"""
记忆巩固模块 (Memory Consolidation)
从短期对话中筛选关键经验，写入长期记忆

核心公式:
I(τ_i) = α * R_i + β * U_i + γ * N_i

其中:
- R_i: 信息丰富度 (Relevance)
- U_i: 独特性 (Uniqueness/Surprise)
- N_i: 新颖度 (Novelty)
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .ltm import HierarchicalLTM, MemoryEntry


@dataclass
class DialogueSegment:
    """对话片段"""
    session_id: int
    dialogue_id: int
    speaker: str
    utterance: str
    response: str = None  # 如果是问答对，存储回复
    embedding: np.ndarray = None
    metadata: dict = None


class DialogueConsolidation:
    """
    对话记忆巩固

    从短期对话中筛选关键信息，计算重要性评分，写入长期记忆
    """

    def __init__(self,
                 ltm: HierarchicalLTM,
                 alpha: float = 0.4,  # 信息丰富度权重
                 beta: float = 0.3,   # 独特性权重
                 gamma: float = 0.3,  # 新颖度权重
                 top_k: int = 5):     # 每个 session 保留的关键片段数
        self.ltm = ltm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k

        # 历史记录（用于计算 surprise）
        self.info_richness_history = []

    def compute_importance(self,
                          segment: DialogueSegment,
                          encoder_func) -> Tuple[float, Dict[str, float]]:
        """
        计算记忆重要性评分

        I = α * R + β * U + γ * N

        Returns:
            (总分, 各项分数明细)
        """
        R = self._compute_relevance(segment)
        U = self._compute_uniqueness(segment)
        N = self._compute_novelty(segment, encoder_func)

        total = self.alpha * R + self.beta * U + self.gamma * N

        breakdown = {
            "relevance": R,
            "uniqueness": U,
            "novelty": N,
            "total": total
        }

        return total, breakdown

    # Weight applied to R for embodied records whose episode_success metadata
    # is False. Successful and absent (dialogue path) keep weight 1.0.
    FAILED_EPISODE_RELEVANCE_WEIGHT = 0.25

    def _compute_relevance(self, segment: DialogueSegment) -> float:
        """
        计算信息丰富度 (Relevance)

        基于:
        1. 对话长度
        2. 是否包含个人信息
        3. 是否是关键问题
        4. (embodied only) episode_success — failed episodes get a R multiplier
           of FAILED_EPISODE_RELEVANCE_WEIGHT. Missing metadata defaults to 1.0
           so the dialogue path is unchanged.
        """
        text = f"{segment.utterance} {segment.response or ''}"

        # 基础分数: 长度归一化
        length_score = min(len(text) / 200.0, 1.0)

        # 个人信息关键词
        personal_keywords = [
            "like", "love", "hate", "favorite", "hobby", "enjoy",
            "my", "i am", "i have", "i want", "i need",
            "喜欢", "爱", "讨厌", "爱好", "我的"
        ]
        personal_score = sum(1 for kw in personal_keywords if kw.lower() in text.lower()) * 0.2
        personal_score = min(personal_score, 1.0)

        # 问题关键词
        question_keywords = ["what", "how", "why", "when", "where", "do you", "are you", "?"]
        question_score = sum(1 for kw in question_keywords if kw.lower() in text.lower()) * 0.1
        question_score = min(question_score, 0.5)

        score = 0.4 * length_score + 0.4 * personal_score + 0.2 * question_score

        meta = segment.metadata or {}
        if "episode_success" in meta and not bool(meta.get("episode_success")):
            score *= self.FAILED_EPISODE_RELEVANCE_WEIGHT

        # 更新历史
        self.info_richness_history.append(score)
        if len(self.info_richness_history) > 100:
            self.info_richness_history.pop(0)

        return score

    def _compute_uniqueness(self, segment: DialogueSegment) -> float:
        """
        计算独特性 (Uniqueness/Surprise)

        基于当前信息丰富度与历史平均的偏差
        """
        if len(self.info_richness_history) <= 1:
            return 0.5  # 数据不足时返回中等分数

        current_score = self.info_richness_history[-1]
        historical_mean = np.mean(self.info_richness_history[:-1])

        # 偏差越大，surprise 越高
        surprise = abs(current_score - historical_mean)

        return min(surprise * 2, 1.0)  # 归一化到 [0, 1]

    def _compute_novelty(self, segment: DialogueSegment, encoder_func) -> float:
        """
        计算新颖度 (Novelty)

        N_i = min ||z_i - z_j||_2  for z_j in LTM

        相对于已有记忆的语义距离
        """
        if segment.embedding is None:
            return 0.5

        # 获取所有层级的 embeddings
        all_embeddings = []

        for level in ["fine", "mid", "coarse"]:
            layer_embeddings = self.ltm.layers[level].get_all_embeddings()
            if layer_embeddings is not None:
                all_embeddings.append(layer_embeddings)

        if len(all_embeddings) == 0:
            return 1.0  # 没有历史记忆时，完全新颖

        all_embeddings = np.vstack(all_embeddings)

        # 计算到所有已有记忆的距离
        distances = np.linalg.norm(all_embeddings - segment.embedding, axis=1)
        min_distance = np.min(distances)

        # 归一化: 距离越大，新颖度越高
        # 假设 L2 距离 5.0 为最大有意义的距离
        novelty = min(min_distance / 5.0, 1.0)

        return novelty

    def consolidate_session(self,
                           segments: List[DialogueSegment],
                           encoder_func,
                           dialogue_id: int) -> Dict[str, List[str]]:
        """
        巩固一个 session 的对话

        1. 计算每个片段的重要性
        2. 筛选 top_k 个关键片段
        3. 写入长期记忆

        Returns:
            写入各层的记忆 ID 列表
        """
        # 计算所有片段的重要性评分
        scored_segments = []
        for seg in segments:
            score, breakdown = self.compute_importance(seg, encoder_func)
            scored_segments.append((seg, score, breakdown))

        # 按重要性排序
        scored_segments.sort(key=lambda x: x[1], reverse=True)

        # 筛选 top_k
        key_segments = scored_segments[:self.top_k]

        # 写入长期记忆
        inserted_ids = {"fine": [], "mid": [], "coarse": []}

        for seg, score, breakdown in key_segments:
            # 写入 Fine 层 (对话片段)
            content = f"Q: {seg.utterance}"
            if seg.response:
                content += f" A: {seg.response}"

            entry_id = self.ltm.insert(
                level="fine",
                embedding=seg.embedding,
                content=content,
                metadata={
                    "dialogue_id": dialogue_id,
                    "session_id": seg.session_id,
                    "importance_score": score,
                    "breakdown": breakdown
                }
            )
            inserted_ids["fine"].append(entry_id)

        return inserted_ids

    def extract_patterns(self,
                        sessions: List[List[DialogueSegment]],
                        encoder_func,
                        dialogue_id: int) -> List[str]:
        """
        从多个 session 中提取对话模式 (写入 Mid 层)

        识别:
        1. 重复出现的话题
        2. 用户偏好模式
        3. 对话风格
        """
        # 收集所有提及的话题/偏好
        all_topics = []
        preference_patterns = []

        for session in sessions:
            for seg in session:
                text = seg.utterance.lower()

                # 提取偏好关键词
                if any(kw in text for kw in ["i like", "i love", "my favorite", "i enjoy"]):
                    preference_patterns.append(seg.utterance)

        # 对相似的偏好进行聚类（简化版: 基于 embedding 相似度）
        if preference_patterns:
            embeddings = [encoder_func(p) for p in preference_patterns]
            embeddings = np.array(embeddings)

            # 简单聚类: 每个偏好作为一个模式
            for i, (pref, emb) in enumerate(zip(preference_patterns, embeddings)):
                self.ltm.insert(
                    level="mid",
                    embedding=emb,
                    content=f"用户偏好: {pref}",
                    metadata={
                        "dialogue_id": dialogue_id,
                        "type": "preference"
                    }
                )

        return preference_patterns

    def extract_persona(self,
                       sessions: List[List[DialogueSegment]],
                       encoder_func,
                       dialogue_id: int,
                       persona_info: List[str] = None) -> str:
        """
        提取/写入用户画像 (Coarse 层)

        结合:
        1. 数据集提供的 persona
        2. 从对话中推断的画像
        """
        # 如果有提供的 persona 信息
        if persona_info:
            persona_text = " | ".join(persona_info)
            persona_emb = encoder_func(persona_text)

            self.ltm.insert(
                level="coarse",
                embedding=persona_emb,
                content=f"用户画像: {persona_text}",
                metadata={
                    "dialogue_id": dialogue_id,
                    "type": "persona",
                    "raw_persona": persona_info
                }
            )
            return persona_text

        # 从对话中推断
        inferred_traits = []
        for session in sessions:
            for seg in session:
                text = seg.utterance.lower()

                # 推断职业
                if "i work" in text or "my job" in text:
                    inferred_traits.append(seg.utterance)

                # 推断家庭
                if "my family" in text or "my kids" in text or "my spouse" in text:
                    inferred_traits.append(seg.utterance)

        if inferred_traits:
            persona_text = " | ".join(inferred_traits[:3])  # 最多保留 3 条
            persona_emb = encoder_func(persona_text)

            self.ltm.insert(
                level="coarse",
                embedding=persona_emb,
                content=f"推断画像: {persona_text}",
                metadata={
                    "dialogue_id": dialogue_id,
                    "type": "inferred_persona"
                }
            )
            return persona_text

        return None
