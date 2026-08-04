"""
回复重排序模块 (Response Reranking)
实现记忆增强的回复选择机制

核心公式 (来自研究提案 Section 3.6):
Score(p_k) = w₁·S_succ + w₂·S_sim + w₃·S_phys

对话场景映射:
Score(response_k) = w₁·S_history + w₂·S_memory + w₃·S_coherence

其中:
- S_history: 该模式的历史成功率 (从 Mid 层获取)
- S_memory: 与检索到的记忆的相似度
- S_coherence: 与当前对话的连贯性

功能:
1. 生成多个候选回复
2. 计算各维度分数
3. 加权融合得到最终分数
4. 返回排序后的回复列表
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ScoredResponse:
    """带分数的回复候选"""
    response: str
    response_embedding: np.ndarray
    scores: Dict[str, float]  # 各维度分数
    final_score: float  # 加权总分
    rank: int = 0  # 排名


@dataclass
class RerankingResult:
    """重排序结果"""
    candidates: List[ScoredResponse]  # 排序后的候选列表
    selected: ScoredResponse  # 最终选择的回复
    debug_info: Dict[str, Any] = None  # 调试信息


class Scorer(ABC):
    """评分器基类"""

    @abstractmethod
    def score(self,
              candidate: str,
              candidate_embedding: np.ndarray,
              context: Dict[str, Any]) -> float:
        """
        计算分数

        Args:
            candidate: 候选回复
            candidate_embedding: 候选回复的 embedding
            context: 上下文信息（包括检索到的记忆等）

        Returns:
            分数 [0, 1]
        """
        pass


class HistorySuccessScorer(Scorer):
    """
    历史成功率评分器 (S_succ / S_history)

    基于 Mid 层聚类的历史成功率
    """

    def __init__(self, mid_layer_memory):
        self.mid_layer_memory = mid_layer_memory

    def score(self,
              candidate: str,
              candidate_embedding: np.ndarray,
              context: Dict[str, Any]) -> float:
        """
        S_history: 基于候选回复所属模式的历史成功率
        """
        # 查找最相关的聚类
        pattern = self.mid_layer_memory.get_pattern_for_context(candidate_embedding)

        if pattern is None:
            return 0.5  # 没有匹配的模式，返回中等分数

        return pattern.success_rate


class MemorySimilarityScorer(Scorer):
    """
    记忆相似度评分器 (S_sim)

    基于与检索到的记忆的语义相似度
    """

    def __init__(self):
        self.weight_fine = 0.5   # Fine 层权重
        self.weight_mid = 0.3    # Mid 层权重
        self.weight_coarse = 0.2  # Coarse 层权重

    def score(self,
              candidate: str,
              candidate_embedding: np.ndarray,
              context: Dict[str, Any]) -> float:
        """
        S_memory: 与检索记忆的加权相似度
        """
        retrieval_results = context.get("retrieval_results", {})

        if not retrieval_results:
            return 0.5

        total_score = 0.0
        total_weight = 0.0

        # Fine 层相似度
        if "fine" in retrieval_results and retrieval_results["fine"]:
            fine_sims = []
            for entry, dist in retrieval_results["fine"]:
                # 距离转相似度 (假设阈值 5.0)
                sim = max(0, 1 - dist / 5.0)
                fine_sims.append(sim)
            score = np.mean(fine_sims)
            total_score += self.weight_fine * score
            total_weight += self.weight_fine

        # Mid 层相似度
        if "mid" in retrieval_results and retrieval_results["mid"]:
            mid_sims = []
            for entry, dist in retrieval_results["mid"]:
                sim = max(0, 1 - dist / 5.0)
                mid_sims.append(sim)
            score = np.mean(mid_sims)
            total_score += self.weight_mid * score
            total_weight += self.weight_mid

        # Coarse 层相似度
        if "coarse" in retrieval_results and retrieval_results["coarse"]:
            coarse_sims = []
            for entry, dist in retrieval_results["coarse"]:
                sim = max(0, 1 - dist / 5.0)
                coarse_sims.append(sim)
            score = np.mean(coarse_sims)
            total_score += self.weight_coarse * score
            total_weight += self.weight_coarse

        if total_weight == 0:
            return 0.5

        return total_score / total_weight


class CoherenceScorer(Scorer):
    """
    连贯性评分器 (S_phys -> S_coherence)

    基于与当前对话上下文的连贯性
    """

    def __init__(self):
        self.stm_window = 5  # 考虑最近 N 轮对话

    def score(self,
              candidate: str,
              candidate_embedding: np.ndarray,
              context: Dict[str, Any]) -> float:
        """
        S_coherence: 与当前对话的连贯性
        """
        stm_context = context.get("stm_context", None)

        if stm_context is None or len(stm_context) == 0:
            return 0.5

        # 如果 stm_context 是字符串列表（utterance 列表）
        if isinstance(stm_context, list):
            recent_turns = stm_context[-self.stm_window:]
        else:
            # 字符串格式：取最后 N 轮
            turns = stm_context.split('\n')
            recent_turns = turns[-self.stm_window:]

        # 计算与最近对话的平均相似度（简化版：基于关键词重叠）
        coherence_score = self._compute_keyword_coherence(
            candidate,
            recent_turns
        )

        return coherence_score

    def _compute_keyword_coherence(self, candidate: str, recent_turns: List[str]) -> float:
        """基于关键词重叠计算连贯性"""
        # 定义关键词类别
        keywords = {
            "topic": ["what", "how", "why", "when", "where", "who", "什么", "怎么", "为什么"],
            "person": ["i", "me", "my", "you", "your", "我", "你", "我的"],
            "action": ["do", "go", "make", "take", "have", "做", "去", "拿"],
            "emotion": ["feel", "think", "hope", "want", "like", "觉得", "想", "喜欢"]
        }

        candidate_lower = candidate.lower()

        scores = []
        for turn in recent_turns:
            turn_lower = turn.lower()
            score = 0.0

            for category, kws in keywords.items():
                if any(kw in candidate_lower for kw in kws):
                    if any(kw in turn_lower for kw in kws):
                        score += 0.25

            scores.append(score)

        return np.mean(scores) if scores else 0.5


class LLMScoringWrapper(Scorer):
    """
    LLM 评分包装器

    使用 LLM 评估回复质量（如果可用）
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def score(self,
              candidate: str,
              candidate_embedding: np.ndarray,
              context: Dict[str, Any]) -> float:
        """
        使用 LLM 评估回复质量
        """
        if self.llm_client is None:
            return 0.5  # 没有 LLM 时返回默认分数

        user_input = context.get("user_input", "")
        prompt = f"""评估以下回复的质量。回复应与用户输入相关、与对话上下文连贯、且内容有信息量。

用户输入: {user_input}
候选回复: {candidate}

请从以下维度评分（0-1分）：
1. 相关性：与用户输入的相关程度
2. 连贯性：与对话上下文的连贯程度
3. 信息量：内容的丰富程度

直接输出一个0-1之间的分数，不要有其他文字。"""

        try:
            response = self.llm_client.generate(prompt)
            # 解析分数（简化处理）
            score = float(response.strip()) if response.strip().replace('.', '').isdigit() else 0.5
            return min(1.0, max(0.0, score))
        except:
            return 0.5


class ResponseReranker:
    """
    回复重排序器

    核心公式:
    Score(response_k) = w₁·S_history + w₂·S_memory + w₃·S_coherence

    论文对应: Section 3.6 记忆增强计划重排序
    """

    def __init__(self,
                 mid_layer_memory=None,
                 ltm=None,
                 llm_client=None,
                 weights: Dict[str, float] = None):
        """
        Args:
            mid_layer_memory: Mid 层记忆管理器（用于 S_history）
            ltm: LTM 实例（用于 S_memory）
            llm_client: LLM 客户端（可选，用于高级评分）
            weights: 各评分器权重
        """
        self.mid_layer_memory = mid_layer_memory
        self.llm_client = llm_client

        # 默认权重
        self.weights = weights or {
            "history": 0.3,   # w₁: 历史成功率
            "memory": 0.4,    # w₂: 记忆相似度
            "coherence": 0.3  # w₃: 连贯性
        }

        # 初始化评分器
        self.scorers = {
            "history": HistorySuccessScorer(mid_layer_memory) if mid_layer_memory else None,
            "memory": MemorySimilarityScorer(),
            "coherence": CoherenceScorer(),
            "llm": LLMScoringWrapper(llm_client) if llm_client else None
        }

    def rerank(self,
              candidates: List[str],
              embeddings: np.ndarray,
              context: Dict[str, Any]) -> RerankingResult:
        """
        对候选回复进行重排序

        Args:
            candidates: 候选回复列表
            embeddings: 候选回复的 embeddings [N, D]
            context: 上下文信息，包含:
                - retrieval_results: 检索结果
                - stm_context: 短期记忆上下文
                - user_input: 用户输入

        Returns:
            RerankingResult: 包含排序结果和调试信息
        """
        scored_candidates = []

        # 对每个候选计算分数
        for i, (candidate, embedding) in enumerate(zip(candidates, embeddings)):
            scores = {}
            total_weight = 0.0
            weighted_sum = 0.0

            # S_history: 历史成功率
            if self.scorers["history"] and self.weights["history"] > 0:
                s_history = self.scorers["history"].score(candidate, embedding, context)
                scores["history"] = s_history
                weighted_sum += self.weights["history"] * s_history
                total_weight += self.weights["history"]

            # S_memory: 记忆相似度
            if self.scorers["memory"] and self.weights["memory"] > 0:
                s_memory = self.scorers["memory"].score(candidate, embedding, context)
                scores["memory"] = s_memory
                weighted_sum += self.weights["memory"] * s_memory
                total_weight += self.weights["memory"]

            # S_coherence: 连贯性
            if self.scorers["coherence"] and self.weights["coherence"] > 0:
                s_coherence = self.scorers["coherence"].score(candidate, embedding, context)
                scores["coherence"] = s_coherence
                weighted_sum += self.weights["coherence"] * s_coherence
                total_weight += self.weights["coherence"]

            # LLM 评分（如果有）
            if self.scorers["llm"]:
                s_llm = self.scorers["llm"].score(candidate, embedding, context)
                scores["llm"] = s_llm
                # LLM 分数作为额外参考，不计入总权重

            # 计算最终分数
            if total_weight > 0:
                final_score = weighted_sum / total_weight
            else:
                final_score = 0.5

            scored_candidates.append(ScoredResponse(
                response=candidate,
                response_embedding=embedding,
                scores=scores,
                final_score=final_score
            ))

        # 按分数排序
        scored_candidates.sort(key=lambda x: x.final_score, reverse=True)

        # 设置排名
        for i, sc in enumerate(scored_candidates):
            sc.rank = i + 1

        # 构建结果
        debug_info = {
            "num_candidates": len(candidates),
            "weights": self.weights,
            "top_scores": [
                {"response": sc.response[:50], "final_score": sc.final_score, "scores": sc.scores}
                for sc in scored_candidates[:3]
            ]
        }

        return RerankingResult(
            candidates=scored_candidates,
            selected=scored_candidates[0] if scored_candidates else None,
            debug_info=debug_info
        )

    def select_top_k(self,
                    rerank_result: RerankingResult,
                    k: int = 3) -> List[ScoredResponse]:
        """选择 top-k 候选"""
        return rerank_result.candidates[:k]

    def update_weights(self, weights: Dict[str, float]):
        """更新权重"""
        self.weights.update(weights)


class CandidateGenerator:
    """
    候选回复生成器

    简单的 LLM 调用生成多个候选
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate(self,
                 prompt: str,
                 num_candidates: int = 5) -> List[str]:
        """
        生成多个候选回复

        简单实现：使用 LLM 生成多个候选
        实际应用中可以使用 beam search、self-consistency 等方法
        """
        if self.llm_client is None:
            return self._mock_generate(prompt, num_candidates)

        # 使用 LLM 生成多个候选
        generation_prompt = f"""{prompt}

请生成 {num_candidates} 个不同的回复候选，每个候选用 [CANDIDATE_i] 标记：

[CANDIDATE_1] 候选1
[CANDIDATE_2] 候选2
...
"""

        try:
            response = self.llm_client.generate(generation_prompt)
            return self._parse_candidates(response, num_candidates)
        except:
            return self._mock_generate(prompt, num_candidates)

    def _parse_candidates(self, response: str, num_candidates: int) -> List[str]:
        """解析 LLM 输出中的候选"""
        candidates = []
        for i in range(1, num_candidates + 1):
            marker = f"[CANDIDATE_{i}]"
            if marker in response:
                start = response.index(marker) + len(marker)
                # 找下一个 marker 或结束
                end = len(response)
                for j in range(i + 1, num_candidates + 1):
                    next_marker = f"[CANDIDATE_{j}]"
                    if next_marker in response:
                        end = response.index(next_marker)
                        break
                candidates.append(response[start:end].strip())
        return candidates if candidates else self._mock_generate("", num_candidates)

    def _mock_generate(self, prompt: str, num_candidates: int) -> List[str]:
        """模拟生成候选（用于测试）"""
        base_responses = [
            "这是一个很好的问题！让我想想...",
            "我理解你的意思。",
            "让我从几个方面来说明。",
            "这很有趣！",
            "你说得对。"
        ]
        return base_responses[:min(num_candidates, len(base_responses))]
