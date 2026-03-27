"""
对话 Agent
整合记忆系统的对话生成模块
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .stm import ShortTermMemory
from .ltm import HierarchicalLTM
from .consolidation import DialogueConsolidation, DialogueSegment
from .encoder import BaseEncoder, get_encoder


class DialogueAgent:
    """
    记忆增强的对话 Agent

    整合:
    1. 短期记忆 (STM) - 当前对话上下文
    2. 长期记忆 (LTM) - 跨 session 的历史记忆
    3. 记忆巩固 - 从对话中提取关键信息
    """

    def __init__(self,
                 encoder: BaseEncoder = None,
                 encoder_type: str = "sentence_transformer",
                 embed_dim: int = 384,
                 stm_max_turns: int = 20,
                 consolidation_top_k: int = 5):

        # 编码器
        if encoder is None:
            encoder = get_encoder(encoder_type)
        self.encoder = encoder
        self.embed_dim = getattr(encoder, 'embed_dim', embed_dim)

        # 记忆系统
        self.stm = ShortTermMemory(max_turns=stm_max_turns)
        self.ltm = HierarchicalLTM(embed_dim=self.embed_dim)
        self.consolidation = DialogueConsolidation(
            ltm=self.ltm,
            top_k=consolidation_top_k
        )

        # 当前对话状态
        self.current_dialogue_id: Optional[int] = None
        self.current_session_id: int = 0

    def start_dialogue(self, dialogue_id: int, session_id: int = 0):
        """开始新的对话"""
        self.current_dialogue_id = dialogue_id
        self.current_session_id = session_id
        self.stm.start_new_session(session_id)

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        处理用户输入

        1. 编码输入
        2. 从 LTM 检索相关记忆
        3. 构建上下文
        4. 存入 STM

        Returns:
            检索结果和上下文信息
        """
        # 编码
        input_embedding = self.encoder.encode(user_input)

        # 存入 STM
        self.stm.add_turn(
            speaker="User",
            utterance=user_input,
            embedding=input_embedding.tolist()
        )

        # 从 LTM 检索
        retrieval_results = self.ltm.multi_scale_search(input_embedding, top_k_per_layer=3)

        # 获取格式化的检索上下文
        retrieval_context = self.ltm.get_retrieval_context(input_embedding, top_k_per_layer=3)

        # 获取 STM 上下文
        stm_context = self.stm.get_context_text()

        return {
            "user_input": user_input,
            "retrieval_results": retrieval_results,
            "retrieval_context": retrieval_context,
            "stm_context": stm_context,
            "stm_turns": len(self.stm)
        }

    def add_assistant_response(self, response: str):
        """添加助手回复到 STM"""
        response_embedding = self.encoder.encode(response)
        self.stm.add_turn(
            speaker="Assistant",
            utterance=response,
            embedding=response_embedding.tolist()
        )

    def consolidate_session(self, persona_info: List[str] = None):
        """
        巩固当前 session 的记忆

        将 STM 中的关键信息写入 LTM
        """
        # 从 STM 提取对话片段
        segments = []
        turns = self.stm.get_recent_context()

        for i, turn in enumerate(turns):
            if turn.speaker == "User":
                # 获取回复（如果有）
                response = None
                if i + 1 < len(turns) and turns[i + 1].speaker == "Assistant":
                    response = turns[i + 1].utterance

                segment = DialogueSegment(
                    session_id=self.current_session_id,
                    dialogue_id=self.current_dialogue_id,
                    speaker=turn.speaker,
                    utterance=turn.utterance,
                    response=response,
                    embedding=np.array(turn.embedding) if turn.embedding else None
                )
                segments.append(segment)

        # 巩固到 LTM
        if segments:
            self.consolidation.consolidate_session(
                segments=segments,
                encoder_func=self.encoder.encode,
                dialogue_id=self.current_dialogue_id
            )

        # 提取并写入用户画像
        if persona_info:
            self.consolidation.extract_persona(
                sessions=[segments],
                encoder_func=self.encoder.encode,
                dialogue_id=self.current_dialogue_id,
                persona_info=persona_info
            )

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统状态"""
        return {
            "stm": {
                "turns": len(self.stm)
            },
            "ltm": self.ltm.stats(),
            "current_dialogue_id": self.current_dialogue_id,
            "current_session_id": self.current_session_id
        }

    def save_memory(self, path: str):
        """保存长期记忆"""
        self.ltm.save(path)

    def load_memory(self, path: str):
        """加载长期记忆"""
        self.ltm.load(path, self.encoder.encode)


class MemoryAugmentedGenerator:
    """
    记忆增强的回复生成器

    可以接入不同的 LLM 后端
    """

    def __init__(self, agent: DialogueAgent, llm_client=None):
        self.agent = agent
        self.llm_client = llm_client

    def build_prompt(self, user_input: str) -> str:
        """构建带记忆上下文的 prompt"""
        # 获取处理结果
        result = self.agent.process_user_input(user_input)

        prompt_parts = [
            "你是一个友好的对话助手。请根据以下信息生成回复。",
            "",
            "【历史对话】",
            result["stm_context"] or "(无历史对话)",
            "",
            "【相关记忆】",
            result["retrieval_context"] or "(无相关记忆)",
            "",
            "【当前输入】",
            user_input,
            "",
            "请生成一个自然、连贯的回复:"
        ]

        return "\n".join(prompt_parts)

    def generate(self, user_input: str) -> str:
        """生成回复"""
        if self.llm_client is None:
            # 简单的模拟回复
            return self._mock_generate(user_input)

        prompt = self.build_prompt(user_input)
        response = self.llm_client.generate(prompt)

        # 将回复存入 STM
        self.agent.add_assistant_response(response)

        return response

    def _mock_generate(self, user_input: str) -> str:
        """模拟生成回复（用于测试）"""
        result = self.agent.process_user_input(user_input)

        # 简单的规则回复
        if "?" in user_input:
            return "这是一个很好的问题！让我想想..."
        else:
            return "我理解你的意思。"
