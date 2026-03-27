"""
短期记忆 (Short-Term Memory)
暂存当前对话上下文，为记忆巩固提供输入
"""

from collections import deque
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DialogueTurn:
    """单轮对话"""
    speaker: str
    utterance: str
    embedding: Optional[List[float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class ShortTermMemory:
    """
    短期记忆

    使用滑动窗口暂存最近的对话轮次
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.buffer: deque = deque(maxlen=max_turns)
        self.current_dialogue_id: Optional[int] = None
        self.current_session_id: Optional[int] = None

    def add_turn(self,
                 speaker: str,
                 utterance: str,
                 embedding: List[float] = None,
                 metadata: dict = None):
        """添加一轮对话"""
        turn = DialogueTurn(
            speaker=speaker,
            utterance=utterance,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.buffer.append(turn)

    def get_recent_context(self, n_turns: int = None) -> List[DialogueTurn]:
        """获取最近 n 轮对话"""
        if n_turns is None:
            return list(self.buffer)
        return list(self.buffer)[-n_turns:]

    def get_context_text(self, n_turns: int = None) -> str:
        """获取格式化的上下文文本"""
        turns = self.get_recent_context(n_turns)
        lines = []
        for turn in turns:
            lines.append(f"{turn.speaker}: {turn.utterance}")
        return "\n".join(lines)

    def get_user_utterances(self) -> List[str]:
        """获取用户说过的所有话"""
        return [t.utterance for t in self.buffer if t.speaker != "Assistant"]

    def clear(self):
        """清空短期记忆"""
        self.buffer.clear()

    def start_new_session(self, session_id: int):
        """开始新的 session"""
        self.current_session_id = session_id
        # 不清空 buffer，保留跨 session 的上下文

    def __len__(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0
