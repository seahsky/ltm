"""
MSC (Multi-Session Chat) 数据加载器

数据格式:
- dialogue_id: 对话组 ID
- session_id: 会话 ID (0-3)
- persona1/persona2: 用户人设
- dialogue: 对话列表
- speaker: 说话者列表
"""

import json
from typing import List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Session:
    """单个对话 Session"""
    dialogue_id: int
    session_id: int
    persona1: List[str]
    persona2: List[str]
    dialogue: List[str]
    speakers: List[str]

    def get_turns(self) -> Iterator[Tuple[str, str]]:
        """迭代获取每轮对话 (speaker, utterance)"""
        for speaker, utterance in zip(self.speakers, self.dialogue):
            yield speaker, utterance

    def get_qa_pairs(self) -> List[Tuple[str, str]]:
        """获取问答对 (交替的对话轮次)"""
        pairs = []
        for i in range(0, len(self.dialogue) - 1, 2):
            q = self.dialogue[i]
            a = self.dialogue[i + 1] if i + 1 < len(self.dialogue) else ""
            pairs.append((q, a))
        return pairs

    def get_context_window(self, window_size: int = 3) -> List[Tuple[List[str], str]]:
        """
        获取滑动窗口上下文

        Returns:
            List of (context_utterances, current_utterance)
        """
        windows = []
        for i in range(len(self.dialogue)):
            start = max(0, i - window_size)
            context = self.dialogue[start:i]
            current = self.dialogue[i]
            windows.append((context, current))
        return windows


@dataclass
class DialogueGroup:
    """一组跨 session 的对话"""
    dialogue_id: int
    sessions: List[Session]

    def get_session(self, session_id: int) -> Session:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None

    def get_all_sessions_up_to(self, session_id: int) -> List[Session]:
        """获取指定 session 之前的所有 session（用于构建历史记忆）"""
        return [s for s in self.sessions if s.session_id <= session_id]


class MSCDataLoader:
    """
    MSC 数据集加载器
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or "/home/ec2-user/studies/ltm_agent/data/msc"
        self._train_data = None
        self._val_data = None
        self._test_data = None

    def _load_split(self, split: str) -> Dict[int, DialogueGroup]:
        """加载指定 split 的数据"""
        file_path = os.path.join(self.data_dir, f"msc_{split}_grouped.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 转换为结构化格式
        groups = {}
        for dialogue_id, sessions_data in raw_data.items():
            dialogue_id = int(dialogue_id)
            sessions = []

            for session_data in sessions_data:
                session = Session(
                    dialogue_id=dialogue_id,
                    session_id=session_data["session_id"],
                    persona1=session_data.get("persona1", []),
                    persona2=session_data.get("persona2", []),
                    dialogue=session_data["dialogue"],
                    speakers=session_data["speaker"]
                )
                sessions.append(session)

            # 按 session_id 排序
            sessions.sort(key=lambda s: s.session_id)
            groups[dialogue_id] = DialogueGroup(dialogue_id, sessions)

        return groups

    @property
    def train(self) -> Dict[int, DialogueGroup]:
        if self._train_data is None:
            self._train_data = self._load_split("train")
        return self._train_data

    @property
    def val(self) -> Dict[int, DialogueGroup]:
        if self._val_data is None:
            self._val_data = self._load_split("val")
        return self._val_data

    @property
    def test(self) -> Dict[int, DialogueGroup]:
        if self._test_data is None:
            self._test_data = self._load_split("test")
        return self._test_data

    def iter_sessions(self, split: str = "train") -> Iterator[Tuple[int, Session]]:
        """迭代所有 session"""
        data = getattr(self, split)
        for dialogue_id, group in data.items():
            for session in group.sessions:
                yield dialogue_id, session

    def get_multi_session_dialogues(self,
                                     split: str = "train",
                                     min_sessions: int = 2) -> List[DialogueGroup]:
        """
        获取有多个 session 的对话（用于测试跨 session 记忆）
        """
        data = getattr(self, split)
        return [g for g in data.values() if len(g.sessions) >= min_sessions]

    def stats(self) -> Dict[str, Any]:
        """统计数据集信息"""
        stats = {}
        for split in ["train", "val", "test"]:
            data = getattr(self, split)
            total_sessions = sum(len(g.sessions) for g in data.values())
            multi_session_count = sum(1 for g in data.values() if len(g.sessions) >= 2)

            session_dist = {}
            for g in data.values():
                n = len(g.sessions)
                session_dist[n] = session_dist.get(n, 0) + 1

            stats[split] = {
                "dialogue_groups": len(data),
                "total_sessions": total_sessions,
                "multi_session_groups": multi_session_count,
                "session_distribution": session_dist
            }

        return stats


def download_msc_dataset(save_dir: str = None):
    """
    下载 MSC 数据集（如果尚未下载）
    """
    save_dir = save_dir or "/home/ec2-user/studies/ltm_agent/data/msc"
    os.makedirs(save_dir, exist_ok=True)

    # 检查是否已下载
    if os.path.exists(os.path.join(save_dir, "msc_train_grouped.json")):
        print(f"MSC 数据集已存在于 {save_dir}")
        return

    print("正在下载 MSC 数据集...")
    from datasets import load_dataset

    ds = load_dataset('nayohan/multi_session_chat')

    # 处理并保存
    from collections import defaultdict

    def group_by_dialogue(split_data):
        grouped = defaultdict(list)
        for sample in split_data:
            grouped[sample['dialoug_id']].append({
                'session_id': sample['session_id'],
                'persona1': sample['persona1'],
                'persona2': sample['persona2'],
                'dialogue': sample['dialogue'],
                'speaker': sample['speaker']
            })
        for dial_id in grouped:
            grouped[dial_id] = sorted(grouped[dial_id], key=lambda x: x['session_id'])
        return dict(grouped)

    for split_name, split_data in [('train', ds['train']), ('val', ds['validation']), ('test', ds['test'])]:
        grouped = group_by_dialogue(split_data)
        save_path = os.path.join(save_dir, f"msc_{split_name}_grouped.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(grouped, f, ensure_ascii=False, indent=2)
        print(f"已保存: {save_path}")

    print("✅ MSC 数据集下载完成!")


if __name__ == "__main__":
    # 测试加载器
    loader = MSCDataLoader()
    print("数据集统计:")
    print(json.dumps(loader.stats(), indent=2, ensure_ascii=False))

    # 打印一个示例
    first_group = list(loader.train.values())[0]
    print(f"\n示例对话组 (ID: {first_group.dialogue_id}):")
    print(f"  Session 数量: {len(first_group.sessions)}")
    for session in first_group.sessions:
        print(f"  Session {session.session_id}: {len(session.dialogue)} 轮对话")
