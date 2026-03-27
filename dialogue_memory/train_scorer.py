"""
重要性评分器训练模块 (Importance Scorer)
用于计算 R_i (对任务成功的贡献)

核心思想:
预测某段对话是否会在后续 session 被提及（话题延续性）
如果会被提及 → 重要 (R_i = 1)
如果不会 → 不重要 (R_i = 0)

训练数据来源 (MSC):
- 从 Session N 提取对话片段
- 检查 Session N+1/2/3 是否提及相同话题
- 是 → label=1, 否 → label=0

监督信号: 约 92% 的对话有话题延续性

使用方法:
1. 训练: scorer = train_scorer(dataset, epochs=10)
2. 推理: importance = compute_importance(scorer, dialogue_segment)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import json
import re


@dataclass
class ImportanceSample:
    """重要性训练样本"""
    dialogue_segment: str
    embedding: np.ndarray
    will_be_mentioned: bool  # 是否在后续 session 被提及
    session_id: int
    dialogue_id: str


class TopicExtractor:
    """
    话题提取器

    从对话中提取关键词/话题
    用于判断话题是否在后续 session 延续
    """

    # 话题关键词
    TOPIC_KEYWORDS = {
        'sports': ['sport', 'basketball', 'football', 'soccer', 'running', 'marathon',
                   'tennis', 'golf', 'baseball', 'hockey', 'swimming', 'cycling',
                   'chase', 'cheetah', 'race', 'run', 'jog'],
        'food': ['food', 'eat', 'cook', 'recipe', 'meal', 'dinner', 'lunch', 'breakfast',
                 'meat', 'vegetable', 'fruit', 'chicken', 'beef', 'pasta', 'jerky',
                 'canning', 'whittle'],
        'hobby': ['hobby', 'enjoy', 'like', 'love', 'favorite', 'passion',
                  'craft', 'art', 'music', 'game', 'read', 'movie', 'book'],
        'family': ['family', 'wife', 'husband', 'kid', 'child', 'son', 'daughter',
                   'parent', 'mother', 'father', 'brother', 'sister', 'marry', 'date'],
        'work': ['work', 'job', 'career', 'office', 'boss', 'coworker', 'weekend',
                 'monday', 'friday', 'salary', 'promotion', 'business'],
        'health': ['health', 'sick', 'injury', 'hospital', 'doctor', 'medicine',
                   'exercise', 'gym', 'pain', 'hurt', 'broken', 'ankle', 'leg'],
        'location': ['live', 'house', 'home', 'city', 'town', 'travel', 'vacation',
                     'visit', 'move', 'apartment', 'neighbor'],
        'pet': ['pet', 'dog', 'cat', 'animal', 'bird', 'fish', 'puppy', 'kitten']
    }

    @classmethod
    def extract_topics(cls, text: str) -> set:
        """从文本中提取话题"""
        text_lower = text.lower()
        topics = set()

        for topic, keywords in cls.TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    topics.add(topic)
                    break

        return topics

    @classmethod
    def extract_entities(cls, text: str) -> set:
        """提取实体（简单实现）"""
        # 简单实现：提取人称代词后的描述
        entities = set()

        # 提取 "I am / I was / I'm ..." 结构
        patterns = [
            r'i am \w+',
            r"i'm \w+",
            r'i was \w+',
            r'i like \w+',
            r'i love \w+',
            r'i enjoy \w+',
            r'i have a \w+',
            r'my \w+',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

        return entities

    @classmethod
    def is_topic_continued(cls,
                           segment: str,
                           later_dialogues: List[str]) -> bool:
        """
        判断话题是否在后续对话中延续

        Args:
            segment: 当前对话片段
            later_dialogues: 后续对话列表

        Returns:
            True if 话题延续
        """
        # 提取当前片段的话题和实体
        segment_topics = cls.extract_topics(segment)
        segment_entities = cls.extract_entities(segment)

        # 检查后续对话
        for dialogue in later_dialogues:
            later_topics = cls.extract_topics(dialogue)
            later_entities = cls.extract_entities(dialogue)

            # 检查话题重叠
            if segment_topics and segment_topics & later_topics:
                return True

            # 检查实体重叠
            if segment_entities and segment_entities & later_entities:
                return True

        return False


class ImportanceDataset(Dataset):
    """
    重要性评分器数据集

    构建方法:
    1. 从 Session N 提取对话片段
    2. 检查 Session N+1/2/3 是否提及相同话题
    3. 是 → label=1, 否 → label=0
    """

    def __init__(self,
                 data: Dict[str, List],
                 encoder,
                 min_session_count: int = 2):
        """
        Args:
            data: MSC 数据 (grouped format)
            encoder: 文本编码器
            min_session_count: 最小 session 数（需要后续 session）
        """
        self.encoder = encoder
        self.samples = []

        for group_id, sessions in data.items():
            if len(sessions) < min_session_count:
                continue

            # 按 session_id 排序
            sessions = sorted(sessions, key=lambda x: x['session_id'])

            # 对每个 session（除了最后一个）提取样本
            for i in range(len(sessions) - 1):
                current_session = sessions[i]
                later_sessions = sessions[i + 1:]

                # 收集后续对话
                later_dialogues = []
                for later_session in later_sessions:
                    later_dialogues.extend(later_session['dialogue'])

                # 对当前 session 的每个对话片段创建样本
                for j, utterance in enumerate(current_session['dialogue']):
                    # 检查是否在后续被提及
                    will_be_mentioned = TopicExtractor.is_topic_continued(
                        utterance,
                        later_dialogues
                    )

                    # 创建样本
                    sample = {
                        'segment': utterance,
                        'dialogue_id': group_id,
                        'session_id': current_session['session_id'],
                        'will_be_mentioned': will_be_mentioned
                    }
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 编码
        emb = self.encoder.encode(sample['segment'])
        label = 1.0 if sample['will_be_mentioned'] else 0.0

        return (
            torch.FloatTensor(emb),
            torch.FloatTensor([label])
        )

    def get_stats(self) -> Dict[str, int]:
        """获取数据集统计"""
        total = len(self.samples)
        positive = sum(1 for s in self.samples if s['will_be_mentioned'])
        negative = total - positive

        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_ratio': positive / total if total > 0 else 0
        }


class ImportanceScorer(nn.Module):
    """
    重要性评分器

    输入: 对话片段的 embedding
    输出: 重要性分数 (0-1)
    """

    def __init__(self, embed_dim: int = 3072, hidden_dim: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScorerTrainer:
    """
    重要性评分器训练器

    训练目标: BCE(predicted_score, label)
    - label=1: 会被后续 session 提及
    - label=0: 不会被提及
    """

    def __init__(self,
                 embed_dim: int = 3072,
                 hidden_dim: int = 512,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ImportanceScorer(embed_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()

    def train_step(self,
                   emb_batch: torch.Tensor,
                   label_batch: torch.Tensor) -> float:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        emb_batch = emb_batch.to(self.device)
        label_batch = label_batch.to(self.device)

        predicted = self.model(emb_batch)
        loss = self.criterion(predicted, label_batch)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for emb_batch, label_batch in dataloader:
                emb_batch = emb_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                predicted = self.model(emb_batch)
                loss = self.criterion(predicted, label_batch)

                total_loss += loss.item()

                # 计算准确率
                pred_labels = (predicted > 0.5).float()
                correct += (pred_labels == label_batch).sum().item()
                total += label_batch.size(0)

                n_batches += 1

        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'accuracy': correct / total if total > 0 else 0.0
        }

    def compute_importance(self, emb: np.ndarray) -> float:
        """
        计算重要性分数

        Args:
            emb: 对话片段的 embedding [D]

        Returns:
            重要性分数 [0, 1]
        """
        self.model.eval()

        with torch.no_grad():
            emb_tensor = torch.FloatTensor(emb).unsqueeze(0).to(self.device)
            importance = self.model(emb_tensor).item()

        return importance

    def compute_importance_batch(self, embs: np.ndarray) -> np.ndarray:
        """
        批量计算重要性

        Args:
            embs: [N, D]

        Returns:
            重要性分数 [N]
        """
        self.model.eval()

        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embs).to(self.device)
            importance = self.model(emb_tensor).squeeze(-1)

        return importance.cpu().numpy()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_scorer(data_path: str,
                 encoder,
                 embed_dim: int = 3072,
                 hidden_dim: int = 512,
                 epochs: int = 10,
                 batch_size: int = 64,
                 val_split: float = 0.1) -> Tuple[ScorerTrainer, Dict]:
    """
    训练重要性评分器

    Args:
        data_path: MSC 数据路径
        encoder: 文本编码器
        embed_dim: embedding 维度
        hidden_dim: 隐藏层维度
        epochs: 训练轮数
        batch_size: 批大小
        val_split: 验证集比例

    Returns:
        (trainer, dataset_stats)
    """
    # 加载数据
    print(f"加载数据: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)

    # 划分训练/验证
    group_ids = list(data.keys())
    np.random.seed(42)
    np.random.shuffle(group_ids)

    val_size = int(len(group_ids) * val_split)
    train_data = {gid: data[gid] for gid in group_ids[val_size:]}
    val_data = {gid: data[gid] for gid in group_ids[:val_size]}

    print(f"训练组数: {len(train_data)}, 验证组数: {len(val_data)}")

    # 构建数据集
    train_dataset = ImportanceDataset(train_data, encoder)
    val_dataset = ImportanceDataset(val_data, encoder)

    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()

    print(f"\n训练集统计:")
    print(f"  总样本: {train_stats['total']}")
    print(f"  正样本: {train_stats['positive']} ({train_stats['positive_ratio']:.1%})")
    print(f"  负样本: {train_stats['negative']}")

    print(f"\n验证集统计:")
    print(f"  总样本: {val_stats['total']}")
    print(f"  正样本: {val_stats['positive']} ({val_stats['positive_ratio']:.1%})")
    print(f"  负样本: {val_stats['negative']}")

    # 创建 data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 训练
    trainer = ScorerTrainer(embed_dim=embed_dim, hidden_dim=hidden_dim)
    best_acc = 0.0

    print(f"\n开始训练 (device: {trainer.device})...")

    for epoch in range(epochs):
        # 训练
        train_losses = []
        for i, (emb, label) in enumerate(train_loader):
            loss = trainer.train_step(emb, label)
            train_losses.append(loss)

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}, Loss: {np.mean(train_losses[-100:]):.4f}")

        # 验证
        val_metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss = {np.mean(train_losses):.4f}, "
              f"Val Loss = {val_metrics['loss']:.4f}, "
              f"Val Acc = {val_metrics['accuracy']:.4f}")

        # 保存最佳模型
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            print(f"  -> 保存最佳模型 (acc = {best_acc:.4f})")

    return trainer, {'train': train_stats, 'val': val_stats}
