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


# ----------------------------------------------------------------------
# Embodied adapter — scores captions from runs/<dir>/episode_*.json
# ----------------------------------------------------------------------


def _split_embodied_samples_by_episode(samples_list, val_split: float, seed: int = 42):
    """Episode-level train/val split for the embodied scorer dataset.

    Same motivation as the predictor split: within-episode caption
    similarity is high enough that step-level random splits leak.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, s in enumerate(samples_list):
        groups[(s.scene_id, s.episode_id)].append(idx)

    keys = sorted(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_split)) if val_split > 0 else 0
    val_keys = set(keys[:n_val])

    train_idx, val_idx = [], []
    for k, idxs in groups.items():
        (val_idx if k in val_keys else train_idx).extend(idxs)
    return sorted(train_idx), sorted(val_idx)


def train_scorer_embodied(run_dirs,
                          encoder,
                          label_mode: str = "soft_spl",
                          epochs: int = 5,
                          batch_size: int = 64,
                          val_split: float = 0.2,
                          keep_per_episode_top_k: Optional[int] = None,
                          save_path: Optional[str] = None,
                          seed: int = 42) -> Tuple[ScorerTrainer, Dict]:
    """Train the importance scorer on captions from embodied run dirs.

    For Phase-1 runs HM3D-ObjectNav had 0 binary successes, so default
    label_mode here is ``soft_spl`` (continuous regression target in
    [0,1]; BCELoss accepts continuous targets and reduces to
    cross-entropy of the soft label).
    """
    from torch.utils.data import Subset
    from embodied_memory.embodied_dataset import EmbodiedImportanceDataset

    embed_dim = int(np.asarray(encoder.encode("probe"), dtype=np.float32).shape[-1])
    print(f"encoder embed_dim probed: {embed_dim}")

    full = EmbodiedImportanceDataset(
        run_dirs, encoder,
        label_mode=label_mode,
        keep_per_episode_top_k=keep_per_episode_top_k,
    )
    if len(full) == 0:
        raise RuntimeError(f"no caption samples found in {list(run_dirs)}")

    stats = full.get_stats()
    print(f"dataset stats: {stats}")
    if label_mode == "success" and stats["positive"] == 0:
        print("WARNING: label_mode=success but 0 positive rows — model will collapse.")

    train_idx, val_idx = _split_embodied_samples_by_episode(full._samples, val_split, seed=seed)
    print(f"samples total={len(full)}  train={len(train_idx)}  val={len(val_idx)}")

    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=batch_size) if val_idx else None

    trainer = ScorerTrainer(embed_dim=embed_dim)
    best_metric = float('-inf')

    print(f"\n开始训练 (device: {trainer.device}) on embodied captions ...")
    for epoch in range(epochs):
        train_losses = []
        for i, (emb, label) in enumerate(train_loader):
            loss = trainer.train_step(emb, label)
            train_losses.append(loss)
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}, Loss: {np.mean(train_losses[-50:]):.4f}")

        msg = f"Epoch {epoch+1}/{epochs}: Train Loss = {np.mean(train_losses):.4f}"
        if val_loader is not None:
            val_metrics = trainer.evaluate(val_loader)
            # For soft labels, accuracy from a 0.5 threshold is not a great
            # signal — track val loss (lower is better) as the save criterion
            # instead. Negate so larger = better, matching best_metric semantics.
            metric = -val_metrics['loss']
            msg += (f", Val Loss = {val_metrics['loss']:.4f}, "
                    f"Val Acc = {val_metrics['accuracy']:.4f}")
            improved = metric > best_metric
            if improved:
                best_metric = metric
        else:
            improved = True
        print(msg)

        if improved and save_path:
            trainer.save(save_path)
            print(f"  -> saved checkpoint to {save_path}")

    if save_path and not val_loader:
        trainer.save(save_path)
        print(f"  -> saved final checkpoint to {save_path}")

    return trainer, stats


def _build_text_encoder(name: str):
    if name == "clip":
        from embodied_memory.perception import CLIPKeyframeEncoder

        class _CLIPTextAdapter:
            def __init__(self):
                self._clip = CLIPKeyframeEncoder()
            def encode(self, text: str):
                return self._clip.encode_text(text)

        return _CLIPTextAdapter()
    if name == "sbert":
        from .encoder import SentenceTransformerEncoder
        return SentenceTransformerEncoder()
    raise ValueError(f"unknown encoder: {name!r} (expected one of: clip, sbert)")


def _cli(argv=None):
    import argparse
    import os

    p = argparse.ArgumentParser(
        description="Train the R_i importance scorer on MSC dialogue or embodied episode logs."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--embodied", nargs="+", metavar="RUN_DIR",
                     help="One or more runs/<dir>/ paths containing episode_*.json files.")
    src.add_argument("--msc", metavar="JSON",
                     help="Path to MSC grouped JSON for dialogue-side training.")
    p.add_argument("--out", default=None,
                   help="Where to save the best checkpoint (.pt). Required for --embodied.")
    p.add_argument("--encoder", default="clip", choices=("clip", "sbert"),
                   help="Text encoder for --embodied (clip matches embodied LTM joint space).")
    p.add_argument("--label-mode", default="soft_spl",
                   choices=("success", "spl", "soft_spl"),
                   help="Embodied label semantics. Default soft_spl because Phase-1 "
                        "has 0 binary successes; success/spl labels will collapse.")
    p.add_argument("--keep-per-episode-top-k", type=int, default=None,
                   help="Optional per-episode subsample (uniform stride) to reduce "
                        "the long tail of nearly-identical captions.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-split", type=float, default=0.2)
    args = p.parse_args(argv)

    if args.embodied:
        if not args.out:
            p.error("--out is required with --embodied")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        encoder = _build_text_encoder(args.encoder)
        train_scorer_embodied(
            run_dirs=args.embodied,
            encoder=encoder,
            label_mode=args.label_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            keep_per_episode_top_k=args.keep_per_episode_top_k,
            save_path=args.out,
        )
        return 0

    # MSC path
    encoder = _build_text_encoder(args.encoder)
    trainer, _ = train_scorer(args.msc, encoder, epochs=args.epochs,
                              batch_size=args.batch_size, val_split=args.val_split)
    if args.out:
        trainer.save(args.out)
        print(f"  -> saved final checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
