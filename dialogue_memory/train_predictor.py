"""
预测模型训练模块 (Prediction Model)
用于计算 U_i (surprise/预测误差)

核心思想:
给定对话历史，预测下一个 utterance。
如果预测不准 → 高 surprise → 值得记忆

训练数据来源 (MSC):
- Session 内: 用前 N-1 句预测第 N 句

使用方法:
1. 训练: train_predictor(dataset, epochs=10)
2. 推理: surprise_score = compute_surprise(model, history, next_utterance)
   - 高损失 → 高 surprise → 值得记忆
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import json


@dataclass
class DialogueTurn:
    """单轮对话"""
    speaker: str
    utterance: str
    embedding: np.ndarray


class PredictionDataset(Dataset):
    """
    预测模型数据集

    从 MSC 数据集构建训练数据:
    - 历史对话 (input)
    - 下一个 utterance (target)
    """

    def __init__(self,
                 data: Dict[str, List],
                 encoder,
                 max_history_len: int = 10):
        """
        Args:
            data: MSC 数据 (grouped format)
            encoder: 文本编码器
            max_history_len: 最大历史长度
        """
        self.encoder = encoder
        self.max_history_len = max_history_len
        self.samples = []

        # 构建样本
        for group_id, sessions in data.items():
            for session in sessions:
                dialogues = session['dialogue']
                speakers = session['speaker']

                # 构建 (history, next_utterance) 对
                for i in range(1, min(len(dialogues), max_history_len + 1)):
                    history = dialogues[:i]
                    next_utterance = dialogues[i] if i < len(dialogues) else None

                    if next_utterance is None:
                        continue

                    self.samples.append({
                        'history': ' '.join(history),
                        'next': next_utterance,
                        'dialogue_id': group_id,
                        'session_id': session['session_id'],
                        'position': i
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 编码历史和目标
        history_emb = self.encoder.encode(sample['history'])
        target_emb = self.encoder.encode(sample['next'])

        return (
            torch.FloatTensor(history_emb),
            torch.FloatTensor(target_emb)
        )


class PredictionMLP(nn.Module):
    """
    预测 MLP

    输入: 历史对话的 embedding
    输出: 预测的下一个 utterance 的 embedding
    """

    def __init__(self, embed_dim: int = 3072, hidden_dim: int = 1024):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionTrainer:
    """
    预测模型训练器

    训练目标: 最小化预测误差
    MSE(predicted_emb, actual_emb)

    推理目标: 计算 surprise
    surprise = MSE(predicted_emb, actual_emb)
    """

    def __init__(self,
                 embed_dim: int = 3072,
                 hidden_dim: int = 1024,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PredictionMLP(embed_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def train_step(self,
                   history_batch: torch.Tensor,
                   target_batch: torch.Tensor) -> float:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        history_batch = history_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        predicted = self.model(history_batch)
        loss = self.criterion(predicted, target_batch)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self,
                dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_surprise = 0.0
        n_batches = 0

        with torch.no_grad():
            for history_batch, target_batch in dataloader:
                history_batch = history_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                predicted = self.model(history_batch)
                loss = self.criterion(predicted, target_batch)

                # 计算 surprise (与损失相同)
                total_loss += loss.item()
                total_surprise += loss.item()
                n_batches += 1

        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'surprise': total_surprise / n_batches if n_batches > 0 else 0.0
        }

    def compute_surprise(self,
                        history_emb: np.ndarray,
                        actual_emb: np.ndarray) -> float:
        """
        计算 surprise (预测误差)

        Args:
            history_emb: 历史对话的 embedding [D]
            actual_emb: 实际下一个 utterance 的 embedding [D]

        Returns:
            surprise 分数 (越高表示越意外，越值得记忆)
        """
        self.model.eval()

        with torch.no_grad():
            history_tensor = torch.FloatTensor(history_emb).unsqueeze(0).to(self.device)
            predicted = self.model(history_tensor)
            target_tensor = torch.FloatTensor(actual_emb).unsqueeze(0).to(self.device)

            # MSE 作为 surprise
            surprise = self.criterion(predicted, target_tensor).item()

        return surprise

    def compute_surprise_batch(self,
                              history_embs: np.ndarray,
                              actual_embs: np.ndarray) -> np.ndarray:
        """
        批量计算 surprise

        Args:
            history_embs: [N, D]
            actual_embs: [N, D]

        Returns:
            surprise 分数 [N]
        """
        self.model.eval()

        with torch.no_grad():
            history_tensor = torch.FloatTensor(history_embs).to(self.device)
            target_tensor = torch.FloatTensor(actual_embs).to(self.device)

            predicted = self.model(history_tensor)
            # MSE per sample
            mse_per_sample = ((predicted - target_tensor) ** 2).mean(dim=1)

        return mse_per_sample.cpu().numpy()

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


def train_predictor(data_path: str,
                    encoder,
                    embed_dim: int = 3072,
                    epochs: int = 10,
                    batch_size: int = 32,
                    val_split: float = 0.1) -> PredictionTrainer:
    """
    训练预测模型

    Args:
        data_path: MSC 数据路径
        encoder: 文本编码器
        embed_dim: embedding 维度
        epochs: 训练轮数
        batch_size: 批大小
        val_split: 验证集比例

    Returns:
        训练好的 trainer
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

    print(f"训练样本: {len(train_data)} 组, 验证样本: {len(val_data)} 组")

    # 构建数据集
    train_dataset = PredictionDataset(train_data, encoder)
    val_dataset = PredictionDataset(val_data, encoder)

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # 创建 data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 训练
    trainer = PredictionTrainer(embed_dim=embed_dim)
    best_val_loss = float('inf')

    print(f"\n开始训练 (device: {trainer.device})...")

    for epoch in range(epochs):
        # 训练
        train_losses = []
        for i, (history, target) in enumerate(train_loader):
            loss = trainer.train_step(history, target)
            train_losses.append(loss)

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}, Loss: {np.mean(train_losses[-100:]):.4f}")

        # 验证
        val_metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss = {np.mean(train_losses):.4f}, "
              f"Val Loss = {val_metrics['loss']:.4f}, "
              f"Val Surprise = {val_metrics['surprise']:.4f}")

        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            print(f"  -> 保存最佳模型 (loss = {best_val_loss:.4f})")

    return trainer
