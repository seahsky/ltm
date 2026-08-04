"""
分层长期记忆系统 (Hierarchical Long-Term Memory)
用于对话场景的记忆存储和检索

三层结构:
- Fine: 对话片段 (具体问答对)
- Mid: 对话模式 (用户偏好/话题模式)
- Coarse: 用户画像 (persona 知识)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import faiss


@dataclass
class MemoryEntry:
    """记忆条目基类"""
    id: str
    embedding: np.ndarray
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class MemoryLayer:
    """单层记忆存储，使用 FAISS 进行向量检索"""

    def __init__(self, name: str, embed_dim: int = 768):
        self.name = name
        self.embed_dim = embed_dim
        self.index = faiss.IndexFlatL2(embed_dim)  # L2 距离索引
        self.entries: List[MemoryEntry] = []
        self.id_counter = 0

    def insert(self, embedding: np.ndarray, content: str, metadata: dict = None) -> str:
        """插入一条记忆"""
        assert embedding.shape[0] == self.embed_dim

        # 生成唯一 ID
        entry_id = f"{self.name}_{self.id_counter}"
        self.id_counter += 1

        # 创建条目
        entry = MemoryEntry(
            id=entry_id,
            embedding=embedding,
            content=content,
            metadata=metadata or {}
        )

        # 添加到 FAISS 索引
        self.index.add(embedding.reshape(1, -1).astype('float32'))

        # 保存条目
        self.entries.append(entry)

        return entry_id

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """检索最相关的记忆"""
        if len(self.entries) == 0:
            return []

        # FAISS 搜索
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, min(top_k, len(self.entries)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.entries):
                results.append((self.entries[idx], float(dist)))

        return results

    def get_all_embeddings(self) -> Optional[np.ndarray]:
        """获取所有 embedding（用于计算新颖度）"""
        if len(self.entries) == 0:
            return None
        return np.array([e.embedding for e in self.entries])

    def __len__(self):
        return len(self.entries)


class HierarchicalLTM:
    """
    分层长期记忆系统

    三层结构:
    - fine: 对话片段层 (具体的问答对)
    - mid: 对话模式层 (用户偏好、话题模式)
    - coarse: 用户画像层 (persona 知识)
    """

    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim

        # 三层记忆
        self.fine = MemoryLayer("fine", embed_dim)      # 对话片段
        self.mid = MemoryLayer("mid", embed_dim)        # 对话模式
        self.coarse = MemoryLayer("coarse", embed_dim)  # 用户画像

        self.layers = {
            "fine": self.fine,
            "mid": self.mid,
            "coarse": self.coarse
        }

    def insert(self, level: str, embedding: np.ndarray, content: str, metadata: dict = None) -> str:
        """向指定层级插入记忆"""
        assert level in self.layers, f"Invalid level: {level}"
        return self.layers[level].insert(embedding, content, metadata)

    def search(self, level: str, query_embedding: np.ndarray, top_k: int = 5):
        """在指定层级检索记忆"""
        assert level in self.layers, f"Invalid level: {level}"
        return self.layers[level].search(query_embedding, top_k)

    def multi_scale_search(self, query_embedding: np.ndarray,
                           top_k_per_layer: int = 3) -> Dict[str, List[Tuple[MemoryEntry, float]]]:
        """
        多尺度检索: 同时在三层进行检索

        Returns:
            {
                "fine": [(entry, distance), ...],
                "mid": [(entry, distance), ...],
                "coarse": [(entry, distance), ...]
            }
        """
        results = {}
        for level_name, layer in self.layers.items():
            results[level_name] = layer.search(query_embedding, top_k_per_layer)
        return results

    def get_retrieval_context(self, query_embedding: np.ndarray,
                               top_k_per_layer: int = 3) -> str:
        """
        获取格式化的检索上下文，用于 LLM 生成
        """
        results = self.multi_scale_search(query_embedding, top_k_per_layer)

        context_parts = []

        # Coarse 层: 用户画像
        if results["coarse"]:
            context_parts.append("【用户画像】")
            for entry, dist in results["coarse"]:
                context_parts.append(f"  - {entry.content}")

        # Mid 层: 对话模式/偏好
        if results["mid"]:
            context_parts.append("\n【对话模式/偏好】")
            for entry, dist in results["mid"]:
                context_parts.append(f"  - {entry.content}")

        # Fine 层: 相关对话片段
        if results["fine"]:
            context_parts.append("\n【相关对话片段】")
            for entry, dist in results["fine"]:
                context_parts.append(f"  - {entry.content}")

        return "\n".join(context_parts)

    def stats(self) -> Dict[str, int]:
        """统计各层记忆数量"""
        return {name: len(layer) for name, layer in self.layers.items()}

    def save(self, path: str):
        """保存记忆到文件"""
        data = {
            "embed_dim": self.embed_dim,
            "layers": {}
        }
        for name, layer in self.layers.items():
            data["layers"][name] = {
                "entries": [e.to_dict() for e in layer.entries],
                "id_counter": layer.id_counter
            }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 单独保存 embeddings
        emb_path = path.replace('.json', '_embeddings.npz')
        embeddings_data = {}
        for name, layer in self.layers.items():
            if len(layer.entries) > 0:
                embeddings_data[name] = np.array([e.embedding for e in layer.entries])
        np.savez(emb_path, **embeddings_data)

    def load(self, path: str, encoder_func):
        """从文件加载记忆（需要 encoder_func 重建 embedding）"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载 embeddings
        emb_path = path.replace('.json', '_embeddings.npz')
        embeddings_data = np.load(emb_path)

        for name, layer_data in data["layers"].items():
            layer = self.layers[name]
            layer.id_counter = layer_data["id_counter"]

            embeddings = embeddings_data[name] if name in embeddings_data else []

            for i, entry_dict in enumerate(layer_data["entries"]):
                entry = MemoryEntry(
                    id=entry_dict["id"],
                    embedding=embeddings[i] if i < len(embeddings) else np.zeros(self.embed_dim),
                    content=entry_dict["content"],
                    metadata=entry_dict["metadata"],
                    timestamp=entry_dict["timestamp"]
                )
                layer.entries.append(entry)
                if len(embeddings) > 0:
                    layer.index.add(embeddings[i].reshape(1, -1).astype('float32'))
