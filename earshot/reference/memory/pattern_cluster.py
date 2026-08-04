"""
模式聚类模块 (Pattern Clustering)
实现 Mid 层聚类机制

核心公式 (来自研究提案 Section 3.4):
C_k = cluster(z_i)  # 对轨迹/对话片段聚类
m_k^mid = (pattern_k, success_rate_k)  # 每个聚类的记忆

功能:
1. 增量聚类 - 新对话片段到来时更新聚类
2. 追踪 success_rate - 每个聚类的成功率
3. 模式提取 - 从聚类中提取可复用的对话模式
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PatternCluster:
    """
    对话模式聚类

    对应研究提案中的 Mid 层记忆:
    m_k^mid = (pattern_k, success_rate_k)
    """
    cluster_id: int
    center: np.ndarray  # 聚类中心 embedding
    pattern_description: str  # 模式描述
    members: List[str] = field(default_factory=list)  # 成员对话片段

    # 统计信息
    total_count: int = 0  # 总出现次数
    success_count: int = 0  # 成功次数
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success_rate(self) -> float:
        """成功率 = 成功次数 / 总次数"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    def update(self, embedding: np.ndarray, content: str, is_successful: bool = True):
        """更新聚类成员"""
        self.members.append(content)
        self.total_count += 1
        if is_successful:
            self.success_count += 1

        # 增量更新中心 (指数移动平均)
        alpha = 0.1  # 学习率
        self.center = (1 - alpha) * self.center + alpha * embedding
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "pattern_description": self.pattern_description,
            "members": self.members,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated
        }


class PatternClusterer:
    """
    对话模式聚类器

    实现:
    1. 增量式 k-means 聚类
    2. 聚类合并 (相似聚类合并)
    3. success_rate 追踪
    """

    def __init__(self,
                 embed_dim: int = 384,
                 distance_threshold: float = 1.5,  # 聚类合并的距离阈值
                 max_clusters: int = 100,  # 最大聚类数
                 min_cluster_size: int = 2):  # 最小聚类大小
        self.embed_dim = embed_dim
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size

        self.clusters: Dict[int, PatternCluster] = {}
        self.cluster_counter = 0

        # 用于模式描述的生成
        self._pattern_templates = [
            "关于 {topic} 的讨论",
            "用户分享 {topic}",
            "询问 {topic} 相关问题",
            "讨论 {topic}",
        ]

    def add_segment(self,
                   embedding: np.ndarray,
                   content: str,
                   pattern_topic: str = None,
                   is_successful: bool = True) -> Tuple[int, bool]:
        """
        添加一个新的对话片段到聚类

        Args:
            embedding: 对话片段的 embedding
            content: 对话内容
            pattern_topic: 模式主题（用于描述）
            is_successful: 该片段是否"成功"（对话继续/用户满意）

        Returns:
            (cluster_id, is_new_cluster) - 所属聚类 ID 和是否为新建聚类
        """
        # 查找最近的聚类
        nearest_cluster_id, min_distance = self._find_nearest_cluster(embedding)

        # 如果距离在阈值内，加入该聚类
        if nearest_cluster_id is not None and min_distance < self.distance_threshold:
            self.clusters[nearest_cluster_id].update(embedding, content, is_successful)
            return nearest_cluster_id, False

        # 否则创建新聚类
        return self._create_cluster(embedding, content, pattern_topic), True

    def _find_nearest_cluster(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """找到最近的聚类"""
        if not self.clusters:
            return None, float('inf')

        min_dist = float('inf')
        min_id = None

        for cluster_id, cluster in self.clusters.items():
            dist = np.linalg.norm(embedding - cluster.center)
            if dist < min_dist:
                min_dist = dist
                min_id = cluster_id

        return min_id, min_dist

    def _create_cluster(self,
                       embedding: np.ndarray,
                       content: str,
                       pattern_topic: str = None) -> int:
        """创建新聚类"""
        # 如果达到最大聚类数，合并最近的聚类
        if len(self.clusters) >= self.max_clusters:
            self._merge_nearest_clusters()

        cluster_id = self.cluster_counter
        self.cluster_counter += 1

        # 生成模式描述
        if pattern_topic:
            pattern_desc = f"模式 {cluster_id}: 关于 {pattern_topic} 的对话"
        else:
            pattern_desc = f"模式 {cluster_id}: {content[:50]}..."

        cluster = PatternCluster(
            cluster_id=cluster_id,
            center=embedding.copy(),
            pattern_description=pattern_desc,
            members=[content]
        )
        cluster.total_count = 1
        cluster.success_count = 1 if cluster else 0

        self.clusters[cluster_id] = cluster
        return cluster_id

    def _merge_nearest_clusters(self):
        """合并最近的聚类"""
        if len(self.clusters) < 2:
            return

        # 找到最近的两个聚类
        min_dist = float('inf')
        merge_pair = None

        cluster_ids = list(self.clusters.keys())
        for i, cid1 in enumerate(cluster_ids):
            for cid2 in cluster_ids[i+1:]:
                dist = np.linalg.norm(
                    self.clusters[cid1].center - self.clusters[cid2].center
                )
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (cid1, cid2)

        if merge_pair and min_dist < self.distance_threshold:
            self._merge_clusters(merge_pair[0], merge_pair[1])

    def _merge_clusters(self, cluster_id1: int, cluster_id2: int):
        """合并两个聚类"""
        c1 = self.clusters[cluster_id1]
        c2 = self.clusters[cluster_id2]

        # 合并成员
        c1.members.extend(c2.members)

        # 合并统计
        c1.total_count += c2.total_count
        c1.success_count += c2.success_count

        # 更新中心 (加权平均)
        total = c1.total_count
        c1.center = (c1.center * c1.total_count + c2.center * c2.total_count) / total

        # 删除被合并的聚类
        del self.clusters[cluster_id2]

    def get_cluster_by_id(self, cluster_id: int) -> Optional[PatternCluster]:
        """获取指定聚类"""
        return self.clusters.get(cluster_id)

    def get_high_success_clusters(self, min_success_rate: float = 0.7) -> List[PatternCluster]:
        """获取高成功率的聚类"""
        return [
            c for c in self.clusters.values()
            if c.success_rate >= min_success_rate and c.total_count >= self.min_cluster_size
        ]

    def get_cluster_for_query(self, query_embedding: np.ndarray) -> Optional[PatternCluster]:
        """获取与查询最相关的聚类"""
        nearest_id, min_dist = self._find_nearest_cluster(query_embedding)
        if nearest_id is not None:
            return self.clusters[nearest_id]
        return None

    def get_all_clusters(self) -> List[PatternCluster]:
        """获取所有聚类"""
        return list(self.clusters.values())

    def stats(self) -> Dict[str, Any]:
        """获取聚类统计"""
        clusters = list(self.clusters.values())
        success_rates = [c.success_rate for c in clusters]

        return {
            "total_clusters": len(clusters),
            "total_members": sum(c.total_count for c in clusters),
            "avg_success_rate": np.mean(success_rates) if success_rates else 0.0,
            "high_success_clusters": len(self.get_high_success_clusters())
        }

    def clear(self):
        """清空所有聚类"""
        self.clusters.clear()
        self.cluster_counter = 0


class MidLayerMemory:
    """
    Mid 层记忆管理器

    整合:
    1. PatternClusterer - 聚类管理
    2. 与 LTM 的交互 - 聚类结果写入 LTM

    对应研究提案 Section 3.4:
    m_k^mid = (pattern_k, success_rate_k)
    """

    def __init__(self,
                 clusterer: PatternClusterer,
                 ltm_layer,
                 encoder_func):
        self.clusterer = clusterer
        self.ltm_layer = ltm_layer  # LTM 的 mid 层
        self.encoder_func = encoder_func

    def add_dialogue_pattern(self,
                            embedding: np.ndarray,
                            content: str,
                            pattern_topic: str = None,
                            is_successful: bool = True) -> Tuple[int, PatternCluster]:
        """
        添加对话模式

        1. 添加到聚类
        2. 如果是新聚类，写入 LTM
        """
        cluster_id, is_new = self.clusterer.add_segment(
            embedding, content, pattern_topic, is_successful
        )
        cluster = self.clusterer.get_cluster_by_id(cluster_id)

        # 如果是新聚类，写入 LTM
        if is_new:
            self._write_cluster_to_ltm(cluster)

        return cluster_id, cluster

    def _write_cluster_to_ltm(self, cluster: PatternCluster):
        """将聚类写入 LTM"""
        content = f"【模式 {cluster.cluster_id}】{cluster.pattern_description}"
        content += f"\n成功率: {cluster.success_rate:.2%}"
        content += f"\n出现次数: {cluster.total_count}"
        content += f"\n示例: {cluster.members[0][:100]}..." if cluster.members else ""

        self.ltm_layer.insert(
            embedding=cluster.center,
            content=content,
            metadata={
                "cluster_id": cluster.cluster_id,
                "pattern_description": cluster.pattern_description,
                "success_rate": cluster.success_rate,
                "total_count": cluster.total_count,
                "type": "pattern_cluster"
            }
        )

    def update_cluster_in_ltm(self, cluster: PatternCluster):
        """更新 LTM 中的聚类信息（需要先删除再插入）"""
        # 简化处理：重新写入
        # 实际应用中可以通过更新而非重建来优化
        self._write_cluster_to_ltm(cluster)

    def get_pattern_for_context(self, context_embedding: np.ndarray) -> Optional[PatternCluster]:
        """获取与当前上下文最相关的模式"""
        return self.clusterer.get_cluster_for_query(context_embedding)

    def get_successful_patterns(self) -> List[PatternCluster]:
        """获取成功的模式"""
        return self.clusterer.get_high_success_clusters()

    def stats(self) -> Dict[str, Any]:
        """统计"""
        cluster_stats = self.clusterer.stats()
        cluster_stats["ltm_size"] = len(self.ltm_layer)
        return cluster_stats
