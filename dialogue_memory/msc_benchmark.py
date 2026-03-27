"""
MSC Benchmark 评估模块

MSC 官方评估任务:
1. Persona Retrieval (Hits@1) - 从候选 persona 中检索正确的
2. Dialogue Generation (Perplexity) - 生成下一句
3. Consistency - 生成内容与 persona 的一致性

我们的记忆系统评估:
1. 检索相关记忆/Persona
2. 检索结果的相关性
3. 跨 Session 记忆利用
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from collections import defaultdict

from .encoder import BaseEncoder
from .ltm import HierarchicalLTM


@dataclass
class RetrievalResult:
    """检索结果"""
    retrieved: str  # 检索到的内容
    relevant: str   # 真正相关的内容
    score: float    # 检索分数
    is_correct: bool  # 是否命中


@dataclass
class BenchmarkResult:
    """Benchmark 结果"""
    task_name: str
    metric: str
    value: float
    details: Dict[str, Any] = None


class PersonaRetrievalBenchmark:
    """
    Persona Retrieval 评估

    任务: 给定当前对话上下文，从候选 persona 中检索正确的 persona
    指标: Hits@1, Recall@k, MRR
    """

    def __init__(self, encoder: BaseEncoder, ltm: HierarchicalLTM = None):
        self.encoder = encoder
        self.ltm = ltm or HierarchicalLTM(embed_dim=encoder.embed_dim)

    def evaluate_retrieval(self,
                         context: str,
                         candidates: List[str],
                         relevant: List[str],
                         k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        评估检索性能

        Args:
            context: 当前对话上下文
            candidates: 候选 persona 列表
            relevant: 真正相关的 persona 列表 (ground truth)
            k_values: 评估的 k 值

        Returns:
            {
                "hits@1": float,
                "recall@3": float,
                "recall@5": float,
                "mrr": float,
            }
        """
        # 编码上下文
        context_emb = self.encoder.encode(context)

        # 编码候选
        candidate_embs = np.array([self.encoder.encode(c) for c in candidates])

        # 计算相似度
        similarities = self._compute_similarities(context_emb, candidate_embs)

        # 排序
        ranked_indices = np.argsort(-similarities)

        # 计算指标
        results = {}
        relevant_set = set(relevant)

        # Hits@K
        for k in k_values:
            top_k_indices = ranked_indices[:k]
            hits = sum(1 for i in top_k_indices if candidates[i] in relevant_set)
            results[f"hits@{k}"] = hits / len(relevant_set) if relevant_set else 0.0

        # Recall@K
        for k in k_values:
            top_k_indices = ranked_indices[:k]
            retrieved = set(candidates[i] for i in top_k_indices)
            recall = len(retrieved & relevant_set) / len(relevant_set) if relevant_set else 0.0
            results[f"recall@{k}"] = recall

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, idx in enumerate(ranked_indices, 1):
            if candidates[idx] in relevant_set:
                mrr = 1.0 / rank
                break
        results["mrr"] = mrr

        return results

    def _compute_similarities(self, query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        cand_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)
        return np.dot(cand_norms, query_norm)

    def evaluate_with_ltm(self,
                         context: str,
                         target_persona: List[str],
                         other_personas: List[str],
                         k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        评估使用 LTM 检索的性能

        Args:
            context: 当前对话上下文
            target_persona: 真正相关的 persona
            other_personas: 其他不相关的 persona
            k_values: 评估的 k 值
        """
        # 合并候选
        all_candidates = target_persona + other_personas
        relevant = target_persona

        return self.evaluate_retrieval(context, all_candidates, relevant, k_values)


class CrossSessionMemoryBenchmark:
    """
    跨 Session 记忆评估

    评估系统是否能利用之前 Session 的信息：
    1. Session N 的对话能检索到 Session 0 的 Persona
    2. Session N 的对话能检索到 Session 0 的关键对话
    """

    def __init__(self, encoder: BaseEncoder, ltm: HierarchicalLTM = None):
        self.encoder = encoder
        self.ltm = ltm or HierarchicalLTM(embed_dim=encoder.embed_dim)

    def build_memory_from_session(self,
                                 session_data: Dict,
                                 level: str = "coarse") -> int:
        """
        将 Session 数据写入 LTM

        Args:
            session_data: Session 数据
            level: 写入的层级 (fine/mid/coarse)

        Returns:
            写入的记忆数量
        """
        count = 0

        # 写入 persona 到 Coarse 层
        for persona in session_data.get('persona1', []):
            emb = self.encoder.encode(persona)
            self.ltm.insert(
                level="coarse",
                embedding=emb,
                content=persona,
                metadata={"type": "persona", "session_id": session_data['session_id']}
            )
            count += 1

        # 写入对话到 Fine 层
        for i, utterance in enumerate(session_data.get('dialogue', [])):
            if session_data['speaker'][i] == 'Speaker 1':  # 只写入用户的话
                emb = self.encoder.encode(utterance)
                self.ltm.insert(
                    level="fine",
                    embedding=emb,
                    content=utterance,
                    metadata={"type": "dialogue", "session_id": session_data['session_id']}
                )
                count += 1

        return count

    def evaluate_cross_session_retrieval(self,
                                       query_session: Dict,
                                       k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        评估跨 Session 检索

        Args:
            query_session: 查询用的 Session (通常是 Session N)
            k_values: 评估的 k 值

        Returns:
            检索结果统计
        """
        results = {"fine": [], "coarse": []}

        # 从 query_session 中采样查询
        for i, utterance in enumerate(query_session.get('dialogue', [])):
            if query_session['speaker'][i] != 'Speaker 1':
                continue

            query_emb = self.encoder.encode(utterance)

            # 检索
            for level in ["fine", "coarse"]:
                retrieval = self.ltm.search(level, query_emb, top_k=max(k_values))

                # 检查是否命中同对话组的记忆
                query_dial_id = query_session.get('dialogue_id', '')
                hits = 0
                for entry, dist in retrieval:
                    entry_dial_id = entry.metadata.get('dialogue_id', '')
                    if entry_dial_id == query_dial_id and dist < 3.0:  # 距离阈值
                        hits += 1

                results[level].append({
                    "query": utterance[:50],
                    "hits": hits,
                    "total": len(retrieval)
                })

        # 汇总
        summary = {}
        for level, level_results in results.items():
            if level_results:
                avg_hits = np.mean([r["hits"] for r in level_results])
                avg_total = np.mean([r["total"] for r in level_results])
                summary[level] = {
                    "avg_hits": avg_hits,
                    "avg_retrieved": avg_total,
                    "hit_rate": avg_hits / avg_total if avg_total > 0 else 0.0
                }

        return summary

    def reset_ltm(self):
        """重置 LTM"""
        self.ltm = HierarchicalLTM(embed_dim=self.encoder.embed_dim)


class DialogueGenerationBenchmark:
    """
    对话生成评估

    由于没有真实 LLM 评估 Perplexity，这里提供一个简化的评估框架
    实际使用时需要接入 LLM API
    """

    def __init__(self, encoder: BaseEncoder):
        self.encoder = encoder

    def evaluate_consistency(self,
                          generated: str,
                          persona: List[str]) -> Dict[str, float]:
        """
        评估生成内容与 Persona 的一致性

        方法:
        1. 计算生成内容与 persona 的语义相似度
        2. 检测 persona 关键词是否在生成内容中出现
        """
        generated_emb = self.encoder.encode(generated)

        # 计算与每个 persona 的相似度
        similarities = []
        for p in persona:
            p_emb = self.encoder.encode(p)
            sim = self._cosine_similarity(generated_emb, p_emb)
            similarities.append(sim)

        return {
            "avg_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities)
        }

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)

    def evaluate_keyword_overlap(self,
                               generated: str,
                               context: str) -> float:
        """
        评估生成内容与上下文的关键词重叠

        返回: 重叠比例 [0, 1]
        """
        # 简单关键词提取
        def extract_keywords(text):
            text = text.lower()
            # 去除停用词
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it'}
            words = [w.strip('.,!?') for w in text.split()]
            return set(w for w in words if len(w) > 2 and w not in stopwords)

        gen_keywords = extract_keywords(generated)
        ctx_keywords = extract_keywords(context)

        if not gen_keywords or not ctx_keywords:
            return 0.0

        overlap = len(gen_keywords & ctx_keywords)
        return overlap / len(gen_keywords)


class MSCEvaluator:
    """
    MSC Benchmark 综合评估器

    整合所有评估任务
    """

    def __init__(self, encoder: BaseEncoder, ltm: HierarchicalLTM = None):
        self.encoder = encoder
        self.persona_retrieval = PersonaRetrievalBenchmark(encoder, ltm)
        self.cross_session = CrossSessionMemoryBenchmark(encoder, ltm)
        self.generation = DialogueGenerationBenchmark(encoder)

    def run_full_evaluation(self,
                           data: Dict[str, List],
                           sample_size: int = 100) -> Dict[str, BenchmarkResult]:
        """
        运行完整评估

        Args:
            data: MSC 数据
            sample_size: 采样大小

        Returns:
            各项评估结果
        """
        results = {}

        print("\n" + "=" * 60)
        print("MSC Benchmark 评估")
        print("=" * 60)

        # 1. Persona Retrieval
        print("\n【1】Persona Retrieval 评估")
        retrieval_results = self._evaluate_persona_retrieval(data, sample_size)
        results["persona_retrieval"] = retrieval_results
        self._print_retrieval_results(retrieval_results)

        # 2. Cross-Session Memory
        print("\n【2】跨 Session 记忆评估")
        cross_session_results = self._evaluate_cross_session(data, sample_size)
        results["cross_session"] = cross_session_results
        self._print_cross_session_results(cross_session_results)

        # 3. Consistency
        print("\n【3】一致性评估")
        consistency_results = self._evaluate_consistency(data, sample_size)
        results["consistency"] = consistency_results
        self._print_consistency_results(consistency_results)

        return results

    def _evaluate_persona_retrieval(self,
                                   data: Dict,
                                   sample_size: int) -> Dict[str, float]:
        """评估 Persona 检索"""
        hits_at_1 = []
        hits_at_3 = []
        hits_at_5 = []
        mrrs = []

        sample_data = list(data.items())[:sample_size]

        for group_id, sessions in sample_data:
            if len(sessions) < 2:
                continue

            # 使用 Session 0 作为记忆，后续 Session 作为查询
            session_0 = sessions[0]

            for session_n in sessions[1:]:
                # 构建上下文
                context = " ".join(session_n['dialogue'][:3])

                # 目标 persona
                target = session_n.get('persona1', [])

                if not target:
                    continue

                # 构建负样本 (从其他对话组采样)
                other_personas = []
                for other_id, other_sessions in list(data.items())[:50]:
                    if other_id != group_id:
                        for s in other_sessions:
                            other_personas.extend(s.get('persona1', []))
                            if len(other_personas) >= 10:
                                break
                    if len(other_personas) >= 10:
                        break

                # 评估
                try:
                    result = self.persona_retrieval.evaluate_with_ltm(
                        context=context,
                        target_persona=target,
                        other_personas=other_personas[:10],
                        k_values=[1, 3, 5]
                    )
                    hits_at_1.append(result.get('hits@1', 0))
                    hits_at_3.append(result.get('recall@3', 0))
                    hits_at_5.append(result.get('recall@5', 0))
                    mrrs.append(result.get('mrr', 0))
                except:
                    pass

        return {
            "hits@1": np.mean(hits_at_1) if hits_at_1 else 0.0,
            "recall@3": np.mean(hits_at_3) if hits_at_3 else 0.0,
            "recall@5": np.mean(hits_at_5) if hits_at_5 else 0.0,
            "mrr": np.mean(mrrs) if mrrs else 0.0,
            "sample_count": len(hits_at_1)
        }

    def _evaluate_cross_session(self,
                               data: Dict,
                               sample_size: int) -> Dict[str, Any]:
        """评估跨 Session 记忆"""
        fine_hit_rates = []
        coarse_hit_rates = []

        sample_data = list(data.items())[:sample_size]

        for group_id, sessions in sample_data:
            if len(sessions) < 2:
                continue

            # 重建 LTM
            self.cross_session.reset_ltm()

            # 写入 Session 0 的记忆
            for session in sessions[:2]:  # 前两个 session 作为记忆
                self.cross_session.build_memory_from_session(session)

            # 用后续 session 测试检索
            for session_n in sessions[2:]:
                result = self.cross_session.evaluate_cross_session_retrieval(
                    session_n, k_values=[3]
                )

                if 'fine' in result:
                    fine_hit_rates.append(result['fine'].get('hit_rate', 0))
                if 'coarse' in result:
                    coarse_hit_rates.append(result['coarse'].get('hit_rate', 0))

        return {
            "fine_hit_rate": np.mean(fine_hit_rates) if fine_hit_rates else 0.0,
            "coarse_hit_rate": np.mean(coarse_hit_rates) if coarse_hit_rates else 0.0,
            "sample_count": len(fine_hit_rates)
        }

    def _evaluate_consistency(self,
                             data: Dict,
                             sample_size: int) -> Dict[str, float]:
        """评估一致性"""
        similarities = []
        overlaps = []

        sample_data = list(data.items())[:sample_size]

        for group_id, sessions in sample_data:
            for session in sessions:
                persona = session.get('persona1', [])
                dialogue = session.get('dialogue', [])

                for utterance in dialogue[:5]:  # 采样前几句
                    if utterance:
                        sim_result = self.generation.evaluate_consistency(utterance, persona)
                        overlap_result = self.generation.evaluate_keyword_overlap(
                            utterance,
                            " ".join(persona)
                        )
                        similarities.append(sim_result['avg_similarity'])
                        overlaps.append(overlap_result)

        return {
            "avg_similarity": np.mean(similarities) if similarities else 0.0,
            "avg_keyword_overlap": np.mean(overlaps) if overlaps else 0.0,
            "sample_count": len(similarities)
        }

    def _print_retrieval_results(self, results: Dict):
        print(f"  Hits@1:  {results['hits@1']:.4f}")
        print(f"  Recall@3: {results['recall@3']:.4f}")
        print(f"  Recall@5: {results['recall@5']:.4f}")
        print(f"  MRR:      {results['mrr']:.4f}")
        print(f"  样本数:   {results['sample_count']}")

    def _print_cross_session_results(self, results: Dict):
        print(f"  Fine 层命中率:  {results['fine_hit_rate']:.4f}")
        print(f"  Coarse 层命中率: {results['coarse_hit_rate']:.4f}")
        print(f"  样本数: {results['sample_count']}")

    def _print_consistency_results(self, results: Dict):
        print(f"  平均相似度: {results['avg_similarity']:.4f}")
        print(f"  关键词重叠: {results['avg_keyword_overlap']:.4f}")
        print(f"  样本数: {results['sample_count']}")


def generate_msc_benchmark_report(results: Dict[str, Any]) -> str:
    """生成 MSC Benchmark 报告"""
    report = f"""# MSC Benchmark 评估报告

生成时间: 2026-03-27

## 评估任务

| 任务 | 描述 | 指标 |
|------|------|------|
| Persona Retrieval | 从候选中检索正确的 persona | Hits@1, Recall@K, MRR |
| Cross-Session Memory | 跨 Session 利用记忆 | Hit Rate |
| Consistency | 生成内容与 Persona 一致性 | Similarity, Overlap |

## 1. Persona Retrieval

| 指标 | 值 |
|------|------|
| Hits@1 | {results['persona_retrieval']['hits@1']:.4f} |
| Recall@3 | {results['persona_retrieval']['recall@3']:.4f} |
| Recall@5 | {results['persona_retrieval']['recall@5']:.4f} |
| MRR | {results['persona_retrieval']['mrr']:.4f} |
| 样本数 | {results['persona_retrieval']['sample_count']} |

## 2. Cross-Session Memory

| 层级 | 命中率 |
|------|--------|
| Fine | {results['cross_session']['fine_hit_rate']:.4f} |
| Coarse | {results['cross_session']['coarse_hit_rate']:.4f} |
| 样本数 | {results['cross_session']['sample_count']} |

## 3. Consistency

| 指标 | 值 |
|------|------|
| 平均相似度 | {results['consistency']['avg_similarity']:.4f} |
| 关键词重叠 | {results['consistency']['avg_keyword_overlap']:.4f} |
| 样本数 | {results['consistency']['sample_count']} |

## 分析

### Persona Retrieval
- Hits@1 表示系统能否在候选中找到最重要的 1 个相关 persona
- MRR (Mean Reciprocal Rank) 综合评估排序质量

### Cross-Session Memory
- 评估系统跨时间利用历史信息的能力
- Fine 层: 具体对话片段
- Coarse 层: Persona 记忆

### Consistency
- 评估生成内容与 Persona 描述的一致性
- 高相似度和高关键词重叠表示一致性好
"""
    return report
