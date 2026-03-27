# MSC Benchmark 评估报告

生成时间: 2026-03-27

## 评估任务

| 任务 | 描述 | 指标 |
|------|------|------|
| Persona Retrieval | 从候选中检索正确的 persona | Hits@1, Recall@K, MRR |
| Cross-Session Memory | 跨 Session 利用记忆 | Hit Rate |
| Consistency | 生成内容与 Persona 一致性 | Similarity, Overlap |

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
