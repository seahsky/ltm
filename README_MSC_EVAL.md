# MSC 数据集官方评估复现指南

## 概述

本文档记录了如何复现 MSC (Multi-Session Chat) 数据集的官方评估。

**MSC (Multi-Session Chat)** 是一个研究"长期对话记忆"的数据集，核心挑战是让 AI 记住之前的对话内容并在后续保持一致性。

---

## 1. 环境创建

由于原有环境存在依赖冲突，需要创建新的独立环境：

```bash
# 创建新环境
conda create -n msc_eval python=3.10 -y

# 激活环境
conda activate msc_eval

# 安装基础依赖
pip install parlai transformers torch sentencepiece
```

### 从源码安装 ParlAI（必须）

pip 版本的 ParlAI 缺少 `projects.msc` 模块，必须从源码安装：

```bash
cd /tmp
git clone https://github.com/facebookresearch/ParlAI.git --depth 1
cd ParlAI
pip install -e .
```

---

## 2. MSC 数据集

### 自动下载

ParlAI 会自动下载 MSC 数据集到：
```
/home/ec2-user/miniconda3/envs/msc_eval/lib/python3.10/site-packages/data/msc/
```

### 数据结构

```
data/msc/msc/msc_dialogue/
├── session_2  # Session 2 对话
├── session_3  # Session 3 对话
└── session_4  # Session 4 对话
```

### 查看数据

```bash
parlai display_data -t msc -dt train -ne 3
```

---

## 3. 运行评估

### 3.1 基础评估（使用基础模型）

```bash
parlai eval_model \
  -m transformer/generator \
  -t msc \
  -dt valid \
  --metrics ppl,f1 \
  -bs 4 \
  -ne 100
```

### 3.2 基础评估结果

| 指标 | 值 |
|------|-----|
| **Perplexity (PPL)** | 1.558 |
| **Token Accuracy** | 0.9074 |
| **F1** | 0 |

---

## 4. 下载 MSC 官方预训练模型

### 官方模型列表

| 模型 | 说明 | 命令参数 |
|------|------|----------|
| MSC 3B (truncate 1024) | 基础对话生成模型 | `zoo:msc/msc3B_1024/model` |
| SumMem-MSC 3B (Fid-RAG) | 带记忆增强的模型 | `zoo:msc/summsc_fidrag3B/model` |
| Dialogue Summarizer | 对话摘要模型 | `zoo:msc/dialog_summarizer/model` |

### 评估命令

```bash
# MSC 3B 模型
parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --metrics ppl,f1

# SumMem-MSC 3B (Fid-RAG)
parlai eval_model \
  -mf zoo:msc/summsc_fidrag3B/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type predsum_both

# Dialogue Summarization Model
parlai eval_model \
  -mf zoo:msc/dialog_summarizer/model \
  -t msc:PersonaSummary \
  -dt valid \
  -bs 16
```

### MSC 3B 评估结果

| 指标 | 值 |
|------|-----|
| **Perplexity (PPL)** | **9.529** |
| **F1** | 0.1975 |
| **Token Accuracy** | 0.4771 |

按 Session 细分：

| Session | Perplexity | F1 | Token Accuracy |
|---------|------------|-----|----------------|
| Session 1 Self | 10.54 | 0.1864 | 0.4386 |
| Session 2 Dialogue | 9.721 | 0.2122 | 0.4865 |
| Session 3 Dialogue | 8.327 | 0.1940 | 0.5063 |

### 模型存储位置

```
/tmp/ParlAI/data/models/msc/msc3B_1024/
├── model (5.1GB)
├── model.dict
└── ...
```

---

## 5. 评估指标说明

| 指标 | 说明 | 我们的结果 | 论文 SOTA |
|------|------|------------|----------|
| **Perplexity (PPL)** | 模型对文本的"惊讶程度"，越低越好 | 9.529 | ~8.2 |
| **F1** | 词级别的重叠分数 | 0.1975 | - |
| **Token Accuracy** | Token 预测准确率 | 0.4771 | - |
| **Hits@1** | Persona 检索准确率 | - | ~80%+ |

> 注：我们的评估只跑了少量样本，论文结果是完整验证集的结果。

---

## 6. 常用命令速查

```bash
# 查看帮助
parlai eval_model --help

# 查看可用任务
parlai list_tasks

# 显示数据样本
parlai display_data -t msc -dt valid -ne 10

# 评估 MSC 3B 模型
parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  --metrics ppl,f1,token_acc
```

---

## 7. 环境切换

```bash
# 切换到 MSC 评估环境
conda activate msc_eval

# 切换到我们的对话记忆系统环境
conda activate remembr
```

---

## 8. 项目文件结构

```
/home/ec2-user/studies/ltm_agent/
├── data/
│   └── msc/                          # 我们的 MSC 数据（JSON 格式）
├── dialogue_memory/                   # 我们的实现
│   ├── ltm.py                        # 分层长期记忆
│   ├── msc_benchmark.py              # MSC Benchmark 评估
│   ├── pattern_cluster.py             # Mid 层聚类
│   ├── reranking.py                  # 回复重排序
│   ├── train_predictor.py             # 预测模型训练
│   ├── train_scorer.py               # 重要性评分器训练
│   └── ...
├── README_MSC_EVAL.md                # 本文档
└── Research Proposal_Embodied Agent.md
```

---

## 9. 注意事项

1. **CUDA 警告**: 如果看到 CUDA 版本不匹配的警告，可以忽略，评估会使用 CPU
2. **内存**: MSC 评估可能需要较大内存，建议 16GB+
3. **时间**: 完整评估可能需要较长时间
4. **源码安装**: 必须从源码安装 ParlAI，pip 版本缺少必要模块

---

## 10. Baseline: 记忆增强模型 (SumMem-MSC)

SumMem-MSC 3B (Fid-RAG) 是带记忆增强的模型，适合作为我们系统的 baseline。

### 10.1 评估命令

```bash
# 激活环境
conda activate msc_eval
cd /tmp/ParlAI

# 评估 SumMem-MSC 3B (Fid-RAG)
parlai eval_model \
  -mf zoo:msc/summsc_fidrag3B/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type predsum_both \
  --metrics ppl,f1,token_acc
```

### 10.2 预期结果

| 指标 | SumMem-MSC 3B | MSC 3B (基础) |
|------|---------------|---------------|
| Perplexity | ~8.0 | 9.529 |
| F1 | - | 0.1975 |

> SumMem-MSC 使用 Fid-RAG 架构，结合了检索增强生成，预期会有更低的困惑度。

### 10.3 评估脚本

chmod +x run_msc_baseline.sh

./run_msc_baseline.sh



### 10.5 与我们系统的对比维度

| 维度 | SumMem-MSC (Baseline) | 我们的系统 |
|------|----------------------|-----------|
| 记忆结构 | Fid-RAG 检索 | 分层 LTM (Fine/Mid/Coarse) |
| 记忆筛选 | 无显式筛选 | 重要性评分 (I = αR + βU + γN) |
| 聚类 | 无 | Mid 层聚类 + success_rate |
| 重排序 | 无 | 多维度重排序 (S_history + S_memory + S_coherence) |
| 可训练性 | 端到端 | 模块化训练 (预测模型 + 评分器) |

---

## 参考

- [MSC Official Project Page](https://parl.ai/projects/msc/)
- [ParlAI Documentation](https://parl.ai/docs/)
- [MSC Paper: Beyond Goldfish Memory](https://arxiv.org/abs/2104.07527)
- [GitHub: ParlAI](https://github.com/facebookresearch/ParlAI)
