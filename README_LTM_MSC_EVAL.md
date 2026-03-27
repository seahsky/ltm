# LTM-MSC 评估方案说明

本文档说明如何将我们的分层长期记忆（LTM）系统接入 MSC 官方评估框架，以及整体架构设计。

---

## 1. 评估对比方案

### 1.1 评估设计

我们在 **相同的评估条件** 下对比三个系统：

| 系统 | 模型 | 记忆来源 | 上下文类型 |
|------|------|----------|-----------|
| **MSC 3B Baseline** | MSC 3B (2.7B) | 无 | `raw_history` |
| **SumMem-MSC 3B** | SumMem-MSC 3B (2.7B) | 官方 Fid-RAG | `predsum_both` |
| **LTM + MSC 3B (Ours)** | MSC 3B (2.7B) | 分层 LTM 检索 | `raw_history` + LTM |

### 1.2 对比方式

```
┌─────────────────────────────────────────────────────────────┐
│                    MSC 验证集 (valid)                        │
│                    20-100 个样本                            │
└─────────────────────┬─────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────────┐
   │MSC 3B   │  │SumMem-  │  │ LTM +       │
   │Baseline  │  │MSC 3B   │  │ MSC 3B      │
   │(无记忆)  │  │(Fid-RAG)│  │(分层LTM)    │
   └────┬────┘  └────┬────┘  └──────┬──────┘
        │             │               │
        ▼             ▼               ▼
   ┌─────────────────────────────────────────────────┐
   │           相同评估指标                              │
   │  • Perplexity (越低越好)                          │
   │  • F1 Score (越高越好)                            │
   │  • Token Accuracy (越高越好)                      │
   └─────────────────────────────────────────────────┘
```

### 1.3 运行命令

**Baseline（无记忆增强）:**
```bash
cd /tmp/ParlAI
conda activate msc_eval

parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --metrics ppl,f1,token_acc \
  -ne 20
```

**LTM + MSC 3B（我们的系统）:**
```bash
cd /tmp/ParlAI
conda activate msc_eval

parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --mutators ltm_augment \
  --metrics ppl,f1,token_acc \
  -ne 20
```

**关键区别**: 只需要添加 `--mutators ltm_augment` 参数

### 1.4 快速对比脚本

```bash
# 一键对比两个系统
cd /home/ec2-user/studies/ltm_agent
./run_msc_quick_eval.sh 20
```

---

## 2. 整体架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ParlAI 评估框架                               │
│                                                                     │
│  ┌───────────────┐      ┌───────────────────────────────────────┐ │
│  │   MSC 数据集   │─────▶│           Mutator 管道                │ │
│  │ (valid.json)  │      │  ┌─────────────────────────────────┐  │ │
│  └───────────────┘      │  │  ltm_augment (我们的实现)        │  │ │
│                         │  │  ┌─────────────────────────────┐  │  │ │
│                         │  │  │   HierarchicalLTMMutator   │  │  │ │
│                         │  │  │  ┌─────────┐  ┌─────────┐  │  │  │ │
│                         │  │  │  │  Fine   │  │ Coarse │  │  │  │ │
│                         │  │  │  │ (对话)  │  │(Persona)│  │  │  │ │
│                         │  │  │  └────┬────┘  └────┬────┘  │  │  │ │
│                         │  │  │       │             │       │  │  │ │
│                         │  │  │       └──────┬──────┘       │  │  │ │
│                         │  │  │              ▼              │  │  │ │
│                         │  │  │     TF-IDF 检索引擎         │  │  │ │
│                         │  │  └─────────────────────────────┘  │  │ │
│                         │  └─────────────────────────────────┘  │  │
│                         └───────────────────────────────────────┘  │
│                                     │                              │
│                                     ▼                              │
│                         ┌───────────────────────┐                 │
│                         │   MSC 3B 生成器       │                 │
│                         │   (2.7B 参数)         │                 │
│                         └───────────┬───────────┘                 │
│                                     │                              │
│                                     ▼                              │
│                         ┌───────────────────────┐                 │
│                         │   评估指标计算        │                 │
│                         │   PPL / F1 / TokenAcc │                │
│                         └───────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
输入文本 (ParlAI 格式):
────────────────────────────
your persona: I like hiking
your persona: I have a dog
Partner: Hello, how are you?
You: I'm doing great!
────────────────────────────
         │
         ▼
┌─────────────────────────────────────────┐
│         ParlAI Mutator 管道             │
│                                         │
│  1. 解析文本，提取 Persona → Coarse 层   │
│  2. 解析文本，提取对话 → Fine 层        │
│  3. 用最后几句对话作为 Query            │
│  4. TF-IDF 检索相关记忆                │
│  5. 记忆注入到输入文本                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
增强后的输入:
────────────────────────────
your persona: I like hiking     ← 从 LTM 检索的 Persona
your persona: I have a dog
your persona: I enjoy outdoors
Your persona: I like hiking     ← 原始 Persona
your persona: I have a dog
Partner: Hello, how are you?
You: I'm doing great!
────────────────────────────
         │
         ▼
┌─────────────────────────────────────────┐
│            MSC 3B 生成器                 │
│                                         │
│  输入: 增强后的上下文                     │
│  输出: 生成的回复                        │
│                                         │
│  计算: PPL, F1, Token Accuracy          │
└─────────────────────────────────────────┘
```

### 2.3 分层 LTM 设计

#### 2.3.1 两层结构

| 层级 | 内容 | 特点 |
|------|------|------|
| **Coarse** | 用户 Persona | 抽象的用户画像，检索时优先级高 |
| **Fine** | 对话片段 | 具体的对话内容，用于理解上下文 |

#### 2.3.2 检索机制

```python
# 检索流程
def retrieve(query, top_k=3):
    # 1. 查询向量 = TF-IDF(最后几句对话)

    # 2. 检索 Coarse 层 (Persona)
    coarse_results = coarse_store.search(query, top_k)

    # 3. 检索 Fine 层 (对话)
    fine_results = fine_store.search(query, top_k)

    # 4. 格式化返回
    return format_memories(coarse_results, fine_results)
```

#### 2.3.3 TF-IDF 实现

```python
class SimpleMemoryStore:
    """轻量级 TF-IDF 检索"""

    def _vectorize(self, text):
        """文本 → TF-IDF 向量"""
        vec = zeros(dim)
        words = tokenize(text)

        for word, freq in word_freq(words):
            if word in vocab:
                idx = vocab[word]
                tf = 1 + log(freq)      # 词频
                idf = self.idf[word]     # 逆文档频率
                vec[idx] = tf * idf

        return normalize(vec)

    def search(self, query, top_k):
        """检索 top-k 相关记忆"""
        query_vec = self._vectorize(query)

        # 余弦相似度
        scores = []
        for content, _, vec in self.entries:
            sim = dot(query_vec, vec)
            scores.append((content, sim))

        return top_k(sorted(scores, key=lambda x: -x[1]))
```

### 2.4 ParlAI Mutator 集成

#### 2.4.1 为什么用 Mutator？

ParlAI Mutator 是数据增强管道的一部分，可以在数据传入模型之前修改数据。这种方式：
- **无需修改模型代码**：直接在数据层面注入 LTM 检索结果
- **与官方框架兼容**：通过 `--mutators` 参数启用
- **可插拔**：可以随时切换不同的 mutator

#### 2.4.2 Mutator 注册

```python
@register_mutator("ltm_augment")
class LTMAugmentMutator(ManyEpisodeMutator):
    """LTM 增强突变器"""

    def many_episode_mutation(self, episode):
        """对每个 episode 注入 LTM 记忆"""
        self.ltm.reset()  # 每个 episode 重置

        history = []
        for message in episode:
            text = message['text']

            # 构建 LTM
            build_ltm_from_text(text, self.ltm)

            # 检索
            query = extract_query(text)  # 最后几句对话
            memories = self.ltm.retrieve(query)

            # 注入
            augmented = memories + "\n" + text
            message['text'] = augmented

            yield [message]
```

### 2.5 与原项目 LTM 的关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                   /home/ec2-user/studies/ltm_agent/                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    dialogue_memory/                          │   │
│  │                                                              │   │
│  │  ├── ltm.py                  ← 完整的分层 LTM (带 FAISS)    │   │
│  │  ├── consolidation.py         ← 记忆巩固 (I = αR + βU + γN)  │   │
│  │  ├── pattern_cluster.py       ← Mid 层聚类 + success_rate    │   │
│  │  ├── reranking.py             ← 回复重排序                   │   │
│  │  ├── train_predictor.py       ← 预测模型训练                  │   │
│  │  ├── train_scorer.py          ← 重要性评分器训练              │   │
│  │  └── msc_benchmark.py         ← MSC 评估封装                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              │ (导出)                              │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  /tmp/ParlAI/parlai/tasks/msc/ltm_mutator.py               │   │
│  │                                                              │   │
│  │  为 ParlAI 评估框架适配的简化版本:                            │   │
│  │  • 使用 TF-IDF 替代 SentenceTransformer (避免依赖冲突)        │   │
│  │  • 两层结构 (Fine + Coarse)                                 │   │
│  │  • 与 ParlAI Mutator 接口兼容                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 当前结果与改进方向

### 3.1 初步评估结果 (20 样本)

| 指标 | MSC 3B Baseline | LTM + MSC 3B (ours) |
|------|------------------|----------------------|
| **Perplexity** | 5.782 | 7.089 (+22.6%) |
| **F1** | 0.2496 | 0.1371 (-45.1%) |
| **Token Accuracy** | 0.5552 | 0.5560 (+0.1%) |

### 3.2 分析

**问题**：
- PPL 上升：TF-IDF 检索质量有限，检索到的记忆可能不够相关
- F1 下降：上下文长度增加导致原始信息被稀释
- Token Accuracy 基本持平

**原因**：
1. **TF-IDF 检索局限**：无法理解语义相似性，只能匹配词形
2. **无记忆筛选**：所有检索到的记忆都被注入，没有重要性评分
3. **无 Mid 层**：缺少主题/模式层的抽象

### 3.3 改进方向

| 组件 | 当前实现 | 改进方案 |
|------|---------|---------|
| **编码器** | TF-IDF | SentenceTransformer (需解决依赖) |
| **记忆筛选** | 无 | 重要性评分 I = αR + βU + γN |
| **Mid 层** | 无 | 主题聚类 + success_rate |
| **重排序** | 无 | 多维度重排序 (S_history + S_memory) |

---

## 4. 文件结构

```
/home/ec2-user/studies/ltm_agent/
├── dialogue_memory/                     # 核心 LTM 实现
│   ├── ltm.py                          # 分层长期记忆 (FAISS)
│   ├── consolidation.py                # 记忆巩固
│   ├── pattern_cluster.py              # Mid 层聚类
│   ├── reranking.py                    # 回复重排序
│   ├── train_predictor.py              # 预测模型
│   ├── train_scorer.py                 # 重要性评分器
│   ├── msc_benchmark.py                # MSC 评估
│   └── ltm_parlai_eval.py             # LTM-ParlAI 集成评估
│
├── eval_results/                        # 评估日志
│   ├── msc3b_baseline.log
│   └── ltm_ours.log
│
├── run_msc_baseline.sh                 # 官方 baseline 脚本
├── run_msc_full_eval.sh                # 完整对比评估脚本
├── run_msc_quick_eval.sh               # 快速对比脚本
│
└── README_LTM_MSC_EVAL.md             # 本文档

/tmp/ParlAI/
└── parlai/tasks/msc/
    └── ltm_mutator.py                  # ParlAI Mutator 实现
```

---

## 5. 依赖关系

```
msc_eval 环境:
├── parlai (源码安装)
├── transformers
├── torch (CUDA 12.1)
├── sentencepiece
└── faiss-cpu  (SumMem-MSC 需要)

注：sentence-transformers 与 ParlAI 依赖冲突，
    所以 ParlAI Mutator 使用简化的 TF-IDF 实现
```

---

## 6. 参考命令

```bash
# 激活环境
conda activate msc_eval
cd /tmp/ParlAI

# 1. MSC 3B Baseline
parlai eval_model -mf zoo:msc/msc3B_1024/model -t msc -dt valid -bs 4 \
  --previous-persona-type raw_history --metrics ppl,f1,token_acc -ne 20

# 2. LTM + MSC 3B (我们的系统)
parlai eval_model -mf zoo:msc/msc3B_1024/model -t msc -dt valid -bs 4 \
  --previous-persona-type raw_history --mutators ltm_augment \
  --metrics ppl,f1,token_acc -ne 20

# 3. SumMem-MSC 3B (需要更多 GPU 内存)
parlai eval_model -mf zoo:msc/summsc_fidrag3B/model -t msc -dt valid -bs 2 \
  --previous-persona-type predsum_both --metrics ppl,f1,token_acc -ne 20
```
