**3 方法（Method）**

**3.1 方法概述（Overview）**

现有的具身智能系统（如
ReMEmbR）已经证明，通过从历史经验中检索相关记忆可以显著提升机器人在复杂任务中的规划能力。然而，这类方法通常将记忆限制在**当前任务或有限的
episode-level memory bank**
中，缺乏一种能够**跨任务、跨环境持续积累和组织经验的长期记忆机制**。因此，机器人难以像人类一样通过长期经验积累逐渐提升能力。

为了解决这一问题，我们提出 **LTM-Embodied Agent**，一种基于 **ReMEmbR**
的终身长期记忆扩展框架。该方法在保留 ReMEmbR
感知、规划和检索机制的基础上，引入一个**分层长期记忆系统（Hierarchical
Long-Term Memory）**，使机器人能够：

- 从短期交互轨迹中自动筛选关键经验

- 将经验压缩为可检索的长期记忆表示

- 在未来任务中通过多粒度检索复用历史经验

- 利用历史成功模式对候选计划进行显式重排序

整体框架如图所示。系统主要由以下四个核心模块组成：

1.  **短期记忆（Short-Term Memory, STM）**

2.  **生物启发的记忆巩固机制（Memory Consolidation）**

3.  **分层长期记忆（Hierarchical Long-Term Memory, LTM）**

4.  **记忆增强规划与重排序（Memory-guided Planning and Re-ranking）**

在任务执行过程中，机器人首先通过 **ReMEmbR backbone**
与环境交互，并将交互轨迹暂存于 STM。当任务结束时，系统通过记忆巩固模块从
STM
中筛选关键经验并写入长期记忆库。在新任务执行时，机器人根据当前观测构造检索查询，从
LTM 中进行多粒度检索，并将相关经验用于规划生成和候选计划重排序。

**3.2 ReMEmbR Backbone**

我们的系统建立在 **ReMEmbR** 的具身智能框架之上。ReMEmbR
的核心思想是通过检索历史经验来辅助机器人进行决策和规划。具体而言，在每个时间步
$t$，机器人接收当前观测：

$$o_{t} = \{ RGB_{t},Depth_{t},Language_{t}\}$$

并通过感知编码器将其映射为特征表示：

$$h_{t} = f_{\theta}(o_{t})$$

其中 $f_{\theta}$表示视觉---语言编码器。

ReMEmbR 使用当前观测表示
$h_{t}$作为查询，从记忆库中检索最相关的历史经验：

$$M_{t} = Retrieve(h_{t}\mathcal{,M)}$$

其中 $\mathcal{M}$表示记忆集合。

随后，检索到的记忆被作为上下文输入到规划模块中，生成候选动作或计划：

$$P = \{ p_{1},p_{2},...,p_{K}\}$$

然而，ReMEmbR 的记忆使用方式仍存在两个重要局限：

1.  **记忆缺乏长期积累机制**：记忆通常仅在有限范围内维护。

2.  **缺乏结构化组织**：不同类型的经验被统一存储，难以支持不同层级的知识复用。

因此，我们在 ReMEmbR
的基础上提出一种**终身分层记忆系统**，用于实现经验的长期存储与高效检索。

Reference:

\[1\] Anwar, A., Welsh, J., Biswas, J., Pouya, S., & Chang, Y. (2025,
May). Remembr: Building and reasoning over long-horizon spatio-temporal
memory for robot navigation. In *2025 IEEE International Conference on
Robotics and Automation (ICRA)* (pp. 2838-2845). IEEE.

**3.3 生物启发的记忆巩固机制（Memory Consolidation）**

机器人在每个任务执行过程中会产生大量交互轨迹：

$$\tau = \{(o_{t},a_{t},r_{t})\}_{t = 1}^{T}$$

其中：

- $o_{t}$表示观测

- $a_{t}$表示动作

- $r_{t}$表示奖励或任务成功信号

若直接存储所有轨迹，将导致记忆规模迅速膨胀，并引入大量噪声经验。因此，我们设计一种**生物启发的记忆巩固机制**，用于从短期轨迹中筛选关键经验并写入长期记忆。

首先将完整轨迹划分为多个片段：

$$\tau = \{\tau_{1},\tau_{2},...,\tau_{n}\}$$

然后对每个片段计算**记忆重要性评分**：

$$I(\tau_{i}) = \alpha R_{i} + \beta U_{i} + \gamma N_{i}$$

其中：

- $R_{i}$表示该片段对任务成功的贡献

- $U_{i}$表示模型预测误差（surprise）

- $N_{i}$表示相对于已有记忆的新颖度

新颖度定义为：

$$N_{i} = \underset{m_{j} \in LTM}{\min} \parallel z_{i} - z_{j} \parallel_{2}$$

其中 $z_{i}$表示轨迹片段编码。

最终仅保留评分最高的若干片段：

$$\mathcal{K =}TopK(\tau_{i},I(\tau_{i}))$$

并将其写入长期记忆库。

该机制受到海马体记忆巩固理论（Hippocampal Memory
Consolidation）启发，可以有效减少冗余经验并提高检索质量。

Reference:

\[1\] Sridhar, A., Pan, J., Sharma, S., & Finn, C. (2025). Memer:
Scaling up memory for robot control via experience retrieval. *arXiv
preprint arXiv:2510.20328*.

\[2\] Lei, M., Cai, H., Cui, Z., Tan, L., Hong, J., Hu, G., \... & Han,
Y. (2025). RoboMemory: A Brain-inspired Multi-memory Agentic Framework
for Lifelong Learning in Physical Embodied Systems. In *NeurIPS 2025
Workshop on Space in Vision, Language, and Embodied AI*.

**3.4 分层长期记忆结构（Hierarchical Long-Term Memory）**

为了支持跨任务知识迁移，我们构建一个**三层记忆结构**：

  ----------------------------------------
  **层级**   **记忆类型**   **内容**
  ---------- -------------- --------------
  Fine       轨迹记忆       具体操作序列

  Mid        成功模式       任务策略

  Coarse     语义知识       affordance /
                            规律
  ----------------------------------------

**细粒度记忆（Fine-grained Memory）**

细粒度记忆记录关键轨迹片段：

$$m_{i}^{fine} = (z_{i},a_{1:T})$$

其中：

$$z_{i} = Encoder(\tau_{i})$$

该层主要用于复现具体操作过程。

**中粒度记忆（Pattern Memory）**

为了抽象出可迁移的任务模式，我们对成功轨迹进行聚类：

$$C_{k} = cluster(z_{i})$$

并为每个 cluster 构建任务模式：

$$m_{k}^{mid} = (pattern_{k},success\_ rate_{k})$$

例如：

open drawer → pull handle\
success rate = 0.82

**粗粒度记忆（Semantic Memory）**

在更高层级，我们抽象出语义知识：

drawer → pullable\
door → rotatable\
cup → graspable

这些语义信息能够在新环境中提供通用指导。

**3.5 多尺度记忆检索（Multi-scale Retrieval）**

当机器人执行新任务时，首先构建查询表示：

$$q = f(o_{current},instruction)$$

随后进行分层检索：

1️⃣ **语义层检索**

检索相关 affordance 知识：

$$M_{coarse} = Retrieve(q,LTM_{coarse})$$

2️⃣ **模式层检索**

获取相关成功策略：

$$M_{mid} = Retrieve(q,LTM_{mid})$$

3️⃣ **轨迹层检索**

获取具体操作经验：

$$M_{fine} = Retrieve(q,LTM_{fine})$$

最终组合为：

$$M = M_{coarse} \cup M_{mid} \cup M_{fine}$$

并输入到规划模块。

Reference:

\[1\] Torne, M., Pertsch, K., Walke, H., Vedder, K., Nair, S., Ichter,
B., \... & Driess, D. (2026). MEM: Multi-Scale Embodied Memory for
Vision Language Action Models. *arXiv preprint arXiv:2603.03596*.

**3.6 记忆增强计划重排序（Memory-guided Plan Re-ranking）**

ReMEmbR 通常通过 LLM 或策略模型生成候选计划：

$$P = \{ p_{1},p_{2},...,p_{K}\}$$

然而，单次生成的计划可能存在不稳定性。因此，我们提出一种**记忆增强的计划重排序机制**。

对于每个候选计划 $p_{k}$，计算综合评分：

$$Score(p_{k}) = w_{1}S_{succ} + w_{2}S_{sim} + w_{3}S_{phys}$$

其中：

- $S_{succ}$：历史成功率

- $S_{sim}$：与历史轨迹的相似度

- $S_{phys}$：物理可行性

最终选择评分最高的计划执行：

$$p^{*} = \arg\underset{p_{k}}{\max}Score(p_{k})$$

该机制能够利用历史经验显式影响决策，从而提高长任务中的稳定性。

Reference:

\[1\] Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David,
B., \... & Zeng, A. (2022). Do as i can, not as i say: Grounding
language in robotic affordances. *arXiv preprint arXiv:2204.01691*.

\[2\] Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L.,
Wiegreffe, S., \... & Clark, P. (2023). Self-refine: Iterative
refinement with self-feedback. *Advances in neural information
processing systems*, *36*, 46534-46594.

\[3\] Wang, Z., Yu, B., Zhao, J., Sun, W., Hou, S., Liang, S., \... &
Gan, Y. (2025, May). Karma: Augmenting embodied ai agents with
long-and-short term memory systems. In *2025 IEEE International
Conference on Robotics and Automation (ICRA)* (pp. 1-8). IEEE.

**方法总结**

我们提出的 **LTM-Embodied Agent** 在 ReMEmbR
框架基础上引入了一个**终身分层长期记忆系统**。该系统通过：

- 生物启发的记忆巩固机制

- 分层记忆组织

- 多尺度记忆检索

- 记忆增强规划重排序

使机器人能够持续积累经验，并在新环境中高效复用历史知识。
