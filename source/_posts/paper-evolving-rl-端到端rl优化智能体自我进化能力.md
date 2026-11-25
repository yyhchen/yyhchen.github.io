---
title: 'Paper精读｜Evolving-RL: 端到端RL优化智能体自我进化能力'
date: '2026-05-13T20:50:49+08:00'
updated: '2026-11-19T23:59:34+08:00'
tags:
- Paper精读
categories:
- 强化学习与后训练
- Paper
excerpt: Evolving-RL 用端到端强化学习将智能体的"经验提取"和"经验利用"作为统一过程联合优化，在 ALFWorld 未见任务上相比 GRPO 基线提升 98.7%。
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜Evolving-RL: 端到端RL优化智能体自我进化能力

> [!abstract] 一句话结论
> **Evolving-RL 用端到端强化学习将智能体的"经验提取"和"经验利用"<span class="mark-strong">作为统一过程联合优化</span>，在 ALFWorld 未见任务上相比 GRPO 基线提升 98.7%。**
> ---

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：** Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability within Agents
> - **arXiv**：** 2605.10663
> - **链接**：** [https://arxiv.org/abs/2605.10663](https://arxiv.org/abs/2605.10663)
> - **作者**：** Zhiyuan Fan, Wenwei Jin, Feng Zhang, Bin Li, Yihong Dong, Yao Hu, Jiawei Li (小红书 & 北京大学)
> - **日期**：** 2026-05-11
> - **标签**：** #自我进化智能体 #强化学习 #RL #经验驱动 #小红书 #AIAgent

---

## 1️⃣ 论文想解决什么问题

现有经验驱动的自我进化智能体存在一个根本性缺陷：**提取（extraction）和利用（utilization）是解耦的。**

- 大多数研究只关注系统层面的设计（如何表示经验、如何管理记忆）

- 少量 RL-based 工作也只优化利用阶段，提取阶段依赖外部更强的模型或手工规则

- **<span class="mark-strong">但两者其实是互相制约的</span>：** 如果提取的经验质量差，策略会学会"忽略"这些经验（因为 bypass 不可靠引导比尝试利用它更稳定）

结果是——自我退化的死结。

---

---

## 2️⃣ 核心洞察

> [!ed-quote] 核心洞察
> 提取和利用不是两个分开的能力——<span class="mark-strong">它们是一个共同进化的闭环</span>。

Evolving-RL 的设计基于一个核心观察：<span class="mark-strong">下游任务上的经验注入效果可以成为提取能力的监督信号</span>。提取 -> 评估 -> 利用 -> 再提取，形成一个完整的反馈闭环。

---

---

## 3️⃣ 方法解读

### 架构

- **单策略双角色：** extractor 和 solver 共享同一组策略参数 pi_theta

- **Extractor：** 给定源任务的交互轨迹 -> 生成 N 个候选技能（textual skill）

- **Solver：** 给定目标任务 + 注入技能 -> 产生动作轨迹

### 训练流程（单次迭代）

1. **源任务求解：** solver 在源任务上无技能交互，得到轨迹 tau

2. **经验提取：** extractor 基于 (x_src, tau, r_src) 生成 N 个候选技能

3. **下游评估：** retriever 拉取 K 个语义相关的下游任务，对每个 (skill, task) 对让 solver 做条件 rollout

4. **双重信号回传：**

  - 聚合下游奖励 -> 优化 extractor（提取更好技能）

  - 技能条件轨迹 -> 优化 solver（更好利用技能）

### 双重监督信号

| **信号** | **来源** | **优化目标** |
| --- | --- | --- |
| 提取奖励 R_skill(e) | 技能的 N 个候选在 K 个下游任务上的聚合表现 | 提升 extractor：技能对下游通用性的度量 |
| 利用轨迹 | 技能条件 rollout 采集的轨迹 | 提升 solver：更好利用技能完成任务 |

两者通过 GRPO 进行优化。所有收集的轨迹都被充分利用。

---

---

## 实验

### 基准

- **ALFWorld**（文本家居任务）：Qwen2.5-7B-Instruct

- **Mind2Web**（网页导航）：Qwen2.5-7B-Instruct

### 基线方法

- 提示工程类：ExpeL, Memento, ReasoningBank

- RL 类：GRPO, SkillRL

### 核心结果

| **基准** | **指标** | **Evolving-RL** | **GRPO 基线** | **相对提升** |
| --- | --- | --- | --- | --- |
| ALFWorld (未见任务) | 成功率 | 96.0% | ~48% | **+98.7%** |
| Mind2Web | 动作准确率 | 明显提升 | — | **+35.8%** |

**消融实验的关键发现：**

- 只有 extractor 和 solver **<span class="mark-strong">联合训练</span>**（co-evolution）才能解锁全部收益

- 单独优化 solver 且使用固定/外部 skill 库 -> 收益有限甚至为负

- Evolving-RL 不仅在 test-time 有技能注入时有效，**去掉技能后仍然明显强过 GRPO**——说明<span class="mark-strong">可复用的过程模式已经内化到模型参数中</span>

---

---

## 理论贡献

Appendix A 用中心极限定理分析了技能评估的可靠性。当候选技能数量 K 适中时，排序正确的概率近似为：

P(correct ranking) is approx Phi(Delta_ab / sigma_ab)

其中 Delta_ab 是真实性能差距，sigma_ab 是估计方差。三条推论直接跟着出来：

- **性能差距越大，评估越可靠**

- **采样方差越小，评估越可靠**

- 当 reward 有界时，可以给出保守下界

---

---

## 5️⃣ 学术评价

**贡献：**

1. **统一视角：** 首次将经验提取和利用纳入一个端到端优化框架，而非分开处理

2. **闭环不用外加燃料：** 下游评估产生的双重信号，天然就够两个能力共同进化用

3. **实验结果强硬：** 98.7% 的提升数字很震撼，且消融实验设计严谨（联合训练 vs 分开训练的对比）

4. **理论支撑：** 有公式化的可靠性分析，不是纯实验 paper

**拿不准的地方：**

- 实验只用了 7B 模型，在更大模型（70B+）上是否仍然有效？

- 技能管理（检索、存储）采用了较简单的策略，正如作者自己承认的

- 评估机制引入的估计噪声在某些情况下可能导致训练不稳定

---

---

## 局限与展望

1. **简单的技能管理：** 当前用了较朴素的检索和存储策略，未来可集成更复杂的进化机制

2. **评估噪声：** 有限采样下的评估误差可能注入训练噪声，需要更强的鲁棒性设计

3. **跨模态/跨领域迁移：** 只在文本家居和网页导航上验证，能否迁移到代码、数学等场景？

4. **部署后适应：** 当前聚焦训练阶段，部署后的持续进化是作者提到的未来方向

---

---

## 8️⃣ 三句速记

> [!summary]- 点开展开
> 1. **提取和利用必须一起训练——分开就是死路。** 先有好技能，才能好好用技能，而好技能来自对"用得好不好"的反馈
>
> 2. **98.7% 的未见任务提升来自 RL 闭环，不是 prompt 工程。** 自我进化能力是能被训练出来的，不是只能靠系统设计
>
> 3. **即使推理时不注入技能，Evolving-RL 训练的模型也更强。** 过程模式内化到了参数里，这是"经验增强型 RL"的重要证据
