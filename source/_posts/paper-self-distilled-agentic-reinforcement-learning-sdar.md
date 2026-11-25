---
title: Paper精读｜Self-Distilled Agentic Reinforcement Learning (SDAR)
date: '2026-05-20T13:00:21+08:00'
updated: '2026-11-19T23:59:34+08:00'
tags:
- Paper精读
categories:
- 强化学习与后训练
- Paper
excerpt: SDAR 通过一个Sigmoid 门控机制将 On-Policy Self-Distillation (OPSD) 作为 GRPO 的辅助目标，让每个 token 自主调节蒸馏强度——强化教师认可的 token、软衰减教师可能因检索技能不完美而产生的误判——在 ALFWorld/WebShop/Search-QA 上稳定超越纯 GRPO 和混合基线。
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜Self-Distilled Agentic Reinforcement Learning (SDAR)

> [!abstract] 一句话结论
> SDAR 通过一个**Sigmoid 门控机制**将 On-Policy Self-Distillation (OPSD) 作为 GRPO 的辅助目标，<span class="mark-strong">让每个 token 自主调节蒸馏强度</span>——强化教师认可的 token、软衰减教师可能因检索技能不完美而产生的误判——在 ALFWorld/WebShop/Search-QA 上稳定超越纯 GRPO 和混合基线。
> ---

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：** Self-Distilled Agentic Reinforcement Learning
> - **arXiv**：** 2605.15155
> - **链接**：** [https://www.alphaxiv.org/abs/2605.15155](https://www.alphaxiv.org/abs/2605.15155)
> - **代码**：** [https://github.com/ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)
> - **作者**：** Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen（浙江大学、美团、清华大学）
> - **日期**：** 2026-05-14

---

## 1️⃣ 论文想解决什么问题

Agent 多轮交互的 post-training 面临一个两难格局：

1. **GRPO 等 RL 方法**提供**轨迹级**奖励信号，粒度粗、对长链决策中局部优劣区分力弱

2. **OPSD（On-Policy Self-Distillation）**能从教师分支提供**token 级**密集指导，但直接迁移到多轮 Agent 会出两个问题：

  - **[不稳定] 多轮 OPSD 漂移**：学生一旦偏离教师支持的轨迹，后续 token 的监督信号越来越不可靠，KL 发散飙升、性能崩溃

  - **[不对称信任] 带技能的教师特权上下文不可靠**：负 gap（教师概率 < 学生概率）可能是因为 token 确实不好，但也可能是因为**技能检索不完美或技能质量差**，不能简单压制

简单把 GRPO + OPSD 加起来（naive GRPO+OPSD）会因这些原因导致训练不稳定。

---

---

## 核心洞察：Gated Self-Distillation

> [!ed-quote] 核心洞察
> SDAR 的洞察是：**<span class="mark-strong">不是所有 token 的教师信号都值得信任</span>**，需要一个自适应门控来决定每个 token 的蒸馏强度。

门控本身的设计一句话就能讲完——拿学生-教师的 gap 过一个 Sigmoid，变成 0 到 1 的蒸馏强度：

```
g_t = σ(β · Δ_t)
```

其中 `Δ_t = log π_θ(y_t|s_t) - log π_T(y_t|s_t^+)` 是 reverse-KL 对齐的学生-教师 gap（带 stop-gradient）。

- **Δ_t > 0（正 gap）** → 教师概率 > 学生概率 → 教师在"鼓励"这个 token → sigmoid 输出接近 1 → **强蒸馏**

- **Δ_t < 0（负 gap）** → 教师概率 < 学生概率 → 可能是误判（技能检索出错）或 token 真不好 → sigmoid 输出接近 0 → **软衰减**

- **stop-gradient 作用于 gate** → 门控值只做加权系数，**不产生自指优化路径**，保证梯度稳定

总损失：

```
L = L_GRPO + λ_SDAR · L_SDAR
L_SDAR = Agg(g_t · log π_T(y_t|s_t^+))
```

**理论保证：** Proposition 1–3 证明梯度被 g_t ∈ (0,1) 严格有界，不会超过无权重似然梯度的上界，且引入在线 token 级课程学习（大 gap 的 token 自动获得更大权重）。

---

---

## 3️⃣ 方法解读

### 技能检索

SDAR 利用 SkillBank（基于 SkillRL），在一个多臂老虎机框架下检索与当前任务最相关的技能文件：

- **UCB Retrieval**：标准 UCB 算法，维护每个技能的历史平均奖励，平衡探索-利用

- **Keyword Matching**：基于任务描述关键词匹配类别标签

- **Full / Random Retrieval**：控制对比

技能只在训练时为教师分支提供（特权上下文），推理时不使用——所以推理时无额外开销。

### SDAR 门控架构 vs 其他混合方案

| **方法** | **特点** |
| --- | --- |
| GRPO+OPSD | 简单加和，不稳定 |
| Skill-SD | 带重要性加权的 K3 散度，耦合度高 |
| RLSD | 用 self-teacher gap 重加权 GRPO advantage，但产生自指梯度 |
| **SDAR** | 分离式辅助目标 + Sigmoid 门控 + stop-gradient，**解耦、有界、自适应** |

---

---

## 实验部分

### 基准与模型

- **模型：** Qwen2.5-Instruct 和 Qwen3-Instruct 系列

- **训练：** 150 steps，8×H800 GPU

- **环境：** ALFWorld（家务文本游戏）、WebShop（网购）、Search-QA（搜索增强 QA）

### 核心结果

| **环境** | **相对 GRPO 提升** |
| --- | --- |
| ALFWorld | **+9.4%** |
| Search-QA | **+7.0%** |
| WebShop-Acc | **+10.2%** |

- SDAR **始终优于**纯 GRPO、纯 OPSD、Skill-GRPO、Skill-SD、RLSD 和 GRPO+OPSD

- **避免了** naive GRPO+OPSD 的多轮发散问题

- 对四种不同质量的技能检索策略（UCB/KM/Full/Random）均表现稳健——<span class="mark-strong">即便用 Random 检索，SDAR 仍稳定压过 GRPO</span>

### 消融分析关键发现

1. **gate 不可少**：去掉 Sigmoid 门控退化为普通蒸馏，性能下降

2. **stop-gradient 不可少**：不 detached 会导致梯度耦合，训练不稳定

3. **λ_SDAR 的超参敏感度低**：0.001–0.1 范围内均有效

4. **技能检索质量影响有限**：SDAR 对低质量检索（Random）有天然鲁棒性，因为负 gap 会被门控自动衰减

---

---

## 5️⃣ 学术评价

**贡献：**

1. 指出 OPSD 迁移到多轮 Agent 的两个核心问题（多轮不稳定 + 不对称信任），问题诊断清晰

2. Sigmoid 门控方案干净利落——不需要复杂的课程调度或阈值截断，一个 σ(·) 就搞定了信任衰减

3. 理论这块给得扎实：Proposition 1–3 把梯度有界性和课程学习性质都形式化了，门控不是拍脑袋

4. 实验覆盖全面：3 种环境 × 3 模型系列 × 6+ 基线对比 × 4 种技能检索质量 × 详细消融

**亮点：**

- 推理时零额外开销（技能只在训练时使用）

- 超参鲁棒（λ_SDAR 在 0.001–0.1 范围有效）

- 代码开源

---

---

## 6️⃣ 局限与可扩展方向

> [!ed-warning] 局限（风险与不确定性）
> 1. **技能检索仍是黑盒**：UCB 和 KM 都是相对简单的检索策略，更复杂的检索（语义嵌入、在线学习检索策略）可能进一步提升
>
> 2. **仅在文本环境验证**：ALFWorld/WebShop/Search-QA 都是文本/网页环境，在代码生成、GUI 操作等更复杂的 Agent 场景未测试
>
> 3. **训练成本**：OPSD 需要一个教师前向传播，比纯 GRPO 多一倍计算量
>
> 4. **Sigmoid 门控是对称的**：大正 gap 和大负 gap 的 gating pattern 严格对称，也许需要一个非对称的温度参数来更激进地压制负 gap

---

---

## 8️⃣ 三句速记

> [!summary]- 点开展开
> 1. **问题：** OPSD 给多轮 Agent 当辅助目标时，学生漂移 + 技能误判导致不稳定
>
> 2. **解法：** 用 σ(β·Δ_t) 做 token 级门控，<span class="mark-strong">正 gap 强蒸馏、负 gap 软衰减</span>，stop-gradient 保梯度稳定
>
> 3. **效果：** <span class="mark-number">比 GRPO 平均涨 9 个点</span>，比所有混合基线都好，且对差技能检索天然鲁棒
