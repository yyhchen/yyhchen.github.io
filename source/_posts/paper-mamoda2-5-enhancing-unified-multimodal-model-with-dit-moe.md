---
title: 'Paper精读｜Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE'
date: '2026-05-06T18:26:30+08:00'
updated: '2026-11-19T23:59:34+08:00'
tags:
- Paper精读
categories:
- 多模态与具身智能
- Paper
excerpt: Mamoda2.5 把理解和生成塞进同一个 AR-Diffusion 框架，容量靠 DiT-MoE 堆——128 专家 Top-8 路由，25B 总参数只激活 3B。VBench 2.0 上拿到顶级生成成绩，视频编辑靠蒸馏加 RL 把推理加速做到 95.9 倍，广告场景成功率 98%。
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE

> [!abstract] 一句话结论
> Mamoda2.5 把理解和生成塞进同一个 AR-Diffusion 框架，容量靠 DiT-MoE 堆——128 专家 Top-8 路由，<span class="mark-number">25B 总参数只激活 3B</span>。VBench 2.0 上拿到顶级生成成绩，视频编辑靠蒸馏加 RL 把推理加速做到 <span class="mark-number">95.9 倍</span>，广告场景成功率 <span class="mark-number">98%</span>。

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：** Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE
> - **arXiv**：** 2605.02641
> - **链接**：** [https://arxiv.org/abs/2605.02641](https://arxiv.org/abs/2605.02641)
> - **作者**：** Yangming Shi, Shixiang Zhu, Tao Shen, Zhimiao Yu, Dengsheng Chen et al. (Mamoda Team, ByteDance)
> - **日期**：** 2026-05-04
> - **主页**：** [https://mamoda25.github.io/](https://mamoda25.github.io/)

---

## 1️⃣ 论文想解决什么问题

理解（问答、描述）和生成（图像/视频创建）要一个模型全包，现有方案大多是分开的架构，或者级联拼起来的复杂设计。另一头的账更现实：高质量视频编辑模型动辄 30 步扩散，推理慢到没法进实际业务。核心挑战一句话就能说清——一个架构里，生成质量要往上提，推理成本要往下压。

---

## 2️⃣ 核心洞察

> [!ed-quote] 核心思路
> 核心思路是：**用 Diffusion Transformer 同时做理解和生成，用 MoE <span class="mark-strong">在不增加推理成本的前提下扩大模型容量</span>。**

三个关键设计点：

> [!important] 三条核心洞察
> 1. **AR-Diffusion 统一框架：** 自回归（AR）管理解任务，扩散（Diffusion）管生成任务，共用同一个 Transformer 骨干
>
> 2. **细粒度 MoE：** 128 个专家 + Top-8 路由。<span class="mark-number">25B 总参数，每次推理只激活 3B</span>，训练成本大幅降低
>
> 3. **联合蒸馏 + RL 框架：** 把 30 步编辑模型压成 4 步，再用强化学习把蒸馏掉的质量补回来

---

## 3️⃣ 方法解读

1. **统一 AR-Diffusion 架构：** 把自回归建模（语言和视觉理解用）和扩散过程（图像/视频生成用）融进单一 Transformer。「统一多模态模型」这个方向光喊口号不行，这篇给了一份能跑起来的实现。

2. **DiT-MoE 设计：** 在 Diffusion Transformer 骨干里引入 MoE，128 个专家各自 specialize 在不同的视觉模式上，Top-8 路由让每个 token 只激活 8 个最相关的专家。

3. **效率设计：** 25B 参数只有 3B 活跃——容量约是密集 3B 模型的 8 倍，计算账单却跟 3B 密集模型一个价。这笔账是全文最清楚的一笔。

4. **推理加速管线：** <span class="mark-strong">蒸馏把 30 步扩散编辑模型压成 4 步</span>，再用 RL 微调弥补蒸馏造成的质量损失。对比开源基线，加速 95.9 倍。

---

## 理论贡献

- **大规模 MoE 在统一多模态模型里站住了：** 128 专家这个级别的 MoE 放进视觉生成架构，这篇用结果验证了它不是纸面设计

- **AR + Diffusion 的统一方案：** 「一个模型做所有事」这条路，交出了一份有竞争力的答卷

- **蒸馏 + RL 的二阶段加速范式：** 先蒸馏砍步数、再用 RL 恢复质量，这个两段式管线别的团队可以直接抄

---

## 4️⃣ 实验部分怎么读

- **VBench 2.0 性能：** 先看标准视频生成基准上的排名——生成质量的硬指标就在这儿

- **OpenVE-Bench 视频编辑：** 重点是与 Kling O1（当前顶级商用产品）的对比——Mamoda2.5 声称匹配了它的性能，这条值得自己掂量

- **推理速度：** 95.9x 加速是核心结果之一，盯它与开源基线的详细对比

- **广告场景部署：** 98% 成功率来自真实业务，不是学术基准——全文最「落地」的一条证据

---

## 5️⃣ 学术评价

Mamoda2.5 的亮点在「大容量 + 低计算」这笔权衡上：25B/3B 的 MoE 让它在推理成本不涨的前提下装下更多知识。95.9x 加速和 98% 的业务成功率，说明这套管线不只是 demo 能跑。话说回来，版本号挂在「2.5」上，它更像一次工程迭代而非范式突破——每样组件单拎出来都有前例，新的是把它们叠起来还跑通了。

---

## 局限

先说处境：一套系统里叠着 AR、Diffusion、MoE、蒸馏、RL 五样东西，消融想做干净本来就难，我理解。但该提的还是要提：

- 架构复杂性高：五层组件的叠加让系统整体工程成本下不来

- 论文的消融实验不足以分离各组件（AR、Diffusion、MoE、蒸馏、RL）的独立贡献——哪个在扛大头，读完说不清

- 评测集中在生成和编辑质量，理解侧（如 VQA、caption）的性能没详细披露

---

## 7️⃣ 最后总结

「展示了字节跳动的工程实力」这种话没什么信息量，拆开看更有用：DiT-MoE 的账——<span class="mark-strong">稀疏激活换大容量，推理成本不涨</span>——是全篇最值得带走的一条；95.9x 加速加上 98% 的业务成功率，说明管线过了真业务的关。它是不是范式级贡献，我持保留意见，2.5 这个版本号自己就承认是迭代。下一步最该补的是消融：五个组件到底哪个在扛大头，这个问题不回答，这篇的参考价值就只能停在「整套抄或不抄」。

---

## 8️⃣ 三句速记

- 25B 参数只激活 3B（128 专家 Top-8 路由的 DiT-MoE）

- AR + Diffusion 统一框架，一个模型同时做理解和生成

- 视频编辑加速 95.9 倍，广告场景 98% 成功率
