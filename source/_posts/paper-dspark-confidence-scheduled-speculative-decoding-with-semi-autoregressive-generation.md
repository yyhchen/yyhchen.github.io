---
title: 'Paper精读｜DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation'
date: '2026-07-03T13:06:29+08:00'
updated: '2026-11-19T23:59:34+08:00'
tags:
- Paper精读
categories:
- 模型架构与推理效率
- Paper
excerpt: 并行草稿快是有代价的：各位置独立预测，接受率随位置一路下滑，这就是后缀衰减。DSpark 用半自回归生成（并行主干 + 轻量序列头）把草稿质量按住，再用置信度调度验证让验证长度跟着系统负载走、专裁低置信度后缀 token——两头一起上，在 DeepSeek-V4 生产部署中拿到 60%–85% 的每用户生成加速。
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

> [!abstract] 一句话结论
> 并行草稿快是有代价的：各位置独立预测，接受率随位置一路下滑，这就是<span class="mark-entity">后缀衰减</span>。DSpark 用**半自回归生成**（并行主干 + 轻量序列头）把草稿质量按住，再用**置信度调度验证**让验证长度跟着系统负载走、专裁低置信度后缀 token——两头一起上，在 DeepSeek-V4 生产部署中拿到 **<span class="mark-number">60%–85%</span>** 的每用户生成加速。

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
> - **alphaXiv**：[2026.dspark](https://www.alphaxiv.org/abs/2026.dspark)
> - **论文PDF**：[alphaXiv PDF](https://pdfs.assets.alphaxiv.org/2026.dsparkv1.pdf)
> - **GitHub**：[DeepSpec](https://github.com/deepseek-ai/DeepSpec)（即将开源）
> - **作者**：Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang（共33人）
> - **机构**：北京大学 / DeepSeek-AI
> - **日期**：2026-06-27
> - **会议**：ICML 2026

---

## 1️⃣ 论文想解决什么问题

自回归推理的账很好算：延迟和输出长度成正比，生成多长就串行多少步——GPU 利用率低，用户干等。**推测解码**（Speculative Decoding）的解法是把「起草」和「验证」拆开：轻量草稿模型一口气提出候选 token 块，目标模型单次前向打包验证，拒绝采样接受最长一致前缀。

并行草稿这条路（如 DFlash）看着快，实际有两个绕不开的坑：

1. **生成质量**：并行模型独立预测每个位置，token 之间的依赖没人管，于是出现**多模态碰撞**（如 "of problem"）和快速的**后缀衰减**——接受率随位置急剧下降

2. **系统效率**：无差别验证整个草稿块，等于把目标模型的算力押到高拒绝风险的 token 上，高并发场景下吞吐量掉得最凶

DSpark 两个坑都接：草稿质量靠半自回归补上依赖，验证开销靠自适应的负载感知调度收敛。

---

## 2️⃣ 核心洞察

> [!important] 三条核心洞察
> 1. **并行 vs 自回归，反直觉的地方在第一个 token**：并行草稿在位置 1 的接受率反而远高于自回归模型（Math 上 0.88 vs 0.81）——O(1) 延迟省下的预算能换更深的网络。推测解码是严格的前缀匹配，<span class="mark-strong">第一个 token 的杠杆效应最大</span>，开头的领先就是全局的领先。
>
> 2. **一点自回归就够治后缀衰减**：2 层 DSpark（带 Markov 头）就能压过 5 层 DFlash（纯并行）——<span class="mark-strong">同样的预算，花在轻量序列建模上比堆并行层划算得多</span>。
>
> 3. **<span class="mark-strong">验证预算要看系统负载的脸色</span>**：低负载时多验几个 token 等于白送；高负载时验证低置信度 token 是在挤占别的请求的批容量。置信度调度器不设静态阈值，直接把分配写成全局吞吐量最大化问题去解。

*（原图未能本地化：Figure 1: DSpark 架构总览 — 并行主干 + 序列头 + 置信度调度器组成推测解码流水线；IMA 媒体接口没有可用媒体编号）*

*图1说明：DSpark 的完整解码流程。给定 prompt tokens ABC，目标模型执行一步生成 anchor token D。然后 DSpark 通过并行主干 + 轻量序列头生成 draft tokens EFGH 及其置信度分数 c₁-c₄。硬件感知前缀调度器评估这些分数，保留前缀 EFG 并丢弃低置信度 token H。最后目标模型并行验证截断后的前缀。图中 E 和 F 被接受，G 被拒绝，模型生成修正 token G* 完成当前轮次。*

---

## 3️⃣ 方法解读

### 组件1：半自回归生成（Semi-Autoregressive Generation）

- **并行主干**：基于 DFlash 架构，单次前向产生所有 γ 个草稿位置的隐藏状态和基础 logits。改动只有一处：把 anchor token 本身也作为预测位置，计算量省下一截

- **序列阶段**：在并行基础上叠加轻量自回归修正，用 autoregressive factorization 引入块内依赖

**两种序列头实现：**

- **Markov 头**（默认）：只依赖前一 token 的一阶转移，低秩分解 B = W₁W₂（r=256）实现，存储和计算开销小到基本可以忽略

- **RNN 头**：维护递归状态 sₖ 积累完整前缀历史，通过门控更新实现。实测只比 Markov 头好一点（主要在长块时），实现却复杂一截——所以默认还是 Markov 头

### 组件2：置信度调度验证（Confidence-Scheduled Verification）

**置信度头（Confidence Head）：**

- 轻量线性投影 + sigmoid，输出每个位置的条件接受概率 cₖ

- 监督信号不绕弯子，直接对着 draft 与 target 分布的总变差距离来教：c*ₖ = 1 - ½‖p_d - p_t‖₁

- **Sequential Temperature Scaling (STS)**：从左到右逐位置校准累积乘积概率，把 ECE <span class="mark-number">从 3%-8% 压到 ~1%</span>

**硬件感知前缀调度器（Algorithm 1）：**

- 验证长度不当超参，直接写成全局吞吐量最大化问题：max Θ = τ · SPS(B)

- 贪心求解：按生存概率 aᵣ,ⱼ 全局排序所有候选 token，逐个准入、查表更新吞吐量

- **早停机制**：吞吐量一降就截断。非预期性（non-anticipating）保证这一切不破坏目标分布的精确恢复——加速不拿分布偏移来换

### 优势估计

- **离线**：Qwen3-4B/8B/14B 上，平均接受长度比 Eagle3（自回归）高 26.7%-30.9%，比 DFlash（并行）高 16.3%-18.3%

- **在线**：DeepSeek-V4-Flash/Pro 生产部署中，匹配吞吐量下每用户生成速度比 MTP-1 基线高 60%-85%

- **系统**：严格 SLA 约束下（如 120 tok/s/user），DSpark 维持非退化吞吐量，MTP-1 已经贴着运行边界

---

## 4️⃣ 实验部分怎么读

### 实验设置

- **目标模型**：Qwen3-4B/8B/14B, Gemma4-12B

- **草稿模型对比**：Eagle3（自回归+TTT）, DFlash（并行+KV injection）, DSpark（半自回归）

- **训练数据**：Open-PerfectBlend（130万样本，含 chat/math/code/instruction）

- **评估基准**：Math（GSM8K/MATH500/AIME25）、Code（MBPP/HumanEval/LCB）、Chat（MT-Bench/Alpaca/Arena-Hard）

### 核心结果

| **目标模型** | **基线** | **DSpark 平均接受长度** | **提升** |
| --- | --- | --- | --- |
| Qwen3-4B | Eagle3 | ~6.1 vs ~4.7 | +30.9% |
| Qwen3-4B | DFlash | ~6.1 vs ~5.2 | +16.3% |
| Qwen3-8B | Eagle3 | ~6.2 vs ~4.9 | +26.7% |
| Qwen3-8B | DFlash | ~6.2 vs ~5.2 | +18.4% |
| Qwen3-14B | Eagle3 | ~6.2 vs ~4.8 | +30.0% |
| Qwen3-14B | DFlash | ~6.2 vs ~5.2 | +18.3% |

**关键发现：**

- 结构化任务（Math/Code）的接受率天生高于开放聊天（Chat）

- DSpark 在所有领域和模型上压着两种基线打

- 草稿块越长（γ 越大），DSpark 相对 DFlash 的优势越大（γ=15 时 Math 提升 <span class="mark-number">30%</span>）——长块正是后缀衰减最重的地方，序列头补的就是这块

### Position-wise 条件接受率分析

*（原图未能本地化：Figure 2: 位置-wise 条件接受率 — DSpark 结合并行模型的高初始接受率和自回归模型的后缀稳定性；IMA 媒体接口没有可用媒体编号）*

*图2说明：在 Qwen3-4B 目标模型上，分别在 Math、Code、Chat 领域绘制的位置-wise 条件接受率曲线。自回归模型（Eagle3）保持稳定甚至上升趋势，但并行模型（DFlash）因独立预测后缀衰减严重（Math 0.87→0.78，Chat 0.72→0.63）。DSpark 继承并行模型的高初始接受率（Math 0.93），同时通过序列头保持高且稳定的条件接受率。*

### 训练细节

- **损失函数**：L = 0.1·L_ce + 0.9·L_tv + 1.0·L_conf

- **位置权重**：wₖ = exp(-(k-1)/γ)，强调靠前的位置

- **训练优化**：Hidden state 通信（O(d) 而非 O(V)）、Anchor-bounded sequence packing

- **部署配置**：γ=5, Markov head, 3 MoE 层 + mHC, 滑动窗口注意力 128

---

## 5️⃣ 学术评价

**优点：**

1. 两个瓶颈各给一个对症的组件：草稿质量归半自回归，算力浪费归置信度调度，没有为凑方法而生的设计

2. 位置-wise 条件接受率分析是真有信息量的：并行模型输在哪个位置、序列头补回了什么，一条曲线说清楚

3. 生产部署的说服力是实打实的：DeepSeek-V4 真实流量下的数字比离线 benchmark 硬得多——推测解码的论文不少，敢上生产线的没几个

4. 检查点和训练框架（DeepSpec）开源，可复现

> [!ed-warning] 局限（风险与不确定性）
> 1. 草稿侧的固定成本收不回：接受率极低的复杂查询，并行主干这笔计算白花——6️⃣ 的难度感知早退就是冲着这个留的口子
>
> 2. 调度器假设硬件容量曲线平滑，实际部署里靠异步近似解决，理论保证有所放松
>
> 3. 只在 DeepSeek-V4 上验过，LLaMA、Mistral 这些架构上的通用性还是问号

---

## 生产部署结果

### DeepSeek-V4 线上性能（Figure 7）

*（原图未能本地化：Figure 7: Throughput vs TPS — DSpark 在吞吐量和每用户生成速度间实现帕累托前沿外移；IMA 媒体接口没有可用媒体编号）*

*图7说明：在 DeepSeek-V4-Flash 和 V4-Pro 上的生产部署结果。左图 V4-Flash：在 80 tok/s/user SLA 下吞吐量提升 51%；120 tok/s/user SLA 下吞吐量提升 661%（因为 MTP-1 在该点已接近崩溃边缘）。右图 V4-Pro：35 tok/s/user SLA 下提升 52%；50 tok/s/user SLA 下提升 406%。DSpark 将吞吐量-交互性帕累托前沿向外推移。*

### 负载自适应调度（Figure 8）

*（原图未能本地化：Figure 8: 负载自适应吞吐量与验证预算 — 调度器随并发增加自动收紧验证预算；IMA 媒体接口没有可用媒体编号）*

*图8说明：上行展示 V4-Flash(a) 和 V4-Pro(b) 在不同并发下的聚合吞吐量；下行展示平均验证预算随并发增加而自动减少。在中低并发（V4-Flash <200请求，V4-Pro <150请求）时，调度器利用空闲算力分配 4-6 token 的验证预算；当系统饱和时，调度器动态收紧验证长度以防止资源争用。*

---

## 6️⃣ 局限与可扩展方向

- **难度感知早退**：对低接受率查询跳过完整块生成，把草稿侧的浪费堵上

- **多轮调度联合优化**：当前每步独立调度，跨步联合还有空间

- **跨架构泛化**：非 MoE 架构上的表现没人验过

- **与量化/蒸馏结合**：和模型压缩技术叠加着用

---

## 7️⃣ 最后总结

读完最确定的一件事：把验证长度选择<span class="mark-strong">从启发式阈值提升为形式化的全局吞吐量最大化问题</span>，是这篇真正立得住的贡献——半自回归救的是草稿质量，这一步救的是系统账本，DeepSeek-V4 的部署能成靠的是两件一起。部署结果里最值钱的也不只是吞吐涨了多少，而是此前够不着的严格交互性 SLA 变成了可交付的服务层级。没解决的我也盯着两处：非 MoE 架构没验，低接受率查询的草稿成本收不回——难度感知早退要是做成了，后一个洞能补上。

---

## 8️⃣ 三句速记

> [!summary]- 点开展开
> 1. **并行主干 + 轻量 Markov/RNN 头**：半自回归保住 O(1) 草稿延迟，同时把后缀衰减按住
>
> 2. **置信度调度验证**：验证长度是解出来的，不是拍出来的——全局吞吐量最大化，随负载裁掉低置信度 token
>
> 3. DeepSeek-V4 生产部署中 **60%-85%** 每用户加速，严格 SLA 下的服务层级从够不着变成可交付
