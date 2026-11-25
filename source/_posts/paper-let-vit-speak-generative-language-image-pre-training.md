---
title: 'Paper精读｜Let ViT Speak: Generative Language-Image Pre-training'
date: '2026-05-06T18:26:29+08:00'
updated: '2026-11-19T23:59:34+08:00'
tags:
- Paper精读
categories:
- 多模态与具身智能
- Paper
excerpt: GenLIP 把视觉预训练砍到只剩一个目标：让 ViT 直接做 next token prediction，从视觉 token 预测语言 token。没有对比损失、没有双塔，8B 样本就匹配/超越了吃 40B 样本的强基线，视觉和文本在同一个 transformer 里联合建模。
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜Let ViT Speak: Generative Language-Image Pre-training

> [!abstract] 一句话结论
> GenLIP 把视觉预训练砍到只剩一个目标：让 ViT 直接做 next token prediction，从视觉 token 预测语言 token。没有对比损失、没有双塔，<span class="mark-number">8B 样本就匹配/超越了吃 40B 样本的强基线</span>，视觉和文本在同一个 transformer 里联合建模。

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：** Let ViT Speak: Generative Language-Image Pre-training
> - **arXiv**：** 2605.00809
> - **链接**：** [https://arxiv.org/abs/2605.00809](https://arxiv.org/abs/2605.00809)
> - **作者**：** Yan Fang, Mengcheng Lan, Zilong Huang, Weixian Lei, Yunqing Zhao, Yujie Zhong, Yingchen Yu, Qi She, Yao Zhao, Yunchao Wei (ByteDance, Beijing Jiaotong University, NTU)
> - **日期**：** 2026-05-01

---

## 1️⃣ 论文想解决什么问题

主流 MLLM 的视觉编码器基本都出自 CLIP/SigLIP 这套对比学习预训练，但这里头一直有个被将就着的错位：视觉编码器是**对比学习**练出来的（双编码器架构），LLM 却是**自回归**的（单解码器架构），两边从训练目标到推理方式都不说同一种语言。将就的代价具体有三笔：1）模态之间要加一个连接器（connector）模块来桥接；2）对比学习需要批构造（batch construction），还得单独养一个文本解码器；3）训练效率和数据利用率都吃亏。

---

## 2️⃣ 核心洞察

核心思想很简单：**<span class="mark-strong">让 ViT 直接说（Speak）语言</span>。** 不要用对比学习来对齐视觉和语言，而是用语言建模目标直接训练 ViT 从视觉 token 预测出语言 token。这样视觉编码器的训练目标（next token prediction）就与 LLM 的推理方式（自回归）天然一致了。

---

## 3️⃣ 方法解读

GenLIP 框架的核心设计：

1. **单一 transformer 架构：** 视觉和文本 token 拼在一起，由同一个 transformer 联合建模。输入就是图像 patches 的视觉 token 加上对应的文本 token。

2. **语言建模目标：** 训练目标是标准的 next token prediction——预测文本序列里的下一个语言 token。没有对比损失，也不需要负样本对。

3. **无需额外解码器：** CLIP 得单独养一个文本编码器，GenLIP 一个前向就把两种模态都处理了。

4. **多分辨率续训：** 基础训练之后，在保持原始宽高比的多分辨率图像上继续训，专补细节敏感任务。

---

## 理论贡献

- **简化预训练管线：** 视觉-语言预训练被简化成标准的语言建模任务，对比学习那一整套复杂度——批构造、负样本、双塔——整个消失

- **统一建模范式：** <span class="mark-strong">视觉和语言可以在同一个自回归框架里联合学习</span>，不需要双编码器——这是全文概念上最硬的一条

- **数据效率：** 8B 样本达到或超过 40B 级别方法的水准——数据量差五倍还能打平，生成式目标对样本的利用率比对比式高，这是全文最有分量的证据

---

## 4️⃣ 实验部分怎么读

- **基础预训练实验：** Recap-DataComp-1B 上 8B 样本训练。盯它与 CLIP、SigLIP 在 14 项多模态基准上的对比——这是范式之争的主战场

- **多分辨率续训实验：** 盯 OCR 和图表理解的提升——这俩正是对比学习方法的老短板，补上了才算闭环

- **扩展性实验：** 盯模型和数据规模的缩放曲线——省数据的路子能不能随规模放大，全看这条线

---

## 5️⃣ 学术评价

GenLIP 打动我的不是它加了什么，是它删了什么：批构造、独立文本编码器、对比损失，全没了。ViT 预训练向「生成式统一」的转变，从此有了一份干净的参考实现。数据效率的声明（<span class="mark-number">8B vs 40B</span>）要是能在更多场景站得住，下一代 MLLM 的视觉编码器大概率会顺着这条路走。

---

## 局限

三条局限指向同一件事——现在的证据面还窄：

- 目前只在 Recap-DataComp-1B 上验证过，换数据源还能不能打，未知

- 与专门设计的高分辨率模型（如专门 OCR 模型）的对比还没做充分

- 生成式目标相对对比学习在表征质量（特征空间结构）上的理论差异，论文没深挖

---

## 7️⃣ 最后总结

GenLIP 用最少的工程复杂度做了一次干净的对照：同样是 ViT 预训练，把对比目标换成语言建模，8B 样本就打平了 40B 级别的对比基线。要是这个数据效率在更大规模、更多数据源上站得住，<span class="mark-strong">CLIP/SigLIP 在 MLLM 里的位置就该换人</span>。眼下它还是单一数据集上的结果；但「视觉编码器和 LLM 说同一种语言」这个论点，经过这篇已经是个可检验的假设，不再是口号。

---

## 8️⃣ 三句速记

- GenLIP = 让 ViT 直接用语言建模目标预测文本 token

- 8B 样本训练，达到 40B 级别对比学习方法的水平

- 多分辨率续训把 OCR 和图表理解补上来——对比学习的传统短板
