---
title: Paper精读｜Thinking with Visual Primitives
date: '2026-05-06T18:26:29+08:00'
updated: '2026-11-19T23:59:34+08:00'
tags:
- Paper精读
categories:
- 多模态与具身智能
- Paper
excerpt: 把点和包围框这些空间标记当成“最小思维单元”直接编进多模态 LLM 的思维链，模型推理时不只会说，还能当场指。视觉 token 预算压到极低，成绩反而追平甚至反超 GPT-5.4、Claude-Sonnet-4.6 和 Gemini-3-Flash。
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜Thinking with Visual Primitives

> [!abstract] 一句话结论
> 把点和包围框这些空间标记当成“<span class="mark-strong">最小思维单元</span>”直接编进多模态 LLM 的思维链，模型推理时不只会说，还能当场指。视觉 token 预算压到极低，成绩反而追平甚至反超 GPT-5.4、Claude-Sonnet-4.6 和 Gemini-3-Flash。

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：** Thinking with Visual Primitives
> - **链接**：** [https://www.alphaxiv.org/abs/visual-primitives](https://www.alphaxiv.org/abs/visual-primitives)
> - **作者**：** Ruijie Lu, Yiyang Ma, Xiaokang Chen, Lingxiao Luo, Zhiyu Wu, Zizheng Pan, Xingchao Liu et al. (DeepSeek-AI, Peking University, Tsinghua University)
> - **日期**：** 2026 年 5 月

---

## 1️⃣ 论文想解决什么问题

多模态 LLM 的推理基本困在语言空间里（Chain-of-Thought）。「Thinking with Images」那一拨靠高分辨率裁剪把感知鸿沟（Perception Gap）补上了，但更根本的瓶颈还搁在那儿——**<span class="mark-entity">引用鸿沟 (Reference Gap)</span>**。自然语言天生带歧义，给不出精确、无歧义的空间指向：一句"左边的红色物体"，放到复杂空间布局里可能对应好几个候选，推理链一走到需要严格定位的环节就断。

---

## 2️⃣ 核心洞察

核心思想是：**让模型在推理时<span class="mark-strong">不仅能“说”，还能“指”</span>。** 人和动物在解决空间问题时，天然地使用指向（pointing）来锚定推理。论文将这个能力赋予 MLLM——将点和包围框作为"最小思维单元"穿插在语言推理过程中。模型不需要描述"那个物体在哪里"（语言描述），而是直接在思维链中输出一个点的坐标，指向图像中的对应位置。

---

## 3️⃣ 方法解读

1. **视觉原语注入：** 在推理过程（CoT）中，穿插表示点坐标和包围框的特殊 token。模型可以在思维序列中输出 $(x,y)$ 坐标对来指向图像中的特定位置，或输出 $(x1,y1,x2,y2)$ 来框定空间范围。

2. **极高效的视觉 token 架构：** 图像 token 预算给得极低，性能却没掉出第一梯队——敢跟闭源旗舰对表，这个效率是前提。

3. **"边推理边指向"的认知轨迹：** 模型在思考的每一步都可以"锚定"到图像的具体物理坐标，思维轨迹和图像内容焊在一起。

4. **紧凑模型规模：** 参数规模比对比方法小一大截，表现却仍有竞争力。

---

## 理论贡献

- **识别并命名了"引用鸿沟"：** 把多模态推理的瓶颈从感知层面（看得到吗）挪到认知层面（指得准吗）——光是把这层瓶颈点破并命名，就值得单独记一条

- **提出视觉原语推理范式：** 空间标记升级为思维的基本单元，不再只是语言 CoT 的配角

- **跨模型性能对比：** 在多个挑战性视觉推理任务上，以极低 token 预算达到或超过 <span class="mark-entity">GPT-5.4、Claude-Sonnet-4.6 和 Gemini-3-Flash</span>

---

## 4️⃣ 实验部分怎么读

论文挑的基准全是需要严格空间定位的那类视觉问答：计数、空间推理、拓扑导航。读实验时盯三处：

- 与 GPT-5.4、Claude-Sonnet-4.6 和 Gemini-3-Flash 的直接对比

- 视觉 token 效率分析——论文的核心卖点之一，值得逐个数字核

- 消融实验：插入视觉原语 token 的具体收益到底有多少

---

## 5️⃣ 学术评价

用一个直观到近乎朴素的点子，去捅一个长期没人命名的问题，这是我最吃这类论文的地方。「边想边指」说起来一句话，但 token 怎么编码、视觉原语和语言推理怎么穿插，每个都是需要精巧设计的环节。跟 GPT-5.4 这类闭源模型比还能不落下风，而且是在模型规模更小、图像 token 预算更低的前提下——结果的说服力主要就来自这个不对称。

---

## 局限

先替作者说一句：点和框是最容易嵌进思维链的空间原语，第一刀切在这儿合理。但该提的还是要提：

- 视觉原语只点了名和包围框，其他空间表示形式（如分割掩码）没探索

- 在通用任务上的提升大概率不如空间密集任务上那么显眼

- 与其他 MLLM 能力（如音频、视频）的整合没涉及

---

## 7️⃣ 最后总结

"引用鸿沟"是这篇论文最重要的概念贡献——它指出多模态推理的<span class="mark-strong">下一个瓶颈不是“看得清不清楚”，而是“指得准不准”</span>。以点为思维单元，等于把认知科学里「指向作为认知锚定」的老观点翻译成了能训练、能推理的 token——这个翻译本身，比任何单项分数都更可能被后来者沿用。

---

## 8️⃣ 三句速记

- 多模态推理的瓶颈不在感知鸿沟，在引用鸿沟——语言指不准空间位置

- 点和包围框当成"思维中的坐标"，模型边推理边指向

- 极低 token 预算下，匹配/反超 GPT-5.4 和 Claude-Sonnet-4.6
