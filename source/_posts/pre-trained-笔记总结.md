---
title: pre-trained 笔记总结
date: 2024-09-25 12:25:46
tags: [LLM, pre-trained]
categories: [LLM]
banner_img: /img/pretrain_task.png
---

# pretrain 笔记总结

>来自知乎作者 ybq [LLM训练-pretrain](https://zhuanlan.zhihu.com/p/718354385)

---


这篇文章是关于如何从零开始进行大型语言模型（LLM）的预训练（pretrain）工作的详细介绍。文章分为多个部分，涵盖了预训练的背景、数据篇、训练篇、评估篇。下面是对文章内容的详尽笔记：

### 背景篇
- 当前，大型模型(dense)如`qwen`、MOE模型`deepseek`、小型模型`minicpm`等已经开源，但训练框架和数据等核心内容仍然闭源。
- 自研预训练模型的意义在于能够为模型迭代做出贡献，优化模型以适应特定领域，以及更好地控制模型的对齐阶段。

<br>

### 数据篇
- **数据爬取**：需要大量数据，可以通过爬网页、购买等方式获取。
- **数据清洗**：使用模型对数据质量打分，选择BERT家族模型进行训练，结合**规则**进行数据过滤。(规则非常重要！)
- **数据去重**：使用minhash等技术对大规模数据进行去重(hadoop, spark)。
- **数据配比**：根据数据类型（如新闻、百科、代码）进行分类，并调整各类数据的比例。
- **数据顺序**：通过**课程学习**的方式，合理安排训练数据的**顺序**，以优化模型学习效果。（学习顺序非常影响效果, 作者推荐 llama 的 [in-context pretrain](#in-context-pretrain)）
- **数据流水线**：pretrain 的两个进程是独立的：“数据处理进程”和“模型训练进程”。避免 GPU 空闲

> 多读各公司的技术报告, 不要随便创新，落地的话跟着别人的架构才能避免踩坑

<br>

### 训练篇
- **Tokenizer**：选择合适的分词器，注意词汇表的扩展和压缩率。（原文有几个细节！**扩词表并不会使得效果特别好**。）
- **模型结构**：建议模仿`llama`的结构，**避免**不必要的创新。（[rope + gqa + rms_norm + swiglu](#rope--gqa--rms_norm--swiglu)，目的是防止踩坑，特别是落地需要快并且省钱）
- **模型参数**：选择合适的模型大小和超参数，考虑训练和推理的算力。
- **训练框架**：选择适合的框架，如Megatron或DeepSpeed，根据训练数据量和模型大小。(`DeepSpeed` 低代码，但是性能损失多，而且慢；`Megatron` bug多，但优化好，是真的。 **T 级别的 token 训练量必须是 megatron**，B 的级别 token 训练量无所谓。)
- **训练技巧**：优化训练效率，分析loss曲线，调整学习率等。（在显存够用的情况下，能不引入 tensor_parallel，pipeline_parallel，sequence_parallel 就不要去引入; **`data_parallel` 几乎是唯一不牺牲算力的并行方式**）

> 一个好模型是“身材匀称”的。也就是说标准的 llama 结构应该是横向和纵向呈某种比例的，所以在增大模型 size 的时候，layer_num 和 hidden_size 要一起递增。
> 
> [rope 的 NTK 外推方法](#rope的ntk外推方法深入解读)已经是各大厂标配的方案：4K/8K + rope 小 base + 90% 数据量 --> 32K/64K + rope 大 base + 10% 数据量

<br>

### 评估篇
- **PPL（Perplexity）**：通过测试集的loss衡量模型效果。
- **Benchmark**：使用改造过的benchmark来评估模型，避免直接使用选择题形式。
- **概率探针**：监控特定token或句子的概率变化，以评估模型的知识掌握程度。

> ppl 只能是自己的模型和自己比，前面说过，由于 tokenizer 的压缩率不同，不同模型的 loss 没有明确可比性，全是字没有词的 tokenizer，loss 一定是最低的
> 
> 现在最主流的 benchmark，形式也都实在是有点单一了。全是选择题，而且是没有 cot 环节的选择题

<br>

### 总结篇
- 文章强调了预训练过程中每个环节的重要性，特别是**数据清洗和处理**。



>[!NOTE] 
> 关于预训练任务基础知识的参考文献可以看这篇[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271),比较久了，里面讲了预训练目标等


