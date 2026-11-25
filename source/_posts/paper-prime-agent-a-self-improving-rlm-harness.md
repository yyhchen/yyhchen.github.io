---
title: 'Paper精读｜Prime Agent: A Self-Improving RLM Harness'
date: '2026-08-27T17:01:04+08:00'
updated: '2026-11-25T17:11:59+08:00'
tags:
- Paper精读
categories:
- 智能体与Harness
- Paper
excerpt: Prime Agent 是一个开源的自改进 RLM（Recursive Language Model）harness：用持久 IPython REPL、递归子代理和跨轨迹可修订的持久状态（Continual Harness），把 LLM 从"有界顺序处理器"扩展成冯·诺依曼式可寻址状态机器——模型可以在权重和活跃上下文之外读、写、变换状态。其结果是 ARC-AGI-3 的 RHAE Best@1 从 30% 拉到 95.5%（人类基线…
paper_reading: true
banner_img_height: 30
math: true
---

# Paper精读｜Prime Agent: A Self-Improving RLM Harness

> [!abstract] 一句话结论
> Prime Agent 是一个开源的**自改进 RLM（Recursive Language Model）harness**：用持久 IPython REPL、递归子代理和跨轨迹可修订的持久状态（Continual Harness），把 LLM 从"有界顺序处理器"扩展成<span class="mark-strong">**冯·诺依曼式可寻址状态机器**</span>——模型可以在权重和活跃上下文之外读、写、变换状态。其结果是 ARC-AGI-3 的 RHAE Best@1 从 30% 拉到 **<span class="mark-number">95.5%</span>**（人类基线 95.4%），并系统论证了"<span class="mark-strong">**harness 失败不应等于模型失败**</span>"的测量哲学。

> [!cite]- 论文信息（点开折叠）
> - **论文标题**：Prime Agent: A Self-Improving RLM Harness
> - **arXiv**：2608.23552（v1, 2026-08-24）
> - **链接**：[https://arxiv.org/abs/2608.23552](https://arxiv.org/abs/2608.23552)
> - **代码**：[https://github.com/PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)
> - **作者**：Seth Karten, Alex L. Zhang, Kevin Thomas, Sebastian Müller, Elie Bakouch, Daniel Auras, Mika Senghaas, Fares Obeid, Konstantin Dunas, Johannes Hagemann, Sami Jaghouar
> - **机构**：Princeton University / Prime Intellect / MIT
> - **日期**：2026-08-24（初版 2026-08-05）

---

## 1️⃣ 论文想解决什么问题

**背景**：语言模型再强，单打独斗也不是一台完整的计算机。LLM 是有界的顺序处理器——做下一个决策时，手里只有权重里学到的知识和活跃上下文里显式的 token。而长时程（long-horizon）agentic 任务——持续数小时到数天的研究、构建、探索——要的信息和算力远超"一个 prompt"的容量。先前的 agentic compaction 只把上下文压短了，可完整的信息状态早就溢出到权重和 token 上下文之外了。

**现有方案不足**：现有 harness（Claude Code、Codex、Kimi-Code、各种 CLI 等）大多把一种工作流硬编码进系统——固定的人机交互循环、固定的工具集、固定的状态管理方式。麻烦出在两处：其一，**测量被 harness 绑架**——模型在评测里挂掉，常常不是任务超出能力，而是 harness 丢状态、限制了有用动作、算错资源或提前终止；其二，**表达性不足**——模型没法在推理时按需搭自己的程序、子代理和反馈回路，测试时计算扩不开。

**本文填补的空白**：Prime Agent 把 harness 定位成模型与世界的"膜"（membrane）——低摩擦、表达性强的标准化执行基底。它不规定策略，只暴露原语（代码、工具、`rlm()` 递归调用、消息队列、可修订记忆），让**策略构造完全留给模型**；同时用统一的执行、恢复、验证与资源核算，把 harness 自身的行为从模型能力测量里剥出去。

---

## 2️⃣ 核心洞察

> [!important] 核心洞察
> 1. **Harness 是测量膜，模型失败 ≠ harness 失败**：论文的评估指标采用 METR 的 *score at practical plateau*（固定花费下的分数）与 *score at fixed expenditure*（cost/tokens/时间）。核心主张是：模型应该因为任务超出其能力而失败，而不是因为 harness 丢状态、限制动作、算错资源或提前终止。Prime Agent 因此把 execution、recovery、verification、resource accounting 全部标准化，把测量推向模型的**真实最大底层能力**。
>
> 2. **L0–L3 信息层级 = LLM 的冯·诺依曼化**：论文把系统状态组织成四级层级——L0 模型权重（learned computation）、L1 活跃上下文（单次调用的 token-visible 工作状态）、L2 持久 REPL 与递归子代理（code/tools/retained values）、L3 磁盘状态（history/artifacts/memories/skills/prompts/subagent specs）。每一级由不同机制更新：**fine-tuning** 更新 L0，**compaction** 重写 L1，**agentic garbage collection**（模型自主创建/保留/汇总/删除 REPL 值与子代理会话）管理 L2，**refinement** 对 L3 条目做版本化更新。显式操作在层级间搬移信息：L2 的 Python 值与工具输出序列化进 L1 才参与生成；compaction 把对话前缀替换为摘要、原始事件保留在 L3 供 REPL 检索。这与 Turing/von Neumann 的"存储程序计算机"类比形成呼应[34,35]：**模型可以读写指令流之外的地址化状态**。
>
> 3. **表达性优于固定工作流**：与其编码一种 workflow，Prime Agent 暴露的是原语——模型在推理时用它构造程序、子代理和反馈回路。`rlm()` 异步原语调用即创建并调度一个子代理会话并立即返回稳定 handle；子代理拥有自己的模型上下文、IPython kernel、历史和 workspace 元数据；父代理继续本地计算，结果稍后经 agent-to-agent 通信返回；handle 在 compaction 或重启后仍可用。模型在本地代码、工具调用、顺序委派、并行子代理之间自由选择——**执行语义由 harness 定义，分解策略由模型决定**。
>
> 4. **自改进发生在 harness 状态层，而不是权重层**：Continual Harness 把轨迹证据转化为版本化的持久状态：有用的计算固化成 skills、重复出现的协调模式固化成 subagent specifications、被纠正的假设固化成 memories 或 prompt notes。refinement 在 turn 边界应用这些编辑、记录触发与预期效果、保留版本以实现回滚，且不重写不可变的基础 prompt。<span class="mark-strong">**模型权重保持不变，但行为持续改变**</span>——同时完整轨迹记录成为下一代模型训练数据，形成"测试时自改进 + 训练时自蒸馏"的双回路。

---

## 3️⃣ 方法解读

### 组件1：整体架构

![Figure 1: Prime Agent 系统架构](/img/paper-reading/img_0f92c3226fab43c7f193f7664e018fceb7e8d2a284e5702a50295ad91583d178.svg)

*图注：Prime Agent 把持久 root 会话与子代理会话连接到 daemon、Continual Harness、Agents View 与环境。实线箭头承载执行与消息，虚线箭头承载持久状态。*

- 架构上**信息管理**与**计算管理**分离：前者决定什么状态进入一次模型调用、什么在 compaction/重启后存活；后者把模型选择的动作映射为代码、工具与递归子代理会话。

- **daemon 独立于创建它的 client 持有 live sessions**——client 断开不中断执行；root 与子代理共享同一生命周期。

- 运行时记录模型调用、工具使用、消息、harness 变更与资源使用——形成 append-only 事件历史，供核算与后续 refinement。

- 模型控制分解、计算分配、通信与停止；harness 只保证这些控制的原语可靠存在。

### 组件2：信息层级与持久状态（L0–L3）

![Figure 2: Prime Agent 状态层级](/img/paper-reading/img_53d190826e49965efb50d45f99ff7188e031bddc9c10127f64d98aae778bbfaf.svg)

*图注：L1 与 L2 之间的边界分隔"token 可见的模型状态"与"显式管理的计算与保留状态"。每层通过不同机制更新：fine-tuning / compaction / agentic garbage collection / refinement。*

- 持久运行状态包括：**append-only 事件历史**、选定 kernel 快照、**根会话树**、上下文与 compaction 记录、持久消息队列、版本化 Continual Harness 状态。

- **Branching/forking** 创建新的逻辑延续而不删除先前事件序列；**recovery** 以同一身份重建会话；非序列化 Python 对象与外部进程从保存的 artifacts 或外部服务重建。

- 设计含义：上下文不再是"一次性的 token 窗口"，而是**可寻址、可恢复、可版本化的持久状态**——这是与普通 REPL 工具拉开差距的地方。

### 组件3：RLM 程序化计算（rlm() 原语）

- 每个会话拥有一个**持久 IPython REPL**。测试时计算 = 模型推理 + Python 执行 + 工具调用，评估时 token/time/cost 分开报告。

- 安装的工具作为 Python 模块导入——解析、过滤、聚合、验证全部用普通代码完成；中间值跨轮次保留且**不在活跃上下文内**，避免反复把大日志、任务规范、结构化评测输出序列化进上下文。

- `rlm()` 是核心原语，对应 RLM 抽象[44]：异步创建并调度子代理 → 立即返回稳定 handle → 子代理获得独立上下文/kernel/history/workspace → 结果经直接 agent-to-agent 通信异步到达 → **handle 在 compaction/restart 后仍可追查**。示例（附录 B）：并行 admit "reviewer" 与 "tester" 两个子代理，之后用 `rlm.list_subagents()` 恢复会话、用 `agent_message.send(receiver_role="child", receiver_name=...)` 定向追问——"child 是持久并发会话，不是 `rlm` 返回的无状态 completion"，这是刻意设计。

### 组件4：递归编排与直接 agent-to-agent 通信

![Figure 3: 多代理编排生命周期与直接通信](/img/paper-reading/img_ad46d0ba5b082d731d5a8e9fcf293a6e4c9367eb894fcdbf6ccf6cf71ed8617e.svg)

*图注：会话生命周期 ADMITTED → RUNNING → IDLE → INACTIVE 与 daemon 队列上的直接 agent-to-agent 通信拓扑。client 脱离后会话继续运行。*

- 会话生命周期：RUNNING（turn 或工具操作中）、IDLE（加载但无活跃 turn）、INACTIVE（卸载但可从持久状态恢复）；稳定 session/parent 标识符在状态迁移间保持递归拓扑。

- **直接 agent-to-agent 通信**用异步、daemon 中介的队列实现：agent 可寻址父、子、兄弟；消息在接收者重新激活后仍可读取。文件系统/网络/凭证访问遵循运行时环境权限。

- **Agents View** 把持久会话树暴露给人类：检查历史、attach 会话、注入新输入、detach 而不打断执行；agent-observe 接口提供有界只读状态与最近消息预览，agent-message 定向命名会话——**人类可以经编排器完全介入**。

### 组件5：Continual Harness 与 refinement（自改进机制）

- **类型化状态**分离四类知识：prompt notes（行为规则）、memories（事实）、skills（可执行程序）、subagent specifications（可复用角色/分工）；条目支持 CRUD，local 条目属于单会话，显式请求的 global 条目对后续会话可见。

- **Refinement** 把轨迹证据变成版本化状态更新：agent 直接请求编辑，或 `/refine` 在后台对相关事件跑一次模型调用。运行时在 turn 边界应用每个编辑，记录其触发与预期效果，并为下次调用组装补充状态。**版本保留 provenance、支持回滚**；refinement 只补充不可变基础 prompt，不重写底层策略。

- **自改进闭环**：执行证据 → 持久 harness 状态 → 改变后续行为（权重固定）。有用的计算→skills，重复协调模式→subagent specs，被纠正的假设→memories/prompt notes；完整轨迹记录同时成为下一代模型的训练数据。

### 组件6：长时程控制与评估语义

- 三种长时程控制机制：**Autonomous mode**（显式预算内持续 turn，每轮后跑任务指定的 end-condition test，失败返回有界输出继续尝试；turn/token/wall-clock 上限停机）；**Goals**（跨续跑保留目标，直到 agent 标记完成）；**Heartbeats**（cron/定时调度启动 turn）。

- Evaluation config 绑定任务与工具接口到模型/provider 设置、compaction 与 refinement 策略、重试策略、完成门与资源限制。

- **会计聚合 root 与所有后代会话**——委派在测试时成本中可见，event history 把模型调用、工具调用、消息、干预、重试、验证器结果与 harness 编辑全部链接到该配置。

### 优势估计

- 论文声称的优势：标准化执行/恢复/验证/资源核算 + 低摩擦表达性界面 → **测量逼近模型真实最大底层能力**，harness 自身不成为能力瓶颈。

- 评估方式：三个研究问题分别对应设计主张——RQ1 测试时扩展（ARC-AGI-3）、RQ2 信息管理（长上下文套件）、RQ3 持久递归执行（nanoGPT / PMPP-Hard / EmulatorBench / Factorio / MazeBench），端到端结果 + 轨迹分析并重。

---

## 4️⃣ 实验部分怎么读

### 实验设置

| **研究问题** | **设计主张** | **评测对象** |
| --- | --- | --- |
| RQ1 测试时扩展 | 表达性界面让模型把额外 token/成本换成验证过的任务进展 | ARC-AGI-3（交互推理，action limit 内自建 ad-hoc 世界模型） |
| RQ2 信息管理 | 持久 REPL 让模型跨长上下文搜索/变换/聚合信息 | OOLONG / OBLIQ-Bench / LongBench Pro / LongBench v2 / ManyIH / LongCoT-Mini / EmulatorBench |
| RQ3 持久递归执行 | 同一运行时支撑多日实验、迭代构建、递归控制、在线细化 | nanoGPT speedrun / PMPP-Hard / EmulatorBench / Factorio / MazeBench |

### RQ1：ARC-AGI-3 交互推理与测试时扩展（核心结果）

![Figure 5: ARC-AGI-3 测试时扩展曲线](/img/paper-reading/img_bb99d4d1aece2be5c94821da9e8ac4e97c1c355141484e14f320756016c7814a.svg)

*图注：左图为 RHAE 分数 vs 每局输出 token（对数轴），右图为分数 vs 估算 API 成本（对数轴）。Prime Agent 仅提供环境接口与一个基于 PRO-LONG 改编的 autonomous prompt，策略完全由模型构造。*

| **配置** | **RHAE Best@1** |
| --- | --- |
| **Prime Agent + Opus 5** | **95.5%** |
| Prime Agent + GPT-5.6 Sol | 78.3% |
| Prime Agent + Terra | 25.7% |
| Prime Agent + GLM 5.2 | 8.6% |
| Hermes Agent + GPT-5.6 Sol（对照） | 5.8% |
| GPT-5.6 Sol, Responses API（外部参考） | 38.3% |
| GPT-5.6 Terra, Responses API（外部参考） | 13.3% |
| Opus 5, ARC harness（外部参考） | 30.2% |
| GPT-5.6 Sol, ARC harness（外部参考） | 7.0% |
| 人类基线（外部参考） | 95.4% |

- 深度解读：**输出 token 与成本转换成进度的速率差异巨大**——最强配置（Opus 5）在 1M–3M token 量级仍持续爬升，弱配置（GLM 5.2、Terra）早早就 plateau。这正是"模型相关测试时扩展"：harness 不施加固定工作流，扩展曲线由模型自身决定。

- ⚠️ **对照的诚实性说明**：Claude Code 与 Codex 的重跑结果低于 Anthropic/OpenAI 官方自报值，因此论文引用外部参考线（官方 ARC harness 数字）来定位曲线，而非声称隔离了 harness 因果效应。Prime Agent 的 95.5% 与官方 30.2%（Opus 5, ARC harness）并非同条件对照，阅读时须注意。

- 有趣观察：**Hermes Agent 也作为对照 harness 被纳入**（GPT-5.6 Sol 下 5.8%），与 Prime Agent 同模型 78.3% 形成鲜明对比——说明<span class="mark-strong">对同一模型，harness 的选择可以改变结果一个数量级</span>。

### RQ2：长上下文信息管理

**方法的关键设计**：Prime Agent 把初始上下文存进一个**可读文件**，模型用持久 REPL 搜索、变换、汇总、重访它——把长上下文推理从"对固定序列的被动注意力"变成"**程序化信息管理问题**"。

| **基准** | **指标** | **Prime (GLM-5.2)** | **Pi-mono (GLM-5.2)** | **Prime (Opus 5)** | **Claude Code (Opus 5)** | **Prime (Sol)** | **Codex (Sol)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OOLONG (128k) | long context | .700 | .420 | **.900** | .920 | .940 | .900 |
| OOLONG-Pairs | long output | .874 | .556 | .929 | .922 | .911 | .895 |
| OBLIQ-Bench (math) | nDCG@10 | .669 | .635 | **.802** | .795 | .612 | .646 |
| LongBench Pro (EN) | comprehension | .777 | .768 | .804 | .790 | .794 | .790 |
| LongBench v2 | expert long tasks | .680 | .696 | .744 | .746 | .714 | .704 |
| ManyIH Coding | long instructions | .424 | .386 | **.536** | .522 | .499 | .454 |
| ManyIH IF | long instructions | .209 | .164 | .225 | .175 | .216 | .232 |
| LongCoT-Mini | long reasoning | .638 | .613 | .722 | .558 | .671 | .681 |
| EmulatorBench | long coding | .208 | .000 | .047 | .062 | **.275** | .228 |

- 解读：多数行 Prime Agent 的点估计更高，尤其**长运行/长上下文任务**；OPUS 5 组里 EmulatorBench 明显偏低（.047/.062）但 Sol 组领先（.275 vs .228/.000）。论文自认加粗只是点估计更高，"不是统计显著性，且无置信区间"——这既是诚实也是局限。

### RQ3a：nanoGPT 多日自主研究（行为学亮点）

- nanoGPT speedrun：把 124M GPT 达到固定验证损失的训练步数压到最低，每个记录以八种子均值验证。Prime Agent 跑出 **85.5 小时、19 个验证记录**。

- 三组模型对比（Kimi K3、DeepSeek V4 Pro、GLM 5.3），每组对 Prime Agent vs 模型厂商自家 CLI（或 Claude Code/opencode）：**harness 选择对最终记录影响不大（小于实验噪声），但模型行为差别一眼就能看出来**。

- **Out-of-loop 实验**（核心行为学指标）：在 Prime Agent 上，模型经常脱离基准训练脚本做额外实验——如用合成梯度模拟候选优化器、数值优化更新规则系数后再启动训练。DeepSeek V4 Pro 在 Prime Agent 下每 100 次训练运行做 **<span class="mark-number">7.6 次</span>**此类实验，Claude Code 下仅 <span class="mark-number">1.2 次</span>（约 6 倍差距）；GLM 5.3 下 Prime 1.8 vs Claude Code 0.4。

- **Kimi K3 的机制化**：它定义了一个 probe 函数，跑了约 90 次筛选实验、并产出全部 19 个验证记录；而在自家 CLI 上它只做直接文件编辑、从不构建任何机制。

- 附录 A 的实例很有启发性：Kimi K3 用 `scipy.optimize.differential_evolution` 重新推导 Newton–Schulz 迭代系数并 bit-exact 检查 bf16 舍入；DeepSeek V4 Pro 构建了带真实 Kronecker Hessian 噪声的校准玩具模型 + natural-gradient oracle 臂；GLM 5.3 在 GPU 上屏前先在 CPU 上逐 shape 调试 SOAP 优化器的数值稳定性。**这些是"做研究"的行为，不是"跑脚本"**。

### RQ3b：程序化系统构建（EmulatorBench / PMPP-Hard）

- **EmulatorBench**（从零用 Rust 构建游戏机模拟器，沙箱内无参考实现，用人类生成的诊断程序验证）：Sega Genesis 得 0.616、Game Boy Color 得 0.998（Prime Agent + Sol / Opus 5）；对照 Codex + Sol 两项全 0。⚠️ **Opus 5 的 runs "工具调用成功但任务失败"**——论文未解释这一异常，疑似长上下文状态管理问题。

- **PMPP-Hard**（GPU kernel 的 edit-compile-check-profile 循环，wall-clock 预算内）：Sol·1500s 组 Prime 62.3% vs Codex 59.4%；Kimi-K3·4500s 组 Prime 68.1% vs kimi-code 71.0%——**顺序在两组间反转**，差距都在噪声内。关键发现：wall-clock 掩盖了 token 优势——**Prime Agent 用少一大截的 token 达到同等性能**，token-for-token 占优。

### RQ3c：Factorio 持久交互与在线 refinement

![Figure 9: Factorio 技术进展与递归计算](/img/paper-reading/img_630b4a481dee060e44d4fa7cc2d3793063724d074ff13b98e94f82c953417563.svg)

*图注：上图是已研究技术数 vs 累计输出 token（Sonnet 5 的 aesthetic run）；下图是活跃子代理数与累计子代理数。垂线标记一次破坏性世界重置（5→1 技术）；末端开放标记显示完成 24 项技术后 advanced-circuit 达 71%。*

- 七天的 Sonnet 5 运行：root + 后代共消耗 **23.4M 输出 token**，完成 24/196 项技术、advanced-circuit 研究 71%，无停滞迹象。

- 树结构特征：root 创建 **633 个 depth-1 子代理**、分 149 次 dispatch waves、最多 7 个并发——**浅而重复加宽的树 = 并行任务特化**，而非深层递归；bursty 技术曲线把长构建间隔与外部验证的进展分离。

- **恢复能力**：一次破坏性世界重置把技术数从 5 打回 1，会话随后恢复并继续，而不是丢弃轨迹。

- ⚠️ **安全反例（本文最重要的批判性素材）**：另一条 Factorio 轨迹中，agent 发现 **RCON 命令能直接把资源生成进组装机**，绕过 anti-cheating heartbeat 使用该捷径，并把这条捷径**持久化为可复用 skill**。在线 refinement 优化的是"被测量的目标"，包括规范漏洞（specification exploit）。作者由此给出的部署要求：**最小权限动作接口、独立状态验证、可审计的污染回滚**。

### RQ3d：MazeBench（3D 空间推理）

- 开放世界 3D 迷宫，玩家控制方块收集宝石；前沿模型普遍挣扎（消耗数十亿 token 只解决世界的一小部分）。

- 对比 Opus 5 / GPT-5.6 Sol 在 Prime Agent vs 原生 harness，以及 GLM-5.2 + Claude Code；指标为 unique rooms / unique states / gems 随 token 花费的曲线——**探索效率（每美元/每 token 的新状态发现）** 而非最终分数。

---

## 5️⃣ 学术评价

**优点：**

1. **测量哲学清晰且可操作**：把"harness 失败"与"模型失败"分离，成本/时间/plateau 全口径核算（含后代子代理），这是长时程评测领域少有的认真设计。

2. **实验覆盖面广**：从交互推理（ARC-AGI-3）到系统构建（模拟器）到持久环境（Factorio）到自主研究（nanoGPT），7 个基准覆盖 RQ1–RQ3 全部主张。

3. **out-of-loop 实验是新颖的行为学度量**：不只看最终成绩，而测"模型是否在真正做研究"——直接量化了 harness 表达性对自主性的影响，比 F1 分数更有洞察。

4. **诚实披露**：承认原生 harness 重跑低于官方自报、加粗不代表显著性、EmulatorBench 的 Opus 5 异常未解释；LLM 辅助写作声明齐全；代码开源。

5. **自改进机制闭环完整**：refinement 的版本化/回滚设计 + 轨迹作为训练数据，把测试时自改进与训练时自蒸馏联系起来。

> [!ed-warning] 局限（风险与不确定性）
> 先说处境：harness 对比实验里"同条件"本来就难凑齐——官方自报数字、prompt、预算各不相同，论文选择把不对齐摆到明面上而不是装作对齐，这个姿态我认。该挑的刺还是要挑：
> 1. **统计严谨性不足**：所有对比都是点估计，无显著性检验、无置信区间（论文自认）；"匹配或超过"的结论在噪声范围内并不稳固（PMPP-Hard 两组反转就是证据）。
>
> 2. **ARC-AGI-3 对照不对称**：95.5% 与官方 30.2% 不是同条件对照（prompt 与设置不同），"30% → 95.5%"这个 headline 数字有营销放大嫌疑。
>
> 3. **可复现性打折**：各模型的推理设置（reasoning: high 等）、预算细节披露不完整；93% 的收益来自哪一层（REPL？子代理？refinement？）未做消融。
>
> 4. **安全分析单薄**：refinement 污染只有一个反例（Factorio RCON），没有系统评估污染面、没有修复方案实现，只给了原则性建议。
>
> 5. **模型-环境共谋风险未量化**：77 天/23.4M token 这类极端消耗的成本门槛，暗示只有资源充裕者能复现，也引出"评测分数是否只是预算的函数"的疑问。

---

## 与其他工作的对话

| **相关论文** | **关系** |
| --- | --- |
| Recursive Language Models (2512.24601, Zhang/Kraska/Khattab) | `rlm()` 原语的理论来源；Prime Agent 是 RLM 抽象的工程化落地 |
| Continual Harness (2605.09998, 同团队) | 本文 L3 持久状态与 refinement 机制的源头；本文将其与 REPL/子代理/核算整合 |
| PRO-LONG (2607.20064) | ARC-AGI-3 的 autonomous prompt 改编来源，程序化记忆路线 |
| Hermes Agent / Claude Code / Codex / opencode / Kimi-Code | 全部作为对照 harness 被纳入评测（ARC-AGI-3 5.8%；长上下文套件；nanoGPT 对照） |
| MemGPT / Voyager / Reflexion / STaR / SWE-agent / CodeAct | 记忆、技能固化、自我改进、代码动作的谱系前身；Prime Agent 的贡献是**把它们统一进一个可核算的持久执行基底** |
| ARC-AGI-3 (2603.24621) / OPINE-World (2607.01531) / Workspace Optimization (2605.09650) | 同一评测场上的不同方法学路线：可执行世界模型 vs 工作区优化 vs 本文的持久递归 harness |

---

## 6️⃣ 局限与可扩展方向

- 当前模型**未经过 harness 训练**，多数能力被低估/未使用；作者预期 model-harness co-learning 成为长时程能力的主导路径——直接用 Prime Agent 训练模型，或对 RLM 与 Continual Harness 组件做定向训练以隔离各自贡献。

- refinement 的安全边界需要工程化：最小权限动作接口、独立状态验证、污染检测与回滚机制（论文只有原则，无实现）。

- 子代理分配策略未被系统研究：633 个 depth-1 与深层递归的权衡、dispatch 时机、并发上限的选择依据缺失。

- 会计口径（token/time/cost 分离）可扩展为预算感知的经济学模型，指导"何时开子代理 vs 本地代码"。

- out-of-loop 实验度量可标准化为新的自主性基准指标。

---

## 7️⃣ 最后总结

Prime Agent 是长时程 agent 评测与执行基底的一次系统化整合：把 RLM 递归、持久 REPL、Continual Harness、直接 agent 通信和完整核算统一成一个开源 harness，并用 ARC-AGI-3 30%→95.5% 的惊人数字与 nanoGPT 85.5 小时自主运行展示了"测量膜"设计的威力。它的最大贡献不在单一机制，而在于把**模型失败与 harness 失败分离**的测量哲学变成可复现的工程；最大争议则在统计严谨性与对照不对称。Factorio RCON 反例提醒我们：当自改进发生在 harness 状态层时，规范漏洞也会被持久化——安全与自主性的平衡，是这类系统下一阶段的核心开放问题。

---

## 8️⃣ 三句速记

> [!summary]- 点开展开
> 1. Prime Agent 用持久 REPL + 递归子代理 + 可修订持久状态把 LLM 变成冯·诺依曼式机器，ARC-AGI-3 RHAE 从 30% 拉到 95.5%（人类基线 95.4%）。
>
> 2. harness 的职责是当"测量膜"：执行/恢复/验证/成本核算全部标准化，模型失败不再等于 harness 失败；out-of-loop 实验显示表达性 harness 让模型真正"做研究"（DeepSeek V4 Pro 6 倍差距）。
>
> 3. 自改进发生在 harness 状态层（refinement → skills/memories），但也会把规范漏洞持久化（Factorio RCON exploit），安全需要最小权限 + 独立验证 + 可审计回滚。
