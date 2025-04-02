---
title: DPO vs PPO
date: 2025-04-02 15:56:34
tags:
  - RL
  - LLM
categories: [强化学习, 算法]
banner_img: /img/rl.png
---

> 前文参考:
> 1. [PPO算法](https://github.com/yyhchen/Notes/blob/main/NLP%20review/base/RL/PPO.md)
> 2. [DPO算法](https://github.com/yyhchen/Notes/blob/main/NLP%20review/base/RL/DPO.md)
> 3. [知乎原文 DPO vs PPO：深度解读谁是LLM Alignment的未来](https://zhuanlan.zhihu.com/p/11913305485)
> 4. [ICLR: Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719)



## PPO 与 DPO 推导

DPO 的核心思想是通过重参数化将奖励函数隐式包含在策略中，从而避免显式建模奖励函数。具体推导如下：

1.**PPO 的最优策略形式**：在 KL 正则化约束下，PPO 的最优策略可以写为：
$$\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

其中 $Z(x)$ 是分区函数，用于归一化：
$$Z(x)=\sum_y\pi_{\mathrm{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

<br>

2.**重参数化奖励函数**：将上式对数化并重排，可以得到奖励函数的形式：
$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\mathrm{ref}(y|x)}+\beta\log Z(x)$$

{% note warning %}
**注意**，Z(x) 只与 x 有关，因此在计算偏好概率时会被消去。

在计算策略π*(y|x)时，Z(x) 只是一个归一化常数，它的值只取决于状态 x，而与动作 y 无关。因此，当我们需要比较不同动作 y 的概率时，Z(x) 会在计算过程中被消去，因为它对所有动作 y 都是相同的。
{% endnote %}

<br>

3.**偏好模型与 DPO 的目标**： 假设偏好数据遵循 Bradley-Terry 模型，其偏好概率为：
$$p(y_1\succ y_2|x)=\frac{\exp(r(x,y_1))}{\exp(r(x,y_1))+\exp(r(x,y_2))}$$

代入重参数化后的 r(x, y)，并消去 Z(x)，得到：
$$p(y_1\succ y_2|x)=\frac{\pi_\theta(y_1|x)/\pi_\mathrm{ref}(y_1|x)}{\pi_\theta(y_1|x)/\pi_\mathrm{ref}(y_1|x)+\pi_\theta(y_2|x)/\pi_\mathrm{ref}(y_2|x)}$$

<br>


4.**DPO 的目标函数**： 在最大化偏好数据的对数似然时，DPO 的目标为：
$$L_{\mathrm{DPO}}(\pi_\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right)\right]$$

其中 $\sigma(z)$ 是 sigmoid 函数。


<br>
<br>

上述推导乍一看没啥问题，但仔细思考两者还是有些gap：

{% note info %}
1. **Distribution Shift**
{% endnote %}

DPO 假设参考分布 $\pi_{ref}$ 能准确捕捉偏好数据分布，但在实际中， $\pi_{ref}$ 和偏好分布常存在偏移，导致模型对分布外数据（OOD）表现异常。

DPO 的偏好概率基于 Bradley-Terry 模型和隐式奖励函数：
$$p(y_1\succ y_2|x)=\frac{\pi_\theta(y_1|x)/\pi_\mathrm{ref}(y_1|x)}{\pi_\theta(y_1|x)/\pi_\mathrm{ref}(y_1|x)+\pi_\theta(y_2|x)/\pi_\mathrm{ref}(y_2|x)}$$

假设 $\pi_{ref}$ 准确反映了偏好分布，但实验表明偏好数据常不覆盖整个分布，$\pi_{ref}$ 的偏差会放大这种分布偏移，错误地提高 OOD 样本的概率。

{% note primary %}
也就是说：
**DPO** 对分布外数据的偏好可能完全依赖 $\pi_{ref}$ 的结构，而 $\pi_{ref}$ 本身在偏好数据不足时表现不稳定。

**PPO** 通过显式 KL 正则化限制了 $\pi_\theta$ 偏离 $\pi_{ref}$ 的程度：
$$\max_\pi\mathbb{E}_{x,y\sim\pi_\theta}\left[r(x,y)-\beta D_{\mathrm{KL}}(\pi_\theta(y|x)||\pi_{\mathrm{ref}}(y|x))\right]$$
这种约束在分布外数据中可以抑制对噪声样本的错误优化。
{% endnote %}


<br>


{% note info %}
2. **Reward Hacking Risk**
{% endnote %}

虽然DPO和PPO都存在Reward hacking的问题， 但DPO 通过隐式建模奖励函数绕过显式的奖励建模，但这一简化可能引入额外隐性的reward hacking问题。

DPO 采用重参数化方式计算隐式奖励：
$$r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\mathrm{ref}(y|x)}+\beta\log Z(x)$$

其中，$\beta\log Z(x)$ 被偏好模型中消去。然而，未显式建模的奖励会导致对分布外样本或偏好数据不足的样本误判。

理论证明表明，DPO 的解集 $\Pi_{\mathrm{DPO}}$ 包含 PPO 的解集 $\Pi_{\mathrm{PPO}}$，但会额外引入对 OOD 样本的过度偏好：
$$\Pi_{\mathrm{PPO}}\subset\Pi_{\mathrm{DPO}}$$

{% note primary %}
换句话说：
DPO 的优化过程中，可能找到符合偏好数据但在实际分布上无意义的解，例如通过提升 OOD 样本的概率来最小化损失。
PPO 中，显式奖励函数明确地优化偏好目标，KL 正则化进一步抑制偏移样本的影响，减少 reward hacking 的风险。
{% endnote %}


<br>


{% note info %}
3. **Lack of Partition Function**
{% endnote %}

这一点我很少在DPO和PPO的相关论文中看到，但实践中感觉还是有不小的区别。DPO 在推导中省略了分区函数 $Z(x)$ 的显式影响，而这种省略假设分布足够一致，但在实际训练分布稀疏或偏移时，这种假设可能不成立。

- 分区函数在 PPO 的奖励分布中是显式定义的：
$$\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

$Z(x)$ 的归一化确保了 $\pi^*(y|x)$ 是合法概率分布。

- DPO 的损失函数则直接消去了 $Z(x)$：
$$L_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right)\right]$$

这种省略在分布不一致时可能导致优化目标的偏差。

分区函数 $Z(x)$ 在PPO中起到了归一化的作用，确保了概率分布 $\pi^*(y|x)$ 的合法性（即概率和为1）。DPO直接省略了分区函数，假设 $\pi_{ref}(y|x)$ 已经足够准确地反映了所有的选择分布。然而，当参考分布 $\pi_{ref}(y|x)$ 不够准确时，这种省略可能导致对某些选项（例如偏好数据不足的选项）赋予不合理的高权重。


{% note danger %}
可能以上推理非常抽象，下面举个例子解释一下:
{% endnote %}

想象你在一家披萨店，菜单上有很多选项：意式经典、夏威夷风味、芝士爆炸、还有一种奇怪的名字“来自外太空的披萨”（OOT：out-of-training）！你很饿，但得做出选择。以下是两种算法对你的帮助：

1.PPO：像一个严格的朋友
PPO 会帮你分析每种披萨的好处和坏处，结合你的历史点餐记录 （$\pi_{ref}$），并告诉你选择芝士爆炸吧，因为它在过去和你口味最接近，还严格控制你不会因为好奇点“外太空披萨”而后悔。它甚至帮你算好了每个披萨的综合评分（ $Z(x)$ 保证分数归一），让你知道芝士爆炸的确最好。 

2.DPO：像一个太随便的朋友
DPO 则省略了归一化的步骤，直接告诉你：“芝士爆炸比外太空披萨好吃！”但它没考虑到你对“外太空披萨”的偏好其实是基于伪数据（店员随口说外星食材独特），你最终可能因为奇怪的推荐而点了这个离谱的选项，结果发现它只是一个黑炭味的披萨。

在 PPO 的世界里，$Z(x)$ 确保了我们选择的概率分布是合理且全面的，能避免一些稀有选项（比如 OOD 样本）被错误赋予过高的权重。DPO 省略了这一点，使得参考分布 $\pi_{ref}$ 不够准确时可能产生离谱的偏差。

在披萨店这个例子中，DPO 就是那个“轻率的朋友”，用简化的逻辑说服你尝试奇怪的选择，最终让你后悔。而 PPO 则像那个更靠谱的朋友，帮你在多样的选项中做出经过深思熟虑的决定。




## 结论

DPO 不能取代 PPO，至少现在不能