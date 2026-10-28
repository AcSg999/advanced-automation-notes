## TRPO 
keyward: KL divergence, surrogate advantage

[spinningup TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html#background)

### Background

TRPO updates policies by taking the largest step possible to improve performance, while satisfying a special constraint on how close the new and old policies are allowed to be.快速收敛但保持限制

This is different from normal policy gradient, which keeps new and old policies close in parameter space（控制$\theta$ old-new不要太大）
 
 recap之前在[西湖大学强化学习笔记](西湖大学强化学习.md) lec 9 学过的 policy gradient&policy update：

$$
\theta_{t+1}
=\theta_t+\alpha\left.\nabla_\theta J(\theta)\right|_{\theta=\theta_t}
=\theta_t+\alpha\mathbb{E}_{S\sim d_{\pi_{\theta_t}},\,A\sim\pi_{\theta_t}(\cdot\mid S)}
\left[
\left.\nabla_\theta\ln\pi(A\mid S,\theta)\right|_{\theta=\theta_t}
q_{\pi_{\theta_t}}(S,A)
\right].
$$

用采样近似期望，并用回报估计 action value 后，得到 stochastic gradient-ascent 更新：

$$
\theta_{t+1}
=\theta_t+\alpha
\left.\nabla_\theta\ln\pi(a_t\mid s_t,\theta)\right|_{\theta=\theta_t}
q_t(s_t,a_t),
\qquad
q_t(s_t,a_t)=G_t=\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k.
$$
普通 policy gradient 通过学习率调节参数更新幅度，但参数变化小有可能动作分布变化大；TRPO 则用 KL 散度**约束新旧策略分布的差异**。

### quick facts

- TRPO is an on-policy algorithm.

    - 补充：用当前策略与环境交互，收集一批数据来更新策略；更新后再用新策略采集下一批，而不是长期从历史经验池中抽数据训练。

- TRPO can be used for environments with either discrete or continuous action spaces.

    - 补充：离散动作是从几个选项中选一个，比如“向左／向右”；连续动作是输出数值，比如关节力矩。TRPO 都能用，只需让策略输出对应的动作分布，例如离散动作的分类分布或连续动作的高斯分布。

- The Spinning Up implementation of TRPO supports parallelization with MPI.

    - 补充：MPI 是多个进程之间传递数据的工具。Spinning Up 可以让多个进程各自运行环境、同时采样，再汇总梯度等信息并同步更新同一个策略，以加快训练.。

### key equations

令 $\pi_\theta$ 表示参数为 $\theta$ 的策略；旧策略为 $\pi_{\theta_k}$，本轮更新时 $\theta_k$ 固定。TRPO 的更新为：

$$
\begin{aligned}
\theta_{k+1}
&=\arg\max_{\theta}\;
\underbrace{
\mathbb{E}_{s,a\sim\pi_{\theta_k}}
\left[
\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}
A^{\pi_{\theta_k}}(s,a)
\right]
}_{\mathcal{L}(\theta_k,\theta)\;\text{：surrogate objective（代理目标）}}
\\[6pt]
\text{s.t.}\quad
&\underbrace{
\mathbb{E}_{s\sim\pi_{\theta_k}}
\left[
D_{KL}\left(
\pi_\theta(\cdot\mid s)\,\|\,\pi_{\theta_k}(\cdot\mid s)
\right)
\right]
}_{\bar{D}_{KL}(\theta\|\theta_k)\;\text{：新旧策略的平均 KL 散度}}
\leq\delta.
\end{aligned}
$$

直观理解：用旧策略采集的数据，寻找让代理目标尽可能大的新策略，同时将动作分布的变化限制在 $\delta$ 以内；$\delta>0$ 是允许的变化上限。

补充：KL 散度衡量两个概率分布的差异，定义为

$$
D_{KL}(P\|Q)=\mathbb{E}_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right]\geq 0.
$$

分布相同时为 $0$，一般不对称，即 $D_{KL}(P\|Q)\neq D_{KL}(Q\|P)$。

surrogate objective:$L(\theta_k,\theta)=\mathbb{E}_{s,a\sim\pi_{\theta_k}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a)\right]$

advantage function:$A^{\pi_{\theta_k}}(s,a)=Q^{\pi_{\theta_k}}(s,a)-V^{\pi_{\theta_k}}(s)$

$$\mathbb{E}_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_{\theta_k}(\cdot\mid s)}
\left[
\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}
A^{\pi_{\theta_k}}(s,a)
\right]
=
\mathbb{E}_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_\theta(\cdot\mid s)}
\left[A^{\pi_{\theta_k}}(s,a)\right]$$
即使动作样本来自旧策略，我们仍然可以用概率比来估计新策略在该状态下的动作分布对应的旧优势。这里对**动作期望的换算是精确的**，surrogate 的近似来自继续使用旧策略的状态分布。


#### 补充：与 lec 10 off-policy actor-critic 对照

将 TRPO 的 surrogate advantage 求梯度，与 [lec 10 加入 baseline 后的 actor 更新](images/126.png)放在一起：

$$
\begin{aligned}
 g_{\mathrm{AC}}(\theta)
 &=\mathbb{E}_{s\sim\rho,\,a\sim\beta(\cdot\mid s)}
 \left[
 \frac{\pi_\theta(a\mid s)}{\beta(a\mid s)}
 \nabla_\theta\ln\pi_\theta(a\mid s)
 A^{\pi_\theta}(s,a)
 \right], \\
 \nabla_\theta\mathcal{L}(\theta_k,\theta)
 &=\mathbb{E}_{s,a\sim\pi_{\theta_k}}
 \left[
 \frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}
 \nabla_\theta\ln\pi_\theta(a\mid s)
 A^{\pi_{\theta_k}}(s,a)
 \right].
\end{aligned}
$$

两式都有“概率比 × log probability 梯度 × 优势”的结构，概率比用于修正给定状态下的动作分布。区别是：lec 10 用行为策略 $\beta$ 的数据估计目标策略 $\pi_\theta$ 的更新方向；TRPO 用本轮旧策略 $\pi_{\theta_k}$ 的数据，并固定旧优势 $A^{\pi_{\theta_k}}$，只对概率比求导。

### 实际TRPO update（以上为theorical ）

recall
$$\begin{aligned} \theta_{k+1}&=\arg\max_{\theta}\; \underbrace{\mathbb{E}_{s,a\sim\pi_{\theta_k}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)} A^{\pi_{\theta_k}}(s,a)\right] }_{\mathcal{L}(\theta_k,\theta)\;\text{：surrogate objective（代理目标）}} \\[6pt]\text{s.t.}\quad&\underbrace{ \mathbb{E}_{s\sim\pi_{\theta_k}}\left[D_{KL}\left( \pi_\theta(\cdot\mid s)\,\|\,\pi_{\theta_k}(\cdot\mid s) \right) \right] }_{\bar{D}_{KL}(\theta\|\theta_k)\;\text{：新旧策略的平均 KL 散度}} \leq\delta. \end{aligned} $$

在 $\theta_k$ 附近对$\mathcal{L}(\theta_k,\theta)$作一阶 Taylor 近似、对 KL 约束作二阶 Taylor 近似，得到 approximate optimization problem：

$$\begin{aligned} \theta_{k+1}&=\arg\max_{\theta}\;\underbrace{g^T(\theta-\theta_k)}_{\approx\,\mathcal{L}(\theta_k,\theta)\;\text{（一阶近似）}} \\[6pt] \text{s.t.}\quad&\underbrace{\frac{1}{2}(\theta-\theta_k)^T H(\theta-\theta_k)}_{\approx\,\bar{D}_{KL}(\theta\|\theta_k)\;\text{（二阶近似）}}\leq\delta. \end{aligned}$$

$g=\left.\nabla_\theta\mathcal{L}(\theta_k,\theta)\right|_{\theta=\theta_k}$ 即代理目标在旧参数 $\theta_k$ 处的梯度，表示目标增长的方向；它也就是旧策略处的 policy gradient。

$H=\left.\nabla_\theta^2\bar{D}_{KL}(\theta\|\theta_k)\right|_{\theta=\theta_k}.$ H：平均 KL 散度$\bar D_{KL}(\theta\|\theta_k)$在旧参数 $\theta_k$ 处的 Hessian（二阶偏导数组成的矩阵），描述参数变化引起的 KL 增长。
$$
D(\theta)\approx
\underbrace{D(\theta_k)}_{\text{零阶项}}
+\underbrace{\nabla_\theta D(\theta_k)^T(\theta-\theta_k)}_{\text{一阶项}}
+\underbrace{\frac12(\theta-\theta_k)^TH(\theta-\theta_k)}_{\text{二阶项}}
$$

两者都对新参数 $\theta$ 求导，再代入 $\theta=\theta_k$；因此在上面的近似优化中，$g$ 和 $H$ 都是固定量。KL 在新旧策略相同时为 $0$ 且梯度为 $0$，所以它的 Taylor 展开从二阶项开始。

近似解：$\theta_{k+1}=\theta_k+\sqrt{\frac{2\delta}{g^T H^{-1}g}}\,H^{-1}g$。用拉格朗日对偶可解出上述近似问题，沿 $H^{-1}g$ 方向前进，并将步长缩放到二阶 KL 约束的边界。


Fisher Information Matrix（Fisher 信息矩阵）：

$$
F(\theta)=\mathbb{E}_{s\sim d_{\pi_\theta},\,a\sim\pi_\theta(\cdot\mid s)}\left[\nabla_\theta\log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)^T\right].
$$

它描述策略概率分布对参数变化的局部敏感程度：沿某个方向改变同样多的参数，分布变化越大，该方向对应的 Fisher 曲率就越大（在通常的正则条件下，这里 $F(\theta_k)=H$）。

Natural Policy Gradient（自然策略梯度）：$\widetilde{\nabla}_\theta J(\theta)=F(\theta)^{-1}\nabla_\theta J(\theta)$。用 $F^{-1}$ 调整普通梯度，让策略分布敏感的方向少走一些、不敏感的方向可以多走一些，从而在相同的局部 KL 变化预算下尽量提高目标；在这里，自然梯度方向就是 $H^{-1}g$。

Natural Policy Gradient 的核心思想：不要只考虑参数变化了多少，还要考虑这种参数变化会让策略概率分布变化多少。

TRPO 更新-回溯线搜索：$\theta_{k+1}=\theta_k+\alpha^j\sqrt{\frac{2\delta}{g^T H^{-1}g}}\,H^{-1}g$。取 $\alpha\in(0,1)$ 并选择满足实际 KL 约束且使代理目标增大的最小非负整数 $j$，逐步缩短步长。

进一步计算此更新，显式计算$H^{-1}$代价很大-->我们令$x=H^{-1}g$,转而求$Hx=g$这个线性方程，直接得到所需的向量$x$。

因为$H$维度很大，用conjugate gradient algorithm（共轭梯度法）迭代逼近$x$。每轮只需要计算$Hv$，$v$是本轮给定的向量。

PyTorch 中可通过两次自动求导直接计算$Hv$：先求 KL 对参数的梯度；第一次求导时将 `create_graph` 设为 `True`，保留可继续求导的计算图。再与固定的$v$做内积，对这个标量求导：

$$
Hv=\left.\nabla_\theta\left[(\nabla_\theta\bar D_{KL}(\theta\|\theta_k))^T v\right]\right|_{\theta=\theta_k}.
$$

这样只计算 Hessian-vector product（Hessian 与向量的乘积），整个过程都不需要显式求出$H$或$H^{-1}$.

### persudo code 伪代码

1. Input：初始策略参数$\theta_0$，初始价值函数参数$\phi_0$；策略为$\pi_\theta$，价值函数为$V_\phi$。

2. Hyperparameters：KL 上限$\delta$，回溯系数$\alpha\in(0,1)$，最大回溯次数$K$。

3. **for** $k=0,1,2,\ldots$，重复以下步骤：

4. 用当前策略$\pi_k=\pi_{\theta_k}$与环境交互，收集轨迹集合$\mathcal D_k=\{\tau_i\}$。其中$\tau_i$是一条轨迹，$|\mathcal D_k|$是轨迹条数；下文hat表示用这批样本得到的估计。

    > $\mathcal D_k=\{\tau_1,\tau_2,\ldots,\tau_N\}$,
    >
    > $\tau_i=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T)$


5. 计算每一步的 rewards-to-go（从该步开始的折扣回报）。

   > $$
   > \hat R_t=\sum_{l=t}^{T}\gamma^{l-t}r_l.
   > $$

6. 基于当前价值函数$V_{\phi_k}$，用优势估计方法计算$\hat A_t$，如 GAE。

7. 估计 policy gradient：

   $$
   \hat g_k=\frac{1}{|\mathcal D_k|}
   \sum_{\tau\in\mathcal D_k}\sum_{t=0}^{T}
   \left.\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right|_{\theta=\theta_k}\hat A_t.
   $$

8. 用 conjugate gradient 求解$\hat H_k\hat x_k=\hat g_k$，得到$\hat x_k\approx\hat H_k^{-1}\hat g_k$。其中$\hat H_k$是样本平均 KL 在$\theta_k$处的 Hessian。

9. 用 backtracking line search 更新策略：

   $$
   \theta_{k+1}=\theta_k+
   \alpha^j\sqrt{\frac{2\delta}{\hat x_k^T\hat H_k\hat x_k}}\,\hat x_k.
   $$

   取$j\in\{0,1,\ldots,K\}$中，使样本代理目标增大且样本平均 KL 不超过$\delta$的最小值；$K$为最大回溯次数，若都不满足，则保留$\theta_k$。

10. 用均方误差拟合价值函数，让$V_\phi(s_t)$接近$\hat R_t$：

    $$
    \phi_{k+1}=\arg\min_\phi\frac{1}{|\mathcal D_k|}
    \sum_{\tau\in\mathcal D_k}\sum_{t=0}^{T}
    \left(V_\phi(s_t)-\hat R_t\right)^2.
    $$

## appendix

### A. importance sampling：怎样用一种分布的样本计算另一种分布的期望？

#### A.1 先区分“分布近似”和“期望换算”

假设我们想计算 $\mathbb E_{x\sim p}[f(x)]$，但手里只有从 $q$ 采到的样本。直接取这些样本的 $f(x)$ 平均值，估计的是 $\mathbb E_q[f(x)]$，通常不是想要的 $\mathbb E_p[f(x)]$。

importance sampling（重要性采样）的做法是：样本仍来自 $q$，但用 $p(x)/q(x)$ 调整每个样本的权重。对离散分布，推导只有一步“乘上再除以 $q(x)$”：

$$
\begin{aligned}
\mathbb E_{x\sim p}[f(x)]
&=\sum_x p(x)f(x)\\
&=\sum_{x:q(x)>0}q(x)\frac{p(x)}{q(x)}f(x)\\
&=\mathbb E_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right].
\end{aligned}
$$

这里要求 $p(x)>0$ 的地方也有 $q(x)>0$，即采样分布覆盖目标分布的 support，让目标分布关心的位置都能被采到。连续变量把求和换成积分，$p,q$ 换成概率密度，道理相同。

**上面的等号是精确恒等式；用有限样本平均代替期望时，才引入采样误差。** 若 $x_i$ 来自 $q$，则

$$
\widehat{\mathbb E_p[f]}
=\frac1N\sum_{i=1}^N\frac{p(x_i)}{q(x_i)}f(x_i).
$$

直觉：目标分布比采样分布更常去的位置，权重大于 $1$；目标分布更少去的位置，权重小于 $1$。样本仍来自 $q$，变化的是每个样本对期望的贡献。

#### A.2 放回 TRPO：固定一个状态，看动作怎么重新加权

固定状态 $s$，令 $q(a)=\pi_{\theta_k}(a\mid s)$、$p(a)=\pi_\theta(a\mid s)$，并取 $f(a)=A^{\pi_{\theta_k}}(s,a)$。于是

$$
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[A^{\pi_{\theta_k}}(s,a)]
=\mathbb E_{a\sim\pi_{\theta_k}(\cdot\mid s)}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a)\right].
$$

这里问的是：**如果在这个状态改用新策略选动作，用旧策略的评价标准来看，平均会好多少？** 所以动作概率用新的，旧 advantage 则作为本轮固定的评分标准。

例如某个状态只有两个动作：

| 动作 | 旧概率 $\pi_{\theta_k}$ | 新概率 $\pi_\theta$ | 旧 advantage | 概率比 |
| --- | --- | --- | --- | --- |
| $a_1$ | $0.5$ | $0.8$ | $+1$ | $1.6$ |
| $a_2$ | $0.5$ | $0.2$ | $-1$ | $0.4$ |

直接按新分布算，结果为 $0.8\times1+0.2\times(-1)=0.6$；按旧分布加权算，结果为 $0.5\times1.6\times1+0.5\times0.4\times(-1)=0.6$。旧策略下这两个 advantage 的均值则为 $0$。

概率比越大，对应样本的影响越大；少数样本权重过大时，估计就容易波动。这也帮助理解 TRPO 为什么要限制新旧策略的差异。

#### A.3 那为什么还叫 surrogate objective？状态分布也会变

前面只固定了一个状态。但策略变了，之后访问哪些状态也会变。用归一化的折扣回报和状态分布把这一层关系写出来（$0<\gamma<1$，初始状态分布固定）：

$$
J(\theta)=(1-\gamma)\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{\infty}\gamma^t r_t\right],
\qquad
d_{\pi_\theta}(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr(s_t=s\mid\pi_\theta).
$$

$d_{\pi_\theta}$ 是归一化后的 discounted state distribution。performance difference identity（性能差异恒等式）给出：

$$
J(\theta)-J(\theta_k)
=\mathbb E_{s\sim d_{\pi_\theta},\,a\sim\pi_\theta(\cdot\mid s)}
[A^{\pi_{\theta_k}}(s,a)].
$$

这个式子从 advantage 的一步展开就能得到：

$$
A^{\pi_{\theta_k}}(s,a)
=\mathbb E[r_t+\gamma V^{\pi_{\theta_k}}(s_{t+1})-V^{\pi_{\theta_k}}(s_t)
\mid s_t=s,a_t=a].
$$

沿新策略轨迹求折扣和，value 项前后抵消，只留下初始状态的 value：

$$
\begin{aligned}
\mathbb E_{\tau\sim\pi_\theta}\sum_{t=0}^{\infty}\gamma^t
A^{\pi_{\theta_k}}(s_t,a_t)
&=\mathbb E_{\tau\sim\pi_\theta}\sum_{t=0}^{\infty}\gamma^t r_t
-\mathbb E_{s_0}[V^{\pi_{\theta_k}}(s_0)]\\
&=\frac{J(\theta)-J(\theta_k)}{1-\gamma}.
\end{aligned}
$$

两边乘上 $1-\gamma$，左侧按 $d_{\pi_\theta}$ 的定义合并，就得到上面的性能差异恒等式。

TRPO 用旧状态分布近似新状态分布，再对动作使用 A.2 的精确换算：

$$
\begin{aligned}
J(\theta)-J(\theta_k)
&\approx\mathbb E_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_\theta}
[A^{\pi_{\theta_k}}(s,a)]\\
&=\underbrace{\mathbb E_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a)\right]}_{\mathcal L(\theta_k,\theta)}.
\end{aligned}
$$

所以，**动作概率比做的是精确换算，状态分布的替换才是这里的 surrogate 近似**。

KL 约束让新旧策略保持接近，从而让这个局部代理更可信。

#### A.4 为什么旧参数处的 surrogate gradient 就是 policy gradient？

本轮更新时，旧状态分布、旧动作分布和旧 advantage 都固定，只对分子 $\pi_\theta$ 求导。利用 $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$：

$$
\begin{aligned}
g
&=\left.\nabla_\theta\mathcal L(\theta_k,\theta)\right|_{\theta=\theta_k}\\
&=\left.\mathbb E_{s,a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}
\nabla_\theta\log\pi_\theta(a\mid s)A^{\pi_{\theta_k}}(s,a)\right]\right|_{\theta=\theta_k}\\
&=\mathbb E_{s,a\sim\pi_{\theta_k}}
\left[\left.\nabla_\theta\log\pi_\theta(a\mid s)\right|_{\theta=\theta_k}
A^{\pi_{\theta_k}}(s,a)\right].
\end{aligned}
$$

最后一步中，新旧参数相同，概率比变成 $1$。而减去 state baseline 不改变 policy gradient，因为固定 $s$ 时

$$
\mathbb E_{a\sim\pi_\theta}
[\nabla_\theta\log\pi_\theta(a\mid s)V^{\pi_{\theta_k}}(s)]
=V^{\pi_{\theta_k}}(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

因此 policy gradient 中的 $Q$ 可以换成 advantage，得到的正是上面的 $g$：

$$
g=\left.\nabla_\theta J(\theta)\right|_{\theta=\theta_k}.
$$

旧策略下 advantage 的均值为 $0$，因此 $\mathcal L(\theta_k,\theta_k)=0$；离开旧参数后的增长，则由上面求出的梯度描述。

### B. backtracking line search：方向定了，沿这个方向走多远？

#### B.1 为什么算出了步长，还要试着缩小？

先看一般的最大化问题 $\max_\theta f(\theta)$。给定一个上升方向 $p$，候选参数为

$$
\theta_{\mathrm{trial}}=\theta_k+t p,\qquad t>0.
$$

选方向回答“往哪走”，line search 回答“沿这个方向走多远”。即使 $\nabla f(\theta_k)^Tp>0$，也只说明足够小的正步长能让目标上升；走远以后，函数可能弯过去，原来的局部预测就不准了。

backtracking（回溯）的思想是：先试完整步长 $t=1$，不合适就乘一个 $\alpha\in(0,1)$，依次试 $1,\alpha,\alpha^2,\ldots$。例如 $\alpha=0.5$，就是先走全程，再试一半、四分之一……**方向保持不变，只缩短这一次的位移。**

一般优化中常用 Armijo sufficient increase 条件：

$$
f(\theta_k+t p)\geq f(\theta_k)+c\,t\,\nabla f(\theta_k)^Tp,
\qquad 0<c<1.
$$

右边的 $t\nabla f^Tp$ 是一阶预测增益，$c$ 表示至少实现其中一部分；比只检查 $f$ 有没有增加更严格。这里写的是最大化版本，最小化问题对应 sufficient decrease。

#### B.2 TRPO 要检查的是两个条件

正文先用局部模型得到完整候选步 $p$：

$$
p=\sqrt{\frac{2\delta}{g^TH^{-1}g}}H^{-1}g,
\qquad \frac12p^THp=\delta.
$$

这个等号针对的是**二阶近似的 KL**。沿 $p$ 改变真正的神经网络参数后，非线性的实际 KL 可能超过 $\delta$，一阶预测的目标增益也可能没有实现，所以还需要回溯。

令 $t_j=\alpha^j$，在本轮同一批旧策略样本上重新计算：

$$
\begin{aligned}
\theta_{\mathrm{trial}}&=\theta_k+t_jp,\\
\Delta\hat{\mathcal L}_j
&=\hat{\mathcal L}(\theta_k,\theta_{\mathrm{trial}})
-\hat{\mathcal L}(\theta_k,\theta_k),\\
\hat D_j&=\widehat{\bar D}_{KL}(\theta_{\mathrm{trial}}\|\theta_k).
\end{aligned}
$$

正文采用的接受条件是 $\Delta\hat{\mathcal L}_j>0$ 且 $\hat D_j\leq\delta$；取第一个满足条件的 $j$。

例如 $\delta=0.01,\alpha=0.5$，完整步测得平均 KL 为 $0.016$，即使代理目标上升也要拒绝；半步测得 KL 为 $0.004$ 且代理目标上升，就接受半步。

操作上始终从同一个 $\theta_k$ 出发，依次试 $\theta_k+p$、$\theta_k+\alpha p$、$\theta_k+\alpha^2p$。旧策略、旧 log probability 和 $\hat A$ 保持固定。最大回溯次数用完仍未通过，就恢复 $\theta_k$。

回溯时重新计算候选策略的完整代理目标和分布 KL，用它们检验局部模型给出的步长。

### C. Lagrangian duality：怎样把 KL 约束放进优化问题？

#### C.1 拉格朗日函数与乘子分别是什么？

这里用到两个对象：Lagrangian function（拉格朗日函数）把目标和约束组合成一个标量函数，Lagrange multiplier（拉格朗日乘子）决定约束的权重。为了和正文的 surrogate $\mathcal L$ 区分，拉格朗日函数记作 $\mathscr L$。

以一个最大化问题为例：

$$
\max_z f(z),\qquad \text{s.t. }c(z)\leq0.
$$

构造

$$
\mathscr L(z,\lambda)=f(z)-\lambda c(z),\qquad \lambda\geq0.
$$

$z$ 是原问题的变量；$\lambda$ 是约束的乘子，可以理解为违反约束的“价格”。$c(z)>0$ 时，目标中减去 $\lambda c(z)$，超出约束会受到惩罚。

#### C.2 primal 与 dual：先固定价格，再寻找最紧的上界

对任意可行的 $z$，因为 $c(z)\leq0$ 且 $\lambda\geq0$，有 $f(z)\leq\mathscr L(z,\lambda)$。再对所有 $z$（包括不可行点）取上确界：

$$
q(\lambda)=\sup_z\mathscr L(z,\lambda),
\qquad
\underbrace{\sup_{z:c(z)\leq0}f(z)}_{\text{primal 最优值}}
\leq q(\lambda).
$$

所以每个 $\lambda$ 都给出原问题最优值的一个上界。dual problem（对偶问题）就是调整 $\lambda$，把这个上界压得尽可能低：

$$
\min_{\lambda\geq0}q(\lambda).
$$

原问题是 max，这里的对偶是 min。上述上界关系叫 weak duality（弱对偶）；若最紧上界等于原问题最优值，就叫 strong duality（强对偶）。下面的 TRPO 局部问题满足强对偶条件，可以沿这条路求出解。

#### C.3 用在 TRPO 的局部近似问题上

令 $\Delta\theta=\theta-\theta_k$，把参数位移作为变量：

$$
\max_{\Delta\theta}g^T\Delta\theta,
\qquad
\frac12\Delta\theta^TH\Delta\theta\leq\delta.
$$

先假设 $H\succ0$（对称正定，即任意非零 $v$ 都有 $v^THv>0$）、$g\neq0$、$\delta>0$。约束区域是椭球，目标是线性的。原点满足严格不等式 $0<\delta$，满足 Slater 条件，因此这个局部凸优化问题具有强对偶。

第一步，写出拉格朗日函数：

$$
\mathscr L(\Delta\theta,\lambda)
=g^T\Delta\theta-\lambda\left(\frac12\Delta\theta^TH\Delta\theta-\delta\right).
$$

第二步，固定 $\lambda>0$，对 $\Delta\theta$ 最大化。其 Hessian 为 $-\lambda H\prec0$，因此令梯度为 $0$ 就能得到唯一最大值：

$$
\nabla_{\Delta\theta}\mathscr L=g-\lambda H\Delta\theta=0
\quad\Longrightarrow\quad
\Delta\theta(\lambda)=\frac1\lambda H^{-1}g.
$$

第三步，代回求出 dual function。记 $C=g^TH^{-1}g>0$，则

$$
q(\lambda)
=\frac C\lambda-\frac{C}{2\lambda}+\lambda\delta
=\frac{C}{2\lambda}+\lambda\delta.
$$

第四步，对 $\lambda$ 最小化：

$$
q'(\lambda)=-\frac{C}{2\lambda^2}+\delta=0
\quad\Longrightarrow\quad
\lambda^*=\sqrt{\frac{C}{2\delta}}.
$$

因为 $q''(\lambda)=C/\lambda^3>0$，这就是唯一最小值。最终

$$
\boxed{
\Delta\theta^*
=\sqrt{\frac{2\delta}{g^TH^{-1}g}}H^{-1}g,
\qquad \theta_{k+1}=\theta_k+\Delta\theta^*.
}
$$

方向来自驻点条件 $g=\lambda H\Delta\theta$，根号系数来自 KL 预算。代入可验证 $\frac12(\Delta\theta^*)^TH\Delta\theta^*=\delta$，刚好走到局部椭球边界。

#### C.4 KKT 条件把这几步连起来

本例的 KKT 条件为：

$$
\begin{aligned}
g-\lambda H\Delta\theta&=0 &&\text{stationarity：驻点条件},\\
\frac12\Delta\theta^TH\Delta\theta-\delta&\leq0 &&\text{primal feasibility：原问题可行},\\
\lambda&\geq0 &&\text{dual feasibility：对偶可行},\\
\lambda\left(\frac12\Delta\theta^TH\Delta\theta-\delta\right)&=0
&&\text{complementary slackness：互补松弛}.
\end{aligned}
$$

互补松弛表示：约束还有余量时，乘子为 $0$；乘子大于 $0$ 时，约束必须恰好取等号。本例 $g\neq0$，驻点条件要求 $\lambda^*>0$，所以最优解会用完局部 KL 预算。

这个解给出了局部模型中的完整候选步；接下来由 D 计算方向，再由 B 的回溯确定最终接受的步长。

### D. conjugate gradient algorithm：只计算矩阵乘向量，怎样求解线性方程？

#### D.1 为什么求解 $Bx=g$ 等价于最小化一个二次函数？

先用 $B$ 表示一个对称正定矩阵，考虑

$$
\min_x\psi(x)=\frac12x^TBx-g^Tx.
$$

因为 $\nabla\psi(x)=Bx-g$、$\nabla^2\psi(x)=B\succ0$，唯一最小值满足 $Bx=g$。所以“解线性方程”也可以变成“沿二次曲面找到最低点”。在 TRPO 中，这里的 $x$ 给出更新方向，随后再按 KL 预算缩放成参数位移。

普通梯度下降每次沿负梯度前进；二次曲面的等高线若很狭长，路径容易来回折返。conjugate gradient（CG，共轭梯度）会利用之前的搜索方向，避免后续更新破坏已经完成的方向上的优化。

#### D.2 conjugate：按曲率矩阵 $B$ 衡量的正交

两个方向 $p_i,p_j$ 若满足

$$
p_i^TBp_j=0,\qquad i\neq j,
$$

就称它们关于 $B$ 共轭。当 $B=I$ 时，这就退化成普通欧氏正交。

为什么这种关系有用？沿 $p_i$ 做完精确线搜索后，有 $p_i^T\nabla\psi(x)=0$。如果随后沿 $p_j$ 移动 $t$，则

$$
p_i^T\nabla\psi(x+t p_j)
=p_i^T\nabla\psi(x)+t\,p_i^TBp_j=0.
$$

也就是说，新的移动保留了之前方向上的最优条件。这就是“共轭方向”的核心直觉。

#### D.3 从 residual 到迭代公式

定义 residual（残差）$r_i=g-Bx_i$，表示线性方程两边还差多少；它也正好是 $-\nabla\psi(x_i)$。从 $x_0=0$ 开始，$r_0=g$，第一条搜索方向取 $p_0=r_0$；若 $g=0$，初始解就已满足方程。

每轮先沿 $p_i$ 精确最小化 $\psi(x_i+\eta p_i)$：

$$
\frac{d}{d\eta}\psi(x_i+\eta p_i)
=-p_i^Tr_i+\eta p_i^TBp_i=0.
$$

所以 $\eta_i=p_i^Tr_i/(p_i^TBp_i)$。CG 中，新残差与之前的搜索方向正交，而 $p_i=r_i+\beta_{i-1}p_{i-1}$，因此 $p_i^Tr_i=r_i^Tr_i$。第一轮 $p_0=r_0$ 也满足这个关系，于是得到熟悉的公式：

$$
\begin{aligned}
z_i&=Bp_i,\\
\eta_i&=\frac{r_i^Tr_i}{p_i^Tz_i},\\
x_{i+1}&=x_i+\eta_i p_i,\\
r_{i+1}&=r_i-\eta_i z_i,\\
\beta_i&=\frac{r_{i+1}^Tr_{i+1}}{r_i^Tr_i},\\
p_{i+1}&=r_{i+1}+\beta_i p_i.
\end{aligned}
$$

最后两行给新残差加一点旧方向，使新方向与旧方向共轭。更具体地，由 $p_{i+1}^TBp_i=0$ 可得

$$
\beta_i=-\frac{r_{i+1}^TBp_i}{p_i^TBp_i}.
$$

在精确算术下，CG 的相邻残差正交，且 $Bp_i=(r_i-r_{i+1})/\eta_i$。代入上式，再用 $\eta_i p_i^TBp_i=r_i^Tr_i$，就得到 $\beta_i=\|r_{i+1}\|^2/\|r_i\|^2$。对称正定二次问题的这些正交关系可以逐轮归纳，并使所有已生成的搜索方向两两共轭。

实际计算时，当 $\|r_i\|$ 足够小，或迭代次数达到上限时停止。精确算术下，$n$ 维正定问题最多 $n$ 轮求得解；神经网络参数很多，通常只运行少量轮数得到近似方向。

这里的 $\eta_i$ 是 CG 内部更新 $x_i$ 的步长；正文的 $\alpha^j$ 是对最终策略位移做回溯，两者解决的是不同层次的问题。

#### D.4 放回 TRPO：damping、Hessian-vector product 与最终缩放

理想推导取 $B=H$，求得 $x\approx H^{-1}g$。但 KL 的 Fisher/Hessian 一般只能保证半正定，可能有零曲率方向；常见实现加入 damping：

$$
B=H+\zeta I,\qquad \zeta>0,
\qquad Bv=Hv+\zeta v.
$$

若 $H\succeq0$，则对非零 $v$ 有 $v^TBv=v^THv+\zeta\|v\|^2>0$，$B$ 变成正定矩阵，求解更稳定。此时 CG 求解 $(H+\zeta I)x=g$，得到经过正则化的方向。

CG 每轮只调用一次 $p_i\mapsto Bp_i$，可以把它理解成一个“输入向量、返回矩阵乘积”的函数。通过正文的两次自动求导得到 $Hp_i$，再加上 $\zeta p_i$ 即可；不用存下整个矩阵，更不用计算逆矩阵。旧策略固定，只对新参数分支求导，HVP 在 $\theta_k$ 处计算。

不加 damping 时，将 CG 返回的 $x$ 缩放为

$$
p=\sqrt{\frac{2\delta}{x^THx}}x.
$$

这正是正文伪代码中的写法。若精确满足 $Hx=g$，则 $x^THx=g^TH^{-1}g$，与 C 中解析式一致；CG 只有近似解时，用 $x^THx$ 直接计算该方向的二次曲率更合适。

使用 damping 时，一种常见做法是连缩放也使用同一个 $B$：

$$
p=\sqrt{\frac{2\delta}{x^TBx}}x,
\qquad x\approx B^{-1}g.
$$

此时 $\frac12p^THp\leq\frac12p^TBp=\delta$，也就是在局部 KL 预算内预留了一些余量。得到完整候选步后，再进入 B 中的回溯。

把三层计算接起来就是：**CG 求方向 $x$ → 按 KL 曲率缩放成候选位移 $p$ → 回溯检查 $\theta_k+\alpha^jp$**。方向、完整步长、最终接受的步长，分别在这三步中确定。


