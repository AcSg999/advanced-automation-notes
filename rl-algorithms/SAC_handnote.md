# Soft actor critic (SAC)

[spinningup sac](https://spinningup.openai.com/en/latest/algorithms/sac.html)


## Background

SAC 用随机策略做强化学习，在追求累计回报的同时奖励策略的熵。它可以反复从经验池取过去的交互，用来评价和改进当前策略，因此属于 off-policy。手记以 CleanRL 的连续动作实现为准。

## 一轮训练

![网络总图原图](images/sac_whole.png)

![伪代码原图](images/sac_original_pseudocode.jpg)

这四张图按早期 SAC 的结构讲，含在线 V 和目标 V。$s$ 是当前状态，$a$ 是实际执行的动作，$r$ 是奖励，$s'$ 是下一状态，$d$ 是终止标记。

### 1. Experience 的产生过程

![状态进入 actor，动作进入环境，转移写入经验池](images/sac_experience.png)

Actor 读入 $s$，采样得到 $a$。环境执行 $a$，返回 $r,s',d$。把这一条转移 $(s,a,r,s',d)$ 存进经验池，未终止时从 $s'$ 继续交互。

### 2. 两个 Q 网络的更新过程

![旧状态和动作进入在线双 Q，下一状态进入目标 V，两边汇合计算回归损失](images/sac_q_update.png)

两个在线 Q 都读回放里的 $(s,a)$。目标 V 读下一状态 $s'$，得到 $\bar V(s')$。令 $\gamma$ 为折扣因子，固定目标为 $y=r+\gamma(1-d)\bar V(s')$。**两个 Q 各自拟合同一个 $y$，两个都更新。**

### 3. V critic 网络的更新过程

![回放状态分别进入在线 V 与 actor，在线 V 拟合双 Q 加熵构造的软价值](images/sac_v_update.png)

在线 V 只读 $s$。同一个 $s$ 也给当前 actor，从中采一个新动作 $\tilde a$。用在线双 Q 的较小值减去 $\alpha\log\pi(\tilde a\mid s)$ 得到软价值目标，其中 $\alpha$ 是熵的权重。固定该目标做回归，只更新在线 V。目标 V 通过滑动平均跟随在线 V。

### 4. Actor 网络的更新过程

![新采动作进入在线双 Q 和对数概率两条支路，合成损失后更新 actor](images/sac_policy_update.png)

Actor 从回放取 $s$，用重参数化采出新动作 $\tilde a$。在线双 Q 取小值给动作打分，加上对数概率项得到损失 $\alpha\log\pi(\tilde a\mid s)-\min_i Q_i(s,\tilde a)$。Q 对动作的梯度传回 actor，这一步只更新 actor。

后面的公式和 Questions 对应 CleanRL。它在更新双 Q 时，直接用**目标双 Q 加熵项**估计下一状态的软价值，接替图中目标 V 的位置。[ICML 2018 SAC 原论文](https://proceedings.mlr.press/v80/haarnoja18b.html) · [CleanRL 源码](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py#L233)

## Key Equations

### 1. 最大熵目标

#### 策略熵

consider entropy as a quantity that measures the randomness, the more random the policy is, the higher the entropy. For discrete actions, a deterministic policy has zero entropy.（在深度学习decision tree中也有出现此概念）
For a discrete random variable $X$ with distribution $P$ and probability mass function $p(x)$, its entropy is:

$$
H(X) = \mathbb{E}_{x \sim P}\left[-\log p(x)\right] = -\sum_x p(x)\log p(x).
$$

连续动作时，$\pi(a\mid s)$ 表示概率密度。给定 $s$ 的微分熵仍可用「负对数密度的平均值」来算：

$$
H(\pi(\cdot\mid s))
=-\int_{\mathcal A}\pi(a\mid s)\log\pi(a\mid s)\,\mathrm da.
$$

连续密度可以大于 1，所以微分熵也可以为负。代码里的 `log_pi` 是一次动作采样的对数密度，`(-log_pi).mean()` 可以用来估计一批状态上的平均熵。

#### 奖励与熵的折中

优化目标变成
$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[
\sum_{t=0}^{\infty} \gamma^t \left(
R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot \mid s_t)\right)
\right)
\right].
$$

其中 $J(\pi)$ 是策略 $\pi$ 的目标函数值，最优策略为 $\pi^*=\arg\max_\pi J(\pi)$。

温度 $\alpha>0$ 调节熵在目标中的权重。动作价值相近时，较大的 $\alpha$ 让策略更愿意把概率分给多个动作。

### 2. 价值函数与 Soft Bellman 方程

符号：$s,a$ 为当前状态、动作，$s',a'$ 为下一时刻状态、动作；$p$ 为状态转移分布，$\pi$ 为策略；$r$ 为期望即时奖励，$\gamma\in[0,1)$ 为折扣因子，$\alpha>0$ 为熵权重；上标 $\pi$ 表示按该策略评估，下标 $\mathrm{soft}$ 表示加入熵奖励。

#### 标准的 Q function 和 V function

$Q^\pi(s,a)$：从状态 $s$ 开始，先执行指定动作 $a$，之后按照策略 $\pi$ 行动，得到的期望折扣累计奖励。

$$
Q^\pi(s,a)
=r(s,a)+\gamma\mathbb{E}_{s'\sim p(\cdot\mid s,a)}
\left[\mathbb{E}_{a'\sim\pi(\cdot\mid s')}
\left[Q^\pi(s',a')\right]\right].
$$

$V^\pi(s)$：从状态 $s$ 开始，按照策略 $\pi$ 行动的期望折扣累计奖励，其 Bellman 递推式如下，其中 $R_{t+1}$ 是执行 $a_t$ 后获得的即时奖励，期望包含策略选动作与环境转移的随机性。

$$
\begin{aligned}
V^\pi(s)
&=\mathbb{E}_{\pi}\left[
R_{t+1}+\gamma V^\pi(s_{t+1})\mid s_t=s
\right]\\
&=\mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[Q^\pi(s,a)\right].
\end{aligned}
$$

代入可得标准 Bellman 方程：

$$
Q^\pi(s,a)
=r(s,a)+\gamma\mathbb{E}_{s'\sim p(\cdot\mid s,a)}
\left[V^\pi(s')\right].
$$

#### 引入熵：Soft Q function 和 Soft V function

Soft Q function：在标准 Q 的递推中，从下一时刻起加入熵奖励 $-\alpha\log\pi(a'\mid s')$。

$$
\begin{aligned}
Q^\pi_{\mathrm{soft}}(s,a)
&=r(s,a)+\gamma\mathbb{E}_{s'\sim p(\cdot\mid s,a)}
\Bigl[\mathbb{E}_{a'\sim\pi(\cdot\mid s')}\bigl[
Q^\pi_{\mathrm{soft}}(s',a')\\
&\hspace{10em}-\alpha\log\pi(a'\mid s')
\bigr]\Bigr].
\end{aligned}
$$

Soft V function：在当前状态下，对 Soft Q 与当前熵奖励之和取动作平均。

$$
V^\pi_{\mathrm{soft}}(s)
=\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[Q^\pi_{\mathrm{soft}}(s,a)-\alpha\log\pi(a\mid s)\right].
$$

#### Soft Bellman 方程与策略评估

将 Soft V 的定义代入 Soft Q，可得：

$$
Q^\pi_{\mathrm{soft}}(s,a)
=r(s,a)+\gamma\mathbb{E}_{s'\sim p(\cdot\mid s,a)}
\left[V^\pi_{\mathrm{soft}}(s')\right].
$$

固定策略时，用这个方程反复评价 Q。在奖励与策略熵有界、精确计算期望的情况下，迭代收敛到该策略的 $Q^\pi_{\mathrm{soft}}$。CleanRL 用当前策略采一个下一动作，以双 Q 和熵项估计上式中的 $V^\pi_{\mathrm{soft}}$，直接构造在线 Q 的训练目标。

### 3. Critic 更新

#### 双 Q 的 Bellman 目标

从回放池取一批转移 $B$，每条为 $(s,a,r,s',d)$，其中 $a$ 是实际执行的旧动作，$d=1$ 表示终止。当前策略 $\pi_\theta$ 在 $s'$ 采新动作 $a'$，其中 $\theta$ 是 actor 参数。两个目标 Q 网络 $\bar Q_1,\bar Q_2$ 用于构造共同的回归目标。

$$
y=r+\gamma(1-d)\Big[\min_{i=1,2}\bar Q_i(s',a')
-\alpha\log\pi_\theta(a'\mid s')\Big].
$$

#### 在线双 Q 的回归损失

两个在线 Q 网络 $Q_1,Q_2$ 都读入回放里的 $(s,a)$，对同一个固定目标 $y$ 做 MSE 回归。

$$
L_Q=\operatorname{mean}_{B}\!\left[(Q_1(s,a)-y)^2\right]
   +\operatorname{mean}_{B}\!\left[(Q_2(s,a)-y)^2\right].
$$

#### 目标双 Q 的 Polyak 更新

设 $\phi_i$ 和 $\bar\phi_i$ 分别为第 $i$ 个在线 Q 和目标 Q 的参数。CleanRL 先复制在线参数，再在每次目标更新时用系数 $\tau$ 做滑动平均，默认 $\tau=0.005$。

$$
\bar\phi_i\leftarrow(1-\tau)\bar\phi_i+\tau\phi_i,
\qquad i=1,2.
$$

### 4. Actor 更新

策略评估固定策略求 Q。到了策略改进，我们固定状态 $s$ 和刚得到的软价值评分 $Q(s,a)$，再问：**如果高分动作都值得尝试，该给每个动作分配多少概率？**

#### EBP：把动作评分变成概率密度

设 $a$ 是要打分的动作，$\alpha>0$ 是温度。EBP（energy-based policy，基于能量的策略）给每个动作一个能量 $E(s,a)$，再用 $e^{-E(s,a)}$ 当权重。希望 Q 越高的动作权重越大，因此取：

$$
-E(s,a)=Q(s,a)/\alpha.
$$

用 $b$ 遍历动作空间 $\mathcal A$，把权重积分得到归一化常数 $Z_Q(s)$，也叫配分函数。除以它，得到理想策略的概率密度 $p_Q$：

$$
p_Q(a\mid s)=\frac{\exp(Q(s,a)/\alpha)}{\int_{\mathcal A}\exp(Q(s,b)/\alpha)\,\mathrm db},\qquad
Z_Q(s)=\int_{\mathcal A}\exp(Q(s,b)/\alpha)\,\mathrm db.
$$

离散动作时改用求和,即softmax归一化。例如两个动作的 Q 为 2 和 1，$\alpha=1$，归一化后的概率约为 73% 和 27%。温度越低，权重越集中在高 Q 动作附近。

#### KL 与 Actor 损失

用 $\pi(a\mid s)$ 表示正在训练的策略。$D_{\mathrm{KL}}(\pi\|p_Q)$ 是从 $\pi$ 采动作时，两者给同一个动作的对数密度差的平均值：

$$
D_{\mathrm{KL}}(\pi\|p_Q)
=\mathbb E_{a\sim\pi}\!\left[\log\pi(a\mid s)-\log p_Q(a\mid s)\right].
$$

代入 $\log p_Q(a\mid s)=Q(s,a)/\alpha-\log Z_Q(s)$，把动作价值和熵放在左边：

$$
\mathbb E_{a\sim\pi}\!\left[Q(s,a)-\alpha\log\pi(a\mid s)\right]
=\alpha\log Z_Q(s)-\alpha D_{\mathrm{KL}}(\pi\|p_Q).
$$

固定 Q 时，右边第一项是常数。最小化 KL 就会提高左边的软价值。代码用当前 actor 采样来估计期望，去掉固定项后，实际最小化的是 $\alpha\log\pi-Q$。CleanRL 用压缩高斯策略近似 $p_Q$，用在线双 Q 中较小的那个给动作打分。

#### 把策略评估和策略改进连起来

记 $\pi_{\mathrm{old}}$ 为当前策略，$\tilde\pi$ 为候选策略，$\Pi$ 为允许的策略族。**评估**固定 $\pi_{\mathrm{old}}$，用前面的 Soft Bellman 方程求出 $Q^{\pi_{\mathrm{old}}}_{\mathrm{soft}}$。**改进**固定这个 Q，构造目标分布 $p_{Q^{\pi_{\mathrm{old}}}_{\mathrm{soft}}}$，再从策略族里找最接近它的策略：

$$
\pi_{\mathrm{new}}(\cdot\mid s)
=\arg\min_{\tilde\pi\in\Pi}
D_{\mathrm{KL}}\!\left(
  \tilde\pi(\cdot\mid s)
  \,\middle\|\, p_{Q^{\pi_{\mathrm{old}}}_{\mathrm{soft}}}(\cdot\mid s)
\right).
$$

上式是在单个状态 $s$ 写的改进目标。CleanRL 用一个共享参数的 actor，在回放池采出的多个状态上做小批次梯度更新。

如果策略能精确表示 $p_Q$，KL 为零，最优软状态价值就是 $\alpha\log Z_Q(s)$。在离散动作下，这是平滑最大值 **log-sum-exp**，输出一个标量。上面的 **softmax** 则输出每个动作的概率。设 $Q^\star$ 是理想的最优软动作价值，把这个最优软状态价值放回 Bellman 方程：

$$
Q^\star(s,a)=r(s,a)
+\gamma\mathbb E_{s'\sim p(\cdot\mid s,a)}
\!\left[\alpha\log Z_{Q^\star}(s')\right].
$$

这里的星号表示策略已经优化到理想分布。CleanRL 的 critic 更新用当前 actor 在下一状态采动作来估计软价值，actor 更新再朝理想分布迈一步。

#### 高斯重参数化与 Tanh 动作压缩

Actor 在状态 $s$ 输出高斯的均值 $\mu_\theta(s)$ 和对数标准差 $\log\sigma_\theta(s)$。按一维写，$\epsilon$ 是标准高斯噪声，$u$ 是压缩前的动作，$c=(a_{\max}-a_{\min})/2$ 和 $b=(a_{\max}+a_{\min})/2$ 分别是动作区间的半宽和中点。

$$
\epsilon\sim\mathcal N(0,1),\qquad
u=\mu_\theta(s)+\sigma_\theta(s)\epsilon,\qquad
a_\theta=c\tanh(u)+b.
$$

`rsample()` 实现前两步。`get_action()` 对 $u$ 用 `tanh` 后缩放到环境的动作范围。`forward()` 里的另一次 `tanh` 用来限制 $\log\sigma_\theta$。

#### 动作变换后的对数密度

用 $p_U(u\mid s)$ 表示压缩前的高斯密度，$\pi_\theta(a_\theta\mid s)$ 表示实际动作的密度。变换前后同一小段的概率相同，即 $\pi_\theta(a_\theta\mid s)\,\mathrm da_\theta=p_U(u\mid s)\,\mathrm du$，因此密度要除以该变换的导数 $\mathrm da_\theta/\mathrm du=c(1-\tanh^2 u)$。

$$
\log\pi_\theta(a_\theta\mid s)
=\log p_U(u\mid s)-\log\!\left[c(1-\tanh^2 u)\right].
$$

多维动作逐维修正后求和，得到代码中的 `log_pi`。CleanRL 在修正项的对数内加 `1e-6`，避免数值上出现 $\log 0$。 [CleanRL 动作采样](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py#L129)

#### 小批次策略损失

从回放批次 $B$ 取状态 $s$，让当前 actor 重新采样 $a_\theta$，用在线双 Q 的较小值评分。

$$
L_\pi(\theta)=\operatorname{mean}_{B}\!\left[
\alpha\log\pi_\theta(a_\theta\mid s)-\min_{i=1,2}Q_i(s,a_\theta)
\right].
$$

### 5. 自动温度调节

#### 目标熵与调温损失

开启 `autotune` 时，CleanRL 取目标熵 $\mathcal H_{\mathrm{tar}}=-n_a$，其中 $n_a$ 是动作维数。优化变量是 $\beta=\log\alpha$，这样 $\alpha=e^\beta$ 始终为正。设 $\widehat{\mathcal H}=-\operatorname{mean}_B[\log\pi_\theta(a_\theta\mid s)]$ 为当前批次的平均熵估计，调温时固定这批对数密度。

$$
L_\alpha(\beta)
=-\operatorname{mean}_B\!\left[e^\beta
\bigl(\log\pi_\theta(a_\theta\mid s)+\mathcal H_{\mathrm{tar}}\bigr)\right]
=e^\beta(\widehat{\mathcal H}-\mathcal H_{\mathrm{tar}}).
$$

#### 温度更新

梯度 $\partial_\beta L_\alpha=e^\beta(\widehat{\mathcal H}-\mathcal H_{\mathrm{tar}})$ 决定调温方向。平均熵低于目标时增大 $\alpha$，高于目标时减小 $\alpha$。代码用 Adam 更新 $\beta$，再取指数得到新温度。关闭自动调温时使用固定的 `args.alpha`。 [CleanRL 温度更新](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

## Questions

**1. SAC 的 critic 是训练两个 Q 网络吗？怎么反传？**

是。CleanRL 有两个在线 Q（`qf1`、`qf2`）和各自的目标副本。一次回归会**同时更新两个在线 Q**，两者都用回放里的旧动作 $a$ 作预测，拟合同一个目标 $y$。

![Critic 更新：在线 Q 读取旧动作，目标 Q 给下一动作估值](images/sac_code_critic.png)

`no_grad()` 固定目标 $y$，`q_optimizer` 对 $L_Q$ 反传后更新在线双 Q。截图公式里的 $\tfrac12$ 会把损失和梯度缩小一半，CleanRL 的 `mse_loss` 直接取平均。目标副本通过滑动平均跟随在线网络。[CleanRL 双 Q 更新](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py#L233)

**2. MSBE minimization 和 clipped double-Q 是什么？**

- **MSBE minimization** 是缩小 Q 预测与贝尔曼平均目标之间的平方误差。对同一个 $(s,a)$，下一状态和下一动作可能不同，因此用 $\mathbb E[y\mid s,a]$ 表示平均目标。代码每次取样本 $y$ 做 MSE。把目标分支固定后，反复用样本做回归会把预测推向这个平均值。
- **Clipped double-Q** 是对**同一个动作**用两个 Q 打分，再取较小值。例如两个目标 Q 对 $a'$ 给 12 和 8，就用 8 构造 $y$。这样可以缓解偶然高估。取小值发生在**构造目标时**，回归时两个在线 Q 都更新。[Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)

**3. SAC 的 actor 怎么更新？重参数化在做什么？**

**重参数化的目的是解决 actor 采样后，Q 对动作的梯度如何传回 actor 的问题。** 为了解决这个问题，`rsample()` 把「从 actor 输出的分布直接采样」转化为「从固定分布采样噪声，再用 actor 输出的均值和标准差计算动作」。

反传时，把本次抽到的 $\epsilon$ 当常数。例如 $\epsilon=0.5$，均值增加 $0.1$ 会让压缩前的动作 $u$ 增加 $0.1$，标准差增加 $0.1$ 会让 $u$ 增加 $0.05$。这给出了策略参数影响动作的路径。

![Actor 更新：新动作经在线 Q 和 log-prob 形成策略损失](images/sac_code_actor.png)

Q 项的梯度走「在线 Q → 新动作 → actor 参数」。CleanRL 的 `backward()` 可以给在线 Q 计算梯度，`actor_optimizer.step()` 只更新它持有的 actor 参数。[CleanRL 动作采样与 actor 更新](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py#L129)

**4. SuperMimic 是让 PPO 穿过世界模型反传吗？更新的是 actor 吗？**

SuperMimic **主方法用短预测轨迹更新 actor**。奖励梯度经过世界模型预测的状态，跨时间步传回前面的动作和 actor。做这一步时，世界模型的参数固定。论文中的 **PPO-WM** 是对照组，它用同一个世界模型产生轨迹，仍靠优势和动作概率比更新策略。普通 PPO 也调用 `backward()`，反传沿当前策略的对数概率走。[SuperMimic 项目页](https://supermimic.github.io/) · [PPO 手记](PPO_handnote.md)

**5. EBP 和 KL 为什么会出现在 actor 的推导里？**

EBP 定义理想的动作分布，KL 衡量 actor 与它的差距。展开 KL 并去掉与 actor 参数无关的归一化项，就得到前面的策略损失。
