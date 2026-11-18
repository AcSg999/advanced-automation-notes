# FlashSAC：从标量 Q 到分布式 Q

> 对应 [FlashSAC 原论文](https://arxiv.org/html/2604.04539v1) 和 [官方项目页](https://holiday-robot.github.io/FlashSAC/)。SAC 的基本公式可以对照 [02_SAC.md](02_SAC.md) 和 [sac_continuous_action.py](sac_continuous_action.py)。

keyword：off-policy, residual block, BatchNorm, RMSNorm, distributional critic, noise repetition, UTD

## 1. 背景（Background）

SAC 已经能从 replay buffer 反复利用旧数据。FlashSAC 想解决的下一个问题是：**有了大量并行仿真数据，怎样用较少的优化步骤，训练一个容量够大、又不会把自身估计误差不断放大的 critic？**

标准 SAC 里的 critic 学的是 $Q(s,a)$。它不能每次都等一条轨迹结束才知道回报，于是用 `真实奖励 + 下一状态的估计价值` 当作目标。估计值来自另一个会随训练更新的网络，这就是 **bootstrapping（自举）**。如果目标网络把某些动作估高了，在线 critic 可能跟着学高，actor 又会更愿意选择这些动作。双 Q 取较小值和慢更新的目标网络已经能缓解这个问题，FlashSAC 继续从特征尺度、权重尺度和价值目标三个地方入手。

### 1.1 Quick Facts

- 保留 SAC 的 off-policy actor–critic、replay、随机策略、双 Q 和自动温度调节。
- **网络改了**。actor 和 critic 都用带跳连的残差 MLP，critic 改用分类分布输出。
- **学习规则改了**。critic 用分布式 Bellman 目标和交叉熵，配合奖励缩放、联合 batch 前向和权重投影。
- **数据采集改了**。训练时短暂复用探索噪声。目标熵由一个目标标准差换算，便于适配不同动作维度。
- 大 batch、更新次数、replay 容量是训练配置。AMP、编译和 CPU/GPU replay 是执行层的选择。论文把这些和网络机制一起使用，实际提速仍要看任务与实现。见 [论文 §4](https://arxiv.org/html/2604.04539v1#S4)。

## 2. 网络结构（Architecture）

下图画的是本地 **LeapHand–Scissors 的 `flashdex` 配置**。actor 接收策略观测，输出动作分布参数。critic 接收更完整的训练观测和一个待打分的动作，两个 Q 分支各自给出一套回报概率。actor 更新时用在线 Q 的期望来给动作打分，critic 更新时用目标 Q 来构造下一状态的目标。

![FlashSAC actor、双 Q 和目标网络架构](images/flash_sac_network.png)

这里的“双 Q”是两套参数。实现将它们放在同一个张量的 ensemble 维度上批量算，仍各自有自己的编码器和输出头。目标双 Q 是另存的慢更新副本。见 [本地模型实现](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/models/flash_sac_model.py#L263) 和 [本轮保存的 agent.json](../regrind/logs/flashsac_jobs/latest_reward_forks/gpu_control/run/params/agent.json)。

### 2.1 Residual connection（残差连接）

设普通 MLP 的某一段是 $F(x)$。串起来的时候，下一段只收到 $F(x)$。残差块给输入多开一条直通路径，把两条路径逐元素相加：

$$
y=x+F(x).
$$

假如某个特征目前已经够用，这个块可以让 $F(x)$ 接近零，让特征基本原样传过去。要学新东西时，再由 $F(x)$ 补充变换。例如 $x=(2,3)$、$F(x)=(0.1,0.4)$，输出就是 $(2.1,3.4)$。加法要求两条路的输出维度相同，所以扩到 $4H$ 的中间层最后要投影回 $H$。

反向传播时，$\partial y/\partial x=I+J_F(x)$。就算变换分支的局部梯度很小，恒等路径仍提供一项直接梯度。它让深层网络更容易保留信息和传递梯度，训练是否稳定仍取决于各块和训练目标。

![FlashSAC 倒残差块，旁路输入和变换分支在末端相加](images/flash_sac_residual.png)

这里 `Linear → BN → ReLU` 执行两次，然后加回输入。**激活前 BatchNorm 的“前”指 ReLU 之前**。`1024 → 4096 → 1024` 是 critic 中一个块的形状，actor 同样有跳连，只是走 `128 → 512 → 128`。这些都是全连接层，输入是观测特征向量。见 [本地残差块实现](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/modules/flash_sac_layers.py#L121)。

### 2.2 BatchNorm、RMSNorm、权重投影分别管什么？

**BatchNorm 管一个 batch 内同一维特征的尺度。** replay 混着不同训练阶段的数据，特征的分布会变。在 ReLU 之前按 batch 统计量调整输入尺度，可以缓解激活长期全关、某些维度数值过大等情况。actor 在采样交互和评估时使用训练积累的运行统计量。

**末端 RMSNorm 管单个样本进入输出头前的整体特征尺度。** 某个罕见观测经过多层后可能产生很大的特征向量。RMSNorm 对该样本按特征维求均方根再缩放，减小整体幅度对 Q 输出的影响。

**权重投影管优化之后的参数尺度。** 本地 `UnitLinear` 一行权重对应一个输出特征。Adam 更新后把每行拉回单位范数，保留它学习到的方向。实现还约束 BN 的仿射参数和 RMSNorm 的缩放参数。这一步发生在参数更新后，前两项发生在前向计算里。相关公式见附录 A。见 [论文 §4.2](https://arxiv.org/html/2604.04539v1#S4.SS2) 和 [本地归一化实现](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/modules/flash_sac_layers.py#L30)。

## 3. Key Equations

### 3.1 从标量 Q 到 categorical critic

先看一步版本，省略真正终止的情况。标准 SAC 计算一个标量目标：

$$
y=r+\gamma\left[\min_i Q_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right],
\qquad a'\sim\pi_\theta(\cdot\mid s').
$$

每个在线 Q 输出一个数，优化 $(Q_{\phi_i}(s,a)-y)^2$。如果两种回报情形的均值一样，标量 Q 用同一个数描述它们。例如，回报固定为 2，和回报一半为 0、一半为 4，均值都是 2。

FlashSAC 给每个 Q 网络固定一排离散值 $z_j$，叫 **atoms（支撑点）**。本地是 $[-5,5]$ 上的 101 个点，间隔 0.1。网络输出每个点的 logit，通过 softmax 得到 $p_{i,j}$。需要标量打分时，取分布的期望：

$$
Q_i(s,a)=\sum_j p_{i,j}(s,a)z_j.
$$

这里表示的是**给定状态动作后，后续软回报的分布**。两个 Q 的分布分别计算期望，挑期望较小的那个 Q，然后拿它的**整个分布**构造目标。逐个 bin 分别取较小概率，会改变分布并可能丢失概率质量。

对下一状态分布的每个支撑点，应用一遍 Bellman 变换，再投影回原来的固定网格：

$$
\tilde z_j=\bar r+\gamma(1-d)\big[z_j-\alpha\log\pi_\theta(a'\mid s')\big].
$$

$d=1$ 表示真正终止。变换后的值通常会落在两个支撑点中间。比如 $1.24$ 落在 $1.2$ 和 $1.3$ 之间，原来这一点的概率质量分给两边，比例是 $0.6$ 和 $0.4$。超出区间的值先裁到边界。设投影后的目标概率为 $m_j$，在线 critic 学的是：

$$
\mathcal L_Q=-\mathbb E_{(s,a)\sim\mathcal D}\!\left[\sum_j m_j\log p_j(s,a)\right].
$$

**改进动机**是让 critic 拟合一组概率，给价值学习更丰富的训练信号。对输出 logit 来说，单样本交叉熵梯度是 $p_j-m_j$，反映的是预测与目标概率的差异。FlashSAC 在论文中用它配合尺度约束来改善优化稳定性。actor 仍用 Q 的**期望**优化原来的最大熵目标，这里没有引入额外的风险偏好。见 [论文 §4.2](https://arxiv.org/html/2604.04539v1#S4.SS2) 和 [本地 TD 投影实现](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/utils/categorical.py#L19)。

### 3.2 Adaptive reward scaling（自适应奖励缩放）

假如目标长期落在 $[-5,5]$ 之外，投影就会把大量概率推到两端。不同好坏程度的回报可能挤到同一个边界点。FlashSAC 跟踪折扣奖励累积量的标准差与见过的最大绝对值，用它们选择缩放因子：

$$
\bar r=\frac{r}{\max\!\left(\sqrt{\mathrm{Var}(G)+\epsilon},\;G_{\mathrm{abs,max}}/G_{\max}\right)}.
$$

标准差项提供尺度基准，最大值项限制极端回报的尺度。缩放让奖励部分更适配支撑区间，完整 TD 目标还含熵项和 bootstrap，所以计算里仍有边界裁剪。Regrind 的训练 `Train/mean_reward` 和完整轨迹评估依据环境的原始奖励。内部缩放只影响 critic 构造训练目标时看到的奖励，查看曲线时要区分这两个位置。见 [本地奖励缩放实现](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/utils/reward_normalization.py#L35)。

### 3.3 Cross-Batch Value Prediction（联合 batch 前向）

一个 replay batch 里有两类状态动作对。$(s,a)$ 是过去实际执行的动作，$(s',a')$ 中的 $a'$ 来自**当前** actor。两类输入的分布可能不同。如果让它们分别通过训练模式的 BatchNorm，均值和方差会随着 batch 来源而变化，连价值预测所用的特征尺度都会受到影响。

FlashSAC 将这两类输入沿 **batch 维**拼为 $2B$ 条，再做前向。**本地版在线 critic 和目标 critic 各跑一次这个联合 batch**，各自在其前向中用覆盖两类样本的 BN 统计。在线网络取前半构成预测，目标网络取后半构成目标。在线和目标网络参数不同，因此两套网络的 BN 数值可以不同。常见疑问是“是不是把两个状态拼成一条更长的观测？”这里拼的是两组样本行，单行输入维度保持原样。

这也解释了部分训练成本。batch 为 4096 时，构造一次 critic 更新的两次 Q 前向各自处理 8192 行，并且每次要计算双 Q，在线侧还要反传。批量算有利于 GPU 利用率，矩阵规模仍然很大。见 [论文 §4.2](https://arxiv.org/html/2604.04539v1#S4.SS2) 和 [本地 critic 更新](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/algorithms/flash_sac.py#L574)。

## 4. Exploration–Exploitation Tradeoff

### 4.1 Noise repetition（重复的是哪一部分？）

把 actor 输出的均值和标准差代入高斯采样，忽略动作区间的仿射变换：

$$
a_t=\tanh\!\big(\mu_\theta(s_t)+\sigma_\theta(s_t)\odot\varepsilon_t\big),
\qquad \varepsilon_t\sim\mathcal N(0,I).
$$

普通采样每一步重新抽 $\varepsilon_t$。FlashSAC 在一个短段内缓存 $\varepsilon$，每次收到新观测仍然重算 $\mu_\theta(s_t)$ 和 $\sigma_\theta(s_t)$。同一份噪声给策略一个持续几步的探索方向，动作仍随观测变化。

| 控制步 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 每步独立采样 | $\varepsilon_1$ | $\varepsilon_2$ | $\varepsilon_3$ | $\varepsilon_4$ | $\varepsilon_5$ |
| 复用示例，先抽到 $k=3$，再抽到 $k=2$ | $\varepsilon_A$ | $\varepsilon_A$ | $\varepsilon_A$ | $\varepsilon_B$ | $\varepsilon_B$ |
| actor 输入 | $s_1$ | $s_2$ | $s_3$ | $s_4$ | $s_5$ |

举个一维例子，前三步缓存 $\varepsilon_A=0.5$，标准差先假设都为 $0.2$。如果 actor 给出的均值依次是 $0.0,0.1,0.3$，`tanh` 前的值就是 $0.1,0.2,0.4$。持续的是 $+0.1$ 的扰动方向。真实训练里标准差也会变，扰动幅度可以随状态调整。

**为什么要持续几步？** 比如机械手需要连续几步朝同一方向微调才能尝试一种抓法。正负独立的小扰动容易在物理系统中相互抵消。持续一小段的偏移更容易产生实际的位移或接触变化。FlashSAC 从幂律分布抽持续长度，短段较多，偶尔出现长段：

$$
P(K=k)=\frac{k^{-\mu}}{\sum_{j=1}^{K_{\max}}j^{-\mu}},\qquad 1\le k\le K_{\max}.
$$

本地设置 $\mu=2$、$K_{\max}=16$。向量化采样时，这批并行环境共享换噪声的时刻，但 `randn_like(mean)` 给各环境、各动作维采各自的噪声值。复用发生在采集数据的 `act()` 路径，训练目标中的下一动作按 actor 分布重新采样，确定性评估使用 `tanh(mean)`。论文 Figure 10 对比了逐步重抽与噪声复用，报告了作者任务上的收益。见 [论文 §4.3、§6.4](https://arxiv.org/html/2604.04539v1#S6.SS4) 和 [本地 act()](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/algorithms/flash_sac.py#L331)。

### 4.2 目标标准差与实际探索噪声

自动温度调节在 SAC 里已有。FlashSAC 用一个参考高斯的标准差 $\sigma_{\mathrm{tgt}}$ 来指定目标熵。对于 $D$ 个独立动作维度：

$$
\mathcal H_{\mathrm{tgt}}=\frac{D}{2}\log\big(2\pi e\,\sigma_{\mathrm{tgt}}^2\big).
$$

把 $\sigma_{\mathrm{tgt}}$ 从 0.15 改成 0.10，会降低自动温度调节的熵目标。actor 输出的实际标准差 $\sigma_\theta(s)$ 仍由网络根据状态给出。**目标熵控制期望探索强度，噪声复用控制扰动持续多久**。这是两件可分别消融的事。

## 5. Pseudocode（对应本地训练循环）

1. 初始化 actor、在线双 Q、目标双 Q、温度和 replay。仿真提供 actor 与 critic 各自的观测组。
2. **Collect**。根据观测计算均值和标准差，按计数器决定复用还是重抽噪声，执行动作。把交互数据写入 replay，并更新奖励尺度统计。预热达到可采样条件才开始优化。
3. **Sample**。取一批历史状态动作对、聚合奖励、下一观测与终止标记。本地配置使用 5-step 回报。
4. **Actor & temperature**。在轮到 actor 更新的批次，用在线双 Q 期望的较小者给新动作打分，最小化 $\alpha\log\pi-Q_{\min}$，并按目标熵更新温度。
5. **Critic**。当前 actor 重新采样下一动作。在线和目标双 Q 各做一次联合 batch 前向，用目标侧的回报分布构造投影后的分类目标，更新在线 Q。
6. **Normalize & EMA**。actor / critic 各自的优化步骤结束后执行权重投影，目标 critic 在每次 critic 更新后跟随在线网络。随后继续采样。

`UTD` 可按“梯度更新次数 / 新交互条数”来读。这里 1024 个环境每推进一个控制步就收集 1024 条数据，做 2 次 critic 批次更新。每批取 4096 条 replay 样本，平均每个新交互对应约 8 次 replay 样本抽取，**这 8 次不是 8 次优化器更新**。多次抽取允许重复命中同一条历史经验。actor 与温度每隔 2 次 critic 更新做一次。这个排序来自 [本地 update()](../regrind/.venv-flashsac/lib/python3.11/site-packages/rsl_rl_flashsac/algorithms/flash_sac.py#L442)。

### 5.1 为什么把大网络、大 batch 和低 UTD 放在一起？

- **并行环境和 replay 提供数据。** 并行仿真让一次控制步能收集多条轨迹的数据，容量大的 replay 让少见的成功或失败经验留得更久。容量一直增大会让近期经验更难被抽到，论文也把容量作为单独的实验因素。
- **大 batch 让每次更新看更多经验。** 梯度和 BN 统计通常更稳定，而且可以把矩阵运算交给 GPU 并行。batch 继续增大也会增加显存和单次更新成本。
- **低 UTD 减少优化器更新。** 每采到一批新数据，少做几轮梯度更新，将预算留给更大的 batch 和模型。这套搭配依赖 critic 在较少的更新下仍能学好。AMP 和编译进一步减少数值运算或执行开销。

论文以墙钟时间检验这套组合，剪刀实验则说明瓶颈会随配置变化。本地 `flashdex` 的大 critic 在同样样本预算下仍比 PPO 花了更多时间，且只更换 replay 设备的收益有限。判断提速是否有效，要同时看每步耗时和**达到同等评估 reward 所需的样本数**。见 [论文速度消融 §6.2](https://arxiv.org/html/2604.04539v1#S6.SS2) 和 [本地耗时分析](../regrind/logs/ppo_jobs/latest/comparison/flashsac_tuning_guide.md)。

## 6. 对照当前剪刀实验

| 已经做过的改动 | 改动在哪一层 | 剪刀实验结果 |
| --- | --- | --- |
| replay CPU → GPU | 数据存储与搬运 | 同一稳定窗口约 186.4 → 182.7 ms/并行步，耗时下降约 2% |
| actor LR → `1e-4` | 优化器学习率 | 140k 父模型到该分支最终权重，确定性验证 reward 从 46.28 提高到 46.70，换评估 seed 后从 46.27 到 46.65 |
| 目标 sigma `0.15 → 0.10` | 探索强度 | 同 GPU 对照相比，训练末段 reward 从 22.09 到 22.75。确定性评估为 46.34，GPU 原参数组为 46.53 |

这三组的残差块、双 Q 和分类 TD 损失都相同。上面的训练 reward 含探索动作和随机参考起点，完整轨迹评估使用确定性动作，两者分别回答不同的问题。见 [分叉实验报告](../regrind/logs/flashsac_jobs/latest_reward_forks/report.md)。

**怎样用实验判断一个设计有没有贡献？** 先确定自己问的是性能、速度还是稳定性，再对准改动的位置。

| 要回答的问题 | 对照怎么设 | 比较什么 |
| --- | --- | --- |
| 分布式 critic 是否值得？ | 同一残差主干和奖励缩放。一个输出 101 个 logits、用交叉熵，另一个输出标量、用 TD 回归 | 固定评估 reward、不同训练 seed 的波动、学习耗时 |
| 跳连是否有效？ | 保持变换分支的宽度和层数，对比 $x+F(x)$ 与 $F(x)$ | 学习曲线、梯度范数、同样本处的评估 reward |
| 噪声复用是否帮助探索？ | 保持目标 sigma 相同，对比 $K=1$ 与当前长度分布 | 早期采样效率、训练动作变化率、确定性评估 |
| 能否保住 reward 又降低 critic 成本？ | 对比现有双 Q、单纯缩宽的双 Q、共享前端但保留独立后段的双 Q | 采样和更新耗时、达到同一评估 reward 所需时间。额外看两条 Q 的一致程度，判断共享是否削弱双 Q 的纠偏作用 |

正式结构消融从头训练，统一样本预算和评估，再画 reward–样本与 reward–时间两张图。共享前端也是一个待验证的设计。按论文 Figure 9 的顺序逐层加回模块，可以回答“加入这一组机制后有什么变化”，每次只关一个组件则更适合检查它在当前完整配置中的作用。见 [论文架构消融 §6.3](https://arxiv.org/html/2604.04539v1#S6.SS3)。

## 7. Documentation

- [FlashSAC 论文](https://arxiv.org/html/2604.04539v1) 的 §4 看设计，Figure 9 看架构消融，Figure 10 看目标熵与噪声复用。
- [官方 RSL-RL 实现，固定到 Regrind 使用的版本](https://github.com/Holiday-Robot/rsl_rl_flashsac/tree/9822b1b7e0b5cce45e7f79d6b3e2c169fa042faa) 可按 `FlashSACBlock` → `FlashSACDoubleCritic` → `act` → `_update_critic` 阅读。
- [Regrind 接入说明](../regrind/FLASH_SAC.md) 记录了运行入口和参数。[本地适配器](../regrind/source/regrind/regrind/modules/flashsac.py) 处理提前截断时的实际 bootstrap 步数，也支持续训时用载入策略预热 replay。

## Appendix

### A. BN、RMSNorm 与权重投影的公式

设一批特征写成 $X\in\mathbb R^{B\times H}$，一行是一个样本。BN 对每一列在 $B$ 个样本上求均值 $\mu_j$、方差 $v_j$，再应用可学习的缩放 $g_j$ 与偏移 $b_j$：

$$
\mathrm{BN}(X)_{ij}=g_j\frac{X_{ij}-\mu_j}{\sqrt{v_j+\epsilon}}+b_j.
$$

RMSNorm 对某一行 $x$ 在 $H$ 个特征上求均方根，不减均值：

$$
\mathrm{RMSNorm}(x)_j=g_j\frac{x_j}{\sqrt{\frac1H\sum_{k=1}^{H}x_k^2+\epsilon}}.
$$

看一个 $2\times 3$ 矩阵就能区分两者。前者每次取一列的两个数，后者每次取一行的三个数。如果在线性层中 $y_j=w_j^\top x$，权重投影则在优化器更新后对参数行做 $w_j\leftarrow w_j/\max(\lVert w_j\rVert_2,\epsilon)$。三种方法操作的对象分别是**一列样本特征、一行样本特征、一行网络参数**。

### B. Actor 怎么通过 critic 的梯度学习？

将采样写成 $a_\theta(s,\epsilon)=\tanh(\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon)$，把它喂给 Q。对固定的噪声，动作对 actor 参数可导。更新 actor 时，梯度沿 $Q(s,a)\to a\to\theta$ 回传，在线 critic 参数保持固定。这叫 **reparameterization（重参数化）**。实现还会对 `tanh` 变换后的 log probability 做 Jacobian 修正。参见 [SAC 学习笔记](02_SAC.md)。

### C. 多步回报、真正终止与时间截断

一步公式里的真实奖励可替换为 $R^{(k)}_t=\sum_{j=0}^{k-1}\gamma^j r_{t+j}$，下一价值项则从第 $k$ 步接上，折扣为 $\gamma^k$。未提前结束时本地取 $k=5$。真正终止时后面没有可接的价值。若只是时间上限截断，任务价值仍可从重置前的最后观测接上，例如第 3 步被截断就用 $\gamma^3$。本地适配器记录每条 replay 的实际 horizon 来处理这种边界。见 [本地 replay 和 critic 适配](../regrind/source/regrind/regrind/modules/flashsac.py#L24)。

### D. 目标 critic 为什么要慢一点？

在线 critic 每批都在变。用它自己的最新值立即构造新目标，会让拟合对象同时移动。目标副本以指数滑动平均跟随：

$$
\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi.
$$

例如一个目标参数为 $1.0$，在线参数为 $1.4$，取 $\tau=0.01$ 后目标变成 $1.004$。这让目标每次只移动一点。双 Q 取小主要针对高估，EMA 主要控制目标移动速度，两者在标准 SAC 中都已有。
