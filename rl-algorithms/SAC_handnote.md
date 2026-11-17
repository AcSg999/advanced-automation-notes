# Soft actor critic (SAC)
## background & keywords
an algorithm that optimizes a stochastic policy in an off-policy way. incorporates the clipped double-Q trick, entropy regularization,

## quick facts
- SAC is an off-policy algorithm.
- The version of SAC implemented here can be used for environments with continuous action spaces and
change it a little bit for discrete action spaces.

## key equations

### Entropy-Regularized Reinforcement Learning

think entropy as a quantity that measures the randomness, the more random the policy is, the higher the entropy. IF the policy is deterministic, the entropy is zero.
For a discrete random variable $X$ with distribution $P$ and probability mass function $p(x)$, its entropy is:

$$
H(X) = \mathbb{E}_{x \sim P}\left[-\log p(x)\right] = -\sum_x p(x)\log p(x).
$$

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi}\left[
\sum_{t=0}^{\infty} \gamma^t \left(
R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot \mid s_t)\right)
\right)
\right].
$$

>$\alpha$>0 is the trade-off coefficient.
