# TRPO: Trust Region Policy Optimization

[Original notes](TRPO_handnote.md) · [Spinning Up: TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html#background)

Keywords: KL divergence, surrogate advantage, natural policy gradient, conjugate gradients, backtracking line search.

## Background

TRPO seeks a policy update that improves performance while limiting how much the new action distribution differs from the old one. The motivation is to make useful progress while keeping the local approximation behind the update reliable.

Recall policy gradients from [Lecture 9 of the reinforcement learning notes](../rl-xihu/西湖大学强化学习.en.md). Let $\theta_t$ be the current policy parameters, $J$ the performance objective, $\alpha>0$ a learning rate, $d_{\pi_{\theta_t}}$ the state weighting appropriate to that objective, and $q_{\pi_{\theta_t}}$ the action-value function:

$$
\begin{aligned}
\theta_{t+1}
&=\theta_t+\alpha\left.\nabla_\theta J(\theta)\right|_{\theta=\theta_t}\\
&=\theta_t+\alpha\mathbb E_{S\sim d_{\pi_{\theta_t}},\,A\sim\pi_{\theta_t}(\cdot\mid S)}
\left[\left.\nabla_\theta\ln\pi(A\mid S,\theta)\right|_{\theta=\theta_t}
q_{\pi_{\theta_t}}(S,A)\right].
\end{aligned}
$$

Replacing the expectation with samples and estimating action value with a return gives stochastic gradient ascent:

$$
\begin{aligned}
\theta_{t+1}
&=\theta_t+\alpha
\left.\nabla_\theta\ln\pi(a_t\mid s_t,\theta)\right|_{\theta=\theta_t}q_t(s_t,a_t),\\
q_t(s_t,a_t)&=G_t=\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k.
\end{aligned}
$$

Here the source indexes a reward received after action $a_t$ as $r_{t+1}$. Ordinary policy gradients control parameter displacement through a learning rate. A small parameter displacement can nevertheless change the action distribution substantially. TRPO measures that distributional change with KL divergence.

## Quick facts

- **On-policy learning.** Collect a batch with the current policy, update it, and collect the next batch with the new policy. The main training loop does not repeatedly sample from a long-lived replay buffer.
- **Discrete and continuous actions.** A categorical policy can select among actions such as left and right; a continuous distribution, such as a Gaussian, can generate joint torques.
- **Parallel sampling.** The Spinning Up implementation supports MPI, a system for exchanging data between processes. Several processes can run environments in parallel, aggregate gradients, and update the same policy together.

## Key equations

Let $\pi_\theta$ denote the new policy. The old parameters $\theta_k$ stay fixed during update $k$. Let $A^{\pi_{\theta_k}}$ be the old policy's advantage and $\delta>0$ the allowed average KL change. The update studied in these notes is

$$
\begin{aligned}
\theta_{k+1}
&=\arg\max_\theta\;
\underbrace{\mathbb E_{s,a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}
A^{\pi_{\theta_k}}(s,a)\right]}_{\mathcal L(\theta_k,\theta):\ \text{surrogate objective}}\\[6pt]
\text{s.t.}\quad
&\underbrace{\mathbb E_{s\sim\pi_{\theta_k}}
\left[D_{KL}\left(\pi_\theta(\cdot\mid s)\,\|\,\pi_{\theta_k}(\cdot\mid s)\right)\right]}_{\bar D_{KL}(\theta\|\theta_k):\ \text{average KL divergence}}
\leq\delta.
\end{aligned}
$$

The notation $s,a\sim\pi_{\theta_k}$ abbreviates states visited under the old policy and actions sampled from that policy. The KL direction above is the convention used in the original notes and is retained throughout this edition.

For probability distributions $P,Q$, KL divergence is

$$
D_{KL}(P\|Q)=\mathbb E_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right]\geq0.
$$

It is zero for identical distributions and is generally asymmetric. In particular, reversing $P$ and $Q$ changes its meaning and usually changes its value.

The advantage measures an action relative to the policy's average performance at the same state:

$$
A^{\pi_{\theta_k}}(s,a)=Q^{\pi_{\theta_k}}(s,a)-V^{\pi_{\theta_k}}(s).
$$

The probability ratio reweights old-policy actions:

$$
\begin{aligned}
&\mathbb E_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a)\right]\\
&\qquad=\mathbb E_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_\theta}
[A^{\pi_{\theta_k}}(s,a)].
\end{aligned}
$$

This action-expectation conversion is exact when the sampling distribution covers the target distribution. The surrogate approximation enters through retaining the old state distribution after the policy changes.

### Comparison with off-policy actor-critic

Let $\beta$ be a behavior policy, $\rho$ the state weighting in the actor update, and $\pi_\theta$ its target policy. The actor update expression discussed in Lecture 10 and the TRPO surrogate gradient share a recognizable structure:

$$
\begin{aligned}
g_{\mathrm{AC}}(\theta)
&=\mathbb E_{s\sim\rho,\,a\sim\beta}
\left[\frac{\pi_\theta(a\mid s)}{\beta(a\mid s)}
\nabla_\theta\ln\pi_\theta(a\mid s)A^{\pi_\theta}(s,a)\right],\\
\nabla_\theta\mathcal L(\theta_k,\theta)
&=\mathbb E_{s,a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}
\nabla_\theta\ln\pi_\theta(a\mid s)A^{\pi_{\theta_k}}(s,a)\right].
\end{aligned}
$$

Both contain a probability ratio, a log-probability gradient, and an advantage. The ratio corrects the conditional action distribution at a fixed state; the state weighting must be handled according to the objective and estimator being used.

The actor-critic expression uses behavior-policy data to update a target policy. TRPO uses data from this iteration's old policy and holds its advantage fixed while differentiating the probability ratio. Reweighting a newly collected on-policy batch during its update is consistent with TRPO's on-policy sampling loop.

## The practical TRPO update

Approximate the surrogate to first order and the average KL to second order around $\theta_k$. Define

$$
g=\left.\nabla_\theta\mathcal L(\theta_k,\theta)\right|_{\theta=\theta_k},
\qquad
H=\left.\nabla_\theta^2\bar D_{KL}(\theta\|\theta_k)\right|_{\theta=\theta_k}.
$$

$g$ is the surrogate gradient at the old parameters. $H$ is a Hessian, the matrix of second partial derivatives; it describes the local KL curvature. Both are held fixed in the local optimization problem:

$$
\begin{aligned}
\max_\theta\quad&g^T(\theta-\theta_k),\\
\text{s.t.}\quad&\frac12(\theta-\theta_k)^TH(\theta-\theta_k)\leq\delta.
\end{aligned}
$$

For $D(\theta)=\bar D_{KL}(\theta\|\theta_k)$, the Taylor expansion is

$$
D(\theta)\approx D(\theta_k)
+\nabla D(\theta_k)^T(\theta-\theta_k)
+\frac12(\theta-\theta_k)^TH(\theta-\theta_k).
$$

At identical policies, $D(\theta_k)=0$ and $\nabla D(\theta_k)=0$, so the expansion starts with the quadratic term.

Under the positive-definite local assumptions derived in Appendix C, the solution is

$$
\theta_{k+1}=\theta_k+
\sqrt{\frac{2\delta}{g^TH^{-1}g}}\,H^{-1}g.
$$

The direction is $H^{-1}g$; the square-root factor scales the displacement to the boundary of the quadratic KL budget.

### Fisher information and the natural policy gradient

Let the score be $\nabla_\theta\log\pi_\theta(a\mid s)$. The Fisher information matrix is

$$
F(\theta)=\mathbb E_{s\sim d_{\pi_\theta},\,a\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)^T\right].
$$

It measures the local sensitivity of the policy distribution to parameter changes. A direction that changes the distribution strongly has high Fisher curvature. With the old-state weighting used for the local KL and the usual regularity assumptions, $F(\theta_k)=H$.

The **natural policy gradient** is

$$
\widetilde\nabla_\theta J(\theta)=F(\theta)^{-1}\nabla_\theta J(\theta).
$$

It rescales the ordinary gradient using distributional sensitivity: take smaller parameter displacements along sensitive directions and larger ones along less sensitive directions. In the local TRPO problem, this direction is $H^{-1}g$.

### Backtracking and computation

Use a backtracking factor $\alpha\in(0,1)$ and choose the first nonnegative integer $j$ whose candidate improves the sampled surrogate and satisfies the measured KL constraint:

$$
\theta_{k+1}=\theta_k+\alpha^j
\sqrt{\frac{2\delta}{g^TH^{-1}g}}\,H^{-1}g.
$$

Explicitly forming $H^{-1}$ is expensive. Instead, solve $Hx=g$ for the required vector $x$ using conjugate gradients. Each iteration only needs the product $Hv$ for a supplied vector $v$.

In PyTorch, this **Hessian-vector product** can be obtained by two differentiations. First differentiate the KL with `create_graph=True`; then differentiate the inner product of that gradient with fixed $v$:

$$
Hv=\left.\nabla_\theta\left[
(\nabla_\theta\bar D_{KL}(\theta\|\theta_k))^Tv
\right]\right|_{\theta=\theta_k}.
$$

The full Hessian and its inverse are never required.

## Pseudocode

1. Initialize policy parameters $\theta_0$ and value-function parameters $\phi_0$. The networks are $\pi_\theta$ and $V_\phi$.
2. Choose KL limit $\delta$, backtracking factor $\alpha\in(0,1)$, and maximum backtracking index $K$.
3. For $k=0,1,2,\ldots$, collect trajectories $\mathcal D_k=\{\tau_1,\ldots,\tau_N\}$ with $\pi_{\theta_k}$. Here $|\mathcal D_k|$ counts trajectories. A hat denotes an estimate from this batch.

   > The pseudocode uses $\tau_i=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T)$, associating $r_t$ with action $a_t$. This shifts the reward subscript relative to the background equations.

4. Compute rewards-to-go:

   $$
   \hat R_t=\sum_{l=t}^{T}\gamma^{l-t}r_l.
   $$

5. Estimate advantages $\hat A_t$ using $V_{\phi_k}$, for example with generalized advantage estimation, GAE. GAE combines TD residuals across several horizons.
6. Estimate the policy gradient:

   $$
   \hat g_k=\frac1{|\mathcal D_k|}\sum_{\tau\in\mathcal D_k}\sum_{t=0}^{T}
   \left.\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right|_{\theta=\theta_k}\hat A_t.
   $$

7. Approximately solve $\hat H_k\hat x_k=\hat g_k$ using conjugate gradients. $\hat H_k$ is the Hessian of the sampled average KL at $\theta_k$.
8. Try candidates

   $$
   \theta_{\mathrm{trial}}=\theta_k+
   \alpha^j\sqrt{\frac{2\delta}{\hat x_k^T\hat H_k\hat x_k}}\,\hat x_k,
   \qquad j=0,\ldots,K.
   $$

   Accept the first candidate that increases the sampled surrogate and has sampled average KL at most $\delta$. Keep $\theta_k$ if all candidates fail.

9. Fit the value function by squared error:

   $$
   \phi_{k+1}=\arg\min_\phi\frac1{|\mathcal D_k|}
   \sum_{\tau\in\mathcal D_k}\sum_{t=0}^{T}
   (V_\phi(s_t)-\hat R_t)^2.
   $$

## Appendix A. Importance sampling

### A.1 Computing one distribution's expectation with another distribution's samples

Suppose we need $\mathbb E_{x\sim p}[f(x)]$ but have samples from $q$. Averaging $f(x)$ over those samples estimates $\mathbb E_q[f(x)]$. Importance sampling instead weights each contribution by $p(x)/q(x)$:

$$
\begin{aligned}
\mathbb E_{x\sim p}[f(x)]
&=\sum_xp(x)f(x)\\
&=\sum_{x:q(x)>0}q(x)\frac{p(x)}{q(x)}f(x)\\
&=\mathbb E_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right].
\end{aligned}
$$

The sampling distribution must have positive probability wherever the target distribution has positive probability. This is the **support coverage** condition. For continuous variables, replace sums with integrals and probabilities with densities.

**The expectation identity is exact.** Sampling error enters when a finite average is used:

$$
\widehat{\mathbb E_p[f]}=\frac1N\sum_{i=1}^{N}\frac{p(x_i)}{q(x_i)}f(x_i),
\qquad x_i\sim q.
$$

Locations more frequent under the target distribution receive weights above $1$; less frequent locations receive weights below $1$. The sampling distribution stays the same while each sample's contribution changes.

### A.2 Reweighting actions at a fixed state

Fix $s$ and set $q(a)=\pi_{\theta_k}(a\mid s)$, $p(a)=\pi_\theta(a\mid s)$, and $f(a)=A^{\pi_{\theta_k}}(s,a)$:

$$
\mathbb E_{a\sim\pi_\theta}[A^{\pi_{\theta_k}}(s,a)]
=\mathbb E_{a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a)\right].
$$

The question is: if the new policy chose actions at this state, how much better would they look under the old policy's fixed advantage scores?

| Action | Old probability | New probability | Old advantage | Probability ratio |
| --- | ---: | ---: | ---: | ---: |
| $a_1$ | $0.5$ | $0.8$ | $+1$ | $1.6$ |
| $a_2$ | $0.5$ | $0.2$ | $-1$ | $0.4$ |

Direct evaluation gives $0.8(1)+0.2(-1)=0.6$. Old-policy reweighting gives $0.5(1.6)(1)+0.5(0.4)(-1)=0.6$. The old-policy mean advantage itself is zero.

Large ratios amplify sample contributions and can increase estimation variance. This helps explain the purpose of limiting policy changes.

### A.3 Why a surrogate is still needed: the state distribution changes

Changing the policy also changes future state visits. For $0<\gamma<1$ and a fixed initial-state distribution, define the normalized discounted objective and occupancy distribution:

$$
\begin{aligned}
J(\theta)&=(1-\gamma)\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{\infty}\gamma^t r_t\right],\\
d_{\pi_\theta}(s)&=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr(s_t=s\mid\pi_\theta).
\end{aligned}
$$

The **performance difference identity** states

$$
J(\theta)-J(\theta_k)
=\mathbb E_{s\sim d_{\pi_\theta},\,a\sim\pi_\theta}
[A^{\pi_{\theta_k}}(s,a)].
$$

To see why, expand the old advantage for one transition:

$$
A^{\pi_{\theta_k}}(s,a)
=\mathbb E[r_t+\gamma V^{\pi_{\theta_k}}(s_{t+1})
-V^{\pi_{\theta_k}}(s_t)\mid s_t=s,a_t=a].
$$

Sum along new-policy trajectories with discounting. Adjacent value terms cancel:

$$
\begin{aligned}
\mathbb E_{\tau\sim\pi_\theta}\sum_{t=0}^{\infty}\gamma^t
A^{\pi_{\theta_k}}(s_t,a_t)
&=\mathbb E_{\tau\sim\pi_\theta}\sum_{t=0}^{\infty}\gamma^t r_t
-\mathbb E_{s_0}[V^{\pi_{\theta_k}}(s_0)]\\
&=\frac{J(\theta)-J(\theta_k)}{1-\gamma}.
\end{aligned}
$$

Multiplying by $1-\gamma$ gives the identity above. TRPO substitutes the old state distribution and then applies the exact action reweighting:

$$
\begin{aligned}
J(\theta)-J(\theta_k)
&\approx\mathbb E_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_\theta}
[A^{\pi_{\theta_k}}(s,a)]\\
&=\underbrace{\mathbb E_{s\sim d_{\pi_{\theta_k}},\,a\sim\pi_{\theta_k}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a)\right]}_{\mathcal L(\theta_k,\theta)}.
\end{aligned}
$$

The approximation is the replacement of the state distribution. The KL constraint keeps the new policy close enough for this local surrogate to be useful.

### A.4 Why the surrogate gradient matches the policy gradient at the old parameters

During this update, the old state distribution, old action distribution, and old advantage stay fixed. Differentiate the numerator using $\nabla\pi_\theta=\pi_\theta\nabla\log\pi_\theta$:

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

The ratio becomes $1$ at identical parameters. Subtracting a state baseline leaves the policy gradient unchanged because

$$
\mathbb E_{a\sim\pi_\theta}
[\nabla_\theta\log\pi_\theta(a\mid s)V^{\pi_{\theta_k}}(s)]
=V^{\pi_{\theta_k}}(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

Thus $Q$ can be replaced by advantage, yielding $g=\left.\nabla_\theta J\right|_{\theta_k}$ with the matching objective convention. Since the old-policy mean advantage is zero, $\mathcal L(\theta_k,\theta_k)=0$; the gradient describes its local increase away from that point.

## Appendix B. Backtracking line search

### B.1 A direction still needs a step length

For $\max_\theta f(\theta)$, let $p$ be an ascent direction and let $t>0$ be a step length:

$$
\theta_{\mathrm{trial}}=\theta_k+t p.
$$

$\nabla f(\theta_k)^Tp>0$ predicts improvement for sufficiently small positive $t$. Farther away, curvature can invalidate the local prediction.

Backtracking tries $t=1,\alpha,\alpha^2,\ldots$, with $\alpha\in(0,1)$. For $\alpha=0.5$, try a full step, a half step, then a quarter step. The direction stays fixed.

A general **Armijo sufficient-increase condition** is

$$
f(\theta_k+t p)\geq f(\theta_k)+c\,t\,\nabla f(\theta_k)^Tp,
\qquad 0<c<1.
$$

The right-hand increment is a fraction of the predicted first-order improvement. A minimization problem uses the corresponding sufficient-decrease condition.

### B.2 The two checks in TRPO

The full candidate step from the local model is

$$
p=\sqrt{\frac{2\delta}{g^TH^{-1}g}}H^{-1}g,
\qquad \frac12p^THp=\delta.
$$

This equality uses the quadratic KL approximation. The actual neural policy can violate the KL budget or fail to realize the predicted surrogate increase.

For $t_j=\alpha^j$, recompute on the same old-policy batch:

$$
\begin{aligned}
\theta_{\mathrm{trial}}&=\theta_k+t_jp,\\
\Delta\hat{\mathcal L}_j
&=\hat{\mathcal L}(\theta_k,\theta_{\mathrm{trial}})
-\hat{\mathcal L}(\theta_k,\theta_k),\\
\hat D_j&=\widehat{\bar D}_{KL}(\theta_{\mathrm{trial}}\|\theta_k).
\end{aligned}
$$

Accept the first candidate with $\Delta\hat{\mathcal L}_j>0$ and $\hat D_j\leq\delta$.

For example, take $\delta=0.01$ and $\alpha=0.5$. A full step with measured KL $0.016$ is rejected even if it improves the surrogate. A half step with KL $0.004$ and positive surrogate improvement is accepted.

Every trial starts from the same $\theta_k$. Keep the old policy, old log probabilities, and estimated advantages fixed. If all trials fail, restore $\theta_k$.

## Appendix C. Lagrangian duality

### C.1 The function and the multiplier

A **Lagrangian function** combines an objective with its constraints. A **Lagrange multiplier** sets the weight of a constraint. Use $\mathscr L$ for the Lagrangian to distinguish it from the surrogate $\mathcal L$.

For a maximization problem with variable $z$,

$$
\max_z f(z),\qquad c(z)\leq0,
$$

define

$$
\mathscr L(z,\lambda)=f(z)-\lambda c(z),\qquad\lambda\geq0.
$$

The multiplier can be viewed as a price for violating the constraint: when $c(z)>0$, the term $-\lambda c(z)$ penalizes that violation.

### C.2 Primal and dual problems

For any feasible $z$, $c(z)\leq0$ and $\lambda\geq0$ imply $f(z)\leq\mathscr L(z,\lambda)$. Therefore

$$
q(\lambda)=\sup_z\mathscr L(z,\lambda),
\qquad
\underbrace{\sup_{z:c(z)\leq0}f(z)}_{\text{primal optimum}}
\leq q(\lambda).
$$

Every multiplier gives an upper bound on the primal optimum. The **dual problem** seeks the tightest such bound:

$$
\min_{\lambda\geq0}q(\lambda).
$$

The bound is called **weak duality**. Equality between the best dual bound and the primal optimum is **strong duality**. The local TRPO problem below has the conditions needed for strong duality.

### C.3 Solving the local TRPO problem

Let $\Delta\theta=\theta-\theta_k$. Consider

$$
\max_{\Delta\theta}g^T\Delta\theta,
\qquad \frac12\Delta\theta^TH\Delta\theta\leq\delta.
$$

Assume $H\succ0$, $g\neq0$, and $\delta>0$. Positive definiteness means $v^THv>0$ for every nonzero $v$. The feasible region is an ellipsoid. The origin is strictly feasible, satisfying Slater's condition, so strong duality applies to this local convex optimization problem.

**Step 1: form the Lagrangian.**

$$
\mathscr L(\Delta\theta,\lambda)
=g^T\Delta\theta-\lambda\left(\frac12\Delta\theta^TH\Delta\theta-\delta\right).
$$

**Step 2: maximize over the displacement for fixed $\lambda>0$.** The Hessian is $-\lambda H\prec0$, so the stationary point is the unique maximizer:

$$
g-\lambda H\Delta\theta=0
\quad\Longrightarrow\quad
\Delta\theta(\lambda)=\frac1\lambda H^{-1}g.
$$

**Step 3: substitute into the dual function.** Write $C=g^TH^{-1}g>0$:

$$
q(\lambda)=\frac C\lambda-\frac{C}{2\lambda}+\lambda\delta
=\frac{C}{2\lambda}+\lambda\delta.
$$

**Step 4: minimize over the multiplier.**

$$
q'(\lambda)=-\frac{C}{2\lambda^2}+\delta=0
\quad\Longrightarrow\quad
\lambda^*=\sqrt{\frac{C}{2\delta}}.
$$

Since $q''(\lambda)=C/\lambda^3>0$, this is the unique minimum. Consequently

$$
\boxed{\Delta\theta^*=\sqrt{\frac{2\delta}{g^TH^{-1}g}}H^{-1}g,
\qquad\theta_{k+1}=\theta_k+\Delta\theta^*.}
$$

Stationarity determines the direction; the KL budget determines the scale. Substitution verifies $\frac12(\Delta\theta^*)^TH\Delta\theta^*=\delta$.

### C.4 KKT conditions

The **Karush-Kuhn-Tucker conditions** combine the requirements used above:

$$
\begin{aligned}
g-\lambda H\Delta\theta&=0 &&\text{stationarity},\\
\frac12\Delta\theta^TH\Delta\theta-\delta&\leq0 &&\text{primal feasibility},\\
\lambda&\geq0 &&\text{dual feasibility},\\
\lambda\left(\frac12\Delta\theta^TH\Delta\theta-\delta\right)&=0
&&\text{complementary slackness}.
\end{aligned}
$$

Complementary slackness says a constraint with unused budget has zero multiplier; a positive multiplier requires a tight constraint. Here $g\neq0$ forces $\lambda^*>0$, so the optimum uses the full local KL budget.

Appendix D computes the direction, and Appendix B selects the displacement that is actually accepted.

## Appendix D. Conjugate gradients

### D.1 Solving a linear system by minimizing a quadratic

For a symmetric positive-definite matrix $B$, define

$$
\psi(x)=\frac12x^TBx-g^Tx.
$$

Its gradient is $Bx-g$, and its Hessian is $B\succ0$. Its unique minimizer therefore solves $Bx=g$. In TRPO, $x$ supplies a direction, which is subsequently scaled to the KL budget.

Ordinary gradient descent can zigzag across elongated quadratic contours. **Conjugate gradients**, CG, constructs directions that preserve progress along earlier directions.

### D.2 Conjugacy: orthogonality measured by curvature

Directions $p_i,p_j$ are $B$-conjugate when

$$
p_i^TBp_j=0,\qquad i\neq j.
$$

For $B=I$, this becomes Euclidean orthogonality. After an exact line search along $p_i$, $p_i^T\nabla\psi(x)=0$. A later displacement along conjugate $p_j$ preserves this condition:

$$
p_i^T\nabla\psi(x+t p_j)
=p_i^T\nabla\psi(x)+t\,p_i^TBp_j=0.
$$

### D.3 From residuals to the iteration

The **residual** $r_i=g-Bx_i$ measures the remaining mismatch in the linear system and equals $-\nabla\psi(x_i)$. Start with $x_0=0$, $r_0=g$, and $p_0=r_0$. If $g=0$, the initial point already solves the system.

Exact minimization along $p_i$ gives

$$
\frac d{d\eta}\psi(x_i+\eta p_i)
=-p_i^Tr_i+\eta p_i^TBp_i=0.
$$

Thus $\eta_i=p_i^Tr_i/(p_i^TBp_i)$. In CG, residuals are orthogonal to earlier search directions, so $p_i^Tr_i=r_i^Tr_i$. The iteration becomes

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

Adding a multiple of the previous direction makes the new direction conjugate. Specifically, $p_{i+1}^TBp_i=0$ yields

$$
\beta_i=-\frac{r_{i+1}^TBp_i}{p_i^TBp_i}.
$$

In exact arithmetic, successive residuals are orthogonal and $Bp_i=(r_i-r_{i+1})/\eta_i$. Combining these identities with $\eta_i p_i^TBp_i=r_i^Tr_i$ gives the squared-residual formula for $\beta_i$. The orthogonality relations extend inductively to all generated directions.

Stop when the residual is sufficiently small or the iteration limit is reached. An $n$-dimensional positive-definite system is solved in at most $n$ iterations in exact arithmetic; neural networks usually use a much smaller number of approximate CG iterations.

$\eta_i$ updates the internal linear-system iterate $x_i$. The factor $\alpha^j$ in backtracking scales the final policy displacement; these step lengths belong to different calculations.

### D.4 Damping, Hessian-vector products, and final scaling

The ideal derivation uses $B=H$. In practice the KL Hessian or Fisher matrix can be positive semidefinite, with zero-curvature directions. **Damping** adds a positive multiple of the identity:

$$
B=H+\zeta I,\qquad\zeta>0,
\qquad Bv=Hv+\zeta v.
$$

For nonzero $v$, $v^TBv=v^THv+\zeta\|v\|^2>0$ when $H\succeq0$. CG then solves $(H+\zeta I)x=g$ for a regularized direction.

CG only needs a function mapping an input vector to $Bp_i$. Compute $Hp_i$ by automatic differentiation and add $\zeta p_i$. Keep the old policy fixed, differentiate the new-parameter branch, and evaluate the HVP at $\theta_k$.

Without damping, scale the returned direction using

$$
p=\sqrt{\frac{2\delta}{x^THx}}x.
$$

An exact solve gives $x^THx=g^TH^{-1}g$. With an approximate solve, directly measuring $x^THx$ evaluates the curvature along the actual returned direction.

A common damped version uses the same $B$ for scaling:

$$
p=\sqrt{\frac{2\delta}{x^TBx}}x,
\qquad x\approx B^{-1}g.
$$

Then $\frac12p^THp\leq\frac12p^TBp=\delta$, leaving some room within the local KL budget. Finally apply backtracking.

The complete computation is **solve for direction $x$ with CG, scale it by KL curvature to obtain candidate displacement $p$, then test $\theta_k+\alpha^jp$ with backtracking**.
