# MLP: Multilayer Perceptrons

[Original notes](mlp.md)

## Network output

A multilayer perceptron combines weighted sums with activation functions. In the network used in the notes:

- $D$ is the number of input features; $i=1,\ldots,D$ indexes them.
- $J$ is the number of hidden units; $j=1,\ldots,J$ indexes them.
- $K$ is the number of output units; $k=1,\ldots,K$ indexes them.
- $v_{ji}$ connects input $i$ to hidden unit $j$, with bias $v_{j0}$.
- $w_{kj}$ connects hidden unit $j$ to output $k$, with bias $w_{k0}$.
- $f$ and $g$ are the hidden and output activation functions.

The network computes

$$
\begin{aligned}
h_j(x)&=f\left(v_{j0}+\sum_{i=1}^{D}x_i v_{ji}\right),\\
o_k(x)&=g\left(w_{k0}+\sum_{j=1}^{J}h_j(x)w_{kj}\right).
\end{aligned}
$$

The diagram's signal flow is input features, hidden activations, then output activations. A weight belongs to a connection; a bias belongs to a receiving unit.

## Example: forward propagation

The illustrated network has 3 input features, two hidden layers of 4 units each, and one output. Let $W_\ell$ and $b_\ell$ denote the weight matrix and bias vector of layer $\ell$; the superscript on $h^1,h^2$ identifies a layer.

$$
\begin{aligned}
h^1&=f(W_1x+b_1),\\
h^2&=f(W_2h^1+b_2),\\
o&=g(W_3h^2+b_3).
\end{aligned}
$$

Here $W_1\in\mathbb R^{4\times3}$, $W_2\in\mathbb R^{4\times4}$, and $W_3\in\mathbb R^{1\times4}$. Bias vectors have shapes $4\times1$, $4\times1$, and $1\times1$.

![Three input units, two four-unit hidden layers, and one output unit](image.png)

$W$ contains all weights in a layer. The row identifies the receiving unit and the column identifies the sending unit. Thus $W_1(2,3)$ connects input feature 3 to hidden unit 2.

The source gives this example with sigmoid hidden layers and a linear output:

```python
# W1, W2, W3 and the biases are assumed to be initialized.
f = lambda x: 1.0 / (1.0 + np.exp(-x))
x = np.random.randn(3, 1)
h1 = f(np.dot(W1, x) + b1)
h2 = f(np.dot(W2, h1) + b2)
out = np.dot(W3, h2) + b3
```

## Backpropagation: gradients of the loss with respect to weights and biases

Training repeats three operations until convergence:

1. Given example $n$, propagate its input forward to obtain predictions.
2. Propagate loss derivatives backward through the network.
3. Update weights and biases using those derivatives.

Given a loss $E$ and differentiable activation functions, the main task is to derive their gradients.

Hidden units have no individually supplied target labels. Instead, compute how the loss changes when a hidden activation changes. One hidden unit can affect several outputs, so those effects must be summed. Once derivatives with respect to hidden activations are known, derivatives with respect to incoming weights follow from the chain rule.

### A one-layer example

The source diagram connects every input directly to every output, with no hidden layer. Its notation is:

![A one-layer network: inputs connect to weighted sums, then activation functions produce outputs](image-2.png)

- $x_i$: value of input unit $i$.
- $w_{ki}$: weight from input $i$ to output $k$.
- $w_{k0}$: output bias.
- $z_k$: weighted input before activation.
- $o_k$: output after applying activation $g$.

$$
z_k=w_{k0}+\sum_{i=1}^{D}w_{ki}x_i,
\qquad o_k=g(z_k).
$$

The special case illustrated in the slides uses the sigmoid:

$$
g(z)=\frac1{1+\exp(-z)}.
$$

Let $E$ be the loss for one example. Distinguish the derivatives with respect to the output and the weighted input:

$$
\delta_k^o\equiv\frac{\partial E}{\partial o_k},
\qquad
\delta_k^z\equiv\frac{\partial E}{\partial z_k}
=\delta_k^o g'(z_k).
$$

The value of $\delta_k^o$ depends on the loss function. When differentiating with respect to $w_{ki}$, keep the input and target fixed. The dependency is

$$
w_{ki}\longrightarrow z_k\longrightarrow o_k\longrightarrow E.
$$

Applying the chain rule gives

$$
\begin{aligned}
\frac{\partial E}{\partial w_{ki}}
&=\underbrace{\frac{\partial E}{\partial o_k}}_{\delta_k^o}
\frac{\partial o_k}{\partial z_k}
\frac{\partial z_k}{\partial w_{ki}}\\
&=\underbrace{\frac{\partial E}{\partial z_k}}_{\delta_k^z}
\frac{\partial z_k}{\partial w_{ki}}
=\delta_k^z x_i.
\end{aligned}
$$

Since $\partial z_k/\partial w_{ki}=x_i$, the weight gradient is the weighted-input loss gradient times the incoming activation. A bias has $\partial z_k/\partial w_{k0}=1$, so its gradient is $\delta_k^z$.

### Worked example: squared error with logistic activation

Let superscript $(n)$ index a training example and let $t_k^{(n)}$ be its target at output $k$. The source uses the following per-example squared-error loss:

$$
E^{(n)}=\sum_k\frac12\left(o_k^{(n)}-t_k^{(n)}\right)^2,
\qquad g(z)=\frac1{1+\exp(-z)}.
$$

The slide's forward and backward calculations are

$$
\begin{aligned}
\text{FP-1:}\quad z_k^{(n)}
&=w_{k0}+\sum_{i=1}^{D}w_{ki}x_i^{(n)},\\
\text{FP-2:}\quad o_k^{(n)}&=g(z_k^{(n)}),\\[4pt]
\text{BP-1:}\quad
\delta_k^o=\frac{\partial E^{(n)}}{\partial o_k^{(n)}}
&=o_k^{(n)}-t_k^{(n)},\\
\text{BP-2:}\quad
\delta_k^z=\frac{\partial E^{(n)}}{\partial z_k^{(n)}}
&=\delta_k^o g'(z_k^{(n)})\\
&=\delta_k^o\underbrace{g(z_k^{(n)})(1-g(z_k^{(n)}))}_{\text{sigmoid derivative}}\\
&=\delta_k^o o_k^{(n)}(1-o_k^{(n)}),\\
\text{BP-3:}\quad
\frac{\partial E^{(n)}}{\partial w_{ki}}
&=\delta_k^z x_i^{(n)}.
\end{aligned}
$$

The Chinese annotations emphasize differentiating for each output unit $k$ and using the sigmoid derivative. These two forward steps and three backward steps are the basis of the matrix formulas below.

### Matrix form

Suppose a layer has $D$ inputs and $K$ outputs. Let $W=[w_{ki}]\in\mathbb R^{K\times D}$; $\mathbf x\in\mathbb R^D$, $\mathbf b,\mathbf z,\mathbf o\in\mathbb R^K$.

**Forward propagation**

$$
\mathbf z=W\mathbf x+\mathbf b,
\qquad \mathbf o=g(\mathbf z).
$$

Matrix multiplication already performs the sum over inputs: $z_k=\sum_iw_{ki}x_i+b_k$.

**Backward propagation for sigmoid and squared error**

Here $\odot$ is elementwise multiplication; $\mathbf1$ is a vector of ones.

$$
\boldsymbol\delta^o=\mathbf o-\mathbf t,
\qquad
\boldsymbol\delta^z=\boldsymbol\delta^o\odot\mathbf o\odot(\mathbf1-\mathbf o).
$$

**Weight and bias gradients**

The derivative with respect to a weight supplies an input value; the derivative with respect to its bias supplies $1$:

$$
\frac{\partial E}{\partial W}
=\boldsymbol\delta^z\mathbf x^T
=[\delta_k^z x_i]_{k,i},
\qquad
\frac{\partial E}{\partial\mathbf b}=\boldsymbol\delta^z.
$$

For a multilayer network, let $\ell$ index layers and let $\mathbf a^{(0)}=\mathbf x$. The same calculation is repeated at each layer:

$$
\mathbf z^{(\ell)}=W^{(\ell)}\mathbf a^{(\ell-1)}+\mathbf b^{(\ell)},
\qquad
\mathbf a^{(\ell)}=g^{(\ell)}(\mathbf z^{(\ell)}).
$$

During backpropagation, a hidden unit collects the effects of all units it influences in the next layer:

$$
\boldsymbol\delta^{z,(\ell)}
=\underbrace{(W^{(\ell+1)})^T\boldsymbol\delta^{z,(\ell+1)}}_{\text{sum of effects from the next layer}}
\odot(g^{(\ell)})'(\mathbf z^{(\ell)}).
$$

The two-layer diagrams illustrate this recursion. For hidden unit $h_j$, its loss derivative is the sum of contributions from all connected output units. The transposed weight matrix performs this weighted sum for the whole layer at once.

## Gradient updates

After backpropagation, move parameters opposite to the loss gradient.

First distinguish the symbols:

- $w_{ki}$ is one scalar weight; $W=[w_{ki}]$ is an entire layer's matrix.
- $\eta>0$ is the learning rate; $\leftarrow$ means assignment.
- $N$ is the number of training examples; superscript $(n)$ indexes an example.
- $E^{(n)}$ is a per-example loss. Following the source convention, $E=\sum_{n=1}^{N}E^{(n)}$ is the summed loss.
- $\partial E/\partial w_{ki}$ is scalar. The matrix $\partial E/\partial W$ contains these partial derivatives and has the same shape as $W$.

### Scalar form

**Batch gradient descent** accumulates gradients over the whole dataset before updating:

$$
w_{ki}\leftarrow w_{ki}-\eta\frac{\partial E}{\partial w_{ki}}
=w_{ki}-\eta\sum_{n=1}^{N}\frac{\partial E^{(n)}}{\partial w_{ki}}.
$$

All terms are evaluated at the same current parameter values. Parameters stay fixed while the sum is accumulated.

**Stochastic gradient descent, SGD**, updates immediately using one example:

$$
w_{ki}\leftarrow w_{ki}-\eta\frac{\partial E^{(n)}}{\partial w_{ki}}.
$$

Examples may be sampled randomly or shuffled each epoch. Different examples can produce different gradient directions, so individual updates fluctuate.

**Mini-batch gradient descent** combines a small group of examples. Let $\mathcal B$ be the batch's index set and $B=|\mathcal B|$ its size. Using an average loss gives

$$
w_{ki}\leftarrow w_{ki}
-\eta\frac1B\sum_{n\in\mathcal B}\frac{\partial E^{(n)}}{\partial w_{ki}}.
$$

### Matrix form: update an entire layer

$$
W\leftarrow W-\eta\frac{\partial E}{\partial W}
=W-\eta\sum_{n=1}^{N}\frac{\partial E^{(n)}}{\partial W}.
$$

Substituting the per-example gradients yields

$$
\begin{aligned}
W&\leftarrow W-\eta\sum_{n=1}^{N}
\boldsymbol\delta^{z,(n)}(\mathbf x^{(n)})^T,\\
\mathbf b&\leftarrow\mathbf b-\eta\sum_{n=1}^{N}\boldsymbol\delta^{z,(n)}.
\end{aligned}
$$

The update slide summarizes the same operation: compute the loss derivatives, then subtract the learning-rate-scaled derivatives from the corresponding weights.

## Adding momentum

Momentum retains a history of gradient contributions. For one weight, let $v$ be an accumulator initialized to zero, $\eta$ the learning rate, and $\gamma\in[0,1)$ the fraction of history retained.

Update the accumulator first, then the weight:

$$
v\leftarrow\underbrace{\gamma v}_{\text{retain history}}
+\underbrace{\eta\frac{\partial E}{\partial w_{ki}}}_{\text{add the current gradient}},
\qquad w_{ki}\leftarrow w_{ki}-v.
$$

Consistently aligned gradients accumulate and accelerate movement. Alternating gradients partly cancel, reducing oscillations. Setting $\gamma=0$ recovers ordinary gradient descent.

The original momentum slide places the weight equation above the accumulator equation. The implementation order is to update $v$ first and then use its new value to update the weight. The notes also describe momentum as potentially helping movement through some local optimization difficulties.
