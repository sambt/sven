# Sven: Singular Value Descent

Sven is a PyTorch optimizer that replaces standard gradient descent with parameter updates computed via the Moore-Penrose pseudoinverse of the per-sample Jacobian matrix. Where SGD computes a single gradient by averaging over the batch, Sven decomposes the loss into individual per-sample components and solves for the minimum-norm parameter update that simultaneously reduces all of them, using a truncated SVD to keep the computation tractable.

In the over-parameterized regime this yields the minimum-norm solution among all updates that minimize the L2 error across the batch, and under favorable conditions can achieve exponential loss decay rather than the power-law behavior typical of first-order methods.

## Installation

```bash
pip install -e .
```

## Quick start

Sven is a near drop-in replacement for a standard PyTorch optimizer, with two differences: (1) the model must be wrapped with `SvenWrapper`, which converts it to a functional form for per-sample Jacobian computation, and (2) the training step calls `loss_and_grad` instead of the usual `loss.backward()`.

```python
import torch
import torch.nn as nn
from sven.nn import SvenWrapper
from sven.opt import Sven

# Define any standard PyTorch model and a per-sample loss function
model = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 1))
loss_fn = lambda pred, y: ((pred - y) ** 2).sum(dim=-1)  # must return shape (B,)

# Wrap the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped = SvenWrapper(model, loss_fn, device)

# Create the optimizer
optimizer = Sven(wrapped, lr=0.1, k=64, rtol=1e-3)

# Training step
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    losses, preds = wrapped.loss_and_grad((xb, yb))
    optimizer.step()
```

See `examples/toy_1d_regression.ipynb` for a complete worked example comparing Sven to Adam.

## Key concepts

### Per-sample loss function

The loss function passed to `SvenWrapper` must return **per-sample** losses with shape `(B,)`, not a scalar. This is because Sven needs the individual loss components to construct the Jacobian matrix.

```python
# Correct: returns (B,) tensor
loss_fn = lambda pred, y: ((pred - y) ** 2).sum(dim=-1)

# Wrong: returns scalar
loss_fn = nn.MSELoss()
```

### Hyperparameters

- **`k`**: Number of singular values to keep in the truncated SVD. Controls the rank of the pseudoinverse approximation. A good starting point is `batch_size // 2`.
- **`rtol`**: Relative tolerance for singular value truncation. Singular values smaller than `rtol * sigma_max` are discarded. Default `1e-3`.
- **`lr`**: Learning rate applied to the pseudoinverse update.
- **`svd_mode`**: Algorithm for computing the truncated SVD. Options: `"torch"` (full SVD then truncate), `"randomized"` (randomized SVD), `"scipy"`, `"lobpcg"`. Default `"torch"`.

### Memory management

The per-sample Jacobian has shape `(B, P)` where `B` is batch size and `P` is the number of parameters, so memory scales as `O(B * P)`. Two options help manage this:

- **`param_fraction`**: Compute the Jacobian with respect to a random subset of parameters each step. Set to e.g. `0.5` to halve memory usage.
- **`microbatch_size`**: Aggregate losses within sub-batches before computing the Jacobian, reducing the effective batch dimension.

## Package structure

```
sven/
├── nn/
│   ├── sven_wrapper.py   # SvenWrapper: functional model wrapper + Jacobian computation
│   └── __init__.py
└── opt/
    ├── sven.py           # Sven optimizer
    ├── pinv.py           # Truncated SVD pseudoinverse implementations
    ├── polyak.py         # PolyakSGD baseline optimizer
    └── __init__.py
```

## How it works

Given a batch $\mathcal{B}$ of data, standard SGD computes the average gradient:

$$\delta\theta = -\eta \frac{1}{B}\sum_{x_\alpha \in \mathcal{B}} \nabla_\theta \ell(x_\alpha;\theta)$$

Sven instead treats each element's contribution to the loss separately. Inspired by the $L_2$ loss, which can be written as a sum of squared residuals, we can express the total loss as

$$L = \sum_{x_\alpha \in \mathcal{B}} \left((\ell(x_\alpha;\theta))^{\kappa/2}\right)^{2/\kappa}$$

where $\kappa > 0$ is a hyperparameter. For a regression-style loss, $L = \sum_\alpha (\mathcal{R}^\alpha)^2$ with $\mathcal{R}^\alpha = f_\theta(x_\alpha) - f(x_\alpha)$. For a generic loss $\ell(x_\alpha;\theta) \equiv \ell^\alpha(\theta)$ (e.g. cross-entropy), $\kappa = 1$ defines a case in which we can view $\sqrt{\ell^\alpha(\theta)}$ as the residuals of an $L_2$-style loss.

In the $L_2$ setting, we can derive a generalizable update rule by considering a first-order linear expansion of our loss in terms of network parameters:

$$L(\theta_0 +\delta\theta) = \sum_{\alpha}\left(\mathcal R^\alpha(\theta_0) + \sum_i M^\alpha_{i} \delta\theta^i\right) ^2+\mathcal{O}\left(|\delta\theta|^2\right)$$

with the Jacobian matrix defined as 

$$M^\alpha_{i} \equiv \left.\frac{\partial \mathcal{R}^\alpha}{\partial \theta^i}\right|_{\theta = \theta_0}.$$

We seek solutions that drive each term of the loss to zero (or as close to zero as it can get in the linear approximation):

$$\mathcal R^\alpha(\theta_0) + \sum_i M^\alpha_{i} \delta\theta^i = 0$$

An exact solution rarely exists, but the closest approximation to one is given by 

$$\delta \theta^i = -(M^+)^i_{\alpha} \mathcal R^{\alpha}(\theta_0)$$

where $M^+$ is the Moore-Penrose pseudoinverse of $M$.

For a generic loss function as written above with $\kappa > 0$, the Sven update rule can be written as

$$\boxed{
\delta \theta^i = - \eta (M^+)^i_{\alpha} \mathcal R_\mathrm{eff}^\alpha(\theta_0), \qquad M^\alpha_{i} \equiv \left.\frac{\partial \mathcal{R}_\mathrm{eff}^\alpha}{\partial \theta^i}\right|_{\theta = \theta_0},}$$

where $\eta$ is a learning rate hyperparameter and $\mathcal{R}_\mathrm{eff}^\alpha = (\ell^\alpha(\theta_0))^{\kappa/2}$.

In practice, while $\kappa = 1$ keeps us in the familiar $L_2$ setting, using $\kappa = 2$ with $\mathcal{R}_\mathrm{eff}^\alpha = \ell^\alpha$ avoids pathologies associated with taking fractional powers of generic loss functions such as cross-entropy. 
