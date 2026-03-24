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

Given a batch of data, standard SGD computes the average gradient:

$$\delta\theta = -\eta \frac{1}{B}\sum_\alpha \nabla_\theta \ell_\alpha(\theta)$$

Sven instead forms the Jacobian matrix $M_{\alpha i} = \partial \ell_\alpha / \partial \theta_i$ and computes:

$$\delta\theta = -\eta \, M^+ \, \boldsymbol{\ell}$$

where $M^+$ is the Moore-Penrose pseudoinverse computed via truncated SVD, and $\boldsymbol{\ell}$ is the vector of per-sample losses. In the over-parameterized regime ($P > B$), this is the minimum-norm update that best satisfies all per-sample loss conditions simultaneously.
