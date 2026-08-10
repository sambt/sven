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

## The Gram-trick optimizer (recommended)

`GramSvenWrapper` + `SvenGram` compute the **exact same update** as `SvenWrapper` + `Sven` without ever materializing the `(B, P)` Jacobian. Instead of a truncated SVD of `J`, they eigendecompose the tiny `B×B` Gram matrix `G = JJᵀ` and recover the update as one weighted backward pass, `δθ = Jᵀw`. The whole step costs roughly two ordinary forward+backward passes: at `P ≈ 1M`, `B = 128` this measures **~400× faster and ~50× less peak memory** than the classic pipeline (5.8 ms / +41 MB vs ~2.4 s / ~+2 GB per step), putting Sven at Adam-class cost. Usage is identical apart from the class names:

```python
from sven.nn import GramSvenWrapper
from sven.opt import SvenGram

wrapped = GramSvenWrapper(model, loss_fn, device)   # capture="hooks" (default)
optimizer = SvenGram(wrapped, lr=0.1, k=64, rtol=1e-3)

for xb, yb in train_loader:
    losses, preds = wrapped.loss_and_grad((xb.to(device), yb.to(device)))
    optimizer.step()
```

Two capture modes build `G`:

- **`capture="hooks"`** (default, fastest): one forward + one backward with per-layer captures. Supports `Linear`, `groups=1` `Conv2d`, and frozen-stats normalization layers; architectures that couple samples across the batch (train-mode batch-stats BatchNorm, dropout, tied weights, …) raise a clear error rather than silently producing a wrong `G`.
- **`capture="chunked"`**: accumulates `G` from per-parameter-group Jacobian blocks via real autodiff — exact for **any** architecture, memory bounded by `chunk_numel`.

Because `cond(G) = cond(J)²`, `G` is accumulated in float64 (`gram_dtype`); this is required for tight `rtol` (≤ ~1e-6) and costs nothing (`G` is only `B×B`). `variable_k` and pre-pseudoinverse RMSProp need the Jacobian itself and are only available with the classic `Sven`.

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
- **`svd_mode`**: Algorithm for computing the truncated SVD. Options: `"torch"` (full SVD then truncate), `"randomized"` (randomized SVD), `"randomized_v2"` (different randomized strategy using eigendecomposition), `"scipy"`, `"lobpcg"`. Default `"torch"`.

### Memory management

The per-sample Jacobian has shape `(B, P)` where `B` is batch size and `P` is the number of parameters, so the classic pipeline's memory scales as `O(B * P)`. The **Gram-trick optimizer above avoids this entirely** and is the recommended fix. Within the Jacobian pipeline itself:

- **`jac_chunk_size`**: bounds the peak memory of the `jacrev` computation by chunking its vmap (a ~3× reduction for one argument, at no accuracy cost).
- **`microbatch_size`**: aggregate losses within sub-batches before computing the Jacobian, reducing the effective row dimension.

### Stochastic parameter updates (`param_fraction` + `mask_mode`)

`param_fraction < 1` updates a random subset of parameters each step. Empirically this barely degrades optimization at fractions ≥ 25 %, and acts as a mild regularizer. The `mask_mode` matters a great deal for **cost**, however:

- **`"elementwise"`** (the legacy default): exact i.i.d. fractions, but **no Jacobian memory savings** — reverse-mode AD materializes the full `(B, P)` cotangent before the masked gather; only the SVD input shrinks.
- **`"tensor"`**: selects whole parameter tensors and differentiates only those — genuine `(B, n_active)` memory, but coarse (a single large layer can dominate the parameter count) and prone to leaving parameters permanently un-updated.
- **`"rows"`** (recommended for the Jacobian pipeline): per-layer output-neuron masking via split active/frozen matmuls — genuine memory *and* compute scaling with the fraction, near-exact fractions, full parameter coverage across steps.
- **Masked Gram** (recommended overall): `GramSvenWrapper(..., param_fraction=f, mask_mode="rows")` + `SvenGram` computes the identical masked update at **~constant memory and milliseconds per step for any fraction** — parameter-fraction scans run at Gram cost. With `capture="chunked"` the same masking works on **any architecture** (transformers included), with leading-axis-slice row semantics and group-bounded memory; see `comparisons/transformer/` for the benchmark harness.

See `reports/2026-07-24_gram_and_stochastic_masking.md` for the measurements behind these recommendations and `comparisons/` for the benchmark harness.

## Package structure

```
sven/
├── nn/
│   ├── sven_wrapper.py   # SvenWrapper: functional model wrapper + Jacobian computation
│   ├── gram_wrapper.py   # GramSvenWrapper: G = JJ^T without materializing J
│   ├── masked_modules.py # Split-matmul Linear/Conv2d twins for mask_mode="rows"
│   └── __init__.py
├── opt/
│   ├── sven.py           # Sven and SvenGram optimizers
│   ├── pinv.py           # Truncated SVD pseudoinverse implementations
│   ├── polyak.py         # PolyakSGD baseline optimizer
│   └── __init__.py
└── jax/                  # JAX mirror: SvenWrapper, GramSvenWrapper, Sven, SvenGram, pinv
    ├── wrapper.py
    ├── gram.py
    ├── sven.py
    └── pinv.py

tests/                    # exactness/guard test suite (pytest, 88 tests)
comparisons/              # optimizer benchmark harness (CIFAR-10 MLP study)
reports/                  # detailed write-up of the Gram trick + masking work
```

A JAX mirror of the full API lives in `sven.jax` (same class names; models are supplied as an `apply_fn` + params pytree). Note the JAX `mask_mode="rows"` masks leading-axis slices per leaf rather than torch's neuron-tied rows, and its masked-Gram memory is group-bounded rather than fraction-constant — see the docstrings.

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

### The Gram trick

The update above never requires $M$ itself. Writing $M = USV^\top$ and $G \equiv MM^\top = US^2U^\top$ — a small $B \times B$ matrix — the truncated pseudoinverse update factors as

$$\delta\theta = -\eta\, V_k S_k^{-1} U_k^\top \mathcal{R} = -\eta\, M^\top \underbrace{U_k S_k^{-2} U_k^\top \mathcal{R}}_{w \,\in\, \mathbb{R}^B},$$

and $M^\top w = \nabla_\theta \sum_\alpha w_\alpha \mathcal{R}^\alpha$ is a single ordinary backward pass. $G$ itself is assembled layer-by-layer from quantities already present in one forward+backward (for a linear layer, $G_l = (g_l g_l^\top) \odot (x_l x_l^\top)$ — the per-example-gradient algebra of Goodfellow 2015, as used in ghost clipping and empirical-NTK computation). The eigendecomposition of $G$ replaces the SVD of $M$ exactly, including the $k$/`rtol` truncation. This is the same sample-space formulation used by MinSR and SPRING for neural quantum states. `SvenGram` implements it; see `reports/` for derivation, measurements, and verification.
