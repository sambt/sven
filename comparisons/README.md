# Optimizer comparisons

Head-to-head comparison of Sven variants against standard PyTorch optimizers on
a ~1M-parameter MLP (3072 → 300 → 300 → 10) trained on a class-stratified
4,096-sample subset of CIFAR-10 (flattened, per-feature standardized; full 10k
test set for validation). CIFAR-10 at this train-set size is deliberately in
Sven's overparametrized regime (P ≈ 250 × N) while remaining hard enough for an
MLP (≈45–50% val accuracy ceiling) that validation curves stay informative —
unlike MNIST, which every optimizer saturates.

## Configs

| name | optimizer | notes |
|---|---|---|
| `sven_classic` | `SvenWrapper` + `Sven` | full Jacobian, `randomized_v3` SVD |
| `sven_frac{10,25,50,80}` | masked `SvenWrapper` + `Sven` | **elementwise** masks (exact fractions), `jac_chunk_size=16` |
| `sven_struct` | block-masked `SvenWrapper` + `Sven` | structural whole-tensor mask at the ~9% this net supports |
| `sven_gram` | `GramSvenWrapper` + `SvenGram` | kernel-trick, hooks capture |
| `adam`, `adamw`, `sgd` | `torch.optim` | untuned common defaults |
| `lbfgs` | `torch.optim.LBFGS` | per-batch closure, strong-Wolfe line search |

The fraction scan is elementwise rather than structural because this
architecture concentrates ~91% of its parameters in `fc1.weight` (3072 → 300):
every whole-tensor budget between 10% and 80% collapses to the same
"everything except `fc1.weight`" ≈ 9% selection. Elementwise masks give exact
fractions but (by construction — see the `SvenWrapper` docstring) no Jacobian
memory savings; `sven_struct` shows the genuine structural savings at the one
fraction this net can express. Sub-tensor (row-block) structural masking would
lift this limitation and is the natural follow-up.

All Sven variants use `lr=0.1, k=128, rtol=1e-4` (the CIFAR scan settings).
Every config shares the same seed, model init, data subset and batch order;
CPU only, 4 torch threads, batch size 128.

## Running

```bash
python run_comparison.py                    # all configs, 20 epochs, ~2 h
python run_comparison.py --configs sven_gram adam --epochs 5
python plot_comparison.py                   # writes results/plots/*.png
```

Each config runs in a fresh subprocess so peak RSS is attributable; the
`probe` config (data + model + one forward, no training) is the memory
baseline subtracted everywhere. Per-step wall times are recorded separately
for the gradient/Jacobian phase and the optimizer-step phase; the wall-time
learning-curve axis uses cumulative optimizer time only (eval excluded).

The CIFAR batches are read directly (no torchvision) from `$CIFAR_DIR` or the
default local copy — see `data.py`.

## Caveats

- Hyperparameters are common defaults, not per-optimizer tuned; read the
  curves as "out-of-the-box" behavior, not best-achievable.
- Wall times are single-run CPU measurements on a shared machine.
- L-BFGS on 128-sample minibatches is a known-awkward baseline (stale curvature
  across batches); it is included for completeness, full-batch L-BFGS is a
  different regime.
