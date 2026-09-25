# Transformer optimizer comparison

Head-to-head comparison of Sven variants against standard PyTorch optimizers
on a byte-level language model: a minimal GPT (manual causal multi-head
attention, pre-LN, GELU MLP, learned tok+pos embeddings, untied head, no
dropout — the architecture on which the chunked Gram capture was verified
exact to ~1e-15) trained on the leading ~3.5 MB of a local OpenWebText
parquet shard. The transformer matters here because the **hooks** Gram
capture cannot run it at all ((B, T, d) Linear inputs, LayerNorm and
Embedding are all guarded) — the **chunked** capture is the only Gram path,
and this suite measures what that costs.

Two model tiers (`model.py`):

| tier | V | T | d | L | H | P | B | used for |
|---|---|---|---|---|---|---|---|---|
| `train` | 256 | 64 | 128 | 4 | 4 | ~0.87M | 32 | the training comparison |
| `profile` | 256 | 128 | 192 | 4 | 6 | ~1.90M | 64 | per-step cost probes only |

Rows are per-sequence mean CE over tokens, shape (B,), so the Sven system is
M=32 (train tier). Data (`data.py`) is read directly from the single
verified local parquet shard with pyarrow — first ~750 docs only, never the
`datasets` library (it stalls fetching missing shards) and never the network.
Train/val are a contiguous disjoint byte split; training windows are seeded
per step so every config sees the identical batch order; validation is a
fixed strided set of 256 windows. Metrics are per-token CE (nats) and
bits-per-byte (CE/ln 2 — tokens are bytes).

## Configs

| name | optimizer | notes |
|---|---|---|
| `sven_classic` | `SvenWrapper` + `Sven` | full (M, P) Jacobian, `randomized_v3` SVD |
| `sven_gram_c{18,20,22}` | chunked `GramSvenWrapper` + `SvenGram` | the `chunk_numel` scan (2^18 / 2^20 / 2^22); the Gram is exact at every chunk size, so updates are identical — only capture memory/time differ |
| `sven_gram_rows{10,25,50}` | masked chunked `GramSvenWrapper` + `SvenGram` | `mask_mode="rows"` at `chunk_numel=2^20` — leading-axis slices per tensor (chunked-rows semantics); on a checkout without masked-chunked support, construction FAILs cleanly and the run continues |
| `adam`, `adamw` | `torch.optim` | lr 3e-4 (AdamW wd 0.01) |
| `sgd` | `torch.optim.SGD` | lr 0.1, momentum 0.9 |

All Sven variants share `lr=0.1, k=32, rtol=1e-4`. The lr comes from a
30-step sanity scan of `sven_gram_c20` over {0.1, 0.3, 1.0}: 0.1 was the
only monotone decrease (val CE 5.60 → 3.10 nats), 0.3 oscillated
(5.60 → 3.22 non-monotone), 1.0 diverged. Every config shares the same
seed, model init (GPT-2-style N(0, 0.02)), data and batch order; CPU only,
4 torch threads.

## Running

```bash
python run_comparison.py                     # all configs, 300 steps
python run_comparison.py --configs sven_gram_c20 adamw --steps 30
python mem_probe.py --sweep-tiers profile train
python plot_comparison.py                    # writes results/plots/*.png
```

Each config runs in a fresh subprocess so peak RSS is attributable; the
`probe` config (data + model + one forward) is the memory baseline
subtracted everywhere. Per-step wall times are recorded separately for the
gradient/Gram phase and the optimizer-step phase; the wall-time
learning-curve axis uses cumulative optimizer time only (eval excluded).

`mem_probe.py` isolates the per-step working set: two optimizer steps on
synthetic batches per config, fresh process, at BOTH tiers, plus the
`chunk_numel` sweep {2^16, 2^18, 2^20, 2^22}. Because the machine is
shared, every probe child runs under a watchdog that aborts the process once
peak RSS crosses `--cap-gb` (default 2 GB); such configs are reported as
over-cap (faded bars in the plots) rather than measured.

## Caveats

- **Profile tier exceeds the 2 GB cap for every jacrev-based capture.** The
  chunked capture's per-group Jacobian block is chunk-bounded, but the
  activation cotangents inside each group's `jacrev` scale with M × the
  activation sizes (the (M, B, H, T, T) attention-score cotangent alone is
  ~1.6 GB at M=B=64, T=128), and multi-tensor groups always run the full
  M-row vmap. A single uncapped reference run at `chunk_numel=2^16` peaked
  at ~4.8 GB and 139 s/step. The populated tradeoff curve therefore comes
  from the train-tier sweep; the profile-tier sweep documents the over-cap
  boundary.
- `sven_classic` exceeds the 2 GB cap even at the train tier: the
  flat-parameter `jacrev` scatters every per-tensor cotangent into (M, P)
  zero-filled transients (one per parameter view).  Run it on a machine with
  more headroom, or read the Gram configs as the memory-honest Sven numbers.
- The chunked-rows mask (`sven_gram_rows*`) selects leading-axis slices per
  parameter tensor, matching `sven.jax.gram` — NOT the hooks-mode
  weight-row+bias pairing.
- Hyperparameters are common defaults plus the one Sven lr scan above; read
  the curves as "out-of-the-box" behavior, not best-achievable.
- Wall times are single-run CPU measurements on a shared machine.
- `torch.func.jacrev` chunking means the chunk scan trades time for memory
  only below `chunk_numel` ≈ the largest tensor (oversized tensors are
  row-chunked); above it, capture memory is dominated by the M-row
  cotangents, not the chunk budget.
