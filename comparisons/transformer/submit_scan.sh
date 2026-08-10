#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/scan-%j.out

# rtol/k/lr sensitivity scan for the Gram Sven transformer (hooks capture).
#
# Findings from the local CPU scan this reproduces and extends (see
# reports/): rtol <= 1e-2 never truncates on this task (full rank kept);
# rtol ~ 0.1 is the optimum; hard k-truncation and lr > 0.1 hurt; Sven
# plateaus after ~300 steps while AdamW keeps descending.
#
# Environment: uv project env from the repo pyproject.toml — no mamba env or
# pip install -e needed.  Run `uv sync --extra evals` once on a login node
# (compute nodes may lack network); `uv run` then reuses the project .venv.
#
# Usage:
#   uv sync --extra evals
#   mkdir -p slurm_logs
#   sbatch comparisons/transformer/submit_scan.sh            # 600-step scan
#   STEPS=1000 sbatch comparisons/transformer/submit_scan.sh

source ~/.bash_profile
cd "$(dirname "$0")"

export OWT_SHARD="${OWT_SHARD:-$HOME/data/openwebtext_shard.parquet}"
STEPS="${STEPS:-600}"
OUT="results_scan"

run () {  # run <tag> <extra args...>
  tag=$1; shift
  uv run --extra evals python train_one.py --device cuda --steps "$STEPS" --out "$OUT" --tag "$tag" "$@"
}

set -x
run scan_adamw       --config adamw
run scan_r1e4        --config sven_gram_c20 --capture hooks                    # default rtol (no truncation)
run scan_r0p03       --config sven_gram_c20 --capture hooks --sven-rtol 0.03
run scan_r0p05       --config sven_gram_c20 --capture hooks --sven-rtol 0.05
run scan_r0p08       --config sven_gram_c20 --capture hooks --sven-rtol 0.08
run scan_r0p1        --config sven_gram_c20 --capture hooks --sven-rtol 0.1
run scan_r0p12       --config sven_gram_c20 --capture hooks --sven-rtol 0.12
run scan_r0p15       --config sven_gram_c20 --capture hooks --sven-rtol 0.15
run scan_r0p2        --config sven_gram_c20 --capture hooks --sven-rtol 0.2
run scan_k16         --config sven_gram_c20 --capture hooks --sven-k 16
run scan_k8          --config sven_gram_c20 --capture hooks --sven-k 8
run scan_lr0p05_r0p1 --config sven_gram_c20 --capture hooks --sven-rtol 0.1 --sven-lr 0.05

# Larger-M scan: M (= batch size, with per-sequence rows) sets the Jacobian
# row count and hence the per-step update-subspace rank.  k = M keeps the
# full row space; rtol = 0.1 is the tolerance optimum from the M=32 scan.
# AdamW references at the same batch sizes keep the comparison fair (bigger
# batches help AdamW too).  Note: different B => different (seeded) batch
# sequences; compare configs at equal B, and any-B curves only vs steps.
run scan_M64_r0p1    --config sven_gram_c20 --capture hooks --sven-rtol 0.1 --batch-size 64  --sven-k 64
run scan_M128_r0p1   --config sven_gram_c20 --capture hooks --sven-rtol 0.1 --batch-size 128 --sven-k 128
run scan_M64_rdef    --config sven_gram_c20 --capture hooks --batch-size 64  --sven-k 64
run scan_adamw_M64   --config adamw --batch-size 64
run scan_adamw_M128  --config adamw --batch-size 128

uv run --extra evals python plot_scan.py --results "$OUT"
