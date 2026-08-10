#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/tfm-%j.out

# Transformer optimizer comparison on GPU.
#
# Prerequisites on the cluster:
#   1. uv, plus one `uv sync --extra evals` on a login node (compute nodes may
#      lack network); `uv run` then reuses the project .venv — no mamba env or
#      pip install -e needed.
#   2. The OpenWebText parquet shard (303 MB), with OWT_SHARD pointing at it:
#      scp from the workstation, or download any plain_text parquet shard of
#      Skylion007/openwebtext.
#
# Usage:
#   uv sync --extra evals
#   mkdir -p slurm_logs
#   sbatch comparisons/transformer/submit_cluster.sh            # full campaign
#   sbatch comparisons/transformer/submit_cluster.sh --steps 5  # smoke first

source ~/.bash_profile
cd "$(dirname "$0")"

export OWT_SHARD="${OWT_SHARD:-$HOME/data/openwebtext_shard.parquet}"

set -x
uv run --extra evals python run_comparison.py --device cuda "$@"
uv run --extra evals python mem_probe.py --device cuda --cap-gb 32
uv run --extra evals python plot_comparison.py
