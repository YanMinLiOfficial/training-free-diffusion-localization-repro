#!/usr/bin/env bash
#SBATCH --job-name=casia1-diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
: "${DATASET_ROOT:?Set DATASET_ROOT to the CASIAv1 directory}"
: "${MANIFEST:?Set MANIFEST to the CSV manifest}"
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/casia1"}
PYTHON=${PYTHON:-python}

mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$ROOT/run_dual_trajectory_probe.py" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" \
  --output-file "$OUTPUT_DIR/shard-$SHARD_INDEX.npz" \
  --sag-root "$ROOT/third_party/Self-Attention-Guidance" \
  --model-path "$ROOT/models/256x256_diffusion_uncond.pt" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX"
