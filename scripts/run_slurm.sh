#!/usr/bin/env bash
#SBATCH --job-name=diffusion-localization
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${DATASET_ROOT:?Set DATASET_ROOT to the dataset directory}"
: "${MANIFEST:?Set MANIFEST to the CSV manifest}"
OUTPUT_FILE=${OUTPUT_FILE:-"$ROOT/results/run.npz"}

python "$ROOT/run_dual_trajectory_probe.py" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" \
  --output-file "$OUTPUT_FILE" \
  --sag-root "$ROOT/third_party/Self-Attention-Guidance" \
  --model-path "$ROOT/models/256x256_diffusion_uncond.pt"
