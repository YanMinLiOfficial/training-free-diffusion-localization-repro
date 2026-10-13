#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SAG_ROOT="$ROOT/third_party/Self-Attention-Guidance"
MODEL_PATH="$ROOT/models/256x256_diffusion_uncond.pt"
SAG_COMMIT=a36f251dac2fc535e9d0670b86527df9eb1c661c

mkdir -p "$ROOT/third_party" "$ROOT/models"
if [[ ! -d "$SAG_ROOT/.git" ]]; then
  git clone https://github.com/SusungHong/Self-Attention-Guidance.git "$SAG_ROOT"
fi
git -C "$SAG_ROOT" fetch --depth 1 origin "$SAG_COMMIT"
git -C "$SAG_ROOT" checkout --detach "$SAG_COMMIT"

python "$ROOT/download_openai_checkpoint.py" "$MODEL_PATH"

printf 'SAG root: %s\nModel: %s\n' "$SAG_ROOT" "$MODEL_PATH"
