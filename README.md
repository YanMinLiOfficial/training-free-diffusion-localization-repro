# Training-Free Diffusion Localization Reproduction

An independent, approximate reproduction of **Training-Free Image
Manipulation Localization Using Diffusion Models** (Zhang, Chang, and Li,
AAAI 2025).

The repository is intentionally explicit about what is reproduced, what is
inferred, and what remains unknown. It is not an official implementation.

## Current status

- The OpenAI ImageNet 256x256 unconditional diffusion checkpoint loads with
  the pinned Self-Attention Guidance (SAG) implementation.
- The conditional and unconditional reverse trajectories run from the same
  noisy image and reuse the same random noise at every reverse step.
- Similarity guidance, SAG, adaptive forward-timestep selection, and reverse
  error-map aggregation are implemented.
- A one-pair `T=10` smoke test completed successfully on an NVIDIA L40S.
- The smoke test validates execution only. It does **not** validate the
  localization claims in the paper.

See [Reproduction status](REPRODUCIBILITY.md) for the numerical smoke-test
result and unresolved implementation choices.

## Method implemented

For an input image `x0`, the runner:

1. Uses fixed SRM residual filters and SSIM to select a forward timestep `T`,
   unless `--fixed-timestep` is supplied.
2. Samples `xT` and starts two reverse trajectories from the same state.
3. Runs an unconditional trajectory and a similarity/SAG-guided trajectory.
4. Reuses identical Gaussian noise for both trajectories at each reverse step.
5. Computes their per-pixel squared difference and aggregates maps from the
   reverse timestep with minimum SRM-SSIM through timestep zero.

The dataset mask is loaded only for evaluation and never enters inference.

## Installation

Python 3.10 and a CUDA-capable PyTorch installation are recommended.

```bash
git clone https://github.com/YanMinLiOfficial/training-free-diffusion-localization-repro.git
cd training-free-diffusion-localization-repro

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the PyTorch build appropriate for your CUDA environment first.
pip install -r requirements.txt

bash scripts/prepare_assets.sh
```

`prepare_assets.sh` checks out SAG at commit
`a36f251dac2fc535e9d0670b86527df9eb1c661c` and downloads the official OpenAI
checkpoint. The checkpoint is accepted only after its size and SHA-256 digest
have been verified.

## Manifest format

The current experiment runner evaluates matched authentic/forged pairs. Its
CSV manifest requires:

```csv
sample_id,spec_id,source_dataset,assigned_tool,authentic,image,mask
example/0001,example-1,dataset,generator,authentic/0001.png,forged/0001.png,masks/0001.png
```

Paths are relative to `--dataset-root`. The pair is an evaluation convenience;
each image is processed independently by the diffusion method.

## Run

```bash
python run_dual_trajectory_probe.py \
  --dataset-root /path/to/dataset \
  --manifest manifests/example.csv \
  --output-file results/run.npz \
  --sag-root third_party/Self-Attention-Guidance \
  --model-path models/256x256_diffusion_uncond.pt \
  --limit 1
```

For a cheap execution check, add `--fixed-timestep 10`. This bypasses the
paper's adaptive timestep selection and must not be used as evidence of method
quality.

Run helper tests with:

```bash
python -m unittest discover -s tests -v
```

## Outputs

The compressed NPZ contains authentic and forged score maps, evaluation masks,
valid-content masks, sample metadata, selected timesteps, aggregation points,
and a JSON-encoded protocol record. A separate `*.diagnostics.json` records
per-image timestep and SSIM diagnostics.

## Important limitations

The paper does not uniquely specify the SRM filter bank, similarity function
`D`, or selected U-Net attention layer. This implementation therefore uses:

- three common fixed 5x5 SRM residual kernels;
- negative pixel MSE for `D`;
- the SAG output attention block at depth 2.

These are testable implementation assumptions, not claims about the authors'
private code. Please do not describe this repository as an exact reproduction
until those details are confirmed.

## Reference

Zhenfei Zhang, Ming-Ching Chang, and Xin Li. "Training-Free Image
Manipulation Localization Using Diffusion Models." AAAI 2025.

## License

The original code in this repository is released under the MIT License.
External checkpoints and the SAG repository retain their own licenses and are
not redistributed here.
