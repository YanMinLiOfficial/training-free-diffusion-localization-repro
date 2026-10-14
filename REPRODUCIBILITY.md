# Reproduction status

## CASIAv1 adaptive smoke test

One CASIAv1 splicing image was processed with adaptive timestep selection on
an NVIDIA L40S. It completed in 25 seconds with exit code 0, selected `T=69`,
and crossed the forward SRM-SSIM threshold at `0.199025`.

The finite 256x256 score map had pixel AUROC `0.6818` and AP `0.1684`, versus
a positive-pixel prevalence of `0.0959`. Its mean score was `0.000684` inside
the mask and `0.000355` outside. These single-image values are diagnostic only.

The reverse SRM-SSIM minimum occurred at `m=0`, so only the final error map was
aggregated. This does not yet match the paper's typical description of a
reverse curve that first falls and then rises. The 20-image pilot records the
complete reverse curves to determine whether this is systematic.

## CASIAv1 20-image pilot

The balanced pilot (10 copy-move and 10 splicing samples) completed with exit
code 0. Adaptive `T` ranged from 36 to 253, with median 118.5. Its localization
metrics were:

| Aggregation | Pixel AUROC | Pixel AP |
| --- | ---: | ---: |
| Pooled pixels | 0.5892 | 0.1073 |
| Mean per image | 0.5656 | 0.1068 |

The positive-pixel prevalence was 0.0710. Mean scores were 0.001436 inside
masks and 0.001072 outside, for a positive mean difference of 0.000364.

All 20 reverse trajectories selected `m=0`; their SRM-SSIM curves decreased to
the final state rather than falling and then rising. Although pooled AUROC is
numerically close to the paper's full-CASIAv1 value of 0.587, this small,
balanced pilot uses a different sample composition and fails to reproduce a
central claimed trajectory behavior. It is therefore not evidence of a
successful benchmark reproduction. A full 920-image run is intentionally
deferred until the unspecified similarity metric, SRM/SSIM convention, and
attention extraction choices are resolved or ablated.

## Confirmed in the smoke test

The first end-to-end smoke test ran on one NVIDIA L40S with a fixed timestep of
10. It completed in 26 seconds with exit code 0 and produced finite 256x256
score maps for both images in the pair.

| Diagnostic | Authentic | Forged |
| --- | ---: | ---: |
| Selected timestep | 10 | 10 |
| Forward SRM-SSIM | 0.721960 | 0.721900 |
| Threshold 0.2 crossed | No | No |
| Aggregation timestep `m` | 0 | 0 |
| Minimum reverse SRM-SSIM | 0.997535 | 0.998260 |

For the forged image, the mean score was `1.03e-5` inside the ground-truth
tampered region and `1.27e-5` outside it. Pixel AUROC was 0.657 and AP was
0.0085, with a positive-pixel prevalence of 0.00576.

This is weak and directionally inconsistent localization evidence: the mean
score inside the tampered region was lower than outside. The authentic image
also had elevated scores at the corresponding coordinates, so spatial or
content bias cannot be excluded.

## What this test establishes

- The model checkpoint and pinned SAG code are compatible.
- The dual-trajectory implementation executes end to end.
- The output and diagnostic contracts are valid and numerically finite.

## What this test does not establish

- It does not reproduce the paper's reported benchmark performance.
- It does not exercise adaptive timestep selection because `T=10` was fixed.
- It does not establish which unspecified implementation assumptions match the
  authors' implementation.
- A single example cannot support a localization or generalization claim.

The raw diagnostic record is stored in
`results/smoke/one-sample-t10.diagnostics.json`. The NPZ score maps and dataset
content are intentionally not published.
