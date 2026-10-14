#!/usr/bin/env python3
"""Training-free dual-trajectory diffusion localization probe.

This is a transparent reimplementation of Zhang et al. (AAAI 2025), built on
the official Self-Attention Guidance fork of OpenAI guided-diffusion. The paper
does not specify the exact SRM filter bank or the selected U-Net attention
block, so both choices are recorded in the output and exposed as arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MODEL_CONFIG = {
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "attention_resolutions": "32,16,8",
    "dropout": 0.0,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "timestep_respacing": "",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "use_fp16": True,
    "use_new_attention_order": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--sag-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--fixed-timestep", type=int)
    parser.add_argument("--max-timestep", type=int, default=400)
    parser.add_argument("--timestep-stride", type=int, default=1)
    parser.add_argument("--forward-ssim-threshold", type=float, default=0.2)
    parser.add_argument("--similarity-scale", type=float, default=1e4)
    parser.add_argument("--attention-scale", type=float, default=1.3)
    parser.add_argument("--attention-threshold", type=float, default=1.3)
    parser.add_argument("--blur-sigma", type=float, default=3.0)
    parser.add_argument(
        "--sel-attn-block", choices=("middle", "output"), default="output"
    )
    parser.add_argument("--sel-attn-depth", type=int, default=2)
    return parser.parse_args()


def load_sag(args: argparse.Namespace):
    root = args.sag_root.resolve()
    if not (root / "guided_diffusion" / "script_util.py").is_file():
        raise FileNotFoundError(
            f"SAG checkout not found at {root}; run prepare_dual_trajectory_assets.sh"
        )
    sys.path.insert(0, str(root))
    try:
        import torchvision.transforms  # noqa: F401
    except ModuleNotFoundError:
        # The SAG fork imports torchvision only for its own sampling helper.
        # This runner uses a local Gaussian blur and calls p_mean_variance
        # directly, so a small import-compatible shim avoids mutating the
        # existing CUDA environment.
        transforms = types.ModuleType("torchvision.transforms")

        class GaussianBlur:
            def __init__(self, kernel_size: int, sigma: float):
                self.kernel_size = kernel_size
                self.sigma = sigma

            def __call__(self, image: torch.Tensor) -> torch.Tensor:
                return gaussian_blur(image, self.sigma, self.kernel_size)

        transforms.GaussianBlur = GaussianBlur
        torchvision = types.ModuleType("torchvision")
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms
    from guided_diffusion.script_util import create_model_and_diffusion

    config = dict(MODEL_CONFIG)
    config.update(
        sel_attn_block=args.sel_attn_block,
        sel_attn_depth=args.sel_attn_depth,
    )
    model, diffusion = create_model_and_diffusion(**config)
    try:
        checkpoint = torch.load(
            args.model_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to("cuda")
    if config["use_fp16"]:
        model.convert_to_fp16()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, diffusion, config


def letterbox(
    image: Image.Image,
    size: int,
    resample: Image.Resampling,
    fill: int | tuple[int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    left = (size - resized[0]) // 2
    top = (size - resized[1]) // 2
    canvas = Image.new(image.mode, (size, size), fill)
    canvas.paste(image.resize(resized, resample), (left, top))
    return canvas, (left, top, left + resized[0], top + resized[1])


def load_image(path: Path, size: int = 256) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as source:
        image, content_box = letterbox(
            source.convert("RGB"), size, Image.Resampling.LANCZOS, (255, 255, 255)
        )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    valid = np.zeros((size, size), dtype=bool)
    left, top, right, bottom = content_box
    valid[top:bottom, left:right] = True
    return tensor, valid


def load_mask(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as source:
        mask, _ = letterbox(source.convert("L"), size, Image.Resampling.NEAREST, 0)
    return np.asarray(mask) > 0


def srm_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Three commonly used fixed SRM residual kernels. The target paper only
    # names SRM and does not identify its exact filter bank.
    kernels = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    scales = torch.tensor([4.0, 12.0, 2.0], device=device, dtype=dtype)
    return (kernels / scales[:, None, None]).unsqueeze(1)


def srm_residuals(image: torch.Tensor) -> torch.Tensor:
    gray = (
        0.2989 * image[:, 0:1]
        + 0.5870 * image[:, 1:2]
        + 0.1140 * image[:, 2:3]
    )
    kernels = srm_kernels(gray.device, gray.dtype)
    return F.conv2d(gray, kernels, padding=2)


def gaussian_kernel(
    size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=torch.float32)
    coordinates -= (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).to(dtype=dtype)
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def gaussian_blur(
    image: torch.Tensor, sigma: float, kernel_size: int = 31
) -> torch.Tensor:
    size = kernel_size
    if size < 3 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be an odd integer >= 3")
    kernel = gaussian_kernel(
        size, sigma, image.shape[1], image.device, image.dtype
    )
    return F.conv2d(image, kernel, padding=size // 2, groups=image.shape[1])


def normalize_pair(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, ...]:
    combined = torch.stack((first, second))
    low = combined.amin(dim=(0, 3, 4), keepdim=False)[..., None, None]
    high = combined.amax(dim=(0, 3, 4), keepdim=False)[..., None, None]
    scale = (high - low).clamp_min(1e-6)
    return (first - low) / scale, (second - low) / scale


def ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    first, second = normalize_pair(first.float(), second.float())
    channels = first.shape[1]
    kernel = gaussian_kernel(11, 1.5, channels, first.device, first.dtype)
    mu_first = F.conv2d(first, kernel, padding=5, groups=channels)
    mu_second = F.conv2d(second, kernel, padding=5, groups=channels)
    sigma_first = F.conv2d(first.square(), kernel, padding=5, groups=channels)
    sigma_first -= mu_first.square()
    sigma_second = F.conv2d(second.square(), kernel, padding=5, groups=channels)
    sigma_second -= mu_second.square()
    covariance = F.conv2d(first * second, kernel, padding=5, groups=channels)
    covariance -= mu_first * mu_second
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_first * mu_second + c1) * (2 * covariance + c2)) / (
        (mu_first.square() + mu_second.square() + c1)
        * (sigma_first + sigma_second + c2)
    )
    return float(score.mean().clamp(-1, 1))


def choose_timestep(
    diffusion,
    x0: torch.Tensor,
    forward_noise: torch.Tensor,
    threshold: float,
    maximum: int,
    stride: int,
) -> tuple[int, float, bool]:
    reference = srm_residuals(x0)
    last_score = 1.0
    for value in range(max(1, stride), maximum + 1, stride):
        timestep = torch.full((1,), value, device=x0.device, dtype=torch.long)
        xt = diffusion.q_sample(x0, timestep, noise=forward_noise)
        last_score = ssim(reference, srm_residuals(xt))
        if last_score < threshold:
            return value, last_score, True
    return maximum, last_score, False


def attention_mask(
    attention: torch.Tensor,
    batch: int,
    heads: int,
    output_shape: tuple[int, int],
    threshold: float,
) -> torch.Tensor:
    tokens = attention.shape[-1]
    resolution = math.isqrt(tokens)
    if resolution * resolution != tokens:
        raise ValueError(f"attention token count is not square: {tokens}")
    importance = attention.reshape(batch, heads, tokens, tokens).mean(1).sum(1)
    mask = (importance > threshold).reshape(batch, 1, resolution, resolution)
    return F.interpolate(mask.float(), output_shape, mode="nearest")


def similarity_gradient(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        variable = current.detach().float().requires_grad_(True)
        similarity = -F.mse_loss(variable, reference.float(), reduction="mean")
        return torch.autograd.grad(similarity, variable)[0].to(current.dtype)


def guided_mean(
    diffusion,
    model,
    current: torch.Tensor,
    timestep: torch.Tensor,
    reference_xt: torch.Tensor,
    similarity_scale: float,
    attention_scale: float,
    attention_threshold: float,
    blur_sigma: float,
    heads: int,
) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        output = diffusion.p_mean_variance(model, current, timestep)
        mask = attention_mask(
            output["attn_map"],
            current.shape[0],
            heads,
            current.shape[-2:],
            attention_threshold,
        )
        blurred_x0 = gaussian_blur(output["pred_xstart"], blur_sigma)
        masked_x0 = (1 - mask) * output["pred_xstart"] + mask * blurred_x0
        masked_xt = diffusion.q_sample(masked_x0, timestep, noise=output["eps"])
        blurred_output = diffusion.p_mean_variance(model, masked_xt, timestep)

    sag_epsilon = output["eps"] + attention_scale * (
        output["eps"] - blurred_output["eps"]
    )
    gradient = similarity_gradient(current, reference_xt)
    index = int(timestep[0])
    alpha_bar = float(diffusion.alphas_cumprod[index])
    similarity_at_t = similarity_scale * math.sqrt((1 - alpha_bar) / alpha_bar)
    sigma_t = math.sqrt(float(diffusion.betas[index]))
    guided_epsilon = sag_epsilon - similarity_at_t * sigma_t * gradient

    predicted_x0 = diffusion._predict_xstart_from_eps(
        x_t=current, t=timestep, eps=guided_epsilon
    ).clamp(-1, 1)
    mean, _, _ = diffusion.q_posterior_mean_variance(
        x_start=predicted_x0, x_t=current, t=timestep
    )
    return mean, output


def aggregate_maps(
    timesteps: list[int], maps: list[torch.Tensor], similarities: list[float]
) -> tuple[torch.Tensor, int, float]:
    minimum_index = int(np.argmin(similarities))
    minimum_timestep = timesteps[minimum_index]
    selected = [score for t, score in zip(timesteps, maps) if t <= minimum_timestep]
    return torch.stack(selected).mean(0), minimum_timestep, similarities[minimum_index]


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (base_seed + offset) % (2**31)


def run_trajectory(
    model,
    diffusion,
    x0: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    forward_noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32)
    forward_noise = forward_noise.to(x0.device, dtype=x0.dtype)
    if args.fixed_timestep is None:
        selected_t, forward_ssim, crossed = choose_timestep(
            diffusion,
            x0,
            forward_noise,
            args.forward_ssim_threshold,
            args.max_timestep,
            args.timestep_stride,
        )
    else:
        selected_t = args.fixed_timestep
        timestep = torch.full((1,), selected_t, device=x0.device, dtype=torch.long)
        xt = diffusion.q_sample(x0, timestep, noise=forward_noise)
        forward_ssim = ssim(srm_residuals(x0), srm_residuals(xt))
        crossed = forward_ssim < args.forward_ssim_threshold

    start_t = torch.full((1,), selected_t, device=x0.device, dtype=torch.long)
    start = diffusion.q_sample(x0, start_t, noise=forward_noise)
    unconditional = start.clone()
    conditional = start.clone()
    timesteps: list[int] = []
    maps: list[torch.Tensor] = []
    similarities: list[float] = []

    for value in range(selected_t, -1, -1):
        timestep = torch.full((1,), value, device=x0.device, dtype=torch.long)
        reference_xt = diffusion.q_sample(x0, timestep, noise=forward_noise)
        with torch.no_grad():
            unconditional_output = diffusion.p_mean_variance(
                model, unconditional, timestep
            )
        conditional_mean, conditional_output = guided_mean(
            diffusion,
            model,
            conditional,
            timestep,
            reference_xt,
            args.similarity_scale,
            args.attention_scale,
            args.attention_threshold,
            args.blur_sigma,
            MODEL_CONFIG["num_heads"],
        )
        step_noise = torch.randn(
            unconditional.shape, generator=generator, dtype=torch.float32
        ).to(unconditional.device, dtype=unconditional.dtype)
        nonzero = 0.0 if value == 0 else 1.0
        unconditional = unconditional_output["mean"] + nonzero * torch.exp(
            0.5 * unconditional_output["log_variance"]
        ) * step_noise
        conditional = conditional_mean + nonzero * torch.exp(
            0.5 * conditional_output["log_variance"]
        ) * step_noise
        state_t = max(0, value - 1)
        score = (conditional.float() - unconditional.float()).square().mean(1)[0]
        similarity = ssim(srm_residuals(conditional), srm_residuals(unconditional))
        # OpenAI's zero-based schedule performs a final model step at t=0.
        # Replace the preceding q(x_0)-like state so timestep zero contributes
        # exactly one (final clean) map to Eq. 13's m+1-map average.
        if state_t == 0 and timesteps and timesteps[-1] == 0:
            maps[-1] = score
            similarities[-1] = similarity
        else:
            timesteps.append(state_t)
            maps.append(score)
            similarities.append(similarity)

    score_map, aggregation_m, minimum_ssim = aggregate_maps(
        timesteps, maps, similarities
    )
    diagnostics = {
        "selected_timestep": selected_t,
        "forward_srm_ssim": forward_ssim,
        "forward_threshold_crossed": crossed,
        "aggregation_m": aggregation_m,
        "minimum_reverse_srm_ssim": minimum_ssim,
        "reverse_timesteps": timesteps,
        "reverse_srm_ssim": similarities,
    }
    return score_map.cpu().numpy().astype(np.float16), diagnostics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.fixed_timestep is not None and not 1 <= args.fixed_timestep < 1000:
        raise ValueError("fixed-timestep must be in [1, 999]")
    if not 1 <= args.max_timestep < 1000:
        raise ValueError("max-timestep must be in [1, 999]")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    model, diffusion, model_config = load_sag(args)
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest selection is empty")

    paired_mode = bool(rows[0].get("authentic"))

    authentic_maps = []
    score_maps = []
    masks = []
    valid_masks = []
    authentic_diagnostics = []
    image_diagnostics = []
    for index, row in enumerate(rows, start=1):
        image, image_valid = load_image(args.dataset_root / row["image"])
        image = image.to("cuda", dtype=torch.float16)
        sample_seed = stable_sample_seed(args.seed, row["sample_id"])
        score_map, image_info = run_trajectory(
            model, diffusion, image, args, sample_seed
        )
        score_maps.append(score_map)
        masks.append(load_mask(args.dataset_root / row["mask"]))
        valid_masks.append(image_valid)
        image_diagnostics.append(image_info)

        if paired_mode:
            authentic, authentic_valid = load_image(
                args.dataset_root / row["authentic"]
            )
            if not np.array_equal(authentic_valid, image_valid):
                raise ValueError(f"content boxes differ for {row['sample_id']}")
            authentic = authentic.to("cuda", dtype=torch.float16)
            authentic_map, authentic_info = run_trajectory(
                model, diffusion, authentic, args, sample_seed
            )
            authentic_maps.append(authentic_map)
            authentic_diagnostics.append(authentic_info)
            progress = (
                f"T(auth/image)={authentic_info['selected_timestep']}/"
                f"{image_info['selected_timestep']}"
            )
        else:
            progress = f"T={image_info['selected_timestep']}"
        print(f"[{index}/{len(rows)}] {row['sample_id']} {progress}", flush=True)

    protocol = {
        "method": "dual conditional/unconditional reverse diffusion",
        "model": "OpenAI ImageNet 256x256 unconditional guided-diffusion",
        "model_sha256": "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1",
        "sag_commit": (
            (args.sag_root / ".git" / "HEAD").read_text().strip()
            if (args.sag_root / ".git" / "HEAD").is_file()
            else "unknown"
        ),
        "model_config": model_config,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "forward_ssim_threshold": args.forward_ssim_threshold,
        "fixed_timestep": args.fixed_timestep,
        "max_timestep": args.max_timestep,
        "timestep_stride": args.timestep_stride,
        "forward_noise_policy": (
            "one epsilon is reused across adaptive-timestep candidates and "
            "to initialize the selected reverse trajectory; paper does not specify this"
        ),
        "reverse_zero_policy": (
            "the final model step at t=0 replaces the preceding zero-labeled map, "
            "so timestep zero contributes once"
        ),
        "similarity_metric": (
            "negative mean squared error to matched forward x_t; "
            "paper leaves D unspecified"
        ),
        "similarity_scale": args.similarity_scale,
        "attention_scale": args.attention_scale,
        "attention_threshold": args.attention_threshold,
        "blur_sigma": args.blur_sigma,
        "srm_filters": "three fixed 5x5 residual kernels; paper filter bank unspecified",
        "attention_source": f"SAG {args.sel_attn_block} block depth {args.sel_attn_depth}",
        "mask_usage": "evaluation only",
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output = dict(
        score_maps=np.stack(score_maps),
        forged_maps=np.stack(score_maps),
        masks=np.stack(masks),
        valid_masks=np.stack(valid_masks),
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        spec_ids=np.asarray([row.get("spec_id", row["sample_id"]) for row in rows]),
        source_datasets=np.asarray(
            [row.get("source_dataset", "unknown") for row in rows]
        ),
        generators=np.asarray([row.get("assigned_tool", "unknown") for row in rows]),
        selected_timesteps=np.asarray(
            [item["selected_timestep"] for item in image_diagnostics]
        ),
        forged_selected_timesteps=np.asarray(
            [item["selected_timestep"] for item in image_diagnostics]
        ),
        aggregation_m=np.asarray(
            [item["aggregation_m"] for item in image_diagnostics]
        ),
        forged_aggregation_m=np.asarray(
            [item["aggregation_m"] for item in image_diagnostics]
        ),
        protocol_json=np.asarray(json.dumps(protocol, sort_keys=True)),
    )
    if paired_mode:
        output.update(
            authentic_maps=np.stack(authentic_maps),
            authentic_selected_timesteps=np.asarray(
                [item["selected_timestep"] for item in authentic_diagnostics]
            ),
            authentic_aggregation_m=np.asarray(
                [item["aggregation_m"] for item in authentic_diagnostics]
            ),
        )
    np.savez_compressed(args.output_file, **output)
    diagnostics_path = args.output_file.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(
        json.dumps(
            {
                "protocol": protocol,
                "samples": [
                    {
                        "sample_id": row["sample_id"],
                        "image": image_info,
                        **(
                            {"authentic": authentic_info}
                            if paired_mode
                            else {}
                        ),
                    }
                    for row, image_info, authentic_info in zip(
                        rows,
                        image_diagnostics,
                        authentic_diagnostics
                        if paired_mode
                        else [None] * len(rows),
                    )
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} dual-trajectory samples to {args.output_file}")


if __name__ == "__main__":
    main()
