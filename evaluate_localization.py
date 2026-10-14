#!/usr/bin/env python3
"""Evaluate one or more dual-trajectory NPZ shards with pixel AUC and AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(np.int64)
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    true_positive = np.cumsum(ranked)
    precision = true_positive / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def load_shards(paths: list[Path]) -> tuple[np.ndarray, ...]:
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("score_maps", "masks", "valid_masks", "sample_ids")
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            map_key = "score_maps" if "score_maps" in shard else "forged_maps"
            arrays["score_maps"].append(shard[map_key].astype(np.float32))
            for key in ("masks", "valid_masks", "sample_ids"):
                arrays[key].append(shard[key])
    result = tuple(np.concatenate(arrays[key]) for key in arrays)
    if len(set(result[-1].tolist())) != len(result[-1]):
        raise ValueError("duplicate sample IDs across shards")
    return result


def evaluate(paths: list[Path]) -> dict:
    scores, masks, valid_masks, sample_ids = load_shards(paths)
    if not np.isfinite(scores).all():
        raise ValueError("score maps contain non-finite values")

    pooled_labels = []
    pooled_scores = []
    per_image_auc = []
    per_image_ap = []
    inside = []
    outside = []
    for score, mask, valid in zip(scores, masks.astype(bool), valid_masks.astype(bool)):
        labels = mask[valid].astype(np.uint8)
        values = score[valid].astype(np.float64)
        pooled_labels.append(labels)
        pooled_scores.append(values)
        per_image_auc.append(roc_auc(labels, values))
        per_image_ap.append(average_precision(labels, values))
        inside.append(float(values[labels == 1].mean()))
        outside.append(float(values[labels == 0].mean()))

    labels = np.concatenate(pooled_labels)
    values = np.concatenate(pooled_scores)
    return {
        "samples": len(sample_ids),
        "positive_pixel_prevalence": float(labels.mean()),
        "pooled_pixel_auc": roc_auc(labels, values),
        "pooled_pixel_ap": average_precision(labels, values),
        "macro_image_auc": float(np.nanmean(per_image_auc)),
        "macro_image_ap": float(np.nanmean(per_image_ap)),
        "mean_score_inside_mask": float(np.mean(inside)),
        "mean_score_outside_mask": float(np.mean(outside)),
        "mean_inside_minus_outside": float(np.mean(np.asarray(inside) - outside)),
        "sample_ids": sample_ids.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.inputs)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
