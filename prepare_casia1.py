#!/usr/bin/env python3
"""Build deterministic CASIAv1 manifests from the Pham ground-truth layout."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


FIELDS = [
    "sample_id",
    "spec_id",
    "source_dataset",
    "assigned_tool",
    "image",
    "mask",
]


def collect_rows(dataset_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    image_root = dataset_root / "images" / "Tp"
    mask_root = dataset_root / "groundtruth" / "CASIA 1.0 groundtruth"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(
            "expected images/Tp and groundtruth/CASIA 1.0 groundtruth under "
            f"{dataset_root}"
        )

    masks = {
        path.stem.removesuffix("_gt"): path
        for path in mask_root.rglob("*.png")
    }
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for image in sorted(image_root.rglob("*.jpg")):
        mask = masks.get(image.stem)
        if mask is None:
            missing.append(image.name)
            continue
        manipulation = "copy-move" if image.parent.name == "CM" else "splicing"
        rows.append(
            {
                "sample_id": image.stem,
                "spec_id": image.stem,
                "source_dataset": "CASIAv1",
                "assigned_tool": manipulation,
                "image": image.relative_to(dataset_root).as_posix(),
                "mask": mask.relative_to(dataset_root).as_posix(),
            }
        )
    return rows, missing


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def balanced_pilot(
    rows: list[dict[str, str]], size: int, seed: int
) -> list[dict[str, str]]:
    if size > len(rows):
        raise ValueError("pilot size exceeds the number of matched samples")
    rng = random.Random(seed)
    groups = {
        label: [row for row in rows if row["assigned_tool"] == label]
        for label in ("copy-move", "splicing")
    }
    first = size // 2
    second = size - first
    selected = rng.sample(groups["copy-move"], first)
    selected += rng.sample(groups["splicing"], second)
    return sorted(selected, key=lambda row: row["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/casia1"))
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()

    rows, missing = collect_rows(args.dataset_root)
    pilot = balanced_pilot(rows, args.pilot_size, args.seed)
    write_manifest(args.output_dir / "all.csv", rows)
    write_manifest(args.output_dir / "pilot-20.csv", pilot)
    (args.output_dir / "missing-masks.txt").write_text(
        "\n".join(missing) + ("\n" if missing else "")
    )
    print(
        f"matched={len(rows)} pilot={len(pilot)} missing_masks={len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
