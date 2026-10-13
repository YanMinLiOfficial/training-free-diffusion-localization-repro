#!/usr/bin/env python3
"""Resumable parallel downloader for the OpenAI unconditional ADM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
SIZE = 2_211_383_297
SHA256 = "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_part(
    index: int,
    start: int,
    end: int,
    directory: Path,
    retries: int,
) -> tuple[int, Path]:
    path = directory / f"part-{index:05d}"
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return index, path
    for attempt in range(1, retries + 1):
        temporary = path.with_suffix(".downloading")
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, received {response.status}")
                with temporary.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"part {index} has {temporary.stat().st_size} bytes, expected {expected}"
                )
            os.replace(temporary, path)
            return index, path
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.output.is_file() and args.output.stat().st_size == SIZE:
        existing_hash = sha256(args.output)
        if existing_hash == SHA256:
            print(f"Checkpoint already verified: {args.output}")
            return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    chunk = args.chunk_mib * 1024 * 1024
    ranges = [
        (index, start, min(start + chunk - 1, SIZE - 1))
        for index, start in enumerate(range(0, SIZE, chunk))
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_part, index, start, end, parts_dir, args.retries
            ): index
            for index, start, end in ranges
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"Downloaded {completed}/{len(ranges)} chunks", flush=True)

    assembled = args.output.with_name(args.output.name + ".assembling")
    with assembled.open("wb") as output:
        for index, _, _ in ranges:
            with (parts_dir / f"part-{index:05d}").open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
    if assembled.stat().st_size != SIZE:
        raise RuntimeError("assembled checkpoint has the wrong size")
    actual_hash = sha256(assembled)
    if actual_hash != SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {actual_hash}")
    os.replace(assembled, args.output)
    for index, _, _ in ranges:
        (parts_dir / f"part-{index:05d}").unlink()
    parts_dir.rmdir()
    print(f"Verified checkpoint: {args.output}")


if __name__ == "__main__":
    main()
