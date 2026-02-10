#!/usr/bin/env python3
"""Count tokens in LitData-optimized datasets via index.json metadata."""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def _format_tokens(value: int) -> str:
    units = ["", "K", "M", "B", "T"]
    size = float(value)
    unit_index = 0
    while size >= 1000 and unit_index < len(units) - 1:
        size /= 1000.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)}"
    return f"{size:.2f}{units[unit_index]}"


def _parse_bits_per_token(data_format) -> Optional[int]:
    if not data_format:
        return None
    # Example: ["no_header_tensor:16"]
    if isinstance(data_format, list):
        data_format = data_format[0] if data_format else None
    if not isinstance(data_format, str):
        return None
    match = re.search(r":(\d+)$", data_format)
    if not match:
        return None
    return int(match.group(1))


def _tokens_from_index(index_path: Path) -> int:
    meta = json.loads(index_path.read_text())
    chunks = meta.get("chunks", [])
    config = meta.get("config", {})

    total = 0
    for chunk in chunks:
        dim = chunk.get("dim")
        if isinstance(dim, int):
            total += dim
            continue

        chunk_bytes = chunk.get("chunk_bytes")
        bits_per_token = _parse_bits_per_token(config.get("data_format"))
        if isinstance(chunk_bytes, int) and bits_per_token:
            total += chunk_bytes * 8 // bits_per_token
            continue

        raise ValueError(f"Cannot determine token count from {index_path}")

    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Count tokens for LitData-optimized datasets.")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="data/qwen3",
        help="Directory containing dataset subfolders with index.json",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    datasets = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not datasets:
        raise SystemExit(f"No datasets found under: {base_dir}")

    total_all = 0
    for dataset in datasets:
        index_path = dataset / "index.json"
        if not index_path.exists():
            print(f"SKIP {dataset.name}: missing index.json")
            continue
        tokens = _tokens_from_index(index_path)
        total_all += tokens
        print(f"{dataset.name}\t{tokens}\t({_format_tokens(tokens)})")

    print(f"TOTAL\t{total_all}\t({_format_tokens(total_all)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
