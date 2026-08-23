"""Build a portable train/validation manifest for LIBERO demonstrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.libero_index import LiberoSampleIndex, build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_split_manifest(
        args.dataset_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    manifest.save(args.output)

    train_index = LiberoSampleIndex(manifest.train)
    val_index = LiberoSampleIndex(manifest.val)
    print(f"manifest: {args.output.expanduser().resolve()}")
    print(
        f"train: {len(manifest.train)} demos, "
        f"{len(train_index)} timestep samples"
    )
    print(f"val: {len(manifest.val)} demos, {len(val_index)} timestep samples")

    if len(train_index):
        first = train_index.locate(0)
        last = train_index.locate(-1)
        print(
            "train boundaries: "
            f"first={first.episode.file}:{first.episode.demo}@{first.timestep}, "
            f"last={last.episode.file}:{last.episode.demo}@{last.timestep}"
        )


if __name__ == "__main__":
    main()
