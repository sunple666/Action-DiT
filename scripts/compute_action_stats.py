"""Compute training-only LIBERO action statistics and save them as JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.action_stats import compute_action_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_action_stats(args.dataset_root, args.manifest)
    stats.save(args.output)

    print(f"stats: {args.output.expanduser().resolve()}")
    print(f"split: {stats.split}")
    print(f"count: {stats.count}")
    print(f"action_dim: {stats.action_dim}")
    print(f"action_min: {list(stats.action_min)}")
    print(f"action_max: {list(stats.action_max)}")
    print(f"action_mean: {list(stats.action_mean)}")
    print(f"action_std: {list(stats.action_std)}")


if __name__ == "__main__":
    main()
