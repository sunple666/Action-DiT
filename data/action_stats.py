"""Compute and persist per-dimension action statistics from raw LIBERO demos."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from data.libero_index import SplitManifest


STATS_VERSION = 1


@dataclass(frozen=True)
class ActionStats:
    version: int
    split: str
    count: int
    action_dim: int
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]

    def save(self, path: str | Path) -> None:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        for key in ("action_min", "action_max", "action_mean", "action_std"):
            payload[key] = list(payload[key])
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActionStats":
        stats_path = Path(path).expanduser().resolve()
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        stats = cls(
            version=int(payload["version"]),
            split=str(payload["split"]),
            count=int(payload["count"]),
            action_dim=int(payload["action_dim"]),
            action_min=tuple(float(value) for value in payload["action_min"]),
            action_max=tuple(float(value) for value in payload["action_max"]),
            action_mean=tuple(float(value) for value in payload["action_mean"]),
            action_std=tuple(float(value) for value in payload["action_std"]),
        )
        stats.validate()
        return stats

    def validate(self) -> None:
        if self.version != STATS_VERSION:
            raise ValueError(
                f"Unsupported action stats version {self.version}; "
                f"expected {STATS_VERSION}"
            )
        if self.split != "train":
            raise ValueError(f"Action statistics must use train split, got {self.split!r}")
        if self.count <= 0 or self.action_dim <= 0:
            raise ValueError("Action statistics have invalid count or dimension")

        arrays = {
            "action_min": np.asarray(self.action_min, dtype=np.float64),
            "action_max": np.asarray(self.action_max, dtype=np.float64),
            "action_mean": np.asarray(self.action_mean, dtype=np.float64),
            "action_std": np.asarray(self.action_std, dtype=np.float64),
        }
        for name, values in arrays.items():
            if values.shape != (self.action_dim,):
                raise ValueError(
                    f"{name} must have shape ({self.action_dim},), got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")

        if np.any(arrays["action_max"] - arrays["action_min"] <= 1e-6):
            raise ValueError("At least one action dimension has almost no variation")
        if np.any(arrays["action_std"] < 0.0):
            raise ValueError("Action standard deviation cannot be negative")


def compute_action_stats(
    dataset_root: str | Path,
    manifest_path: str | Path,
) -> ActionStats:
    """Scan each raw training action exactly once and return streaming statistics."""

    root = Path(dataset_root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_file}")

    manifest = SplitManifest.load(manifest_file)
    episodes = manifest.train
    if not episodes:
        raise ValueError("Training split contains no episodes")

    episodes_by_file: dict[str, list] = defaultdict(list)
    for episode in episodes:
        episodes_by_file[episode.file].append(episode)

    action_dim = 7
    action_min = np.full(action_dim, np.inf, dtype=np.float64)
    action_max = np.full(action_dim, -np.inf, dtype=np.float64)
    action_sum = np.zeros(action_dim, dtype=np.float64)
    action_squared_sum = np.zeros(action_dim, dtype=np.float64)
    count = 0

    for relative_file, file_episodes in episodes_by_file.items():
        full_path = (root / relative_file).resolve()
        if not full_path.is_relative_to(root):
            raise ValueError(f"HDF5 path escapes dataset root: {relative_file}")
        if not full_path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {full_path}")

        with h5py.File(full_path, "r") as file:
            for episode in file_episodes:
                actions = np.asarray(
                    file["data"][episode.demo]["actions"],
                    dtype=np.float64,
                )
                if actions.ndim != 2 or actions.shape[1] != action_dim:
                    raise ValueError(
                        f"Expected actions [T, {action_dim}], got {actions.shape} at "
                        f"{relative_file}:{episode.demo}"
                    )
                if actions.shape[0] != episode.length:
                    raise ValueError(
                        f"Manifest length {episode.length} differs from HDF5 length "
                        f"{actions.shape[0]} at {relative_file}:{episode.demo}"
                    )
                if not np.isfinite(actions).all():
                    raise ValueError(
                        f"Non-finite actions found at {relative_file}:{episode.demo}"
                    )

                action_min = np.minimum(action_min, actions.min(axis=0))
                action_max = np.maximum(action_max, actions.max(axis=0))
                action_sum += actions.sum(axis=0)
                action_squared_sum += np.square(actions).sum(axis=0)
                count += actions.shape[0]

    if count == 0:
        raise ValueError("No training actions were found")

    action_mean = action_sum / count
    action_variance = action_squared_sum / count - np.square(action_mean)
    action_std = np.sqrt(np.maximum(action_variance, 0.0))

    stats = ActionStats(
        version=STATS_VERSION,
        split="train",
        count=count,
        action_dim=action_dim,
        action_min=tuple(action_min.tolist()),
        action_max=tuple(action_max.tolist()),
        action_mean=tuple(action_mean.tolist()),
        action_std=tuple(action_std.tolist()),
    )
    stats.validate()
    return stats
