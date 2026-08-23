"""Portable episode splits and sample indexing for LIBERO HDF5 datasets."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py


SplitName = Literal["train", "val"]
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class EpisodeRef:
    """Metadata needed to locate one demonstration without loading its arrays."""

    file: str
    demo: str
    length: int
    instruction: str


@dataclass(frozen=True)
class SampleRef:
    """The episode and local timestep addressed by one global dataset index."""

    episode: EpisodeRef
    timestep: int


@dataclass(frozen=True)
class SplitManifest:
    """A reproducible train/validation split containing portable relative paths."""

    seed: int
    val_ratio: float
    train: tuple[EpisodeRef, ...]
    val: tuple[EpisodeRef, ...]
    version: int = MANIFEST_VERSION

    def episodes(self, split: SplitName) -> tuple[EpisodeRef, ...]:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        raise ValueError(f"Unknown split: {split!r}")

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "train": [asdict(episode) for episode in self.train],
            "val": [asdict(episode) for episode in self.val],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(payload["version"])
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest version {version}; expected {MANIFEST_VERSION}"
            )
        return cls(
            version=version,
            seed=int(payload["seed"]),
            val_ratio=float(payload["val_ratio"]),
            train=tuple(EpisodeRef(**item) for item in payload["train"]),
            val=tuple(EpisodeRef(**item) for item in payload["val"]),
        )


class LiberoSampleIndex:
    """Map contiguous global sample indices to episode-local timesteps."""

    def __init__(self, episodes: Iterable[EpisodeRef]) -> None:
        self.episodes = tuple(episodes)
        self._cumulative_ends: list[int] = []
        total = 0
        for episode in self.episodes:
            if episode.length <= 0:
                raise ValueError(
                    f"Episode {episode.file}:{episode.demo} has invalid length "
                    f"{episode.length}"
                )
            total += episode.length
            self._cumulative_ends.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def locate(self, index: int) -> SampleRef:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"Sample index {index} is outside [0, {self._length})")

        episode_index = bisect_right(self._cumulative_ends, index)
        episode_start = (
            0 if episode_index == 0 else self._cumulative_ends[episode_index - 1]
        )
        return SampleRef(
            episode=self.episodes[episode_index],
            timestep=index - episode_start,
        )


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", maxsplit=1)[-1]), name
    except ValueError:
        return 0, name


def _decode_instruction(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("The HDF5 problem_info attribute must contain JSON text")
    problem_info = json.loads(value)
    instruction = problem_info.get("language_instruction")
    if isinstance(instruction, list):
        instruction = " ".join(str(part) for part in instruction)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("problem_info does not contain a language instruction")
    return instruction.strip()


def _scan_file(dataset_root: Path, file_path: Path) -> list[EpisodeRef]:
    relative_file = file_path.relative_to(dataset_root).as_posix()
    episodes: list[EpisodeRef] = []

    with h5py.File(file_path, "r") as file:
        if "data" not in file:
            raise ValueError(f"{file_path} has no root 'data' group")
        data = file["data"]
        if "problem_info" not in data.attrs:
            raise ValueError(f"{file_path} has no data/problem_info attribute")
        instruction = _decode_instruction(data.attrs["problem_info"])

        for demo_name in sorted(data.keys(), key=_demo_sort_key):
            demo = data[demo_name]
            if "actions" not in demo or "obs" not in demo:
                raise ValueError(f"{relative_file}:{demo_name} lacks actions or obs")

            actions = demo["actions"]
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(
                    f"{relative_file}:{demo_name} expected actions [T, 7], "
                    f"got {actions.shape}"
                )
            length = int(actions.shape[0])
            if length == 0:
                raise ValueError(f"{relative_file}:{demo_name} is empty")

            obs = demo["obs"]
            required_obs = {
                "agentview_rgb": (128, 128, 3),
                "ee_states": (6,),
                "gripper_states": (2,),
            }
            for key, trailing_shape in required_obs.items():
                if key not in obs:
                    raise ValueError(f"{relative_file}:{demo_name} lacks obs/{key}")
                expected_shape = (length, *trailing_shape)
                if obs[key].shape != expected_shape:
                    raise ValueError(
                        f"{relative_file}:{demo_name}/obs/{key} expected "
                        f"{expected_shape}, got {obs[key].shape}"
                    )

            episodes.append(
                EpisodeRef(
                    file=relative_file,
                    demo=demo_name,
                    length=length,
                    instruction=instruction,
                )
            )

    return episodes


def _split_one_file(
    episodes: list[EpisodeRef], val_ratio: float, seed: int
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    if len(episodes) < 2:
        raise ValueError(
            f"At least two demos are required to create train/val splits; "
            f"found {len(episodes)} in {episodes[0].file if episodes else 'an empty file'}"
        )

    num_val = round(len(episodes) * val_ratio)
    num_val = min(max(num_val, 1), len(episodes) - 1)

    def rank(episode: EpisodeRef) -> bytes:
        identity = f"{seed}\0{episode.file}\0{episode.demo}".encode("utf-8")
        return hashlib.sha256(identity).digest()

    validation_keys = {
        (episode.file, episode.demo)
        for episode in sorted(episodes, key=rank)[:num_val]
    }
    train = [
        episode
        for episode in episodes
        if (episode.file, episode.demo) not in validation_keys
    ]
    val = [
        episode
        for episode in episodes
        if (episode.file, episode.demo) in validation_keys
    ]
    return train, val


def build_split_manifest(
    dataset_root: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> SplitManifest:
    """Scan all HDF5 files and split the demos independently within each task."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")
    files = sorted(root.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {root}")

    train: list[EpisodeRef] = []
    val: list[EpisodeRef] = []
    for file_path in files:
        file_episodes = _scan_file(root, file_path)
        file_train, file_val = _split_one_file(file_episodes, val_ratio, seed)
        train.extend(file_train)
        val.extend(file_val)

    return SplitManifest(
        seed=seed,
        val_ratio=val_ratio,
        train=tuple(train),
        val=tuple(val),
    )
