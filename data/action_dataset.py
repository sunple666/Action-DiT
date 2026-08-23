"""PyTorch dataset for LIBERO action-chunk diffusion training."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.libero_index import LiberoSampleIndex, SplitManifest


SplitName = Literal["train", "val"]


class ActionDataset(Dataset):
    """Read time-aligned LIBERO observations and future action chunks."""

    image_mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32
    ).view(3, 1, 1)
    image_std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=torch.float32
    ).view(3, 1, 1)

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        split: SplitName,
        action_chunk: int = 16,
        max_open_files: int = 8,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(
                f"Dataset root does not exist: {self.dataset_root}"
            )

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Split manifest does not exist: {self.manifest_path}"
            )

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if not isinstance(action_chunk, int) or action_chunk <= 0:
            raise ValueError(
                f"action_chunk must be a positive integer, got {action_chunk!r}"
            )
        if not isinstance(max_open_files, int) or max_open_files <= 0:
            raise ValueError(
                "max_open_files must be a positive integer, "
                f"got {max_open_files!r}"
            )

        self.split = split
        self.action_chunk = action_chunk
        self.max_open_files = max_open_files

        manifest = SplitManifest.load(self.manifest_path)
        self.episodes = manifest.episodes(self.split)
        if not self.episodes:
            raise ValueError(f"The {self.split!r} split contains no episodes")

        missing_files = sorted(
            {
                episode.file
                for episode in self.episodes
                if not (self.dataset_root / episode.file).is_file()
            }
        )
        if missing_files:
            missing_preview = ", ".join(missing_files[:3])
            raise FileNotFoundError(
                f"{len(missing_files)} HDF5 file(s) referenced by the manifest "
                f"were not found under {self.dataset_root}: {missing_preview}"
            )

        self.index = LiberoSampleIndex(self.episodes)
        self._open_files: OrderedDict[str, h5py.File] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _get_hdf5_file(self, relative_path: str) -> h5py.File:
        if relative_path in self._open_files:
            handle = self._open_files[relative_path]
            self._open_files.move_to_end(relative_path)
            return handle

        full_path = (self.dataset_root / relative_path).resolve()
        if not full_path.is_relative_to(self.dataset_root):
            raise ValueError(f"HDF5 path escapes dataset root: {relative_path}")
        if not full_path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {full_path}")

        handle = h5py.File(full_path, "r")
        self._open_files[relative_path] = handle

        if len(self._open_files) > self.max_open_files:
            _, oldest_handle = self._open_files.popitem(last=False)
            oldest_handle.close()

        return handle

    def close(self) -> None:
        for handle in self._open_files.values():
            handle.close()
        self._open_files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open_files"] = OrderedDict()
        return state

    @staticmethod
    def _process_image(image: np.ndarray) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a NumPy array, got {type(image).__name__}"
            )
        if image.shape != (128, 128, 3):
            raise ValueError(f"Expected image shape (128, 128, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image, got {image.dtype}")

        image_tensor = torch.from_numpy(image.copy())
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()
        image_tensor = image_tensor.to(dtype=torch.float32).div(255.0)
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return (image_tensor - ActionDataset.image_mean) / ActionDataset.image_std

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample_ref = self.index.locate(idx)
        episode = sample_ref.episode
        timestep = sample_ref.timestep

        file = self._get_hdf5_file(episode.file)
        demo = file["data"][episode.demo]
        obs = demo["obs"]

        ee_state = np.asarray(obs["ee_states"][timestep], dtype=np.float32)
        gripper_state = np.asarray(
            obs["gripper_states"][timestep], dtype=np.float32
        )
        state_array = np.concatenate((ee_state, gripper_state), axis=0)
        if state_array.shape != (8,):
            raise ValueError(
                f"Expected state shape (8,), got {state_array.shape} at "
                f"{episode.file}:{episode.demo}@{timestep}"
            )
        state = torch.from_numpy(state_array)

        image = np.asarray(obs["agentview_rgb"][timestep], dtype=np.uint8)
        observation = self._process_image(image)

        action_end = min(timestep + self.action_chunk, episode.length)
        valid_action_array = np.asarray(
            demo["actions"][timestep:action_end], dtype=np.float32
        )
        if valid_action_array.ndim != 2 or valid_action_array.shape[1] != 7:
            raise ValueError(
                f"Expected valid actions [T, 7], got {valid_action_array.shape} at "
                f"{episode.file}:{episode.demo}@{timestep}"
            )
        valid_actions = torch.from_numpy(valid_action_array)
        valid_length = valid_actions.shape[0]

        action = torch.zeros((self.action_chunk, 7), dtype=torch.float32)
        action[:valid_length] = valid_actions

        action_mask = torch.zeros(self.action_chunk, dtype=torch.bool)
        action_mask[:valid_length] = True

        return {
            "action": action,
            "action_mask": action_mask,
            "state": state,
            "observation": observation,
            "text": episode.instruction,
        }
