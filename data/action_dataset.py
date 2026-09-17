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
        post_transition_steps: int = 4,
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
        if not isinstance(post_transition_steps, int) or post_transition_steps < 0:
            raise ValueError(
                "post_transition_steps must be a non-negative integer, "
                f"got {post_transition_steps!r}"
            )
        if not isinstance(max_open_files, int) or max_open_files <= 0:
            raise ValueError(
                "max_open_files must be a positive integer, "
                f"got {max_open_files!r}"
            )

        self.split = split
        self.action_chunk = action_chunk
        self.post_transition_steps = post_transition_steps
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

        self.index = LiberoSampleIndex(self.episodes,trim_end=1)
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

    def transition_sampling_weights(
        self,
        *,
        transition_window: int,
        oversample_factor: float,
    ) -> tuple[torch.Tensor, int]:
        """Return per-observation weights near gripper-command transitions.

        Dataset item ``t`` predicts ``action[t + 1]``.  An item is considered
        transition-near when that first target action is within
        ``transition_window`` actions of a gripper sign change.  The returned
        tensor is suitable for ``WeightedRandomSampler``.
        """
        if not isinstance(transition_window, int) or transition_window < 0:
            raise ValueError("transition_window must be a non-negative integer")
        if not np.isfinite(oversample_factor) or oversample_factor < 1.0:
            raise ValueError("oversample_factor must be finite and at least 1")

        weights = torch.ones(len(self), dtype=torch.double)
        near_count = 0
        dataset_start = 0
        for episode in self.episodes:
            file = self._get_hdf5_file(episode.file)
            actions = np.asarray(
                file["data"][episode.demo]["actions"], dtype=np.float32
            )
            signs = actions[:, -1] >= 0.0
            transitions = np.flatnonzero(signs[1:] != signs[:-1]) + 1
            target_indices = np.arange(1, episode.length)
            if transitions.size:
                distances = np.abs(
                    target_indices[:, None] - transitions[None, :]
                )
                near_transition = distances.min(axis=1) <= transition_window
            else:
                near_transition = np.zeros(
                    episode.length - 1, dtype=np.bool_
                )

            episode_length = episode.length - 1
            episode_weights = weights[
                dataset_start : dataset_start + episode_length
            ]
            episode_weights[torch.from_numpy(near_transition)] = oversample_factor
            near_count += int(near_transition.sum())
            dataset_start += episode_length

        if dataset_start != len(self):
            raise RuntimeError(
                f"Built {dataset_start} sampling weights for dataset of "
                f"length {len(self)}"
            )
        return weights, near_count

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

        agentview_image = np.asarray(obs["agentview_rgb"][timestep], dtype=np.uint8)
        wrist_image=np.asarray(obs["eye_in_hand_rgb"][timestep], dtype=np.uint8)
        agentview_observation = self._process_image(agentview_image)
        wrist_observation=self._process_image(wrist_image)

        action_start = timestep + 1
        action_end = min(action_start + self.action_chunk, episode.length)
        valid_action_array = np.asarray(
            demo["actions"][action_start:action_end], dtype=np.float32
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

        transition_mask = torch.zeros(self.action_chunk, dtype=torch.bool)
        post_transition_mask = torch.zeros(self.action_chunk, dtype=torch.bool)
        if valid_length:
            history_start = max(
                0,
                action_start - self.post_transition_steps - 1,
            )
            gripper_history = np.asarray(
                demo["actions"][history_start:action_end, -1],
                dtype=np.float32,
            )
            history_positive = gripper_history >= 0.0
            transition_indices = (
                np.flatnonzero(history_positive[1:] != history_positive[:-1])
                + history_start
                + 1
            )
            for transition_index in transition_indices.tolist():
                relative_transition = transition_index - action_start
                if 0 <= relative_transition < valid_length:
                    transition_mask[relative_transition] = True

                if self.post_transition_steps:
                    post_start = max(transition_index + 1, action_start)
                    post_end = min(
                        transition_index + 1 + self.post_transition_steps,
                        action_end,
                    )
                    if post_start < post_end:
                        post_transition_mask[
                            post_start - action_start : post_end - action_start
                        ] = True
            post_transition_mask &= ~transition_mask

        return {
            "action": action,
            "action_mask": action_mask,
            "action_transition_mask": transition_mask,
            "action_post_transition_mask": post_transition_mask,
            "state": state,
            "agentview_observation": agentview_observation,
            "wrist_observation": wrist_observation,
            "text": episode.instruction,
        }
