"""Offline action-alignment diagnostics for a trained ActionDiT checkpoint.

This script samples held-out LIBERO observations, generates action chunks with
the same DDIM path used by ``infer.py``, and compares those predictions against
two candidate targets:

* current alignment: observation[t] -> actions[t:t + H]
* next alignment:    observation[t] -> actions[t + 1:t + 1 + H]

The comparison describes which correspondence the trained model learned.  It
cannot, by itself, prove which action should causally follow an observation;
that final question must be confirmed with demonstration replay or retraining.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.action_dataset import ActionDataset
from data.action_normalizer import ActionNormalizer
from diffusion import create_diffusion
from infer import (
    ACTION_CHUNK,
    ACTION_DIM,
    DEFAULT_STORAGE_ROOT,
    PROJECT_ROOT,
    build_model,
    default_dino_repo,
    default_dino_weights,
    default_qwen_model_path,
    infer_action_chunk,
    load_model_weights,
    precision_dtype,
    require_directory,
    require_file,
    seed_everything,
)


RUNTIME_STORAGE_ROOT = Path(
    os.environ.get("STORE", str(DEFAULT_STORAGE_ROOT))
).expanduser()
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def default_dataset_root() -> Path:
    configured = os.environ.get("DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_STORAGE_ROOT / "datasets" / "libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated validation actions with action[t] and "
            "action[t+1] targets."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "libero_goal_split.json",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "libero_goal_action_stats.json",
    )
    parser.add_argument("--dino_repo", type=Path, default=default_dino_repo())
    parser.add_argument(
        "--dino_weights", type=Path, default=default_dino_weights()
    )
    parser.add_argument(
        "--qwen_model_path", type=Path, default=default_qwen_model_path()
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of eligible frames, stratified across instructions.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--results_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiment_c_offline_alignment.json",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num_samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.ddim_steps <= 0:
        parser.error("--ddim_steps must be positive")
    return args


class OfflineAlignmentDataset(Dataset):
    """Expose current and one-step-shifted targets for selected observations."""

    def __init__(self, base: ActionDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        dataset_index = self.indices[item]
        sample = self.base[dataset_index]
        sample_ref = self.base.index.locate(dataset_index)
        episode = sample_ref.episode
        timestep = sample_ref.timestep

        # Eligible indices always have at least one action after timestep.
        if timestep + 1 >= episode.length:
            raise RuntimeError("Selected an ineligible final demonstration frame")

        file = self.base._get_hdf5_file(episode.file)
        actions = file["data"][episode.demo]["actions"]

        current_end = min(timestep + ACTION_CHUNK, episode.length)
        current_array = np.asarray(
            actions[timestep:current_end], dtype=np.float32
        )
        current_length = int(current_array.shape[0])
        current_action = torch.zeros(ACTION_CHUNK, ACTION_DIM, dtype=torch.float32)
        current_action[:current_length] = torch.from_numpy(current_array)
        current_mask = torch.zeros(ACTION_CHUNK, dtype=torch.bool)
        current_mask[:current_length] = True

        next_end = min(timestep + 1 + ACTION_CHUNK, episode.length)
        next_array = np.asarray(
            actions[timestep + 1 : next_end], dtype=np.float32
        )
        next_length = int(next_array.shape[0])

        next_action = torch.zeros(ACTION_CHUNK, ACTION_DIM, dtype=torch.float32)
        next_action[:next_length] = torch.from_numpy(next_array)
        next_mask = torch.zeros(ACTION_CHUNK, dtype=torch.bool)
        next_mask[:next_length] = True

        return {
            "state": sample["state"],
            "agentview_observation": sample["agentview_observation"],
            "wrist_observation": sample["wrist_observation"],
            "text": sample["text"],
            "current_action": current_action,
            "current_mask": current_mask,
            "next_action": next_action,
            "next_mask": next_mask,
            "dataset_index": dataset_index,
            "timestep": timestep,
            "episode": f"{episode.file}:{episode.demo}",
        }


def choose_stratified_indices(
    dataset: ActionDataset,
    num_samples: int,
    seed: int,
) -> list[int]:
    by_instruction: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        sample_ref = dataset.index.locate(index)
        if sample_ref.timestep + 1 < sample_ref.episode.length:
            by_instruction[sample_ref.episode.instruction].append(index)

    if not by_instruction:
        raise RuntimeError("No frames have a valid action[t+1] target")

    total_eligible = sum(len(indices) for indices in by_instruction.values())
    if num_samples > total_eligible:
        raise ValueError(
            f"Requested {num_samples} samples, but only {total_eligible} are eligible"
        )

    rng = random.Random(seed)
    instructions = sorted(by_instruction)
    allocation = {instruction: num_samples // len(instructions) for instruction in instructions}
    for instruction in instructions[: num_samples % len(instructions)]:
        allocation[instruction] += 1

    selected: list[int] = []
    deficit = 0
    for instruction in instructions:
        available = by_instruction[instruction]
        requested = allocation[instruction]
        take = min(requested, len(available))
        selected.extend(rng.sample(available, take))
        deficit += requested - take

    if deficit:
        selected_set = set(selected)
        remaining = [
            index
            for instruction in instructions
            for index in by_instruction[instruction]
            if index not in selected_set
        ]
        selected.extend(rng.sample(remaining, deficit))

    rng.shuffle(selected)
    return selected


def masked_metrics(
    predicted_normalized: torch.Tensor,
    predicted_raw: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    normalizer: ActionNormalizer,
) -> dict[str, Any]:
    target_normalized = normalizer.normalize(target_raw, clamp=True).cpu()
    mask = target_mask.to(dtype=torch.bool).cpu()
    predicted_normalized = predicted_normalized.cpu()
    predicted_raw = predicted_raw.cpu()
    target_raw = target_raw.cpu()

    normalized_error = (predicted_normalized - target_normalized).abs()
    raw_error = (predicted_raw - target_raw).abs()
    expanded_mask = mask.unsqueeze(-1)
    valid_steps = int(mask.sum().item())
    if valid_steps == 0:
        raise ValueError("Metric mask contains no valid action steps")

    normalized_per_dim = (
        (normalized_error * expanded_mask).sum(dim=(0, 1)) / valid_steps
    )
    raw_per_dim = (raw_error * expanded_mask).sum(dim=(0, 1)) / valid_steps

    first_normalized = normalized_error[:, 0].mean(dim=0)
    first_raw = raw_error[:, 0].mean(dim=0)
    predicted_gripper_open = predicted_raw[:, 0, -1] >= 0
    target_gripper_open = target_raw[:, 0, -1] >= 0

    return {
        "normalized_mae": float(normalized_per_dim.mean().item()),
        "normalized_mae_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, normalized_per_dim.tolist())
        },
        "raw_mae_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, raw_per_dim.tolist())
        },
        "first_action_normalized_mae": float(first_normalized.mean().item()),
        "first_action_normalized_mae_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, first_normalized.tolist())
        },
        "first_action_raw_mae_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, first_raw.tolist())
        },
        "first_action_gripper_sign_accuracy": float(
            (predicted_gripper_open == target_gripper_open).float().mean().item()
        ),
        "valid_action_steps": valid_steps,
    }


def summarize_subset(
    predicted_normalized: torch.Tensor,
    predicted_raw: torch.Tensor,
    current_raw: torch.Tensor,
    current_mask: torch.Tensor,
    next_raw: torch.Tensor,
    next_mask: torch.Tensor,
    normalizer: ActionNormalizer,
) -> dict[str, Any]:
    current = masked_metrics(
        predicted_normalized,
        predicted_raw,
        current_raw,
        current_mask,
        normalizer,
    )
    following = masked_metrics(
        predicted_normalized,
        predicted_raw,
        next_raw,
        next_mask,
        normalizer,
    )

    current_normalized = normalizer.normalize(current_raw, clamp=True).cpu()
    next_normalized = normalizer.normalize(next_raw, clamp=True).cpu()
    common_mask = (current_mask & next_mask).cpu().unsqueeze(-1)
    common_steps = int(common_mask[..., 0].sum().item())
    target_step_change = (
        ((current_normalized - next_normalized).abs() * common_mask).sum()
        / (common_steps * ACTION_DIM)
    )

    current_mae = current["normalized_mae"]
    next_mae = following["normalized_mae"]
    preferred = "current_action_t" if current_mae < next_mae else "next_action_t_plus_1"

    return {
        "current_alignment": current,
        "next_alignment": following,
        "preferred_by_lower_normalized_mae": preferred,
        "next_minus_current_normalized_mae": next_mae - current_mae,
        "target_t_to_t_plus_1_normalized_change": float(target_step_change.item()),
        "prediction_saturation_fraction": float(
            (predicted_normalized.abs() >= 0.999).float().mean().item()
        ),
        "predicted_raw_mean_per_dim": {
            name: float(value)
            for name, value in zip(
                ACTION_NAMES, predicted_raw.mean(dim=(0, 1)).tolist()
            )
        },
        "predicted_raw_std_per_dim": {
            name: float(value)
            for name, value in zip(
                ACTION_NAMES, predicted_raw.std(dim=(0, 1)).tolist()
            )
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Offline ActionDiT inference requires a CUDA GPU")
    device = torch.device("cuda")
    amp_dtype = precision_dtype(args.precision)

    checkpoint_path = require_file(args.checkpoint, "checkpoint")
    dataset_root = require_directory(args.dataset_root, "dataset root")
    manifest_path = require_file(args.manifest_path, "split manifest")
    stats_path = require_file(args.stats_path, "action statistics")
    dino_repo = require_directory(args.dino_repo, "DINOv2 repository")
    dino_weights = require_file(args.dino_weights, "DINOv2 weights")
    qwen_model_path = require_directory(args.qwen_model_path, "Qwen model")

    base_dataset = ActionDataset(
        dataset_root,
        manifest_path,
        args.split,
        action_chunk=ACTION_CHUNK,
    )
    selected_indices = choose_stratified_indices(
        base_dataset,
        args.num_samples,
        args.seed,
    )
    dataset = OfflineAlignmentDataset(base_dataset, selected_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        device=device,
        dino_repo=dino_repo,
        dino_weights=dino_weights,
        qwen_model_path=qwen_model_path,
        amp_dtype=amp_dtype,
    )
    load_model_weights(model, checkpoint_path)
    model.eval()

    diffusion = create_diffusion(
        timestep_respacing=f"ddim{args.ddim_steps}",
        noise_schedule="linear",
        diffusion_steps=1000,
        learn_sigma=True,
    )
    normalizer = ActionNormalizer.from_stats_file(stats_path).to(device)
    metric_normalizer = ActionNormalizer.from_stats_file(stats_path)

    collected: dict[str, list[Any]] = defaultdict(list)
    processed = 0
    try:
        for batch in loader:
            states = batch["state"].to(device, non_blocking=True)
            agentview_observations = batch["agentview_observation"].to(
                device, non_blocking=True
            )
            wrist_observations = batch["wrist_observation"].to(
                device, non_blocking=True
            )
            texts = list(batch["text"])

            predicted_normalized, predicted_raw = infer_action_chunk(
                model=model,
                diffusion=diffusion,
                normalizer=normalizer,
                states=states,
                agentview_observations=agentview_observations,
                wrist_observations=wrist_observations,
                language=texts,
                eta=0.0,
                amp_dtype=amp_dtype,
            )

            collected["predicted_normalized"].append(predicted_normalized.cpu())
            collected["predicted_raw"].append(predicted_raw.cpu())
            for key in ("current_action", "current_mask", "next_action", "next_mask"):
                collected[key].append(batch[key].cpu())
            collected["text"].extend(texts)

            processed += states.shape[0]
            print(f"processed {processed}/{len(dataset)} samples", flush=True)
    finally:
        base_dataset.close()

    predicted_normalized = torch.cat(collected["predicted_normalized"])
    predicted_raw = torch.cat(collected["predicted_raw"])
    current_raw = torch.cat(collected["current_action"])
    current_mask = torch.cat(collected["current_mask"])
    next_raw = torch.cat(collected["next_action"])
    next_mask = torch.cat(collected["next_mask"])
    texts = collected["text"]

    overall = summarize_subset(
        predicted_normalized,
        predicted_raw,
        current_raw,
        current_mask,
        next_raw,
        next_mask,
        metric_normalizer,
    )

    per_instruction: dict[str, Any] = {}
    for instruction in sorted(set(texts)):
        subset = torch.tensor(
            [text == instruction for text in texts],
            dtype=torch.bool,
        )
        per_instruction[instruction] = {
            "num_samples": int(subset.sum().item()),
            **summarize_subset(
                predicted_normalized[subset],
                predicted_raw[subset],
                current_raw[subset],
                current_mask[subset],
                next_raw[subset],
                next_mask[subset],
                metric_normalizer,
            ),
        }

    results = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "split": args.split,
        "seed": args.seed,
        "num_samples": len(dataset),
        "ddim_steps": args.ddim_steps,
        "precision": args.precision,
        "overall": overall,
        "per_instruction": per_instruction,
        "interpretation_note": (
            "Lower MAE identifies which indexed target the trained model more closely "
            "reproduces; it does not alone establish causal control alignment. If the "
            "target t-to-t+1 change is much smaller than both prediction MAEs, this "
            "diagnostic is inconclusive because adjacent actions are too similar."
        ),
    }

    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nOffline action-alignment summary")
    print(
        "  current action[t] normalized MAE: "
        f"{overall['current_alignment']['normalized_mae']:.6f}"
    )
    print(
        "  next action[t+1] normalized MAE:  "
        f"{overall['next_alignment']['normalized_mae']:.6f}"
    )
    print(
        "  first-action current/next MAE:     "
        f"{overall['current_alignment']['first_action_normalized_mae']:.6f} / "
        f"{overall['next_alignment']['first_action_normalized_mae']:.6f}"
    )
    print(
        "  target t -> t+1 normalized change: "
        f"{overall['target_t_to_t_plus_1_normalized_change']:.6f}"
    )
    print(
        "  lower-error alignment:             "
        f"{overall['preferred_by_lower_normalized_mae']}"
    )
    print(
        "  prediction saturation fraction:    "
        f"{overall['prediction_saturation_fraction']:.2%}"
    )
    print(f"saved results: {results_path}")


if __name__ == "__main__":
    main()
