"""Measure ActionDiT errors around gripper-transition phases.

The ordinary offline MAE is dominated by the many frames in which the robot is
only moving through free space.  This diagnostic balances evaluation samples
across five phases defined by the *first target action* (action[t + 1]):

* ``pre_gripper_transition``: within N actions before the closest transition;
* ``at_gripper_transition``: the target action changes gripper sign;
* ``post_gripper_transition``: within N actions after the closest transition;
* ``high_translation_motion``: far from a transition and above a data-derived
  translation-magnitude percentile;
* ``regular_motion``: all remaining actions.

The transition direction is reported as ``to_positive`` or ``to_negative``.
This deliberately avoids assuming which sign means open/closed for a particular
LIBERO controller configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
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
    infer_action_ensemble,
    load_model_weights,
    precision_dtype,
    require_directory,
    require_file,
    seed_everything,
)
from infer_offline import ACTION_NAMES, masked_metrics


RUNTIME_STORAGE_ROOT = Path(
    os.environ.get("STORE", str(DEFAULT_STORAGE_ROOT))
).expanduser()
PHASES = (
    "pre_gripper_transition",
    "at_gripper_transition",
    "post_gripper_transition",
    "high_translation_motion",
    "regular_motion",
)


def default_dataset_root() -> Path:
    configured = os.environ.get("DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_STORAGE_ROOT / "datasets" / "libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report validation action errors around gripper transitions."
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
    parser.add_argument("--dino_weights", type=Path, default=default_dino_weights())
    parser.add_argument(
        "--qwen_model_path", type=Path, default=default_qwen_model_path()
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--samples_per_phase",
        type=int,
        default=40,
        help="Maximum samples selected for each phase.",
    )
    parser.add_argument(
        "--transition_window",
        type=int,
        default=8,
        help="Number of actions before/after a gripper transition.",
    )
    parser.add_argument(
        "--high_motion_percentile",
        type=float,
        default=75.0,
        help=(
            "Dataset percentile of raw xyz action L2 magnitude used to define "
            "high_translation_motion."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--num_action_samples", type=int, default=1)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--results_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "contact_phase_error_val.json",
    )
    args = parser.parse_args()

    if args.samples_per_phase <= 0:
        parser.error("--samples_per_phase must be positive")
    if args.transition_window <= 0:
        parser.error("--transition_window must be positive")
    if not 0.0 < args.high_motion_percentile < 100.0:
        parser.error("--high_motion_percentile must be in (0, 100)")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.ddim_steps <= 0:
        parser.error("--ddim_steps must be positive")
    if args.num_action_samples <= 0:
        parser.error("--num_action_samples must be positive")
    return args


@dataclass(frozen=True)
class PhaseCandidate:
    dataset_index: int
    phase: str
    transition_direction: str
    transition_offset: int


def _transition_direction(gripper_value: float) -> str:
    return "to_positive" if gripper_value >= 0.0 else "to_negative"


def build_phase_candidates(
    dataset: ActionDataset,
    transition_window: int,
    high_motion_percentile: float,
) -> tuple[dict[str, list[PhaseCandidate]], float]:
    """Classify every eligible first target action without loading images."""
    episode_actions: list[np.ndarray] = []
    translation_magnitudes: list[np.ndarray] = []

    for episode in dataset.episodes:
        file = dataset._get_hdf5_file(episode.file)
        actions = np.asarray(
            file["data"][episode.demo]["actions"], dtype=np.float32
        )
        if actions.shape != (episode.length, ACTION_DIM):
            raise ValueError(
                f"Unexpected actions shape at {episode.file}:{episode.demo}: "
                f"{actions.shape}"
            )
        episode_actions.append(actions)
        # Dataset observation t predicts action t+1, so action 0 is ineligible.
        translation_magnitudes.append(
            np.linalg.norm(actions[1:, :3], axis=1)
        )

    all_magnitudes = np.concatenate(translation_magnitudes)
    high_motion_threshold = float(
        np.percentile(all_magnitudes, high_motion_percentile)
    )

    by_phase: dict[str, list[PhaseCandidate]] = {
        phase: [] for phase in PHASES
    }
    dataset_start = 0
    for episode, actions, magnitudes in zip(
        dataset.episodes, episode_actions, translation_magnitudes
    ):
        signs = actions[:, -1] >= 0.0
        transitions = np.flatnonzero(signs[1:] != signs[:-1]) + 1

        for local_offset, (action_index, magnitude) in enumerate(
            zip(range(1, episode.length), magnitudes)
        ):
            phase = "regular_motion"
            direction = "none"
            transition_offset = 0

            if transitions.size:
                distances = transitions - action_index
                nearest_position = int(np.argmin(np.abs(distances)))
                nearest_transition = int(transitions[nearest_position])
                transition_offset = int(distances[nearest_position])
                if abs(transition_offset) <= transition_window:
                    direction = _transition_direction(
                        float(actions[nearest_transition, -1])
                    )
                    if transition_offset > 0:
                        phase = "pre_gripper_transition"
                    elif transition_offset < 0:
                        phase = "post_gripper_transition"
                    else:
                        phase = "at_gripper_transition"

            if phase == "regular_motion" and magnitude >= high_motion_threshold:
                phase = "high_translation_motion"
                transition_offset = 0

            by_phase[phase].append(
                PhaseCandidate(
                    dataset_index=dataset_start + local_offset,
                    phase=phase,
                    transition_direction=direction,
                    transition_offset=transition_offset,
                )
            )

        dataset_start += episode.length - 1

    if dataset_start != len(dataset):
        raise RuntimeError(
            f"Phase indexing produced {dataset_start} samples, dataset has "
            f"{len(dataset)}"
        )
    return by_phase, high_motion_threshold


def _balanced_sample(
    candidates: list[PhaseCandidate],
    dataset: ActionDataset,
    limit: int,
    rng: random.Random,
) -> list[PhaseCandidate]:
    """Sample a phase approximately evenly across language instructions."""
    if len(candidates) <= limit:
        selected = list(candidates)
        rng.shuffle(selected)
        return selected

    by_instruction: dict[str, list[PhaseCandidate]] = defaultdict(list)
    for candidate in candidates:
        instruction = dataset.index.locate(
            candidate.dataset_index
        ).episode.instruction
        by_instruction[instruction].append(candidate)

    instructions = sorted(by_instruction)
    allocation = {
        instruction: limit // len(instructions) for instruction in instructions
    }
    for instruction in instructions[: limit % len(instructions)]:
        allocation[instruction] += 1

    selected: list[PhaseCandidate] = []
    for instruction in instructions:
        pool = by_instruction[instruction]
        take = min(allocation[instruction], len(pool))
        selected.extend(rng.sample(pool, take))

    if len(selected) < limit:
        selected_indices = {item.dataset_index for item in selected}
        remaining = [
            item for item in candidates if item.dataset_index not in selected_indices
        ]
        selected.extend(rng.sample(remaining, limit - len(selected)))

    rng.shuffle(selected)
    return selected


class ContactPhaseDataset(Dataset):
    def __init__(
        self, base: ActionDataset, candidates: list[PhaseCandidate]
    ) -> None:
        self.base = base
        self.candidates = candidates

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, item: int) -> dict[str, Any]:
        candidate = self.candidates[item]
        sample = self.base[candidate.dataset_index]
        sample_ref = self.base.index.locate(candidate.dataset_index)
        return {
            **sample,
            "dataset_index": candidate.dataset_index,
            "phase": candidate.phase,
            "transition_direction": candidate.transition_direction,
            "transition_offset": candidate.transition_offset,
            "timestep": sample_ref.timestep,
            "episode": f"{sample_ref.episode.file}:{sample_ref.episode.demo}",
        }


def summarize_group(
    predicted_normalized: torch.Tensor,
    predicted_raw: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    normalizer: ActionNormalizer,
) -> dict[str, Any]:
    metrics = masked_metrics(
        predicted_normalized,
        predicted_raw,
        target_raw,
        target_mask,
        normalizer,
    )
    target_normalized = normalizer.normalize(target_raw, clamp=True).cpu()
    predicted_normalized = predicted_normalized.cpu()
    predicted_raw = predicted_raw.cpu()
    target_raw = target_raw.cpu()

    first_normalized_error = predicted_normalized[:, 0] - target_normalized[:, 0]
    first_raw_error = predicted_raw[:, 0] - target_raw[:, 0]
    target_positive = target_raw[:, 0, -1] >= 0.0
    predicted_positive = predicted_raw[:, 0, -1] >= 0.0

    return {
        "num_samples": int(target_raw.shape[0]),
        "metrics": metrics,
        "first_action_normalized_translation_l2": float(
            torch.linalg.vector_norm(first_normalized_error[:, :3], dim=-1)
            .mean()
            .item()
        ),
        "first_action_raw_translation_l2": float(
            torch.linalg.vector_norm(first_raw_error[:, :3], dim=-1).mean().item()
        ),
        "first_action_normalized_rotation_l2": float(
            torch.linalg.vector_norm(first_normalized_error[:, 3:6], dim=-1)
            .mean()
            .item()
        ),
        "first_action_raw_rotation_l2": float(
            torch.linalg.vector_norm(first_raw_error[:, 3:6], dim=-1).mean().item()
        ),
        "first_action_gripper_confusion": {
            "target_negative_predicted_negative": int(
                ((~target_positive) & (~predicted_positive)).sum().item()
            ),
            "target_negative_predicted_positive": int(
                ((~target_positive) & predicted_positive).sum().item()
            ),
            "target_positive_predicted_negative": int(
                (target_positive & (~predicted_positive)).sum().item()
            ),
            "target_positive_predicted_positive": int(
                (target_positive & predicted_positive).sum().item()
            ),
        },
        "target_first_action_raw_mean_per_dim": {
            name: float(value)
            for name, value in zip(
                ACTION_NAMES, target_raw[:, 0].mean(dim=0).tolist()
            )
        },
        "predicted_first_action_raw_mean_per_dim": {
            name: float(value)
            for name, value in zip(
                ACTION_NAMES, predicted_raw[:, 0].mean(dim=0).tolist()
            )
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Contact-phase inference requires a CUDA GPU")
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
    candidates_by_phase, high_motion_threshold = build_phase_candidates(
        base_dataset,
        args.transition_window,
        args.high_motion_percentile,
    )
    rng = random.Random(args.seed)
    selected: list[PhaseCandidate] = []
    selected_counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    for phase in PHASES:
        available_counts[phase] = len(candidates_by_phase[phase])
        phase_selection = _balanced_sample(
            candidates_by_phase[phase],
            base_dataset,
            args.samples_per_phase,
            rng,
        )
        selected_counts[phase] = len(phase_selection)
        selected.extend(phase_selection)

    if not selected:
        base_dataset.close()
        raise RuntimeError("No eligible contact-phase samples were found")
    rng.shuffle(selected)
    dataset = ContactPhaseDataset(base_dataset, selected)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("available samples by phase:")
    for phase in PHASES:
        print(
            f"  {phase}: {available_counts[phase]} available, "
            f"{selected_counts[phase]} selected"
        )
    print(f"raw xyz high-motion threshold: {high_motion_threshold:.6f}")

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
            agentview = batch["agentview_observation"].to(
                device, non_blocking=True
            )
            wrist = batch["wrist_observation"].to(device, non_blocking=True)
            texts = list(batch["text"])
            predicted_normalized, predicted_raw = infer_action_ensemble(
                model=model,
                diffusion=diffusion,
                normalizer=normalizer,
                states=states,
                agentview_observations=agentview,
                wrist_observations=wrist,
                language=texts,
                eta=0.0,
                amp_dtype=amp_dtype,
                num_action_samples=args.num_action_samples,
            )

            collected["predicted_normalized"].append(
                predicted_normalized.cpu()
            )
            collected["predicted_raw"].append(predicted_raw.cpu())
            collected["target_raw"].append(batch["action"].cpu())
            collected["target_mask"].append(batch["action_mask"].cpu())
            for key in ("phase", "transition_direction", "text"):
                collected[key].extend(list(batch[key]))

            processed += states.shape[0]
            print(f"processed {processed}/{len(dataset)} samples", flush=True)
    finally:
        base_dataset.close()

    predicted_normalized = torch.cat(collected["predicted_normalized"])
    predicted_raw = torch.cat(collected["predicted_raw"])
    target_raw = torch.cat(collected["target_raw"])
    target_mask = torch.cat(collected["target_mask"])
    phases = collected["phase"]
    directions = collected["transition_direction"]
    texts = collected["text"]

    def group_summary(selector: torch.Tensor) -> dict[str, Any]:
        return summarize_group(
            predicted_normalized[selector],
            predicted_raw[selector],
            target_raw[selector],
            target_mask[selector],
            metric_normalizer,
        )

    overall_selector = torch.ones(len(dataset), dtype=torch.bool)
    per_phase: dict[str, Any] = {}
    for phase in PHASES:
        selector = torch.tensor([value == phase for value in phases])
        if selector.any():
            per_phase[phase] = {
                "num_available": available_counts[phase],
                **group_summary(selector),
            }
        else:
            per_phase[phase] = {
                "num_available": available_counts[phase],
                "num_samples": 0,
            }

    per_transition_direction: dict[str, Any] = {}
    for direction in ("to_negative", "to_positive"):
        selector = torch.tensor([value == direction for value in directions])
        if selector.any():
            per_transition_direction[direction] = group_summary(selector)

    per_instruction: dict[str, Any] = {}
    for instruction in sorted(set(texts)):
        selector = torch.tensor([value == instruction for value in texts])
        per_instruction[instruction] = group_summary(selector)

    results = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "split": args.split,
        "seed": args.seed,
        "ddim_steps": args.ddim_steps,
        "num_action_samples": args.num_action_samples,
        "precision": args.precision,
        "transition_window": args.transition_window,
        "high_motion_percentile": args.high_motion_percentile,
        "high_motion_raw_xyz_l2_threshold": high_motion_threshold,
        "phase_definition": (
            "Phases are assigned from the first aligned target action "
            "action[t+1]. Pre/at/post use the closest gripper-sign transition; "
            "high motion applies only outside the transition window."
        ),
        "available_samples_by_phase": available_counts,
        "selected_samples_by_phase": selected_counts,
        "overall_balanced_selection": group_summary(overall_selector),
        "per_phase": per_phase,
        "per_transition_direction": per_transition_direction,
        "per_instruction": per_instruction,
        "interpretation_note": (
            "Compare first-action errors across phases rather than treating the "
            "balanced overall value as a natural dataset average. A much larger "
            "pre/at/post-transition error supports contact-focused sampling or "
            "loss weighting. This remains teacher-forced offline evaluation and "
            "does not measure recovery from closed-loop state drift."
        ),
    }

    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nContact-phase first-action summary")
    print("  phase                         n   norm MAE   xyz L2   gripper")
    for phase in PHASES:
        summary = per_phase[phase]
        if summary["num_samples"] == 0:
            print(f"  {phase:29s}  0   n/a        n/a      n/a")
            continue
        metrics = summary["metrics"]
        print(
            f"  {phase:29s} "
            f"{summary['num_samples']:3d}   "
            f"{metrics['first_action_normalized_mae']:.6f}   "
            f"{summary['first_action_normalized_translation_l2']:.6f}   "
            f"{metrics['first_action_gripper_sign_accuracy']:.1%}"
        )
    print(f"saved results: {results_path}")


if __name__ == "__main__":
    main()
