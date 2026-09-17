"""Measure whether ActionDiT uses its wrist-camera observation.

For every selected validation observation, this script generates actions with
the same state, language, agent-view image, and initial diffusion noise under
three wrist-camera conditions:

* normal: the matching wrist image;
* zero: a zero-valued normalized image tensor;
* shuffled: a wrist image from a different selected observation.

It reports target-action errors for every condition and action deltas relative
to the normal condition.  Holding the diffusion noise fixed is essential here:
otherwise sampling variation can be mistaken for wrist-camera sensitivity.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from infer_offline import (
    ACTION_NAMES,
    OfflineAlignmentDataset,
    choose_stratified_indices,
    masked_metrics,
)


RUNTIME_STORAGE_ROOT = Path(
    os.environ.get("STORE", str(DEFAULT_STORAGE_ROOT))
).expanduser()
CONDITIONS = ("normal", "zero", "shuffled")


def default_dataset_root() -> Path:
    configured = os.environ.get("DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_STORAGE_ROOT / "datasets" / "libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline wrist-camera ablation with matched diffusion noise."
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
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--results_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "wrist_ablation_val.json",
    )
    args = parser.parse_args()

    if args.num_samples < 2:
        parser.error("--num_samples must be at least 2 for shuffled ablation")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.ddim_steps <= 0:
        parser.error("--ddim_steps must be positive")
    return args


class WristAblationDataset(Dataset):
    """Add a different sample's wrist image to alignment diagnostics."""

    def __init__(self, base: ActionDataset, indices: list[int]) -> None:
        if len(indices) < 2:
            raise ValueError("Wrist ablation needs at least two samples")
        self.base = base
        self.alignment = OfflineAlignmentDataset(base, indices)
        # The selected indices are already randomly ordered.  A one-place
        # rotation is a deterministic derangement, including across batches.
        self.shuffled_indices = indices[1:] + indices[:1]

    def __len__(self) -> int:
        return len(self.alignment)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.alignment[item]
        shuffled_sample = self.base[self.shuffled_indices[item]]
        sample["shuffled_wrist_observation"] = shuffled_sample[
            "wrist_observation"
        ]
        return sample


def prediction_delta(
    reference_normalized: torch.Tensor,
    reference_raw: torch.Tensor,
    ablated_normalized: torch.Tensor,
    ablated_raw: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Summarize output changes while ignoring padded target positions."""
    mask = mask.to(dtype=torch.bool).cpu()
    expanded_mask = mask.unsqueeze(-1)
    valid_steps = int(mask.sum().item())
    if valid_steps == 0:
        raise ValueError("Ablation mask contains no valid action steps")

    normalized_delta = (reference_normalized - ablated_normalized).abs().cpu()
    raw_delta = (reference_raw - ablated_raw).abs().cpu()
    normalized_per_dim = (
        (normalized_delta * expanded_mask).sum(dim=(0, 1)) / valid_steps
    )
    raw_per_dim = (raw_delta * expanded_mask).sum(dim=(0, 1)) / valid_steps
    first_normalized = normalized_delta[:, 0].mean(dim=0)
    first_raw = raw_delta[:, 0].mean(dim=0)

    return {
        "normalized_mean_abs_delta": float(normalized_per_dim.mean().item()),
        "normalized_mean_abs_delta_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, normalized_per_dim.tolist())
        },
        "raw_mean_abs_delta_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, raw_per_dim.tolist())
        },
        "first_action_normalized_mean_abs_delta": float(
            first_normalized.mean().item()
        ),
        "first_action_normalized_mean_abs_delta_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, first_normalized.tolist())
        },
        "first_action_raw_mean_abs_delta_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, first_raw.tolist())
        },
        "first_action_gripper_sign_flip_fraction": float(
            (
                (reference_raw[:, 0, -1] >= 0)
                != (ablated_raw[:, 0, -1] >= 0)
            )
            .float()
            .mean()
            .item()
        ),
        "valid_action_steps": valid_steps,
    }


def condition_metrics(
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
    metrics["prediction_saturation_fraction"] = float(
        (predicted_normalized.abs() >= 0.999).float().mean().item()
    )
    return metrics


def summarize_subset(
    predictions: dict[str, dict[str, torch.Tensor]],
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    normalizer: ActionNormalizer,
) -> dict[str, Any]:
    conditions = {
        name: condition_metrics(
            values["normalized"],
            values["raw"],
            target_raw,
            target_mask,
            normalizer,
        )
        for name, values in predictions.items()
    }
    normal_mae = conditions["normal"]["normalized_mae"]
    normal_first_mae = conditions["normal"]["first_action_normalized_mae"]
    for name in ("zero", "shuffled"):
        conditions[name]["normalized_mae_change_from_normal"] = (
            conditions[name]["normalized_mae"] - normal_mae
        )
        conditions[name]["first_action_mae_change_from_normal"] = (
            conditions[name]["first_action_normalized_mae"] - normal_first_mae
        )

    sensitivities = {
        f"normal_vs_{name}": prediction_delta(
            predictions["normal"]["normalized"],
            predictions["normal"]["raw"],
            predictions[name]["normalized"],
            predictions[name]["raw"],
            target_mask,
        )
        for name in ("zero", "shuffled")
    }
    return {"conditions": conditions, "sensitivities": sensitivities}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Wrist ablation requires a CUDA GPU")
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
    dataset = WristAblationDataset(base_dataset, selected_indices)
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

    noise_seed = args.seed + 1_000_000
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(noise_seed)
    collected: dict[str, list[Any]] = defaultdict(list)
    input_delta_sums = {"zero": 0.0, "shuffled": 0.0}
    input_element_count = 0
    processed = 0

    try:
        for batch in loader:
            states = batch["state"].to(device, non_blocking=True)
            agentview = batch["agentview_observation"].to(
                device, non_blocking=True
            )
            normal_wrist = batch["wrist_observation"].to(
                device, non_blocking=True
            )
            shuffled_wrist = batch["shuffled_wrist_observation"].to(
                device, non_blocking=True
            )
            zero_wrist = torch.zeros_like(normal_wrist)
            texts = list(batch["text"])
            initial_noise = torch.randn(
                states.shape[0],
                ACTION_CHUNK,
                ACTION_DIM,
                device=device,
                dtype=states.dtype,
                generator=noise_generator,
            )

            wrist_inputs = {
                "normal": normal_wrist,
                "zero": zero_wrist,
                "shuffled": shuffled_wrist,
            }
            for condition, wrist in wrist_inputs.items():
                predicted_normalized, predicted_raw = infer_action_chunk(
                    model=model,
                    diffusion=diffusion,
                    normalizer=normalizer,
                    states=states,
                    agentview_observations=agentview,
                    wrist_observations=wrist,
                    language=texts,
                    eta=0.0,
                    amp_dtype=amp_dtype,
                    initial_noise=initial_noise,
                )
                collected[f"{condition}_normalized"].append(
                    predicted_normalized.cpu()
                )
                collected[f"{condition}_raw"].append(predicted_raw.cpu())

            input_delta_sums["zero"] += float(
                (normal_wrist - zero_wrist).abs().sum().item()
            )
            input_delta_sums["shuffled"] += float(
                (normal_wrist - shuffled_wrist).abs().sum().item()
            )
            input_element_count += normal_wrist.numel()
            collected["next_action"].append(batch["next_action"].cpu())
            collected["next_mask"].append(batch["next_mask"].cpu())
            collected["text"].extend(texts)

            processed += states.shape[0]
            print(f"processed {processed}/{len(dataset)} samples", flush=True)
    finally:
        base_dataset.close()

    predictions = {
        condition: {
            "normalized": torch.cat(collected[f"{condition}_normalized"]),
            "raw": torch.cat(collected[f"{condition}_raw"]),
        }
        for condition in CONDITIONS
    }
    target_raw = torch.cat(collected["next_action"])
    target_mask = torch.cat(collected["next_mask"])
    texts = collected["text"]
    overall = summarize_subset(
        predictions,
        target_raw,
        target_mask,
        metric_normalizer,
    )

    per_instruction: dict[str, Any] = {}
    for instruction in sorted(set(texts)):
        subset = torch.tensor(
            [text == instruction for text in texts], dtype=torch.bool
        )
        subset_predictions = {
            condition: {
                key: value[subset] for key, value in values.items()
            }
            for condition, values in predictions.items()
        }
        per_instruction[instruction] = {
            "num_samples": int(subset.sum().item()),
            **summarize_subset(
                subset_predictions,
                target_raw[subset],
                target_mask[subset],
                metric_normalizer,
            ),
        }

    results = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "split": args.split,
        "seed": args.seed,
        "diffusion_noise_seed": noise_seed,
        "num_samples": len(dataset),
        "ddim_steps": args.ddim_steps,
        "precision": args.precision,
        "input_mean_abs_delta": {
            name: value / input_element_count
            for name, value in input_delta_sums.items()
        },
        "overall": overall,
        "per_instruction": per_instruction,
        "interpretation": {
            "ignored_wrist": (
                "Normal-vs-ablated action deltas and MAE changes are both near zero."
            ),
            "useful_wrist": (
                "Zeroing or shuffling increases MAE, and generated actions change."
            ),
            "influential_but_not_useful": (
                "Generated actions change, but ablation does not increase MAE."
            ),
        },
    }

    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    normal = overall["conditions"]["normal"]
    zero = overall["conditions"]["zero"]
    shuffled = overall["conditions"]["shuffled"]
    print("\nWrist-camera ablation summary")
    print(f"  normal normalized MAE:          {normal['normalized_mae']:.6f}")
    print(
        "  zero normalized MAE:            "
        f"{zero['normalized_mae']:.6f} "
        f"(change {zero['normalized_mae_change_from_normal']:+.6f})"
    )
    print(
        "  shuffled normalized MAE:        "
        f"{shuffled['normalized_mae']:.6f} "
        f"(change {shuffled['normalized_mae_change_from_normal']:+.6f})"
    )
    for name in ("zero", "shuffled"):
        sensitivity = overall["sensitivities"][f"normal_vs_{name}"]
        print(
            f"  normal-vs-{name} action delta: "
            f"{sensitivity['normalized_mean_abs_delta']:.6f}"
        )
        print(
            f"  first-action delta ({name}):     "
            f"{sensitivity['first_action_normalized_mean_abs_delta']:.6f}"
        )
    print(f"saved results: {results_path}")


if __name__ == "__main__":
    main()
