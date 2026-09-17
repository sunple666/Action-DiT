"""Diagnose ActionDiT sampling variance with multi-noise ensembles.

For each held-out observation, this script draws several independent initial
DDIM noises while keeping the observation, language, and eta fixed.  It then
compares prefix ensembles (by default K=1,2,4,8), a medoid sample, and an
oracle best-of-K upper bound against the recorded next-action chunk.

The oracle uses the target action and is diagnostic only.  It must never be
used for online policy evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

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


def default_dataset_root() -> Path:
    configured = os.environ.get("DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_STORAGE_ROOT / "datasets" / "libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-noise ActionDiT action ensembles."
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
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=(
            "Base observation batch size. The effective inference batch is "
            "batch_size * max(sample_counts). Reduce this if CUDA runs out "
            "of memory."
        ),
    )
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--sample_counts",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
        help="Ensemble prefix sizes to evaluate.",
    )
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--results_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "noise_ensemble_val.json",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num_samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.ddim_steps <= 0:
        parser.error("--ddim_steps must be positive")
    if not args.sample_counts or any(value <= 0 for value in args.sample_counts):
        parser.error("--sample_counts must contain positive integers")
    if sorted(set(args.sample_counts)) != list(args.sample_counts):
        parser.error("--sample_counts must be unique and strictly increasing")
    return args


def repeat_language(texts: list[str], repeats: int) -> list[str]:
    return [text for text in texts for _ in range(repeats)]


def gather_candidates(
    candidates: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """Select one complete action chunk per observation from [K,N,H,A]."""
    observation_indices = torch.arange(candidates.shape[1])
    return candidates[candidate_indices, observation_indices]


def ensemble_predictions(
    samples: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    normalizer: ActionNormalizer,
) -> dict[str, torch.Tensor]:
    """Build deployable ensembles and a target-dependent oracle."""
    continuous_mean = samples.mean(dim=0)

    majority_gripper = continuous_mean.clone()
    positive_fraction = (samples[..., -1] >= 0).float().mean(dim=0)
    majority_gripper[..., -1] = torch.where(
        positive_fraction >= 0.5,
        torch.ones_like(positive_fraction),
        -torch.ones_like(positive_fraction),
    )

    # Select the real sampled chunk closest to the ensemble mean. This keeps
    # the result on the model's sampled action manifold.
    medoid_distance = (
        (samples - continuous_mean.unsqueeze(0)).square().mean(dim=(2, 3))
    )
    medoid = gather_candidates(samples, medoid_distance.argmin(dim=0))

    # Diagnostic upper bound: target actions are used to choose the sample.
    target_normalized = normalizer.normalize(target_raw, clamp=True).cpu()
    mask = target_mask.to(dtype=torch.bool).cpu().unsqueeze(0).unsqueeze(-1)
    valid_steps = target_mask.to(dtype=torch.bool).sum(dim=1).clamp_min(1)
    oracle_error = (
        ((samples - target_normalized.unsqueeze(0)).abs() * mask).sum(dim=(2, 3))
        / (valid_steps.unsqueeze(0) * ACTION_DIM)
    )
    oracle = gather_candidates(samples, oracle_error.argmin(dim=0))

    return {
        "continuous_mean": continuous_mean,
        "mean_with_majority_gripper": majority_gripper,
        "medoid": medoid,
        "oracle_best_of_k": oracle,
    }


def metrics_for_normalized_prediction(
    prediction: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    normalizer: ActionNormalizer,
) -> dict[str, Any]:
    prediction = prediction.clamp(-1, 1).cpu()
    prediction_raw = normalizer.denormalize(prediction)
    return masked_metrics(
        prediction,
        prediction_raw,
        target_raw,
        target_mask,
        normalizer,
    )


def sampling_variance(
    samples: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, Any]:
    """Average standard deviation across independent initial noises."""
    std = samples.std(dim=0, unbiased=False)
    mask = target_mask.to(dtype=torch.bool).cpu()
    expanded_mask = mask.unsqueeze(-1)
    valid_steps = int(mask.sum().item())
    per_dim = (std * expanded_mask).sum(dim=(0, 1)) / valid_steps
    valid_by_horizon = mask.sum(dim=0)
    safe_counts = valid_by_horizon.clamp_min(1)
    by_horizon = (
        (std * expanded_mask).sum(dim=(0, 2))
        / (safe_counts * std.shape[-1])
    )
    first_per_dim = std[:, 0].mean(dim=0)
    return {
        "normalized_std": float(per_dim.mean().item()),
        "normalized_std_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, per_dim.tolist())
        },
        "first_action_normalized_std": float(first_per_dim.mean().item()),
        "first_action_normalized_std_per_dim": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, first_per_dim.tolist())
        },
        "normalized_std_by_horizon": [
            float(value) if int(count) > 0 else None
            for value, count in zip(by_horizon.tolist(), valid_by_horizon)
        ],
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalized_mae": metrics["normalized_mae"],
        "first_action_normalized_mae": metrics[
            "first_action_normalized_mae"
        ],
        "first_action_gripper_sign_accuracy": metrics[
            "first_action_gripper_sign_accuracy"
        ],
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Noise-ensemble inference requires a CUDA GPU")
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
    gpu_normalizer = ActionNormalizer.from_stats_file(stats_path).to(device)
    metric_normalizer = ActionNormalizer.from_stats_file(stats_path)

    max_samples = max(args.sample_counts)
    noise_seed = args.seed + 2_000_000
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(noise_seed)
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
            batch_size = states.shape[0]

            repeated_states = states.repeat_interleave(max_samples, dim=0)
            repeated_agentview = agentview.repeat_interleave(
                max_samples, dim=0
            )
            repeated_wrist = wrist.repeat_interleave(max_samples, dim=0)
            repeated_texts = repeat_language(texts, max_samples)
            initial_noise = torch.randn(
                batch_size,
                max_samples,
                ACTION_CHUNK,
                ACTION_DIM,
                device=device,
                dtype=states.dtype,
                generator=noise_generator,
            ).reshape(batch_size * max_samples, ACTION_CHUNK, ACTION_DIM)

            predicted_normalized, _ = infer_action_chunk(
                model=model,
                diffusion=diffusion,
                normalizer=gpu_normalizer,
                states=repeated_states,
                agentview_observations=repeated_agentview,
                wrist_observations=repeated_wrist,
                language=repeated_texts,
                eta=0.0,
                amp_dtype=amp_dtype,
                initial_noise=initial_noise,
            )
            predicted_normalized = (
                predicted_normalized.reshape(
                    batch_size,
                    max_samples,
                    ACTION_CHUNK,
                    ACTION_DIM,
                )
                .permute(1, 0, 2, 3)
                .cpu()
            )
            collected["predicted_normalized"].append(predicted_normalized)
            collected["next_action"].append(batch["next_action"].cpu())
            collected["next_mask"].append(batch["next_mask"].cpu())
            collected["text"].extend(texts)

            processed += batch_size
            print(f"processed {processed}/{len(dataset)} samples", flush=True)
    finally:
        base_dataset.close()

    samples = torch.cat(collected["predicted_normalized"], dim=1)
    target_raw = torch.cat(collected["next_action"])
    target_mask = torch.cat(collected["next_mask"])
    texts = collected["text"]

    results_by_k: dict[str, Any] = {}
    primary_predictions: dict[int, torch.Tensor] = {}
    for sample_count in args.sample_counts:
        prefix_samples = samples[:sample_count]
        predictions = ensemble_predictions(
            prefix_samples,
            target_raw,
            target_mask,
            metric_normalizer,
        )
        strategy_metrics = {
            name: metrics_for_normalized_prediction(
                prediction,
                target_raw,
                target_mask,
                metric_normalizer,
            )
            for name, prediction in predictions.items()
        }
        primary_predictions[sample_count] = predictions[
            "mean_with_majority_gripper"
        ]

        individual_metrics = [
            metrics_for_normalized_prediction(
                prefix_samples[index],
                target_raw,
                target_mask,
                metric_normalizer,
            )
            for index in range(sample_count)
        ]
        individual_maes = torch.tensor(
            [metric["normalized_mae"] for metric in individual_metrics]
        )
        individual_first_maes = torch.tensor(
            [
                metric["first_action_normalized_mae"]
                for metric in individual_metrics
            ]
        )
        results_by_k[str(sample_count)] = {
            "strategies": strategy_metrics,
            "sampling_variance": sampling_variance(
                prefix_samples, target_mask
            ),
            "individual_sample_normalized_mae_mean": float(
                individual_maes.mean().item()
            ),
            "individual_sample_normalized_mae_std": float(
                individual_maes.std(unbiased=False).item()
            ),
            "individual_sample_first_action_mae_mean": float(
                individual_first_maes.mean().item()
            ),
            "individual_sample_first_action_mae_std": float(
                individual_first_maes.std(unbiased=False).item()
            ),
        }

    per_instruction: dict[str, Any] = {}
    for instruction in sorted(set(texts)):
        subset = torch.tensor(
            [text == instruction for text in texts], dtype=torch.bool
        )
        per_instruction[instruction] = {
            "num_samples": int(subset.sum().item()),
            "ensemble_by_k": {
                str(sample_count): compact_metrics(
                    metrics_for_normalized_prediction(
                        prediction[subset],
                        target_raw[subset],
                        target_mask[subset],
                        metric_normalizer,
                    )
                )
                for sample_count, prediction in primary_predictions.items()
            },
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
        "sample_counts": args.sample_counts,
        "results_by_k": results_by_k,
        "per_instruction": per_instruction,
        "interpretation": {
            "deployable_strategy": (
                "mean_with_majority_gripper uses no target information and can "
                "be tested online."
            ),
            "medoid": (
                "Medoid selects the sampled chunk closest to the sample mean, "
                "without target information."
            ),
            "oracle_warning": (
                "oracle_best_of_k selects actions using the recorded target. "
                "It is an offline upper bound and cannot be used online."
            ),
        },
    }

    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nMulti-noise ensemble summary")
    print(
        "  K | individual | continuous | majority | medoid | oracle | noise std"
    )
    for sample_count in args.sample_counts:
        result = results_by_k[str(sample_count)]
        strategies = result["strategies"]
        print(
            f"  {sample_count:>2} | "
            f"{result['individual_sample_normalized_mae_mean']:.6f} | "
            f"{strategies['continuous_mean']['normalized_mae']:.6f} | "
            f"{strategies['mean_with_majority_gripper']['normalized_mae']:.6f} | "
            f"{strategies['medoid']['normalized_mae']:.6f} | "
            f"{strategies['oracle_best_of_k']['normalized_mae']:.6f} | "
            f"{result['sampling_variance']['normalized_std']:.6f}"
        )
    print(f"saved results: {results_path}")


if __name__ == "__main__":
    main()
