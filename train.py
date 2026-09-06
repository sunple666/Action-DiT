from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_STORAGE_ROOT = Path("/root/autodl-tmp/actiondit_storage")
PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_STORAGE_ROOT = Path(
    os.environ.get("STORE", str(DEFAULT_STORAGE_ROOT))
).expanduser()

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from actiondit.action_dit import ActionDiT
from data.action_dataset import ActionDataset
from data.action_normalizer import ActionNormalizer
from diffusion import create_diffusion


ACTION_DIM = 7
ACTION_CHUNK = 16
STATE_DIM = 8
TIME_DIM = 128
HIDDEN_DIM = 256
LEARN_SIGMA = True

LOGGER = logging.getLogger("actiondit.train")


def format_duration(seconds: float) -> str:
    """Format a duration as DDd HH:MM:SS or HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


def create_run_directory(output_root: Path) -> Path:
    """Create a unique output directory for one training invocation."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root.expanduser() / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> Path:
    """Write training messages to both the console and a persistent log file."""
    log_path = run_dir / "train.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    return log_path


def _storage_torch_home() -> Path:
    return RUNTIME_STORAGE_ROOT / "cache" / "torch"


def _default_dino_repo() -> str:
    configured = os.environ.get("ACTIONDIT_DINO_REPO") or os.environ.get("DINO_REPO")
    if configured:
        return configured
    return str(_storage_torch_home() / "hub" / "facebookresearch_dinov2_main")


def _default_dino_weights() -> str | None:
    configured = os.environ.get("ACTIONDIT_DINO_WEIGHTS") or os.environ.get(
        "DINO_WEIGHTS"
    )
    if configured:
        return configured
    return str(
        _storage_torch_home()
        / "hub"
        / "checkpoints"
        / "dinov2_vits14_pretrain.pth"
    )


def _default_dataset_root() -> Path:
    configured = os.environ.get("DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_STORAGE_ROOT / "datasets" / "libero"


def _default_qwen_model_path() -> str:
    configured = os.environ.get("ACTIONDIT_QWEN_MODEL") or os.environ.get(
        "QWEN_MODEL"
    )
    if configured:
        return configured
    return str(RUNTIME_STORAGE_ROOT / "models" / "Qwen3-Embedding-0.6B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=_default_dataset_root())
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
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--qwen_model_path", default=_default_qwen_model_path())
    parser.add_argument("--dino_repo", default=_default_dino_repo())
    parser.add_argument("--dino_weights", default=_default_dino_weights())

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps; 0 means no step limit.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument(
        "--val_batches",
        type=int,
        default=0,
        help="Maximum validation batches; 0 evaluates the full validation set.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()

    positive_names = (
        "num_epochs",
        "batch_size",
        "log_every",
        "val_every",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_steps < 0 or args.num_workers < 0 or args.val_batches < 0:
        parser.error(
            "--max_steps, --num_workers, and --val_batches cannot be negative"
        )
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0 <= args.min_lr <= args.lr:
        parser.error("--min_lr must be between 0 and --lr")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def precision_dtype(precision: str) -> torch.dtype | None:
    return {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]


def validate_batch(batch: dict[str, Any]) -> int:
    batch_size = batch["action"].shape[0]
    expected = {
        "action": (batch_size, ACTION_CHUNK, ACTION_DIM),
        "action_mask": (batch_size, ACTION_CHUNK),
        "state": (batch_size, STATE_DIM),
        "observation": (batch_size, 3, 224, 224),
    }
    for key, shape in expected.items():
        if tuple(batch[key].shape) != shape:
            raise ValueError(f"Expected {key} shape {shape}, got {batch[key].shape}")
    if len(batch["text"]) != batch_size:
        raise ValueError("Text batch size does not match tensor batch size")
    if not batch["action_mask"].any(dim=1).all().item():
        raise ValueError("A sample contains no valid actions")
    return batch_size


def compute_loss(
    model: ActionDiT,
    diffusion,
    normalizer: ActionNormalizer,
    batch: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    *,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = validate_batch(batch)
    raw_actions = batch["action"].to(device, non_blocking=True)
    action_mask = batch["action_mask"].to(
        device, dtype=torch.bool, non_blocking=True
    )
    states = batch["state"].to(device, non_blocking=True)
    observations = batch["observation"].to(device, non_blocking=True)
    texts = batch["text"]
    timesteps = torch.randint(
        0,
        diffusion.num_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    normalized_actions = normalizer.normalize(raw_actions, clamp=not training)

    with torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
    ):
        condition = model.encode_static_condition(states, observations, texts)
        loss_dict = diffusion.training_losses(
            model=model,
            x_start=normalized_actions,
            t=timesteps,
            model_kwargs={
                "condition": condition,
                "action_mask": action_mask,
            },
            loss_mask=action_mask,
        )
        loss = loss_dict["loss"].mean()

    zero = torch.zeros((), device=device)
    metrics = {
        "loss": float(loss.detach()),
        "loss_mse": float(loss_dict["mse"].mean().detach()),
        "loss_vb": float(loss_dict.get("vb", zero).mean().detach()),
    }
    return loss, metrics


def evaluate(
    model: ActionDiT,
    diffusion,
    normalizer: ActionNormalizer,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int,
) -> dict[str, float]:
    if max_batches < 0:
        raise ValueError("max_batches cannot be negative")
    model.eval()
    totals = {"loss": 0.0, "loss_mse": 0.0, "loss_vb": 0.0}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            _, metrics = compute_loss(
                model,
                diffusion,
                normalizer,
                batch,
                device,
                amp_dtype,
                training=False,
            )
            batch_size = int(batch["action"].shape[0])
            for key in totals:
                totals[key] += metrics[key] * batch_size
            count += batch_size
    if count == 0:
        raise RuntimeError("Validation loader produced no samples")
    model.train()
    return {key: value / count for key, value in totals.items()}


def trainable_state_dict(model: ActionDiT) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def _serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: ActionDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "args": _serialized_args(args),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    LOGGER.info(
        "checkpoint saved: %s (best validation loss=%.5f)",
        path,
        best_val_loss,
    )


def load_checkpoint(
    path: Path,
    model: ActionDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int, float]:
    checkpoint_path = path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    format_version = checkpoint.get("format_version")
    if format_version not in (1, 2):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    scheduler_state = checkpoint.get("scheduler")
    if scheduler_state is None:
        raise ValueError(
            "Checkpoint does not contain an LR scheduler state and cannot "
            f"resume cosine annealing accurately: {checkpoint_path}"
        )
    saved_t_max = int(scheduler_state["T_max"])
    if saved_t_max != scheduler.T_max:
        raise ValueError(
            "Checkpoint LR schedule is incompatible with this run: "
            f"saved T_max={saved_t_max}, current T_max={scheduler.T_max}. "
            "Keep batch size, num_epochs, and max_steps consistent when resuming."
        )
    saved_eta_min = float(scheduler_state["eta_min"])
    if saved_eta_min != scheduler.eta_min:
        raise ValueError(
            "Checkpoint LR schedule is incompatible with this run: "
            f"saved eta_min={saved_eta_min}, current eta_min={scheduler.eta_min}."
        )
    saved_base_lrs = [float(lr) for lr in scheduler_state["base_lrs"]]
    current_base_lrs = [float(lr) for lr in scheduler.base_lrs]
    if saved_base_lrs != current_base_lrs:
        raise ValueError(
            "Checkpoint LR schedule is incompatible with this run: "
            f"saved base_lrs={saved_base_lrs}, "
            f"current base_lrs={current_base_lrs}."
        )
    saved_global_step = int(checkpoint["global_step"])
    saved_scheduler_step = int(scheduler_state["last_epoch"])
    if saved_scheduler_step != saved_global_step:
        raise ValueError(
            "Checkpoint scheduler and optimizer-step counters disagree: "
            f"scheduler last_epoch={saved_scheduler_step}, "
            f"global_step={saved_global_step}."
        )
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys[:5]}"
        )
    scheduler.load_state_dict(scheduler_state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    epoch = int(checkpoint["epoch"])
    global_step = saved_global_step
    best_val_loss = float(checkpoint["best_val_loss"])
    LOGGER.info(
        "resumed: %s (epoch=%d, step=%d, lr=%.6e, "
        "frozen keys reloaded from pretrained models=%d)",
        checkpoint_path,
        epoch,
        global_step,
        scheduler.get_last_lr()[0],
        len(incompatible.missing_keys),
    )
    return epoch, global_step, best_val_loss


def make_loader(
    dataset: ActionDataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def main(args: argparse.Namespace) -> None:
    os.environ["TORCH_HOME"] = str(_storage_torch_home())
    args.output_dir = create_run_directory(args.output_dir)
    log_path = setup_logging(args.output_dir)
    LOGGER.info("output directory: %s", args.output_dir.resolve())
    LOGGER.info("log file: %s", log_path.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("ActionDiT training requires a CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision fp16")

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dino_repo_path = Path(args.dino_repo).expanduser().resolve()
    if not dino_repo_path.is_dir():
        raise NotADirectoryError(f"DINOv2 repository does not exist: {dino_repo_path}")
    dino_weights_path = Path(args.dino_weights).expanduser().resolve()
    if not dino_weights_path.is_file():
        raise FileNotFoundError(f"DINOv2 weights do not exist: {dino_weights_path}")
    qwen_model_path = Path(args.qwen_model_path).expanduser().resolve()
    if not qwen_model_path.is_dir():
        raise NotADirectoryError(f"Qwen model does not exist: {qwen_model_path}")
    args.dino_repo = str(dino_repo_path)
    args.dino_weights = str(dino_weights_path)
    args.qwen_model_path = str(qwen_model_path)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(_serialized_args(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("DINOv2 repository: %s", dino_repo_path)
    LOGGER.info("DINOv2 weights: %s", dino_weights_path)
    LOGGER.info("Qwen model: %s", qwen_model_path)

    amp_dtype = precision_dtype(args.precision)
    qwen_dtype = torch.float32 if args.precision == "fp32" else amp_dtype
    model = ActionDiT(
        action_dim=ACTION_DIM,
        time_dim=TIME_DIM,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        depth=6,
        dino_dim=384,
        qwen_dim=1024,
        action_chunk=ACTION_CHUNK,
        learn_sigma=LEARN_SIGMA,
        dino_repo=args.dino_repo,
        dino_weights=args.dino_weights,
        qwen_model_path=args.qwen_model_path,
        qwen_dtype=qwen_dtype,
    ).to(device)
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule="linear",
        diffusion_steps=1000,
        learn_sigma=LEARN_SIGMA,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    normalizer = ActionNormalizer.from_stats_file(args.stats_path).to(device)

    train_dataset = ActionDataset(
        args.dataset_root,
        args.manifest_path,
        "train",
        action_chunk=ACTION_CHUNK,
    )
    val_dataset = ActionDataset(
        args.dataset_root,
        args.manifest_path,
        "val",
        action_chunk=ACTION_CHUNK,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        train_dataset, args, shuffle=True, generator=generator
    )
    val_loader = make_loader(
        val_dataset, args, shuffle=False, generator=generator
    )

    planned_steps = len(train_loader) * args.num_epochs
    total_steps = (
        planned_steps
        if args.max_steps == 0
        else min(args.max_steps, planned_steps)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
        )
        if global_step > total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} exceeds this run's "
                f"total_steps={total_steps}"
            )
        # Every invocation writes to a new run directory. Preserve the exact
        # state used to resume even if continued training is interrupted before
        # the first validation.
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            start_epoch,
            global_step,
            best_val_loss,
        )

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("device: %s", torch.cuda.get_device_name(device))
    LOGGER.info("precision: %s", args.precision)
    LOGGER.info(
        "train samples: %d, val samples: %d", len(train_dataset), len(val_dataset)
    )
    LOGGER.info(
        "parameters: trainable=%s, total=%s",
        f"{trainable_count:,}",
        f"{total_count:,}",
    )
    LOGGER.info(
        "lr schedule: cosine peak=%.6e min=%.6e total_steps=%d",
        args.lr,
        args.min_lr,
        total_steps,
    )

    writer = SummaryWriter(log_dir=args.output_dir / "tensorboard")
    model.train()
    final_epoch = start_epoch
    stop = False
    invocation_start_step = global_step
    training_start_time = time.monotonic()
    LOGGER.info(
        "training timer started: start_step=%d remaining_steps=%d",
        invocation_start_step,
        total_steps - invocation_start_step,
    )

    def timing_snapshot() -> tuple[float, float]:
        """Return elapsed time and estimated remaining time."""
        elapsed_seconds = time.monotonic() - training_start_time
        completed_steps = global_step - invocation_start_step
        if completed_steps <= 0:
            return elapsed_seconds, 0.0
        average_step_seconds = elapsed_seconds / completed_steps
        eta_seconds = average_step_seconds * max(total_steps - global_step, 0)
        return elapsed_seconds, eta_seconds

    try:
        for epoch in range(start_epoch, args.num_epochs):
            final_epoch = epoch
            for batch in train_loader:
                if global_step >= total_steps:
                    stop = True
                    break
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_loss(
                    model,
                    diffusion,
                    normalizer,
                    batch,
                    device,
                    amp_dtype,
                    training=True,
                )
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(trainable_parameters, args.grad_clip)
                else:
                    grad_norm = torch.zeros((), device=device)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    elapsed_seconds, eta_seconds = timing_snapshot()
                    LOGGER.info(
                        "epoch=%d step=%d loss=%.5f mse=%.5f vb=%.5f "
                        "grad=%.4f max_vram_gb=%.2f lr=%.6e "
                        "elapsed=%s eta=%s",
                        epoch,
                        global_step,
                        metrics["loss"],
                        metrics["loss_mse"],
                        metrics["loss_vb"],
                        float(grad_norm),
                        memory_gb,
                        scheduler.get_last_lr()[0],
                        format_duration(elapsed_seconds),
                        format_duration(eta_seconds),
                    )
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, global_step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar(
                        "train/learning_rate",
                        scheduler.get_last_lr()[0],
                        global_step,
                    )
                    writer.add_scalar("system/max_vram_gb", memory_gb, global_step)
                    writer.add_scalar(
                        "system/elapsed_hours", elapsed_seconds / 3600, global_step
                    )
                    writer.add_scalar(
                        "system/eta_hours", eta_seconds / 3600, global_step
                    )
                if global_step % args.val_every == 0:
                    val_metrics = evaluate(
                        model,
                        diffusion,
                        normalizer,
                        val_loader,
                        device,
                        amp_dtype,
                        args.val_batches,
                    )
                    elapsed_seconds, eta_seconds = timing_snapshot()
                    LOGGER.info(
                        "validation step=%d loss=%.5f mse=%.5f vb=%.5f "
                        "elapsed=%s eta=%s",
                        global_step,
                        val_metrics["loss"],
                        val_metrics["loss_mse"],
                        val_metrics["loss_vb"],
                        format_duration(elapsed_seconds),
                        format_duration(eta_seconds),
                    )
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"val/{key}", value, global_step)
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        save_checkpoint(
                            args.output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            args,
                            epoch,
                            global_step,
                            best_val_loss,
                        )
                    save_checkpoint(
                        args.output_dir / "last.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        epoch,
                        global_step,
                        best_val_loss,
                    )
                    writer.add_scalar("val/best_loss", best_val_loss, global_step)

                if global_step >= total_steps:
                    stop = True
                    break
            if stop:
                break

        final_val_metrics = evaluate(
            model,
            diffusion,
            normalizer,
            val_loader,
            device,
            amp_dtype,
            args.val_batches,
        )
        for key, value in final_val_metrics.items():
            writer.add_scalar(f"val_final/{key}", value, global_step)
        LOGGER.info(
            "final validation step=%d loss=%.5f",
            global_step,
            final_val_metrics["loss"],
        )
        if final_val_metrics["loss"] < best_val_loss:
            best_val_loss = final_val_metrics["loss"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                final_epoch,
                global_step,
                best_val_loss,
            )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            final_epoch,
            global_step,
            best_val_loss,
        )
        total_elapsed_seconds = time.monotonic() - training_start_time
        LOGGER.info(
            "training finished: elapsed=%s optimizer_steps_this_run=%d "
            "final_step=%d",
            format_duration(total_elapsed_seconds),
            global_step - invocation_start_step,
            global_step,
        )
    finally:
        writer.close()
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception:
        LOGGER.exception("training failed")
        raise
