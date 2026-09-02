from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _default_dino_repo() -> str:
    configured = os.environ.get("ACTIONDIT_DINO_REPO")
    if configured:
        return configured
    torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    cached_repo = torch_home / "hub" / "facebookresearch_dinov2_main"
    return str(cached_repo) if cached_repo.is_dir() else "facebookresearch/dinov2"


def _default_dino_weights() -> str | None:
    configured = os.environ.get("ACTIONDIT_DINO_WEIGHTS")
    if configured:
        return configured
    torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    cached_weights = torch_home / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    return str(cached_weights) if cached_weights.is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="Action-DiT/outputs")
    parser.add_argument(
        "--qwen_model_path",
        default=os.environ.get(
            "ACTIONDIT_QWEN_MODEL", "Qwen/Qwen3-Embedding-0.6B"
        ),
    )
    parser.add_argument("--dino_repo", default=_default_dino_repo())
    parser.add_argument("--dino_weights", default=_default_dino_weights())

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps; 0 means no step limit.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()

    positive_names = (
        "num_epochs",
        "batch_size",
        "log_every",
        "val_every",
        "val_batches",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_steps < 0 or args.num_workers < 0:
        parser.error("--max_steps and --num_workers cannot be negative")
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
        condition = model.encode_condition(states, observations, texts)
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
    model.eval()
    totals = {"loss": 0.0, "loss_mse": 0.0, "loss_vb": 0.0}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
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
            for key in totals:
                totals[key] += metrics[key]
            count += 1
    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
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
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
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
        "best checkpoint updated: %s (validation loss=%.5f)",
        path,
        best_val_loss,
    )


def load_checkpoint(
    path: Path,
    model: ActionDiT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int]:
    checkpoint_path = path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys[:5]}"
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    LOGGER.info(
        "resumed: %s (epoch=%d, step=%d, "
        "frozen keys reloaded from pretrained models=%d)",
        checkpoint_path,
        epoch,
        global_step,
        len(incompatible.missing_keys),
    )
    return epoch, global_step


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
    (args.output_dir / "run_config.json").write_text(
        json.dumps(_serialized_args(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scaler
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

    writer = SummaryWriter(log_dir=args.output_dir / "tensorboard")
    model.train()
    best_val_loss = float("inf")
    final_epoch = start_epoch
    stop = False

    try:
        for epoch in range(start_epoch, args.num_epochs):
            final_epoch = epoch
            for batch in train_loader:
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
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    LOGGER.info(
                        "epoch=%d step=%d loss=%.5f mse=%.5f vb=%.5f "
                        "grad=%.4f max_vram_gb=%.2f",
                        epoch,
                        global_step,
                        metrics["loss"],
                        metrics["loss_mse"],
                        metrics["loss_vb"],
                        float(grad_norm),
                        memory_gb,
                    )
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, global_step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar("system/max_vram_gb", memory_gb, global_step)

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
                    LOGGER.info(
                        "validation step=%d loss=%.5f mse=%.5f vb=%.5f",
                        global_step,
                        val_metrics["loss"],
                        val_metrics["loss_mse"],
                        val_metrics["loss_vb"],
                    )
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"val/{key}", value, global_step)
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        save_checkpoint(
                            args.output_dir / "best.pt",
                            model,
                            optimizer,
                            scaler,
                            args,
                            epoch,
                            global_step,
                            best_val_loss,
                        )
                    writer.add_scalar("val/best_loss", best_val_loss, global_step)

                if args.max_steps and global_step >= args.max_steps:
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
                scaler,
                args,
                final_epoch,
                global_step,
                best_val_loss,
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
