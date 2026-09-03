"""Parse ActionDiT's train.log and save training/validation loss curves."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib

# The script is primarily intended for headless training servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FIELD_PATTERN = re.compile(
    rf"\b(epoch|step|loss|mse|vb|grad)=({NUMBER}|nan|inf|-inf)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ActionDiT loss curves from train.log."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        help=(
            "Path to train.log or a run directory. If omitted, use the newest "
            "outputs/run_*/train.log."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory searched when LOG is omitted (default: outputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: <run_dir>/loss_curve.png).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window; 1 disables smoothing (default: 1).",
    )
    parser.add_argument(
        "--include-step-one",
        action="store_true",
        help="Include step=1 in the plot (excluded by default).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image resolution (default: 160).",
    )
    return parser.parse_args()


def resolve_log_path(log: Path | None, outputs_dir: Path) -> Path:
    if log is not None:
        path = log.expanduser()
        if path.is_dir():
            path = path / "train.log"
        if not path.is_file():
            raise FileNotFoundError(f"Training log not found: {path}")
        return path.resolve()

    candidates = list(outputs_dir.expanduser().glob("run_*/train.log"))
    if not candidates:
        raise FileNotFoundError(
            f"No train.log found under {outputs_dir}. Pass the log path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def extract_fields(line: str) -> dict[str, float]:
    return {key.lower(): float(value) for key, value in FIELD_PATTERN.findall(line)}


def parse_log(
    log_path: Path, *, include_step_one: bool = False
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    train_records: list[dict[str, float]] = []
    val_records: list[dict[str, float]] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = extract_fields(line)
            if "step" not in fields or "loss" not in fields:
                continue
            if not include_step_one and fields["step"] == 1:
                continue
            if "validation step=" in line:
                val_records.append(fields)
            elif "epoch=" in line:
                train_records.append(fields)

    if not train_records:
        raise ValueError(f"No training loss records found in {log_path}")
    return train_records, val_records


def values(records: Iterable[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    points = [(record["step"], record[key]) for record in records if key in record]
    return [point[0] for point in points], [point[1] for point in points]


def moving_average(sequence: list[float], window: int) -> list[float]:
    if window <= 1:
        return sequence.copy()

    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(sequence):
        running_sum += value
        if index >= window:
            running_sum -= sequence[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def plot_metric(
    axis: plt.Axes,
    train_records: list[dict[str, float]],
    val_records: list[dict[str, float]],
    key: str,
    label: str,
    color: str,
    smooth: int,
) -> None:
    train_steps, train_values = values(train_records, key)
    if train_values:
        if smooth == 1:
            axis.plot(
                train_steps,
                train_values,
                color=color,
                linewidth=1.2,
                label=f"train {label}",
            )
        else:
            axis.plot(
                train_steps,
                train_values,
                color=color,
                alpha=0.18,
                linewidth=0.8,
                label=f"train {label} (raw)",
            )
            axis.plot(
                train_steps,
                moving_average(train_values, smooth),
                color=color,
                linewidth=1.8,
                label=f"train {label} (MA {smooth})",
            )

    val_steps, val_values = values(val_records, key)
    if val_values:
        axis.plot(
            val_steps,
            val_values,
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            linestyle="--",
            label=f"validation {label}",
        )


def create_figure(
    train_records: list[dict[str, float]],
    val_records: list[dict[str, float]],
    log_path: Path,
    smooth: int,
) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (loss_axis, component_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )

    plot_metric(
        loss_axis,
        train_records,
        val_records,
        key="loss",
        label="loss",
        color="#1f77b4",
        smooth=smooth,
    )
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title(f"ActionDiT training curves — {log_path.parent.name}")
    loss_axis.legend(loc="best")

    plot_metric(
        component_axis,
        train_records,
        val_records,
        key="mse",
        label="MSE",
        color="#2ca02c",
        smooth=smooth,
    )
    plot_metric(
        component_axis,
        train_records,
        val_records,
        key="vb",
        label="VB",
        color="#d62728",
        smooth=smooth,
    )
    component_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    component_axis.set_xlabel("Global step")
    component_axis.set_ylabel("Component loss")
    component_axis.legend(loc="best", ncol=2)

    return figure


def main() -> None:
    args = parse_args()
    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")

    log_path = resolve_log_path(args.log, args.outputs_dir)
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else log_path.parent / "loss_curve.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_records, val_records = parse_log(
        log_path, include_step_one=args.include_step_one
    )
    figure = create_figure(
        train_records=train_records,
        val_records=val_records,
        log_path=log_path,
        smooth=args.smooth,
    )
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)

    print(f"Read log: {log_path}")
    print(f"Training points: {len(train_records)}")
    print(f"Validation points: {len(val_records)}")
    print(f"Saved figure: {output_path.resolve()}")


if __name__ == "__main__":
    main()
