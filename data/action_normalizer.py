from pathlib import Path

import torch
from torch import nn

from data.action_stats import ActionStats

class ActionNormalizer(nn.Module):
    def __init__(self, action_min, action_max, eps=1e-6):
        super().__init__()
        action_min = torch.as_tensor(action_min, dtype=torch.float32).detach().clone()
        action_max = torch.as_tensor(action_max, dtype=torch.float32).detach().clone()
        if action_min.ndim != 1 or action_max.ndim != 1:
            raise ValueError("action_min and action_max must be one-dimensional")
        if action_min.shape != action_max.shape:
            raise ValueError("action_min and action_max must have the same shape")
        if not torch.isfinite(action_min).all() or not torch.isfinite(action_max).all():
            raise ValueError("Action bounds must be finite")
        if not torch.all(action_max - action_min > eps):
            raise ValueError("Every action dimension must have a non-zero range")

        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)
        self.eps = eps

    @classmethod
    def from_stats_file(cls, path: str | Path, eps: float = 1e-6):
        stats = ActionStats.load(path)
        return cls(stats.action_min, stats.action_max, eps=eps)

    def normalize(self, raw_action, *, clamp=False):
        if raw_action.shape[-1] != self.action_min.shape[0]:
            raise ValueError(
                f"Expected action dimension {self.action_min.shape[0]}, "
                f"got {raw_action.shape[-1]}"
            )
        if not torch.isfinite(raw_action).all():
            raise ValueError("Raw actions contain non-finite values")

        normalized = -1 + 2 * (raw_action - self.action_min) / (
            self.action_max - self.action_min
        )
        if clamp:
            return normalized.clamp(-1, 1)
        if not torch.all(
            (normalized >= -1 - self.eps) & (normalized <= 1 + self.eps)
        ):
            raise ValueError("Raw actions fall outside the fitted training range")
        return normalized

    def denormalize(self, normalized_action):
        if normalized_action.shape[-1] != self.action_min.shape[0]:
            raise ValueError(
                f"Expected action dimension {self.action_min.shape[0]}, "
                f"got {normalized_action.shape[-1]}"
            )
        if not torch.isfinite(normalized_action).all():
            raise ValueError("Normalized actions contain non-finite values")
        if not torch.all(
            (normalized_action >= -1 - self.eps)
            & (normalized_action <= 1 + self.eps)
        ):
            raise ValueError("Normalized actions must lie in [-1, 1]")

        normalized_action = normalized_action.clamp(-1, 1)
        return self.action_min + (normalized_action + 1) * (
            self.action_max - self.action_min
        ) / 2
