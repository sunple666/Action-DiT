from pathlib import Path

import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 encoder loaded from Hub or a local clone."""

    def __init__(self, repo: str = "facebookresearch/dinov2", weights: str | None = None):
        super().__init__()

        is_local = Path(repo).expanduser().is_dir()
        load_kwargs = {
            "repo_or_dir": str(Path(repo).expanduser().resolve()) if is_local else repo,
            "model": "dinov2_vits14",
            "source": "local" if is_local else "github",
            "pretrained": True,
            "trust_repo": True,
        }
        if weights is not None:
            weights_path = Path(weights).expanduser().resolve()
            if not weights_path.is_file():
                raise FileNotFoundError(f"DINOv2 weights do not exist: {weights_path}")
            load_kwargs["weights"] = str(weights_path)

        self.dino = torch.hub.load(**load_kwargs)
        self.dino.requires_grad_(False)
        self.dino.eval()

    def train(self, mode: bool = True):
        # The backbone is frozen and must stay in evaluation mode when the
        # parent ActionDiT module switches to training mode.
        super().train(False)
        self.dino.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            return self.dino(images)
