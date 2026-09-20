from pathlib import Path

import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 encoder loaded from Hub or a local clone."""

    def __init__(self, repo: str = "facebookresearch/dinov2", weights: str | None = None, unfreeze_last_n_layers = 0):
        super().__init__()

        self.unfreeze_last_n_layers = unfreeze_last_n_layers

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
        if self.unfreeze_last_n_layers > 0:
            # Unfreeze the last n layers of the DINOv2 encoder
            for block in self.dino.blocks[-self.unfreeze_last_n_layers:]:
                block.requires_grad_(True)
            self.dino.norm.requires_grad_(True)

        self.dino.eval()

    def train(self, mode: bool = True):
        # The backbone is frozen and must stay in evaluation mode when the
        # parent ActionDiT module switches to training mode.
        super().train(mode)
        self.dino.eval()
        if mode and self.unfreeze_last_n_layers > 0:
            for block in self.dino.blocks[-self.unfreeze_last_n_layers:]:
                block.train(mode)
            self.dino.norm.train(mode)

        return self

    def forward(self, images):
        if self.unfreeze_last_n_layers > 0:
            return self.dino(images)
        with torch.no_grad():
            return self.dino(images)

    def forward_patch_tokens(self,images):
        if self.unfreeze_last_n_layers > 0:
            features = self.dino.forward_features(images)
            return features["x_norm_patchtokens"]
        with torch.no_grad():
            features=self.dino.forward_features(images)
            return features["x_norm_patchtokens"]

