import torch
import torch.nn as nn

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        repo_path=(r"D:\model_cache\torch_hub\facebookresearch_dinov2_main")
        weights_path=(r"D:\model_cache\torch_hub\checkpoints\dinov2_vits14_pretrain.pth")
        self.dino=torch.hub.load(
            repo_or_dir=repo_path,
            model="dinov2_vits14",
            source="local",
            pretrained=True,
            weights=weights_path
        )

        for param in self.dino.parameters():
            param.requires_grad=False

        self.dino.eval()

    def forward(self,images):
        ret=self.dino(images)
        return ret