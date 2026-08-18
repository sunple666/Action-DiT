from pathlib import Path

import torch


hub_directory = Path(r"D:\model_cache\torch_hub")
hub_directory.mkdir(parents=True, exist_ok=True)

# 告诉 PyTorch Hub 在哪里保存模型
torch.hub.set_dir(str(hub_directory))

print("Torch Hub directory:")
print(torch.hub.get_dir())


model = torch.hub.load(
    repo_or_dir="facebookresearch/dinov2",
    model="dinov2_vits14",
    trust_repo=True,
)

model.eval()

print("DINOv2 downloaded successfully")