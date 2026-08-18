import torch
from actiondit.action_dit import ActionDiT
from actiondit.vision_encoder import DINOv2Encoder
from actiondit.language_encoder import QwenEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size=2
action_chunk=16
action_dim=7
time_dim=128
state_dim=7
hidden_dim=256

model=ActionDiT(action_dim,time_dim,state_dim,hidden_dim).to(device)

noisy_actions=torch.randn(batch_size,action_chunk,action_dim).to(device)

time_steps=torch.randint(low=1,high=100,size=(batch_size,)).to(device)

states=torch.randn(batch_size,action_dim).to(device)

observations=torch.randn(batch_size,3,224,224).to(device)

texts=["Hello, how are you?"]*batch_size

predicted_actions=model(noisy_actions,time_steps,states,observations,texts)

loss=torch.mean((predicted_actions-noisy_actions)**2)

loss.backward()

