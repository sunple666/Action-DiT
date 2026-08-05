import torch
from actiondit.action_dit import ActionDiT

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size=2
action_chunk=16
action_dim=7
time_dim=batch_size
state_dim=7
hidden_dim=256

model=ActionDiT(action_dim,time_dim,state_dim,hidden_dim).to(device)

noisy_actions=torch.randn(batch_size,action_chunk,action_dim).to(device)

time_steps=torch.randint(low=1,high=100,size=(batch_size,)).to(device)

states=torch.randn(batch_size,action_dim).to(device)

predicted_actions=model(noisy_actions,time_steps,states)

loss=torch.mean((predicted_actions-noisy_actions)**2)

loss.backward()

