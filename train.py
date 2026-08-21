import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from actiondit.action_dit import ActionDiT
from data.action_normalizer import ActionNormalizer
from diffusion import create_diffusion
from data.action_dataset import ActionDataset
import argparse

device="cuda" if torch.cuda.is_available() else "cpu"

batch_size=2
action_dim=7
action_chunk=16
state_dim=8
time_dim=128
hidden_dim=256
learn_sigma=True

parser=argparse.ArgumentParser()
parser.add_argument("--num_epochs",type=int,default=1000)
parser.add_argument("--batch_size",type=int,default=4)
args=parser.parse_args()
batch_size=args.batch_size
epochs=args.num_epochs

model=ActionDiT(
    action_dim=action_dim,
    time_dim=time_dim,
    state_dim=state_dim,
    hidden_dim=hidden_dim,
    depth=6,
    dino_dim=384,
    qwen_dim=1024,
    action_chunk=action_chunk,
    learn_sigma=learn_sigma
).to(device)

diffusion=create_diffusion(
    timestep_respacing="",
    noise_schedule="linear",
    diffusion_steps=1000,
    learn_sigma=learn_sigma
)

action_min=torch.tensor([-1.0]*action_dim,dtype=torch.float32).to(device)
action_max=torch.tensor([1.0]*action_dim,dtype=torch.float32).to(device)

normalizer=ActionNormalizer(action_min,action_max).to(device)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

train_dataset = ActionDataset()

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

def train_step(model,diffusion,normalizer,optimizer,batch,device):
    raw_actions=batch["action"].to(device)
    assert raw_actions.shape==(batch_size,action_chunk,action_dim)
    states=batch["state"].to(device)
    assert states.shape==(batch_size,state_dim)
    observations=batch["observation"].to(device)
    assert observations.shape==(batch_size,3,224,224)
    texts=batch["text"]
    timesteps=torch.randint(
        low=0,
        high=diffusion.num_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long
        )

    normalized_actions=normalizer.normalize(raw_actions)

    model_kwargs={
        "state_embeddings":states,
        "observation_embeddings":observations,
        "language_embeddings":texts
    }

    loss_dict=diffusion.training_losses(
        model=model,
        x_start=normalized_actions,
        t=timesteps,
        model_kwargs=model_kwargs
    )

    loss=loss_dict["loss"].mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    metrics={
        "loss":loss.detach().item(),
        "loss_mse":loss_dict["mse"].mean().detach().item(),
        "loss_vb":loss_dict["vb"].mean().detach().item()
    }

    return metrics

def main(args):
    num_epochs=args.num_epochs
    batch={
        "action":torch.rand(batch_size,action_chunk,action_dim,dtype=torch.float32).to(device),
        "state":torch.rand(batch_size,state_dim,dtype=torch.float32).to(device),
        "observation":torch.rand(batch_size,3,224,224,dtype=torch.float32).to(device),
        "text":["This is a test sentence."]*batch_size
    }

    model.train()
    model.o_embedder.eval()
    model.l_embedder.eval()

    for epoch in range(num_epochs):
        for step,batch in enumerate(train_loader):
            metrics=train_step(model,diffusion,normalizer,optimizer,batch,device)
            if step%10==0:
                print(
                    f"epoch {epoch}, step {step}, loss: {metrics['loss']:.4f}, mse: {metrics['loss_mse']:.4f}, vb: {metrics['loss_vb']:.4f}"
                )

if __name__=="__main__":
    main(args)