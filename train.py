import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from actiondit.action_dit import ActionDiT
from data.action_normalizer import ActionNormalizer
from diffusion import create_diffusion
from data.action_dataset import ActionDataset
import argparse

device="cuda" if torch.cuda.is_available() else "cpu"
assert device=="cuda"

action_dim=7
action_chunk=16
state_dim=8
time_dim=128
hidden_dim=256
learn_sigma=True

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_root",type=str,required=True)
    parser.add_argument("--manifest_path",type=str,required=True)
    parser.add_argument("--num_epochs",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--max_steps",type=int,default=1)
    parser.add_argument("--lr",type=float,default=1e-4)

    args=parser.parse_args()
    return args


def train_step(model,diffusion,normalizer,optimizer,batch,device):
    raw_actions=batch["action"].to(device)
    current_batch_size=raw_actions.shape[0]
    assert raw_actions.shape==(current_batch_size,action_chunk,action_dim)
    action_mask=batch["action_mask"].to(device,dtype=torch.bool)
    assert action_mask.shape==(current_batch_size,action_chunk)
    if not action_mask.any(dim=1).all().item():
        raise ValueError("无有效动作")
    states=batch["state"].to(device)
    assert states.shape==(current_batch_size,state_dim)
    observations=batch["observation"].to(device)
    assert observations.shape==(current_batch_size,3,224,224)
    texts=batch["text"]
    assert len(texts)==current_batch_size
    timesteps=torch.randint(
        low=0,
        high=diffusion.num_timesteps,
        size=(current_batch_size,),
        device=device,
        dtype=torch.long
        )

    normalized_actions=normalizer.normalize(raw_actions)

    condition=model.encode_condition(states,observations,texts)

    model_kwargs={
        "condition":condition,
        "action_mask":action_mask
    }

    loss_dict=diffusion.training_losses(
        model=model,
        x_start=normalized_actions,
        t=timesteps,
        model_kwargs=model_kwargs,
        loss_mask=action_mask
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

    optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.01
    )

    train_dataset = ActionDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        split="train",
        action_chunk=action_chunk,
    )

    train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
    )

    normalizer=ActionNormalizer.from_stats_file("configs/libero_action_stats.json").to(device)



    num_epochs=args.num_epochs

    model.train()
    model.o_embedder.dino.eval()
    model.l_embedder.qwen.eval()

    current_steps=0
    max_steps=args.max_steps
    should_stop=False
    
    for epoch in range(num_epochs):
        for _,batch in enumerate(train_loader):
            metrics=train_step(model,diffusion,normalizer,optimizer,batch,device)
            if current_steps%100==0:
                print(
                    f"epoch {epoch}, step {current_steps}, loss: {metrics['loss']:.4f}, mse: {metrics['loss_mse']:.4f}, vb: {metrics['loss_vb']:.4f}"
                )
            current_steps+=1
            if current_steps>=max_steps:
                should_stop=True
                print("训练完成")
                break
        if should_stop:
            break

    train_dataset.close()

if __name__=="__main__":
    args=get_args()
    main(args)
