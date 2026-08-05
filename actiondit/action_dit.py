import torch
import torch.nn as nn
import math

def modulate(x,scale,shift):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class ActionEmbedder(nn.Module):
    def __init__(self,action_dim,hidden_dim):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=action_dim,out_features=hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
    def forward(self,x):
        x=self.mlp(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self,time_dim,hidden_dim):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=time_dim,out_features=hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
        self.time_dim=time_dim
        self.hidden_dim=hidden_dim

    @staticmethod
    def timestep_embedding(t,dim,max_period=10000):
        half=dim//2
        freq=torch.exp(-torch.arange(start=0,end=half,dtype=torch.float32)/half*math.log(max_period))
        tmp=t.unsqueeze(1)*freq.unsqueeze(0)
        ret=torch.cat([torch.cos(tmp),torch.sin(tmp)],dim=1)
        if dim%2==1:
            ret=torch.cat([ret,torch.zeros_like(ret[:,:1])],dim=1)
        return ret

    def forward(self,t):
        emb=TimestepEmbedder.timestep_embedding(t,dim=self.time_dim)
        emb=self.mlp(emb)
        return emb

class StateEmbedder(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=state_dim,out_features=hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
    def forward(self,x):
        x=self.mlp(x)
        return x

class FinalLayer(nn.Module):
    def __init__(self,hidden_dim,action_dim):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim,out_features=action_dim,bias=True)
        )
    def forward(self,x):
        x=self.mlp(x)
        return x

class ActionDiTBlock(nn.Module):
    def __init__(self,hidden_dim,mlp_hidden_dim,num_heads,hidden_size):
        super().__init__()
        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim)
        self.attn=nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=num_heads,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=hidden_dim,out_features=mlp_hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=mlp_hidden_dim,out_features=hidden_dim,bias=True)
        )
        self.condition_mlp=nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size*6,bias=True)
        )
    def forward(self,x,c):
        attn_gate,mlp_gate,attn_scale,attn_shift,mlp_scale,mlp_shift=self.condition_mlp(c).chunk(6,dim=-1)
        attn_input=modulate(self.norm1(x),scale=attn_scale,shift=attn_shift)
        h=x+attn_gate.unsqueeze(1)*self.attn(attn_input,attn_input,attn_input)[0]
        x=h+mlp_gate.unsqueeze(1)*self.mlp(modulate(self.norm2(h),scale=mlp_scale,shift=mlp_shift))
        return x

class ActionDiT(nn.Module):
    def __init__(self,action_dim,time_dim,state_dim,hidden_dim,depth=6):
        super().__init__()

        self.x_embedder=ActionEmbedder(action_dim,hidden_dim)
        self.t_embedder=TimestepEmbedder(time_dim,hidden_dim)
        self.s_embedder=StateEmbedder(state_dim,hidden_dim)

        self.blocks=nn.ModuleList([
            ActionDiTBlock(hidden_dim=hidden_dim,mlp_hidden_dim=hidden_dim*4,num_heads=8,hidden_size=hidden_dim)
            for _ in range(depth)
        ])
        self.final_layer=FinalLayer(hidden_dim,action_dim)

    def forward(
            self,
            predicted_actions,
            timesteps,
            state_embeddings,
            observation_embeddings=None,
            language_embeddings=None):
        
        predicted_actions=self.x_embedder(predicted_actions)
        timesteps=self.t_embedder(timesteps)
        state_embeddings=self.s_embedder(state_embeddings)
        condition=timesteps+state_embeddings
        for block in self.blocks:
            predicted_actions=block(predicted_actions,condition)
        predicted_actions=self.final_layer(predicted_actions)
        return predicted_actions

