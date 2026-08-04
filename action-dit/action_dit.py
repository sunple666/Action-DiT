import torch
import torch.nn as nn
import math

def modulate(x,scale,shift):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class ActionEmbedder(nn.module):
    def __init__(self,action_dim,hidden_dim):
        super().__init__()
        self.mlp=nn.sequential(
            nn.linear(in_features=action_dim,out_features=hidden_dim,bias=True),
            nn.SiLu(),
            nn.linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
    def forward(self,x):
        x=self.mlp(x)
        return x


class TimestepEmbedder(nn.module):
    def __init__(self,time_dim,hidden_dim):
        super().__init__()
        self.mlp=nn.sequential(
            nn.linear(in_features=time_dim,out_features=hidden_dim,bias=True),
            nn.SiLu(),
            nn.linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
        self.original_dim=time_dim
        self.hidden_dim=hidden_dim

    @staticmethod
    def timestep_embedding(t,dim,max_period=10000):
        half=dim//2
        freq=torch.exp(-torch.arrange(start=0,end=half,dtype=torch.float32)/half*math.log(max_period))
        tmp=t.unsqueeze(1)*freq.unsqueeze(0)
        ret=torch.cat(torch.cos(tmp),torch.sin(tmp),dim=1)
        if dim%2==1:
            ret=torch.cat(ret,torch.zero_like(ret[:,:1]),dim=1)
        return ret

    def forward(self,t):
        emb=TimestepEmbedder.timestep_embedding(t,dim=self.original_dim)
        emb=self.mlp(emb)
        return emb

class StateEmbedder(nn.module):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.mlp=nn.sequential(
            nn.linear(in_features=state_dim,out_features=hidden_dim,bias=True),
            nn.SiLu(),
            nn.linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
    def forward(self,x):
        x=self.mlp(x)
        return x

class ActionDiTBlock(nn.module):
    def __init__(self,hidden_dim,mlp_hidden_dim,num_heads,hidden_size):
        super().__init__()
        self.attn=nn.attention(hidden_dim,hidden_dim,num_heads=num_heads)
        self.mlp=nn.sequential(
            nn.linear(in_features=hidden_dim,out_features=mlp_hidden_dim,bias=True),
            nn.SiLU(),
            nn.linear(in_features=mlp_hidden_dim,out_features=hidden_dim,bias=True)
        )
        self.condition_mlp=nn.sequential(
            nn.SiLU(),
            nn.linear(in_features=hidden_size,out_features=hidden_size*6,bias=True)
        )
    def forward(self,x,c):
        attn_gate,mlp_gate,attn_scale,attn_shift,mlp_scale,mlp_shift=self.condition_mlp(c).chunk(6,dim=-1)
        h=x+attn_gate*self.attn(modulate(nn.layernorm(x),scale=attn_scale,shift=attn_shift))
        x=h+mlp_gate*self.mlp(modulate(nn.layernorm(h),scale=mlp_scale,shift=mlp_shift))
        return x

class ActionDiT(nn.module):
    def __init__(self):
        super().__init__()

        self.x_embedder=ActionEmbedder()
        self.t_embedder=TimestepEmbedder()
        self.s_embedder=StateEmbedder()


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

