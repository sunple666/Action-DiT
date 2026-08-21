import torch
import torch.nn as nn
import math
from actiondit.vision_encoder import DINOv2Encoder
from actiondit.language_encoder import QwenEncoder

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
        freq=torch.exp(-torch.arange(start=0,end=half,dtype=torch.float32,device=t.device)/half*math.log(max_period))
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

class ObservationEmbedder(nn.Module):
    def __init__(self,dino_dim,hidden_dim):
        super().__init__()
        self.dino=DINOv2Encoder()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=dino_dim,out_features=hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        )
    def forward(self,x):
        x=self.dino(x)
        x=self.mlp(x)
        return x

class LanguageEmbedder(nn.Module):
    def __init__(self,qwen_dim,hidden_dim):
        super().__init__()
        self.qwen=QwenEncoder()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=qwen_dim,out_features=hidden_dim,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)            
        )
    def forward(self,x):
        x=self.qwen(x)
        x=self.mlp(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self,hidden_dim,out_dim):
        super().__init__()
        self.norm=nn.LayerNorm(hidden_dim)
        self.mlp=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*2, bias=True)
        )
        self.linear=nn.Linear(in_features=hidden_dim,out_features=out_dim,bias=True)

        nn.init.constant_(self.mlp[-1].weight,0)
        nn.init.constant_(self.mlp[-1].bias,0)
        nn.init.constant_(self.linear.weight,0)
        nn.init.constant_(self.linear.bias,0) 
        
    def forward(self,x,c):
        shift, scale = self.mlp(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x

class ActionDiTBlock(nn.Module):
    def __init__(self,hidden_dim,mlp_hidden_dim,num_heads,hidden_size):
        super().__init__()
        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim)
        self.attn=nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=num_heads,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=hidden_dim,out_features=mlp_hidden_dim,bias=True),
            nn.GELU(),
            nn.Linear(in_features=mlp_hidden_dim,out_features=hidden_dim,bias=True)
        )
        self.condition_mlp=nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size*6,bias=True)
        )
        nn.init.constant_(self.condition_mlp[-1].bias,0)
        nn.init.constant_(self.condition_mlp[-1].weight,0)

    def forward(self,x,c):
        attn_gate,mlp_gate,attn_scale,attn_shift,mlp_scale,mlp_shift=self.condition_mlp(c).chunk(6,dim=-1)
        attn_input=modulate(self.norm1(x),scale=attn_scale,shift=attn_shift)
        h=x+attn_gate.unsqueeze(1)*self.attn(attn_input,attn_input,attn_input)[0]
        x=h+mlp_gate.unsqueeze(1)*self.mlp(modulate(self.norm2(h),scale=mlp_scale,shift=mlp_shift))
        return x

class ActionDiT(nn.Module):
    def __init__(self,action_dim,time_dim,state_dim,hidden_dim,depth=6,dino_dim=384,qwen_dim=1024,action_chunk=16,learn_sigma=True):
        super().__init__()

        self.out_dim=2*action_dim if learn_sigma else action_dim
        self.action_chunk=action_chunk

        self.x_embedder=ActionEmbedder(action_dim,hidden_dim)
        self.action_pos_embeddings=nn.Parameter(torch.zeros(1,action_chunk,hidden_dim))
        nn.init.normal_(self.action_pos_embeddings,mean=0.0,std=0.02)
        self.t_embedder=TimestepEmbedder(time_dim,hidden_dim)
        self.s_embedder=StateEmbedder(state_dim,hidden_dim)
        self.o_embedder=ObservationEmbedder(dino_dim,hidden_dim)
        self.l_embedder=LanguageEmbedder(qwen_dim,hidden_dim)

        self.blocks=nn.ModuleList([
            ActionDiTBlock(hidden_dim=hidden_dim,mlp_hidden_dim=hidden_dim*4,num_heads=8,hidden_size=hidden_dim)
            for _ in range(depth)
        ])
        self.final_layer=FinalLayer(hidden_dim,self.out_dim)

    def forward(
            self,
            noisy_actions,#动作[B,H,A]
            timesteps,#时间步[B]
            state_embeddings,#状态[B,S]
            observation_embeddings=None,#图像[B,3,224,224]
            language_embeddings=None):#语言List[B]
        assert noisy_actions.shape[1]==self.action_chunk
        noisy_actions=self.x_embedder(noisy_actions)#[B,H,A]->[B,H,D]
        noisy_actions=noisy_actions+self.action_pos_embeddings#[B,H,D]+[1,H,D]->[B,H,D]
        timesteps=self.t_embedder(timesteps)#[B]->[B,D]
        state_embeddings=self.s_embedder(state_embeddings)#[B,S]->[B,D]
        observation_embeddings=self.o_embedder(observation_embeddings)#[B,3,224,224]->[B,384]->[B,D]
        language_embeddings=self.l_embedder(language_embeddings)#List[B]->[B,1024]->[B,D]
        condition=timesteps+state_embeddings+observation_embeddings+language_embeddings

        for block in self.blocks:
            noisy_actions=block(noisy_actions,condition)
        output=self.final_layer(noisy_actions,condition)
        return output
