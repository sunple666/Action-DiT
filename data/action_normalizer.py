import torch
from torch import nn

class ActionNormalizer(nn.Module):
    def __init__(self,action_min,action_max,eps=1e-6):
        super().__init__()
        action_min=torch.as_tensor(action_min,dtype=torch.float32).detach().clone()
        action_max=torch.as_tensor(action_max,dtype=torch.float32).detach().clone()
        assert action_min.ndim==action_max.ndim==1
        assert action_min.shape==action_max.shape
        assert torch.isfinite(action_min).all() and torch.isfinite(action_max).all()
        assert torch.all(action_max-action_min>eps)
        self.register_buffer("action_min",action_min)
        self.register_buffer("action_max",action_max)
        self.eps=eps


    def normalize(self,raw_action):
        assert raw_action.shape[-1]==self.action_min.shape[0]
        assert torch.all((raw_action>=self.action_min-self.eps) & (raw_action<=self.action_max+self.eps))
        return -1+2*(raw_action-self.action_min)/(self.action_max-self.action_min)

    def denormalize(self,normalized_action):
        assert normalized_action.shape[-1]==self.action_min.shape[0]
        assert torch.all((normalized_action>=-1-self.eps) & (normalized_action<=1+self.eps))
        normalized_action=normalized_action.clamp(-1,1)
        return self.action_min+(normalized_action+1)*(self.action_max-self.action_min)/2