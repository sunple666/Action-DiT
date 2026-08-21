import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel

class QwenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_path=r"D:\model_cache\Qwen3-Embedding-0.6B"
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)
        self.qwen=AutoModel.from_pretrained(model_path,dtype=torch.float32)
        for param in self.qwen.parameters():
            param.requires_grad=False
        self.qwen.eval()

    def forward(self,texts):
        device="cuda" if torch.cuda.is_available() else "cpu"
        tokenized=self.tokenizer(texts,padding=True,truncation=True,max_length=64,return_tensors="pt").to(device)
        ret=self.qwen(**tokenized)
        return ret.last_hidden_state[:,-1,:]