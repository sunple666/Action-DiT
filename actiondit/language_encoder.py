import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel

class QwenEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,padding_side="left")
        self.qwen=AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            use_cache=False,
        )
        self.qwen.requires_grad_(False)
        self.qwen.eval()
        self._embedding_cache: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(False)
        self.qwen.eval()
        return self

    def forward(self,texts):
        device=next(self.qwen.parameters()).device
        missing_texts = list(dict.fromkeys(
            text for text in texts if text not in self._embedding_cache
        ))
        if missing_texts:
            tokenized=self.tokenizer(
                missing_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                embeddings=self.qwen(**tokenized).last_hidden_state[:,-1,:]
            for text, embedding in zip(missing_texts, embeddings):
                self._embedding_cache[text] = embedding.detach()
        return torch.stack([self._embedding_cache[text] for text in texts])
