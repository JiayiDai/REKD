import torch
import torch.nn as nn
from transformers import ViTModel

class ViT(nn.Module):
    def __init__(self, args, encoding=False, if_t=False):
        super().__init__()
        if if_t:
            self.vit = ViTModel.from_pretrained(args.model_form_t, cache_dir="hf_cache", local_files_only=True)
        else:
            self.vit = ViTModel.from_pretrained(args.model_form, cache_dir="hf_cache", local_files_only=True)
        self.encoding = encoding
        self.embedding = self.vit.embeddings
        self.hidden_size = self.vit.config.hidden_size

    def forward(self, x):#x is patch_embeddings for f; pixel values for g
        if self.encoding:
            outputs = self.vit.encoder(x)
            features = outputs.last_hidden_state#[128, 197, 768]
            #features = self.vit.layernorm(features)
            cls_token = features[:, 0]  # (B, 768)
            return cls_token
        else:
            outputs = self.vit(x)
            features = outputs.last_hidden_state
            return features[:, 1:]#[128, 196, 768]