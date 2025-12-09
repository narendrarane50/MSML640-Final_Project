from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import trunc_normal_, PatchEmbed, MLP, Attention, TransformerBlock


class MAEEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, patch_size, in_chans):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x