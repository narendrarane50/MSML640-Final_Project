from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal
from models.mae import MAEEncoder, MAEDecoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import trunc_normal_, PatchEmbed, MLP, Attention, TransformerBlock

class AUToken(nn.Module):
    """Encodes a vector of AUs (B, num_aus) into a single token (B, 1, D)."""
    def __init__(self, num_aus: int, out_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_aus, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        self.token_cls = nn.Parameter(torch.zeros(1, 1, out_dim))  
        trunc_normal_(self.token_cls, std=0.02)

    def forward(self, aus: torch.Tensor) -> torch.Tensor:
        z = self.mlp(aus)  
        z = z.unsqueeze(1)  
        
        return z + self.token_cls.expand(z.size(0), -1, -1)


class AUFiLM(nn.Module):
    
    def __init__(self, num_aus: int, hidden: int, dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.proj = nn.Sequential(
            nn.Linear(num_aus, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim * depth),  
        )

    def forward(self, aus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = aus.size(0)
        gb = self.proj(aus)  
        gb = gb.view(B, self.depth, 2, -1)  
        gamma = gb[:, :, 0, :].unsqueeze(2)  
        beta = gb[:, :, 1, :].unsqueeze(2)   
        return gamma, beta

@dataclass
class AUConditionedMAEConfig:
    
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    decoder_mlp_ratio: float = 4.0

    
    mask_ratio: float = 0.50

    
    num_aus: int = 20
    conditioning: Literal["tokens", "film", "both"] = "both"
    au_hidden: int = 128
    film_hidden: int = 256

    
    num_classes: Optional[int] = 7  
    cls_pool: Literal["cls", "mean"] = "cls"

    
    recon_loss: Literal["l2", "l1"] = "l2"


class AUConditionedMAE(nn.Module):
    def __init__(self, cfg: AUConditionedMAEConfig):
        super().__init__()
        self.cfg = cfg

        
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        
        self.use_tokens = cfg.conditioning in ("tokens", "both")
        self.use_film = cfg.conditioning in ("film", "both")

        if self.use_tokens:
            self.au_token = AUToken(cfg.num_aus, cfg.embed_dim, hidden=cfg.au_hidden)

        if self.use_film:
            self.film = AUFiLM(cfg.num_aus, hidden=cfg.film_hidden, dim=cfg.embed_dim, depth=cfg.depth)

        
        self.encoder = MAEEncoder(cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio)

        self.encoder_to_decoder = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        self.decoder = MAEDecoder(
        embed_dim=cfg.decoder_embed_dim,
        depth=cfg.decoder_depth,
        num_heads=cfg.decoder_num_heads,
        mlp_ratio=cfg.decoder_mlp_ratio,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        )

        
        if cfg.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.num_classes)
            )
        else:
            self.classifier = None

        
        if self.use_film:
            
            self.film_blocks = nn.ModuleList()
            for blk in self.encoder.blocks:
                self.film_blocks.append(blk)
            

    

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        
        p = self.cfg.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.cfg.image_size, "Input size must match config.image_size"
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        
        p = self.cfg.patch_size
        B, N, D = patches.shape
        C = self.cfg.in_chans
        H = W = self.cfg.image_size
        h = H // p
        w = W // p
        x = patches.view(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x

    

    def forward(
        self,
        imgs: torch.Tensor,           
        aus: Optional[torch.Tensor],  
        labels: Optional[torch.Tensor] = None,  
        mask_ratio: Optional[float] = None,
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        
        cfg = self.cfg
        B = imgs.size(0)
        mask_ratio = cfg.mask_ratio if mask_ratio is None else mask_ratio

        
        x = self.patch_embed(imgs)  
        x = x + self.pos_embed[:, 1:self.num_patches + 1, :]  

        
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio)

        
        cls = self.cls_token.expand(B, -1, -1)
        seq = [cls]

        if self.use_tokens:
            if aus is None:
                raise ValueError("AUs are required when conditioning mode includes 'tokens' or 'both'.")
            au_tok = self.au_token(aus)  
            seq.append(au_tok)

        seq.append(x_vis)
        x_enc = torch.cat(seq, dim=1)  

        
        pos = []
        pos.append(self.pos_embed[:, :1, :])  
        if self.use_tokens:
            
            pos.append(self.pos_embed[:, 1:2, :])
        
        if len(pos) > 0:
            x_enc[:, :len(pos), :] = x_enc[:, :len(pos), :] + torch.cat(pos, dim=1)

        
        if self.use_film:
            if aus is None:
                raise ValueError("AUs are required when conditioning mode includes 'film' or 'both'.")
            gamma, beta = self.film(aus)  

            
            for i, blk in enumerate(self.encoder.blocks):
                
                h = blk.norm1(x_enc)
                h = h * (1 + gamma[:, i]) + beta[:, i]
                x_enc = x_enc + blk.attn(h)
                
                x_enc = x_enc + blk.mlp(blk.norm2(x_enc))
            x_enc = self.encoder.norm(x_enc)
        else:
            x_enc = self.encoder(x_enc)

        
        out: Dict[str, Any] = {}
        if self.classifier is not None:
            if cfg.cls_pool == "cls":
                pooled = x_enc[:, 0, :]
            else:
                pooled = x_enc.mean(dim=1)
            logits = self.classifier(pooled)
            out["logits"] = logits
            if labels is not None:
                
                out["loss_cls"] = F.cross_entropy(logits, labels, label_smoothing=0.1)

        
        x_dec = self.encoder_to_decoder(x_enc)

        
        cls_dec = x_dec[:, :1, :]
        x_vis_dec = x_dec[:, (2 if self.use_tokens else 1):, :]  

        
        B, N, _ = x.shape  
        len_keep = x_vis_dec.size(1)
        len_mask = N - len_keep

        mask_tokens = self.mask_token.expand(B, len_mask, -1)
        x_ = torch.cat([x_vis_dec, mask_tokens], dim=1)  
        
        index = ids_restore.unsqueeze(-1).repeat(1, 1, x_.size(-1))
        x_ = torch.gather(x_, dim=1, index=index)
        
        x_full = torch.cat([cls_dec, x_], dim=1)  

        pred = self.decoder(x_full)  
        pred_patches = pred[:, 1:, :]  

        
        target = self.patchify(imgs)
        if cfg.recon_loss == "l2":
            loss_recon_per_patch = (pred_patches - target) ** 2
        else:
            loss_recon_per_patch = (pred_patches - target).abs()

        
        loss_recon = (loss_recon_per_patch * mask.unsqueeze(-1)).sum() / (mask.sum() * target.size(-1) + 1e-8)

        out["pred"] = pred_patches
        out["mask"] = mask
        out["loss_recon"] = loss_recon

        
        with torch.no_grad():
            recon_img = self.unpatchify(pred_patches)
            out["recon_img"] = recon_img.clamp(0, 1)

        
        if "loss_cls" in out and return_loss:
            out["loss_total"] = loss_recon + 0.3 * out["loss_cls"]
        elif return_loss:
            out["loss_total"] = loss_recon

        return out
