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
        self.token_cls = nn.Parameter(torch.zeros(1, 1, out_dim))  # a learned AU "class" token bias
        trunc_normal_(self.token_cls, std=0.02)

    def forward(self, aus: torch.Tensor) -> torch.Tensor:
        z = self.mlp(aus)  # (B, D)
        z = z.unsqueeze(1)  # (B, 1, D)
        # Add learned bias token
        return z + self.token_cls.expand(z.size(0), -1, -1)


class AUFiLM(nn.Module):
    """Per-block FiLM: produce (gamma, beta) from AU embedding and modulate normalized features."""
    def __init__(self, num_aus: int, hidden: int, dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.proj = nn.Sequential(
            nn.Linear(num_aus, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim * depth),  # for each block: gamma and beta of size dim
        )

    def forward(self, aus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns gamma, beta each shaped (depth, B, 1, D)
        B = aus.size(0)
        gb = self.proj(aus)  # (B, 2*dim*depth)
        gb = gb.view(B, self.depth, 2, -1)  # (B, depth, 2, D)
        gamma = gb[:, :, 0, :].unsqueeze(2)  # (B, depth, 1, D)
        beta = gb[:, :, 1, :].unsqueeze(2)   # (B, depth, 1, D)
        return gamma, beta

@dataclass
class AUConditionedMAEConfig:
    # Image / patches
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    # Encoder (ViT)
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Decoder
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    decoder_mlp_ratio: float = 4.0

    # Masking
    mask_ratio: float = 0.50

    # AU conditioning
    num_aus: int = 20
    conditioning: Literal["tokens", "film", "both"] = "both"
    au_hidden: int = 128
    film_hidden: int = 256

    # Classification head (FER)
    num_classes: Optional[int] = 7  # set to None to disable classifier
    cls_pool: Literal["cls", "mean"] = "cls"

    # Loss
    recon_loss: Literal["l2", "l1"] = "l2"


class AUConditionedMAE(nn.Module):
    def __init__(self, cfg: AUConditionedMAEConfig):
        super().__init__()
        self.cfg = cfg

        # Patch embed + positional encodings (encoder side)
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        # AU conditioning modules
        self.use_tokens = cfg.conditioning in ("tokens", "both")
        self.use_film = cfg.conditioning in ("film", "both")

        if self.use_tokens:
            self.au_token = AUToken(cfg.num_aus, cfg.embed_dim, hidden=cfg.au_hidden)

        if self.use_film:
            self.film = AUFiLM(cfg.num_aus, hidden=cfg.film_hidden, dim=cfg.embed_dim, depth=cfg.depth)

        # Encoder / Decoder
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

        # Optional classifier head (FER)
        if cfg.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.num_classes)
            )
        else:
            self.classifier = None

        # For FiLM modulation we need to hook into encoder blocks' norms
        if self.use_film:
            # Replace encoder blocks with FiLM-aware versions
            self.film_blocks = nn.ModuleList()
            for blk in self.encoder.blocks:
                self.film_blocks.append(blk)
            # No structural change needed; FiLM applied around norm1 in forward path

    # -----------------------
    # Masking helpers
    # -----------------------

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        x: (B, N, D) without cls / au tokens.
        Returns:
          x_masked: subset of patches kept
          mask: (B, N) with 1 for removed, 0 for kept
          ids_restore: indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 3, H, W) -> (B, N, patch_dim)
        """
        p = self.cfg.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.cfg.image_size, "Input size must match config.image_size"
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, patch_dim) -> (B, 3, H, W)
        """
        p = self.cfg.patch_size
        B, N, D = patches.shape
        C = self.cfg.in_chans
        H = W = self.cfg.image_size
        h = H // p
        w = W // p
        x = patches.view(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x

    # -----------------------
    # Forward
    # -----------------------

    def forward(
        self,
        imgs: torch.Tensor,           # (B, C, H, W)
        aus: Optional[torch.Tensor],  # (B, num_aus) floats in [0..1] or intensities
        labels: Optional[torch.Tensor] = None,  # (B,) FER labels if classifier enabled
        mask_ratio: Optional[float] = None,
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          - 'loss_recon': reconstruction loss (if return_loss)
          - 'pred': decoder patch predictions
          - 'mask': binary mask (1=removed)
          - 'loss_cls' & 'logits' if classifier is enabled and labels provided
          - 'recon_img' reconstructed image (for visualization)
        """
        cfg = self.cfg
        B = imgs.size(0)
        mask_ratio = cfg.mask_ratio if mask_ratio is None else mask_ratio

        # ---- Encoder tokens
        x = self.patch_embed(imgs)  # (B, N, D)
        x = x + self.pos_embed[:, 1:self.num_patches + 1, :]  # add positional (skip cls slot)

        # Random masking over patch tokens (no cls/AU in mask)
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Prepare sequence: [CLS] (+ [AU]) + visible patches
        cls = self.cls_token.expand(B, -1, -1)
        seq = [cls]

        if self.use_tokens:
            if aus is None:
                raise ValueError("AUs are required when conditioning mode includes 'tokens' or 'both'.")
            au_tok = self.au_token(aus)  # (B,1,D)
            seq.append(au_tok)

        seq.append(x_vis)
        x_enc = torch.cat(seq, dim=1)  # (B, 1(+1) + N_vis, D)

        # Add pos embed for cls (and we already added patch pos above); AU token needs a position too.
        # Reuse cls positional slot for cls; for AU we shift by one (simple & works well).
        # Build a small positional slice matching sequence length:
        pos = []
        pos.append(self.pos_embed[:, :1, :])  # cls
        if self.use_tokens:
            # Use cls pos for AU as well (or a learnable separate parameter; this works fine)
            pos.append(self.pos_embed[:, 1:2, :])
        # no pos for visible patches here because x_vis already had positional added earlier
        if len(pos) > 0:
            x_enc[:, :len(pos), :] = x_enc[:, :len(pos), :] + torch.cat(pos, dim=1)

        # ---- FiLM modulation (if enabled): apply around each encoder block's norm1
        if self.use_film:
            if aus is None:
                raise ValueError("AUs are required when conditioning mode includes 'film' or 'both'.")
            gamma, beta = self.film(aus)  # (B, depth, 1, D)

            # Manual unroll to insert FiLM after norm1 in each block
            for i, blk in enumerate(self.encoder.blocks):
                # x = x + attn(norm1(x))  with FiLM on norm1(x)
                h = blk.norm1(x_enc)
                h = h * (1 + gamma[:, i]) + beta[:, i]
                x_enc = x_enc + blk.attn(h)
                # x = x + mlp(norm2(x))
                x_enc = x_enc + blk.mlp(blk.norm2(x_enc))
            x_enc = self.encoder.norm(x_enc)
        else:
            x_enc = self.encoder(x_enc)

        # ---- Optional classifier (FER)
        out: Dict[str, Any] = {}
        if self.classifier is not None:
            if cfg.cls_pool == "cls":
                pooled = x_enc[:, 0, :]
            else:
                pooled = x_enc.mean(dim=1)
            logits = self.classifier(pooled)
            out["logits"] = logits
            if labels is not None:
                # out["loss_cls"] = F.cross_entropy(logits, labels)
                out["loss_cls"] = F.cross_entropy(logits, labels, label_smoothing=0.1)

        # ---- Decoder: re-insert masked tokens
        x_dec = self.encoder_to_decoder(x_enc)

        # Remove special tokens to build full-length sequence for decoder:
        # We only keep the CLS for decoder input; AU token is primarily encoder-side context.
        cls_dec = x_dec[:, :1, :]
        x_vis_dec = x_dec[:, (2 if self.use_tokens else 1):, :]  # strip cls (+ AU if present)

        # Build full tokens including mask tokens
        B, N, _ = x.shape  # original patch token count (no cls/au)
        len_keep = x_vis_dec.size(1)
        len_mask = N - len_keep

        mask_tokens = self.mask_token.expand(B, len_mask, -1)
        x_ = torch.cat([x_vis_dec, mask_tokens], dim=1)  # still shuffled order
        # Unshuffle to original order
        index = ids_restore.unsqueeze(-1).repeat(1, 1, x_.size(-1))
        x_ = torch.gather(x_, dim=1, index=index)
        # Add back CLS at the front
        x_full = torch.cat([cls_dec, x_], dim=1)  # (B, 1 + N, Dd)

        pred = self.decoder(x_full)  # (B, 1+N, patch_dim)
        pred_patches = pred[:, 1:, :]  # drop cls

        # Reconstruction target
        target = self.patchify(imgs)
        if cfg.recon_loss == "l2":
            loss_recon_per_patch = (pred_patches - target) ** 2
        else:
            loss_recon_per_patch = (pred_patches - target).abs()

        # Only compute loss on masked patches (like MAE)
        loss_recon = (loss_recon_per_patch * mask.unsqueeze(-1)).sum() / (mask.sum() * target.size(-1) + 1e-8)

        out["pred"] = pred_patches
        out["mask"] = mask
        out["loss_recon"] = loss_recon

        # Reconstructed image (for visualization)
        with torch.no_grad():
            recon_img = self.unpatchify(pred_patches)
            out["recon_img"] = recon_img.clamp(0, 1)

        # Total loss if both heads present
        if "loss_cls" in out and return_loss:
            out["loss_total"] = loss_recon + 0.3 * out["loss_cls"]
        elif return_loss:
            out["loss_total"] = loss_recon

        return out
