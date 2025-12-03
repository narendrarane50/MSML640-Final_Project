import torch
import torch.nn as nn
import torch.nn.functional as F

class AUConditionedFER(nn.Module):
    def __init__(self, mae_model, num_classes=7, freeze_encoder=False):
        super().__init__()
        self.encoder = mae_model.encoder
        self.patch_embed = mae_model.patch_embed
        self.au_condition = mae_model.au_condition
        self.embed_dim = mae_model.embed_dim

        # Optional: freeze encoder weights
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.patch_embed.parameters():
                p.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, imgs, aus):
        B = imgs.shape[0]

        # ---- Patchify ----
        x = self.patch_embed(imgs)            # (B, embed_dim, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)      # (B, N, embed_dim)
        N = x.shape[1]

        # ---- AU Conditioning ----
        au_embed = self.au_condition(aus, N)
        x = x + au_embed

        # ---- Encode ----
        x = self.encoder(x)

        # ---- Aggregate patches ----
        x = x.mean(dim=1)                     # Global average pooling across patches

        # ---- Classify ----
        logits = self.classifier(x)
        return logits
