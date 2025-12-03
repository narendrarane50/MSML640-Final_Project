import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

class AUConditionedMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, au_dim=20, au_embed_dim=128):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=3, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Project AUs into same embedding dimension as patches
        self.au_embed = nn.Sequential(
            nn.Linear(au_dim, au_embed_dim),
            nn.GELU(),
            nn.Linear(au_embed_dim, embed_dim)
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=2048)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # Decoder (maps from latent dim → RGB patches)
        patch_dim = 3 * patch_size * patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_dim)
        )

    def forward(self, imgs, aus):
        """
        imgs: (B, 3, H, W)
        aus:  (B, au_dim)
        """
        # 1. Patch embedding
        x = self.patch_embed(imgs)  # (B, N, D)
        B, N, D = x.shape

        # 2. AU conditioning
        au_token = self.au_embed(aus).unsqueeze(1).repeat(1, N, 1)
        x = x + au_token

        # 3. Encode + Decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)  # (B, N, patch_dim)

        # 4. Reconstruct image from patches
        patch_size = self.patch_size
        recon = decoded.view(B, int(imgs.size(2)/patch_size), int(imgs.size(3)/patch_size), 3, patch_size, patch_size)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, 3, imgs.size(2), imgs.size(3))  # (B, 3, H, W)

        return recon

class AUConditionedFER(nn.Module):
    def __init__(self, pretrained_mae, num_classes=7):
        super().__init__()
        self.encoder = pretrained_mae.encoder
        self.patch_embed = pretrained_mae.patch_embed
        self.au_condition = pretrained_mae.au_condition
        self.embed_dim = pretrained_mae.embed_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, imgs, aus):
        x = self.patch_embed(imgs)                          # (B, N, 768)
        au_token = self.au_condition(aus, x.shape[1])       # (B, N, 768)
        x = x + au_token
        features = self.encoder(x).mean(dim=1)              # mean pooling over patches
        return self.classifier(features)
