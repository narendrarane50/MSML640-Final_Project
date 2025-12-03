# models/pose_normalizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseNormalizer(nn.Module):
    """
    Enhanced Spatial Transformer Network (2D affine) for pose normalization.
    Includes BatchNorm and Dropout for more stable training on RAF-DB.
    """
    def __init__(self, in_ch=3, input_size=224):
        super().__init__()
        # Localization network (predicts affine parameters)
        self.localization = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)    # 56 -> 28
        )

        # Dynamically determine flattened size
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        with torch.no_grad():
            out = self.localization(dummy)
            flat_dim = out.view(1, -1).size(1)

        # Fully connected layers to predict 2x3 affine transform
        self.fc_loc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(64, 6)
        )

        # Initialize affine transform to near-identity
        self.fc_loc[-1].weight.data.zero_()
        identity_bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        noise = torch.randn_like(identity_bias) * 0.01
        self.fc_loc[-1].bias.data.copy_(identity_bias + noise)

    def forward(self, x):
        # Predict affine transform
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        # Apply affine transformation
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        x_out = F.grid_sample(
            x, grid, align_corners=False, mode="bilinear", padding_mode="border"
        )
        return x_out
