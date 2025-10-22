# models/fer_backbone.py
import torch.nn as nn
import torchvision.models as tvm

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(*list(m.children())[:-1])  # pool -> [B, 2048, 1, 1]
        self.output_dim = 2048

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)

class FERClassifier(nn.Module):
    def __init__(self, backbone, num_classes=7, use_pose_normalizer=False):
        super().__init__()
        self.use_pose_normalizer = use_pose_normalizer
        self.backbone = backbone
        self.fc = nn.Linear(backbone.output_dim, num_classes)
        # pose normalizer injected externally to keep ablations clean
        self.pose_normalizer = None

    def attach_pose_normalizer(self, pose_normalizer):
        self.pose_normalizer = pose_normalizer
        self.use_pose_normalizer = True

    def forward(self, x):
        if self.use_pose_normalizer and self.pose_normalizer is not None:
            x = self.pose_normalizer(x)
        feats = self.backbone(x)
        logits = self.fc(feats)
        return logits
