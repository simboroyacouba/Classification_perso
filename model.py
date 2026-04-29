"""
Modèle de classification personnalisé - Toitures cadastrales
Architecture: EfficientNet-B3 + CBAM + CustomPoolingConcatenate + ClassificationHead
Reproduit l'esprit du modèle original (attention résiduelle + pooling custom)
en PyTorch pour comparaison homogène avec les autres modèles.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


# =============================================================================
# BLOCS D'ATTENTION (CBAM)
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ResidualAttentionBlock(nn.Module):
    """CBAM avec connexion résiduelle — équivalent du ResidualAttentionBlock original."""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x)) + x


# =============================================================================
# POOLING CUSTOM
# =============================================================================

class CustomPoolingConcatenate(nn.Module):
    """
    Concatène avg pool et max pool globaux → 2 × in_channels features.
    Équivalent du CustomPoolingConcatenate original.
    """
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        a = self.avg(x).flatten(1)
        m = self.max(x).flatten(1)
        return torch.cat([a, m], dim=1)


# =============================================================================
# TÊTE DE CLASSIFICATION
# =============================================================================

class ClassificationHead(nn.Module):
    """FC layers pour la classification finale."""
    def __init__(self, in_features, num_classes, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# =============================================================================
# MODÈLE COMPLET
# =============================================================================

class PersoClassifier(nn.Module):
    """
    EfficientNet-B3 + ResidualAttentionBlock + CustomPoolingConcatenate + ClassificationHead.
    Entrée : (B, 3, 299, 299) normalisé ImageNet.
    Sortie  : (B, num_classes) logits.
    """
    IMAGE_SIZE       = 299
    BACKBONE_CHANNELS = 1536  # canaux de sortie EfficientNet-B3

    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights          = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        effnet           = efficientnet_b3(weights=weights)
        self.backbone    = effnet.features          # → (B, 1536, H, W)
        self.attention   = ResidualAttentionBlock(self.BACKBONE_CHANNELS)
        self.pool        = CustomPoolingConcatenate()   # → (B, 3072)
        self.head        = ClassificationHead(self.BACKBONE_CHANNELS * 2, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)
        return self.head(x)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


def build_model(num_classes, pretrained=True, freeze_backbone=False):
    return PersoClassifier(num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
