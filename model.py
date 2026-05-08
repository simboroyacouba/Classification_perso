"""
Modèle de classification personnalisé - Toitures cadastrales
Backbones  : efficientnet_b0, efficientnet_b3, efficientnet_b4, resnet50
Attention  : none, se, cbam
Pooling    : avg + max concat → CustomPoolingConcatenate
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    resnet50,        ResNet50_Weights,
)

# Taille d'image recommandée par backbone
BACKBONE_IMAGE_SIZES = {
    "efficientnet_b0": 224,
    "efficientnet_b3": 299,
    "efficientnet_b4": 380,
    "resnet50":        224,
}


# =============================================================================
# BLOCS D'ATTENTION
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation — attention canal uniquement."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
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
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ResidualAttentionBlock(nn.Module):
    """CBAM avec connexion résiduelle — canal + spatial."""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x)) + x


def build_attention(attention_type, in_channels):
    """Fabrique le bloc d'attention selon le type."""
    if attention_type == 'cbam':
        return ResidualAttentionBlock(in_channels)
    elif attention_type == 'se':
        return SEBlock(in_channels)
    else:
        return nn.Identity()


# =============================================================================
# POOLING CUSTOM
# =============================================================================

class CustomPoolingConcatenate(nn.Module):
    """Concatène avg pool et max pool globaux → 2 × in_channels features."""
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avg(x).flatten(1), self.max(x).flatten(1)], dim=1)


# =============================================================================
# TÊTE DE CLASSIFICATION
# =============================================================================

class ClassificationHead(nn.Module):
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
# BACKBONE FACTORY
# =============================================================================

def _build_backbone(backbone_name, pretrained):
    """Retourne (backbone, out_channels)."""
    if backbone_name == "efficientnet_b0":
        w = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return efficientnet_b0(weights=w).features, 1280
    elif backbone_name == "efficientnet_b3":
        w = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        return efficientnet_b3(weights=w).features, 1536
    elif backbone_name == "efficientnet_b4":
        w = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        return efficientnet_b4(weights=w).features, 1792
    elif backbone_name == "resnet50":
        w   = ResNet50_Weights.DEFAULT if pretrained else None
        net = resnet50(weights=w)
        backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        return backbone, 2048
    else:
        raise ValueError(
            f"Backbone inconnu: '{backbone_name}'. "
            f"Choisir parmi: efficientnet_b0, efficientnet_b3, efficientnet_b4, resnet50"
        )


# =============================================================================
# MODÈLE COMPLET
# =============================================================================

class PersoClassifier(nn.Module):
    """
    Backbone + Attention + CustomPoolingConcatenate + ClassificationHead.
    Entrée  : (B, 3, H, W) normalisé ImageNet.
    Sortie  : (B, num_classes) logits.
    """
    def __init__(self, num_classes, backbone_name="efficientnet_b3",
                 attention_type="cbam", pretrained=True,
                 freeze_backbone=False, dropout=0.4):
        super().__init__()
        self.backbone_name  = backbone_name
        self.attention_type = attention_type

        self.backbone, out_ch = _build_backbone(backbone_name, pretrained)
        self.attention = build_attention(attention_type, out_ch)
        self.pool      = CustomPoolingConcatenate()
        self.head      = ClassificationHead(out_ch * 2, num_classes, dropout)

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)
        return self.head(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(num_classes, backbone="efficientnet_b3", attention="cbam",
                pretrained=True, freeze_backbone=False, dropout=0.4):
    """Point d'entrée unique — rétrocompatible avec l'ancienne signature."""
    return PersoClassifier(
        num_classes,
        backbone_name=backbone,
        attention_type=attention,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
