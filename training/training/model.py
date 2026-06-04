"""
EfficientNet-B0 backbone with a parameter-regression head.

Input : (B, 3, 224, 224) normalized RGB image (ImageNet stats).
Output: (B, 15)           enhancement parameters in their valid ranges.

The output can be fed directly into pipeline.apply_params().
"""

import torch
import torch.nn as nn
import timm

from pipeline import NUM_PARAMS, PARAM_RANGES


class PhotoEnhancerNet(nn.Module):
    """
    Predicts 15 global enhancement parameters from a single image.
    Uses an EfficientNet-B0 feature extractor (fast, 5.3 M params).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Feature extractor — EfficientNet-B0 pretrained on ImageNet-1k
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,       # drop the classifier head
            global_pool="avg",   # global average pooling
        )
        feat_dim = self.backbone.num_features  # 1280 for B0

        # Multi-layer regression head
        self.head = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(feat_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, NUM_PARAMS),
        )

        # Separate output scaling for symmetric vs positive-only params
        self._sym_idx = [i for i, (lo, _) in enumerate(PARAM_RANGES) if lo < 0]  # tanh × 100
        self._pos_idx = [i for i, (lo, _) in enumerate(PARAM_RANGES) if lo == 0] # sigmoid × 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224) float32 — ImageNet-normalised
        returns: (B, 15) float32 in valid parameter ranges
        """
        feats  = self.backbone(x)          # (B, 1280)
        raw    = self.head(feats)          # (B, 15)

        out = torch.empty_like(raw)
        out[:, self._sym_idx] = torch.tanh(raw[:, self._sym_idx]) * 100.0
        out[:, self._pos_idx] = torch.sigmoid(raw[:, self._pos_idx]) * 100.0
        return out
