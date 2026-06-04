"""
PhotoEnhancerNet — multi-scale EfficientNet-B0 with a differentiable image-stats branch.

Input : (B, 3, 224, 224) normalized RGB image (ImageNet stats).
Output: (B, 15)           enhancement parameters in their valid ranges.

Architecture:
  backbone   EfficientNet-B0 (pretrained), feature map (B, 1280, 7, 7)
  pooling    global avg-pool (B, 1280) + 2×2 regional avg-pool (B, 1024)
  stats      differentiable image-statistics encoder (B, 8) — luminance,
             dark/bright ratio, per-channel means, warm ratio. These are
             the same signals the rule-based system uses, now learnable.
  head       linear + residual block → 15 params
"""

import torch
import torch.nn as nn
import timm

from pipeline import NUM_PARAMS, PARAM_RANGES


class ImageStatsEncoder(nn.Module):
    """
    Compute differentiable image statistics from an ImageNet-normalised input.
    Outputs an 8-dim vector of scene-relevant scalars that match the features
    used by the C# rule-based classifier (luminance, dark/bright ratio, RGB
    means, warm ratio).  Soft sigmoid thresholds keep gradients flowing.
    """

    NUM_STATS = 8

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) ImageNet-normalised → (B, 8)"""
        rgb = x * self.std + self.mean          # unnormalize to [0, 1]

        lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]  # (B, H, W)

        lum_mean = lum.mean(dim=[1, 2])         # (B,)

        # Manual variance avoids std() edge cases in ONNX
        lum_var  = ((lum - lum_mean[:, None, None]) ** 2).mean(dim=[1, 2])
        lum_std  = (lum_var + 1e-8).sqrt()

        # Soft thresholds for differentiability (sigmoid ≈ heaviside)
        dark_ratio   = torch.sigmoid((0.22 - lum) * 20).mean(dim=[1, 2])
        bright_ratio = torch.sigmoid((lum - 0.78) * 20).mean(dim=[1, 2])

        r_mean = rgb[:, 0].mean(dim=[1, 2])
        g_mean = rgb[:, 1].mean(dim=[1, 2])
        b_mean = rgb[:, 2].mean(dim=[1, 2])

        # Warm ratio: pixels where R > B by a meaningful margin (sunset signal)
        warm = torch.sigmoid((rgb[:, 0] - rgb[:, 2] - 0.10) * 20).mean(dim=[1, 2])

        return torch.stack(
            [lum_mean, lum_std, dark_ratio, bright_ratio, r_mean, g_mean, b_mean, warm],
            dim=1,
        )  # (B, 8)


class PhotoEnhancerNet(nn.Module):
    """
    Predicts 15 global enhancement parameters from a single image.

    Compared to the original single-vector design:
    - Multi-scale pooling captures both global context and regional patterns
      (2×2 grid picks up top/bottom sky vs ground, left/right imbalance, etc.)
    - Stats branch injects photometric prior knowledge directly, reducing the
      amount the backbone must re-learn from scratch
    - Residual head block stabilises training and prevents gradient vanishing
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Feature extractor — return feature map, apply our own pooling below
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",     # (B, 1280, 7, 7) for 224×224 input
        )
        feat_ch = self.backbone.num_features    # 1280 for B0

        # Multi-scale pooling
        self.global_pool   = nn.AdaptiveAvgPool2d(1)    # → (B, 1280, 1, 1)
        self.regional_pool = nn.AdaptiveAvgPool2d(2)    # → (B, C, 2, 2) after projection

        # 1×1 conv reduces channel count before regional flatten (keeps combined dim reasonable)
        self.region_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        # combined_dim = 1280 (global) + 256*4 (2×2 regional) + 8 (stats)
        combined_dim = feat_ch + 256 * 4 + ImageStatsEncoder.NUM_STATS

        self.stats_enc = ImageStatsEncoder()

        # Regression head with one residual block
        hidden = 512
        self.head_in = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.head_res = nn.Sequential(
            nn.Dropout(p=0.20),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.head_out = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(hidden, 256),
            nn.SiLU(),
            nn.Linear(256, NUM_PARAMS),
        )

        self._sym_idx = [i for i, (lo, _) in enumerate(PARAM_RANGES) if lo < 0]   # tanh × 100
        self._pos_idx = [i for i, (lo, _) in enumerate(PARAM_RANGES) if lo == 0]  # sigmoid × 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224) float32 — ImageNet-normalised
        returns: (B, 15) float32 in valid parameter ranges
        """
        feat_map = self.backbone(x)                         # (B, 1280, 7, 7)

        # Global average pooling
        g = self.global_pool(feat_map).flatten(1)           # (B, 1280)

        # 2×2 regional pooling (captures spatial layout)
        r = self.regional_pool(self.region_proj(feat_map))  # (B, 256, 2, 2)
        r = r.flatten(1)                                    # (B, 1024)

        # Differentiable image statistics
        stats = self.stats_enc(x)                           # (B, 8)

        combined = torch.cat([g, r, stats], dim=1)          # (B, 2312)

        h   = self.head_in(combined)
        h   = h + self.head_res(h)      # residual connection
        raw = self.head_out(h)          # (B, 15)

        out = torch.empty_like(raw)
        out[:, self._sym_idx] = torch.tanh(raw[:, self._sym_idx]) * 100.0
        out[:, self._pos_idx] = torch.sigmoid(raw[:, self._pos_idx]) * 100.0
        return out
