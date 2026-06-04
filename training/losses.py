"""
Training losses for the photo enhancement model.

  PhotoLoss = w_l1   * L1
            + w_perc * Perceptual   (VGG19, 4 feature levels)
            + w_ssim * (1 - SSIM)
            + w_color * ColorMoment
            + param regularisation  (parameter-aware weights)

- L1          per-pixel accuracy, stable baseline
- Perceptual  VGG19 feature matching across 4 levels — preserves texture
              and structure without enforcing exact pixel positions
- SSIM        structural similarity — rewards matching local contrast patterns
- ColorMoment matches per-channel mean and std — prevents color drift in
              predicted outputs
- Reg         parameter-aware regularisation; exposure/shadows penalised
              lightly (often legitimately large), tint/vignette penalised
              more heavily (rarely needed in large magnitudes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from pipeline import PARAM_NAMES


# ---------------------------------------------------------------------------
# Structural Similarity (SSIM)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 3):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute mean SSIM between two (B, C, H, W) images in [0, 1].
    Returns a scalar in [0, 1] (higher = more similar).
    """
    C = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, C).to(img1.device)

    pad = window_size // 2
    img1_p = F.pad(img1, [pad] * 4, mode="reflect")
    img2_p = F.pad(img2, [pad] * 4, mode="reflect")

    mu1 = F.conv2d(img1_p, kernel, groups=C)
    mu2 = F.conv2d(img2_p, kernel, groups=C)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_p ** 2, kernel, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2_p ** 2, kernel, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1_p * img2_p, kernel, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Color moment matching
# ---------------------------------------------------------------------------

def color_moment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match the first two moments (mean, std) of each RGB channel.
    Fast O(B·C·H·W) and fully differentiable.  Prevents the model from
    shifting overall color balance or crushing contrast.
    """
    p_mean = pred.mean(dim=[2, 3])    # (B, C)
    t_mean = target.mean(dim=[2, 3])
    p_std  = pred.std(dim=[2, 3])
    t_std  = target.std(dim=[2, 3])
    return F.l1_loss(p_mean, t_mean) + F.l1_loss(p_std, t_std)


# ---------------------------------------------------------------------------
# Perceptual / VGG loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Feature-matching loss using four frozen VGG19 levels:
      relu1_2  — edges and fine texture (weight 0.5 — avoid over-sharpening)
      relu2_2  — mid-level texture (weight 1.0)
      relu3_3  — shapes and structure (weight 0.5)
      relu4_3  — high-level semantics (weight 0.25)
    """

    def __init__(self):
        super().__init__()
        vgg   = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        feats = vgg.features
        self.slice0 = feats[:4]    # relu1_2
        self.slice1 = feats[:9]    # relu2_2
        self.slice2 = feats[:18]   # relu3_3
        self.slice3 = feats[:27]   # relu4_3
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._norm(pred)
        t = self._norm(target)
        loss  = F.l1_loss(self.slice0(p), self.slice0(t)) * 0.50   # low-level edges
        loss += F.l1_loss(self.slice1(p), self.slice1(t)) * 1.00   # mid-level texture
        loss += F.l1_loss(self.slice2(p), self.slice2(t)) * 0.50   # structure
        loss += F.l1_loss(self.slice3(p), self.slice3(t)) * 0.25   # semantics
        return loss


# ---------------------------------------------------------------------------
# Parameter-aware regularisation weights
# ---------------------------------------------------------------------------
# Indexed to match PARAM_NAMES in pipeline.py.
# Higher weight = stronger penalty for large magnitudes.
#   Exposure and shadows frequently need large corrections → low penalty.
#   Tint is rarely needed → high penalty.
_PARAM_REG_W = torch.tensor([
    0.30,  # exposure      — legitimate to push hard in dark scenes
    0.50,  # brilliance
    0.50,  # highlights
    0.30,  # shadows       — legitimate to push hard in dark scenes
    0.50,  # contrast
    0.50,  # brightness
    0.60,  # black_point
    0.50,  # saturation
    0.40,  # vibrance
    0.50,  # warmth
    0.90,  # tint          — rarely needed, penalise strongly
    0.60,  # sharpness
    0.60,  # definition
    0.40,  # noise         — legitimately large for night scenes
    0.70,  # vignette      — stylistic; penalise to keep subtle
], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------

class PhotoLoss(nn.Module):
    def __init__(
        self,
        w_l1:   float = 1.0,
        w_perc: float = 0.10,
        w_ssim: float = 0.50,
        w_color: float = 0.20,
    ):
        super().__init__()
        self.w_l1    = w_l1
        self.w_perc  = w_perc
        self.w_ssim  = w_ssim
        self.w_color = w_color
        self.perc    = VGGPerceptualLoss()
        self.register_buffer("param_reg_w", _PARAM_REG_W)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        l1_loss    = F.l1_loss(pred, target)
        perc_loss  = self.perc(pred, target)
        ssim_loss  = 1.0 - ssim(pred, target)
        color_loss = color_moment_loss(pred, target)

        # Parameter-aware magnitude regularisation
        reg_loss = (params.abs() / 100.0 * self.param_reg_w.to(params.device)).mean() * 0.02

        loss = (self.w_l1   * l1_loss
              + self.w_perc * perc_loss
              + self.w_ssim * ssim_loss
              + self.w_color * color_loss
              + reg_loss)

        return loss, {
            "l1":    l1_loss.item(),
            "perc":  perc_loss.item(),
            "ssim":  ssim_loss.item(),
            "color": color_loss.item(),
            "reg":   reg_loss.item(),
        }
