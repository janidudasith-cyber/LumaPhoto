"""
Training losses for the photo enhancement model.

  PhotoLoss = w_l1 * L1 + w_perc * Perceptual + w_ssim * (1 - SSIM)

- L1         : per-pixel accuracy, fast, stable
- Perceptual : VGG19 feature matching — preserves texture and structure
               without enforcing exact pixel positions
- SSIM       : structural similarity — rewards matching local contrast patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


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

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d((img1_p ** 2), kernel, groups=C) - mu1_sq
    sigma2_sq = F.conv2d((img2_p ** 2), kernel, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1_p * img2_p, kernel, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Perceptual / VGG loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Compute feature-matching loss using layers from a frozen VGG19.
    Uses relu2_2 and relu3_3 features for a good texture / structure balance.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        feats = vgg.features
        self.slice1 = feats[:9]   # up to relu2_2
        self.slice2 = feats[:18]  # up to relu3_3
        for p in self.parameters():
            p.requires_grad_(False)

        # VGG normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._norm(pred)
        t = self._norm(target)
        loss  = F.l1_loss(self.slice1(p), self.slice1(t))
        loss += F.l1_loss(self.slice2(p), self.slice2(t)) * 0.5
        return loss


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------

class PhotoLoss(nn.Module):
    def __init__(self, w_l1: float = 1.0, w_perc: float = 0.1, w_ssim: float = 0.5):
        super().__init__()
        self.w_l1   = w_l1
        self.w_perc = w_perc
        self.w_ssim = w_ssim
        self.perc   = VGGPerceptualLoss()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        l1_loss   = F.l1_loss(pred, target)
        perc_loss = self.perc(pred, target)
        ssim_loss = 1.0 - ssim(pred, target)

        # Small regulariser: penalise very large parameter magnitudes to prevent
        # the model from producing unrealistic extreme edits.
        reg_loss = (params.abs() / 100.0).mean() * 0.01

        loss = (self.w_l1   * l1_loss
              + self.w_perc * perc_loss
              + self.w_ssim * ssim_loss
              + reg_loss)

        return loss, {
            "l1":   l1_loss.item(),
            "perc": perc_loss.item(),
            "ssim": ssim_loss.item(),
            "reg":  reg_loss.item(),
        }
