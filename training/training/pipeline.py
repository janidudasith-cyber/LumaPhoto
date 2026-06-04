"""
Differentiable implementation of LumaPhoto's AdjustPixel pipeline in PyTorch.
Mirrors ImageProcessor.cs so gradients can flow through enhancement parameters
during training.

Parameter order (index 0-14):
  0  exposure      [-100, 100]
  1  brilliance    [-100, 100]
  2  highlights    [-100, 100]
  3  shadows       [-100, 100]
  4  contrast      [-100, 100]
  5  brightness    [-100, 100]
  6  black_point   [-100, 100]
  7  saturation    [-100, 100]
  8  vibrance      [-100, 100]
  9  warmth        [-100, 100]
  10 tint          [-100, 100]
  11 sharpness     [0, 100]
  12 definition    [0, 100]
  13 noise         [0, 100]
  14 vignette      [0, 100]
"""

import torch
import torch.nn.functional as F

PARAM_NAMES = [
    "exposure", "brilliance", "highlights", "shadows", "contrast",
    "brightness", "black_point", "saturation", "vibrance", "warmth",
    "tint", "sharpness", "definition", "noise", "vignette",
]
NUM_PARAMS = len(PARAM_NAMES)

# Valid ranges for clamping
PARAM_RANGES = [
    (-100, 100),  # exposure
    (-100, 100),  # brilliance
    (-100, 100),  # highlights
    (-100, 100),  # shadows
    (-100, 100),  # contrast
    (-100, 100),  # brightness
    (-100, 100),  # black_point
    (-100, 100),  # saturation
    (-100, 100),  # vibrance
    (-100, 100),  # warmth
    (-100, 100),  # tint
    (0,    100),  # sharpness
    (0,    100),  # definition
    (0,    100),  # noise
    (0,    100),  # vignette
]


def apply_params(img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Apply enhancement parameters to a batch of images.

    img:    (B, 3, H, W) float32 in [0, 1]   (R, G, B channel order)
    params: (B, 15)      float32, in the ranges defined by PARAM_RANGES

    Returns: (B, 3, H, W) float32 in [0, 1]
    """
    B, _, H, W = img.shape

    # Unpack all params — each is (B,)
    (exposure, brilliance, highlights, shadows, contrast,
     brightness, black_point, saturation, vibrance, warmth,
     tint, _sharpness, _definition, _noise, vignette) = params.unbind(1)

    # Work in the [0, 255] pixel range to match the C# implementation exactly.
    x = img * 255.0  # (B, 3, H, W)

    # --- Exposure ---
    exp_factor = torch.pow(2.0, exposure / 100.0).view(B, 1, 1, 1)
    x = x * exp_factor

    # --- Brightness ---
    br = (brightness * 1.4).view(B, 1, 1, 1)
    x = x + br

    # --- Contrast ---
    co = (1.0 + contrast / 100.0).view(B, 1, 1, 1)
    x = (x - 128.0) * co + 128.0

    # --- Highlights / Shadows ---
    lum = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    hi  = torch.clamp((lum - 128.0) / 127.0, 0.0, 1.0)
    sh  = torch.clamp((128.0 - lum) / 128.0, 0.0, 1.0)
    hiD = (highlights.view(B, 1, 1, 1) * -0.9) * hi
    shD = (shadows.view(B, 1, 1, 1) *  1.1) * sh
    x = x + hiD + shD

    # --- Brilliance (midtone push) ---
    lum = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    mid_push = (128.0 - lum) * (brilliance.view(B, 1, 1, 1) / 100.0) * 0.36
    x = x + mid_push

    # --- Black Point ---
    # Positive BP → crush blacks (deepen shadows, adds depth).
    # Negative BP → lift blacks (faded/milky look, not used by auto enhance).
    # Use a smooth differentiable approximation via softplus to avoid hard branches.
    bp = black_point.view(B, 1, 1, 1)
    pos_bp = F.softplus(bp) - F.softplus(torch.zeros_like(bp))  # ≈ max(bp, 0)

    bl = pos_bp * 1.25
    scale = 255.0 / torch.clamp(255.0 - bl, min=1.0)
    x = (x - bl) * scale

    # --- Saturation ---
    lum = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    sat = (1.0 + saturation.view(B, 1, 1, 1) / 100.0)
    x = lum + (x - lum) * sat

    # --- Vibrance ---
    vib  = (vibrance.view(B, 1, 1, 1) / 100.0)
    maxc = x.max(dim=1, keepdim=True)[0]
    avg  = x.mean(dim=1, keepdim=True)
    boost = 1.0 + vib * (1.0 - torch.abs(maxc - avg) / 128.0)
    x = avg + (x - avg) * boost

    # --- Warmth / Tint ---
    w = (warmth * 0.55).view(B, 1, 1, 1)
    t = (tint   * 0.28).view(B, 1, 1, 1)
    r_ch = x[:, 0:1] + w + t
    g_ch = x[:, 1:2] - t * (0.18 / 0.28)
    b_ch = x[:, 2:3] - w + t
    x = torch.cat([r_ch, g_ch, b_ch], dim=1)

    # --- Vignette ---
    vy  = torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype)
    vx  = torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype)
    yy, xx = torch.meshgrid(vy, vx, indexing="ij")
    dist = (xx * xx + yy * yy).sqrt().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    vig_factor = 1.0 - torch.clamp(dist - 0.35, min=0.0) * (vignette.view(B, 1, 1, 1) / 85.0)
    x = x * vig_factor

    # Back to [0, 1]
    return torch.clamp(x / 255.0, 0.0, 1.0)


def params_to_dict(params: torch.Tensor) -> dict:
    """Convert a (15,) parameter tensor to a named dict."""
    return {name: params[i].item() for i, name in enumerate(PARAM_NAMES)}
