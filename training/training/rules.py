"""
Python port of ImageProcessor.Analyze + ComputeAutoParams from C#.

This lets us generate training labels (15 enhancement parameters) for any
image without external datasets.  The NN then learns to replicate and
generalise these rules — training entirely from the user's own photos.
"""

import math
import numpy as np
from PIL import Image


# ── Scene types (must match C# SceneType enum order) ──────────────────────────

SCENES = ["General", "Daylight", "Portrait", "Landscape", "Sunset",
          "LowLight", "Night", "HDR", "HighKey"]

PARAM_NAMES = [
    "exposure", "brilliance", "highlights", "shadows", "contrast",
    "brightness", "black_point", "saturation", "vibrance", "warmth",
    "tint", "sharpness", "definition", "noise", "vignette",
]
NUM_PARAMS = len(PARAM_NAMES)

# P indices (shorthand)
EXP, BRI, HIG, SHA, CON, BRT, BPT, SAT, VIB, WRM, TNT, SHP, DEF, NRS, VIG = range(15)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def analyze(img: Image.Image) -> dict:
    """
    Port of ImageProcessor.Analyze.
    img: PIL Image (RGB).
    Returns a dict with all scene statistics + scene name.
    """
    img_np = np.array(img.convert("RGB"), dtype=np.float32)  # (H, W, 3)
    h, w = img_np.shape[:2]

    # Sample every Nth pixel — ≥ 40 000 samples
    step = max(1, int(math.sqrt(w * h / 40000.0)))
    flat = img_np[::step, ::step].reshape(-1, 3)  # (N, 3)

    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    lum = r * 0.299 + g * 0.587 + b * 0.114
    count = len(flat)

    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
    l_mean = lum.mean()

    # Variance pass
    l_std    = float(lum.std())
    contrast = float(((r - r_mean)**2 + (g - g_mean)**2 + (b - b_mean)**2).mean()**0.5 / 3)

    dark_ratio    = float((lum < 55).sum()  / count)
    bright_ratio  = float((lum > 200).sum() / count)
    hotspot_ratio = float((lum > 230).sum() / count)

    # Skin tone
    rg_ratio = r / np.maximum(g, 1.0)
    skin = ((r > 100) & (r > g) & (g > b) & (r > b * 1.15) &
            (rg_ratio >= 1.05) & (rg_ratio <= 1.65) & (lum > 60) & (lum < 210))
    skin_ratio = float(skin.sum() / count)

    # Sky
    sky  = (b > r * 1.1) & (b > g * 1.05) & (lum > 100)
    sky_ratio = float(sky.sum() / count)

    # Green vegetation
    green = (g > r * 1.1) & (g > b * 1.1) & (lum > 40)
    green_ratio = float(green.sum() / count)

    # Warm (sunset/fire)
    warm = (r > 180) & (r > g * 1.3) & (r > b * 1.8)
    warm_ratio = float(warm.sum() / count)

    wb = float(r_mean - b_mean)
    cs = float(max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean))

    # ── Scene classification — same priority order as C# ──────────────────────
    if l_mean < 42 and dark_ratio > 0.55 and 0.004 < hotspot_ratio < 0.18:
        scene = "Night"
    elif l_std > 68 and dark_ratio > 0.12 and bright_ratio > 0.12:
        scene = "HDR"
    elif skin_ratio > 0.12:
        scene = "Portrait"
    elif warm_ratio > 0.12 and 70 < l_mean < 165:
        scene = "Sunset"
    elif sky_ratio > 0.15 or green_ratio > 0.20:
        scene = "Landscape"
    elif l_mean < 62 or dark_ratio > 0.45:
        scene = "LowLight"
    elif l_mean > 162 and dark_ratio < 0.05:
        scene = "HighKey"
    elif l_mean > 95 and l_std > 28:
        scene = "Daylight"
    else:
        scene = "General"

    return dict(
        scene=scene, l_mean=float(l_mean), l_std=l_std, contrast=contrast,
        dark_ratio=dark_ratio, bright_ratio=bright_ratio,
        r_mean=float(r_mean), g_mean=float(g_mean), b_mean=float(b_mean),
        skin_ratio=skin_ratio, sky_ratio=sky_ratio,
        green_ratio=green_ratio, warm_ratio=warm_ratio,
        hotspot_ratio=hotspot_ratio, wb=wb, cs=cs,
    )


def compute_auto_params(a: dict) -> np.ndarray:
    """
    Port of ImageProcessor.ComputeAutoParams.
    Returns a (15,) float32 array in the same parameter order as pipeline.py.
    """
    p = np.zeros(NUM_PARAMS, dtype=np.float32)
    scene = a["scene"]
    l     = a["l_mean"]
    wb    = a["wb"]
    cs    = a["cs"]
    dr    = a["dark_ratio"]
    br    = a["bright_ratio"]
    hs    = a["hotspot_ratio"]
    co    = a["contrast"]

    if scene == "Night":
        p[EXP] = _clamp(round((80 - l) * 0.55), 0, 60)
        p[BRI] = 20
        p[HIG] = -38 if hs > 0.04 else -22
        p[SHA] = 45 if dr > 0.70 else 35
        p[CON] = 22
        p[BRT] = 5
        p[BPT] = 8
        p[VIB] = 18
        p[WRM] = 10 if wb < -15 else 0
        p[SHP] = 8
        p[DEF] = 10
        p[NRS] = 45 if l < 30 else 32
        p[VIG] = 16

    elif scene == "HDR":
        p[EXP] = _clamp(round((114 - l) * 0.30), -30, 28)
        p[BRI] = 15
        p[HIG] = _clamp(round(-br * 125), -72, -25)
        p[SHA] = _clamp(round(dr * 88), 22, 58)
        p[CON] = -5
        p[BPT] = 12
        p[SAT] = 8 if cs < 20 else 4
        p[VIB] = 20 if cs < 20 else 14
        p[WRM] = (_clamp(-round(wb * .25), -20, -5) if wb > 30
                  else _clamp(round(-wb * .25), 5, 20) if wb < -30 else 0)
        p[SHP] = 10
        p[DEF] = 20
        p[NRS] = 15 if l < 80 else 5
        p[VIG] = 10

    elif scene == "Portrait":
        p[EXP] = _clamp(round((122 - l) * 0.42), -55, 50)
        p[BRI] = 20 if l < 110 else (14 if l < 140 else 8)
        p[HIG] = -22 if br > 0.10 else -5
        p[SHA] = 32 if dr > 0.20 else (20 if dr > 0.10 else 10)
        p[CON] = 12 if co < 30 else (7 if co < 50 else 2)
        p[BRT] = 5 if l < 80 else 0
        p[BPT] = 8
        p[SAT] = 8 if cs < 20 else (4 if cs < 40 else 0)
        p[VIB] = 18 if cs < 20 else (12 if cs < 40 else 6)
        p[WRM] = -8 if wb > 40 else (15 if wb < -20 else (8 if wb < 0 else 3))
        p[SHP] = 8
        p[DEF] = 12
        p[NRS] = 15 if l < 70 else 0
        p[VIG] = 10

    elif scene == "Sunset":
        p[EXP] = _clamp(round((100 - l) * 0.42), -50, 36)
        p[BRI] = 6
        p[HIG] = (_clamp(round(-br * 100), -65, -22) if br > 0.15 else -22)
        p[SHA] = 15
        p[CON] = 18 if co < 35 else 12
        p[BPT] = 14
        p[SAT] = 22
        p[VIB] = 32
        p[WRM] = 0 if wb > 20 else 12
        p[SHP] = 12
        p[DEF] = 16
        p[VIG] = 16

    elif scene == "Landscape":
        p[EXP] = _clamp(round((108 - l) * 0.42), -55, 50)
        p[BRI] = 14 if l < 110 else 10
        p[HIG] = (_clamp(round(-br * 95), -65, -8) if br > 0.15
                  else (-18 if br > 0.06 else -5))
        p[SHA] = (_clamp(round(dr * 72), 12, 46) if dr > 0.30
                  else (15 if dr > 0.12 else 6))
        p[CON] = 24 if co < 40 else (14 if co < 60 else 7)
        p[BRT] = 5 if l < 80 else 0
        p[BPT] = 12
        p[SAT] = 18 if cs < 20 else (12 if cs < 40 else 6)
        p[VIB] = 28 if cs < 20 else (20 if cs < 40 else 12)
        p[WRM] = (_clamp(-round(wb * .28), -28, -5) if wb > 35
                  else _clamp(round(-wb * .25), 5, 22) if wb < -25 else 0)
        p[SHP] = 14
        p[DEF] = 20
        p[VIG] = 12

    elif scene == "LowLight":
        p[EXP] = _clamp(round((95 - l) * 0.42), -20, 52)
        p[BRI] = 14
        p[HIG] = -18 if br > 0.06 else -5
        p[SHA] = 26 if dr > 0.50 else (18 if dr > 0.35 else 12)
        p[CON] = 22 if co < 35 else (14 if co < 55 else 6)
        p[BRT] = 5 if l < 80 else 0
        p[BPT] = 8
        p[SAT] = 8 if cs < 15 else 4
        p[VIB] = 20 if cs < 15 else 14
        p[WRM] = (_clamp(-round(wb * .3), -25, -5) if wb > 30
                  else _clamp(round(-wb * .3), 5, 25) if wb < -30 else 0)
        p[SHP] = 10
        p[DEF] = 14
        p[NRS] = 25 if dr > 0.50 else 20
        p[VIG] = 8

    elif scene == "HighKey":
        p[EXP] = _clamp(round((140 - l) * 0.35), -30, 18)
        p[BRI] = 6
        p[HIG] = _clamp(round(-br * 85), -52, -10)
        p[SHA] = 4
        p[CON] = 14 if co < 25 else 6
        p[BPT] = 5
        p[SAT] = 8 if cs < 15 else 4
        p[VIB] = 14 if cs < 15 else 8
        p[WRM] = -8 if wb > 30 else (8 if wb < -30 else 0)
        p[SHP] = 12
        p[DEF] = 14
        p[VIG] = 6

    elif scene == "Daylight":
        p[EXP] = _clamp(round((112 - l) * 0.30), -25, 22)
        p[BRI] = 12 if l < 115 else 8
        p[HIG] = -20 if br > 0.12 else (-12 if br > 0.06 else -4)
        p[SHA] = 12 if dr > 0.15 else 5
        p[CON] = 16 if co < 40 else (10 if co < 58 else 2)
        p[BPT] = 10
        p[SAT] = 10 if cs < 20 else (5 if cs < 40 else 0)
        p[VIB] = 18 if cs < 20 else (12 if cs < 40 else 8)
        p[WRM] = (_clamp(-round(wb * .25), -22, -5) if wb > 30
                  else _clamp(round(-wb * .25), 5, 22) if wb < -30 else 0)
        p[SHP] = 12
        p[DEF] = 16
        p[VIG] = 10

    else:  # General
        p[EXP] = _clamp(round((112 - l) * 0.42), -55, 50)
        p[BRI] = 16 if l < 100 else (10 if l < 130 else 6)
        p[HIG] = (_clamp(round(-br * 95), -65, -8) if br > 0.15
                  else (-16 if br > 0.06 else -4))
        p[SHA] = (_clamp(round(dr * 72), 12, 46) if dr > 0.30
                  else (15 if dr > 0.12 else 6))
        p[CON] = (_clamp(round(20 + (35 - co) * .4), 10, 32) if co < 35
                  else (10 if co < 55 else 2))
        p[BRT] = 5 if l < 80 else 0
        p[BPT] = 10
        p[SAT] = 12 if cs < 15 else (6 if cs < 35 else 2)
        p[VIB] = 22 if cs < 15 else (16 if cs < 35 else 10)
        p[WRM] = (_clamp(-round(wb * .3), -28, -5) if wb > 30
                  else _clamp(round(-wb * .3), 5, 28) if wb < -30 else 0)
        p[SHP] = 12
        p[DEF] = 18
        p[NRS] = 12 if l < 70 else 0
        p[VIG] = 10

    # Final clamp
    for i in range(11):  p[i] = _clamp(p[i], -100, 100)
    for i in range(11, 15): p[i] = _clamp(p[i], 0, 100)

    return p


def label_image(img: Image.Image) -> np.ndarray:
    """Analyze an image and return its 15-parameter enhancement label."""
    return compute_auto_params(analyze(img))
