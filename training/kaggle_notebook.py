# =============================================================================
# LumaPhoto — Kaggle Training Notebook
# Train PhotoEnhancerNet on FiveK + PPR10K + LOL → export enhancer_params.onnx
#
# HOW TO USE ON KAGGLE
# ─────────────────────
# 1. Create a new Kaggle notebook:
#    https://www.kaggle.com/notebooks → New Notebook → set to GPU T4 x2
#
# 2. Add datasets to your notebook via "+ Add data":
#
#    REQUIRED — pick at least one:
#    a) PPR10K 512px  ← upload once as a private dataset (5 GB, 33K portrait
#                       pairs — the most important one for portrait/landscape fix)
#                       Download from https://github.com/csjliang/PPR10K
#                       Get the "512px resized" version from their Google Drive.
#    b) "soumikrakshit/lol-dataset" — search Kaggle, add as-is (~1 GB, public)
#
#    OPTIONAL — FiveK JPEG (~1.5 GB, NOT the 50 GB RAW version):
#    c) Search Kaggle for "adobe fivek preprocessed" and add it.
#       Or skip it — PPR10K covers portraits better anyway.
#
#    AUTO — the script downloads DIV2K (~7 GB) for synthetic augmentation.
#
# 3. Paste this entire file into a single Kaggle code cell and run it.
#
# 4. When training finishes, enhancer_params.onnx appears in the
#    Kaggle output tab (right sidebar). Download it and drop it next to
#    LumaPhoto.exe — the app picks it up automatically on next launch.
#
# EXPECTED TRAINING TIME (Kaggle T4 GPU)
# ───────────────────────────────────────
#  LOL only               (~500 pairs)   ~20 min   (proof of concept only)
#  PPR10K 512px          (~33K pairs)    ~7 hours  ← good for portrait fix
#  PPR10K + LOL          (~33.5K pairs)  ~7.5 hours
#  + FiveK JPEG          (~38.5K pairs)  ~9 hours  ← best overall quality
#  + DIV2K synthetic     (+800 images)   +1 hour
#
# =============================================================================

import os, sys, subprocess, math, time, copy, random, shutil
from pathlib import Path

# ── 1. Install dependencies ───────────────────────────────────────────────────
print("Installing dependencies…")
subprocess.run(["pip", "install", "-q", "timm", "onnx", "onnxruntime"], check=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("⚠  No GPU found. Go to: Settings → Accelerator → GPU T4 x2")

# ── 2. Locate datasets ────────────────────────────────────────────────────────
# Kaggle mounts added datasets under /kaggle/input/<dataset-slug>/

# Print everything in /kaggle/input so we can see exact paths
print("=== /kaggle/input contents ===")
for item in sorted(Path("/kaggle/input").rglob("*")):
    if item.is_dir():
        print(f"  DIR  {item}")
    elif item.stat().st_size > 0:
        print(f"  FILE {item}  ({item.stat().st_size // 1024 // 1024} MB)")
print("==============================")

# Scan /kaggle/input and auto-detect datasets by their contents
FIVEK_ROOT   = None
PPR10K_ROOT  = None
LOL_ROOT     = None

for dirpath in sorted(Path("/kaggle/input").rglob("*")):
    if not dirpath.is_dir():
        continue
    contents = {p.name for p in dirpath.iterdir()}
    if "input" in contents and "expertC" in contents and FIVEK_ROOT is None:
        FIVEK_ROOT = str(dirpath)
    if "source" in contents and "target_a" in contents and PPR10K_ROOT is None:
        PPR10K_ROOT = str(dirpath)
    if "our485" in contents and LOL_ROOT is None:
        LOL_ROOT = str(dirpath)

# PPR10K: if not found as extracted folders, it may be a zip — extract it
if PPR10K_ROOT is None:
    import zipfile, tarfile
    for fpath in Path("/kaggle/input").rglob("*"):
        if not fpath.is_file():
            continue
        _out = Path("/kaggle/working/ppr10k")
        _out.mkdir(parents=True, exist_ok=True)
        extracted = False
        try:
            if zipfile.is_zipfile(fpath):
                print(f"Extracting PPR10K zip: {fpath.name} ...")
                with zipfile.ZipFile(fpath) as z:
                    z.extractall(_out)
                extracted = True
            elif tarfile.is_tarfile(fpath):
                print(f"Extracting PPR10K tar: {fpath.name} ...")
                with tarfile.open(fpath) as t:
                    t.extractall(_out)
                extracted = True
        except Exception as e:
            print(f"  Could not extract {fpath.name}: {e}")
        if extracted:
            for dp in sorted(_out.rglob("*")):
                if dp.is_dir():
                    contents = {p.name for p in dp.iterdir()}
                    if "source" in contents and "target_a" in contents:
                        PPR10K_ROOT = str(dp)
                        print(f"  PPR10K found at: {PPR10K_ROOT}")
                        break
            if PPR10K_ROOT:
                break

print(f"FiveK  : {FIVEK_ROOT  or 'not found'}")
print(f"PPR10K : {PPR10K_ROOT or 'not found'}")
print(f"LOL    : {LOL_ROOT    or 'not found'}")

if not PPR10K_ROOT and not LOL_ROOT and not FIVEK_ROOT:
    raise RuntimeError("No dataset found. Check that your datasets are attached in the Input panel.")

# ── 3. Auto-download DIV2K as synthetic augmentation source ──────────────────
DIV2K_DIR = Path("./data/div2k")
if not DIV2K_DIR.exists():
    print("Downloading DIV2K (800 high-res images for synthetic augmentation)…")
    DIV2K_DIR.mkdir(parents=True, exist_ok=True)
    url  = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    arch = DIV2K_DIR / "DIV2K_train_HR.zip"
    subprocess.run(["wget", "-q", "-O", str(arch), url], check=True)
    subprocess.run(["unzip", "-q", str(arch), "-d", str(DIV2K_DIR)], check=True)
    arch.unlink()
    print(f"DIV2K ready at {DIV2K_DIR}")

DIV2K_IMAGES = str(DIV2K_DIR / "DIV2K_train_HR")
if not Path(DIV2K_IMAGES).exists():
    DIV2K_IMAGES = str(DIV2K_DIR)   # flat layout fallback


# ── 4. Dataset loaders ────────────────────────────────────────────────────────

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

def _to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.functional.to_tensor(img)

def _resize_min(img: Image.Image, min_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) < min_side:
        s = min_side / min(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
    return img

def _sync_crop_flip(inp: Image.Image, tgt: Image.Image,
                    crop: int, flip: bool) -> tuple:
    """Apply identical random crop + optional flip to both images."""
    seed = random.randint(0, 2**31)
    aug  = transforms.Compose([
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(p=1.0 if flip else 0.0),
    ])
    torch.manual_seed(seed); random.seed(seed)
    inp_t = aug(_to_tensor(inp))
    torch.manual_seed(seed); random.seed(seed)
    tgt_t = aug(_to_tensor(tgt))
    return inp_t, tgt_t


def _find_fivek_dirs(root: str):
    """
    Auto-detect input and target directories for any FiveK community upload.
    Different Kaggle datasets use different subfolder names; this tries them all.
    Returns (input_dir, target_dir) or raises FileNotFoundError.
    """
    root = Path(root)

    # Walk up to 2 levels to handle datasets that add a wrapper folder
    search_roots = [root] + list(root.iterdir()) if root.is_dir() else [root]
    search_roots = [p for p in search_roots if p.is_dir()]

    INPUT_NAMES  = ["input", "Input", "INPUT", "raw", "source", "original", "low"]
    TARGET_NAMES = ["expertA", "ExpertA", "experta", "expert_a", "expert-a",
                    "expertC", "ExpertC", "expertc", "expert_c", "expert-c",
                    "retouched", "enhanced", "target", "high", "output"]

    for base in search_roots:
        inp_dir = next((base / n for n in INPUT_NAMES if (base / n).is_dir()), None)
        tgt_dir = next((base / n for n in TARGET_NAMES if (base / n).is_dir()), None)
        if inp_dir and tgt_dir:
            print(f"  FiveK layout detected: {inp_dir.name}/ → {tgt_dir.name}/")
            return inp_dir, tgt_dir

    # Last resort: find any two sibling dirs that contain images
    img_dirs = [p for p in root.rglob("*") if p.is_dir()
                and any(f.suffix.lower() in EXTS for f in p.iterdir()
                        if f.is_file())]
    if len(img_dirs) >= 2:
        print(f"  FiveK fallback layout: {img_dirs[0].name}/ → {img_dirs[1].name}/")
        return img_dirs[0], img_dirs[1]

    raise FileNotFoundError(
        f"Could not find input+target subdirs in {root}.\n"
        f"Contents: {[p.name for p in root.iterdir()]}"
    )


class FiveKDataset(Dataset):
    def __init__(self, root: str, crop: int = 256):
        self.inp_dir, self.tgt_dir = _find_fivek_dirs(root)
        # Match files that exist in both directories
        inp_names = {f.name for f in self.inp_dir.iterdir() if f.suffix.lower() in EXTS}
        tgt_names = {f.name for f in self.tgt_dir.iterdir() if f.suffix.lower() in EXTS}
        self.files = sorted(inp_names & tgt_names)
        if not self.files:
            # Try matching by stem if extensions differ (e.g. .tif input, .jpg target)
            inp_stems = {f.stem: f.name for f in self.inp_dir.iterdir() if f.suffix.lower() in EXTS}
            tgt_stems = {f.stem: f.name for f in self.tgt_dir.iterdir() if f.suffix.lower() in EXTS}
            common = sorted(inp_stems.keys() & tgt_stems.keys())
            self.inp_names = [inp_stems[s] for s in common]
            self.tgt_names = [tgt_stems[s] for s in common]
            self.files = common          # stems used as index keys
            self._stem_mode = True
        else:
            self.inp_names = self.tgt_names = self.files
            self._stem_mode = False
        self.crop = crop
        print(f"  FiveK: {len(self.files)} matched pairs")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        inp_name = self.inp_names[idx]
        tgt_name = self.tgt_names[idx]
        inp  = _resize_min(Image.open(self.inp_dir / inp_name).convert("RGB"), 288)
        tgt  = Image.open(self.tgt_dir / tgt_name).convert("RGB").resize(inp.size, Image.BICUBIC)
        flip = random.random() > 0.5
        return _sync_crop_flip(inp, tgt, self.crop, flip)


class PPR10KDataset(Dataset):
    def __init__(self, root: str, experts=("a","b","c"), crop: int = 256):
        src = Path(root) / "source"
        tdirs = [Path(root) / f"target_{e}" for e in experts]
        names = sorted(f.name for f in src.iterdir() if f.suffix.lower() in EXTS)
        self.pairs = [(src / n, td / n) for n in names for td in tdirs
                      if (td / n).exists()]
        self.crop = crop

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp_p, tgt_p = self.pairs[idx]
        inp  = _resize_min(Image.open(inp_p).convert("RGB"), 288)
        tgt  = Image.open(tgt_p).convert("RGB").resize(inp.size, Image.BICUBIC)
        flip = random.random() > 0.5
        return _sync_crop_flip(inp, tgt, self.crop, flip)


class LOLDataset(Dataset):
    def __init__(self, root: str, crop: int = 256):
        self.low_dir  = Path(root) / "our485" / "low"
        self.high_dir = Path(root) / "our485" / "high"
        self.files = sorted(f.name for f in self.low_dir.iterdir()
                            if f.suffix.lower() in EXTS)
        self.crop = crop

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        inp  = _resize_min(Image.open(self.low_dir  / name).convert("RGB"), 288)
        tgt  = Image.open(self.high_dir / name).convert("RGB").resize(inp.size, Image.BICUBIC)
        flip = random.random() > 0.5
        return _sync_crop_flip(inp, tgt, self.crop, flip)


class SyntheticDataset(Dataset):
    """Degrades clean images to create input→target pairs. No labeling needed."""
    def __init__(self, image_dir: str, crop: int = 256, max_images: int = 5000):
        self.files = sorted(p for p in Path(image_dir).rglob("*")
                            if p.suffix.lower() in EXTS)[:max_images]
        self.crop  = crop

    def __len__(self): return len(self.files)

    def _degrade(self, x: torch.Tensor) -> torch.Tensor:
        ev  = random.uniform(-1.5, 1.5)
        x   = x * (2.0 ** ev)
        co  = random.uniform(0.60, 1.60)
        x   = (x - 0.5) * co + 0.5
        lum = x[0:1]*0.299 + x[1:2]*0.587 + x[2:3]*0.114
        sat = random.uniform(0.3, 1.8)
        x   = lum + (x - lum) * sat
        wt  = random.uniform(-0.12, 0.12)
        x[0], x[2] = x[0] + wt, x[2] - wt
        if random.random() < 0.3: x = x - random.uniform(0.0, 0.12)
        if random.random() < 0.3: x = x * random.uniform(1.1, 1.4)
        return torch.clamp(x, 0.0, 1.0)

    def __getitem__(self, idx):
        img = _resize_min(Image.open(self.files[idx]).convert("RGB"), 288)
        seed = random.randint(0, 2**31)
        aug  = transforms.Compose([transforms.RandomCrop(self.crop),
                                   transforms.RandomHorizontalFlip()])
        torch.manual_seed(seed); random.seed(seed)
        clean = aug(_to_tensor(img))
        return self._degrade(clean), clean


# Build combined dataset
datasets = []
if FIVEK_ROOT:
    ds = FiveKDataset(FIVEK_ROOT); datasets.append(ds)
    print(f"FiveK:    {len(ds):>6,} pairs (auto-detected layout)")
if PPR10K_ROOT:
    ds = PPR10KDataset(PPR10K_ROOT); datasets.append(ds)
    print(f"PPR10K:   {len(ds):>6,} pairs")
if LOL_ROOT:
    ds = LOLDataset(LOL_ROOT); datasets.append(ds)
    print(f"LOL:      {len(ds):>6,} pairs")
if Path(DIV2K_IMAGES).exists():
    ds = SyntheticDataset(DIV2K_IMAGES, max_images=800); datasets.append(ds)
    print(f"Synthetic:{len(ds):>6,} images (DIV2K)")

full_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
total   = sum(len(d) for d in datasets)
print(f"\nTotal: {total:,} training pairs")

BATCH = 16
loader = DataLoader(full_ds, batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
print(f"Batches/epoch: {len(loader):,}  |  Batch size: {BATCH}")


# ── 5. Model — multi-scale + image-statistics branch ─────────────────────────

PARAM_NAMES  = ["exposure","brilliance","highlights","shadows","contrast",
                "brightness","black_point","saturation","vibrance","warmth",
                "tint","sharpness","definition","noise","vignette"]
NUM_PARAMS   = 15
PARAM_RANGES = [(-100,100)]*11 + [(0,100)]*4
SYM_IDX      = [i for i,(lo,_) in enumerate(PARAM_RANGES) if lo < 0]
POS_IDX      = [i for i,(lo,_) in enumerate(PARAM_RANGES) if lo == 0]

import timm

class ImageStatsEncoder(nn.Module):
    NUM_STATS = 8
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x):
        rgb  = x * self.std + self.mean
        lum  = 0.299*rgb[:,0] + 0.587*rgb[:,1] + 0.114*rgb[:,2]
        lm   = lum.mean(dim=[1,2])
        lvar = ((lum - lm[:,None,None])**2).mean(dim=[1,2])
        ls   = (lvar + 1e-8).sqrt()
        dr   = torch.sigmoid((0.22 - lum)*20).mean(dim=[1,2])
        br   = torch.sigmoid((lum - 0.78)*20).mean(dim=[1,2])
        rm, gm, bm = rgb[:,0].mean(dim=[1,2]), rgb[:,1].mean(dim=[1,2]), rgb[:,2].mean(dim=[1,2])
        warm = torch.sigmoid((rgb[:,0] - rgb[:,2] - 0.10)*20).mean(dim=[1,2])
        return torch.stack([lm, ls, dr, br, rm, gm, bm, warm], dim=1)


class PhotoEnhancerNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained,
            num_classes=0, global_pool="")
        feat_ch = self.backbone.num_features  # 1280

        self.global_pool   = nn.AdaptiveAvgPool2d(1)
        self.regional_pool = nn.AdaptiveAvgPool2d(2)
        self.region_proj   = nn.Sequential(
            nn.Conv2d(feat_ch, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
        self.stats_enc = ImageStatsEncoder()

        combined = feat_ch + 256*4 + ImageStatsEncoder.NUM_STATS   # 2312
        self.head_in  = nn.Sequential(nn.Linear(combined, 512), nn.SiLU(), nn.LayerNorm(512))
        self.head_res = nn.Sequential(nn.Dropout(0.20), nn.Linear(512, 512),
                                      nn.SiLU(), nn.LayerNorm(512))
        self.head_out = nn.Sequential(nn.Dropout(0.15), nn.Linear(512, 256),
                                      nn.SiLU(), nn.Linear(256, NUM_PARAMS))

    def forward(self, x):
        fm   = self.backbone(x)                             # (B,1280,7,7)
        g    = self.global_pool(fm).flatten(1)              # (B,1280)
        r    = self.regional_pool(self.region_proj(fm)).flatten(1)  # (B,1024)
        st   = self.stats_enc(x)                            # (B,8)
        h    = self.head_in(torch.cat([g, r, st], 1))
        h    = h + self.head_res(h)
        raw  = self.head_out(h)
        out  = torch.empty_like(raw)
        out[:, SYM_IDX] = torch.tanh(raw[:, SYM_IDX]) * 100.0
        out[:, POS_IDX] = torch.sigmoid(raw[:, POS_IDX]) * 100.0
        return out


# ── 6. Loss function ──────────────────────────────────────────────────────────
import torchvision.models as tv_models

def _gaussian_kernel(sz=11, sigma=1.5, ch=3):
    x = torch.arange(sz, dtype=torch.float32) - sz//2
    k1 = torch.exp(-0.5*(x/sigma)**2); k1 /= k1.sum()
    k2 = (k1.unsqueeze(1) @ k1.unsqueeze(0)).expand(ch, 1, sz, sz)
    return k2.contiguous()

def ssim_loss(img1, img2):
    C, K = img1.shape[1], 11
    pad  = K//2
    kern = _gaussian_kernel(K, 1.5, C).to(img1.device)
    i1p  = F.pad(img1, [pad]*4, mode="reflect")
    i2p  = F.pad(img2, [pad]*4, mode="reflect")
    mu1  = F.conv2d(i1p, kern, groups=C)
    mu2  = F.conv2d(i2p, kern, groups=C)
    s1   = F.conv2d(i1p**2, kern, groups=C) - mu1**2
    s2   = F.conv2d(i2p**2, kern, groups=C) - mu2**2
    s12  = F.conv2d(i1p*i2p, kern, groups=C) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    m = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return 1.0 - m.mean()

def color_moment_loss(pred, tgt):
    return (F.l1_loss(pred.mean(dim=[2,3]), tgt.mean(dim=[2,3])) +
            F.l1_loss(pred.std(dim=[2,3]),  tgt.std(dim=[2,3])))

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        f = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.s0 = f[:4];  self.s1 = f[:9]
        self.s2 = f[:18]; self.s3 = f[:27]
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("m", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("s", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, p, t):
        p, t = (p-self.m)/self.s, (t-self.m)/self.s
        return (F.l1_loss(self.s0(p), self.s0(t)) * 0.50 +
                F.l1_loss(self.s1(p), self.s1(t)) * 1.00 +
                F.l1_loss(self.s2(p), self.s2(t)) * 0.50 +
                F.l1_loss(self.s3(p), self.s3(t)) * 0.25)

# Parameter-aware regularisation weights
_REG_W = torch.tensor([0.30,0.50,0.50,0.30,0.50,0.50,0.60,
                        0.50,0.40,0.50,0.90,0.60,0.60,0.40,0.70])


# ── 7. Differentiable enhancement pipeline (mirrors C# ImageProcessor) ────────

MEAN_T = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD_T  = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)

def apply_params(img, params):
    """img: (B,3,H,W) in [0,1] | params: (B,15) → (B,3,H,W) in [0,1]"""
    B,_,H,W = img.shape
    (ex,br,hi,sh,co,bt,bp,sat,vib,wm,tn,
     _shp,_def,_nrs,vig) = params.unbind(1)
    x = img * 255.0
    x = x * torch.pow(2.0, ex/100.0).view(B,1,1,1)            # exposure
    x = x + (bt*1.4).view(B,1,1,1)                             # brightness
    x = (x-128.0)*(1.0+co/100.0).view(B,1,1,1) + 128.0        # contrast
    lum = x[:,0:1]*.299 + x[:,1:2]*.587 + x[:,2:3]*.114
    hi_ = torch.clamp((lum-128.0)/127.0, 0.0, 1.0)
    sh_ = torch.clamp((128.0-lum)/128.0, 0.0, 1.0)
    x = x + (hi.view(B,1,1,1)*-0.9)*hi_ + (sh.view(B,1,1,1)*1.1)*sh_  # hi/sh
    lum = x[:,0:1]*.299 + x[:,1:2]*.587 + x[:,2:3]*.114
    x = x + (128.0-lum)*(br.view(B,1,1,1)/100.0)*0.36          # brilliance
    bp_ = F.softplus(bp.view(B,1,1,1)); bl = bp_*1.25           # black point
    x = (x-bl)*(255.0/torch.clamp(255.0-bl, min=1.0))
    lum = x[:,0:1]*.299 + x[:,1:2]*.587 + x[:,2:3]*.114
    x = lum + (x-lum)*(1.0+sat.view(B,1,1,1)/100.0)            # saturation
    maxc = x.max(1,keepdim=True)[0]; avg = x.mean(1,keepdim=True)
    x = avg+(x-avg)*(1.0+vib.view(B,1,1,1)/100.0*(1.0-torch.abs(maxc-avg)/128.0)) # vibrance
    w_ = (wm*0.55).view(B,1,1,1); t_ = (tn*0.28).view(B,1,1,1)
    x = torch.cat([x[:,0:1]+w_+t_, x[:,1:2]-t_*(0.18/0.28), x[:,2:3]-w_+t_], 1)  # warmth/tint
    vy = torch.linspace(-1,1,H,device=img.device); vx = torch.linspace(-1,1,W,device=img.device)
    yy,xx = torch.meshgrid(vy,vx,indexing="ij")
    dist = (xx**2+yy**2).sqrt().unsqueeze(0).unsqueeze(0)
    x = x*(1.0-torch.clamp(dist-0.35,min=0.0)*(vig.view(B,1,1,1)/85.0))  # vignette
    return torch.clamp(x/255.0, 0.0, 1.0)


# ── 8. EMA ────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay*self.shadow[k] + (1-self.decay)*v

    def state_dict(self): return self.shadow


# ── 9. Training loop ──────────────────────────────────────────────────────────

EPOCHS       = 60
LR           = 3e-4
WARMUP_EP    = 3
CKPT_DIR     = Path("/kaggle/working/checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

model = PhotoEnhancerNet(pretrained=True).to(DEVICE)
vgg   = VGGLoss().to(DEVICE)
ema   = EMA(model)
reg_w = _REG_W.to(DEVICE)

optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": LR * 0.1},
    {"params": list(model.head_in.parameters()) +
               list(model.head_res.parameters()) +
               list(model.head_out.parameters()) +
               list(model.region_proj.parameters()) +
               list(model.stats_enc.parameters()), "lr": LR},
], weight_decay=1e-4)

total_steps  = EPOCHS * len(loader)
warmup_steps = WARMUP_EP * len(loader)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

best_loss  = math.inf
step_count = 0
t_start    = time.time()

print(f"\nStarting training — {EPOCHS} epochs, {len(loader)} batches/epoch")
print(f"Estimated time on T4: {EPOCHS*len(loader)*BATCH/300/3600:.1f} hours")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for inp, tgt in loader:
        inp = inp.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        # Normalize to 224×224 for backbone
        model_inp = F.interpolate(inp, (224,224), mode="bilinear", align_corners=False)
        model_inp = (model_inp - MEAN_T) / STD_T

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            params = model(model_inp)
            pred   = apply_params(inp, params)

            l1    = F.l1_loss(pred, tgt)
            perc  = vgg(pred, tgt)
            ssim  = ssim_loss(pred, tgt)
            color = color_moment_loss(pred, tgt)

            loss = l1 + 0.10*perc + 0.50*ssim + 0.20*color

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        epoch_loss += loss.item()
        step_count += 1

    avg  = epoch_loss / len(loader)
    elapsed = time.time() - t0
    remaining = (EPOCHS - epoch - 1) * elapsed / 3600
    print(f"Ep {epoch+1:02d}/{EPOCHS}  loss={avg:.4f}  "
          f"({elapsed:.0f}s/ep, ~{remaining:.1f}h left)")

    # Save checkpoints
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "best_loss": best_loss},
               CKPT_DIR / "last.pt")

    if avg < best_loss:
        best_loss = avg
        torch.save(ema.state_dict(), CKPT_DIR / "best.pt")
        print(f"  ↳ New best (EMA weights): {best_loss:.4f}")

total_time = (time.time() - t_start) / 3600
print(f"\nTraining complete in {total_time:.2f}h  |  Best loss: {best_loss:.4f}")


# ── 10. Export ONNX ───────────────────────────────────────────────────────────

ONNX_PATH = "/kaggle/working/enhancer_params.onnx"

model.eval().cpu()
state = torch.load(CKPT_DIR / "best.pt", map_location="cpu")
model.load_state_dict(state)

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        opset_version    = 17,
        input_names      = ["input"],
        output_names     = ["params"],
        dynamic_axes     = {"input": {0:"batch"}, "params": {0:"batch"}},
        do_constant_folding = True,
    )

import onnxruntime as ort
sess    = ort.InferenceSession(ONNX_PATH)
out_arr = sess.run(None, {"input": np.random.randn(1,3,224,224).astype(np.float32)})[0]
params_dict = {n: round(float(v), 1) for n, v in zip(PARAM_NAMES, out_arr[0])}
print("\nONNX export verified ✓")
print("Sample output:", params_dict)
print(f"\nFile saved to: {ONNX_PATH}")
print("Download it from the Kaggle output tab (right sidebar) and place it")
print("next to LumaPhoto.exe — the app will pick it up automatically on next launch.")
