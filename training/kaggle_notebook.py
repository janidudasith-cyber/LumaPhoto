# =============================================================================
# LumaPhoto — Kaggle Training Notebook (multi-expert queue)
# Trains one PhotoEnhancerNet per FiveK expert, sequentially, in a single run.
# → exports fivek_expert_<x>.onnx + .json manifest for each expert in EXPERTS
#
# HOW TO USE ON KAGGLE
# ─────────────────────
# 1. New Notebook → Settings: Accelerator = GPU (T4), Internet = ON
# 2. "+ Add Input" → attach the FiveK dataset:  weipengzhang/adobe-fivek
#    (contains raw/ + a/ b/ c/ d/ e/ expert folders)
# 3. Paste this entire file into one code cell and run it.
#    It trains every expert in EXPERTS, one after the other, and exports each
#    ONNX as soon as that expert finishes — so even if the session dies midway,
#    every completed expert is already sitting in /kaggle/working.
#
# SKIP / RESUME BEHAVIOUR
#   • If fivek_expert_<x>.onnx already exists in /kaggle/working OR in any
#     attached input dataset → that expert is SKIPPED.
#   • If last_expert_<x>.pt exists (working dir or attached previous output)
#     → that expert RESUMES mid-training.
#   • To resume after a dead session: save this notebook's output as a
#     dataset (or use "+ Add Input" → your previous notebook's output),
#     re-attach it, and re-run. It picks up where it left off.
#
# OUTPUT (per expert, in /kaggle/working → notebook Output tab)
#   fivek_expert_c.onnx + .json  ← natural  (Auto Enhance slider centre)
#   fivek_expert_a.onnx + .json  ← vibrant  (slider right)
#   fivek_expert_e.onnx + .json  ← dramatic (slider left)
#   Download each .onnx + its .json manifest and place next to LumaPhoto.exe.
#   The app REJECTS any .onnx without its matching .json manifest.
#
# EXPECTED TIME (Kaggle T4, FiveK 5000 pairs + DIV2K, 60 ep)
#   ~3–4 h per expert → two experts ≈ 6–8 h (fits one 12 h session)
# =============================================================================

# ── Expert queue ──────────────────────────────────────────────────────────────
# All three power the Auto Enhance slider:
#   slider right (+100) → Expert A  (vibrant / punchy)
#   slider middle (  0) → Expert C  (natural / balanced)
#   slider left  (-100) → Expert E  (moody / dramatic)
# C is listed too — it's skipped automatically if its .onnx is found (e.g.
# attach your previous output, or it just trains again if not found).
EXPERTS       = ["a", "e"]   # ← C already trained; add "c" to retrain it
FORCE_RETRAIN = []           # e.g. ["a"] to retrain even if its .onnx exists
# =============================================================================

import os, sys, subprocess, math, time, copy, random, json
from pathlib import Path

WORK = Path("/kaggle/working")
TMP  = Path("/kaggle/tmp"); TMP.mkdir(parents=True, exist_ok=True)

# ── 1. Install dependencies ───────────────────────────────────────────────────
print("Installing dependencies…")
subprocess.run(["pip", "install", "-q", "timm", "onnx", "onnxruntime", "onnxscript"], check=True)

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
    print("⚠  No GPU found. Settings → Accelerator → GPU T4")

# ── 2. Validate expert queue + skip already-trained experts ───────────────────
EXPERTS       = [e.lower().strip() for e in EXPERTS]
FORCE_RETRAIN = [e.lower().strip() for e in FORCE_RETRAIN]
for e in EXPERTS:
    assert e in ("a", "b", "c", "d", "e"), f"EXPERTS entries must be a/b/c/d/e, got: {e}"

def onnx_name(expert): return f"fivek_expert_{expert}.onnx"

def find_existing(filename):
    """Look for a file in /kaggle/working and any attached input dataset."""
    if (WORK / filename).exists():
        return WORK / filename
    for p in Path("/kaggle/input").rglob(filename):
        return p
    return None

def is_done(expert):
    return find_existing(onnx_name(expert)) is not None and expert not in FORCE_RETRAIN

todo = [e for e in EXPERTS if not is_done(e)]
done = [e for e in EXPERTS if is_done(e)]
if done:
    print(f"Already trained (onnx found, skipping): {', '.join(e.upper() for e in done)}")
if not todo:
    raise SystemExit("All experts already trained — nothing to do. "
                     "Add an expert to FORCE_RETRAIN to retrain it.")
print(f"Training queue: {' → '.join(e.upper() for e in todo)}")

# ── 3. Locate the FiveK dataset under /kaggle/input ───────────────────────────
def _find_fivek_root():
    """Find the dir containing raw/ (or input/) + per-expert letter folders."""
    for d in sorted(Path("/kaggle/input").rglob("*")):
        if not d.is_dir():
            continue
        has_input  = (d / "raw").is_dir() or (d / "input").is_dir()
        has_expert = any((d / x).is_dir() for x in ("a", "b", "c", "d", "e"))
        if has_input and has_expert:
            return d
    return None

FIVEK_ROOT = _find_fivek_root()
if FIVEK_ROOT is None:
    print("=== /kaggle/input contents ===")
    for item in sorted(Path("/kaggle/input").glob("*/*")):
        print(f"  {item}")
    raise RuntimeError("FiveK not found. '+ Add Input' → weipengzhang/adobe-fivek")

FIVEK_INPUT = FIVEK_ROOT / ("raw" if (FIVEK_ROOT / "raw").is_dir() else "input")
print(f"FiveK root: {FIVEK_ROOT}")
print(f"FiveK input ({FIVEK_INPUT.name}/): {len(list(FIVEK_INPUT.iterdir()))} images")
for e in todo:
    tgt = FIVEK_ROOT / e
    if not tgt.is_dir():
        print(f"⚠  Missing target folder for Expert {e.upper()}: {tgt}")
        print(f"   Available: {[p.name for p in FIVEK_ROOT.iterdir() if p.is_dir()]}")
        raise RuntimeError(f"Expert {e.upper()} folder not found in dataset")
    print(f"  Expert {e.upper()}: {len(list(tgt.iterdir()))} target images ✓")

# ── 4. Auto-download DIV2K for synthetic augmentation ─────────────────────────
# Lives in /kaggle/tmp so it isn't persisted into the (size-limited) output.
DIV2K_DIR = TMP / "div2k"
if not DIV2K_DIR.exists():
    print("Downloading DIV2K (800 high-res images for synthetic augmentation)…")
    DIV2K_DIR.mkdir(parents=True, exist_ok=True)
    url  = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    arch = DIV2K_DIR / "DIV2K_train_HR.zip"
    try:
        subprocess.run(["wget", "-q", "-O", str(arch), url], check=True)
        subprocess.run(["unzip", "-q", str(arch), "-d", str(DIV2K_DIR)], check=True)
        arch.unlink()
        print("DIV2K ready.")
    except subprocess.CalledProcessError:
        print("⚠  DIV2K download failed — continuing with FiveK only.")

DIV2K_IMAGES = str(DIV2K_DIR / "DIV2K_train_HR")
if not Path(DIV2K_IMAGES).exists():
    DIV2K_IMAGES = str(DIV2K_DIR)

# ── 4b. One-time downscale pass (CRITICAL for speed) ──────────────────────────
# FiveK ships full-resolution JPEGs (~3–6 MP). Decoding two of them per pair,
# every batch, every epoch makes training CPU-bound: ~30 min/epoch on Kaggle.
# Training only ever sees 256-px crops, so we shrink everything ONCE to
# min-side 320 on disk (~10–15 min total) → epochs drop to ~3 min.
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SMALL    = TMP / "small"
PRE_SIDE = 320

def _shrink_one(args):
    src, dst = args
    try:
        im = Image.open(src)
        im.draft("RGB", (PRE_SIDE, PRE_SIDE))   # fast JPEG DCT-scaled decode
        im = im.convert("RGB")
        w, h = im.size
        s = PRE_SIDE / min(w, h)
        if s < 1.0:
            im = im.resize((max(PRE_SIDE, round(w*s)), max(PRE_SIDE, round(h*s))),
                           Image.BICUBIC)
        im.save(dst, "JPEG", quality=92)
    except Exception as ex_:
        print(f"  skip {src.name}: {ex_}")

def shrink_dir(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, dst_dir / (p.stem + ".jpg"))
            for p in sorted(Path(src_dir).iterdir())
            if p.suffix.lower() in EXTS and not (dst_dir / (p.stem + ".jpg")).exists()]
    if not jobs:
        print(f"  {dst_dir.name}/: ready ({len(list(dst_dir.iterdir()))} images)")
        return
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i, _ in enumerate(pool.map(_shrink_one, jobs, chunksize=32), 1):
            if i % 1000 == 0:
                print(f"  {dst_dir.name}/: {i}/{len(jobs)}", flush=True)
    print(f"  {dst_dir.name}/: {len(jobs)} images shrunk in {time.time()-t0:.0f}s", flush=True)

print("Preparing downscaled training copies (one-time)…", flush=True)
shrink_dir(FIVEK_INPUT, SMALL / "raw")
for e in todo:
    shrink_dir(FIVEK_ROOT / e, SMALL / e)
if Path(DIV2K_IMAGES).exists():
    shrink_dir(DIV2K_IMAGES, SMALL / "div2k")
    DIV2K_IMAGES = str(SMALL / "div2k")
FIVEK_TRAIN_ROOT = SMALL   # loaders read the small copies, not /kaggle/input


# ── 5. Dataset loaders ────────────────────────────────────────────────────────

def _to_tensor(img): return transforms.functional.to_tensor(img)

def _resize_min(img, min_side):
    w, h = img.size
    if min(w, h) < min_side:
        s = min_side / min(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.BICUBIC)
    return img

def _sync_crop_flip(inp, tgt, crop, flip):
    seed = random.randint(0, 2**31)
    aug  = transforms.Compose([transforms.RandomCrop(crop),
                                transforms.RandomHorizontalFlip(p=1.0 if flip else 0.0)])
    torch.manual_seed(seed); random.seed(seed); inp_t = aug(_to_tensor(inp))
    torch.manual_seed(seed); random.seed(seed); tgt_t = aug(_to_tensor(tgt))
    return inp_t, tgt_t


class FiveKDataset(Dataset):
    """MIT-Adobe FiveK — 5 000 diverse-scene pairs (raw → chosen expert retouch).
    Expects: <root>/raw/*.jpg  (or input/) and  <root>/<expert>/*.jpg"""
    def __init__(self, root, expert_dir, crop=256):
        root = Path(root)
        self.inp_dir = root / "raw" if (root / "raw").is_dir() else root / "input"
        self.tgt_dir = root / expert_dir
        names = sorted(f.name for f in self.inp_dir.iterdir()
                       if f.suffix.lower() in EXTS)
        self.pairs = [(self.inp_dir/n, self.tgt_dir/n)
                      for n in names if (self.tgt_dir/n).exists()]
        self.crop = crop
        print(f"  FiveK:  {len(self.pairs):,} pairs (expert {expert_dir[-1].upper()})")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp_p, tgt_p = self.pairs[idx]
        inp  = _resize_min(Image.open(inp_p).convert("RGB"), 288)
        tgt  = Image.open(tgt_p).convert("RGB").resize(inp.size, Image.BICUBIC)
        return _sync_crop_flip(inp, tgt, self.crop, random.random() > 0.5)


class SyntheticDataset(Dataset):
    def __init__(self, image_dir, crop=256, max_images=5000):
        self.files = sorted(p for p in Path(image_dir).rglob("*")
                            if p.suffix.lower() in EXTS)[:max_images]
        self.crop  = crop

    def __len__(self): return len(self.files)

    def _degrade(self, x):
        ev  = random.uniform(-1.5, 1.5);   x = x * (2.0**ev)
        co  = random.uniform(0.60, 1.60);  x = (x-0.5)*co + 0.5
        lum = x[0:1]*0.299 + x[1:2]*0.587 + x[2:3]*0.114
        sat = random.uniform(0.3, 1.8);    x = lum + (x-lum)*sat
        wt  = random.uniform(-0.12, 0.12); x[0], x[2] = x[0]+wt, x[2]-wt
        if random.random() < 0.3: x = x - random.uniform(0.0, 0.12)
        if random.random() < 0.3: x = x * random.uniform(1.1, 1.4)
        return torch.clamp(x, 0.0, 1.0)

    def __getitem__(self, idx):
        img  = _resize_min(Image.open(self.files[idx]).convert("RGB"), 288)
        seed = random.randint(0, 2**31)
        aug  = transforms.Compose([transforms.RandomCrop(self.crop),
                                   transforms.RandomHorizontalFlip()])
        torch.manual_seed(seed); random.seed(seed); clean = aug(_to_tensor(img))
        return self._degrade(clean.clone()), clean


BATCH = 16

def build_loader(expert):
    """Fresh DataLoader targeting the given expert's retouches (small copies)."""
    datasets = [FiveKDataset(FIVEK_TRAIN_ROOT, expert)]
    if Path(DIV2K_IMAGES).exists():
        ds = SyntheticDataset(DIV2K_IMAGES, max_images=800); datasets.append(ds)
        print(f"  Synthetic: {len(ds):,} images (DIV2K augmentation)")
    full_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    total   = sum(len(d) for d in datasets)
    print(f"  Total: {total:,} training pairs  (Expert {expert.upper()} target style)")
    return DataLoader(full_ds, batch_size=BATCH, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True,
                      persistent_workers=True, prefetch_factor=4)


# ── 6. Model ──────────────────────────────────────────────────────────────────

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
        ls   = ((lum - lm[:,None,None])**2).mean(dim=[1,2]).add(1e-8).sqrt()
        dr   = torch.sigmoid((0.22-lum)*20).mean(dim=[1,2])
        br   = torch.sigmoid((lum-0.78)*20).mean(dim=[1,2])
        rm, gm, bm = rgb[:,0].mean(dim=[1,2]), rgb[:,1].mean(dim=[1,2]), rgb[:,2].mean(dim=[1,2])
        warm = torch.sigmoid((rgb[:,0]-rgb[:,2]-0.10)*20).mean(dim=[1,2])
        return torch.stack([lm,ls,dr,br,rm,gm,bm,warm], dim=1)


class PhotoEnhancerNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone    = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                             num_classes=0, global_pool="")
        feat_ch          = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regional_pool = nn.AdaptiveAvgPool2d(2)
        self.region_proj = nn.Sequential(nn.Conv2d(feat_ch,256,1,bias=False),
                                         nn.BatchNorm2d(256), nn.SiLU())
        self.stats_enc   = ImageStatsEncoder()
        combined         = feat_ch + 256*4 + ImageStatsEncoder.NUM_STATS
        self.head_in     = nn.Sequential(nn.Linear(combined,512), nn.SiLU(), nn.LayerNorm(512))
        self.head_res    = nn.Sequential(nn.Dropout(0.20), nn.Linear(512,512),
                                         nn.SiLU(), nn.LayerNorm(512))
        self.head_out    = nn.Sequential(nn.Dropout(0.15), nn.Linear(512,256),
                                         nn.SiLU(), nn.Linear(256,NUM_PARAMS))

    def forward(self, x):
        fm  = self.backbone(x)
        g   = self.global_pool(fm).flatten(1)
        r   = self.regional_pool(self.region_proj(fm)).flatten(1)
        st  = self.stats_enc(x)
        h   = self.head_in(torch.cat([g,r,st],1))
        h   = h + self.head_res(h)
        raw = self.head_out(h)
        out = torch.empty_like(raw)
        out[:,SYM_IDX] = torch.tanh(raw[:,SYM_IDX]) * 100.0
        out[:,POS_IDX] = torch.sigmoid(raw[:,POS_IDX]) * 100.0
        return out


# ── 7. Loss ───────────────────────────────────────────────────────────────────
import torchvision.models as tv_models

def _gaussian_kernel(sz=11, sigma=1.5, ch=3):
    x  = torch.arange(sz, dtype=torch.float32) - sz//2
    k1 = torch.exp(-0.5*(x/sigma)**2); k1 /= k1.sum()
    return (k1.unsqueeze(1) @ k1.unsqueeze(0)).expand(ch,1,sz,sz).contiguous()

def ssim_loss(a, b):
    C, K, pad = a.shape[1], 11, 5
    k = _gaussian_kernel(K,1.5,C).to(a.device)
    ap, bp = F.pad(a,[pad]*4,"reflect"), F.pad(b,[pad]*4,"reflect")
    mu1,mu2 = F.conv2d(ap,k,groups=C), F.conv2d(bp,k,groups=C)
    s1  = F.conv2d(ap**2,k,groups=C)-mu1**2
    s2  = F.conv2d(bp**2,k,groups=C)-mu2**2
    s12 = F.conv2d(ap*bp,k,groups=C)-mu1*mu2
    C1,C2 = 0.01**2, 0.03**2
    return 1.0-((2*mu1*mu2+C1)*(2*s12+C2)/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()

def color_moment_loss(p, t):
    return (F.l1_loss(p.mean(dim=[2,3]),t.mean(dim=[2,3])) +
            F.l1_loss(p.std(dim=[2,3]), t.std(dim=[2,3])))

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        f = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.s0,self.s1,self.s2,self.s3 = f[:4],f[:9],f[:18],f[:27]
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("m", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("s", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, p, t):
        p,t = (p-self.m)/self.s, (t-self.m)/self.s
        return (F.l1_loss(self.s0(p),self.s0(t))*0.50 + F.l1_loss(self.s1(p),self.s1(t))*1.00 +
                F.l1_loss(self.s2(p),self.s2(t))*0.50 + F.l1_loss(self.s3(p),self.s3(t))*0.25)


# ── 8. Differentiable pipeline (mirrors C# ImageProcessor) ────────────────────

MEAN_T = torch.tensor([0.485,0.456,0.406],device=DEVICE).view(1,3,1,1)
STD_T  = torch.tensor([0.229,0.224,0.225],device=DEVICE).view(1,3,1,1)

def apply_params(img, params):
    B,_,H,W = img.shape
    (ex,br,hi,sh,co,bt,bp,sat,vib,wm,tn,_s,_d,_n,vig) = params.unbind(1)
    x = img*255.0
    x = x*torch.pow(2.0,ex/100.0).view(B,1,1,1)
    x = x+(bt*1.4).view(B,1,1,1)
    x = (x-128.0)*(1.0+co/100.0).view(B,1,1,1)+128.0
    lum = x[:,0:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114
    x = x+(hi.view(B,1,1,1)*-0.9)*torch.clamp((lum-128)/127,0,1)+(sh.view(B,1,1,1)*1.1)*torch.clamp((128-lum)/128,0,1)
    lum = x[:,0:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114
    x = x+(128-lum)*(br.view(B,1,1,1)/100)*0.36
    bl = F.softplus(bp.view(B,1,1,1))*1.25
    x = (x-bl)*(255/torch.clamp(255-bl,min=1.0))
    lum = x[:,0:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114
    x = lum+(x-lum)*(1+sat.view(B,1,1,1)/100)
    mc=x.max(1,keepdim=True)[0]; av=x.mean(1,keepdim=True)
    x = av+(x-av)*(1+vib.view(B,1,1,1)/100*(1-torch.abs(mc-av)/128))
    w_=(wm*0.55).view(B,1,1,1); t_=(tn*0.28).view(B,1,1,1)
    x = torch.cat([x[:,0:1]+w_+t_, x[:,1:2]-t_*(0.18/0.28), x[:,2:3]-w_+t_],1)
    vy=torch.linspace(-1,1,H,device=img.device); vx=torch.linspace(-1,1,W,device=img.device)
    yy,xx=torch.meshgrid(vy,vx,indexing="ij")
    x = x*(1-torch.clamp((xx**2+yy**2).sqrt().unsqueeze(0).unsqueeze(0)-0.35,min=0)*(vig.view(B,1,1,1)/85))
    return torch.clamp(x/255.0,0.0,1.0)


# ── 9. EMA ────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay=decay; self.shadow=copy.deepcopy(model.state_dict())
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                self.shadow[k]=self.decay*self.shadow[k]+(1-self.decay)*v
    def state_dict(self): return self.shadow


# ── 10. Training + export, one expert at a time ───────────────────────────────

EPOCHS    = 60
LR        = 3e-4
WARMUP_EP = 3
CKPT_DIR  = WORK / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

import onnxruntime as ort

vgg = VGGLoss().to(DEVICE)   # frozen — shared across all expert runs

def train_expert(expert):
    """Train one expert end-to-end (resumable) and export its ONNX to /kaggle/working."""
    suffix = f"_expert_{expert}"
    loader = build_loader(expert)

    model = PhotoEnhancerNet(pretrained=True).to(DEVICE)
    ema   = EMA(model)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),  "lr": LR*0.1},
        {"params": list(model.head_in.parameters()) +
                   list(model.head_res.parameters()) +
                   list(model.head_out.parameters()) +
                   list(model.region_proj.parameters()) +
                   list(model.stats_enc.parameters()), "lr": LR},
    ], weight_decay=1e-4)

    total_steps  = EPOCHS * len(loader)
    warmup_steps = WARMUP_EP * len(loader)

    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        prog = (step-warmup_steps) / max(1, total_steps-warmup_steps)
        return max(1e-3, 0.5*(1.0+math.cos(math.pi*prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_loss   = math.inf
    start_epoch = 0

    # Resume: check working dir first, then any attached previous-run output
    resume_path = CKPT_DIR / f"last{suffix}.pt"
    if not resume_path.exists():
        resume_path = find_existing(f"last{suffix}.pt") or resume_path
    if resume_path.exists():
        print(f"Resuming Expert {expert.upper()} from {resume_path} …")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        ema.shadow = ckpt["ema"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_loss   = ckpt.get("best_loss", math.inf)
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch+1}/{EPOCHS}  |  best so far: {best_loss:.4f}")
        # Carry the matching best EMA weights forward if available
        prev_best = find_existing(f"best{suffix}.pt")
        if prev_best and not (CKPT_DIR / f"best{suffix}.pt").exists():
            import shutil as _sh
            _sh.copy(prev_best, CKPT_DIR / f"best{suffix}.pt")
    else:
        print(f"\nStarting Expert {expert.upper()} — {EPOCHS} epochs, {len(loader)} batches/epoch")
        print("(per-epoch time is printed after each epoch; expect ~3 min/ep on T4)")

    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for inp, tgt in loader:
            inp = inp.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            model_inp = F.interpolate(inp,(224,224),mode="bilinear",align_corners=False)
            model_inp = (model_inp - MEAN_T) / STD_T

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                params = model(model_inp)
                pred   = apply_params(inp, params)
                loss   = F.l1_loss(pred,tgt) + 0.10*vgg(pred,tgt) + 0.50*ssim_loss(pred,tgt) + 0.20*color_moment_loss(pred,tgt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            epoch_loss += loss.item()

        avg     = epoch_loss / len(loader)
        elapsed = time.time() - t0
        remaining = (EPOCHS-epoch-1)*elapsed/3600
        print(f"[{expert.upper()}] Ep {epoch+1:02d}/{EPOCHS}  loss={avg:.4f}  ({elapsed:.0f}s/ep, ~{remaining:.1f}h left)",
              flush=True)

        # Checkpoint every epoch so a dead session loses at most one epoch
        ckpt = {"epoch":epoch,"model":model.state_dict(),"ema":ema.state_dict(),
                "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"best_loss":best_loss}
        torch.save(ckpt, CKPT_DIR/f"last{suffix}.pt")

        if avg < best_loss:
            best_loss = avg
            torch.save(ema.state_dict(), CKPT_DIR/f"best{suffix}.pt")
            print(f"  ↳ New best (EMA weights): {best_loss:.4f}", flush=True)

    print(f"\nExpert {expert.upper()} trained in {(time.time()-t_start)/3600:.2f}h  |  Best loss: {best_loss:.4f}")

    # ── Export ONNX + manifest for this expert ────────────────────────────────
    onnx_path = str(WORK / onnx_name(expert))
    best_path = CKPT_DIR / f"best{suffix}.pt"
    if not best_path.exists():
        best_path = find_existing(f"best{suffix}.pt")   # resumed with no new best

    model.eval().cpu()
    model.load_state_dict(torch.load(best_path, map_location="cpu"))

    dummy = torch.randn(1,3,224,224)
    with torch.no_grad():
        torch.onnx.export(model, dummy, onnx_path,
                          opset_version=17, input_names=["input"], output_names=["params"],
                          dynamic_axes={"input":{0:"batch"},"params":{0:"batch"}},
                          do_constant_folding=True)

    # PyTorch's dynamo exporter may split weights into a sidecar .onnx.data
    # file. Re-save as ONE self-contained .onnx — the app installer only
    # bundles .onnx + .json, so a .data dependency would break packaging.
    import onnx as _onnx
    data_file = Path(onnx_path + ".data")
    if data_file.exists():
        _m = _onnx.load(onnx_path)          # pulls external data in
        _onnx.save(_m, onnx_path)           # writes single file (<2 GB inlines)
        data_file.unlink()
        print("  merged .onnx.data into single .onnx ✓")

    # Write the manifest the app REQUIRES to trust this model.
    # NeuralEnhancer.IsTrustedParamModel() silently ignores any .onnx whose
    # sidecar .json is missing or whose profile/param_count/expert don't match.
    manifest_path = str(Path(onnx_path).with_suffix(".json"))
    json.dump({
        "training_profile": "fivek-v2",
        "dataset": "MIT-Adobe FiveK",
        "expert": expert,
        "param_count": NUM_PARAMS,
        "input": "float32 [1, 3, 224, 224] ImageNet-normalised RGB",
        "output": "float32 [1, 15] LumaPhoto slider parameters",
    }, open(manifest_path, "w"), indent=2)

    sess    = ort.InferenceSession(onnx_path)
    out_arr = sess.run(None, {"input": np.random.randn(1,3,224,224).astype(np.float32)})[0]
    print(f"ONNX export verified ✓  ({onnx_name(expert)})")
    print("Sample:", {n:round(float(v),1) for n,v in zip(PARAM_NAMES, out_arr[0])})
    print(f"Saved to /kaggle/working: {onnx_name(expert)}  (+ manifest {Path(manifest_path).name})")

    # Free GPU memory before the next expert
    del model, ema, optimizer, scheduler, scaler, loader
    torch.cuda.empty_cache()


# ── 11. Run the queue ─────────────────────────────────────────────────────────

for i, expert in enumerate(todo, 1):
    print(f"\n{'='*70}\n  EXPERT {expert.upper()}  ({i}/{len(todo)})\n{'='*70}")
    train_expert(expert)

print(f"\n{'='*70}\nAll done. Files in /kaggle/working (notebook Output tab):")
for e in EXPERTS:
    found = find_existing(onnx_name(e))
    print(f"  {onnx_name(e)}  {'✓ ' + str(found) if found else '✗ missing'}")
print("""
Slider mapping:
  fivek_expert_c.onnx  ← natural  (slider centre)
  fivek_expert_a.onnx  ← vibrant  (slider right)
  fivek_expert_e.onnx  ← dramatic (slider left)
Download each .onnx + its .json manifest and place ALL of them next to
LumaPhoto.exe — the app rejects any .onnx missing its .json manifest.""")
