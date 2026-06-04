# =============================================================================
# LumaPhoto — Google Colab Training Notebook
# Train PhotoEnhancerNet on PPR10K + DIV2K synthetic → export enhancer_params.onnx
#
# HOW TO USE
# ──────────
# 1. Go to https://colab.research.google.com → New Notebook
# 2. Runtime → Change runtime type → GPU (T4)
# 3. Paste this entire file into a single code cell and run it.
#
# BEFORE RUNNING — set the PPR10K folder URLs below:
#   Open https://drive.google.com/drive/folders/1kB2OSAGy8uc0xUXaMKoPB0HMSc-rkrLW
#   Navigate into the 360p folder → right-click "source" → Get link → paste as PPR10K_SOURCE_URL
#   Do the same for "target_a"                              → paste as PPR10K_TARGET_URL
#
# OUTPUT
#   enhancer_params.onnx is saved to your Google Drive at:
#   My Drive/LumaPhoto/enhancer_params.onnx
#   Download it and place it next to LumaPhoto.exe
#
# EXPECTED TIME (Colab T4)
#   PPR10K 360p (~8K pairs)   ~2–3 hours
#   + DIV2K synthetic         +30 min
# =============================================================================

# ── Paste your PPR10K subfolder URLs here ────────────────────────────────────
PPR10K_SOURCE_URL = ""   # e.g. "https://drive.google.com/drive/folders/XXXXX"
PPR10K_TARGET_URL = ""   # e.g. "https://drive.google.com/drive/folders/YYYYY"
# If you only have the top-level folder URL, paste it here and leave above blank:
PPR10K_ROOT_URL   = "https://drive.google.com/drive/folders/1kB2OSAGy8uc0xUXaMKoPB0HMSc-rkrLW"
# =============================================================================

import os, sys, subprocess, math, time, copy, random
from pathlib import Path

# ── 1. Mount Google Drive (saves checkpoints so they survive session resets) ──
from google.colab import drive
drive.mount("/content/drive")
DRIVE_OUT = Path("/content/drive/MyDrive/LumaPhoto")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print(f"Checkpoints will be saved to: {DRIVE_OUT}")

# ── 2. Install dependencies ───────────────────────────────────────────────────
print("Installing dependencies…")
subprocess.run(["pip", "install", "-q", "timm", "onnx", "onnxruntime", "onnxscript", "gdown"], check=True)

import gdown
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
    print("⚠  No GPU found. Runtime → Change runtime type → GPU")

# ── 3. Download PPR10K from Google Drive ─────────────────────────────────────
PPR10K_ROOT = Path("/content/ppr10k")
PPR10K_ROOT.mkdir(exist_ok=True)

def _is_file_url(url):
    return "/file/d/" in url

def _download_and_extract(url, out_dir, label):
    """Download a file or folder URL from Google Drive and extract if needed."""
    import zipfile, tarfile
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _is_file_url(url):
        # Single file (likely a zip) — download then extract
        tmp = out_dir.parent / f"_tmp_{label}.zip"
        print(f"Downloading {label} (zip)…")
        gdown.download(url, str(tmp), quiet=False, fuzzy=True)
        print(f"Extracting {label}…")
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as z: z.extractall(out_dir)
        elif tarfile.is_tarfile(tmp):
            with tarfile.open(tmp) as t: t.extractall(out_dir)
        else:
            raise RuntimeError(f"Downloaded file is not a zip/tar: {tmp}")
        tmp.unlink()
        # If extraction created a single subfolder, use that as the dir
        children = [p for p in out_dir.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return out_dir
    else:
        # Folder URL
        print(f"Downloading {label} (folder)…")
        gdown.download_folder(url, output=str(out_dir), quiet=False)
        return out_dir

if PPR10K_SOURCE_URL and PPR10K_TARGET_URL:
    src_dir = _download_and_extract(PPR10K_SOURCE_URL, PPR10K_ROOT / "source",  "source")
    tgt_dir = _download_and_extract(PPR10K_TARGET_URL, PPR10K_ROOT / "target_a", "target_a")
    # Ensure standard layout: ppr10k/source/ and ppr10k/target_a/
    import shutil
    if src_dir != PPR10K_ROOT / "source":
        if (PPR10K_ROOT / "source").exists(): shutil.rmtree(PPR10K_ROOT / "source")
        src_dir.rename(PPR10K_ROOT / "source")
    if tgt_dir != PPR10K_ROOT / "target_a":
        if (PPR10K_ROOT / "target_a").exists(): shutil.rmtree(PPR10K_ROOT / "target_a")
        tgt_dir.rename(PPR10K_ROOT / "target_a")
else:
    # Download top-level folder and auto-detect source/target_a inside
    print("Downloading PPR10K (this may take a while)…")
    gdown.download_folder(PPR10K_ROOT_URL, output=str(PPR10K_ROOT), quiet=False)
    for dp in sorted(PPR10K_ROOT.rglob("*")):
        if dp.is_dir():
            contents = {p.name for p in dp.iterdir() if p.is_dir()}
            if "source" in contents and "target_a" in contents:
                PPR10K_ROOT = dp
                print(f"PPR10K found at: {PPR10K_ROOT}")
                break

print(f"PPR10K source: {PPR10K_ROOT / 'source'} — exists: {(PPR10K_ROOT / 'source').exists()}")
print(f"PPR10K target_a: {PPR10K_ROOT / 'target_a'} — exists: {(PPR10K_ROOT / 'target_a').exists()}")

# ── 4. Download DIV2K for synthetic augmentation ──────────────────────────────
DIV2K_DIR = Path("/content/div2k")
if not DIV2K_DIR.exists():
    print("Downloading DIV2K (800 high-res images for synthetic augmentation)…")
    DIV2K_DIR.mkdir(parents=True, exist_ok=True)
    url  = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    arch = DIV2K_DIR / "DIV2K_train_HR.zip"
    subprocess.run(["wget", "-q", "-O", str(arch), url], check=True)
    subprocess.run(["unzip", "-q", str(arch), "-d", str(DIV2K_DIR)], check=True)
    arch.unlink()
    print("DIV2K ready.")

DIV2K_IMAGES = str(DIV2K_DIR / "DIV2K_train_HR")
if not Path(DIV2K_IMAGES).exists():
    DIV2K_IMAGES = str(DIV2K_DIR)


# ── 5. Dataset loaders ────────────────────────────────────────────────────────

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

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


class PPR10KDataset(Dataset):
    def __init__(self, root, experts=("a",), crop=256):
        src   = Path(root) / "source"
        tdirs = [Path(root) / f"target_{e}" for e in experts]
        names = sorted(f.name for f in src.iterdir() if f.suffix.lower() in EXTS)
        self.pairs = [(src/n, td/n) for n in names for td in tdirs if (td/n).exists()]
        self.crop  = crop
        print(f"  PPR10K: {len(self.pairs):,} pairs (expert A)")

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


# Build combined dataset
datasets = []
if (PPR10K_ROOT / "source").exists() and (PPR10K_ROOT / "target_a").exists():
    ds = PPR10KDataset(PPR10K_ROOT); datasets.append(ds)
else:
    print("⚠  PPR10K not found — check your URLs above")

if Path(DIV2K_IMAGES).exists():
    ds = SyntheticDataset(DIV2K_IMAGES, max_images=800); datasets.append(ds)
    print(f"  Synthetic: {len(ds):,} images (DIV2K)")

if not datasets:
    raise RuntimeError("No datasets found. Check PPR10K URLs.")

full_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
total   = sum(len(d) for d in datasets)
print(f"\nTotal: {total:,} training pairs")

BATCH  = 16
loader = DataLoader(full_ds, batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
print(f"Batches/epoch: {len(loader):,}  |  Batch size: {BATCH}")


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


# ── 8. Differentiable pipeline ────────────────────────────────────────────────

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


# ── 10. Training ──────────────────────────────────────────────────────────────

EPOCHS    = 60
LR        = 3e-4
WARMUP_EP = 3
CKPT_DIR  = Path("/content/checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

model = PhotoEnhancerNet(pretrained=True).to(DEVICE)
vgg   = VGGLoss().to(DEVICE)
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

best_loss  = math.inf
start_epoch = 0

# ── Resume from checkpoint if available ──────────────────────────────────────
resume_path = DRIVE_OUT / "last.pt"
if resume_path.exists():
    print(f"Resuming from {resume_path} …")
    ckpt = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    ema.shadow = ckpt["ema"]
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    best_loss   = ckpt.get("best_loss", math.inf)
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed at epoch {start_epoch+1}/{EPOCHS}  |  best so far: {best_loss:.4f}")
else:
    print(f"\nStarting training — {EPOCHS} epochs, {len(loader)} batches/epoch")
    print(f"Estimated time on T4: {EPOCHS*len(loader)*BATCH/300/3600:.1f} hours")

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
    print(f"Ep {epoch+1:02d}/{EPOCHS}  loss={avg:.4f}  ({elapsed:.0f}s/ep, ~{remaining:.1f}h left)")

    # Save to local + Google Drive so a session reset doesn't lose progress
    ckpt = {"epoch":epoch,"model":model.state_dict(),"ema":ema.state_dict(),
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"best_loss":best_loss}
    torch.save(ckpt, CKPT_DIR/"last.pt")
    torch.save(ckpt, DRIVE_OUT/"last.pt")   # Drive backup

    if avg < best_loss:
        best_loss = avg
        torch.save(ema.state_dict(), CKPT_DIR/"best.pt")
        torch.save(ema.state_dict(), DRIVE_OUT/"best.pt")   # Drive backup
        print(f"  ↳ New best (EMA weights): {best_loss:.4f}")

print(f"\nTraining complete in {(time.time()-t_start)/3600:.2f}h  |  Best loss: {best_loss:.4f}")


# ── 11. Export ONNX ───────────────────────────────────────────────────────────

ONNX_PATH = str(DRIVE_OUT / "enhancer_params.onnx")

model.eval().cpu()
model.load_state_dict(torch.load(CKPT_DIR/"best.pt", map_location="cpu"))

dummy = torch.randn(1,3,224,224)
with torch.no_grad():
    torch.onnx.export(model, dummy, ONNX_PATH,
                      opset_version=17, input_names=["input"], output_names=["params"],
                      dynamic_axes={"input":{0:"batch"},"params":{0:"batch"}},
                      do_constant_folding=True)

import onnxruntime as ort
sess    = ort.InferenceSession(ONNX_PATH)
out_arr = sess.run(None, {"input": np.random.randn(1,3,224,224).astype(np.float32)})[0]
print("\nONNX export verified ✓")
print("Sample:", {n:round(float(v),1) for n,v in zip(PARAM_NAMES, out_arr[0])})
print(f"\nSaved to Google Drive: {ONNX_PATH}")
print("Download it from Google Drive and place it next to LumaPhoto.exe")
