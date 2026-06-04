# ── Paste this entire script into ONE Colab cell and run it ───────────────────
# Runtime → Change runtime type → T4 GPU first

# 1. Install deps
import subprocess
subprocess.run(["pip", "install", "-q", "timm", "onnx", "onnxruntime", "tqdm"], check=True)

# 2. Check GPU
import subprocess
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(r.stdout if r.returncode == 0 else "NO GPU — change runtime to T4 first!")

# 3. Mount Drive
from google.colab import drive
drive.mount("/content/drive")
import os
os.makedirs("/content/drive/MyDrive/LumaPhoto", exist_ok=True)

# 4. Download photos — Open Images V7 (5 000, Google-hosted, no login)
#    + Lorem Picsum (1 084 bonus photos, always works)
import json, urllib.request
from pathlib import Path

PHOTO_DIR = "/content/photos"
Path(PHOTO_DIR).mkdir(exist_ok=True)

# ── Source 1: Open Images V7 via fiftyone ─────────────────────────────────────
#    Google's Open Images has 9M diverse real photos. fiftyone downloads
#    exactly the subset you ask for — no account, no API key needed.
print("Installing fiftyone...")
subprocess.run(["pip", "install", "-q", "fiftyone"], check=True)

import fiftyone as fo
import fiftyone.zoo as foz

print("Downloading 5 000 photos from Open Images V7 (Google)...")
oi_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split        = "validation",   # 41K images — fast to index
    max_samples  = 5000,
    seed         = 42,
    shuffle      = True,
    only_matching = False,
)
open_images_files = [s.filepath for s in oi_dataset
                     if Path(s.filepath).exists()]
print(f"Open Images: {len(open_images_files)} photos")

# ── Source 2: Lorem Picsum (extra 1 084 real Unsplash photos, 100% reliable) ──
print("\nDownloading bonus photos from Lorem Picsum...")
all_picsum = []
for page in range(1, 12):
    try:
        with urllib.request.urlopen(
                f"https://picsum.photos/v2/list?page={page}&limit=100",
                timeout=10) as r:
            batch = json.loads(r.read())
        if not batch: break
        all_picsum.extend(batch)
    except:
        break

picsum_files = []
for photo in all_picsum:
    pid   = photo["id"]
    fname = Path(PHOTO_DIR) / f"picsum_{pid}.jpg"
    if not fname.exists():
        try:
            urllib.request.urlretrieve(
                f"https://picsum.photos/id/{pid}/960/720", fname)
        except:
            continue
    picsum_files.append(str(fname))
print(f"Picsum: {len(picsum_files)} photos")

# ── Combine ────────────────────────────────────────────────────────────────────
IMAGE_FILES = list({f for f in open_images_files + picsum_files
                    if Path(f).exists()})

if len(IMAGE_FILES) == 0:
    raise RuntimeError(
        "No photos downloaded — check Colab internet connection.\n"
        "Try: Runtime → Disconnect and delete runtime → reconnect.")

print(f"\nTotal: {len(IMAGE_FILES)} photos ready for training")

# 5. Label every photo using the rule-based analysis (port of the C# code)
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

def analyze(img):
    a = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    step = max(1, int(math.sqrt(w * h / 40000)))
    f = a[::step, ::step].reshape(-1, 3)
    r, g, b = f[:, 0], f[:, 1], f[:, 2]
    lum = r * 0.299 + g * 0.587 + b * 0.114
    n = len(f)
    rm, gm, bm, lm = r.mean(), g.mean(), b.mean(), lum.mean()
    ls = float(lum.std())
    co = float(((r-rm)**2 + (g-gm)**2 + (b-bm)**2).mean()**0.5 / 3)
    dr = float((lum < 55).sum() / n)
    br = float((lum > 200).sum() / n)
    hs = float((lum > 230).sum() / n)
    rg2 = r / np.maximum(g, 1)
    sk_r  = float(((r>100)&(r>g)&(g>b)&(r>b*1.15)&(rg2>=1.05)&(rg2<=1.65)&(lum>60)&(lum<210)).sum()/n)
    sky_r = float(((b>r*1.1)&(b>g*1.05)&(lum>100)).sum()/n)
    gr_r  = float(((g>r*1.1)&(g>b*1.1)&(lum>40)).sum()/n)
    wm_r  = float(((r>180)&(r>g*1.3)&(r>b*1.8)).sum()/n)
    wb = float(rm - bm)
    cs = float(max(rm, gm, bm) - min(rm, gm, bm))
    if   lm < 42 and dr > .55 and .004 < hs < .18:  sc = "Night"
    elif ls > 68 and dr > .12 and br > .12:           sc = "HDR"
    elif sk_r > .12:                                  sc = "Portrait"
    elif wm_r > .12 and 70 < lm < 165:               sc = "Sunset"
    elif sky_r > .15 or gr_r > .20:                  sc = "Landscape"
    elif lm < 62 or dr > .45:                         sc = "LowLight"
    elif lm > 162 and dr < .05:                       sc = "HighKey"
    elif lm > 95 and ls > 28:                         sc = "Daylight"
    else:                                              sc = "General"
    return dict(sc=sc, l=lm, ls=ls, co=co, dr=dr, br=br, wb=wb, cs=cs, hs=hs)

def label(img):
    def c(v, lo, hi): return max(lo, min(hi, v))
    a = analyze(img)
    sc, l, wb, cs, dr, br, hs, co = a["sc"], a["l"], a["wb"], a["cs"], a["dr"], a["br"], a["hs"], a["co"]
    p = np.zeros(15, dtype=np.float32)
    EXP,BRI,HIG,SHA,CON,BRT,BPT,SAT,VIB,WRM,TNT,SHP,DEF,NRS,VIG = range(15)
    if sc == "Night":
        p[EXP]=c(round((80-l)*.55),0,60); p[BRI]=20; p[HIG]=-38 if hs>.04 else -22
        p[SHA]=45 if dr>.70 else 35; p[CON]=22; p[BRT]=5; p[BPT]=8
        p[VIB]=18; p[WRM]=10 if wb<-15 else 0; p[SHP]=8; p[DEF]=10
        p[NRS]=45 if l<30 else 32; p[VIG]=16
    elif sc == "HDR":
        p[EXP]=c(round((114-l)*.30),-30,28); p[BRI]=15
        p[HIG]=c(round(-br*125),-72,-25); p[SHA]=c(round(dr*88),22,58)
        p[CON]=-5; p[BPT]=12; p[SAT]=8 if cs<20 else 4; p[VIB]=20 if cs<20 else 14
        p[WRM]=c(-round(wb*.25),-20,-5) if wb>30 else c(round(-wb*.25),5,20) if wb<-30 else 0
        p[SHP]=10; p[DEF]=20; p[NRS]=15 if l<80 else 5; p[VIG]=10
    elif sc == "Portrait":
        p[EXP]=c(round((122-l)*.42),-55,50); p[BRI]=20 if l<110 else 14 if l<140 else 8
        p[HIG]=-22 if br>.10 else -5; p[SHA]=32 if dr>.20 else 20 if dr>.10 else 10
        p[CON]=12 if co<30 else 7 if co<50 else 2; p[BRT]=5 if l<80 else 0; p[BPT]=8
        p[SAT]=8 if cs<20 else 4 if cs<40 else 0; p[VIB]=18 if cs<20 else 12 if cs<40 else 6
        p[WRM]=-8 if wb>40 else 15 if wb<-20 else 8 if wb<0 else 3
        p[SHP]=8; p[DEF]=12; p[NRS]=15 if l<70 else 0; p[VIG]=10
    elif sc == "Sunset":
        p[EXP]=c(round((100-l)*.42),-50,36); p[BRI]=6
        p[HIG]=c(round(-br*100),-65,-22) if br>.15 else -22; p[SHA]=15
        p[CON]=18 if co<35 else 12; p[BPT]=14; p[SAT]=22; p[VIB]=32
        p[WRM]=0 if wb>20 else 12; p[SHP]=12; p[DEF]=16; p[VIG]=16
    elif sc == "Landscape":
        p[EXP]=c(round((108-l)*.42),-55,50); p[BRI]=14 if l<110 else 10
        p[HIG]=c(round(-br*95),-65,-8) if br>.15 else -18 if br>.06 else -5
        p[SHA]=c(round(dr*72),12,46) if dr>.30 else 15 if dr>.12 else 6
        p[CON]=24 if co<40 else 14 if co<60 else 7; p[BRT]=5 if l<80 else 0; p[BPT]=12
        p[SAT]=18 if cs<20 else 12 if cs<40 else 6; p[VIB]=28 if cs<20 else 20 if cs<40 else 12
        p[WRM]=c(-round(wb*.28),-28,-5) if wb>35 else c(round(-wb*.25),5,22) if wb<-25 else 0
        p[SHP]=14; p[DEF]=20; p[VIG]=12
    elif sc == "LowLight":
        p[EXP]=c(round((95-l)*.42),-20,52); p[BRI]=14
        p[HIG]=-18 if br>.06 else -5; p[SHA]=26 if dr>.50 else 18 if dr>.35 else 12
        p[CON]=22 if co<35 else 14 if co<55 else 6; p[BRT]=5 if l<80 else 0; p[BPT]=8
        p[SAT]=8 if cs<15 else 4; p[VIB]=20 if cs<15 else 14
        p[WRM]=c(-round(wb*.3),-25,-5) if wb>30 else c(round(-wb*.3),5,25) if wb<-30 else 0
        p[SHP]=10; p[DEF]=14; p[NRS]=25 if dr>.50 else 20; p[VIG]=8
    elif sc == "HighKey":
        p[EXP]=c(round((140-l)*.35),-30,18); p[BRI]=6; p[HIG]=c(round(-br*85),-52,-10)
        p[SHA]=4; p[CON]=14 if co<25 else 6; p[BPT]=5
        p[SAT]=8 if cs<15 else 4; p[VIB]=14 if cs<15 else 8
        p[WRM]=-8 if wb>30 else 8 if wb<-30 else 0; p[SHP]=12; p[DEF]=14; p[VIG]=6
    elif sc == "Daylight":
        p[EXP]=c(round((112-l)*.30),-25,22); p[BRI]=12 if l<115 else 8
        p[HIG]=-20 if br>.12 else -12 if br>.06 else -4
        p[SHA]=12 if dr>.15 else 5; p[CON]=16 if co<40 else 10 if co<58 else 2; p[BPT]=10
        p[SAT]=10 if cs<20 else 5 if cs<40 else 0; p[VIB]=18 if cs<20 else 12 if cs<40 else 8
        p[WRM]=c(-round(wb*.25),-22,-5) if wb>30 else c(round(-wb*.25),5,22) if wb<-30 else 0
        p[SHP]=12; p[DEF]=16; p[VIG]=10
    else:
        p[EXP]=c(round((112-l)*.42),-55,50); p[BRI]=16 if l<100 else 10 if l<130 else 6
        p[HIG]=c(round(-br*95),-65,-8) if br>.15 else -16 if br>.06 else -4
        p[SHA]=c(round(dr*72),12,46) if dr>.30 else 15 if dr>.12 else 6
        p[CON]=c(round(20+(35-co)*.4),10,32) if co<35 else 10 if co<55 else 2
        p[BRT]=5 if l<80 else 0; p[BPT]=10
        p[SAT]=12 if cs<15 else 6 if cs<35 else 2; p[VIB]=22 if cs<15 else 16 if cs<35 else 10
        p[WRM]=c(-round(wb*.3),-28,-5) if wb>30 else c(round(-wb*.3),5,28) if wb<-30 else 0
        p[SHP]=12; p[DEF]=18; p[NRS]=12 if l<70 else 0; p[VIG]=10
    for i in range(11):    p[i] = c(p[i], -100, 100)
    for i in range(11,15): p[i] = c(p[i], 0, 100)
    return p

PARAM_NAMES = ["exposure","brilliance","highlights","shadows","contrast",
               "brightness","black_point","saturation","vibrance","warmth",
               "tint","sharpness","definition","noise","vignette"]
NUM_PARAMS = 15

labels = np.zeros((len(IMAGE_FILES), NUM_PARAMS), dtype=np.float32)
for i, path in enumerate(tqdm(IMAGE_FILES, desc="Labelling")):
    try:
        labels[i] = label(Image.open(path))
    except:
        pass
print("Labelling done")

# 6. Build model
import torch, torch.nn as nn
import torch.nn.functional as F
import timm

class PhotoEnhancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True,
                                          num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(.35), nn.Linear(d, 512), nn.SiLU(),
            nn.LayerNorm(512), nn.Dropout(.2),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, NUM_PARAMS))
        self._s = list(range(11))   # symmetric [-100, 100]
        self._p = list(range(11,15)) # positive  [0, 100]
    def forward(self, x):
        raw = self.head(self.backbone(x))
        out = torch.empty_like(raw)
        out[:, self._s] = torch.tanh(raw[:, self._s]) * 100.0
        out[:, self._p] = torch.sigmoid(raw[:, self._p]) * 100.0
        return out

# 7. Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class PhotoDataset(Dataset):
    def __init__(self, files, labels):
        self.files  = files
        self.labels = labels
        self.aug = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.03),
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert("RGB")
            w, h = img.size
            if min(w,h) < 288:
                s = 288/min(w,h)
                img = img.resize((int(w*s), int(h*s)), Image.BICUBIC)
            t = self.aug(transforms.functional.to_tensor(img))
        except:
            t = torch.zeros(3, 256, 256)
        return t, torch.from_numpy(self.labels[i])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

dataset = PhotoDataset(IMAGE_FILES, labels)
loader  = DataLoader(dataset, batch_size=32, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)

model = PhotoEnhancerNet().to(DEVICE)
MEAN  = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD   = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)

optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 3e-5},
    {"params": model.head.parameters(),     "lr": 3e-4},
], weight_decay=1e-4)

EPOCHS = 60
sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
scaler = torch.cuda.amp.GradScaler()
W      = torch.ones(NUM_PARAMS, device=DEVICE)
W[[0,2,4,7]] = 2.0  # exposure, highlights, contrast, saturation weighted higher

# 8. Train
import math, time

os.makedirs("/content/ckpt", exist_ok=True)
best_loss = math.inf

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    t0    = time.time()
    for img_t, lbl in loader:
        img_t = img_t.to(DEVICE, non_blocking=True)
        lbl   = lbl.to(DEVICE, non_blocking=True)
        inp   = (F.interpolate(img_t,(224,224),mode="bilinear",align_corners=False) - MEAN) / STD
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = ((model(inp) - lbl).pow(2) * W).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    sched.step()
    avg = total / len(loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss={avg:.4f}  ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), "/content/ckpt/last.pt")
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "/content/ckpt/best.pt")
        print(f"  ↳ best saved")

print(f"\nDone. Best loss: {best_loss:.4f}")

# 9. Export ONNX
import torch.onnx, onnxruntime as ort, shutil

model.eval()
model.load_state_dict(torch.load("/content/ckpt/best.pt", map_location="cpu"))
model = model.cpu()

with torch.no_grad():
    torch.onnx.export(
        model, torch.randn(1,3,224,224), "/content/enhancer_params.onnx",
        opset_version=17, input_names=["input"], output_names=["params"],
        dynamic_axes={"input":{0:"batch"}, "params":{0:"batch"}},
        do_constant_folding=True)

# Verify
out = ort.InferenceSession("/content/enhancer_params.onnx").run(
    None, {"input": np.random.randn(1,3,224,224).astype(np.float32)})[0]
print("Output:", {n: round(float(v),1) for n,v in zip(PARAM_NAMES, out[0])})

# Save to Drive + download
shutil.copy("/content/enhancer_params.onnx",
            "/content/drive/MyDrive/LumaPhoto/enhancer_params.onnx")
print("Saved to Google Drive.")

from google.colab import files
files.download("/content/enhancer_params.onnx")
print("\nDone! Place enhancer_params.onnx next to LumaPhoto.exe")
