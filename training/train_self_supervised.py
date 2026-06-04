"""
Self-supervised training — no external datasets required.

How it works:
  1. Scans any folder of photos you already own (or downloads free ones from
     Wikimedia Commons — no login, no account).
  2. Runs the rule-based analysis (ported from C#) on every image to generate
     the 15-parameter enhancement labels automatically.
  3. Trains the EfficientNet-B0 model to predict those labels directly.
  4. Exports enhancer_params.onnx — drop it next to LumaPhoto.exe.

Quality: on-par with or better than the rule-based system alone, because the
NN can generalise to images the rules misclassify.

Usage:
    # Use your own photos (any folder):
    python train_self_supervised.py --photos "C:/Users/You/Pictures"

    # Download free Wikimedia photos + use your own:
    python train_self_supervised.py --photos "C:/Users/You/Pictures" --wikimedia 5000

    # Wikimedia only:
    python train_self_supervised.py --wikimedia 8000 --out ./data/wiki

    # Full options:
    python train_self_supervised.py --photos ./my_photos --wikimedia 5000 \\
        --epochs 60 --batch 32 --output enhancer_params.onnx
"""

import argparse
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import PhotoEnhancerNet
from rules import label_image, PARAM_NAMES, NUM_PARAMS

# ── ImageNet normalisation ─────────────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


# ── Wikimedia Commons downloader ──────────────────────────────────────────────

WIKIMEDIA_CATEGORIES = [
    "Landscape photographs", "Portrait photographs", "Sunset photographs",
    "Night photographs", "HDR photographs", "Nature photographs",
    "Urban photography", "Street photography", "Travel photography",
    "People photographs", "Architecture photographs", "Flower photographs",
    "Mountain photographs", "Forest photographs", "Beach photographs",
    "Sky photographs", "City photographs", "Food photography",
    "Sports photographs", "Wildlife photographs",
]

def _wikimedia_images_for_category(category: str, limit: int = 50) -> List[str]:
    """Return up to `limit` image URLs from a Wikimedia Commons category."""
    api = (
        "https://commons.wikimedia.org/w/api.php"
        "?action=query&format=json"
        f"&list=categorymembers&cmtitle=Category:{category.replace(' ', '_')}"
        f"&cmtype=file&cmlimit={limit}"
    )
    try:
        with urllib.request.urlopen(api, timeout=10) as r:
            data = json.loads(r.read())
        members = data.get("query", {}).get("categorymembers", [])
        titles  = [m["title"] for m in members if m["title"].lower().endswith(
                   (".jpg", ".jpeg", ".png"))]

        urls = []
        for title in titles[:limit]:
            img_api = (
                "https://commons.wikimedia.org/w/api.php"
                "?action=query&format=json&prop=imageinfo"
                "&iiprop=url&iiurlwidth=800"
                f"&titles={urllib.parse.quote(title)}"
            )
            with urllib.request.urlopen(img_api, timeout=10) as r2:
                d2 = json.loads(r2.read())
            pages = d2.get("query", {}).get("pages", {})
            for page in pages.values():
                info = page.get("imageinfo", [])
                if info:
                    urls.append(info[0].get("thumburl") or info[0].get("url", ""))
        return [u for u in urls if u]
    except Exception:
        return []


import urllib.parse


def download_wikimedia_photos(out_dir: str, target: int = 3000) -> List[str]:
    """Download up to `target` photos from Wikimedia Commons. No login required."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = list(out.glob("*.jpg"))
    if len(existing) >= target:
        print(f"Wikimedia: {len(existing)} photos already in {out}")
        return [str(p) for p in existing]

    print(f"Downloading up to {target} free photos from Wikimedia Commons…")
    downloaded = list(existing)
    per_cat    = max(10, target // len(WIKIMEDIA_CATEGORIES) + 1)

    for cat in WIKIMEDIA_CATEGORIES:
        if len(downloaded) >= target:
            break
        urls = _wikimedia_images_for_category(cat, per_cat)
        for url in urls:
            if len(downloaded) >= target:
                break
            fname = out / f"wiki_{len(downloaded):05d}.jpg"
            if fname.exists():
                downloaded.append(fname)
                continue
            try:
                urllib.request.urlretrieve(url, fname)
                downloaded.append(fname)
            except Exception:
                pass
        print(f"  [{cat}] → {len(downloaded)}/{target}")

    print(f"Wikimedia: {len(downloaded)} photos saved to {out}")
    return [str(p) for p in downloaded]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SelfSupervisedDataset(Dataset):
    """
    Loads images from disk, computes rule-based labels on-the-fly.

    The label (15 float params) is cached after the first computation
    so repeat epochs are fast.  Cache is saved as a .npy file alongside
    the image list so it survives between runs.
    """

    def __init__(
        self,
        image_files: List[str],
        cache_dir: str = "./label_cache",
        crop_size: int = 256,
        augment: bool = True,
    ):
        self.files     = image_files
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache     = {}       # index → np.ndarray(15)
        self.crop_size = crop_size
        self.augment   = augment

        # Spatial augmentation
        aug_list = [transforms.RandomCrop(crop_size)]
        if augment:
            aug_list += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.1),
            ]
        self.spatial_aug = transforms.Compose(aug_list)

        # Colour augmentation (applied to input only — does NOT affect the label
        # because the label was computed on the original unaugmented image)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03
        ) if augment else None

        # Load any previously cached labels
        cache_file = self.cache_dir / "labels.npy"
        meta_file  = self.cache_dir / "files.json"
        if cache_file.exists() and meta_file.exists():
            with open(meta_file) as f:
                cached_files = json.load(f)
            if cached_files == image_files:
                arr = np.load(cache_file)
                self.cache = {i: arr[i] for i in range(len(arr))}
                print(f"Loaded {len(self.cache)} cached labels.")

    def _resize_min(self, img: Image.Image, min_side: int) -> Image.Image:
        w, h = img.size
        if min(w, h) < min_side:
            s = min_side / min(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
        return img

    def _get_label(self, idx: int, img: Image.Image) -> np.ndarray:
        if idx in self.cache:
            return self.cache[idx]
        label = label_image(img)
        self.cache[idx] = label
        return label

    def save_cache(self):
        arr = np.stack([self.cache[i] for i in range(len(self.files))])
        np.save(self.cache_dir / "labels.npy", arr)
        with open(self.cache_dir / "files.json", "w") as f:
            json.dump(self.files, f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (300, 300), (128, 128, 128))

        img = self._resize_min(img, self.crop_size + 32)

        # Compute label BEFORE colour jitter (label is for the clean image)
        label = self._get_label(idx, img)

        # Convert to tensor and crop
        seed = random.randint(0, 2**31)
        torch.manual_seed(seed)
        img_t = self.spatial_aug(transforms.functional.to_tensor(img))

        # Optional colour jitter on input only
        if self.color_jitter is not None:
            img_t = self.color_jitter(img_t)

        return img_t, torch.from_numpy(label)


# ── Training ──────────────────────────────────────────────────────────────────

def make_backbone_input(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    small = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
    return (small - mean) / std


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect image files ────────────────────────────────────────────────────
    all_files: List[str] = []

    if args.photos:
        for root_dir in args.photos:
            found = [str(p) for p in Path(root_dir).rglob("*")
                     if p.suffix.lower() in PHOTO_EXTS]
            print(f"Found {len(found)} photos in {root_dir}")
            all_files.extend(found)

    if args.wikimedia > 0:
        wiki_dir = args.out or "./data/wikimedia"
        wiki_files = download_wikimedia_photos(wiki_dir, args.wikimedia)
        all_files.extend(wiki_files)

    if not all_files:
        raise RuntimeError(
            "No images found.\n"
            "Provide --photos <folder> and/or --wikimedia <count>"
        )

    # Deduplicate and shuffle
    all_files = sorted(set(all_files))
    random.shuffle(all_files)
    print(f"\nTotal images: {len(all_files)}")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    cache_dir = Path(args.cache_dir)
    dataset   = SelfSupervisedDataset(all_files, str(cache_dir), args.crop_size)

    # Pre-warm label cache in a single pass (fast — no NN involved)
    uncached = [i for i in range(len(dataset)) if i not in dataset.cache]
    if uncached:
        print(f"Computing labels for {len(uncached)} images…")
        for i in tqdm(uncached, ncols=70):
            try:
                img = Image.open(all_files[i]).convert("RGB")
                dataset._get_label(i, img)
            except Exception:
                pass
        dataset.save_cache()
        print("Label cache saved.")

    loader = DataLoader(
        dataset,
        batch_size    = args.batch,
        shuffle       = True,
        num_workers   = min(4, os.cpu_count() or 1),
        pin_memory    = device.type == "cuda",
        drop_last     = True,
    )
    print(f"Batches/epoch: {len(loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PhotoEnhancerNet(pretrained=True).to(device)

    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(),     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss   = math.inf

    if args.resume and (ckpt_dir / "last.pt").exists():
        ckpt        = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", math.inf)
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss = 0.0
        t0      = time.time()

        for img_t, labels in loader:
            img_t  = img_t.to(device, non_blocking=True)    # (B, 3, H, W)
            labels = labels.to(device, non_blocking=True)   # (B, 15)

            backbone_inp = make_backbone_input(img_t, mean, std)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(backbone_inp)   # (B, 15)

                # Weighted MSE: exposure/contrast/highlights are more perceptually
                # important than tint or noise; scale their loss up.
                weights = torch.ones(NUM_PARAMS, device=device)
                weights[[0, 2, 4, 7]] = 2.0   # exposure, highlights, contrast, saturation
                loss = ((pred - labels).pow(2) * weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()

        scheduler.step()

        avg     = ep_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:03d}/{args.epochs}  loss={avg:.4f}  ({elapsed:.0f}s)")

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }, ckpt_dir / "last.pt")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ↳ Best saved  ({best_loss:.4f})")

    print(f"\nTraining done. Best loss: {best_loss:.4f}")

    # ── Export ONNX ───────────────────────────────────────────────────────────
    model.eval()
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location="cpu"))
    model = model.cpu()

    import torch.onnx
    dummy     = torch.randn(1, 3, 224, 224)
    out_path  = args.output

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out_path,
            opset_version=17,
            input_names=["input"], output_names=["params"],
            dynamic_axes={"input": {0: "batch"}, "params": {0: "batch"}},
            do_constant_folding=True,
        )

    print(f"\nExported → {out_path}")
    print("Place enhancer_params.onnx next to LumaPhoto.exe and restart the app.")

    # Quick verification
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path)
    out  = sess.run(None, {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})[0]
    print(f"\nSample output for random input:")
    for name, val in zip(PARAM_NAMES, out[0]):
        print(f"  {name:12s}: {val:+.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="Self-supervised LumaPhoto NN training")
    p.add_argument("--photos",     nargs="+", default=None,
                   help="One or more folders of your own photos (any format)")
    p.add_argument("--wikimedia",  type=int, default=0,
                   help="Number of free Wikimedia Commons photos to download (0 = skip)")
    p.add_argument("--out",        default="./data/wikimedia",
                   help="Where to save downloaded Wikimedia photos")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--crop_size",  type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--cache_dir",  default="./label_cache")
    p.add_argument("--output",     default="enhancer_params.onnx")
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
