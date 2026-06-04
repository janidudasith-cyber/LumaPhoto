"""
Dataset loaders for three public image-enhancement datasets:

  1. MIT-Adobe FiveK  — 5 000 RAW→expert-retouched pairs.
     Download: https://data.csail.mit.edu/graphics/fivek/  (free, requires agreement)
     Pre-processed JPG mirrors are also available — see README.md.

  2. PPR10K           — 11 161 images × 3 expert retouches = 33 483 pairs.
     Paper / data:  https://github.com/csjliang/PPR10K
     Google Drive / Baidu Yun links are listed in the repository.

  3. DPED             — iPhone / BlackBerry / Sony vs. Canon DSLR patches.
     Download: https://people.ee.ethz.ch/~ihnatova/dped.html  (free)
     run `python download_data.py --dataset dped` to fetch automatically.

  4. SyntheticDataset — Self-supervised: randomly degrades any folder of
     high-quality images (no pairing needed).  Works with Open Images,
     Unsplash Lite, DIV2K, Flickr2K, COCO, etc.

All datasets return (input_img, target_img) tensors in [0, 1].
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

# ImageNet normalisation constants (used for the model backbone input)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalize   = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
_unnormalize = transforms.Normalize(
    [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    [1.0 / s for s in IMAGENET_STD],
)


# ---------------------------------------------------------------------------
# Shared augmentation helpers
# ---------------------------------------------------------------------------

def _make_transforms(crop_size: int = 256, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
        ])
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
    ])


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → float32 (C, H, W) in [0, 1]"""
    return transforms.functional.to_tensor(img)


def _resize_min(img: Image.Image, min_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) < min_side:
        scale = min_side / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img


# ---------------------------------------------------------------------------
# MIT-Adobe FiveK
# ---------------------------------------------------------------------------

class FiveKDataset(Dataset):
    """
    Expects a folder structure produced by the official download or any
    pre-processed JPG mirror:

        <root>/
            input/          ← original / input images
                a0001.jpg
                a0002.jpg
                ...
            expertC/        ← retouched by Expert C (highest quality)
                a0001.jpg
                a0002.jpg
                ...

    Expert C is the most commonly cited reference in literature.  If you
    downloaded a different expert's folder, pass `expert="expertA"`, etc.
    """

    def __init__(
        self,
        root: str,
        expert: str = "expertC",
        crop_size: int = 256,
        train: bool = True,
    ):
        self.input_dir  = Path(root) / "input"
        self.target_dir = Path(root) / expert

        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(
                f"FiveK root '{root}' must contain 'input/' and '{expert}/' subdirs.\n"
                "Download from https://data.csail.mit.edu/graphics/fivek/"
            )

        self.files = sorted(f.name for f in self.input_dir.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        self.aug   = _make_transforms(crop_size, train)
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        inp  = Image.open(self.input_dir  / name).convert("RGB")
        tgt  = Image.open(self.target_dir / name).convert("RGB")

        # Resize both so the short side ≥ 288 (leaves room for 256-crop)
        inp = _resize_min(inp, 288)
        tgt = tgt.resize(inp.size, Image.BICUBIC)

        # Apply the same random crop / flip to both
        seed = random.randint(0, 2**31)
        torch.manual_seed(seed)
        random.seed(seed)
        inp_t = self.aug(_to_tensor(inp))

        torch.manual_seed(seed)
        random.seed(seed)
        tgt_t = self.aug(_to_tensor(tgt))

        return inp_t, tgt_t


# ---------------------------------------------------------------------------
# PPR10K
# ---------------------------------------------------------------------------

class PPR10KDataset(Dataset):
    """
    Structure after extracting the PPR10K download:

        <root>/
            source/          ← original photos
                0001.tif
                ...
            target_a/        ← expert A retouches
            target_b/
            target_c/

    All three experts are used by default (triples the dataset size).
    """

    def __init__(
        self,
        root: str,
        experts: tuple = ("a", "b", "c"),
        crop_size: int = 256,
        train: bool = True,
    ):
        self.source_dir = Path(root) / "source"
        self.target_dirs = [Path(root) / f"target_{e}" for e in experts]

        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"PPR10K root '{root}' must contain 'source/' and 'target_a/b/c/' subdirs.\n"
                "Download from https://github.com/csjliang/PPR10K"
            )

        src_files = sorted(f.name for f in self.source_dir.iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        # Build (source_name, target_dir) pairs
        self.pairs = [
            (name, tdir)
            for name in src_files
            for tdir in self.target_dirs
            if (tdir / name).exists()
        ]

        self.aug   = _make_transforms(crop_size, train)
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        name, tdir = self.pairs[idx]
        inp = Image.open(self.source_dir / name).convert("RGB")
        tgt = Image.open(tdir / name).convert("RGB")

        inp = _resize_min(inp, 288)
        tgt = tgt.resize(inp.size, Image.BICUBIC)

        seed = random.randint(0, 2**31)
        torch.manual_seed(seed); random.seed(seed)
        inp_t = self.aug(_to_tensor(inp))

        torch.manual_seed(seed); random.seed(seed)
        tgt_t = self.aug(_to_tensor(tgt))

        return inp_t, tgt_t


# ---------------------------------------------------------------------------
# DPED
# ---------------------------------------------------------------------------

class DPEDDataset(Dataset):
    """
    DPED: DSLR Photo Enhancement Dataset.
    Uses phone-vs-DSLR patch pairs (128×128) as training pairs.
    The phone image is `input`, the DSLR patch is `target`.

    Structure after extracting:
        <root>/
            iphone/          ← iPhone patches (or blackberry / sony)
                train/
                    patch_xxxxx.jpg
            canon/           ← Canon DSLR reference patches
                train/
                    patch_xxxxx.jpg

    Download: https://people.ee.ethz.ch/~ihnatova/dped.html
    Or run:   python download_data.py --dataset dped
    """

    def __init__(
        self,
        root: str,
        phone: str = "iphone",
        dslr: str = "canon",
        split: str = "train",
        crop_size: int = 128,
    ):
        phone_dir = Path(root) / phone / split
        dslr_dir  = Path(root) / dslr  / split

        if not phone_dir.exists() or not dslr_dir.exists():
            raise FileNotFoundError(
                f"DPED dirs not found: {phone_dir} / {dslr_dir}\n"
                "Download from https://people.ee.ethz.ch/~ihnatova/dped.html"
            )

        phone_files = {f.name for f in phone_dir.iterdir() if f.suffix == ".jpg"}
        dslr_files  = {f.name for f in dslr_dir.iterdir()  if f.suffix == ".jpg"}
        common = sorted(phone_files & dslr_files)

        self.phone_dir = phone_dir
        self.dslr_dir  = dslr_dir
        self.files     = common
        self.crop      = crop_size
        self.train     = split == "train"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        inp  = _to_tensor(Image.open(self.phone_dir / name).convert("RGB"))
        tgt  = _to_tensor(Image.open(self.dslr_dir  / name).convert("RGB"))

        if self.train and inp.shape[-1] >= self.crop and inp.shape[-2] >= self.crop:
            i = random.randint(0, inp.shape[-2] - self.crop)
            j = random.randint(0, inp.shape[-1] - self.crop)
            inp = inp[:, i:i+self.crop, j:j+self.crop]
            tgt = tgt[:, i:i+self.crop, j:j+self.crop]

            if random.random() > 0.5:
                inp = torch.flip(inp, [-1])
                tgt = torch.flip(tgt, [-1])

        return inp, tgt


# ---------------------------------------------------------------------------
# SyntheticDataset — self-supervised on any image folder
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Self-supervised enhancement dataset.

    Takes any folder of high-quality images (Unsplash Lite, DIV2K, Flickr2K,
    Open Images, COCO, etc.) and creates training pairs by randomly degrading
    the image.  The "target" is the original; the "input" is the degraded version.

    Synthetic degradations:
      - Random exposure shift (±1.5 stops)
      - Random contrast change (±40%)
      - Random saturation shift
      - Random warm/cool color shift
      - Random highlight blow-out (overexposure)
      - Random shadow crush

    This gives unlimited training pairs from any photo collection.
    """

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}

    def __init__(
        self,
        image_dir: str,
        crop_size: int = 256,
        train: bool = True,
        max_images: Optional[int] = None,
    ):
        self.files = sorted(
            p for p in Path(image_dir).rglob("*") if p.suffix.lower() in self.EXTS
        )
        if max_images:
            self.files = self.files[:max_images]
        if not self.files:
            raise FileNotFoundError(f"No images found in '{image_dir}'")

        self.aug       = _make_transforms(crop_size, train)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.files)

    def _degrade(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random photographic degradations to a clean image patch."""
        x = img.clone()

        # Random exposure shift
        ev = random.uniform(-1.5, 1.5)
        x = x * (2.0 ** ev)

        # Random contrast
        co = random.uniform(0.6, 1.6)
        x = (x - 0.5) * co + 0.5

        # Random saturation
        lum = x[0:1] * 0.299 + x[1:2] * 0.587 + x[2:3] * 0.114
        sat = random.uniform(0.3, 1.8)
        x = lum + (x - lum) * sat

        # Random warmth (colour temperature shift)
        wt = random.uniform(-0.15, 0.15)
        x[0] = x[0] + wt
        x[2] = x[2] - wt

        # Occasional shadow crush / highlight blow
        if random.random() < 0.3:
            crush = random.uniform(0.0, 0.12)
            x = x - crush
        if random.random() < 0.3:
            scale = random.uniform(1.1, 1.4)
            x = x * scale

        return torch.clamp(x, 0.0, 1.0)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        img = _resize_min(img, 288)

        # Same spatial crop for both input and target
        seed = random.randint(0, 2**31)
        torch.manual_seed(seed); random.seed(seed)
        clean = self.aug(_to_tensor(img))

        degraded = self._degrade(clean)
        return degraded, clean


# ---------------------------------------------------------------------------
# Factory — build a combined training dataset from available sources
# ---------------------------------------------------------------------------

def build_dataset(
    fivek_root:     Optional[str] = None,
    ppr10k_root:    Optional[str] = None,
    dped_root:      Optional[str] = None,
    synthetic_dirs: Optional[list] = None,
    crop_size:      int = 256,
    train:          bool = True,
) -> Dataset:
    """
    Build a combined dataset from whichever sources are available.
    At least one source must be provided.
    """
    datasets = []

    if fivek_root and Path(fivek_root).exists():
        try:
            ds = FiveKDataset(fivek_root, expert="expertA", crop_size=crop_size, train=train)
            datasets.append(ds)
            print(f"FiveK: {len(ds)} pairs")
        except FileNotFoundError as e:
            print(f"FiveK skipped: {e}")

    if ppr10k_root and Path(ppr10k_root).exists():
        try:
            ds = PPR10KDataset(ppr10k_root, crop_size=crop_size, train=train)
            datasets.append(ds)
            print(f"PPR10K: {len(ds)} pairs")
        except FileNotFoundError as e:
            print(f"PPR10K skipped: {e}")

    if dped_root and Path(dped_root).exists():
        try:
            ds = DPEDDataset(dped_root, crop_size=min(crop_size, 128), split="train" if train else "test")
            datasets.append(ds)
            print(f"DPED: {len(ds)} pairs")
        except FileNotFoundError as e:
            print(f"DPED skipped: {e}")

    if synthetic_dirs:
        for d in synthetic_dirs:
            if d and Path(d).exists():
                try:
                    ds = SyntheticDataset(d, crop_size=crop_size, train=train)
                    datasets.append(ds)
                    print(f"Synthetic ({d}): {len(ds)} images")
                except FileNotFoundError as e:
                    print(f"Synthetic skipped: {e}")

    if not datasets:
        raise RuntimeError(
            "No datasets found. Provide at least one of:\n"
            "  --fivek_root, --ppr10k_root, --dped_root, --synthetic_dirs"
        )

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
