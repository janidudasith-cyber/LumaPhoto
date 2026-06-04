"""
Dataset download helpers.

Automatic downloads:
    python download_data.py --dataset dped      --out ./data/dped
    python download_data.py --dataset div2k     --out ./data/div2k
    python download_data.py --dataset flickr2k  --out ./data/flickr2k
    python download_data.py --dataset lol       --out ./data/lol

Instructions for manual downloads (requires free registration/agreement):
    python download_data.py --dataset fivek
    python download_data.py --dataset ppr10k
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
import urllib.request

# ---------------------------------------------------------------------------
# DPED — direct HTTP download from ETH Zurich
# 3 phone cameras vs Canon DSLR, ~24 K training patches each
# ---------------------------------------------------------------------------
DPED_URLS = {
    "iphone_train":  "https://people.ee.ethz.ch/~ihnatova/dped/iphone_train.zip",
    "iphone_test":   "https://people.ee.ethz.ch/~ihnatova/dped/iphone_test.zip",
    "canon_train":   "https://people.ee.ethz.ch/~ihnatova/dped/canon_train.zip",
    "canon_test":    "https://people.ee.ethz.ch/~ihnatova/dped/canon_test.zip",
}

# ---------------------------------------------------------------------------
# DIV2K — 800 high-resolution training images, no specific pairing needed
# (used as synthetic dataset source)
# ---------------------------------------------------------------------------
DIV2K_URLS = {
    "train_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

# ---------------------------------------------------------------------------
# Flickr2K — 2650 high-resolution images (used as synthetic source)
# ---------------------------------------------------------------------------
FLICKR2K_URL = "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"


def _progress(count, block, total):
    done = count * block
    if total > 0:
        pct = min(100, done * 100 // total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {done // 1024 // 1024:,} MB", end="", flush=True)


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {url.split('/')[-1]} …")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _extract(archive: Path, out_dir: Path):
    print(f"  Extracting {archive.name} …")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif archive.suffix in (".tar", ".tgz") or archive.name.endswith(".tar.gz"):
        import tarfile
        with tarfile.open(archive) as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive.suffix}")
    print(f"  Extracted to {out_dir}")


def download_dped(out_dir: str):
    out = Path(out_dir)
    print("Downloading DPED …")
    for name, url in DPED_URLS.items():
        arch = out / f"{name}.zip"
        _download(url, arch)
        _extract(arch, out)
    print(f"DPED ready at {out}")
    print("Expected layout:")
    print("  dped/iphone/train/  dped/iphone/test/")
    print("  dped/canon/train/   dped/canon/test/")


def download_div2k(out_dir: str):
    out = Path(out_dir)
    print("Downloading DIV2K …")
    for name, url in DIV2K_URLS.items():
        arch = out / f"{name}.zip"
        _download(url, arch)
        _extract(arch, out)
    print(f"DIV2K ready at {out}  (use as --synthetic_dirs)")


def download_flickr2k(out_dir: str):
    out = Path(out_dir)
    print("Downloading Flickr2K …")
    arch = out / "Flickr2K.tar"
    _download(FLICKR2K_URL, arch)
    _extract(arch, out)
    print(f"Flickr2K ready at {out}  (use as --synthetic_dirs)")


# ---------------------------------------------------------------------------
# LOL — Low-Light dataset (485 training pairs, ~1 GB)
# Auto-downloadable via Kaggle CLI.  No account needed if running in a
# Kaggle notebook (the dataset is public: soumikrakshit/lol-dataset).
# ---------------------------------------------------------------------------

LOL_KAGGLE_DATASET = "soumikrakshit/lol-dataset"

def download_lol(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Downloading LOL dataset via Kaggle CLI …")
    print(f"  Target: {out}")
    ret = subprocess.run(
        ["kaggle", "datasets", "download", "-d", LOL_KAGGLE_DATASET,
         "--unzip", "-p", str(out)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print("Kaggle CLI failed. Install with:  pip install kaggle")
        print("Then set up ~/.kaggle/kaggle.json (API token from kaggle.com/account)")
        print("\nOr download manually from:")
        print(f"  https://www.kaggle.com/datasets/{LOL_KAGGLE_DATASET}")
        print(f"and extract to {out}/")
        return
    print(f"LOL ready at {out}")
    print("Expected layout:")
    print("  lol/our485/low/   lol/our485/high/")
    print("  lol/eval15/low/   lol/eval15/high/")

import subprocess


# ---------------------------------------------------------------------------
# Manual-download instructions
# ---------------------------------------------------------------------------

FIVEK_INSTRUCTIONS = """
MIT-Adobe FiveK — 5 000 photos retouched by Expert C.

⚠  DO NOT download the full archive — it is 50 GB of RAW (DNG) files
   you don't need.  Get the pre-processed JPEG version instead (~1.5 GB):

Option A — Kaggle (easiest, no account needed to browse):
  1. Go to https://www.kaggle.com/datasets
  2. Search: "MIT Adobe FiveK JPEG"  or  "adobe fivek preprocessed"
  3. Download the dataset that contains input/ and expertC/ folders.
  4. Extract so the structure is:
       data/fivek/input/     ← a0001.jpg … a5000.jpg
       data/fivek/expertC/   ← a0001.jpg … a5000.jpg

Option B — MIT CSAIL individual JPEG exports (~1.5 GB total):
  1. Go to https://data.csail.mit.edu/graphics/fivek/
  2. Accept the free license agreement.
  3. Download ONLY the two JPEG zip files:
       fivek_dataset/HalfAndHalf/JPG/input.zip     (~750 MB)
       fivek_dataset/HalfAndHalf/JPG/expertC.zip   (~750 MB)
     (Skip the DNG/XMP files — those are the 50 GB you don't want.)
  4. Extract both into data/fivek/.

Option C — Skip FiveK entirely:
  PPR10K alone (33 K portrait pairs) is sufficient and better for fixing
  the portrait/landscape misclassification.  FiveK adds general scene
  variety but is not required.

Step 3 — Train:
  python train.py --fivek_root ./data/fivek ...
"""

PPR10K_INSTRUCTIONS = """
PPR10K — 11 161 portrait/people photos, retouched by 3 experts = 33 483 pairs.
This is the best dataset for fixing portrait vs landscape misclassification.

Step 1 — Download from the authors' GitHub releases:
  https://github.com/csjliang/PPR10K

  Google Drive and Baidu Yun links are in the README.
  The full-resolution archive is ~35 GB.
  A 512 px resized version (~5 GB) works equally well for training.

Step 2 — Extract so the folder structure is:
  data/ppr10k/source/    ← original photos
  data/ppr10k/target_a/  ← expert A retouches
  data/ppr10k/target_b/  ← expert B retouches
  data/ppr10k/target_c/  ← expert C retouches

Step 3 — Train:
  python train.py --ppr10k_root ./data/ppr10k ...
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download enhancement training datasets")
    p.add_argument("--dataset", required=True,
                   choices=["dped", "div2k", "flickr2k", "lol", "fivek", "ppr10k"])
    p.add_argument("--out", default="./data", help="Output directory")
    args = p.parse_args()

    if args.dataset == "dped":
        download_dped(args.out)
    elif args.dataset == "div2k":
        download_div2k(args.out)
    elif args.dataset == "flickr2k":
        download_flickr2k(args.out)
    elif args.dataset == "lol":
        download_lol(args.out)
    elif args.dataset == "fivek":
        print(FIVEK_INSTRUCTIONS)
    elif args.dataset == "ppr10k":
        print(PPR10K_INSTRUCTIONS)
