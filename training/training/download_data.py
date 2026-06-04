"""
Dataset download helpers.

Usage:
    python download_data.py --dataset dped      --out ./data/dped
    python download_data.py --dataset div2k     --out ./data/div2k
    python download_data.py --dataset flickr2k  --out ./data/flickr2k

MIT-Adobe FiveK and PPR10K require manual download (see instructions below).
"""

import argparse
import os
import shutil
import time
import zipfile
from pathlib import Path
import urllib.request
import urllib.error

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


def _gdrive_url(file_id: str) -> str:
    """Convert a Google Drive file ID to a direct download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _gdrive_confirm_url(file_id: str, confirm_token: str) -> str:
    return f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"


def _download(url: str, dest: Path, retries: int = 5, chunk_size: int = 1024 * 1024):
    """
    Download url → dest with resume support and retry logic.
    - Resumes partial downloads using HTTP Range header.
    - Retries up to `retries` times with exponential back-off.
    - Handles Google Drive large-file confirmation tokens automatically.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check for a Google Drive file ID pattern in the URL
    is_gdrive = "drive.google.com" in url or "docs.google.com" in url

    # If a partial file exists, check it isn't an HTML error page from a previous attempt.
    # If it is, delete it so we start fresh rather than appending to garbage.
    if dest.exists() and dest.stat().st_size > 0:
        with open(dest, "rb") as _f:
            header = _f.read(16)
        if header.lstrip().startswith(b"<!") or header.lstrip().startswith(b"<html"):
            print(f"  Corrupt file detected (HTML error page) — deleting and restarting: {dest.name}")
            dest.unlink()

    existing = dest.stat().st_size if dest.exists() else 0
    if existing:
        print(f"  Resuming {dest.name} from {existing // 1024 // 1024:,} MB …")
    else:
        print(f"  Downloading {dest.name} …")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)
            if existing:
                req.add_header("Range", f"bytes={existing}-")

            with urllib.request.urlopen(req, timeout=60) as resp:
                # Handle Google Drive virus-scan confirmation page
                if is_gdrive and resp.headers.get("Content-Type", "").startswith("text/html"):
                    html = resp.read().decode("utf-8", errors="replace")
                    # Extract confirm token from the warning page
                    token = None
                    for part in html.split("confirm="):
                        if len(part) > 3:
                            token = part.split("&")[0].split('"')[0]
                            break
                    if token:
                        # Re-extract file ID and build confirmed URL
                        fid = url.split("id=")[-1].split("&")[0]
                        url = _gdrive_confirm_url(fid, token)
                        req = urllib.request.Request(url)
                        if existing:
                            req.add_header("Range", f"bytes={existing}-")
                        resp.close()
                        resp = urllib.request.urlopen(req, timeout=60)

                # Abort early if the server returned an HTML error page instead of the file.
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    body_preview = resp.read(512).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Server returned an HTML page instead of the file.\n"
                        f"URL: {url}\nPreview: {body_preview[:200]}"
                    )

                total_str = resp.headers.get("Content-Length") or resp.headers.get("content-range", "").split("/")[-1]
                total = int(total_str) if total_str and total_str.isdigit() else 0
                downloaded = existing

                mode = "ab" if existing else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            full_total = total + existing
                            pct = min(100, downloaded * 100 // full_total)
                            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                            print(f"\r  [{bar}] {pct:3d}%  {downloaded // 1024 // 1024:,} MB", end="", flush=True)
                        else:
                            print(f"\r  {downloaded // 1024 // 1024:,} MB downloaded", end="", flush=True)

            # Reject suspiciously small files — a real dataset archive is never < 100 KB.
            final_size = dest.stat().st_size if dest.exists() else 0
            if final_size < 100 * 1024:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded file is only {final_size} bytes — server probably returned "
                    f"an error page or redirect instead of the real file.\n"
                    f"URL: {url}"
                )

            print(f"\n  Done: {dest.name}  ({final_size // 1024 // 1024:,} MB)")
            return

        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as exc:
            existing = dest.stat().st_size if dest.exists() else 0
            if attempt < retries:
                wait = 2 ** attempt
                print(f"\n  Error: {exc}. Retry {attempt}/{retries - 1} in {wait}s …")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Download failed after {retries} attempts: {url}") from exc


def _extract(archive: Path, out_dir: Path):
    print(f"  Extracting {archive.name} …")
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(out_dir)
        elif archive.suffix in (".tar", ".tgz") or archive.name.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(archive) as t:
                t.extractall(out_dir)
        else:
            raise ValueError(f"Unknown archive format: {archive.suffix}")
    except (zipfile.BadZipFile, Exception) as exc:
        # The archive is corrupt (likely an HTML error page saved by a previous run).
        # Delete it so the next run downloads it fresh.
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"Archive '{archive.name}' is corrupt and has been deleted.\n"
            f"Re-run the download command to fetch it again.\n"
            f"Original error: {exc}"
        ) from exc
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
# Manual-download instructions
# ---------------------------------------------------------------------------

FIVEK_INSTRUCTIONS = """
MIT-Adobe FiveK requires a free registration agreement at:
  https://data.csail.mit.edu/graphics/fivek/

After downloading, extract so the structure is:
  data/fivek/input/     ← original images (a0001.jpg … a5000.jpg)
  data/fivek/expertC/   ← retouched by Expert C

A pre-processed JPG version is also available from community mirrors.
Search GitHub for "adobe fivek preprocessed jpg" for ready-to-use links.
"""

PPR10K_INSTRUCTIONS = """
PPR10K is available from the authors' GitHub page:
  https://github.com/csjliang/PPR10K

Download links (Google Drive / Baidu Yun) are listed in the README.
After downloading, extract so the structure is:
  data/ppr10k/source/    ← original photos
  data/ppr10k/target_a/  ← expert A retouches
  data/ppr10k/target_b/
  data/ppr10k/target_c/
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download enhancement training datasets")
    p.add_argument("--dataset", required=True,
                   choices=["dped", "div2k", "flickr2k", "fivek", "ppr10k"])
    p.add_argument("--out", default="./data", help="Output directory")
    args = p.parse_args()

    if args.dataset == "dped":
        download_dped(args.out)
    elif args.dataset == "div2k":
        download_div2k(args.out)
    elif args.dataset == "flickr2k":
        download_flickr2k(args.out)
    elif args.dataset == "fivek":
        print(FIVEK_INSTRUCTIONS)
    elif args.dataset == "ppr10k":
        print(PPR10K_INSTRUCTIONS)
