"""
Train the LumaPhoto PhotoEnhancerNet on one or more enhancement datasets.

Quick start:
    # With FiveK + synthetic data from any photos folder:
    python train.py --fivek_root ./data/fivek --synthetic_dirs ./data/photos

    # Multi-dataset:
    python train.py \\
        --fivek_root   ./data/fivek \\
        --ppr10k_root  ./data/ppr10k \\
        --dped_root    ./data/dped \\
        --synthetic_dirs ./data/open_images ./data/flickr2k

After training, run:
    python export_onnx.py --checkpoint ./checkpoints/best.pt
to produce enhancer_params.onnx for the LumaPhoto app.
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import build_dataset
from losses import PhotoLoss
from model import PhotoEnhancerNet
from pipeline import apply_params

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalize   = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
_unnormalize = transforms.Normalize(
    [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    [1.0 / s for s in IMAGENET_STD],
)


def make_model_input(img: torch.Tensor) -> torch.Tensor:
    """Resize to 224×224 and apply ImageNet normalisation for the backbone."""
    small = torch.nn.functional.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
    return _normalize(small)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = build_dataset(
        fivek_root     = args.fivek_root,
        ppr10k_root    = args.ppr10k_root,
        dped_root      = args.dped_root,
        synthetic_dirs = args.synthetic_dirs,
        crop_size      = args.crop_size,
        train          = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size    = args.batch_size,
        shuffle       = True,
        num_workers   = args.workers,
        pin_memory    = device.type == "cuda",
        drop_last     = True,
    )
    print(f"Training samples: {len(train_ds)} | Batches/epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = PhotoEnhancerNet(pretrained=True).to(device)
    criterion = PhotoLoss(w_l1=1.0, w_perc=0.1, w_ssim=0.5).to(device)

    # Separate learning rates: backbone gets a lower LR (it's pretrained)
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=1, eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_loss   = math.inf
    ckpt_dir    = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and (ckpt_dir / "last.pt").exists():
        ckpt        = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", math.inf)
        print(f"Resumed from epoch {start_epoch}")

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()
        detail: dict = {}

        for step, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True)   # (B, 3, H, W) in [0, 1]
            tgt = tgt.to(device, non_blocking=True)

            # Build normalised 224×224 tensor for the backbone
            model_inp = make_model_input(inp)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                params = model(model_inp)             # (B, 15)
                pred   = apply_params(inp, params)    # (B, 3, H, W) in [0, 1]
                loss, detail = criterion(pred, tgt, params)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % max(1, len(train_loader) // 10) == 0:
                lr_now = optimizer.param_groups[-1]["lr"]
                print(
                    f"  E{epoch:03d} [{step:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"l1={detail['l1']:.4f}  "
                    f"perc={detail['perc']:.4f}  "
                    f"ssim={detail['ssim']:.4f}  "
                    f"lr={lr_now:.2e}"
                )

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - epoch_start
        print(f"Epoch {epoch:03d} — avg loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

        # Save checkpoints
        ckpt_state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(ckpt_state, ckpt_dir / "last.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ↳ New best: {best_loss:.4f}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Run: python export_onnx.py --checkpoint {ckpt_dir / 'best.pt'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train LumaPhoto PhotoEnhancerNet")

    # Dataset sources
    p.add_argument("--fivek_root",     type=str, default=None)
    p.add_argument("--ppr10k_root",    type=str, default=None)
    p.add_argument("--dped_root",      type=str, default=None)
    p.add_argument("--synthetic_dirs", type=str, nargs="+", default=None)

    # Training hyper-parameters
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--crop_size",   type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--workers",     type=int,   default=4)

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume",         action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
