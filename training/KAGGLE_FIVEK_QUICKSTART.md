# Kaggle FiveK Training Quickstart

This is the fastest free path for LumaPhoto Auto Enhance training without tying up your PC.

## 1. Upload Files As Kaggle Datasets

You currently have:

```text
D:\Downloads\fivek\
  input\
  expertC\

D:\Downloads\fivek.zip
```

Also upload the current LumaPhoto training source package:

```text
D:\Downloads\LumaPhoto\LumaPhoto\kaggle_lumaphoto_training_src.zip
```

On Kaggle:

1. Go to `https://www.kaggle.com/datasets`.
2. Click `New Dataset`.
3. Upload `D:\Downloads\fivek.zip`.
4. Name it something like `lumaphoto-fivek-c`.
5. Keep it private if you want.
6. Create another small private dataset for `kaggle_lumaphoto_training_src.zip`, or add that zip to the same dataset.

This dataset can train only the center Natural model because it contains Expert C only.

For the full slider, upload datasets that also contain:

```text
expertE\  -> Dramatic left
expertA\  -> Bright right
```

## 2. Create A Kaggle Notebook

1. Go to `https://www.kaggle.com/code`.
2. Click `New Notebook`.
3. Open notebook settings.
4. Set Accelerator to `GPU T4 x2` or any available GPU.
5. Add your `lumaphoto-fivek-c` dataset in the right-side `Add data` panel.
6. Add the dataset containing `kaggle_lumaphoto_training_src.zip`.

## 3. Paste This Cell

Change `EXPERT` to `c`, `e`, or `a`.

```python
EXPERT = "c"       # c = Natural, e = Dramatic, a = Bright
EPOCHS = 30        # try 20-30 first; raise to 60 if quality is not enough
BATCH_SIZE = 16    # lower to 8 if Kaggle reports CUDA out-of-memory

import os, sys, subprocess
from pathlib import Path
import shutil, zipfile

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "onnx", "onnxruntime"], check=True)

TRAINING = Path("/kaggle/working/lumaphoto_training")
TRAINING.mkdir(parents=True, exist_ok=True)

src_zip = next(Path("/kaggle/input").rglob("kaggle_lumaphoto_training_src.zip"), None)
if src_zip:
    with zipfile.ZipFile(src_zip) as z:
        z.extractall(TRAINING)
    nested = TRAINING / "training"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), TRAINING / item.name)
        nested.rmdir()
else:
    train_py = next(Path("/kaggle/input").rglob("train.py"), None)
    if train_py is None:
        raise RuntimeError("Could not find kaggle_lumaphoto_training_src.zip or train.py in attached datasets.")
    for name in ["train.py", "dataset.py", "model.py", "pipeline.py", "losses.py", "export_onnx.py", "requirements.txt"]:
        src = train_py.parent / name
        if src.exists():
            shutil.copy(src, TRAINING / name)

os.chdir(TRAINING)

def find_fivek_root():
    roots = []
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_dir():
            names = {x.name.lower() for x in p.iterdir() if x.is_dir()}
            has_input = "input" in names or "raw" in names
            has_expert = f"expert{EXPERT}".lower() in names or EXPERT.lower() in names
            if has_input and has_expert:
                roots.append(p)
    if roots:
        return roots[0]
    raise RuntimeError("Could not find FiveK layout. Expected input/raw plus expert folder.")

fivek_root = find_fivek_root()
print("FiveK root:", fivek_root)

ckpt_dir = Path(f"/kaggle/working/checkpoints_{EXPERT}")
subprocess.run([
    sys.executable, "train.py",
    "--fivek_root", str(fivek_root),
    "--fivek_expert", EXPERT,
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--workers", "2",
    "--checkpoint_dir", str(ckpt_dir),
    "--resume",
], check=True)

out_path = Path(f"/kaggle/working/fivek_expert_{EXPERT}.onnx")
subprocess.run([
    sys.executable, "export_onnx.py",
    "--checkpoint", str(ckpt_dir / "best.pt"),
    "--expert", EXPERT,
    "--output", str(out_path),
], check=True)

print("Done. Download these from Kaggle Output:")
print(out_path)
print(out_path.with_suffix(".json"))
```

## 4. Download Outputs

After the notebook finishes, download:

```text
fivek_expert_c.onnx
fivek_expert_c.json
```

Place both in:

```text
D:\Downloads\LumaPhoto\LumaPhoto\publish\
```

For the full slider, repeat for `e` and `a`.

## Cost-Friendly Settings

Start with:

```text
EPOCHS = 20 or 30
BATCH_SIZE = 16
```

Only train to 60 epochs if the exported model still looks weak.
