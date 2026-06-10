# LumaPhoto Auto Enhance Neural Training

LumaPhoto's neural Auto Enhance model predicts the same 15 edit parameters used by the app sliders. It does not generate pixels. This keeps Auto Enhance fast, adjustable, and easy to fall back from.

## Three Auto Enhance Models

The Auto Enhance slider uses three FiveK expert models:

```text
-100  Dramatic  -> fivek_expert_e.onnx  (Expert E)
   0  Natural   -> fivek_expert_c.onnx  (Expert C)
+100  Bright    -> fivek_expert_a.onnx  (Expert A, shaped to stay colorful)
```

Train Expert C first so the center position works, then train E and A.

## Recommended First Model

Train one general-purpose model first:

```powershell
python training\train.py `
  --fivek_root .\data\fivek `
  --fivek_expert c `
  --synthetic_dirs .\data\photos `
  --epochs 60 `
  --batch_size 16 `
  --checkpoint_dir .\training\checkpoints `
  --resume
```

Use FiveK Expert C for the default model because it is the most natural/balanced target.

Then repeat with:

```powershell
--fivek_expert e
--fivek_expert a
```

Expected FiveK layouts:

```text
fivek/
  input/ or raw/
  expertC/ or c/
```

## Export

```powershell
python training\export_onnx.py `
  --checkpoint .\training\checkpoints\best.pt `
  --expert c
```

The exporter writes both `fivek_expert_c.onnx` and `fivek_expert_c.json` by default. Copy both files next to `LumaPhoto.exe`, and repeat for experts `e` and `a`.

For release builds, place these six files in `publish/`:

```text
fivek_expert_e.onnx
fivek_expert_e.json
fivek_expert_c.onnx
fivek_expert_c.json
fivek_expert_a.onnx
fivek_expert_a.json
```

On startup, the app loads each model only when the JSON manifest says it is a `fivek-v2` 15-parameter model with the matching expert.

Older unlabeled models are ignored on purpose, because the early PPR10K model was portrait-biased and can make general photos too bright, dull, and desaturated.

## App Contract

The ONNX model must keep this contract:

```text
input   float32 [1, 3, 224, 224]  ImageNet-normalised RGB
output  float32 [1, 15]           LumaPhoto slider parameters
```

The parameter order is defined in `training/pipeline.py` as `PARAM_NAMES` and must match `RawToState()` in `LumaPhoto/NeuralEnhancer.cs`.

## Quality Notes

Use diverse scenes, not only portraits. FiveK Expert C plus synthetic degradation from high-quality general photos is a good baseline. Add LOL low-light data if night/indoor photos still look weak.
