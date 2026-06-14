# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run (C# App)

**Quickest build** — double-click `build.bat`, or from the repo root:
```
dotnet publish LumaPhoto\LumaPhoto.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -o publish
```
Output: `publish\LumaPhoto.exe` + `publish\enhancer_params.onnx`.

**Debug run:**
```
dotnet run --project LumaPhoto\LumaPhoto.csproj
```

There are no automated tests.

## Installer

`installer.iss` — Inno Setup script at the repo root. Open in Inno Setup Compiler and press F9 to build.
Output: `installer_output\LumaPhoto-Setup-v1.0.exe` — a standard Windows installer (Program Files entry, Start Menu shortcut, uninstaller).
Bundles `publish\LumaPhoto.exe` + `publish\enhancer_params.onnx`.
The `SetupIconFile` points to `LumaPhoto\LumaPhoto.ico` (not the exe — extracting from a 167 MB single-file exe causes a compile error).

## Training Pipeline (Python)

**Primary training platform: Google Colab** (free T4 GPU).
Use `training/colab_notebook.py` — paste into Colab cells and run in order.
Checkpoints are saved to Google Drive at `MyDrive/LumaPhoto/` after every epoch (`last.pt` + `best.pt`).
If the Colab session disconnects, re-running Cell 4 auto-resumes from `last.pt` on Drive.

**Training data:**
- PPR10K Expert A (11,161 portrait pairs) — primary dataset, most vibrant retouches
- DIV2K (800 images) — synthetic augmentation
- Download links stored in `colab_notebook.py` header comments

**Export to ONNX** (after training) — Cell 4 does this automatically, or run the export cell separately:
loads `best.pt` from Drive → exports `enhancer_params.onnx` to `MyDrive/LumaPhoto/enhancer_params.onnx`.
Download and place next to `LumaPhoto.exe`.

**Local training** (optional, CUDA GPU recommended):
```
pip install -r training/requirements.txt
python training/train.py --fivek_root ./data/fivek --synthetic_dirs ./data/photos
```

**File layout:**
- `training/colab_notebook.py` — primary self-contained Colab notebook (use this)
- `training/kaggle_notebook.py` — Kaggle variant (kept for reference)
- `training/training/` — modular training code (dataset.py, model.py, losses.py, pipeline.py, train.py)
- `training/` root files — older versions kept for reference

## Architecture

### C# WPF App

The app is .NET 8 / WPF, targeting `win-x64` as a self-contained single-file executable.

**`MainWindow.xaml.cs`** is a monolithic class that owns all UI state and event handling. Key state fields:
- `_sourcePixels` / `_sourceW` / `_sourceH` — the currently loaded image as a raw BGRA byte array. This is the immutable source; all edits are re-rendered from it on every change.
- `_adj` (`AdjustmentState`) — the 15+ current slider values applied to the source.
- `_history` / `_future` — undo/redo stacks of `HistorySnapshot` (pixel clone + adj clone + markup strokes).
- `_autoBaseParams` / `_autoPreState` — auto-enhance state; `_autoPreState` is the adj saved before Auto was toggled on so it can be fully restored.

**`ImageProcessor.cs`** — all pixel math. Uses `unsafe` pointer operations for throughput. Key methods: `LoadImageFile`, `Analyze` (rule-based scene detection), `ComputeAutoParams`, `RefineWithNN`, `Render` / `RenderToBuffer` / `BufferToBitmap`.

**`NeuralEnhancer.cs`** — ONNX Runtime inference with two operating modes:
- **Mode 1** (preferred): loads `enhancer_params.onnx` next to the exe, runs `PredictParams()` which returns a full `AdjustmentState` directly. Runs both full-image and center-80%-crop inference and averages.
- **Mode 2** (fallback): loads `places365_mobilenet.onnx` + `places365_classes.txt`, classifies scene type, and returns `SceneWeights` used by `RefineWithNN` to blend rule-based params.

**`SliderRow.xaml/.cs`** — reusable labeled slider component. Raises `DragStarted`, `DragCompleted`, and `CommitChange` events; `SetValueSilent` updates the slider without firing change events.

### Inspector tabs

The inspector panel has **4 tabs**: Adjust · Filters · Crop · Markup.
The Layers tab was removed. There are no adjustment layers or layer compositing in the codebase.

### Render path

```
Slider drag → SyncAdjFromSliders() → ScheduleRender() [25 ms debounce]
  → DoRenderAsync() [background thread]
  → ImageProcessor.RenderToBuffer()
  → ImageProcessor.BufferToBitmap()
  → PhotoDisplay.Source = WriteableBitmap
```

During slider drag (`_draggingTransform = true`), sharpness/definition/noise passes are skipped for responsiveness.

### Export

Default export format is **JPEG** (FilterIndex = 1 in SaveFileDialog). Format order: JPEG → PNG → WebP → TIFF.
Batch export also defaults to JPEG (`SelectedIndex = 0` on the format ComboBox).

### Auto Enhance flow

1. `AutoBtn_Click` → `ImageProcessor.Analyze()` (synchronous, rule-based) → `ComputeAutoParams()` → stores as `_autoBaseParams`.
2. `RunNeuralEnhancerAsync()` fires in background → `NeuralEnhancer.PredictParams()` or `.Analyze()` → updates `_autoBaseParams` on the dispatcher.
3. `ApplyAutoStyleAtSliderValue(v)` blends the three FiveK expert endpoints (`_autoDramaticParams`/`_autoNaturalParams`/`_autoBrightParams`) when the models are loaded — Dramatic (v=−100) → Natural (v=0) → Bright (v=+100); it falls back to a pre-auto↔auto interpolation when only rule-based params exist.

### Python training

**`pipeline.py`** is a differentiable PyTorch re-implementation of `ImageProcessor.cs`'s `AdjustPixel` method. The 15 parameters and their index order are defined by `PARAM_NAMES` — **this list is the shared contract between Python and C#**. If you add, remove, or reorder parameters here, you must update `RawToState()` in `NeuralEnhancer.cs` and the corresponding slider in `ImageProcessor.cs` to match.

**`model.py`** — `PhotoEnhancerNet`: EfficientNet-B0 backbone (via `timm`) + 2×2 regional pooling + differentiable `ImageStatsEncoder` (8 photometric stats that mirror the C# rule-based features) + residual linear head → 15 params. Symmetric params use `tanh × 100`; positive-only params use `sigmoid × 100`.

**`losses.py`** — `PhotoLoss`: L1 + VGG perceptual + SSIM + color moment matching. No regularization term (removed — was making outputs too dull).

**`dataset.py`** — `build_dataset()` combines FiveK (Expert A), PPR10K (Expert A), DPED, and synthetic datasets.

**`train.py`** — standard training loop with EMA (decay=0.9999). `best.pt` saves EMA weights only; `last.pt` saves everything for resuming. Use `--resume` to continue an interrupted run.

### Dependencies

The project uses only **MIT-licensed / Windows built-in** libraries:
- `Microsoft.ML.OnnxRuntime` (MIT) — ONNX model inference
- WPF + .NET 8 (Windows) — UI and image I/O

ImageSharp was removed. The app is commercially safe with no paid license requirements.

### Model deployment

| File(s) next to exe | Mode |
|---|---|
| `enhancer_params.onnx` | Direct parameter prediction (best quality) |
| `places365_mobilenet.onnx` + `places365_classes.txt` | Scene classification fallback |
| Neither | Rule-based auto enhance only |

`NeuralEnhancer` is loaded in a background `Task` at startup so the window opens immediately. It is disposed in `MainWindow.Closed`.

### GitHub / distribution

- Source code on GitHub (C#, Python, scripts) — `.gitignore` excludes `publish/`, `bin/`, `obj/`, `*.onnx`, `*.pt`, datasets
- Installer distributed via GitHub Releases as a binary attachment (`LumaPhoto-Setup-v1.0.exe`)
- Trained model (`enhancer_params.onnx`) stored on Google Drive — download and place next to exe after training
