# Luma Photo Editor

A native Windows desktop photo editor built with C# + WPF (.NET 8). Everything runs
locally — no account, no upload, no per-image cost.

## Install

Download the latest `LumaPhoto-Setup-vX.Y.exe` from
[Releases](https://github.com/janidudasith-cyber/LumaPhoto/releases) and run it.
The app checks for newer releases on startup and can update itself.

## Requirements

- Windows 10 or 11 (64-bit)
- Nothing else to install — the installer ships a self-contained build.

To build from source you additionally need the
[.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).

## Build from source

1. **Install the .NET 8 SDK** (Windows → x64 → SDK installer).
2. **Double-click `build.bat`** — restores, compiles, and produces a single
   `LumaPhoto.exe` in `publish\`.
3. **Run `publish\LumaPhoto.exe`.**

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for architecture, the training pipeline, and
installer details.

## Features

The inspector has six tabs: **Adjust · Filters · Crop · Markup · Design · Layers**.

**Editing**

- **Auto Enhance** — per-image analysis with an intensity slider. Blends three
  trained FiveK expert styles (Dramatic ↔ Natural ↔ Bright) when the models are
  present, falling back to rule-based enhancement otherwise.
- **15 manual adjustments** — Exposure, Brilliance, Highlights, Shadows, Contrast,
  Brightness, Black Point, Saturation, Vibrance, Warmth, Tint, Sharpness, Definition,
  Noise Reduction, Vignette.
- **Curves** and an **HSL mix** for per-channel colour work.
- **10 filters** — Original, Vivid, Dramatic, Mono, Silvertone, Noir, Warm, Cool,
  Fade, Process — each with an intensity slider.
- **Crop** — free or fixed aspect ratio, drag handles.
- **Transform** — rotate left/right, flip horizontal/vertical.
- **Compare** — hold the Compare button to see the original.

**Design**

- **Remove Background** — one-click subject cut-out, running locally via ONNX.
  Three edge styles (Balanced / Hair / Product); switching styles re-cuts from the
  original so presets never compound. Uses your GPU via DirectML when available.
  A larger, more accurate model can be downloaded from inside the app — see
  [Background-removal models](#background-removal-models).
- **Collage** — split, stack, grid, and feature layouts, with drag-to-reposition
  inside each slot.
- **Frames** and a configurable **text watermark** (position, opacity, size).
- **Markup** — pen, line, rectangle, arrow; 9 colours; adjustable brush size;
  undo/clear.
- **Layers** — visibility and export toggles for the Watermark, Markup, Frame, and
  Photo layers. This is not a compositing stack, and there are no adjustment layers.

## File formats

| | Formats |
|---|---|
| **Open** | JPG, PNG, BMP, WebP, TIFF, GIF, HEIC/HEIF |
| **Export** | JPEG (default), PNG, TIFF, BMP, GIF |

**WebP and HEIC can be opened but not exported** — Windows ships decoders for both
but no encoders.

**Export as PNG to keep a background-removal cut-out.** JPEG has no alpha channel and
silently flattens transparency onto a solid background.

## Background-removal models

Only the ~4.7 MB **U²-Net Lite** model ships in the installer, which keeps the
download small. It handles hard-edged subjects well.

For group photos, a link under the Remove Background button downloads the ~176 MB
**U²-Net Human** model once; the app then uses it automatically. The general models
are salient-object detectors that tend to keep only the most prominent person, while
the human-segmentation model masks everyone in frame.

The best model present is selected automatically at launch, and the active model and
execution provider are shown under the button. Advanced users can drop other
supported `.onnx` files into `%LOCALAPPDATA%\LumaPhoto\Models` — see
[`LumaPhoto.Vision/Assets/Models/README.md`](LumaPhoto.Vision/Assets/Models/README.md).

## Performance

Rendering runs on a 25 ms debounce so slider dragging stays smooth, and pixel work
uses `unsafe` pointer operations for throughput. During a drag, the sharpness,
definition, and noise passes are skipped and restored on release.

Background-removal inference runs off the UI thread and is serialised through a
semaphore — ONNX sessions are reused for the life of the window, never rebuilt
per image.

## Licence

Dependencies are MIT-licensed or built into Windows (`Microsoft.ML.OnnxRuntime`,
WPF, .NET 8). The bundled U²-Net weights are Apache-2.0. Note that
`isnet-general-use.onnx`, if you add it by hand, has an unverified weight licence —
check before any commercial use.
