# Luma Photo Editor

A native Windows desktop photo editor built with C# + WPF (.NET 8).

## Requirements

- Windows 10 or 11 (64-bit)
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) — free download from Microsoft

## Build & Run

1. **Install .NET 8 SDK** from https://dotnet.microsoft.com/download/dotnet/8.0  
   (Choose: Windows → x64 → SDK installer)

2. **Double-click `build.bat`**  
   This will restore packages, compile, and produce a single `LumaPhoto.exe` in the `publish\` folder.

3. **Run `publish\LumaPhoto.exe`** — no installer needed, just double-click.

## Features

- **Open** JPG, PNG, WebP, BMP — drag & drop or file picker
- **Auto Enhance** — smart per-image analysis with adjustable intensity slider
- **15 manual adjustments** — Exposure, Brilliance, Highlights, Shadows, Contrast, Brightness, Black Point, Saturation, Vibrance, Warmth, Tint, Sharpness, Definition, Noise Reduction, Vignette
- **10 Filters** — Original, Vivid, Dramatic, Mono, Silvertone, Noir, Warm, Cool, Fade, Process — each with intensity slider
- **Crop** — free or fixed aspect ratio, drag handles
- **Transform** — rotate left/right, flip horizontal/vertical
- **Markup** — pen, line, rectangle, arrow tools; 9 colors; adjustable brush size; undo/clear
- **Compare** — hold Compare button to see original
- **Export** — save as PNG or JPEG

## Performance

Rendering runs on a debounced 30ms timer so slider dragging stays smooth. Image processing uses unsafe pointer operations in C# for maximum pixel throughput.
