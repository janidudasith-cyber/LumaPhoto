# Model files go here

The `.onnx` weights are **not** committed — `.gitignore` excludes `*.onnx`, and they
are too large for git. Download and drop them in this folder.

| File | Size | Used by | How it reaches users | Licence |
|---|---|---|---|---|
| `u2netp.onnx` | ~4.7 MB | `ModelDescriptor.U2NetP` (fallback default) | ships in installer | Apache-2.0 ✅ |
| `u2net_human_seg.onnx` | ~176 MB | `ModelDescriptor.U2NetHumanSeg` — **use this for group photos**; the general models keep only the most salient person, this one masks everyone in frame | in-app download | Apache-2.0 ✅ |
| `u2net.onnx` | ~176 MB | `ModelDescriptor.U2Net` (better hair/fur on non-people subjects) | manual drop-in | Apache-2.0 ✅ |
| `isnet-general-use.onnx` | ~173 MB | `ModelDescriptor.IsNet` | manual drop-in | ⚠️ VERIFY BEFORE COMMERCIAL USE |

Only the lite model ships in the installer. The human-seg model is fetched on demand
by `ModelDownloader` (see the link under the Remove Background button) into this same
folder next to the exe; the other two can be dropped in by hand. Keeping the installer
at ~5 MB matters more than shipping weights most users never touch.

The app picks the **best model present automatically** (IS-Net → Human → Full → Lite;
see `BgModelPreference` in `BackgroundRemoval.cs`) — drop a better `.onnx` in this
folder next to the installed exe and the next launch uses it. The active model is
shown under the Remove Background button.

Verified 2026-08-29: `u2netp`, `u2net_human_seg`, and `u2net` all export input
`input.1` at `[1,3,320,320]` with the fused prediction as the first output, so they
are drop-in for `U2NetBackgroundRemover` with the existing descriptors.

Source: the U²-Net ONNX exports published by the `danielgatis/rembg` project
(`https://github.com/danielgatis/rembg` → releases). Verify the licence of the
specific `.onnx` file before packaging it for distribution.

Build action is already configured in `LumaPhoto.Vision.csproj`: any `.onnx` here
is copied next to the exe as a loose file (`ExcludeFromSingleFile`), so
`AppContext.BaseDirectory` resolution keeps working in a single-file publish.

Without a model present, `OnnxSessionManager` throws `FileNotFoundException` on
first use — the UI should catch that and tell the user to install the model rather
than crashing.
