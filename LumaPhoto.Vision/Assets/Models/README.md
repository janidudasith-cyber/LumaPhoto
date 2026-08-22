# Model files go here

The `.onnx` weights are **not** committed — `.gitignore` excludes `*.onnx`, and they
are too large for git. Download and drop them in this folder.

| File | Size | Used by | Licence |
|---|---|---|---|
| `u2netp.onnx` | ~4.7 MB | `ModelDescriptor.U2NetP` (default, ships in installer) | Apache-2.0 ✅ |
| `u2net.onnx` | ~176 MB | `ModelDescriptor.U2Net` (optional download, better hair/fur) | Apache-2.0 ✅ |
| `isnet-general-use.onnx` | ~173 MB | `ModelDescriptor.IsNet` | ⚠️ VERIFY BEFORE COMMERCIAL USE |

Source: the U²-Net ONNX exports published by the `danielgatis/rembg` project
(`https://github.com/danielgatis/rembg` → releases). Verify the licence of the
specific `.onnx` you download before shipping it in a paid build.

Build action is already configured in `LumaPhoto.Vision.csproj`: any `.onnx` here
is copied next to the exe as a loose file (`ExcludeFromSingleFile`), so
`AppContext.BaseDirectory` resolution keeps working in a single-file publish.

Without a model present, `OnnxSessionManager` throws `FileNotFoundException` on
first use — the UI should catch that and tell the user to install the model rather
than crashing.
