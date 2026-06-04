"""
Export a trained PhotoEnhancerNet checkpoint to ONNX format.

Usage:
    python export_onnx.py --checkpoint ./checkpoints/best.pt

The resulting enhancer_params.onnx should be placed next to LumaPhoto.exe.
The app detects it automatically and uses it instead of places365_mobilenet.onnx.

Model I/O contract:
    Input:  "input"  — float32 [1, 3, 224, 224] — ImageNet-normalised RGB
    Output: "params" — float32 [1, 15]           — enhancement parameters
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from model import PhotoEnhancerNet


def export(checkpoint: str, out_path: str, opset: int = 17):
    device = torch.device("cpu")
    model  = PhotoEnhancerNet(pretrained=False)

    state = torch.load(checkpoint, map_location=device)
    # Support both raw state_dict and full checkpoint dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version     = opset,
            input_names       = ["input"],
            output_names      = ["params"],
            dynamic_axes      = {"input": {0: "batch"}, "params": {0: "batch"}},
            do_constant_folding = True,
        )

    print(f"Exported → {out_path}")

    # Quick sanity check
    import onnxruntime as ort
    import numpy as np

    sess    = ort.InferenceSession(out_path)
    inp     = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out_ort = sess.run(None, {"input": inp})[0]
    print(f"ONNX output shape: {out_ort.shape}")
    print(f"Sample params: {dict(zip(['exposure','brilliance','highlights','shadows','contrast'], out_ort[0][:5].round(1)))}")
    print("Sanity check passed ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--output",     default=None,  help="Output .onnx path (default: next to checkpoint)")
    p.add_argument("--opset",      type=int, default=17)
    args = p.parse_args()

    out = args.output or str(Path(args.checkpoint).with_name("enhancer_params.onnx"))
    export(args.checkpoint, out, args.opset)
