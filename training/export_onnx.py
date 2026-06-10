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
import json
from pathlib import Path

import torch
import torch.onnx

from model import PhotoEnhancerNet


def export(checkpoint: str, out_path: str, opset: int = 17, expert: str = "c"):
    device = torch.device("cpu")
    model  = PhotoEnhancerNet(pretrained=False)

    state = torch.load(checkpoint, map_location=device)
    # Support raw state_dict, best.pt EMA state_dict, and full last.pt checkpoints.
    # Prefer EMA weights when present; they are what the app should ship.
    if isinstance(state, dict) and "ema" in state:
        state = state["ema"]
    elif isinstance(state, dict) and "model" in state:
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

    manifest = {
        "training_profile": "fivek-v2",
        "dataset": "MIT-Adobe FiveK",
        "expert": expert,
        "param_count": 15,
        "input": "float32 [1, 3, 224, 224] ImageNet-normalised RGB",
        "output": "float32 [1, 15] LumaPhoto slider parameters",
    }
    manifest_path = str(Path(out_path).with_suffix(".json"))
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest → {manifest_path}")

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
    p.add_argument("--output",     default=None,  help="Output .onnx path (default: fivek_expert_<expert>.onnx next to checkpoint)")
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--expert",     default="c", help="FiveK expert used for this model: c/a/e")
    args = p.parse_args()

    out = args.output or str(Path(args.checkpoint).with_name(f"fivek_expert_{args.expert.lower()}.onnx"))
    export(args.checkpoint, out, args.opset, args.expert)
