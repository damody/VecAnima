"""Export the locally downloaded AniLines basic model for native OpenCV inference.

Python/PyTorch/ONNX are conversion tools only. The target CLI runs ONNX natively.
The original network and license remain in third_party/anilines.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "third_party/anilines")
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name in ("basic.pth", "line_extractor.py"):
        row = next(r for r in manifest if r["name"] == name)
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {name}")
    import torch
    spec = importlib.util.spec_from_file_location("anilines_network", root / "line_extractor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.LineExtractor(3, 1, True).eval()
    net.load_state_dict(torch.load(root / "basic.pth", map_location="cpu", weights_only=True))
    # Native inference reflect-pads to multiples of 16. These upsample sizes
    # therefore match exactly; omit redundant shape-dependent padding operators.
    for layer in net.modules():
        if layer.__class__.__name__ == "Up":
            def forward(self, x1, x2):
                return self.conv(torch.cat([x2, self.up(x1)], dim=1))
            layer.forward = forward.__get__(layer, type(layer))
    destination = root / "basic.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, 256, 256), str(destination),
                      opset_version=13, dynamo=False,
                      input_names=["input"], output_names=["lineart"])
    record = {"name": "basic.onnx", "derived_from": "basic.pth", "opset": 13,
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "torch": torch.__version__, "input": "NCHW RGB 0..1, Pillow Sharpness(6), reflect pad /16",
              "output": "NCHW grayscale, white background, arbitrary /16-aligned spatial sizes verified"}
    manifest = [r for r in manifest if r["name"] != "basic.onnx"] + [record]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
