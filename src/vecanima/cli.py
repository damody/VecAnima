import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .media import binary, probe, run
from .pipeline import analyze, build
from .vectorize import Settings


def main():
    parser = argparse.ArgumentParser(description="VecAnima animation vectorization experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check project-local dependencies")
    p = sub.add_parser("probe", help="Inspect source media")
    p.add_argument("source", type=Path)
    p = sub.add_parser("analyze", help="Create source metadata and timestamped thumbnails")
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=12)
    p = sub.add_parser("vectorize", help="Run the flat-color and dark-mask baseline")
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--colors", type=int, default=24)
    p.add_argument("--epsilon", type=float, default=.65)
    p.add_argument("--stroke-threshold", type=int, default=65)
    p.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            import scipy
            import skimage
            result = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__,
                      "scipy": scipy.__version__, "scikit_image": skimage.__version__,
                      "sift": hasattr(cv2, "SIFT_create"), "l0_smoothing": hasattr(cv2.ximgproc, "l0Smooth")}
            for n in ("ffmpeg", "ffprobe"):
                result[n] = {"path": binary(n), "version": run(n, ["-version"]).stdout.splitlines()[0]}
        else:
            if not args.source.is_file():
                raise ValueError(f"Source does not exist: {args.source}")
            if args.command == "probe":
                result = probe(args.source)
            elif args.command == "analyze":
                if not 1 <= args.count <= 100:
                    raise ValueError("count must be between 1 and 100")
                result = analyze(args.source, args.out, args.count)
            else:
                if not 0 <= args.start or not 0 < args.duration or args.width < 32 or args.width % 2:
                    raise ValueError("start >= 0, duration > 0, and an even width >= 32 are required")
                if not 2 <= args.colors <= 256 or not 0 <= args.epsilon <= 10 or not 0 <= args.stroke_threshold <= 255:
                    raise ValueError("colors: 2..256, epsilon: 0..10, stroke-threshold: 0..255")
                result = build(args.source, args.out, args.start, args.duration, args.width,
                               Settings(args.colors, args.epsilon, args.stroke_threshold), not args.no_preview)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
