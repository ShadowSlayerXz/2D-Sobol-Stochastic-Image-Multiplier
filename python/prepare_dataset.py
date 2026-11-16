#!/usr/bin/env python3
"""Convert dataset_collection/*.png to binary PPM (P6) for the C++ demo (no OpenCV needed).
The 768x512 image is centre-cropped to 500x500 so every pair has matching dimensions."""
import sys, pathlib
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results" / "ppm"
out.mkdir(parents=True, exist_ok=True)
for p in sorted((root / "dataset_collection").rglob("*.png")):
    im = Image.open(p).convert("RGB")
    name = p.stem
    if im.size != (500, 500):
        w, h = im.size
        l, t = (w - 500) // 2, (h - 500) // 2
        im = im.crop((l, t, l + 500, t + 500))
        name += "_crop500"
    im.save(out / f"{name}.ppm")
    print("wrote", out / f"{name}.ppm", im.size)
