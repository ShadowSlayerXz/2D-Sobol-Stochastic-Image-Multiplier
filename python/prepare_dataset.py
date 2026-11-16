#!/usr/bin/env python3
"""Convert images to binary PPM (P6) for the C++ demo (no OpenCV needed).

Default:  python python/prepare_dataset.py [out_dir]
    converts dataset_collection/**/*.png; non-500x500 images are centre-cropped to 500x500
    so every pair in the fixed demo has matching dimensions.
Batch:    python python/prepare_dataset.py --dir <image_dir> <out_dir>
    converts every png/jpg/bmp/tif in <image_dir> at its ORIGINAL size (use with
    `image_demo --batch dirA dirB`, which pairs files and centre-crops each pair to a common size;
    e.g. Kodak or McMaster downloaded by you).
"""
import sys, pathlib
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
if len(sys.argv) > 1 and sys.argv[1] == "--dir":
    src, out = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            im = Image.open(p).convert("RGB"); im.save(out / f"{p.stem}.ppm"); print("wrote", out / f"{p.stem}.ppm", im.size)
    sys.exit(0)

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
