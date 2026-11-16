#!/usr/bin/env python3
"""Adds SSIM to results/image_metrics.csv rows and draws plots + a visual comparison.
SSIM is implemented here (Wang et al. 2004, 7x7 uniform window, K1=0.01, K2=0.03, mean over
RGB channels) because scikit-image is not installed. Run after build/image_demo."""
import pathlib, csv, collections
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
res = root / "results"; plots = res / "plots"; plots.mkdir(exist_ok=True)

def ssim_channel(a, b, win=7):
    a = a.astype(np.float64); b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = uniform_filter(a, win), uniform_filter(b, win)
    saa = uniform_filter(a * a, win) - ma * ma
    sbb = uniform_filter(b * b, win) - mb * mb
    sab = uniform_filter(a * b, win) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))
    return s.mean()

def ssim(a, b):
    return float(np.mean([ssim_channel(a[..., c], b[..., c]) for c in range(3)]))

def load(p): return np.array(Image.open(p).convert("RGB"))

rows = list(csv.DictReader(open(res / "image_metrics.csv")))
for r in rows:
    r["ssim"] = ""
    if r["method"] == "conventional_int": r["ssim"] = "1"; continue
    d = res / "images" / r["pair"]
    # PPM written by C++ (first seed); load with PIL
    r["ssim"] = f"{ssim(load(d / 'exact.ppm'), load(d / f'{r['method']}_N{r['N']}.ppm')):.6f}"
with open(res / "image_metrics_ssim.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# Plot 1: MAE vs N (real image pair), log y
methods = ["random_independent", "same_reference", "sobol_std", "digital_shift", "adaptive_shift"]
for pair in ["real500_x_real768crop", "real500_x_checkerboard"]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in methods:
        rr = sorted([r for r in rows if r["pair"] == pair and r["method"] == m], key=lambda r: int(r["N"]))
        N = [int(r["N"]) for r in rr]
        ax[0].plot(N, [max(float(r["mae"]), 1e-3) for r in rr], marker="o", label=m)
        ax[1].plot(N, [float(r["ssim"]) for r in rr], marker="o", label=m)
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("stream length N"); ax[0].set_ylabel("MAE (0-255 levels)")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("stream length N"); ax[1].set_ylabel("SSIM vs exact product")
    ax[0].legend(fontsize=7); fig.suptitle(pair); fig.tight_layout(); fig.savefig(plots / f"quality_vs_N_{pair}.png", dpi=130); plt.close(fig)

# Plot 2: visual comparison at N=256 on the real pair
d = res / "images" / "real500_x_real768crop"
names = ["exact"] + [f"{m}_N256" for m in methods]
fig, axs = plt.subplots(2, 6, figsize=(18, 6.4))
for i, n in enumerate(names):
    img, ex = load(d / f"{n}.ppm"), load(d / "exact.ppm")
    axs[0, i].imshow(img); axs[0, i].set_title(n, fontsize=8); axs[0, i].axis("off")
    axs[1, i].imshow(np.abs(img.astype(int) - ex.astype(int)).mean(-1), cmap="magma", vmin=0, vmax=40)
    axs[1, i].set_title("|error| (0-40)", fontsize=8); axs[1, i].axis("off")
fig.tight_layout(); fig.savefig(plots / "visual_comparison_N256.png", dpi=110); plt.close(fig)

# Console summary: mean over the real-image pair + checkerboard at each N
print(f"{'method':20s} {'N':>5s} {'MAE':>7s} {'PSNR':>7s} {'SSIM':>8s}   (pair real500_x_real768crop)")
for r in rows:
    if r["pair"] == "real500_x_real768crop" and r["method"] != "conventional_int":
        print(f"{r['method']:20s} {r['N']:>5s} {float(r['mae']):7.3f} {float(r['psnr']):7.2f} {float(r['ssim']):8.5f}")
print("wrote", res / "image_metrics_ssim.csv", "and", plots)
