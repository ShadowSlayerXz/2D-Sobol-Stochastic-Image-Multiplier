#!/usr/bin/env python3
"""Post-process results/ written by build/image_demo (and run_extra).

Adds, for the SAVED image of each row (first seed), metrics computed independently of the C++ code:
  MAE / MSE / PSNR (numpy, cross-checks the C++ per-image numbers for single-trial rows),
  SSIM        (scikit-image structural_similarity, 7x7 uniform window, data_range 255, RGB channels),
  Delta-E 2000 (scikit-image deltaE_ciede2000 on sRGB->CIELAB, D65): mean and 95th percentile.
Writes results/image_metrics_full.csv plus plots in results/plots/.

Usage: python python/analyze_images.py [results_dir]
"""
import csv, pathlib, sys
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parent.parent
res = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results"
plots = res / "plots"; plots.mkdir(exist_ok=True)
METHODS = ["random_independent", "same_reference", "sobol_std", "digital_shift", "adaptive_shift"]


def load(p): return np.array(Image.open(p).convert("RGB"))


def image_metrics(got, ref):
    d = got.astype(np.float64) - ref.astype(np.float64)
    mse = float((d ** 2).mean())
    ssim = structural_similarity(ref, got, channel_axis=2, data_range=255, win_size=7)
    de = deltaE_ciede2000(rgb2lab(ref), rgb2lab(got))
    return dict(img_mae=float(np.abs(d).mean()), img_mse=mse,
                img_psnr=99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse),
                ssim=float(ssim), de00_mean=float(de.mean()), de00_p95=float(np.percentile(de, 95)))


rows = list(csv.DictReader(open(res / "image_metrics.csv")))
for r in rows:
    if r["method"] == "conventional_int":
        r.update(img_mae=0, img_mse=0, img_psnr=99, ssim=1, de00_mean=0, de00_p95=0); continue
    d = res / "images" / r["pair"]
    r.update(image_metrics(load(d / f"{r['method']}_N{r['N']}.ppm"), load(d / "exact.ppm")))
fields = list(rows[0].keys())
with open(res / "image_metrics_full.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

# ---- plots: quality vs N for the real pair and the checkerboard pair ---------------------
pairs = [p for p in ["real500_x_real768crop", "real500_x_checkerboard"] if any(r["pair"] == p for r in rows)]
for pair in pairs:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for m in METHODS:
        rr = sorted([r for r in rows if r["pair"] == pair and r["method"] == m], key=lambda r: int(r["N"]))
        N = [int(r["N"]) for r in rr]
        ax[0].plot(N, [max(float(r["mae"]), 1e-3) for r in rr], marker="o", label=m)
        ax[1].plot(N, [r["ssim"] for r in rr], marker="o", label=m)
        ax[2].plot(N, [r["de00_mean"] for r in rr], marker="o", label=m)
    for a, yl in zip(ax, ["MAE (0-255 levels)", "SSIM vs exact product", "mean Delta-E 2000"]):
        a.set_xscale("log", base=2); a.set_xlabel("stream length N"); a.set_ylabel(yl)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7); fig.suptitle(pair); fig.tight_layout()
    fig.savefig(plots / f"quality_vs_N_{pair}.png", dpi=130); plt.close(fig)

if pairs:
    d = res / "images" / pairs[0]
    names = ["exact"] + [f"{m}_N256" for m in METHODS]
    fig, axs = plt.subplots(2, len(names), figsize=(18, 6.4)); ex = load(d / "exact.ppm")
    for i, n in enumerate(names):
        img = load(d / f"{n}.ppm")
        axs[0, i].imshow(img); axs[0, i].set_title(n, fontsize=8); axs[0, i].axis("off")
        axs[1, i].imshow(np.abs(img.astype(int) - ex.astype(int)).mean(-1), cmap="magma", vmin=0, vmax=40)
        axs[1, i].set_title("|error| (0-40)", fontsize=8); axs[1, i].axis("off")
    fig.tight_layout(); fig.savefig(plots / "visual_comparison_N256.png", dpi=110); plt.close(fig)

# ---- timing plot: scalar vs AVX2 AND+popcount ---------------------------------------------
tpath = res / "timing.csv"
if tpath.exists():
    trows = list(csv.DictReader(open(tpath)))
    fig, ax = plt.subplots(figsize=(6, 4)); width = 0.35
    ns = sorted({int(r["N"]) for r in trows if r["stage"] == "and_popcount"})
    for k, path in enumerate(["scalar", "avx2"]):
        vals = [next(float(r["mean_ms"]) for r in trows if r["stage"] == "and_popcount" and int(r["N"]) == n and r["path_requested"] == path) for n in ns]
        errs = [next(float(r["std_ms"]) for r in trows if r["stage"] == "and_popcount" and int(r["N"]) == n and r["path_requested"] == path) for n in ns]
        ax.bar(np.arange(len(ns)) + k * width, vals, width, yerr=errs, label=path)
    ax.set_xticks(np.arange(len(ns)) + width / 2); ax.set_xticklabels(ns); ax.set_xlabel("stream length N")
    ax.set_ylabel("AND+popcount, 250k px x 3 ch (ms)"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "timing_and_popcount.png", dpi=130); plt.close(fig)

# ---- rotation plot --------------------------------------------------------------------------
rpath = res / "rotation.csv"
if rpath.exists():
    rr = [r for r in csv.DictReader(open(rpath)) if r["N"] == "256"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for src in ["random_independent", "same_reference", "sobol_std"]:
        s = [r for r in rr if r["source"] == src]
        x = [int(r["offset"]) for r in s]
        ax[0].plot(range(len(x)), [float(r["mean_abs_scc"]) for r in s], marker="o", label=src)
        ax[1].plot(range(len(x)), [float(r["mae_mult"]) * 1e3 for r in s], marker="o", label=src)
        ax[0].set_xticks(range(len(x))); ax[0].set_xticklabels(x); ax[1].set_xticks(range(len(x))); ax[1].set_xticklabels(x)
    ax[0].set_ylabel("mean |SCC|"); ax[1].set_ylabel("multiplication MAE (x1e-3)"); ax[1].set_yscale("log")
    for a in ax: a.set_xlabel("rotation offset of stream B (N=256)")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "rotation_N256.png", dpi=130); plt.close(fig)

# ---- console summary -------------------------------------------------------------------------
if pairs:
    print(f"{'method':20s} {'N':>5s} {'MAE':>7s} {'PSNR':>7s} {'SSIM':>8s} {'dE00':>7s}   (pair {pairs[0]}; MAE/PSNR = mean over seeds, SSIM/dE00 = saved first-seed image)")
    for r in rows:
        if r["pair"] == pairs[0] and r["method"] != "conventional_int":
            print(f"{r['method']:20s} {r['N']:>5s} {float(r['mae']):7.3f} {float(r['psnr']):7.2f} {r['ssim']:8.5f} {r['de00_mean']:7.3f}")
print("wrote", res / "image_metrics_full.csv", "and", plots)
