# 2D Sobol Stochastic Image Multiplier

Stochastic-computing image multiplication (C++17, CMake). Each 8-bit channel value becomes an N-bit
stream (bit k = 1 iff reference_k < value/255); two images are multiplied by AND-ing their streams and
counting ones: `round(255 * popcount(SA[a] & SB[b]) / N)`. The reference numbers come from a 2D Sobol
sequence (x for image A, y for image B), optionally scrambled (Gap 2 correlation-control stack).

## Build and run

```
cmake -S . -B build -G Ninja            # or omit -G on Linux/macOS
cmake --build build
ctest --test-dir build                  # test_sobol, test_correlation, test_avx2_kernel, test_image_path

python python/prepare_dataset.py        # dataset_collection/*.png -> results/ppm/*.ppm
./build/image_demo results/ppm results  # accuracy + scalar-vs-AVX2 timing  -> results/*.csv, results/images
./build/run_experiments results         # Gap 2 generator experiments        -> results/summary.csv
./build/run_extra results results/ppm   # rotation, operation-aware, selection-bias -> results/*.csv
python python/analyze_images.py         # SSIM, Delta-E 2000, plots          -> results/image_metrics_full.csv

# batch mode over two folders (pairs the i-th file of each, centre-crops to the common size)
python python/prepare_dataset.py --dir <folderA> results/a && python python/prepare_dataset.py --dir <folderB> results/b
./build/image_demo --batch results/a results/b results/batch
```

Requirements: CMake >= 3.18, a C++17 compiler (tested: MSYS2 UCRT64 GCC 16.1.0 + Ninja on Windows 11);
Python with `numpy pillow scikit-image matplotlib` for the analysis scripts. AVX2 is compiled in only if the
compiler accepts `-mavx2` on an x86 target and is **re-checked at run time**; otherwise a scalar fallback
runs (every program prints which kernel it used). OpenCV is optional (`find_package(OpenCV QUIET)`).

## What is implemented

| Area | Where |
|---|---|
| 2D Sobol generator (Joe-Kuo dims 1..10, Gray-code XOR update) | `include/sobol_generator.hpp`, `src/sobol_generator.cpp` |
| Correlation control: digital shift / linear-matrix / Owen scrambles, metrics, adaptive seed selection | `scrambling.*`, `sc_metrics.*`, `seed_selector.*`, `src/experiments.cpp` |
| Scalar + AVX2 AND/popcount kernels (pshufb nibble-LUT popcount, scalar tail, runtime dispatch) | `include/avx2_kernels.hpp`, `src/avx2_kernels.cpp` |
| Image I/O, stream banks, stochastic multiplier | `include/sc_image.hpp`, `src/sc_image.cpp` |
| Image demo with CIs, scalar-vs-AVX2 check + timing, batch mode, optional OpenCV baseline | `src/image_demo.cpp` |
| Rotation, operation-aware correlation, selection-bias experiments | `src/extra_experiments.cpp` |
| SSIM / Delta-E 2000 / plots | `python/analyze_images.py` |
| Hardware mapping (design sketch + labelled estimates) | `docs/hardware_mapping.md` |

## Measured results

All numbers below were produced by the programs above on one machine: AMD Ryzen 7 6800H, Windows 11,
GCC 16.1.0, `-O3` Release, AVX2 compiled and supported. Seeds are fixed (deterministic). `+-` is a 95 % CI
of the mean over seeds (`1.96*std/sqrt(trials)`); rows with a single deterministic trial have no CI.

### Image quality vs. exact product (real 500x500 x real 768x512-crop; MAE in 0-255 levels, mean over seeds)

| method | N | MAE | PSNR dB | SSIM | mean dE2000 |
|---|---|---|---|---|---|
| random independent | 128 / 256 / 1024 | 5.52 / 4.39 / 2.15 | 31.1 / 33.6 / 39.4 | 0.949 / 0.964 / 0.994 | 2.73 / 2.38 / 1.00 |
| same reference (fully correlated) | 128 / 256 / 1024 | 22.8 / 22.1 / 21.4 | 19.1 / 19.3 / 19.5 | 0.806 / 0.811 / 0.817 | 10.8 / 10.6 / 10.5 |
| Sobol dims (1,2) | 128 / 256 / 1024 | 1.65 / 0.72 / 0.17 | 42.0 / 48.3 / 55.9 | 0.989 / 0.996 / 0.999 | 0.79 / 0.40 / 0.13 |
| digital-shift Sobol | 128 / 256 / 1024 | 0.98 / 0.51 / 0.15 | 45.9 / 50.4 / 56.4 | 0.992 / 0.997 / 0.999 | 0.58 / 0.38 / 0.11 |
| adaptive (best of 32 shifts) | 128 / 256 / 1024 | 0.93 / 0.70 / 0.145 | 46.2 / 48.5 / 56.5 | 0.992 / 0.996 / 0.999 | 0.58 / 0.40 / 0.11 |

MAE/PSNR are means over seeds; SSIM and Delta-E are computed (scikit-image) on the saved first-seed image,
so they are single-sample. Per-row CIs are in `results/image_metrics.csv`. Sobol streams are roughly
6-12x more accurate than random streams of the same length; the fully-correlated control collapses
multiplication to `min(a,b)` and is far worse. Delta-E 2000 below ~1 is reached by Sobol/shifted Sobol at
N = 1024 and by scrambled Sobol at N >= 256 on this image pair. **No claim is made for the original
targets (SSIM >= 0.99 everywhere, Delta-E < 0.85 everywhere):** SSIM >= 0.99 holds for Sobol-type streams on
this pair at N >= 128 but not for random streams at N <= 256, and it was only tested on 2 real images.

### AVX2 vs scalar (`results/timing.csv`; 500x500x3 channels, 3 warm-up + 15 timed reps)

AND+popcount stage, mean ms (std in the CSV): N=128: scalar 1.98 / AVX2 2.51; N=256: 2.64 / 2.73;
N=1024: 6.09 / 4.32. Scalar and AVX2 outputs were **byte-identical** for every pair, method and length,
and `test_avx2_kernel` checks all 65 536 level pairs at N = 128/256/1024. **AVX2 only wins at N = 1024
(~29 % faster stage time); at N <= 256 it is slower or equal**, because the hardware `popcnt` on 64-bit
words is already cheap and one 256-bit vector (N = 256) or none (N = 128: tail only) does not amortise the
horizontal sum. Stream encoding/packing (0.05-0.45 ms) and Sobol generation (< 0.01 ms) are negligible next
to the AND+popcount stage. One machine, one compiler, one run: treat the exact ratios as indicative.

### Rotation de-correlation (`results/rotation.csv`, N = 256, test grid 0.1..0.9)

Circularly rotating stream B before the AND: for the **fully-correlated same-reference** streams a rotation
of 3 cuts mean |SCC| from 1.00 to 0.23 and multiplication MAE from 103e-3 to 17e-3, but offsets 16/64/128
restore SCC ~ 1 (periodic structure). Random streams are unaffected, and for **Sobol (1,2) rotation gave no
benefit**: MAE stays 2.2-2.9e-3 for all offsets (no systematic trend or harm was seen), and the unrotated
Sobol streams are already ~6x better than the best rotated same-reference case (17e-3). Sobol rows are
single deterministic trials (no CI). Conclusion: rotation is a cheap fix for a shared-reference design, not
an improvement on 2D Sobol.

### Operation-aware correlation (`results/ops.csv`, N = 256)

Mean |error| x1e-3 over the 9x9 grid: same-reference gives multiply 103.6 (wrong), min 1.7, max 1.7, |a-b| 1.2
(right: those ops need SCC +1); Sobol (1,2) gives multiply 2.95 (right: needs SCC 0) but min 99, max 103,
|a-b| 202 (wrong). Random streams are wrong for the +1 ops and noisy (19e-3) for multiply. So the correlation
target must follow the operation; "same reference" is the correct choice for min/max/|a-b| only.

### Is adaptive seed selection just selection bias? (`results/selection_bias.csv`, 40 independent pools of 32 seeds)

Gain = (average candidate error) - (selected candidate error), selected on the calibration grid, scored elsewhere:

| N | evaluation | gain (+-CI95) |
|---|---|---|
| 64 | calibration grid (in-sample) | -0.05 +- 0.31 e-3 |
| 64 | held-out grid | -0.28 +- 0.31 e-3 |
| 64 | image pixels (MAE in levels) | -0.106 +- 0.060 |
| 256 | calibration grid (in-sample) | +0.14 +- 0.06 e-3 |
| 256 | held-out grid | +0.14 +- 0.10 e-3 |
| 256 | image pixels (MAE in levels) | -0.017 +- 0.018 |

At N = 64 selection does not help; at N = 256 there is a small positive gain on grid metrics that survives
on a disjoint grid but is **not detectable on real image pixels** (CI contains 0). The image-level
"adaptive" gains in the table above are therefore within seed-to-seed noise and should not be presented as
an improvement over a plain digital shift.

## What could not be run / known limits

* **OpenCV baseline**: code is written (`SC_HAVE_OPENCV`) but OpenCV is not installed here, so it was never
  compiled or timed. The conventional baseline reported is the scalar integer multiply (~0.6 ms/image).
* **Kodak / McMaster batch runs**: batch mode works (tested on the repo's 2 real x 4 synthetic images) but
  those datasets are not in the repo and were not downloaded.
* **Linux/WSL, other CPUs, MSVC/Clang**: only the Windows/MinGW GCC build above was run.
* **Hardware**: `docs/hardware_mapping.md` is an analytic estimate; nothing was synthesised.
* Dataset is small (2 real + 4 synthetic images); quality conclusions are not generalised beyond it.
