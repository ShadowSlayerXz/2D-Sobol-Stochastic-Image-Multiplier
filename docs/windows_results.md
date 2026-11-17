# Windows Results (AMD Ryzen 7 6800H, Windows 11)

Consolidated record of everything measured on the Windows machine. Raw data is in `results/`
(`summary.csv`, `image_metrics.csv`, `timing.csv`, `rotation.csv`, `ops.csv`, `selection_bias.csv`,
and the logs `exp_log.txt`, `demo_log.txt`, `extra_log.txt`).

**Environment:** AMD Ryzen 7 6800H (8 cores / 16 threads), Windows 11, MSYS2 UCRT64 GCC 16.1.0 + Ninja,
`-O3 -DNDEBUG` Release, AVX2 compiled and supported, OpenCV 4.13.0 (MSYS2) for the baseline.
Seeds are fixed (deterministic). `+-` is a 95% CI over seeds; single-trial deterministic rows have no CI.
All timings are one machine, one compiler, one run: indicative only.

## Image quality vs. exact product (real 500x500 x real 768x512-crop; MAE in 0-255 levels, mean over seeds)

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

## AVX2 vs scalar (`results/timing.csv`; 500x500x3 channels, 3 warm-up + 15 timed reps)

AND+popcount stage, mean ms (std in the CSV), latest run: N=128: scalar 2.01 / AVX2 2.43; N=256: 2.57 / 2.83;
N=1024: 6.06 / 4.27 (three separate runs gave the same pattern). Scalar and AVX2 outputs were **byte-identical** for every pair, method and length,
and `test_avx2_kernel` checks all 65 536 level pairs at N = 128/256/1024. **AVX2 only wins at N = 1024
(~29 % faster stage time); at N <= 256 it is slower or equal**, because the hardware `popcnt` on 64-bit
words is already cheap and one 256-bit vector (N = 256) or none (N = 128: tail only) does not amortise the
horizontal sum. Stream encoding/packing (0.05-0.45 ms) and Sobol generation (< 0.01 ms) are negligible next
to the AND+popcount stage. One machine, one compiler, one run: treat the exact ratios as indicative.

## CPU optimisation (bit-identical output) and the OpenCV comparison

Optimisations, all producing byte-for-byte the same image as the direct AND+popcount path (checked by
`test_avx2_kernel` and on every `image_demo` run):

1. **Product lookup table** (`buildProductLutInto` / `lookupProduct`): the result depends only on the (a, b) level
   pair, so compute all 65 536 results once with the AND+popcount kernel, then do one table read per pixel.
2. **Fast stream encoding** (`buildBank`): put each bit at the first level where it turns on, then prefix-OR across
   levels: O(N + 256*W) instead of O(256*N). Encoding no longer shows up in the profile.
3. **Count->level table** instead of an integer divide per table entry; **tuned kernel** (`tunedKernel`: AVX2 popcount
   only when >= 8 words, i.e. N >= 512; hardware popcnt otherwise); **reused buffers** (no per-call allocation).
4. **`__restrict` in the lookup loop**: the `uint8_t*` pointers could alias, so the compiler reloaded constantly; adding
   `restrict` cut the single-thread lookup pass from 1.08 ms to 0.35 ms (found by profiling).
5. **Thread pool** (`include/thread_pool.hpp`) splitting table build and lookup over 4/8/16 threads.

Full pipeline (Sobol generation + encoding + table + lookup), 500x500x3 image, mean ms (std in `timing.csv`):

| N | direct AND+popcount, scalar | direct, AVX2 | LUT, 1 thread | LUT, 4 thr | LUT, 8 thr | LUT, 16 thr |
|---|---|---|---|---|---|---|
| 128 | 2.2 | 2.6 | 0.52 | 0.14 | 0.105 | 0.125 |
| 256 | 2.7 | 3.0 | 0.58 | 0.15 | 0.105 | 0.129 |
| 1024 | 6.5 | 4.9 | 0.68 | 0.19 | 0.134 | 0.166 |

(direct columns from an earlier run of the same program; they were ~2.7 / ~6.5 ms in every run.) Best configuration
is 8 threads (the 6800H has 8 physical cores / 16 logical): **roughly 20-50x faster than the direct path**, and
output remains identical.

**OpenCV `cv::multiply`** (OpenCV 4.13.0 from MSYS2, scale 1/255, same image): **0.134-0.137 ms**, exact result. It gives
the same time with 1, 8 or the default 16 threads, i.e. it runs effectively single-threaded on this size. So:

* single thread: optimised stochastic path 0.52-0.68 ms vs OpenCV 0.134 ms -> OpenCV is ~4-5x faster;
* 8 threads: optimised stochastic path 0.105-0.134 ms vs single-thread OpenCV 0.134 ms -> about equal.

**Read this comparison with care.** The stochastic path wins nothing by being stochastic: the speed comes from the lookup
table and from threading. A binary multiply could use the very same table and the same threads, and cv::multiply split
over 8 threads by hand would almost certainly be faster still (not measured). The honest conclusion is that the
software version can be made *competitive*, not *better*; its case remains the hardware area/energy argument in
`docs/hardware_mapping.md`. All timings: one machine, one compiler, one run.

## Rotation de-correlation (`results/rotation.csv`, N = 256, test grid 0.1..0.9)

Circularly rotating stream B before the AND: for the **fully-correlated same-reference** streams a rotation
of 3 cuts mean |SCC| from 1.00 to 0.23 and multiplication MAE from 103e-3 to 17e-3, but offsets 16/64/128
restore SCC ~ 1 (periodic structure). Random streams are unaffected, and for **Sobol (1,2) rotation gave no
benefit**: MAE stays 2.2-2.9e-3 for all offsets (no systematic trend or harm was seen), and the unrotated
Sobol streams are already ~6x better than the best rotated same-reference case (17e-3). Sobol rows are
single deterministic trials (no CI). Conclusion: rotation is a cheap fix for a shared-reference design, not
an improvement on 2D Sobol.

## Operation-aware correlation (`results/ops.csv`, N = 256)

Mean |error| x1e-3 over the 9x9 grid: same-reference gives multiply 103.6 (wrong), min 1.7, max 1.7, |a-b| 1.2
(right: those ops need SCC +1); Sobol (1,2) gives multiply 2.95 (right: needs SCC 0) but min 99, max 103,
|a-b| 202 (wrong). Random streams are wrong for the +1 ops and noisy (19e-3) for multiply. So the correlation
target must follow the operation; "same reference" is the correct choice for min/max/|a-b| only.

## Is adaptive seed selection just selection bias? (`results/selection_bias.csv`, 40 independent pools of 32 seeds)

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

## Scalar numeric experiments (`results/exp_log.txt`, `results/summary.csv`)

Stage 5, 81 (pA,pB) pairs on the 0.1..0.9 grid, N = 256 (MAE x1e-3, lower is better):

| config | mean abs SCC | MAE (x1e-3) | MSE (x1e-6) |
|---|---|---|---|
| random independent | 0.1280 | 18.222 | 579.9 |
| same reference (dim1,dim1) | 1.0000 | 103.588 | 14398.4 |
| Sobol dims (1,2) | 0.0381 | 2.951 | 13.3 |
| digital shift | 0.0336 | 2.317 | 8.4 |
| linear matrix scramble | 0.0341 | 2.295 | 8.3 |
| Owen scramble | 0.0340 | 2.273 | 8.1 |
| adaptive seed (digital shift) | 0.0227 | 2.078 | 6.5 |
| adaptive seed (linear matrix) | 0.0340 | 2.286 | 8.3 |
| adaptive seed (Owen) | 0.0355 | 2.290 | 8.1 |
| adaptive joint dims + shift | 0.0260 | 2.117 | 6.9 |

Stage 2 (45 direction-number pairs from dims 1..10): dims (1,2) are near the top at every N, but the best
pair varies with N (e.g. best (1,10) at N=8, (5,8) at N=16, (1,2) at N=32, (2,5) at N=64, (1,7) at N=128,
(1,4) at N=256) and some pairs are far worse (e.g. (1,5) at N=8, (8,9) at N=16, (2,8) at N=32/64).
Full ranking per N is in `results/exp_log.txt`.

## OpenCV `cv::multiply` vs the Sobol stochastic multiplier (all results)

Same 500x500x3 image pair (real500 x real768 crop), same machine, 3 warm-up + 15 timed reps (45 for the threaded LUT rows). Source: `results/timing.csv`, `results/demo_log.txt`. Ratio = Sobol time / OpenCV single-thread time (0.134 ms); **<1 means Sobol is faster**.

### OpenCV baseline (exact result, MAE vs exact = 0)

| configuration | mean ms | min ms | std ms |
|---|---|---|---|
| default (16 threads reported) | 0.137 | 0.135 | 0.004 |
| 1 thread | 0.134 | 0.133 | 0.002 |
| 8 threads | 0.135 | 0.134 | 0.001 |

OpenCV is flat across thread counts (effectively single-threaded at this size).

### Sobol stochastic multiplier, full pipeline (Sobol generation + encoding + multiply), mean ms and ratio to OpenCV

| N | direct scalar | direct AVX2 | LUT scalar | LUT AVX2 | fast 1 thr | fast 4 thr | fast 8 thr | fast 16 thr |
|---|---|---|---|---|---|---|---|---|
| 128 | 2.076 (15.5x) | 2.532 (18.9x) | 0.943 (7.0x) | 1.017 (7.6x) | 0.515 (3.8x) | 0.142 (1.1x) | 0.105 (0.8x) | 0.125 (0.9x) |
| 256 | 2.635 (19.7x) | 2.948 (22.0x) | 1.018 (7.6x) | 1.044 (7.8x) | 0.579 (4.3x) | 0.151 (1.1x) | 0.105 (0.8x) | 0.129 (1.0x) |
| 1024 | 6.318 (47.1x) | 4.310 (32.1x) | 1.350 (10.1x) | 1.138 (8.5x) | 0.677 (5.0x) | 0.194 (1.4x) | 0.134 (1.0x) | 0.166 (1.2x) |

### Accuracy next to the timing (N = 1024, real500 x real768 crop)

| method | MAE (0-255 levels) | PSNR dB | time |
|---|---|---|---|
| OpenCV `cv::multiply` (exact, rounded) | 0 | inf | 0.134 ms (1 thr) |
| Sobol (1,2), fast 8 thr | 0.169 | 55.9 | 0.134 ms |
| digital-shift Sobol, fast 8 thr | 0.149 | 56.4 | 0.134 ms |
| random independent, fast 8 thr | 2.150 | 39.4 | 0.134 ms |

(The fast path costs the same regardless of stream type, since the table and lookup dominate; stream choice only changes accuracy.)

### Findings

* **Single thread:** optimised Sobol 0.52 / 0.58 / 0.68 ms at N = 128 / 256 / 1024 vs OpenCV 0.134 ms, so OpenCV is about 3.8-5.0x faster.
* **8 threads:** Sobol 0.105 / 0.105 / 0.134 ms vs single-thread OpenCV 0.134 ms: 0.8x, 0.8x and 1.0x, i.e. equal or slightly faster at N <= 256 and equal at N = 1024. OpenCV got no benefit from extra threads, so the comparison is thread-count-asymmetric.
* **16 threads is slower than 8** (0.125-0.166 ms): the 6800H has 8 physical cores, so SMT adds overhead.
* **Versus the direct (non-LUT) stochastic path** the optimisations give roughly 20-50x (e.g. N=1024: 6.3 ms scalar -> 0.134 ms).
* **Caveat:** the speed comes from the product lookup table and threading, not from stochastic arithmetic. A binary multiply could use the same table and threads, and OpenCV split over 8 threads by hand was not measured and would likely win. The stochastic version is competitive, not better; its case is the hardware area/energy argument in `docs/hardware_mapping.md`.
* One machine, one compiler, one run each; ratios are indicative.
