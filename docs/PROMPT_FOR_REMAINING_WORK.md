# Prompt to finish the project (paste into Claude Code on an x86 machine / WSL2 Ubuntu)

I'm working on a university COA project: 2D Sobol stochastic image multiplication (C++17, CMake).
Repo: https://github.com/ShadowSlayerXz/2D-Sobol-Stochastic-Image-Multiplier
My professor requires an AVX2 implementation, so this must be built and tested on x86-64
(WSL2 Ubuntu 22.04 with GCC 11+, or Linux). Check `grep avx2 /proc/cpuinfo` first.

## Already done (read the code before changing anything)
- `include/sobol_generator.hpp`, `src/sobol_generator.cpp`: 2D Sobol generator (Joe-Kuo dims 1..10, `nextRaw()`).
- `scrambling.*`, `sc_metrics.*`, `seed_selector.*`, `src/experiments.cpp`: Gap 2 correlation-reduction stack.
- `src/image_demo.cpp`: scalar RGB multiplier. Per channel: `popcount(SA[a] & SB[b])`,
  decoded as `round(255*count/N)`, where SA/SB are 256 packed N-bit streams (one per pixel level).
  Compares random / same-reference / Sobol / digital-shift / adaptive at N = 128, 256, 1024.
- `python/prepare_dataset.py`, `python/analyze_images.py` (SSIM, plots), `tests/*` (3 CTest programs).
- Dataset: `dataset_collection/` (2 real, 4 synthetic images).
- Build/run: `cmake -S . -B build && cmake --build build && ctest --test-dir build`, then
  `python3 python/prepare_dataset.py && ./build/image_demo results/ppm results && python3 python/analyze_images.py`.

## Tasks (do them in this order, run tests after each, commit after each)
1. **AVX2 kernel** (`include/avx2_kernels.hpp`, `src/avx2_kernels.cpp`): pack 256 stochastic bits into
   `__m256i`; AND with `_mm256_and_si256`; popcount (use `_mm_popcnt_u64` on the four 64-bit lanes, or a
   pshufb nibble-LUT popcount); keep a scalar fallback and report at runtime which path ran.
   Compile AVX2 only where supported (`-mavx2 -mpopcnt` via CMake `check_cxx_compiler_flag`, `#ifdef __AVX2__`).
2. **`tests/test_avx2_kernel.cpp`**: scalar and AVX2 must give bit-identical counts for N = 128, 256, 1024
   over all 256x256 level pairs; register with CTest.
3. **Hook into `image_demo`**: add a scalar-vs-AVX2 path, verify identical output images, and time them
   separately (Sobol generation, encoding/packing, AND+popcount, full pipeline) with warm-up, >= 10 repetitions,
   mean/min/max/std. Record CPU model, compiler, flags, OS and which path ran into the CSV.
4. **Circular-rotation de-correlation**: rotate stream B by offsets (0, 1, 3, 7, 16, 64, ...) before the AND.
   Measure SCC and multiplication error per offset for random, same-reference and Sobol streams.
   Report honestly if rotation hurts true 2D Sobol (it can break the net property).
5. **Metrics**: add Delta-E 2000 (own implementation or `colour-science`), MAE/MSE/PSNR/SSIM
   (`scikit-image` if installable). Add an optional OpenCV baseline (`find_package(OpenCV QUIET)`,
   `cv::multiply` with scale 1/255) and time it against the stochastic path.
6. **Statistics**: 95% confidence intervals (mean +- 1.96*std/sqrt(trials)) in `experiments.cpp` and
   `image_demo.cpp`; check that adaptive seed selection gains are not selection bias (select on one grid,
   score on a disjoint one, and on image pixel-value distributions).
7. **Operation-aware correlation**: AND needs SCC 0; min (AND) / max (OR) need SCC +1; |a-b| (XOR) needs +1.
   Show that the same-reference control is correct for those ops and wrong for multiplication.
8. **Batch mode**: let `image_demo` run over directories (e.g. Kodak, McMaster) pairing images, cropping to a
   common size.
9. **Hardware mapping** (plan section 15/16): write a doc with registers (SOBOL_INDEX, SOBOL_X/Y, DIR ROM,
   BIT_COUNTER, ONES_COUNT), the FSM, comparators, AND datapath, popcount/accumulator, and a gate-count
   estimate vs an 8x8 array multiplier. Label estimates as estimates, not measurements.
10. Update README/docs with measured results only. Do not claim targets (SSIM >= 0.99, DeltaE00 < 0.85,
    >98% gate savings) unless measured.

Rules: keep tests passing, keep deterministic seeds, don't invent numbers, tell me what could not be run.
Do not add Claude as a co-author in commits.
