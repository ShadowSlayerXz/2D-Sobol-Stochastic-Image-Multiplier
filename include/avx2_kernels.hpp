// =====================================================================================
// avx2_kernels.hpp — AND + popcount kernels for packed stochastic bitstreams
// =====================================================================================
// The hot loop of the stochastic multiplier is   count = popcount(SA & SB)   over N-bit
// streams packed into 64-bit words. Two implementations are provided:
//
//   Scalar  one 64-bit AND + hardware/builtin popcount per word.
//   AVX2    256 stream bits per step: _mm256_and_si256, then a pshufb nibble-LUT popcount
//           (the classic Mula method) summed with _mm256_sad_epu8. A scalar tail handles
//           the last (words % 4) words, so N = 128 (2 words) falls back to the tail only.
//
// The AVX2 translation unit is compiled with -mavx2 -mpopcnt only where the compiler
// supports it (CMake defines SC_HAVE_AVX2); the CPU is checked again at RUN time, so a
// binary built with AVX2 still runs safely on a machine without it.
// =====================================================================================
#pragma once
#include <cstdint>

enum class KernelPath { Scalar, AVX2 };

// popcount(a[0..words) & b[0..words))
using AndPopcountFn = int (*)(const uint64_t* a, const uint64_t* b, int words);

int andPopcountScalar(const uint64_t* a, const uint64_t* b, int words);
int andPopcountAvx2(const uint64_t* a, const uint64_t* b, int words);   // scalar if not compiled in

bool avx2Compiled();    // was the AVX2 code path built into this binary?
bool avx2Supported();   // compiled AND the running CPU reports AVX2 + POPCNT

// Resolve a path to a function. Requesting AVX2 when unavailable returns the scalar kernel;
// use effectivePath() to learn which one actually runs.
AndPopcountFn kernelFor(KernelPath path);
KernelPath effectivePath(KernelPath requested);
KernelPath bestPath();
// Fastest kernel for streams of `words` 64-bit words, from the measurements in README: the AVX2 nibble-LUT
// popcount only beats 64-bit hardware popcnt once there are >= 8 words (N >= 512).
AndPopcountFn tunedKernel(int words);
const char* pathName(KernelPath path);
