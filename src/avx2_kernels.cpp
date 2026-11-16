// avx2_kernels.cpp — see avx2_kernels.hpp.
#include "avx2_kernels.hpp"

#if defined(SC_HAVE_AVX2) && defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#define SC_AVX2_BUILD 1
#include <immintrin.h>
#endif

int andPopcountScalar(const uint64_t* a, const uint64_t* b, int words) {
    int count = 0;
    for (int j = 0; j < words; j++) count += __builtin_popcountll(a[j] & b[j]);
    return count;
}

#ifdef SC_AVX2_BUILD
int andPopcountAvx2(const uint64_t* a, const uint64_t* b, int words) {
    // Popcount of each nibble 0..15, replicated in both 128-bit lanes (pshufb works per lane).
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                         0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i lowMask = _mm256_set1_epi8(0x0f);
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc = zero;                                   // four 64-bit partial sums
    int j = 0;
    for (; j + 4 <= words; j += 4) {                      // 4 words = 256 stream bits
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
        __m256i v = _mm256_and_si256(va, vb);
        __m256i lo = _mm256_and_si256(v, lowMask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
        __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(cnt, zero));   // bytes -> 64-bit sums
    }
    // horizontal sum of the four 64-bit lanes without a store/reload
    __m128i s2 = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    int count = static_cast<int>(_mm_cvtsi128_si64(_mm_add_epi64(s2, _mm_unpackhi_epi64(s2, s2))));
    for (; j < words; j++) count += static_cast<int>(_mm_popcnt_u64(a[j] & b[j]));   // tail
    return count;
}
bool avx2Compiled() { return true; }
#else
int andPopcountAvx2(const uint64_t* a, const uint64_t* b, int words) { return andPopcountScalar(a, b, words); }
bool avx2Compiled() { return false; }
#endif

bool avx2Supported() {
#if defined(SC_AVX2_BUILD) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
#else
    return false;
#endif
}

KernelPath effectivePath(KernelPath requested) {
    return (requested == KernelPath::AVX2 && avx2Supported()) ? KernelPath::AVX2 : KernelPath::Scalar;
}
AndPopcountFn kernelFor(KernelPath path) {
    return effectivePath(path) == KernelPath::AVX2 ? andPopcountAvx2 : andPopcountScalar;
}
KernelPath bestPath() { return effectivePath(KernelPath::AVX2); }
const char* pathName(KernelPath path) { return path == KernelPath::AVX2 ? "avx2" : "scalar"; }
