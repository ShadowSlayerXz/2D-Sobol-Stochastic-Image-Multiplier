// test_avx2_kernel.cpp — scalar and AVX2 AND+popcount must agree bit for bit.
// Checks every one of the 256x256 level pairs for N = 128, 256, 1024 on several reference
// sources, plus random packed words of awkward lengths (tail handling) and the whole image path.
#include "sc_image.hpp"

#include <cstdio>
#include <random>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    std::printf("AVX2 compiled: %s, supported at run time: %s\n", avx2Compiled() ? "yes" : "no", avx2Supported() ? "yes" : "no");
    if (!avx2Supported()) std::printf("NOTE: AVX2 unavailable here; AVX2 entry point == scalar fallback, so this run only checks the fallback.\n");

    AndPopcountFn scalar = kernelFor(KernelPath::Scalar), avx = kernelFor(KernelPath::AVX2);

    // 1) all 256x256 level pairs, three generators, three lengths
    for (int n : {128, 256, 1024}) {
        std::vector<RefPair> sources;
        sources.push_back(fromSobol(SobolConfig{}, n));
        sources.push_back(fromRandom(n, 7));
        { SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = 3; sources.push_back(fromSobol(c, n)); }
        for (size_t s = 0; s < sources.size(); s++) {
            StreamBank ba = buildBank(sources[s].x), bb = buildBank(sources[s].y);
            long mismatches = 0;
            for (int a = 0; a < 256; a++)
                for (int b = 0; b < 256; b++)
                    if (scalar(ba.at(a), bb.at(b), ba.words) != avx(ba.at(a), bb.at(b), ba.words)) mismatches++;
            CHECK(mismatches == 0, "N=%d source=%zu: %ld mismatching level pairs", n, s, mismatches);
        }
        std::printf("N=%4d: 65536 level pairs x 3 sources compared\n", n);
    }

    // 2) arbitrary word counts (exercises the scalar tail) on random data, incl. all-ones / zeros
    std::mt19937_64 rng(12345);
    for (int words = 1; words <= 33; words++) {
        std::vector<uint64_t> a(words), b(words);
        for (int rep = 0; rep < 50; rep++) {
            for (int i = 0; i < words; i++) { a[i] = rng(); b[i] = rng(); }
            if (rep == 0) for (int i = 0; i < words; i++) { a[i] = ~0ull; b[i] = ~0ull; }
            if (rep == 1) for (int i = 0; i < words; i++) { a[i] = 0; }
            CHECK(scalar(a.data(), b.data(), words) == avx(a.data(), b.data(), words), "words=%d rep=%d", words, rep);
        }
    }

    // 3) identical output images through the whole multiplier
    Image A, B; A.w = B.w = 64; A.h = B.h = 48;
    A.px.resize(64 * 48 * 3); B.px.resize(64 * 48 * 3);
    for (size_t i = 0; i < A.px.size(); i++) { A.px[i] = rng() & 255; B.px[i] = rng() & 255; }
    for (int n : {128, 256, 1024}) {
        RefPair rp = fromSobol(SobolConfig{}, n);
        StreamBank ba = buildBank(rp.x), bb = buildBank(rp.y);
        Image o1 = stochasticProduct(A, B, ba, bb, scalar), o2 = stochasticProduct(A, B, ba, bb, avx);
        CHECK(o1.px == o2.px, "image output differs at N=%d", n);
    }

    std::printf(failures ? "test_avx2_kernel: %d FAILURE(S)\n" : "test_avx2_kernel: all passed\n", failures);
    return failures ? 1 : 0;
}
