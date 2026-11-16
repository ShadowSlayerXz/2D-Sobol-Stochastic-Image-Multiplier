// test_neon_kernel.cpp — NEON AND+popcount must equal the scalar kernel bit for bit.
// Skips (passes) on non-ARM64 builds, where the NEON entry point is the scalar fallback.
#include "sc_image.hpp"
#include <cstdio>
#include <random>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    std::printf("NEON compiled: %s\n", neonCompiled() ? "yes" : "no");
    if (!neonSupported()) { std::printf("NOTE: not an ARM64 build; nothing to compare.\ntest_neon_kernel: all passed\n"); return 0; }
    AndPopcountFn scalar = kernelFor(KernelPath::Scalar), neon = kernelFor(KernelPath::NEON);
    CHECK(neon == andPopcountNeon, "kernelFor(NEON) did not resolve to the NEON kernel");

    for (int n : {128, 256, 1024}) {
        std::vector<RefPair> src = {fromSobol(SobolConfig{}, n), fromRandom(n, 7)};
        for (size_t s = 0; s < src.size(); s++) {
            StreamBank ba = buildBank(src[s].x), bb = buildBank(src[s].y);
            long bad = 0;
            for (int a = 0; a < 256; a++)
                for (int b = 0; b < 256; b++)
                    if (scalar(ba.at(a), bb.at(b), ba.words) != neon(ba.at(a), bb.at(b), ba.words)) bad++;
            CHECK(bad == 0, "N=%d source=%zu: %ld mismatching level pairs", n, s, bad);
        }
        std::printf("N=%4d: 65536 level pairs x 2 sources compared\n", n);
    }
    std::mt19937_64 rng(777);
    for (int words = 0; words <= 40; words++) {          // covers the 4-, 2- and 1-word tails
        std::vector<uint64_t> a(words + 1), b(words + 1);
        for (int rep = 0; rep < 100; rep++) {
            for (int i = 0; i < words; i++) { a[i] = rng(); b[i] = rng(); }
            if (rep == 0) for (int i = 0; i < words; i++) a[i] = b[i] = ~0ull;
            if (rep == 1) for (int i = 0; i < words; i++) a[i] = 0;
            CHECK(scalar(a.data(), b.data(), words) == neon(a.data(), b.data(), words), "words=%d rep=%d", words, rep);
        }
    }
    std::printf(failures ? "test_neon_kernel: %d FAILURE(S)\n" : "test_neon_kernel: all passed\n", failures);
    return failures ? 1 : 0;
}
