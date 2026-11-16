// Checks the packed AND+popcount identity against the textbook bitstream definition, and the
// 0/255 endpoints. Re-implements the tiny bank builder so it tests the maths, not image_demo.
#include "sc_metrics.hpp"
#include <cstdio>
#include <cstdlib>

static int failures = 0;
static void check(bool ok, const char* name) { std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name); if (!ok) failures++; }

int main() {
    const int N = 256;
    SobolGenerator g; std::vector<uint32_t> rx, ry;
    for (int i = 0; i < N; i++) { auto p = g.nextRaw(); rx.push_back(p.first); ry.push_back(p.second); }
    auto bit = [](uint32_t ref, int level) { return (uint64_t)ref * 255u < ((uint64_t)level << 32); };

    bool allOnes = true, allZeros = true;
    for (int k = 0; k < N; k++) { allOnes &= bit(rx[k], 255); allZeros &= !bit(rx[k], 0); }
    check(allOnes, "level 255 -> all-ones stream");
    check(allZeros, "level 0 -> all-zeros stream");

    // Integer test must equal the floating-point definition ref < level/255 (away from ties).
    bool same = true;
    for (int level = 0; level < 256; level++)
        for (int k = 0; k < N; k++) same &= (bit(rx[k], level) == ((double)rx[k] / 4294967296.0 < level / 255.0));
    check(same, "integer threshold == floating-point threshold");

    // AND+popcount product for 0.75-ish x 0.5-ish levels matches makeStream-based product.
    std::vector<double> dx(rx.begin(), rx.end()), dy(ry.begin(), ry.end());
    for (auto& v : dx) v /= 4294967296.0;
    for (auto& v : dy) v /= 4294967296.0;
    bool prod = true;
    for (int a = 0; a < 256; a += 17)
        for (int b = 0; b < 256; b += 17) {
            int c = 0; for (int k = 0; k < N; k++) c += bit(rx[k], a) && bit(ry[k], b);
            Bitstream sa = makeStream(dx, a / 255.0), sb = makeStream(dy, b / 255.0);
            prod &= (c == (int)(andProduct(sa, sb) * N + 0.5));
            // product accuracy: within 4/N of exact for N=256 Sobol
            prod &= std::abs(c / (double)N - (a / 255.0) * (b / 255.0)) < 4.0 / N + 1e-9;
        }
    check(prod, "packed/integer product == stream product, error < 4/N");
    std::printf(failures ? "FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
