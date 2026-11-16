// =====================================================================================
// test_correlation.cpp — validation for the Gap 2 additions
//   1. direction numbers for dimensions 1..10 match scipy (256 points each)
//   2. every scrambler keeps the (0,m,2)-net property and is deterministic / seed-sensitive
//   3. SCC / Pearson / E_mult behave correctly on the worked examples in the Gap 2 notes
//   4. star discrepancy and the seed selector behave sensibly
// =====================================================================================
#include "sc_metrics.hpp"
#include "seed_selector.hpp"
#include "sobol_reference.inc"

#include <cmath>
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}
static Bitstream bits(const char* s) {
    Bitstream b;
    for (; *s; s++) b.push_back(*s == '1');
    return b;
}
static bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

// 1. every dimension against scipy, via the (dimX, dimY) config so both coordinates are exercised
static void testDirectionNumbers() {
    for (int d = 1; d <= SobolGenerator::MAX_DIM; d++) {
        SobolConfig cfg; cfg.dimX = d; cfg.dimY = 11 - d;
        SobolGenerator gen(cfg);
        bool ok = true;
        for (int n = 0; n < 256; n++) {
            auto raw = gen.nextRaw();
            if ((raw.first >> 24) != REF[d - 1][n] % 256 || (raw.second >> 24) != REF[10 - d][n] % 256) ok = false;
        }
        char name[64];
        std::snprintf(name, sizeof(name), "dimension %d (and %d) matches scipy, 256 points", d, 11 - d);
        check(ok, name);
    }
}

// 2. net property: first 4^k points put exactly one point per cell
static bool isNet(const SobolConfig& cfg, int k) {
    int side = 1 << k, n = side * side;
    std::vector<int> cells(n, 0);
    auto pts = generatePoints(cfg, n);
    for (auto& p : pts) cells[int(p.second * side) * side + int(p.first * side)]++;
    for (int c : cells) if (c != 1) return false;
    return true;
}
static void testScramblers() {
    ScrambleMode modes[3] = {ScrambleMode::DigitalShift, ScrambleMode::LinearMatrix, ScrambleMode::Owen};
    for (ScrambleMode m : modes) {
        bool ok = true;
        for (uint64_t seed = 1; seed <= 20; seed++) {
            SobolConfig cfg; cfg.scramble = m; cfg.seed = seed;
            for (int k = 2; k <= 4; k++) ok = ok && isNet(cfg, k);
        }
        char name[80];
        std::snprintf(name, sizeof(name), "%s keeps net property (20 seeds, N=16/64/256)", scrambleName(m));
        check(ok, name);

        SobolConfig a, b; a.scramble = b.scramble = m; a.seed = 5; b.seed = 6;
        SobolGenerator g1(a), g2(a), g3(b);
        bool same = true, diff = false;
        for (int i = 0; i < 64; i++) {
            auto p1 = g1.nextRaw(), p2 = g2.nextRaw(), p3 = g3.nextRaw();
            same = same && p1 == p2;
            diff = diff || p1 != p3;
        }
        std::snprintf(name, sizeof(name), "%s is deterministic per seed and differs across seeds", scrambleName(m));
        check(same && diff, name);

        // bijection sanity on the 32-bit scrambler: distinct inputs -> distinct outputs
        Scrambler s(m, 99);
        bool inj = true;
        for (uint32_t x = 0; x < 5000 && inj; x++) inj = s.apply(x * 0x9E3779B1u) != s.apply((x + 1) * 0x9E3779B1u);
        std::snprintf(name, sizeof(name), "%s maps distinct inputs to distinct outputs (sample)", scrambleName(m));
        check(inj, name);
    }
    SobolConfig plain;
    SobolGenerator g(plain);
    check(g.next() == SobolGenerator::Point2D(0.f, 0.f), "default config still starts at (0,0)");
}

// 3. the worked examples from the Gap 2 notes
static void testMetrics() {
    Bitstream A = bits("10101010"), B = bits("11001100"), C = bits("10101010");
    check(near(andProduct(A, B), 0.25) && near(multError(A, B), 0.0), "notes example: A=10101010, B=11001100 -> 0.25, E_mult 0");
    check(near(andProduct(A, C), 0.5) && near(multError(A, C), 0.25), "notes example: identical streams -> 0.5, E_mult 0.25");
    check(near(scc(A, C), 1.0) && near(pearson(A, C), 1.0), "identical streams: SCC = Pearson = +1");
    check(near(scc(A, B), 0.0) && near(pearson(A, B), 0.0), "independent pair: SCC = Pearson = 0");
    check(near(scc(A, bits("01010101")), -1.0), "complementary streams: SCC = -1");
    check(near(hammingFraction(A, bits("01010101")), 1.0) && near(hammingFraction(A, C), 0.0), "Hamming fraction");
    check(near(autocorrelation(A, 1), -1.0), "alternating stream has lag-1 autocorrelation -1");
    check(near(pearson(bits("1111"), bits("1010")), 0.0), "constant stream gives Pearson 0 (no divide by zero)");
    check(near(scc(bits("1110"), bits("1000")), 1.0), "nested ones (3/4 vs 1/4): SCC = +1");
}

// 4. discrepancy and selector
static void testDiscrepancyAndSelector() {
    std::vector<SobolGenerator::Point2D> pts = generatePoints(SobolConfig{}, 16);
    double d = starDiscrepancy2D(pts);
    check(d > 0 && d < 0.2, "star discrepancy of 16 Sobol points is small");
    std::vector<SobolGenerator::Point2D> diag;
    for (int i = 0; i < 16; i++) diag.push_back({i / 16.0f, i / 16.0f});
    check(starDiscrepancy2D(diag) > 1.5 * d, "diagonal point set has larger discrepancy");

    std::vector<double> grid = {0.15, 0.35, 0.55, 0.75};
    std::vector<SobolConfig> cands;
    for (uint64_t s = 1; s <= 8; s++) { SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = s; cands.push_back(c); }
    Objective obj;
    ScoredConfig best = selectBest(cands, 64, grid, obj);
    bool minimal = true;
    for (auto& c : cands) minimal = minimal && scoreConfig(c, 64, grid, obj).J >= best.J - 1e-15;
    check(minimal, "selectBest returns the lowest-J candidate");

    // The correlated control (same dimension for A and B) must score far worse.
    SobolConfig bad; bad.dimX = bad.dimY = 1;
    check(scoreConfig(bad, 64, grid, obj).J > 3 * best.J, "same-dimension pair is penalised by J");
}

int main() {
    testDirectionNumbers();
    testScramblers();
    testMetrics();
    testDiscrepancyAndSelector();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
