// =====================================================================================
// test_sobol.cpp — Phase 1 validation for SobolGenerator
// =====================================================================================
// Each test checks one property the stochastic multiplier depends on. If any fail,
// later phases will produce wrong products, so this must be green before Phase 2.
//
// Reference points are the unscrambled 2D Sobol sequence (Joe-Kuo direction numbers,
// Gray-code order), identical to python: scipy.stats.qmc.Sobol(d=2, scramble=False).random(16)
//
// Run with:  ./build/test_sobol   (exit code 0 = all passed, 1 = something failed)
// =====================================================================================

#include "sobol_generator.hpp"

#include <cstdio>  // std::printf, std::snprintf
#include <vector>  // std::vector: a resizable array

// Counts failed checks. 'static' at file level means only this file can see it.
static int failures = 0;

// Tiny test helper: prints PASS/FAIL and remembers failures.
// (A real project might use GoogleTest or Catch2. This keeps things dependency-free.)
static void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

// -------------------------------------------------------------------------------------
// Test 1: the first 16 points must match the reference exactly.
// All values are multiples of 1/16, which floats store exactly, so comparing with ==
// is safe here. (For general float results you would compare within a tolerance.)
// -------------------------------------------------------------------------------------
static void testFirst16Points() {
    const float expected[16][2] = {
        {0.0f, 0.0f},       {0.5f, 0.5f},       {0.75f, 0.25f},     {0.25f, 0.75f},
        {0.375f, 0.375f},   {0.875f, 0.875f},   {0.625f, 0.125f},   {0.125f, 0.625f},
        {0.1875f, 0.3125f}, {0.6875f, 0.8125f}, {0.9375f, 0.0625f}, {0.4375f, 0.5625f},
        {0.3125f, 0.1875f}, {0.8125f, 0.6875f}, {0.5625f, 0.4375f}, {0.0625f, 0.9375f},
    };

    SobolGenerator gen;
    bool ok = true;
    for (int i = 0; i < 16; i++) {
        SobolGenerator::Point2D p = gen.next();
        if (p.first != expected[i][0] || p.second != expected[i][1]) {
            std::printf("  point %d: got (%g, %g), expected (%g, %g)\n",
                        i, p.first, p.second, expected[i][0], expected[i][1]);
            ok = false;
        }
    }
    check(ok, "first 16 points match reference");
}

// -------------------------------------------------------------------------------------
// Test 2: the net property, the key reason for using Sobol.
// The first N = 2^(2k) points put exactly one point in every (1/2^k x 1/2^k) cell of
// the unit square. We split the square into side x side cells, drop each point into
// its cell, and check that every cell got exactly 1 point.
//   k=2 -> N=16   (4x4 grid)    matches the 16-bit example in the notes
//   k=4 -> N=256  (16x16 grid)  the main stream length (one AVX2 register)
//   k=5 -> N=1024 (32x32 grid)  the high-accuracy stream length
// If the Y direction numbers were wrong, some cells would get 2 points and others 0.
// -------------------------------------------------------------------------------------
static void testNetProperty(int k) {
    const int side = 1 << k;     // 2^k cells along each axis
    const int n = side * side;   // total number of points = number of cells
    std::vector<int> cellCount(n, 0);   // n counters, all starting at 0

    SobolGenerator gen;
    for (int i = 0; i < n; i++) {
        SobolGenerator::Point2D p = gen.next();
        // Which column / row does this point fall in? Multiplying by side and
        // truncating to int gives a cell index from 0 to side-1.
        int cx = static_cast<int>(p.first * side);
        int cy = static_cast<int>(p.second * side);
        cellCount[cy * side + cx]++;    // 2D (row, col) flattened into a 1D index
    }

    bool ok = true;
    for (int c : cellCount) {   // range-based for: visits every element of cellCount
        if (c != 1) ok = false;
    }

    char name[64];
    std::snprintf(name, sizeof(name), "net property N=%d (%dx%d cells)", n, side, side);
    check(ok, name);
}

// -------------------------------------------------------------------------------------
// Test 3: every point must lie in [0,1), i.e. 0 is allowed and 1 is not.
// This catches the float-rounding bug where a state near 2^32 would become 1.0.
// -------------------------------------------------------------------------------------
static void testRange() {
    SobolGenerator gen;
    bool ok = true;
    for (int i = 0; i < 100000; i++) {
        SobolGenerator::Point2D p = gen.next();
        if (p.first < 0.0f || p.first >= 1.0f || p.second < 0.0f || p.second >= 1.0f) ok = false;
    }
    check(ok, "100000 points lie in [0,1)");
}

// -------------------------------------------------------------------------------------
// Test 4: determinism. The plan (§11) requires reproducible generation, so the same
// points must come out again after reset(). We also check currentIndex() counts right.
// -------------------------------------------------------------------------------------
static void testResetDeterminism() {
    SobolGenerator gen;
    std::vector<SobolGenerator::Point2D> first;
    for (int i = 0; i < 256; i++) first.push_back(gen.next());   // remember 256 points
    check(gen.currentIndex() == 256, "currentIndex() == 256 after 256 points");

    gen.reset();
    check(gen.currentIndex() == 0, "currentIndex() == 0 after reset()");

    bool ok = true;
    for (int i = 0; i < 256; i++) {
        if (gen.next() != first[i]) ok = false;   // std::pair supports == / != directly
    }
    check(ok, "sequence repeats identically after reset()");
}

// -------------------------------------------------------------------------------------
// Test 5: the example from the project notes, done the stochastic-computing way.
//   A = 0.75 -> stream bit is 1 when x < 0.75
//   B = 0.50 -> stream bit is 1 when y < 0.50
//   AND the two bits and count the 1s over N = 16 points.
// The rectangle [0,0.75) x [0,0.5) covers 3 x 2 = 6 of the 16 grid cells, and each cell
// holds exactly one point, so the count must be exactly 6 -> 6/16 = 0.375 = 0.75 * 0.5.
// (Compare with the correlated repeating patterns in the notes, which gave 8/16.)
// -------------------------------------------------------------------------------------
static void testNotesExample() {
    SobolGenerator gen;
    int ones = 0;
    for (int i = 0; i < 16; i++) {
        SobolGenerator::Point2D p = gen.next();
        bool a = p.first < 0.75f;    // bit i of A's stochastic stream
        bool b = p.second < 0.50f;   // bit i of B's stochastic stream
        if (a && b) ones++;          // AND gate, then count the 1s (popcount)
    }
    std::printf("  0.75 x 0.50 -> %d/16 = %g\n", ones, ones / 16.0);
    check(ones == 6, "0.75 x 0.50 = 6/16 = 0.375 with N=16");
}

int main() {
    testFirst16Points();
    testNetProperty(2);  // N = 16
    testNetProperty(4);  // N = 256
    testNetProperty(5);  // N = 1024
    testRange();
    testResetDeterminism();
    testNotesExample();

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
                failures, failures == 1 ? "" : "s");
    // Returning non-zero tells CTest / the shell that the test failed.
    return failures ? 1 : 0;
}
