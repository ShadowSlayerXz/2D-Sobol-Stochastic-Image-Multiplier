// sobol_generator.cpp — implementation of the 2D Sobol generator.
// Read the big comment at the top of sobol_generator.hpp first for the theory.

#include "sobol_generator.hpp"

// -------------------------------------------------------------------------------------
// Joe-Kuo primitive polynomials and initial direction integers for dimensions 2..10
// (new-joe-kuo-6.21201). s = polynomial degree, a = its middle coefficients (bit s-2 is
// a_1 ... bit 0 is a_{s-1}), m[] = the first s direction integers (odd, m_i < 2^i).
// Dimension 1 needs no entry: it is van der Corput (m_i = 1).
// Verified against scipy.stats.qmc.Sobol for the first 256 points (tests/test_directions).
// -------------------------------------------------------------------------------------
namespace {
struct JoeKuoEntry { int s; uint32_t a; uint32_t m[5]; };
const JoeKuoEntry JOE_KUO[SobolGenerator::MAX_DIM + 1] = {
    {0, 0, {0}},                    // unused slot 0
    {0, 0, {0}},                    // dimension 1: van der Corput (handled separately)
    {1, 0, {1}},                    // dimension 2
    {2, 1, {1, 3}},                 // dimension 3
    {3, 1, {1, 3, 1}},              // dimension 4
    {3, 2, {1, 1, 1}},              // dimension 5
    {4, 1, {1, 1, 3, 3}},           // dimension 6
    {4, 4, {1, 3, 5, 13}},          // dimension 7
    {5, 2, {1, 1, 5, 5, 17}},       // dimension 8
    {5, 4, {1, 1, 5, 5, 5}},        // dimension 9
    {5, 7, {1, 1, 7, 11, 19}},      // dimension 10
};
} // namespace

// -------------------------------------------------------------------------------------
// buildDirectionNumbers: direction numbers v_1..v_32 of Sobol dimension d, in fixed point.
//
// For degree s with coefficients a_1..a_{s-1}, the recurrence on integers is
//     m_i = 2a_1 m_{i-1} ^ 4a_2 m_{i-2} ^ ... ^ 2^{s-1} a_{s-1} m_{i-s+1} ^ 2^s m_{i-s} ^ m_{i-s}
// and with v_i = m_i / 2^i it becomes, in fixed point,
//     v_i = v_{i-s} ^ (v_{i-s} >> s) ^ XOR over k with a_k=1 of v_{i-k}
// (for s = 1 this is the old one-liner v_i = v_{i-1} ^ (v_{i-1} >> 1)).
// -------------------------------------------------------------------------------------
std::array<uint32_t, SobolGenerator::BIT_WIDTH> SobolGenerator::buildDirectionNumbers(int d) {
    std::array<uint32_t, BIT_WIDTH> v{};
    if (d <= 1) {
        // Dimension 1: v_i = 1/2^i, i.e. a single 1 bit shifted into place.
        for (int i = 1; i <= BIT_WIDTH; i++) v[i - 1] = static_cast<uint32_t>(1u) << (BIT_WIDTH - i);
        return v;
    }
    const JoeKuoEntry& e = JOE_KUO[d];
    for (int i = 1; i <= e.s; i++) v[i - 1] = e.m[i - 1] << (BIT_WIDTH - i);
    for (int i = e.s + 1; i <= BIT_WIDTH; i++) {
        uint32_t prev = v[i - e.s - 1];               // v_{i-s}
        uint32_t next = prev ^ (prev >> e.s);
        for (int k = 1; k < e.s; k++)
            if ((e.a >> (e.s - 1 - k)) & 1u) next ^= v[i - k - 1];
        v[i - 1] = next;
    }
    return v;
}

// -------------------------------------------------------------------------------------
// Constructors: set the start state and build both direction-number tables.
// Point 0 of every (unscrambled) Sobol sequence is (0,0), and we start from point 0.
// -------------------------------------------------------------------------------------
SobolGenerator::SobolGenerator() : SobolGenerator(SobolConfig{}) {}

SobolGenerator::SobolGenerator(const SobolConfig& config)
    : config_(config),
      // Independent scrambler per coordinate: derive two seeds from the master seed.
      scramblerX_(config.scramble, config.seed * 2 + 1),
      scramblerY_(config.scramble, config.seed * 2 + 2)
{
    index = 0;
    stateX = 0;
    stateY = 0;
    directionNumbersX = buildDirectionNumbers(config.dimX);
    directionNumbersY = buildDirectionNumbers(config.dimY);
}

// -------------------------------------------------------------------------------------
// rightmostZeroBit: which direction number to XOR in when stepping from point k to k+1.
//
// Going from gray(k) to gray(k+1) flips exactly one bit, and that bit sits at the
// position of the rightmost 0 in k. Positions are counted from 1:
//
//     k = 0  (...000) -> 1        k = 1 (...001) -> 2
//     k = 2  (...010) -> 1        k = 3 (...011) -> 3
//     k = 7  (...111) -> 4
//
// The loop shifts away the trailing 1s and counts them. It is a "count trailing ones"
// operation; hardware (and __builtin_ctz(~k) + 1) can do it in one instruction.
// -------------------------------------------------------------------------------------
int SobolGenerator::rightmostZeroBit(uint32_t k) {
    int position = 1;

    while (k & 1u) {    // while the lowest bit is 1 ...
        k >>= 1;        // ... drop it
        position++;     // ... and move one position to the left
    }
    return position;
}

// -------------------------------------------------------------------------------------
// toUnit: convert a 32-bit fixed-point state to a float in [0,1).
//
// A float only has a 24-bit mantissa. Converting all 32 bits would round, and states
// very close to 2^32 would round UP to exactly 1.0, which is outside [0,1).
// Keeping only the top 24 bits (state >> 8) and scaling by 1/2^24 gives a value that
// fits exactly in a float and is always < 1.0.
// -------------------------------------------------------------------------------------
float SobolGenerator::toUnit(uint32_t state) {
    // keep the top 24 bits so the value is exact in a float and can never round up to 1.0
    return static_cast<float>(state >> 8) * (1.0f / 16777216.0f); // 2 pow 24
}

// -------------------------------------------------------------------------------------
// next: return the current point, then advance by one step.
//
// Worked example of the first few calls (showing only the X column; Y works the same
// way with its own direction numbers):
//
//   call | returns x | index before | rightmost 0 | XOR in | new stateX
//   -----+-----------+--------------+-------------+--------+-----------
//     1  |   0       |   0 (000)    |      1      |  1/2   |  0.5
//     2  |   0.5     |   1 (001)    |      2      |  1/4   |  0.75
//     3  |   0.75    |   2 (010)    |      1      |  1/2   |  0.25
//     4  |   0.25    |   3 (011)    |      3      |  1/8   |  0.375
// -------------------------------------------------------------------------------------
std::pair<uint32_t, uint32_t> SobolGenerator::nextRaw() {
    // return the current point first so the sequence starts at point 0 = (0,0);
    // the first 2^k points then form a complete Sobol block
    std::pair<uint32_t, uint32_t> raw(scramblerX_.apply(stateX), scramblerY_.apply(stateY));

    // Advance: find the single Gray-code bit that flips, and XOR the matching
    // direction number into each coordinate. b is 1-based, while arrays are 0-based,
    // hence [b - 1].
    int b = rightmostZeroBit(index);
    stateX ^= directionNumbersX[b - 1];
    stateY ^= directionNumbersY[b - 1];
    index++;

    return raw;
}

SobolGenerator::Point2D SobolGenerator::next() {
    std::pair<uint32_t, uint32_t> raw = nextRaw();
    return Point2D(toUnit(raw.first), toUnit(raw.second));
}

// -------------------------------------------------------------------------------------
// reset: go back to point 0. Direction numbers never change, so only the state resets.
// -------------------------------------------------------------------------------------
void SobolGenerator::reset() {
    index = 0;
    stateX = 0;
    stateY = 0;
}

// currentIndex: how many points next() has handed out since construction/reset.
uint32_t SobolGenerator::currentIndex() const {
    return index;
}
