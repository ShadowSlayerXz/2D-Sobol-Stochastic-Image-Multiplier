// scrambling.cpp — see scrambling.hpp for what each mode means.
#include "scrambling.hpp"

const char* scrambleName(ScrambleMode mode) {
    switch (mode) {
        case ScrambleMode::None:         return "none";
        case ScrambleMode::DigitalShift: return "digital_shift";
        case ScrambleMode::LinearMatrix: return "linear_matrix";
        case ScrambleMode::Owen:         return "owen";
    }
    return "?";
}

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t reverseBits(uint32_t x) {
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00u) >> 8) | ((x & 0x00FF00FFu) << 8);
    x = ((x & 0xF0F0F0F0u) >> 4) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x & 0xCCCCCCCCu) >> 2) | ((x & 0x33333333u) << 2);
    x = ((x & 0xAAAAAAAAu) >> 1) | ((x & 0x55555555u) << 1);
    return x;
}

static int parity(uint32_t x) {
    x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    return static_cast<int>(x & 1u);
}

// Laine-Karras style permutation: after bit reversal, "x ^= x * odd" makes each bit
// depend on all lower bits, i.e. (once reversed back) on all more-significant bits.
static uint32_t laineKarras(uint32_t x, uint32_t seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

Scrambler::Scrambler(ScrambleMode mode, uint64_t seed) : mode_(mode) {
    uint64_t s = seed;
    shift_ = static_cast<uint32_t>(splitmix64(s));
    if (mode == ScrambleMode::LinearMatrix) {
        // Row r (r = 0 is the MSB, bit 31) may use input bits at MSB-rank 0..r,
        // i.e. bit positions 31 down to 31-r. The diagonal (bit 31-r) is forced to 1.
        for (int r = 0; r < 32; r++) {
            uint32_t allowed = (r == 31) ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> (r + 1));
            // allowed has the top r+1 bits set
            uint32_t m = static_cast<uint32_t>(splitmix64(s)) & allowed;
            m |= 1u << (31 - r);
            rows_[r] = m;
        }
    }
}

uint32_t Scrambler::apply(uint32_t state) const {
    switch (mode_) {
        case ScrambleMode::None:
            return state;
        case ScrambleMode::DigitalShift:
            return state ^ shift_;
        case ScrambleMode::LinearMatrix: {
            uint32_t out = 0;
            for (int r = 0; r < 32; r++)
                out |= static_cast<uint32_t>(parity(state & rows_[r])) << (31 - r);
            return out ^ shift_;
        }
        case ScrambleMode::Owen:
            return reverseBits(laineKarras(reverseBits(state), shift_));
    }
    return state;
}
