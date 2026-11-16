// =====================================================================================
// scrambling.hpp — cheap, net-preserving scrambles for 32-bit fixed-point Sobol states
// =====================================================================================
// Research Gap 2, Approach B. A scrambler is a bijection on 32-bit fixed-point values
// (an integer s stands for s / 2^32). It is applied to each coordinate AFTER the Sobol
// state is computed, so the underlying Gray-code XOR recurrence is untouched.
//
//   None          identity (the standard, unscrambled Sobol baseline)
//   DigitalShift  s ^ shift.                 1 XOR per coordinate. Preserves every net.
//   LinearMatrix  (L * s) ^ shift over GF(2) with L random lower-triangular, unit
//                 diagonal ("LMS"). ~32 AND/XOR-parity trees. Preserves nets because L is
//                 invertible and an output bit only depends on equal-or-more-significant
//                 input bits.
//   Owen          Hash-based nested uniform scramble (Burley 2020, Laine-Karras hash on
//                 bit-reversed input). Approximates true Owen scrambling: each bit is
//                 flipped depending on all MORE significant bits. Needs 32-bit multipliers
//                 (the most expensive option in hardware).
//
// Each coordinate gets its own independent scrambler derived from one master seed.
// =====================================================================================
#pragma once
#include <array>
#include <cstdint>

enum class ScrambleMode { None, DigitalShift, LinearMatrix, Owen };

// Printable name for tables / CSV.
const char* scrambleName(ScrambleMode mode);

// splitmix64: tiny, well-mixed seed expander (deterministic across platforms).
uint64_t splitmix64(uint64_t& state);

class Scrambler
{
public:
    Scrambler(ScrambleMode mode = ScrambleMode::None, uint64_t seed = 0);

    // Scramble one 32-bit fixed-point value. Bijective for every mode.
    uint32_t apply(uint32_t state) const;

    ScrambleMode mode() const { return mode_; }

private:
    ScrambleMode mode_;
    uint32_t shift_ = 0;                  // digital shift / Owen seed
    // rows_[r] = mask of input bits XORed (parity) to give output bit at MSB-rank r.
    std::array<uint32_t, 32> rows_{};
};
