// =====================================================================================
// sobol_generator.hpp — 2D Sobol low-discrepancy sequence generator
// =====================================================================================
//
// WHAT THIS IS FOR (the big picture)
// ----------------------------------
// In stochastic computing a number p in [0,1] is stored as a stream of N bits where
// roughly p*N of the bits are 1. To build that stream we compare p against N
// "reference" numbers r_0 ... r_{N-1}:
//
//        bit_k = 1  if r_k < p,   else 0
//
// To multiply two numbers A and B we build one stream for each and AND them together:
//
//        C_k = A_k AND B_k        result ≈ (number of 1s in C) / N
//
// This only gives A*B if the two streams are INDEPENDENT. If A and B use the same
// reference numbers, the 1s line up and AND gives min(A,B) instead (the 8/16 mistake
// in the handwritten notes).
//
// This class produces the reference numbers: a sequence of 2D points (x_k, y_k).
//   - x_k is used to build image A's stream
//   - y_k is used to build image B's stream
// The two coordinates of a Sobol sequence are designed to be "orthogonal", so the
// streams they create are as independent as possible.
//
// WHY SOBOL AND NOT rand()?
// -------------------------
// Random points clump together and leave gaps, so results are noisy. Sobol points fill
// the square evenly: the first 2^(2k) points put EXACTLY ONE point in every
// (1/2^k x 1/2^k) grid cell. Counting points inside the rectangle [0,A) x [0,B)
// therefore counts grid cells, which gives the area A*B with very little error.
//
// HOW IT WORKS (the algorithm)
// ----------------------------
// Each dimension has 32 "direction numbers" v_1 ... v_32. Each one is a binary fraction
// stored as a 32-bit fixed-point integer (the integer value / 2^32 is the fraction).
//
// Point n is the XOR of the direction numbers picked out by the bits of gray(n),
// where gray(n) = n ^ (n >> 1) is the Gray code of n.
//
// Consecutive Gray codes differ in exactly ONE bit, so we never recompute from scratch.
// Each step XORs a single direction number into the running state:
//
//        c      = position of the rightmost 0 bit of n
//        state ^= v_c
//
// That is one XOR per dimension per point: very cheap in hardware, which is why this
// suits a COA project. (This is the Antonov–Saleev Gray-code method.)
// =====================================================================================

#pragma once
#include <cstdint> // for integer types like uint32_t, uint_64_t
#include <array>   // std::array: fixed-size array that knows its own size (like a C array but safer)
#include <utility> // we will be using std::pair for returning (x,y) pairs
#include "scrambling.hpp"

// -------------------------------------------------------------------------------------
// SobolConfig — everything the correlation-control module (Gap 2) can vary.
//   dimX / dimY : which Sobol dimensions (1-based, 1..MAX_DIM) feed stream A and stream B.
//                 Dimension 1 is van der Corput; dimension 2 is the classic pair.
//                 (Approach A: different direction numbers.)
//   scramble    : post-processing applied to each coordinate (Approach B).
//   seed        : master seed for the scramblers (Approach C searches over this).
// The default config reproduces the original unscrambled (dim 1, dim 2) generator.
// -------------------------------------------------------------------------------------
struct SobolConfig
{
    int dimX = 1;
    int dimY = 2;
    ScrambleMode scramble = ScrambleMode::None;
    uint64_t seed = 0;
};

class SobolGenerator
{
    public:
        // A 2D point. p.first is x (used for image A), p.second is y (used for image B).
        using Point2D = std::pair<float, float> ;// creating an alias for (x(float),y(float)) type

        // Fills in the direction number tables and sets the sequence to point 0.
        SobolGenerator(); // constructor (standard dims 1 and 2, no scrambling)
        explicit SobolGenerator(const SobolConfig& config); // choose dimensions / scrambling / seed

        // Number of Sobol dimensions with built-in Joe-Kuo direction numbers.
        static constexpr int MAX_DIM = 10;

        const SobolConfig& config() const { return config_; }

        // Same as next() but returns the raw 32-bit fixed-point coordinates
        // (value = raw / 2^32), so streams can be built with exact integer compares.
        std::pair<uint32_t, uint32_t> nextRaw();

        // Returns the current point, then moves one step along the sequence.
        // Call order:  1st call -> (0,0),  2nd -> (0.5,0.5),  3rd -> (0.75,0.25), ...
        // Starting at point 0 matters: the "one point per grid cell" guarantee only holds
        // for a complete block of points 0 .. 2^k - 1.
        Point2D next(); // returns the current (x,y) and then increments teh internal state by one step (first call gives point 0 = (0,0))

        // Goes back to point 0. The sequence is deterministic, so after reset() the same
        // points come out again in the same order (important for reproducible benchmarks).
        void reset(); // resets teh generator back to index 0

        // 'const' at the end means this function promises not to modify the object.
        uint32_t currentIndex() const; // this returns how many points have been generated since the last reset

    private:
        // 'static constexpr' = one compile-time constant shared by all objects,
        // so it can be used as an array size below.
        static constexpr int BIT_WIDTH = 32; // we are taking 32 bits precision fr a dimension

        // Direction numbers, stored in fixed point: element [i-1] holds v_i * 2^32.
        //   X (dimension 1): v_i = 1/2, 1/4, 1/8, ...        -> bits 1000..., 0100..., 0010...
        //   Y (dimension 2): v_i = 1/2, 3/4, 5/8, 15/16, ... -> from the polynomial x + 1
        // In hardware these would be a small ROM (the "DIR_X / DIR_Y" registers in the plan).
        std::array<uint32_t, BIT_WIDTH> directionNumbersX;// fr storeing  32 predefined sobol direction no's used to generate the x coordinates
        std::array<uint32_t, BIT_WIDTH> directionNumbersY;// fr storeing  32 predefined sobol direction no's used to generate the y coordinates

        SobolConfig config_;
        Scrambler scramblerX_;   // applied to stateX on output
        Scrambler scramblerY_;   // applied to stateY on output

        // Builds the 32 direction numbers of Sobol dimension d (1-based) from its
        // Joe-Kuo primitive polynomial and initial m values.
        static std::array<uint32_t, BIT_WIDTH> buildDirectionNumbers(int d);

        // The current point in fixed point: x = stateX / 2^32, y = stateY / 2^32.
        // Keeping integers (not floats) makes each update an exact bitwise XOR.
        uint32_t stateX;// fr storing the current raw 32-bit sobol state for the x coordinate
        uint32_t stateY;// fr storing the current raw 32-bit sobol state for the y coordinate

        // Which point we are on (n). Its bit pattern decides which direction number to use next.
        uint32_t index;// fr storing the current position in teh sobol sequence and to determine which direction no will be used next

        // 'static' member functions don't use any object data, just their argument.
        static int rightmostZeroBit(uint32_t k); //finds the position of the rightmost zero bit  so next() knows which direction no to XOR into the state
        static float toUnit(uint32_t state); // converts a raw 32-bit state to a float in [0,1)
};
