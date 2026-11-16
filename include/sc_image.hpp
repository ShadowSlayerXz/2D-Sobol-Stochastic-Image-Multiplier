// =====================================================================================
// sc_image.hpp — RGB images, packed stream banks and the stochastic image multiplier
// =====================================================================================
// Shared by image_demo and the tests. For every 8-bit channel the product is
//     round(255 * popcount(SA[a] & SB[b]) / N)
// where SA[v] / SB[v] are the N-bit streams for level v (bit k = 1 iff ref_k < v/255, tested
// exactly in integers), built from ONE shared reference sequence per operand — the same
// structure as one SNG feeding a whole array of AND gates.
// =====================================================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "avx2_kernels.hpp"
#include "sobol_generator.hpp"
#include "thread_pool.hpp"

struct Image { int w = 0, h = 0; std::vector<uint8_t> px; };   // interleaved RGB

bool readPPM(const std::string& path, Image& im);
bool writePPM(const std::string& path, const Image& im);
Image centerCrop(const Image& im, int w, int h);

// Streams for all 256 levels: words[level * W + j] holds bits 64j .. 64j+63.
struct StreamBank {
    int n = 0, words = 0;
    std::vector<uint64_t> bits;
    const uint64_t* at(int level) const { return &bits[static_cast<size_t>(level) * words]; }
};
StreamBank buildBank(const std::vector<uint32_t>& ref);

// A pair of reference sequences (stream A's and stream B's).
struct RefPair { std::vector<uint32_t> x, y; };
RefPair fromSobol(const SobolConfig& cfg, int n);
RefPair fromRandom(int n, uint64_t seed);

Image exactProduct(const Image& a, const Image& b);
Image stochasticProduct(const Image& a, const Image& b, const StreamBank& ba, const StreamBank& bb,
                        AndPopcountFn kernel);

// Lookup-table variant: the stochastic product depends only on the (a, b) level pair, so compute all
// 256x256 results once with the AND+popcount kernel (lut[a*256+b]) and then do one table read per
// pixel. Output is bit-identical to stochasticProduct().
// In-place, optionally multithreaded variants (pool may be null = single thread). `lut` must hold 65536
// bytes; `out` must hold `count` bytes. No allocation, so repeated calls reuse the same buffers.
void buildProductLutInto(const StreamBank& ba, const StreamBank& bb, AndPopcountFn kernel, uint8_t* lut,
                         ThreadPool* pool = nullptr);
void lookupProduct(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t count, const uint8_t* lut,
                   ThreadPool* pool = nullptr);
std::vector<uint8_t> buildProductLut(const StreamBank& ba, const StreamBank& bb, AndPopcountFn kernel);
Image stochasticProductLut(const Image& a, const Image& b, const std::vector<uint8_t>& lut);

// Per-channel stochastic result for a single level pair (used by tests / error analysis).
inline int stochasticLevel(int a, int b, const StreamBank& ba, const StreamBank& bb, AndPopcountFn kernel) {
    int count = kernel(ba.at(a), bb.at(b), ba.words);
    return (count * 255 + ba.n / 2) / ba.n;
}

struct ImageStats { double mae = 0, mse = 0, psnr = 0; };
ImageStats compareImages(const Image& got, const Image& exact);
