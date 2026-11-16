// =====================================================================================
// sc_metrics.hpp — correlation / accuracy metrics for stochastic bitstreams
// =====================================================================================
// Implements the metric table of Research Gap 2 (§4): cross-correlation (Pearson and the
// stochastic-computing-specific SCC), autocorrelation, MAE, MSE, discrepancy, Hamming
// distance and the AND-multiplication error E_mult = |P(A&B) - P(A)P(B)|.
// =====================================================================================
#pragma once
#include <cstdint>
#include <vector>
#include "sobol_generator.hpp"

using Bitstream = std::vector<uint8_t>;   // one 0/1 per element

// ---- stream construction --------------------------------------------------------------
// bit_k = 1 if ref_k < p. `ref` are the reference numbers (a Sobol coordinate, say).
Bitstream makeStream(const std::vector<double>& ref, double p);

// First n points of a generator configuration.
std::vector<SobolGenerator::Point2D> generatePoints(const SobolConfig& cfg, int n);
std::vector<double> coordinate(const std::vector<SobolGenerator::Point2D>& pts, bool useY);

// ---- pairwise stream metrics ----------------------------------------------------------
int    popcount(const Bitstream& s);
double pearson(const Bitstream& a, const Bitstream& b);       // 0 if either stream is constant
// SCC of Alaghi & Hayes: +1 = maximally overlapping 1s, 0 = independent, -1 = minimally overlapping.
double scc(const Bitstream& a, const Bitstream& b);
double hammingFraction(const Bitstream& a, const Bitstream& b);  // fraction of differing bits
double autocorrelation(const Bitstream& a, int lag);          // Pearson of a[k] vs a[k+lag]
double andProduct(const Bitstream& a, const Bitstream& b);    // P(A & B) = popcount(A&B)/N
double multError(const Bitstream& a, const Bitstream& b);     // |P(A&B) - P(A)P(B)|

// ---- point-set metric -----------------------------------------------------------------
// Exact 2D star discrepancy D*_N = sup over boxes [0,x)x[0,y) of | count/N - x*y |, O(N^2).
double starDiscrepancy2D(const std::vector<SobolGenerator::Point2D>& pts);

// ---- aggregate over a probability grid ------------------------------------------------
struct Metrics {
    double absScc = 0;        // mean |SCC| over all (pA, pB) pairs
    double absPearson = 0;    // mean |Pearson|
    double mae = 0;           // mean |P(A&B) - pA*pB|  (error vs the exact product)
    double mse = 0;           // mean (P(A&B) - pA*pB)^2
    double emult = 0;         // mean |P(A&B) - P(A)P(B)| using the measured stream probabilities
    double hamming = 0;       // mean normalised Hamming distance A vs B
    double autocorr = 0;      // mean |lag-1 autocorrelation| of both streams
    double disc = 0;          // 2D star discrepancy of the point set
};
Metrics evaluatePoints(const std::vector<SobolGenerator::Point2D>& pts, const std::vector<double>& grid);
