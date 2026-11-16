// =====================================================================================
// seed_selector.hpp — Approach C: adaptive, correlation-aware configuration search
// =====================================================================================
// Given a list of candidate generator configurations (direction-number pair x scrambler x
// seed), score each one with the objective from the Gap 2 write-up
//
//        J = alpha * mean|SCC|  +  beta * D*
//
// where mean|SCC| is the cross-correlation over a CALIBRATION grid of probabilities and
// D* is the 2D star discrepancy of the first N points (the "distribution error"). The
// candidate with the lowest J wins. Experiments then evaluate the winner on a DIFFERENT
// (held-out) probability grid so the result is not just overfitting the calibration set.
// =====================================================================================
#pragma once
#include <vector>
#include "sc_metrics.hpp"

struct Objective {
    double alpha = 1.0;   // weight on mean |SCC|
    double beta = 1.0;    // weight on star discrepancy
};

struct ScoredConfig {
    SobolConfig cfg;
    double J = 0, absScc = 0, disc = 0;
};

// Score one configuration on N points.
ScoredConfig scoreConfig(const SobolConfig& cfg, int n, const std::vector<double>& calibrationGrid,
                         const Objective& obj);

// Score all candidates and return the best (lowest J). Ties keep the earliest candidate.
ScoredConfig selectBest(const std::vector<SobolConfig>& candidates, int n,
                        const std::vector<double>& calibrationGrid, const Objective& obj);
