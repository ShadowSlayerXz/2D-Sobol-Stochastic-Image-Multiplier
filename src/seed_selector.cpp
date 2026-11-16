// seed_selector.cpp — see seed_selector.hpp.
#include "seed_selector.hpp"
#include <cmath>

ScoredConfig scoreConfig(const SobolConfig& cfg, int n, const std::vector<double>& grid, const Objective& obj) {
    std::vector<SobolGenerator::Point2D> pts = generatePoints(cfg, n);
    std::vector<double> rx = coordinate(pts, false), ry = coordinate(pts, true);
    std::vector<Bitstream> sa, sb;
    for (double p : grid) { sa.push_back(makeStream(rx, p)); sb.push_back(makeStream(ry, p)); }

    double sum = 0;
    for (size_t i = 0; i < grid.size(); i++)
        for (size_t j = 0; j < grid.size(); j++) sum += std::fabs(scc(sa[i], sb[j]));

    ScoredConfig s;
    s.cfg = cfg;
    s.absScc = sum / (grid.size() * grid.size());
    s.disc = starDiscrepancy2D(pts);
    s.J = obj.alpha * s.absScc + obj.beta * s.disc;
    return s;
}

ScoredConfig selectBest(const std::vector<SobolConfig>& candidates, int n,
                        const std::vector<double>& grid, const Objective& obj) {
    ScoredConfig best;
    bool first = true;
    for (const SobolConfig& c : candidates) {
        ScoredConfig s = scoreConfig(c, n, grid, obj);
        if (first || s.J < best.J) { best = s; first = false; }
    }
    return best;
}
