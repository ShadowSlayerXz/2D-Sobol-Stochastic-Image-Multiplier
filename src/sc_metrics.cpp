// sc_metrics.cpp — see sc_metrics.hpp.
#include "sc_metrics.hpp"
#include <algorithm>
#include <cmath>

Bitstream makeStream(const std::vector<double>& ref, double p) {
    Bitstream s(ref.size());
    for (size_t k = 0; k < ref.size(); k++) s[k] = ref[k] < p ? 1 : 0;
    return s;
}

std::vector<SobolGenerator::Point2D> generatePoints(const SobolConfig& cfg, int n) {
    SobolGenerator gen(cfg);
    std::vector<SobolGenerator::Point2D> pts;
    pts.reserve(n);
    for (int i = 0; i < n; i++) pts.push_back(gen.next());
    return pts;
}

std::vector<double> coordinate(const std::vector<SobolGenerator::Point2D>& pts, bool useY) {
    std::vector<double> c;
    c.reserve(pts.size());
    for (const auto& p : pts) c.push_back(useY ? p.second : p.first);
    return c;
}

int popcount(const Bitstream& s) {
    int n = 0;
    for (uint8_t b : s) n += b;
    return n;
}

// Joint counts: a = #(1,1), pa = #ones in A, pb = #ones in B.
static void joint(const Bitstream& x, const Bitstream& y, int& a, int& pa, int& pb) {
    a = pa = pb = 0;
    for (size_t k = 0; k < x.size(); k++) {
        a += x[k] & y[k];
        pa += x[k];
        pb += y[k];
    }
}

double pearson(const Bitstream& x, const Bitstream& y) {
    int a, pa, pb;
    joint(x, y, a, pa, pb);
    double n = static_cast<double>(x.size());
    double varA = static_cast<double>(pa) * (n - pa);
    double varB = static_cast<double>(pb) * (n - pb);
    if (varA <= 0 || varB <= 0) return 0.0;
    return (n * a - static_cast<double>(pa) * pb) / std::sqrt(varA * varB);
}

double scc(const Bitstream& x, const Bitstream& y) {
    int a, pa, pb;
    joint(x, y, a, pa, pb);
    double n = static_cast<double>(x.size());
    double numer = n * a - static_cast<double>(pa) * pb;      // = ad - bc
    double denom;
    if (numer > 0) denom = n * std::min(pa, pb) - static_cast<double>(pa) * pb;
    else           denom = static_cast<double>(pa) * pb - n * std::max(0, pa + pb - static_cast<int>(x.size()));
    if (denom == 0) return 0.0;
    return numer / denom;
}

double hammingFraction(const Bitstream& x, const Bitstream& y) {
    int d = 0;
    for (size_t k = 0; k < x.size(); k++) d += x[k] != y[k];
    return static_cast<double>(d) / x.size();
}

double autocorrelation(const Bitstream& s, int lag) {
    if (lag <= 0 || static_cast<size_t>(lag) >= s.size()) return 0.0;
    Bitstream head(s.begin(), s.end() - lag), tail(s.begin() + lag, s.end());
    return pearson(head, tail);
}

double andProduct(const Bitstream& x, const Bitstream& y) {
    int a, pa, pb;
    joint(x, y, a, pa, pb);
    return static_cast<double>(a) / x.size();
}

double multError(const Bitstream& x, const Bitstream& y) {
    double n = static_cast<double>(x.size());
    return std::fabs(andProduct(x, y) - (popcount(x) / n) * (popcount(y) / n));
}

double starDiscrepancy2D(const std::vector<SobolGenerator::Point2D>& pts) {
    // O(N^2): sort by y once; for every x-corner sweep upward through y keeping running
    // counts of points with x < cx ("open") and x <= cx ("closed").
    const size_t n = pts.size();
    std::vector<SobolGenerator::Point2D> byY(pts);
    std::sort(byY.begin(), byY.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    std::vector<double> xs;
    for (const auto& p : pts) xs.push_back(p.first);
    xs.push_back(1.0);

    double best = 0;
    for (double cx : xs) {
        size_t open = 0, closed = 0;      // among points with y strictly below the current group
        size_t i = 0;
        while (i < n) {
            double cy = byY[i].second;
            size_t j = i, openG = 0, closedG = 0;     // the group of points with y == cy
            for (; j < n && byY[j].second == cy; j++) {
                openG += byY[j].first < cx;
                closedG += byY[j].first <= cx;
            }
            double vol = cx * cy;
            // box [0,cx)x[0,cy) holds only earlier groups; box [0,cx]x[0,cy] also holds this one
            best = std::max(best, std::fabs(open / double(n) - vol));
            best = std::max(best, std::fabs((closed + closedG) / double(n) - vol));
            open += openG; closed += closedG;
            i = j;
        }
        // cy = 1: every point is below
        best = std::max(best, std::fabs(open / double(n) - cx));
        best = std::max(best, std::fabs(closed / double(n) - cx));
    }
    return best;
}

Metrics evaluatePoints(const std::vector<SobolGenerator::Point2D>& pts, const std::vector<double>& grid) {
    Metrics m;
    std::vector<double> rx = coordinate(pts, false), ry = coordinate(pts, true);
    std::vector<Bitstream> sa, sb;
    for (double p : grid) { sa.push_back(makeStream(rx, p)); sb.push_back(makeStream(ry, p)); }

    int pairs = 0;
    for (size_t i = 0; i < grid.size(); i++) {
        m.autocorr += 0.5 * (std::fabs(autocorrelation(sa[i], 1)) + std::fabs(autocorrelation(sb[i], 1)));
        for (size_t j = 0; j < grid.size(); j++) {
            double err = andProduct(sa[i], sb[j]) - grid[i] * grid[j];
            m.absScc += std::fabs(scc(sa[i], sb[j]));
            m.absPearson += std::fabs(pearson(sa[i], sb[j]));
            m.mae += std::fabs(err);
            m.mse += err * err;
            m.emult += multError(sa[i], sb[j]);
            m.hamming += hammingFraction(sa[i], sb[j]);
            pairs++;
        }
    }
    m.absScc /= pairs; m.absPearson /= pairs; m.mae /= pairs; m.mse /= pairs;
    m.emult /= pairs; m.hamming /= pairs; m.autocorr /= grid.size();
    m.disc = starDiscrepancy2D(pts);
    return m;
}
