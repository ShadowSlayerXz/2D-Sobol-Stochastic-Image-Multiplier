// =====================================================================================
// extra_experiments.cpp — rotation de-correlation, operation-aware correlation, and the
// selection-bias check for adaptive seed selection  (tasks 4, 6, 7)
// =====================================================================================
//   rotation : circular-rotate stream B by an offset before the AND; measure SCC and the
//              multiplication error for random / same-reference / Sobol streams.
//   ops      : AND (multiply) needs SCC 0; AND (min), OR (max), XOR (|a-b|) need SCC +1.
//              Shows the same-reference control is right for the last three, wrong for multiply.
//   selbias  : adaptive selection picks the best of 32 digital-shift seeds on a calibration
//              grid; it is then scored on a disjoint grid and on image pixel pairs, against the
//              average candidate in the same pool (what a blind pick would give). Reports the
//              mean gain over many independent pools with a 95% CI.
//
// Usage: run_extra [out_dir] [ppm_dir]       writes <out_dir>/rotation.csv, ops.csv, selection_bias.csv
// =====================================================================================
#include "sc_image.hpp"
#include "seed_selector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<double> testGrid() { std::vector<double> g; for (int i = 1; i <= 9; i++) g.push_back(i / 10.0); return g; }
std::vector<double> calibrationGrid() { std::vector<double> g; for (int i = 0; i < 10; i++) g.push_back((2 * i + 1) / 20.0); return g; }

struct MeanCI { double mean = 0, ci = 0; };
MeanCI meanCI(const std::vector<double>& v) {
    MeanCI r; double k = static_cast<double>(v.size());
    for (double x : v) r.mean += x;
    r.mean /= k;
    if (v.size() > 1) {
        double var = 0; for (double x : v) var += (x - r.mean) * (x - r.mean);
        r.ci = 1.96 * std::sqrt(var / (k - 1)) / std::sqrt(k);   // sample std-dev, 95% CI of the mean
    }
    return r;
}

struct Src { std::string name; int seeds; std::function<std::vector<SobolGenerator::Point2D>(int n, int seed)> pts; };

std::vector<SobolGenerator::Point2D> randomPoints(int n, int seed) {
    std::mt19937 ex(1000 + seed), ey(5000 + seed);
    std::vector<SobolGenerator::Point2D> p;
    for (int i = 0; i < n; i++) p.push_back({static_cast<float>(ex() / 4294967296.0), static_cast<float>(ey() / 4294967296.0)});
    return p;
}

std::vector<Src> sources() {
    return {
        {"random_independent", 20, [](int n, int s) { return randomPoints(n, s + 1); }},
        {"same_reference", 1, [](int n, int) { SobolConfig c; c.dimX = c.dimY = 1; return generatePoints(c, n); }},
        {"sobol_std", 1, [](int n, int) { return generatePoints(SobolConfig{}, n); }},
    };
}

Bitstream rotated(const Bitstream& s, int offset) {
    Bitstream r(s.size());
    for (size_t i = 0; i < s.size(); i++) r[(i + offset) % s.size()] = s[i];
    return r;
}

// ---- task 4 --------------------------------------------------------------------------------
void rotationExperiment(const std::string& out) {
    std::ofstream csv(out + "/rotation.csv");
    csv << "N,source,offset,trials,mean_abs_scc,mean_abs_scc_ci95,mae_mult,mae_mult_ci95\n";
    std::printf("\n=== Task 4: circular rotation of stream B before the AND (test grid 0.1..0.9) ===\n");
    std::printf("%5s %-19s %6s %9s %9s\n", "N", "source", "offset", "mean|SCC|", "MAE(x1e3)");
    const std::vector<double> grid = testGrid();
    for (int n : {64, 256, 1024}) {
        for (const Src& src : sources()) {
            for (int off : {0, 1, 3, 7, 16, 64, 128, 512}) {
                if (off >= n) continue;
                std::vector<double> sccs, maes;
                for (int s = 0; s < src.seeds; s++) {
                    auto pts = src.pts(n, s);
                    auto rx = coordinate(pts, false), ry = coordinate(pts, true);
                    double sSum = 0, eSum = 0;
                    for (double pa : grid) {
                        Bitstream a = makeStream(rx, pa);
                        for (double pb : grid) {
                            Bitstream b = rotated(makeStream(ry, pb), off);
                            sSum += std::fabs(scc(a, b));
                            eSum += std::fabs(andProduct(a, b) - pa * pb);
                        }
                    }
                    sccs.push_back(sSum / (grid.size() * grid.size())); maes.push_back(eSum / (grid.size() * grid.size()));
                }
                MeanCI c = meanCI(sccs), e = meanCI(maes);
                csv << n << "," << src.name << "," << off << "," << src.seeds << "," << c.mean << "," << c.ci << "," << e.mean << "," << e.ci << "\n";
                std::printf("%5d %-19s %6d %9.4f %9.3f\n", n, src.name.c_str(), off, c.mean, e.mean * 1e3);
            }
        }
    }
}

// ---- task 7 --------------------------------------------------------------------------------
void opsExperiment(const std::string& out) {
    std::ofstream csv(out + "/ops.csv");
    csv << "N,source,operation,needs_scc,mean_scc_measured,mean_abs_error,mean_abs_error_ci95\n";
    std::printf("\n=== Task 7: operation-aware correlation (N=256, test grid 0.1..0.9) ===\n");
    std::printf("%-19s %-22s %-9s %10s %12s\n", "source", "operation", "needs SCC", "measured", "mean|err|x1e3");
    const std::vector<double> grid = testGrid();
    const int n = 256;
    struct Op { const char* name; const char* needs; };
    const Op ops[] = {{"multiply (AND)", "0"}, {"min (AND)", "+1"}, {"max (OR)", "+1"}, {"|a-b| (XOR)", "+1"}};
    for (const Src& src : sources()) {
        for (int o = 0; o < 4; o++) {
            std::vector<double> errs, sccs;
            for (int s = 0; s < src.seeds; s++) {
                auto pts = src.pts(n, s);
                auto rx = coordinate(pts, false), ry = coordinate(pts, true);
                double eSum = 0, cSum = 0;
                for (double pa : grid) {
                    Bitstream a = makeStream(rx, pa);
                    for (double pb : grid) {
                        Bitstream b = makeStream(ry, pb);
                        int cnt = 0;
                        for (size_t k = 0; k < a.size(); k++)
                            cnt += (o == 2) ? (a[k] | b[k]) : (o == 3) ? (a[k] ^ b[k]) : (a[k] & b[k]);
                        double got = double(cnt) / n;
                        double want = o == 0 ? pa * pb : o == 1 ? std::min(pa, pb) : o == 2 ? std::max(pa, pb) : std::fabs(pa - pb);
                        eSum += std::fabs(got - want); cSum += scc(a, b);
                    }
                }
                errs.push_back(eSum / (grid.size() * grid.size())); sccs.push_back(cSum / (grid.size() * grid.size()));
            }
            MeanCI e = meanCI(errs), c = meanCI(sccs);
            csv << n << "," << src.name << "," << ops[o].name << "," << ops[o].needs << "," << c.mean << "," << e.mean << "," << e.ci << "\n";
            std::printf("%-19s %-22s %-9s %10.3f %12.2f\n", src.name.c_str(), ops[o].name, ops[o].needs, c.mean, e.mean * 1e3);
        }
    }
}

// ---- task 6 --------------------------------------------------------------------------------
double pixelError(const Image& A, const Image& B, const Image& exact, const SobolConfig& cfg, int n) {
    RefPair rp = fromSobol(cfg, n);
    StreamBank ba = buildBank(rp.x), bb = buildBank(rp.y);
    return compareImages(stochasticProduct(A, B, ba, bb, kernelFor(KernelPath::Scalar)), exact).mae;
}

void selectionBias(const std::string& out, const std::string& ppmDir) {
    std::ofstream csv(out + "/selection_bias.csv");
    csv << "N,evaluation,repeats,gain_mean,gain_ci95,selected_mean,pool_mean,oracle_mean,selected_beats_pool_frac\n";
    std::printf("\n=== Task 6: is the adaptive-selection gain just selection bias? ===\n");
    std::printf("Select best of 32 digital-shift seeds on the calibration grid (J = mean|SCC| + D*); score on data the\n"
                "selection never saw. gain = pool-average error - selected error (positive = selection helps).\n");

    Image A, B, exact; bool haveImages = false;
    {
        Image b2;
        if (readPPM(ppmDir + "/real_image_500x500.ppm", A) && readPPM(ppmDir + "/real_image_768x512_crop500.ppm", b2) &&
            A.w == b2.w && A.h == b2.h) { B = b2; exact = exactProduct(A, B); haveImages = true; }
        else std::printf("(PPM images not found in %s: pixel-distribution evaluation skipped)\n", ppmDir.c_str());
    }
    const int REPEATS = 40, POOL = 32;
    std::printf("%5s %-34s %10s %18s %10s %10s %10s %8s\n", "N", "evaluation", "repeats", "gain (+-CI95)", "selected", "pool avg", "oracle", "win%");
    for (int n : {64, 256}) {
        struct Eval { std::string name; std::vector<double> sel, pool, oracle, gain; int wins = 0; };
        std::vector<Eval> ev(haveImages ? 3 : 2);
        ev[0].name = "calibration grid (in-sample)"; ev[1].name = "held-out grid 0.1..0.9 (MAE x1e3)";
        if (haveImages) ev[2].name = "image pixels, MAE in levels";
        for (int r = 0; r < REPEATS; r++) {
            std::vector<SobolConfig> cand;
            for (int s = 0; s < POOL; s++) { SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = 1 + r * POOL + s; cand.push_back(c); }
            ScoredConfig best = selectBest(cand, n, calibrationGrid(), Objective{});
            auto errOf = [&](int which, const SobolConfig& c) {
                if (which == 0) { auto p = generatePoints(c, n); return evaluatePoints(p, calibrationGrid()).mae * 1e3; }
                if (which == 1) { auto p = generatePoints(c, n); return evaluatePoints(p, testGrid()).mae * 1e3; }
                return pixelError(A, B, exact, c, n);
            };
            for (size_t w = 0; w < ev.size(); w++) {
                double selErr = errOf(static_cast<int>(w), best.cfg), sum = 0, mn = 1e18;
                for (const auto& c : cand) { double e = errOf(static_cast<int>(w), c); sum += e; mn = std::min(mn, e); }
                double poolAvg = sum / POOL;
                ev[w].sel.push_back(selErr); ev[w].pool.push_back(poolAvg); ev[w].oracle.push_back(mn);
                ev[w].gain.push_back(poolAvg - selErr); if (selErr < poolAvg) ev[w].wins++;
            }
        }
        for (auto& e : ev) {
            MeanCI g = meanCI(e.gain), s = meanCI(e.sel), p = meanCI(e.pool), o = meanCI(e.oracle);
            double winFrac = double(e.wins) / REPEATS;
            csv << n << "," << e.name << "," << REPEATS << "," << g.mean << "," << g.ci << "," << s.mean << "," << p.mean << "," << o.mean << "," << winFrac << "\n";
            std::printf("%5d %-34s %10d %9.4f +-%7.4f %10.4f %10.4f %10.4f %7.0f%%\n", n, e.name.c_str(), REPEATS, g.mean, g.ci, s.mean, p.mean, o.mean, winFrac * 100);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : "results";
    std::string ppm = argc > 2 ? argv[2] : out + "/ppm";
    std::filesystem::create_directories(out);
    rotationExperiment(out);
    opsExperiment(out);
    selectionBias(out, ppm);
    return 0;
}
