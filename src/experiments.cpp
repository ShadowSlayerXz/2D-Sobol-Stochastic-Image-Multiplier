// =====================================================================================
// experiments.cpp — Gap 2 experimental pipeline (stages 1-5)
// =====================================================================================
//   Stage 0  controls: independent random streams, and the fully-correlated same-reference case
//   Stage 1  standard 2D Sobol (dims 1,2) baseline
//   Stage 2  Approach A: every pair of Sobol dimensions 1..10 (different direction numbers)
//   Stage 3  Approach B: digital shift / linear-matrix / Owen scrambling, many random seeds
//   Stage 4  Approach C: adaptive selection minimising J = alpha*mean|SCC| + beta*D*
//            (selected on a calibration probability grid, scored on a held-out test grid)
//   Stage 5  side-by-side stochastic-multiplication comparison
//
// Usage: run_experiments [output_dir]     (default: results/)
// Writes <output_dir>/summary.csv and prints the headline tables.
// =====================================================================================
#include "seed_selector.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <filesystem>

namespace {

const int LENGTHS[] = {8, 16, 32, 64, 128, 256};
const int NUM_TRIALS = 100;        // random seeds per scrambling mode (stage 3)
const int ADAPT_CANDIDATES = 32;   // seeds examined per adaptive selection (stage 4)
const int ADAPT_REPEATS = 20;      // independent adaptive selections to average over

std::vector<double> testGrid() { std::vector<double> g; for (int i = 1; i <= 9; i++) g.push_back(i / 10.0); return g; }
std::vector<double> calibrationGrid() { std::vector<double> g; for (int i = 0; i < 10; i++) g.push_back((2 * i + 1) / 20.0); return g; }

struct Row {
    std::string stage, config;
    int n = 0, trials = 1;
    Metrics mean;          // averaged over trials
    double maeStd = 0;     // std-dev of MAE across trials
    double nsPerPoint = 0; // generation latency
    double selectMs = 0;   // one-off search cost (stage 4 only)
};

// Accumulates per-trial metrics into a Row.
struct Accum {
    std::vector<Metrics> all;
    void add(const Metrics& m) { all.push_back(m); }
    void finish(Row& r) const {
        Metrics s;
        for (const Metrics& m : all) {
            s.absScc += m.absScc; s.absPearson += m.absPearson; s.mae += m.mae; s.mse += m.mse;
            s.emult += m.emult; s.hamming += m.hamming; s.autocorr += m.autocorr; s.disc += m.disc;
        }
        double k = static_cast<double>(all.size());
        s.absScc /= k; s.absPearson /= k; s.mae /= k; s.mse /= k; s.emult /= k;
        s.hamming /= k; s.autocorr /= k; s.disc /= k;
        double var = 0;
        for (const Metrics& m : all) var += (m.mae - s.mae) * (m.mae - s.mae);
        r.mean = s;
        r.maeStd = std::sqrt(var / k);
        r.trials = static_cast<int>(all.size());
    }
};

// ns per generated point, measured over a long run so timer resolution is irrelevant.
double latencyNsPerPoint(const SobolConfig& cfg) {
    const int reps = 1 << 20;
    SobolGenerator gen(cfg);
    uint32_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) { auto p = gen.nextRaw(); sink ^= p.first + p.second; }
    auto t1 = std::chrono::steady_clock::now();
    volatile uint32_t keep = sink; (void)keep;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

double randomLatencyNsPerPoint() {
    const int reps = 1 << 20;
    std::mt19937 a(1), b(2);
    uint32_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) sink ^= a() + b();
    auto t1 = std::chrono::steady_clock::now();
    volatile uint32_t keep = sink; (void)keep;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

Row runDeterministic(const std::string& stage, const std::string& label, int n, const SobolConfig& cfg,
                     const std::vector<double>& grid) {
    Row r; r.stage = stage; r.config = label; r.n = n;
    Accum a; a.add(evaluatePoints(generatePoints(cfg, n), grid)); a.finish(r);
    r.nsPerPoint = latencyNsPerPoint(cfg);
    return r;
}

Row runTrials(const std::string& stage, const std::string& label, int n, ScrambleMode mode,
              const std::vector<double>& grid) {
    Row r; r.stage = stage; r.config = label; r.n = n;
    Accum a;
    for (int t = 1; t <= NUM_TRIALS; t++) {
        SobolConfig cfg; cfg.scramble = mode; cfg.seed = static_cast<uint64_t>(t);
        a.add(evaluatePoints(generatePoints(cfg, n), grid));
    }
    a.finish(r);
    SobolConfig lat; lat.scramble = mode; lat.seed = 1;
    r.nsPerPoint = latencyNsPerPoint(lat);
    return r;
}

Row runRandom(int n, const std::vector<double>& grid) {
    Row r; r.stage = "0_control"; r.config = "random_independent"; r.n = n;
    Accum a;
    for (int t = 1; t <= NUM_TRIALS; t++) {
        std::mt19937 ex(1000 + t), ey(5000 + t);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::vector<SobolGenerator::Point2D> pts;
        for (int i = 0; i < n; i++) { float x = u(ex); float y = u(ey); pts.push_back({x, y}); }
        a.add(evaluatePoints(pts, grid));
    }
    a.finish(r);
    r.nsPerPoint = randomLatencyNsPerPoint();
    return r;
}

// Adaptive selection: ADAPT_REPEATS independent searches, each over ADAPT_CANDIDATES-or-more
// candidates, chosen on the calibration grid and scored on the test grid.
Row runAdaptive(const std::string& label, int n, const std::vector<std::vector<SobolConfig>>& pools,
                const std::vector<double>& cal, const std::vector<double>& test, const Objective& obj) {
    Row r; r.stage = "4_adaptive"; r.config = label; r.n = n;
    Accum a;
    double totalMs = 0;
    SobolConfig last;
    for (const auto& pool : pools) {
        auto t0 = std::chrono::steady_clock::now();
        ScoredConfig best = selectBest(pool, n, cal, obj);
        auto t1 = std::chrono::steady_clock::now();
        totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        a.add(evaluatePoints(generatePoints(best.cfg, n), test));
        last = best.cfg;
    }
    a.finish(r);
    r.selectMs = totalMs / pools.size();
    r.nsPerPoint = latencyNsPerPoint(last);   // after selection the generator costs the same as any scrambled one
    return r;
}

void printHeader() {
    std::printf("%-30s %4s %9s %9s %10s %10s %10s %9s %8s\n",
                "config", "N", "mean|SCC|", "|pearson|", "MAE(x1e3)", "MSE(x1e6)", "MAE std", "disc D*", "ns/pt");
}
void printRow(const Row& r) {
    std::printf("%-30s %4d %9.4f %9.4f %10.3f %10.3f %10.3f %9.4f %8.2f\n", r.config.c_str(), r.n,
                r.mean.absScc, r.mean.absPearson, r.mean.mae * 1e3, r.mean.mse * 1e6, r.maeStd * 1e3,
                r.mean.disc, r.nsPerPoint);
}

} // namespace

int main(int argc, char** argv) {
    std::string outDir = argc > 1 ? argv[1] : "results";
    std::filesystem::create_directories(outDir);

    const std::vector<double> test = testGrid(), cal = calibrationGrid();
    Objective obj;   // alpha = beta = 1
    std::vector<Row> rows;

    for (int n : LENGTHS) {
        // ---- Stage 0 controls
        rows.push_back(runRandom(n, test));
        SobolConfig same; same.dimX = same.dimY = 1;
        rows.push_back(runDeterministic("0_control", "same_reference(dim1,dim1)", n, same, test));

        // ---- Stage 1 baseline
        rows.push_back(runDeterministic("1_baseline", "sobol_dims_1_2", n, SobolConfig{}, test));

        // ---- Stage 2 direction-number pairs
        for (int i = 1; i <= SobolGenerator::MAX_DIM; i++)
            for (int j = i + 1; j <= SobolGenerator::MAX_DIM; j++) {
                SobolConfig c; c.dimX = i; c.dimY = j;
                rows.push_back(runDeterministic("2_direction", "dims_" + std::to_string(i) + "_" + std::to_string(j), n, c, test));
            }

        // ---- Stage 3 scrambling (dims 1,2)
        ScrambleMode modes[3] = {ScrambleMode::DigitalShift, ScrambleMode::LinearMatrix, ScrambleMode::Owen};
        for (ScrambleMode m : modes)
            rows.push_back(runTrials("3_scramble", std::string("scramble_") + scrambleName(m), n, m, test));

        // ---- Stage 4 adaptive: seed search per scrambler (dims 1,2 fixed)
        for (ScrambleMode m : modes) {
            std::vector<std::vector<SobolConfig>> pools;
            for (int r = 0; r < ADAPT_REPEATS; r++) {
                std::vector<SobolConfig> pool;
                for (int k = 0; k < ADAPT_CANDIDATES; k++) {
                    SobolConfig c; c.scramble = m; c.seed = 100000ull + r * ADAPT_CANDIDATES + k;
                    pool.push_back(c);
                }
                pools.push_back(pool);
            }
            rows.push_back(runAdaptive(std::string("adaptive_seed_") + scrambleName(m), n, pools, cal, test, obj));
        }
        // ---- Stage 4 adaptive: joint search over (dimension pair x digital-shift seed)
        {
            std::vector<std::vector<SobolConfig>> pools;
            for (int r = 0; r < ADAPT_REPEATS; r++) {
                std::vector<SobolConfig> pool;
                for (int i = 1; i <= SobolGenerator::MAX_DIM; i++)
                    for (int j = i + 1; j <= SobolGenerator::MAX_DIM; j++)
                        for (int k = 0; k < 8; k++) {
                            SobolConfig c; c.dimX = i; c.dimY = j; c.scramble = ScrambleMode::DigitalShift;
                            c.seed = 200000ull + r * 1000 + k;
                            pool.push_back(c);
                        }
                pools.push_back(pool);
            }
            rows.push_back(runAdaptive("adaptive_joint_dims+shift", n, pools, cal, test, obj));
        }
    }

    // ---- CSV (every row, including all 45 direction pairs)
    std::string csvPath = outDir + "/summary.csv";
    if (FILE* f = std::fopen(csvPath.c_str(), "w")) {
        std::fprintf(f, "stage,config,N,trials,mean_abs_scc,mean_abs_pearson,mae,mse,emult,hamming,autocorr_lag1,"
                        "star_discrepancy,mae_std,ns_per_point,select_ms\n");
        for (const Row& r : rows)
            std::fprintf(f, "%s,%s,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.4g,%.4g\n",
                         r.stage.c_str(), r.config.c_str(), r.n, r.trials, r.mean.absScc, r.mean.absPearson,
                         r.mean.mae, r.mean.mse, r.mean.emult, r.mean.hamming, r.mean.autocorr, r.mean.disc,
                         r.maeStd, 1.96 * r.maeStd / std::sqrt(static_cast<double>(r.trials)), r.nsPerPoint, r.selectMs);
        std::fclose(f);
        std::printf("wrote %s (%zu rows)\n\n", csvPath.c_str(), rows.size());
    }

    // ---- Stage 2 headline: best / worst pair per N (oracle: ranked on the test grid itself)
    std::printf("=== Stage 2: direction-number pairs (45 pairs from dims 1..10), ranked by MAE ===\n");
    printHeader();
    for (int n : LENGTHS) {
        const Row *best = nullptr, *worst = nullptr, *base = nullptr;
        for (const Row& r : rows) {
            if (r.n != n) continue;
            if (r.stage == "1_baseline") base = &r;
            if (r.stage != "2_direction") continue;
            if (!best || r.mean.mae < best->mean.mae) best = &r;
            if (!worst || r.mean.mae > worst->mean.mae) worst = &r;
        }
        Row b = *best, w = *worst; b.config = "BEST  " + b.config; w.config = "WORST " + w.config;
        printRow(*base); printRow(b); printRow(w);
    }

    // ---- Stage 5: side-by-side multiplication comparison
    std::printf("\n=== Stage 5: stochastic multiplication, 81 (pA,pB) pairs, test grid 0.1..0.9 ===\n");
    printHeader();
    const char* order[] = {"random_independent", "same_reference(dim1,dim1)", "sobol_dims_1_2", "scramble_digital_shift",
                           "scramble_linear_matrix", "scramble_owen", "adaptive_seed_digital_shift",
                           "adaptive_seed_linear_matrix", "adaptive_seed_owen", "adaptive_joint_dims+shift"};
    for (int n : LENGTHS) {
        for (const char* name : order)
            for (const Row& r : rows)
                if (r.n == n && r.config == name) printRow(r);
        std::printf("\n");
    }

    std::printf("Adaptive search cost (ms per selection, %d candidates / %d for joint):\n", ADAPT_CANDIDATES, 45 * 8);
    for (int n : LENGTHS)
        for (const Row& r : rows)
            if (r.n == n && r.stage == "4_adaptive") std::printf("  N=%-4d %-30s %8.3f ms\n", n, r.config.c_str(), r.selectMs);
    return 0;
}
