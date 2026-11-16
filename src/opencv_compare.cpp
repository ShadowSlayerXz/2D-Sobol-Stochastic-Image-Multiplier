// =====================================================================================
// opencv_compare.cpp — stochastic (Sobol AND + popcount) vs OpenCV cv::multiply
// =====================================================================================
// Baseline: cv::multiply(A, B, out, 1.0/255) on 8-bit 3-channel images (OpenCV's own vectorised
// code). Stochastic: per channel popcount(SA[a] & SB[b]) with 64-bit words, scalar C++.
// Reports accuracy of the stochastic result against the OpenCV result and timing of
//   (1) cv::multiply,  (2) stochastic AND+popcount kernel only,  (3) kernel + stream-bank build.
// Usage: opencv_compare <image_a.png> <image_b.png> <out.csv>   (images must have equal size)
// =====================================================================================
#include "sobol_generator.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>

namespace {
struct Bank { int n, words; std::vector<uint64_t> bits; const uint64_t* at(int l) const { return &bits[size_t(l) * words]; } };

Bank buildBank(const std::vector<uint32_t>& ref) {
    Bank b; b.n = int(ref.size()); b.words = (b.n + 63) / 64; b.bits.assign(size_t(256) * b.words, 0);
    for (int l = 0; l < 256; l++)
        for (int k = 0; k < b.n; k++)
            if (uint64_t(ref[k]) * 255u < (uint64_t(l) << 32)) b.bits[size_t(l) * b.words + k / 64] |= 1ull << (k % 64);
    return b;
}

void stochMultiply(const cv::Mat& A, const cv::Mat& B, cv::Mat& out, const Bank& ba, const Bank& bb) {
    const uint8_t *pa = A.data, *pb = B.data; uint8_t* po = out.data;
    const size_t total = A.total() * A.channels();
    for (size_t i = 0; i < total; i++) {
        const uint64_t *sa = ba.at(pa[i]), *sb = bb.at(pb[i]);
        int c = 0;
        for (int j = 0; j < ba.words; j++) c += __builtin_popcountll(sa[j] & sb[j]);
        po[i] = uint8_t((c * 255 + ba.n / 2) / ba.n);
    }
}

// ---- Optimisation 1: product LUT -------------------------------------------------------
// The stochastic output depends only on the pair (a,b), so precompute all 65536 results once.
// Built in O(N + 65536) instead of 65536 * N/64 popcounts: bit k is 1 in stream A for every
// level a >= ax[k] (ax[k] = floor(ref*255/2^32)+1, the smallest level whose threshold exceeds ref),
// likewise for B, so count[a][b] = #{k : ax[k] <= a and by[k] <= b} = 2D prefix sum of a histogram.
std::vector<uint8_t> buildLut(const std::vector<uint32_t>& rx, const std::vector<uint32_t>& ry) {
    const int n = int(rx.size());
    std::vector<int> cnt(256 * 256, 0);
    for (int k = 0; k < n; k++) {
        int ax = int((uint64_t(rx[k]) * 255u) >> 32) + 1, by = int((uint64_t(ry[k]) * 255u) >> 32) + 1;
        cnt[ax * 256 + by]++;            // ax,by in 1..255
    }
    std::vector<uint8_t> lut(256 * 256);
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++) {
            if (a) cnt[a * 256 + b] += cnt[(a - 1) * 256 + b];
            if (b) cnt[a * 256 + b] += cnt[a * 256 + b - 1];
            if (a && b) cnt[a * 256 + b] -= cnt[(a - 1) * 256 + b - 1];
            lut[a * 256 + b] = uint8_t((cnt[a * 256 + b] * 255 + n / 2) / n);
        }
    return lut;
}

// Per-pixel work is one table lookup. 4x unrolled so loads overlap.
void lutRange(const uint8_t* pa, const uint8_t* pb, uint8_t* po, size_t lo, size_t hi, const uint8_t* lut) {
    size_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        po[i]     = lut[(pa[i] << 8) | pb[i]];
        po[i + 1] = lut[(pa[i + 1] << 8) | pb[i + 1]];
        po[i + 2] = lut[(pa[i + 2] << 8) | pb[i + 2]];
        po[i + 3] = lut[(pa[i + 3] << 8) | pb[i + 3]];
    }
    for (; i < hi; i++) po[i] = lut[(pa[i] << 8) | pb[i]];
}
void lutMultiply(const cv::Mat& A, const cv::Mat& B, cv::Mat& out, const std::vector<uint8_t>& lut) {
    lutRange(A.data, B.data, out.data, 0, A.total() * A.channels(), lut.data());
}

// ---- Optimisation 2: persistent thread pool (no thread-creation cost per image) ---------
class Pool {
public:
    explicit Pool(unsigned n) : n_(n) {
        for (unsigned t = 0; t < n_; t++) th_.emplace_back([this, t] {
            unsigned seen = 0;
            for (;;) {
                { std::unique_lock<std::mutex> l(m_); cv_.wait(l, [&] { return gen_ != seen || quit_; }); if (quit_) return; seen = gen_; }
                job_(t);
                { std::lock_guard<std::mutex> l(m_); if (--pending_ == 0) done_.notify_one(); }
            }
        });
    }
    ~Pool() { { std::lock_guard<std::mutex> l(m_); quit_ = true; } cv_.notify_all(); for (auto& t : th_) t.join(); }
    template <class F> void run(F f) {     // run f(threadIndex) on every worker and wait
        { std::lock_guard<std::mutex> l(m_); job_ = f; pending_ = n_; gen_++; }
        cv_.notify_all();
        std::unique_lock<std::mutex> l(m_); done_.wait(l, [&] { return pending_ == 0; });
    }
    unsigned size() const { return n_; }
private:
    unsigned n_, gen_ = 0, pending_ = 0; bool quit_ = false;
    std::mutex m_; std::condition_variable cv_, done_; std::function<void(unsigned)> job_;
    std::vector<std::thread> th_;
};

void lutMultiplyMT(const cv::Mat& A, const cv::Mat& B, cv::Mat& out, const std::vector<uint8_t>& lut, Pool& pool) {
    const size_t total = A.total() * A.channels(), T = pool.size();
    pool.run([&](unsigned t) { lutRange(A.data, B.data, out.data, total * t / T, total * (t + 1) / T, lut.data()); });
}
void stochMultiplyMT(const cv::Mat& A, const cv::Mat& B, cv::Mat& out, const Bank& ba, const Bank& bb, Pool& pool) {
    const size_t total = A.total() * A.channels(), T = pool.size();
    pool.run([&](unsigned t) {
        for (size_t i = total * t / T; i < total * (t + 1) / T; i++) {
            const uint64_t *sa = ba.at(A.data[i]), *sb = bb.at(B.data[i]);
            int c = 0;
            for (int j = 0; j < ba.words; j++) c += __builtin_popcountll(sa[j] & sb[j]);
            out.data[i] = uint8_t((c * 255 + ba.n / 2) / ba.n);
        }
    });
}

template <class F> void timeIt(F f, int reps, double& mean, double& mn, double& sd) {
    f();  // warm-up
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::steady_clock::now(); f();
        auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    mean = 0; mn = 1e18; for (double v : t) { mean += v; mn = std::min(mn, v); } mean /= reps;
    double var = 0; for (double v : t) var += (v - mean) * (v - mean); sd = std::sqrt(var / reps);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> refs(const SobolConfig& c, int n) {
    SobolGenerator g(c); std::vector<uint32_t> x, y;
    for (int i = 0; i < n; i++) { auto p = g.nextRaw(); x.push_back(p.first); y.push_back(p.second); }
    return {x, y};
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s a.png b.png out.csv\n", argv[0]); return 2; }
    cv::Mat A = cv::imread(argv[1], cv::IMREAD_COLOR), B = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (A.empty() || B.empty() || A.size() != B.size()) { std::fprintf(stderr, "cannot load / size mismatch\n"); return 1; }
    const int REPS = 30;
    cv::Mat ref(A.size(), A.type());
    double cm, cmin, csd;
    timeIt([&] { cv::multiply(A, B, ref, 1.0 / 255.0); }, REPS, cm, cmin, csd);

    unsigned T = std::max(1u, std::thread::hardware_concurrency());
    Pool pool(T);
    std::ofstream csv(argv[3]);
    csv << "method,variant,N,mae_vs_opencv,psnr_vs_opencv,max_abs_err,bit_exact_vs_scalar,ms_mean,ms_min,ms_std,setup_ms,opencv_ms_mean,ratio_vs_opencv\n";
    std::printf("OpenCV %s, image %dx%d, cv::multiply: mean %.3f ms (min %.3f, sd %.3f), OpenCV threads %d, pool threads %u\n",
                CV_VERSION, A.cols, A.rows, cm, cmin, csd, cv::getNumThreads(), T);
    std::printf("%-14s %-22s %5s %7s %7s %6s %6s %9s %9s %9s\n", "method", "variant", "N", "MAE", "PSNR", "maxErr", "exact", "ms mean", "setup ms", "x OpenCV");

    struct Cfg { const char* name; SobolConfig c; };
    SobolConfig shift; shift.scramble = ScrambleMode::DigitalShift; shift.seed = 1;
    for (const Cfg& cfg : {Cfg{"sobol_std", SobolConfig{}}, Cfg{"digital_shift", shift}})
        for (int n : {128, 256, 1024}) {
            auto r = refs(cfg.c, n);
            auto now = [] { return std::chrono::steady_clock::now(); };
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            auto t0 = now(); Bank ba = buildBank(r.first), bb = buildBank(r.second); double bankMs = ms(t0, now());
            t0 = now(); std::vector<uint8_t> lut = buildLut(r.first, r.second); double lutMs = ms(t0, now());

            cv::Mat scalarOut(A.size(), A.type());
            stochMultiply(A, B, scalarOut, ba, bb);
            struct V { const char* name; std::function<void(cv::Mat&)> fn; double setup; };
            std::vector<V> vs = {
                {"scalar_popcount", [&](cv::Mat& o) { stochMultiply(A, B, o, ba, bb); }, bankMs},
                {"popcount_threads", [&](cv::Mat& o) { stochMultiplyMT(A, B, o, ba, bb, pool); }, bankMs},
                {"lut", [&](cv::Mat& o) { lutMultiply(A, B, o, lut); }, lutMs},
                {"lut_threads", [&](cv::Mat& o) { lutMultiplyMT(A, B, o, lut, pool); }, lutMs},
            };
            for (V& v : vs) {
                cv::Mat out(A.size(), A.type());
                double km, kmin, ksd;
                timeIt([&] { v.fn(out); }, REPS, km, kmin, ksd);
                double ae = 0, se = 0; int mx = 0; bool exact = true;
                for (size_t i = 0; i < A.total() * 3; i++) {
                    int d = int(out.data[i]) - int(ref.data[i]); ae += std::abs(d); se += double(d) * d; mx = std::max(mx, std::abs(d));
                    exact &= out.data[i] == scalarOut.data[i];
                }
                double tot = double(A.total() * 3), mae = ae / tot, mse = se / tot;
                double psnr = mse == 0 ? 99.0 : 10 * std::log10(255.0 * 255.0 / mse);
                csv << cfg.name << "," << v.name << "," << n << "," << mae << "," << psnr << "," << mx << "," << exact << "," << km << ","
                    << kmin << "," << ksd << "," << v.setup << "," << cm << "," << km / cm << "\n";
                std::printf("%-14s %-22s %5d %7.3f %7.2f %6d %6s %9.4f %9.4f %9.2f\n", cfg.name, v.name, n, mae, psnr, mx,
                            exact ? "yes" : "NO", km, v.setup, km / cm);
            }
        }
    return 0;
}
