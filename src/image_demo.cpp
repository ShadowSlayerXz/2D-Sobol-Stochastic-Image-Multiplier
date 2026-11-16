// =====================================================================================
// image_demo.cpp — Phase 6: stochastic RGB image multiplication with the Gap 2 generators
// =====================================================================================
// For each image pair (A, B) and stream length N, every 8-bit channel is multiplied by
//     SA[a] & SB[b]  ->  popcount  ->  round(255 * count / N)
// where SA[v] / SB[v] are the N-bit streams for level v (bit k = 1 iff ref_k < v/255, tested
// exactly in integers) built from one shared reference sequence per operand — the same
// structure as one SNG feeding a whole array of AND gates.
//
// Reference-sequence sources compared: random independent (mt19937), same reference
// (fully correlated control), standard Sobol dims 1,2, digital-shift Sobol, and adaptive
// (best of 32 seeds by J = mean|SCC| + D*). Random and shifted rows are averaged over
// 10 seeds (mean +- std); the saved image uses the first seed.
//
// Usage: image_demo <ppm_dir> <out_dir>      (ppm_dir from python/prepare_dataset.py)
// Writes <out_dir>/image_metrics.csv and <out_dir>/images/<pair>/<method>_N<N>.ppm
// =====================================================================================
#include "seed_selector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct Image { int w = 0, h = 0; std::vector<uint8_t> px; };   // interleaved RGB

bool readPPM(const std::string& path, Image& im) {
    std::ifstream f(path, std::ios::binary);
    std::string magic; int maxv = 0;
    if (!(f >> magic >> im.w >> im.h >> maxv) || magic != "P6" || maxv != 255) return false;
    f.get();
    im.px.resize(static_cast<size_t>(im.w) * im.h * 3);
    f.read(reinterpret_cast<char*>(im.px.data()), im.px.size());
    return static_cast<bool>(f);
}

void writePPM(const std::string& path, const Image& im) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << im.w << " " << im.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(im.px.data()), im.px.size());
}

// Streams for all 256 levels: words[level * W + j] holds bits 64j .. 64j+63.
struct StreamBank {
    int n = 0, words = 0;
    std::vector<uint64_t> bits;
    const uint64_t* at(int level) const { return &bits[static_cast<size_t>(level) * words]; }
};

StreamBank buildBank(const std::vector<uint32_t>& ref) {
    StreamBank b; b.n = static_cast<int>(ref.size()); b.words = (b.n + 63) / 64;
    b.bits.assign(static_cast<size_t>(256) * b.words, 0);
    for (int level = 0; level < 256; level++)
        for (int k = 0; k < b.n; k++)
            // ref/2^32 < level/255  <=>  ref*255 < level*2^32 (exact; level 255 -> all ones)
            if (static_cast<uint64_t>(ref[k]) * 255u < (static_cast<uint64_t>(level) << 32))
                b.bits[static_cast<size_t>(level) * b.words + k / 64] |= 1ull << (k % 64);
    return b;
}

// One generator choice = a pair of reference sequences (stream A's and stream B's).
struct RefPair { std::vector<uint32_t> x, y; };

RefPair fromSobol(const SobolConfig& cfg, int n) {
    RefPair r; SobolGenerator g(cfg);
    for (int i = 0; i < n; i++) { auto p = g.nextRaw(); r.x.push_back(p.first); r.y.push_back(p.second); }
    return r;
}
RefPair fromRandom(int n, uint64_t seed) {
    RefPair r; std::mt19937 ex(static_cast<uint32_t>(1000 + seed)), ey(static_cast<uint32_t>(5000 + seed));
    for (int i = 0; i < n; i++) { r.x.push_back(ex()); r.y.push_back(ey()); }
    return r;
}

std::vector<double> calibrationGrid() { std::vector<double> g; for (int i = 0; i < 10; i++) g.push_back((2 * i + 1) / 20.0); return g; }

struct Method {
    std::string name;
    int seeds;                                         // trials averaged
    std::function<RefPair(int n, int seed)> make;
};

struct Stats { double mae = 0, mse = 0, psnr = 0; };

Stats compare(const Image& got, const Image& exact) {
    double ae = 0, se = 0;
    for (size_t i = 0; i < got.px.size(); i++) {
        double d = double(got.px[i]) - double(exact.px[i]);
        ae += std::fabs(d); se += d * d;
    }
    Stats s; s.mae = ae / got.px.size(); s.mse = se / got.px.size();
    s.psnr = s.mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / s.mse);
    return s;
}

Image exactProduct(const Image& a, const Image& b) {
    Image o = a;
    for (size_t i = 0; i < a.px.size(); i++) o.px[i] = static_cast<uint8_t>((a.px[i] * b.px[i] + 127) / 255);
    return o;
}

// The stochastic multiplier: AND + popcount per channel. Returns the output image.
Image stochasticProduct(const Image& a, const Image& b, const StreamBank& ba, const StreamBank& bb) {
    Image o = a;
    const int n = ba.n, W = ba.words;
    for (size_t i = 0; i < a.px.size(); i++) {
        const uint64_t* sa = ba.at(a.px[i]);
        const uint64_t* sb = bb.at(b.px[i]);
        int count = 0;
        for (int j = 0; j < W; j++) count += __builtin_popcountll(sa[j] & sb[j]);
        o.px[i] = static_cast<uint8_t>((count * 255 + n / 2) / n);   // round(255*count/N)
    }
    return o;
}

bool mkdirp(const std::string& p) { return mkdir(p.c_str(), 0755) == 0 || errno == EEXIST; }

} // namespace

int main(int argc, char** argv) {
    std::string ppmDir = argc > 1 ? argv[1] : "results/ppm";
    std::string outDir = argc > 2 ? argv[2] : "results";
    mkdirp(outDir); mkdirp(outDir + "/images");

    struct Pair { std::string name, a, b; };
    const std::vector<Pair> pairs = {
        {"real500_x_real768crop", "real_image_500x500", "real_image_768x512_crop500"},
        {"real500_x_checkerboard", "real_image_500x500", "checkerboard"},
        {"real768crop_x_solidred", "real_image_768x512_crop500", "solid_red"},
        {"real500_x_white", "real_image_500x500", "white"},
        {"real500_x_black", "real_image_500x500", "black"},
    };
    const int lengths[] = {128, 256, 1024};
    const int SEEDS = 10;

    const std::vector<Method> methods = {
        {"random_independent", SEEDS, [](int n, int s) { return fromRandom(n, s + 1); }},
        {"same_reference", 1, [](int n, int) { SobolConfig c; c.dimX = 1; c.dimY = 1; return fromSobol(c, n); }},
        {"sobol_std", 1, [](int n, int) { return fromSobol(SobolConfig{}, n); }},
        {"digital_shift", SEEDS, [](int n, int s) {
            SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = s + 1; return fromSobol(c, n); }},
        {"adaptive_shift", 1, [](int n, int) {
            std::vector<SobolConfig> cand;
            for (int s = 1; s <= 32; s++) { SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = s; cand.push_back(c); }
            return fromSobol(selectBest(cand, n, calibrationGrid(), Objective{}).cfg, n); }},
    };

    std::ofstream csv(outDir + "/image_metrics.csv");
    csv << "pair,method,N,seeds,mae,mae_std,mse,psnr,psnr_std,ms_per_image\n";
    std::printf("%-26s %-18s %5s %8s %8s %8s %9s\n", "pair", "method", "N", "MAE", "MAEstd", "PSNR", "ms/img");

    for (const Pair& pr : pairs) {
        Image A, B;
        if (!readPPM(ppmDir + "/" + pr.a + ".ppm", A) || !readPPM(ppmDir + "/" + pr.b + ".ppm", B) ||
            A.w != B.w || A.h != B.h) {
            std::fprintf(stderr, "cannot load pair %s (run python/prepare_dataset.py first)\n", pr.name.c_str());
            return 1;
        }
        mkdirp(outDir + "/images/" + pr.name);
        Image exact = exactProduct(A, B);
        writePPM(outDir + "/images/" + pr.name + "/exact.ppm", exact);

        // Conventional baseline timing (scalar integer multiply), best of 5.
        double convMs = 1e9;
        for (int r = 0; r < 5; r++) {
            auto t0 = std::chrono::steady_clock::now();
            Image tmp = exactProduct(A, B);
            auto t1 = std::chrono::steady_clock::now();
            volatile uint8_t keep = tmp.px[0]; (void)keep;
            convMs = std::min(convMs, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        csv << pr.name << ",conventional_int,0,1,0,0,0,99,0," << convMs << "\n";

        for (int n : lengths) {
            for (const Method& m : methods) {
                std::vector<Stats> all; double msBest = 1e9;
                for (int s = 0; s < m.seeds; s++) {
                    RefPair rp = m.make(n, s);
                    StreamBank ba = buildBank(rp.x), bb = buildBank(rp.y);
                    auto t0 = std::chrono::steady_clock::now();
                    Image out = stochasticProduct(A, B, ba, bb);
                    auto t1 = std::chrono::steady_clock::now();
                    msBest = std::min(msBest, std::chrono::duration<double, std::milli>(t1 - t0).count());
                    all.push_back(compare(out, exact));
                    if (s == 0) writePPM(outDir + "/images/" + pr.name + "/" + m.name + "_N" + std::to_string(n) + ".ppm", out);
                }
                double mae = 0, mse = 0, psnr = 0;
                for (auto& s : all) { mae += s.mae; mse += s.mse; psnr += s.psnr; }
                double k = all.size(); mae /= k; mse /= k; psnr /= k;
                double vm = 0, vp = 0;
                for (auto& s : all) { vm += (s.mae - mae) * (s.mae - mae); vp += (s.psnr - psnr) * (s.psnr - psnr); }
                double maeStd = std::sqrt(vm / k), psnrStd = std::sqrt(vp / k);
                csv << pr.name << "," << m.name << "," << n << "," << all.size() << "," << mae << "," << maeStd << ","
                    << mse << "," << psnr << "," << psnrStd << "," << msBest << "\n";
                std::printf("%-26s %-18s %5d %8.3f %8.3f %8.2f %9.2f\n", pr.name.c_str(), m.name.c_str(), n, mae, maeStd, psnr, msBest);
            }
        }
        std::printf("%-26s %-18s %5s %8s %8s %8s %9.2f\n", pr.name.c_str(), "conventional_int", "-", "0", "-", "inf", convMs);
    }
    std::printf("\nWrote %s/image_metrics.csv and images under %s/images/\n", outDir.c_str(), outDir.c_str());
    return 0;
}
