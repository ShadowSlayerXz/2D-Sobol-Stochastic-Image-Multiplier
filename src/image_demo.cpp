// =====================================================================================
// image_demo.cpp — stochastic RGB image multiplication with the Gap 2 generators
// =====================================================================================
// For each image pair (A, B) and stream length N, every 8-bit channel is multiplied by
//     SA[a] & SB[b]  ->  popcount  ->  round(255 * count / N)
// (see sc_image.hpp). Reference-sequence sources compared: random independent (mt19937),
// same reference (fully correlated control), standard Sobol dims 1,2, digital-shift Sobol,
// and adaptive (best of 32 seeds by J = mean|SCC| + D*). Random and shifted rows are
// averaged over 10 seeds (mean, std and a 95% CI = 1.96*std/sqrt(trials)); the saved image
// uses the first seed.
//
// The AND+popcount step runs on the best available kernel (AVX2 if the CPU has it). For
// every output the scalar kernel is also run and the two images must be byte-identical.
// A separate timing section measures scalar vs AVX2 stage by stage (warm-up + repetitions,
// mean/min/max/std) and records the machine/compiler/flags in timing.csv.
//
// Usage:
//   image_demo [ppm_dir] [out_dir]                    (ppm_dir from python/prepare_dataset.py)
//   image_demo --batch <dirA> <dirB> [out_dir]        pair the i-th PPM of dirA with the i-th of
//                                                     dirB, centre-cropped to the common size
// Writes <out_dir>/image_metrics.csv, <out_dir>/timing.csv, <out_dir>/images/<pair>/<method>_N<N>.ppm
// =====================================================================================
#include "sc_image.hpp"
#include "seed_selector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#ifdef SC_HAVE_OPENCV
#include <opencv2/core.hpp>
#endif
#ifndef SC_BUILD_FLAGS
#define SC_BUILD_FLAGS "unknown"
#endif
#ifndef SC_BUILD_TYPE
#define SC_BUILD_TYPE "unknown"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<double> calibrationGrid() { std::vector<double> g; for (int i = 0; i < 10; i++) g.push_back((2 * i + 1) / 20.0); return g; }

struct Method {
    std::string name;
    int seeds;                                         // trials averaged
    std::function<RefPair(int n, int seed)> make;
};

std::vector<Method> makeMethods(int seeds) {
    return {
        {"random_independent", seeds, [](int n, int s) { return fromRandom(n, s + 1); }},
        {"same_reference", 1, [](int n, int) { SobolConfig c; c.dimX = 1; c.dimY = 1; return fromSobol(c, n); }},
        {"sobol_std", 1, [](int n, int) { return fromSobol(SobolConfig{}, n); }},
        {"digital_shift", seeds, [](int n, int s) {
            SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = s + 1; return fromSobol(c, n); }},
        {"adaptive_shift", 1, [](int n, int) {
            std::vector<SobolConfig> cand;
            for (int s = 1; s <= 32; s++) { SobolConfig c; c.scramble = ScrambleMode::DigitalShift; c.seed = s; cand.push_back(c); }
            return fromSobol(selectBest(cand, n, calibrationGrid(), Objective{}).cfg, n); }},
    };
}

// ---- machine description for the timing CSV ---------------------------------------------
std::string cpuModel() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned r[12] = {};
    unsigned maxExt = __get_cpuid_max(0x80000000u, nullptr);
    if (maxExt >= 0x80000004u) {
        for (unsigned i = 0; i < 3; i++) __get_cpuid(0x80000002u + i, &r[i * 4], &r[i * 4 + 1], &r[i * 4 + 2], &r[i * 4 + 3]);
        std::string s(reinterpret_cast<char*>(r), 48);
        s = s.c_str();
        size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
        return a == std::string::npos ? "unknown" : s.substr(a, b - a + 1);
    }
#endif
    return "unknown";
}
std::string osName() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown";
#endif
}
std::string csvSafe(std::string s) { std::replace(s.begin(), s.end(), ',', ';'); return s; }

// ---- timing statistics --------------------------------------------------------------------
struct TimeStats { double mean = 0, min = 0, max = 0, std = 0; int reps = 0; };
TimeStats summarize(const std::vector<double>& v) {
    TimeStats t; t.reps = static_cast<int>(v.size());
    t.min = *std::min_element(v.begin(), v.end()); t.max = *std::max_element(v.begin(), v.end());
    for (double x : v) t.mean += x;
    t.mean /= v.size();
    double var = 0; for (double x : v) var += (x - t.mean) * (x - t.mean);
    t.std = v.size() > 1 ? std::sqrt(var / (v.size() - 1)) : 0;   // sample std-dev
    return t;
}
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

constexpr int WARMUPS = 3, REPS = 15;

// Time Sobol generation, stream encoding/packing, AND+popcount and the full pipeline for
// one kernel path. The N-bit pipeline is the standard (unscrambled) 2D Sobol multiplier.
void timePath(const std::string& pair, const Image& A, const Image& B, int n, KernelPath requested,
              std::ofstream& csv, const std::string& envPrefix) {
    KernelPath ran = effectivePath(requested);
    AndPopcountFn kernel = kernelFor(requested);
    std::vector<double> tGen, tEnc, tAnd, tFull, tLut, tFullLut;
    volatile uint8_t sink = 0;
    for (int r = -WARMUPS; r < REPS; r++) {
        auto t0 = Clock::now();
        RefPair rp = fromSobol(SobolConfig{}, n);
        double g = msSince(t0);
        t0 = Clock::now();
        StreamBank ba = buildBank(rp.x), bb = buildBank(rp.y);
        double e = msSince(t0);
        t0 = Clock::now();
        Image out = stochasticProduct(A, B, ba, bb, kernel);
        double a = msSince(t0);
        sink = sink ^ out.px[out.px.size() / 2];
        t0 = Clock::now();
        Image outLut = stochasticProductLut(A, B, buildProductLut(ba, bb, kernel));
        double l = msSince(t0);
        if (outLut.px != out.px) std::fprintf(stderr, "LUT MISMATCH\n");
        if (r >= 0) { tGen.push_back(g); tEnc.push_back(e); tAnd.push_back(a); tFull.push_back(g + e + a);
                      tLut.push_back(l); tFullLut.push_back(g + e + l); }
    }
    (void)sink;
    auto emit = [&](const char* stage, const std::vector<double>& v) {
        TimeStats t = summarize(v);
        csv << envPrefix << pair << "," << n << "," << pathName(requested) << "," << pathName(ran) << "," << stage << ","
            << t.reps << "," << t.mean << "," << t.min << "," << t.max << "," << t.std << "\n";
        std::printf("  N=%-5d %-6s(ran %-6s) %-14s mean %9.3f  min %9.3f  max %9.3f  std %8.3f ms\n",
                    n, pathName(requested), pathName(ran), stage, t.mean, t.min, t.max, t.std);
    };
    emit("sobol_generate", tGen); emit("encode_pack", tEnc); emit("and_popcount", tAnd); emit("full_pipeline", tFull);
    emit("lut_product", tLut); emit("full_pipeline_lut", tFullLut);
}

#ifdef SC_HAVE_OPENCV
void timeOpenCV(const std::string& pair, const Image& A, const Image& B, const Image& exact,
                std::ofstream& csv, const std::string& envPrefix) {
    cv::Mat ma(A.h, A.w, CV_8UC3, const_cast<uint8_t*>(A.px.data()));
    cv::Mat mb(B.h, B.w, CV_8UC3, const_cast<uint8_t*>(B.px.data()));
    cv::Mat out; std::vector<double> t;
    for (int r = -WARMUPS; r < REPS; r++) {
        auto t0 = Clock::now(); cv::multiply(ma, mb, out, 1.0 / 255.0); double ms = msSince(t0);
        if (r >= 0) t.push_back(ms);
    }
    Image o = A; std::copy(out.data, out.data + o.px.size(), o.px.begin());
    ImageStats s = compareImages(o, exact);
    TimeStats ts = summarize(t);
    csv << envPrefix << pair << ",0,opencv,opencv,cv_multiply," << ts.reps << "," << ts.mean << "," << ts.min << "," << ts.max << "," << ts.std << "\n";
    std::printf("  OpenCV cv::multiply: mean %.3f ms (MAE vs exact %.3f)\n", ts.mean, s.mae);
}
#endif

// ---- accuracy over methods / lengths -------------------------------------------------------
struct PairResult { bool ok = true; };

bool runPair(const std::string& name, const Image& A, const Image& B, const std::string& outDir,
             std::ofstream& csv, std::ofstream& timingCsv, const std::string& envPrefix,
             const std::vector<Method>& methods, bool doTiming) {
    const int lengths[] = {128, 256, 1024};
    AndPopcountFn fast = kernelFor(KernelPath::AVX2), scalar = kernelFor(KernelPath::Scalar);
    fs::create_directories(outDir + "/images/" + name);
    Image exact = exactProduct(A, B);
    writePPM(outDir + "/images/" + name + "/exact.ppm", exact);
    bool allIdentical = true;

    double convMs = 1e9;                                   // conventional scalar integer multiply, best of 5
    for (int r = 0; r < 5; r++) {
        auto t0 = Clock::now(); Image tmp = exactProduct(A, B); double ms = msSince(t0);
        volatile uint8_t keep = tmp.px[0]; (void)keep; convMs = std::min(convMs, ms);
    }
    csv << name << ",conventional_int,0,1,0,0,0,0,99,0,0," << convMs << ",1\n";

    for (int n : lengths) {
        for (const Method& m : methods) {
            std::vector<ImageStats> all; double msBest = 1e9; bool identical = true;
            for (int s = 0; s < m.seeds; s++) {
                RefPair rp = m.make(n, s);
                StreamBank ba = buildBank(rp.x), bb = buildBank(rp.y);
                auto t0 = Clock::now();
                Image out = stochasticProduct(A, B, ba, bb, fast);
                msBest = std::min(msBest, msSince(t0));
                if (stochasticProduct(A, B, ba, bb, scalar).px != out.px) identical = false;
                all.push_back(compareImages(out, exact));
                if (s == 0) writePPM(outDir + "/images/" + name + "/" + m.name + "_N" + std::to_string(n) + ".ppm", out);
            }
            allIdentical = allIdentical && identical;
            double k = static_cast<double>(all.size()), mae = 0, mse = 0, psnr = 0;
            for (auto& s : all) { mae += s.mae; mse += s.mse; psnr += s.psnr; }
            mae /= k; mse /= k; psnr /= k;
            double vm = 0, vp = 0;
            for (auto& s : all) { vm += (s.mae - mae) * (s.mae - mae); vp += (s.psnr - psnr) * (s.psnr - psnr); }
            double maeStd = std::sqrt(vm / k), psnrStd = std::sqrt(vp / k);
            double maeCi = 1.96 * maeStd / std::sqrt(k);   // 95% CI half-width of the mean (0 for single-trial rows)
            csv << name << "," << m.name << "," << n << "," << all.size() << "," << mae << "," << maeStd << "," << maeCi << ","
                << mse << "," << psnr << "," << psnrStd << "," << 1.96 * psnrStd / std::sqrt(k) << "," << msBest << "," << (identical ? 1 : 0) << "\n";
            std::printf("%-26s %-18s %5d %8.3f +-%6.3f %8.2f %9.2f %s\n", name.c_str(), m.name.c_str(), n, mae, maeCi, psnr, msBest,
                        identical ? "" : "  SCALAR/AVX2 MISMATCH");
        }
    }
    std::printf("%-26s %-18s %5s %8s          %8s %9.2f\n", name.c_str(), "conventional_int", "-", "0", "inf", convMs);

    if (doTiming) {
        std::printf("Timing %s (%d warm-up + %d timed reps per row):\n", name.c_str(), WARMUPS, REPS);
        for (int n : lengths) {
            timePath(name, A, B, n, KernelPath::Scalar, timingCsv, envPrefix);
            timePath(name, A, B, n, KernelPath::AVX2, timingCsv, envPrefix);
        }
#ifdef SC_HAVE_OPENCV
        timeOpenCV(name, A, B, exact, timingCsv, envPrefix);
#endif
    }
    return allIdentical;
}

std::vector<std::string> listPPM(const std::string& dir) {
    std::vector<std::string> v;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".ppm") v.push_back(e.path().string());
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

int main(int argc, char** argv) {
    bool batch = argc > 1 && std::string(argv[1]) == "--batch";
    std::string ppmDir, dirA, dirB, outDir;
    if (batch) {
        if (argc < 4) { std::fprintf(stderr, "usage: image_demo --batch <dirA> <dirB> [out_dir]\n"); return 2; }
        dirA = argv[2]; dirB = argv[3]; outDir = argc > 4 ? argv[4] : "results/batch";
    } else {
        ppmDir = argc > 1 ? argv[1] : "results/ppm"; outDir = argc > 2 ? argv[2] : "results";
    }
    fs::create_directories(outDir);

    // Environment recorded on every timing row.
    std::string envPrefix = csvSafe(cpuModel()) + "," + csvSafe(
#if defined(__VERSION__)
        std::string("gcc/clang ") + __VERSION__
#else
        std::string("unknown")
#endif
        ) + "," + csvSafe(std::string(SC_BUILD_FLAGS)) + "," + SC_BUILD_TYPE + "," + osName() + "," +
        (avx2Compiled() ? "1" : "0") + "," + (avx2Supported() ? "1" : "0") + ",";
    std::printf("CPU: %s | OS: %s | build: %s | AVX2 compiled=%d supported=%d -> fast kernel = %s\n",
                cpuModel().c_str(), osName().c_str(), SC_BUILD_TYPE, avx2Compiled(), avx2Supported(), pathName(bestPath()));

    std::ofstream csv(outDir + "/image_metrics.csv"), timingCsv(outDir + "/timing.csv");
    csv << "pair,method,N,seeds,mae,mae_std,mae_ci95,mse,psnr,psnr_std,psnr_ci95,ms_per_image,scalar_avx2_identical\n";
    timingCsv << "cpu,compiler,flags,build_type,os,avx2_compiled,avx2_supported,pair,N,path_requested,path_ran,stage,reps,mean_ms,min_ms,max_ms,std_ms\n";
    std::printf("%-26s %-18s %5s %8s %7s %8s %9s\n", "pair", "method", "N", "MAE", "+-CI95", "PSNR", "ms/img");
    const std::vector<Method> methods = makeMethods(10);
    bool ok = true;

    if (!batch) {
        struct Pair { std::string name, a, b; };
        const std::vector<Pair> pairs = {
            {"real500_x_real768crop", "real_image_500x500", "real_image_768x512_crop500"},
            {"real500_x_checkerboard", "real_image_500x500", "checkerboard"},
            {"real768crop_x_solidred", "real_image_768x512_crop500", "solid_red"},
            {"real500_x_white", "real_image_500x500", "white"},
            {"real500_x_black", "real_image_500x500", "black"},
        };
        bool first = true;
        for (const Pair& pr : pairs) {
            Image A, B;
            if (!readPPM(ppmDir + "/" + pr.a + ".ppm", A) || !readPPM(ppmDir + "/" + pr.b + ".ppm", B) || A.w != B.w || A.h != B.h) {
                std::fprintf(stderr, "cannot load pair %s (run python/prepare_dataset.py first)\n", pr.name.c_str());
                return 1;
            }
            ok &= runPair(pr.name, A, B, outDir, csv, timingCsv, envPrefix, methods, first);   // timing on the first (real) pair
            first = false;
        }
    } else {
        std::vector<std::string> fa = listPPM(dirA), fb = listPPM(dirB);
        size_t count = std::min(fa.size(), fb.size());
        if (count == 0) { std::fprintf(stderr, "no .ppm files found in both directories\n"); return 1; }
        for (size_t i = 0; i < count; i++) {
            Image A, B;
            if (!readPPM(fa[i], A) || !readPPM(fb[i], B)) { std::fprintf(stderr, "cannot read %s / %s\n", fa[i].c_str(), fb[i].c_str()); return 1; }
            int w = std::min(A.w, B.w), h = std::min(A.h, B.h);
            A = centerCrop(A, w, h); B = centerCrop(B, w, h);
            std::string name = fs::path(fa[i]).stem().string() + "_x_" + fs::path(fb[i]).stem().string();
            std::printf("--- batch pair %zu/%zu: %s (%dx%d)\n", i + 1, count, name.c_str(), w, h);
            ok &= runPair(name, A, B, outDir, csv, timingCsv, envPrefix, methods, i == 0);
        }
    }
    std::printf("\nScalar and AVX2 outputs %s for every method/length.\n", ok ? "were byte-identical" : "DIFFERED (BUG)");
    std::printf("Wrote %s/image_metrics.csv, %s/timing.csv and images under %s/images/\n", outDir.c_str(), outDir.c_str(), outDir.c_str());
    return ok ? 0 : 1;
}
