// sc_image.cpp — see sc_image.hpp.
#include "sc_image.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

bool readPPM(const std::string& path, Image& im) {
    std::ifstream f(path, std::ios::binary);
    std::string magic; int maxv = 0;
    if (!(f >> magic >> im.w >> im.h >> maxv) || magic != "P6" || maxv != 255) return false;
    f.get();
    im.px.resize(static_cast<size_t>(im.w) * im.h * 3);
    f.read(reinterpret_cast<char*>(im.px.data()), static_cast<std::streamsize>(im.px.size()));
    return static_cast<bool>(f);
}

bool writePPM(const std::string& path, const Image& im) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << im.w << " " << im.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(im.px.data()), static_cast<std::streamsize>(im.px.size()));
    return static_cast<bool>(f);
}

Image centerCrop(const Image& im, int w, int h) {
    Image o; o.w = w; o.h = h; o.px.resize(static_cast<size_t>(w) * h * 3);
    int x0 = (im.w - w) / 2, y0 = (im.h - h) / 2;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w * 3; x++)
            o.px[static_cast<size_t>(y) * w * 3 + x] = im.px[static_cast<size_t>(y + y0) * im.w * 3 + x0 * 3 + x];
    return o;
}

StreamBank buildBank(const std::vector<uint32_t>& ref) {
    StreamBank b; b.n = static_cast<int>(ref.size()); b.words = (b.n + 63) / 64;
    b.bits.assign(static_cast<size_t>(256) * b.words, 0);
    // bit k is set for level L iff ref_k*255 < L*2^32 (exact integer test), i.e. iff
    // L >= t_k = floor(ref_k*255 / 2^32) + 1  (always in 1..255). So drop each bit into its
    // first level t_k, then OR every level into the next: O(N + 256*W) instead of O(256*N).
    for (int k = 0; k < b.n; k++) {
        int t = static_cast<int>((static_cast<uint64_t>(ref[k]) * 255u) >> 32) + 1;
        b.bits[static_cast<size_t>(t) * b.words + k / 64] |= 1ull << (k % 64);
    }
    for (int level = 1; level < 256; level++)
        for (int j = 0; j < b.words; j++)
            b.bits[static_cast<size_t>(level) * b.words + j] |= b.bits[static_cast<size_t>(level - 1) * b.words + j];
    return b;
}

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

Image exactProduct(const Image& a, const Image& b) {
    Image o = a;
    for (size_t i = 0; i < a.px.size(); i++) o.px[i] = static_cast<uint8_t>((a.px[i] * b.px[i] + 127) / 255);
    return o;
}

Image stochasticProduct(const Image& a, const Image& b, const StreamBank& ba, const StreamBank& bb,
                        AndPopcountFn kernel) {
    Image o = a;
    const int n = ba.n, W = ba.words;
    for (size_t i = 0; i < a.px.size(); i++) {
        int count = kernel(ba.at(a.px[i]), bb.at(b.px[i]), W);
        o.px[i] = static_cast<uint8_t>((count * 255 + n / 2) / n);   // round(255*count/N)
    }
    return o;
}

void buildProductLutInto(const StreamBank& ba, const StreamBank& bb, AndPopcountFn kernel, uint8_t* lut,
                         ThreadPool* pool) {
    const int n = ba.n, W = ba.words;
    std::vector<uint8_t> levelOf(n + 1);                      // count -> round(255*count/N), no per-entry divide
    for (int c = 0; c <= n; c++) levelOf[c] = static_cast<uint8_t>((c * 255 + n / 2) / n);
    auto row = [&](int a) {
        const uint64_t* sa = ba.at(a);
        for (int b = 0; b < 256; b++) lut[a * 256 + b] = levelOf[kernel(sa, bb.at(b), W)];
    };
    if (pool && pool->size() > 1) pool->run(64, [&](int g) { for (int a = g * 4; a < g * 4 + 4; a++) row(a); });
    else for (int a = 0; a < 256; a++) row(a);
}

void lookupProduct(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t count, const uint8_t* lut,
                   ThreadPool* pool) {
    auto span = [&](size_t lo, size_t hi) {
        const uint8_t* __restrict pa = a; const uint8_t* __restrict pb = b;
        const uint8_t* __restrict tl = lut; uint8_t* __restrict po = out;
        for (size_t i = lo; i < hi; i++) po[i] = tl[(static_cast<size_t>(pa[i]) << 8) | pb[i]];
    };
    const size_t chunk = 1 << 16;                              // 64 KiB of pixels per task
    if (pool && pool->size() > 1 && count > 4 * chunk) {
        int tasks = static_cast<int>((count + chunk - 1) / chunk);
        pool->run(tasks, [&](int t) { span(t * chunk, std::min(count, (t + 1) * chunk)); });
    } else span(0, count);
}

std::vector<uint8_t> buildProductLut(const StreamBank& ba, const StreamBank& bb, AndPopcountFn kernel) {
    std::vector<uint8_t> lut(256 * 256);
    buildProductLutInto(ba, bb, kernel, lut.data(), nullptr);
    return lut;
}

Image stochasticProductLut(const Image& a, const Image& b, const std::vector<uint8_t>& lut) {
    Image o = a;
    for (size_t i = 0; i < a.px.size(); i++) o.px[i] = lut[a.px[i] * 256 + b.px[i]];
    return o;
}

ImageStats compareImages(const Image& got, const Image& exact) {
    double ae = 0, se = 0;
    for (size_t i = 0; i < got.px.size(); i++) {
        double d = double(got.px[i]) - double(exact.px[i]);
        ae += std::fabs(d); se += d * d;
    }
    ImageStats s; s.mae = ae / got.px.size(); s.mse = se / got.px.size();
    s.psnr = s.mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / s.mse);
    return s;
}
