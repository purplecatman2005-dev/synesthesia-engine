// test_fusion.cpp — 成功基準の検証
//   FFT往復 / 正弦波ピーク / STFT-ISTFT再構成 / UnifiedTensorエイリアシング /
//   WAV往復 / BMP往復 / Conv2D恒等カーネル / Attention行和=1

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "audio_io.hpp"
#include "fft.hpp"
#include "image_io.hpp"
#include "nn.hpp"
#include "tensor.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace syn;

static int g_failed = 0;

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) {                                             \
            std::printf("  [PASS] %s\n", msg);                  \
        } else {                                                \
            std::printf("  [FAIL] %s\n", msg);                  \
            ++g_failed;                                         \
        }                                                       \
    } while (0)

static void test_fft_roundtrip() {
    std::printf("FFT往復:\n");
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<std::complex<float>> a(1024), orig;
    for (auto& x : a) x = {d(rng), d(rng)};
    orig = a;
    fft_inplace(a, false);
    fft_inplace(a, true);
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - orig[i]));
    CHECK(err < 1e-4f, "fft→ifft 最大誤差 < 1e-4");
}

static void test_fft_sine_peak() {
    std::printf("正弦波のピーク位置:\n");
    const int n = 1024, k = 37;
    std::vector<std::complex<float>> a(n);
    for (int i = 0; i < n; ++i) a[i] = {std::sin(2.0f * kPi * k * i / n), 0.0f};
    fft_inplace(a, false);
    int peak = 0;
    for (int i = 1; i < n / 2; ++i)
        if (std::abs(a[i]) > std::abs(a[peak])) peak = i;
    CHECK(peak == k, "bin37の正弦波 → ピークがbin37");
}

static void test_stft_istft() {
    std::printf("STFT→ISTFT再構成:\n");
    StftConfig cfg;
    const int len = 44100;
    std::vector<float> sig(len);
    for (int i = 0; i < len; ++i)
        sig[i] = 0.5f * std::sin(2.0f * kPi * 440.0f * i / 44100.0f) +
                 0.3f * std::sin(2.0f * kPi * 1234.5f * i / 44100.0f);

    UnifiedTensor u = stft(sig, cfg);
    SpectralView spec = u.as_spectrum(stft_frame_count(sig.size(), cfg), cfg.bins());
    std::vector<float> rec = istft(spec, cfg);

    float err = 0.0f;
    const int margin = cfg.fft_size; // 端は窓の影響があるため内部のみ検証
    for (int i = margin; i < len - margin; ++i) err = std::max(err, std::abs(rec[i] - sig[i]));
    CHECK(err < 1e-3f, "内部サンプルの最大誤差 < 1e-3");
}

static void test_unified_aliasing() {
    std::printf("UnifiedTensorエイリアシング (等価データの核):\n");
    UnifiedTensor u(4 * 8 * 2); // 4フレーム x 8ビン
    SpectralView spec = u.as_spectrum(4, 8);
    ImageView img = u.as_image(4, 8, 2);

    spec.at(2, 5) = {3.5f, -1.25f};
    CHECK(img.at(2, 5, 0) == 3.5f && img.at(2, 5, 1) == -1.25f,
          "スペクトルへの書き込みが画像ビューから見える");

    img.at(1, 3, 0) = 7.0f;
    img.at(1, 3, 1) = 2.0f;
    CHECK(spec.at(1, 3) == std::complex<float>(7.0f, 2.0f),
          "画像ピクセルの書き込みがスペクトルから見える");

    img.at(2, 5, 0) *= 2.0f;
    img.at(2, 5, 1) *= 2.0f;
    CHECK(std::abs(spec.magnitude(2, 5) - 2.0f * std::abs(std::complex<float>(3.5f, -1.25f))) < 1e-5f,
          "画像側のゲイン操作＝スペクトルのマグニチュード変調");
}

static void test_wav_roundtrip() {
    std::printf("WAV往復:\n");
    AudioBuffer a;
    a.sample_rate = 44100;
    a.samples.resize(1000);
    for (int i = 0; i < 1000; ++i) a.samples[i] = 0.8f * std::sin(0.05f * i);
    write_wav("tmp_test.wav", a);
    AudioBuffer b = read_wav("tmp_test.wav");
    std::remove("tmp_test.wav");

    bool ok = b.sample_rate == 44100 && b.samples.size() == 1000;
    float err = 0.0f;
    if (ok)
        for (int i = 0; i < 1000; ++i) err = std::max(err, std::abs(a.samples[i] - b.samples[i]));
    CHECK(ok && err < 2.0f / 32768.0f, "PCM16量子化誤差の範囲内で一致");
}

static void test_bmp_roundtrip() {
    std::printf("BMP往復:\n");
    Tensor3 img(13, 17, 3); // パディングが発生する奇数幅
    std::mt19937 rng(2);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& v : img.v) v = d(rng) / 255.0f;
    write_bmp("tmp_test.bmp", img);
    Tensor3 back = read_bmp("tmp_test.bmp");
    std::remove("tmp_test.bmp");

    bool ok = back.h == 13 && back.w == 17;
    float err = 0.0f;
    if (ok)
        for (size_t i = 0; i < img.v.size(); ++i) err = std::max(err, std::abs(img.v[i] - back.v[i]));
    CHECK(ok && err < 1.0f / 255.0f, "8bit量子化の範囲内で一致");
}

static void test_conv_identity() {
    std::printf("Conv2D恒等カーネル:\n");
    std::mt19937 rng(3);
    Conv2D conv(3, 1, 1, rng);
    // 中央のみ1の恒等カーネルに上書き
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) conv.weight(0, ky, kx, 0) = (ky == 1 && kx == 1) ? 1.0f : 0.0f;

    Tensor3 in(8, 8, 1);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : in.v) v = d(rng);
    Tensor3 out = conv.forward(in);

    float err = 0.0f;
    for (size_t i = 0; i < in.v.size(); ++i) err = std::max(err, std::abs(out.v[i] - in.v[i]));
    CHECK(out.h == 8 && out.w == 8 && err < 1e-6f, "恒等カーネル → 入力を完全再現");
}

static void test_vit_attention() {
    std::printf("ViT Attention:\n");
    TinyViT vit;
    Tensor3 gray(64, 64, 1);
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    for (auto& v : gray.v) v = d(rng);

    ViTOutput out = vit.forward(gray);
    bool shape_ok = out.tokens.r == 64 && out.tokens.c == TinyViT::kDim &&
                    out.attention.r == 64 && out.attention.c == 64;
    CHECK(shape_ok, "トークン64x32 / Attention 64x64 の形状");

    float row_err = 0.0f;
    for (int i = 0; i < out.attention.r; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < out.attention.c; ++j) sum += out.attention.at(i, j);
        row_err = std::max(row_err, std::abs(sum - 1.0f));
    }
    CHECK(row_err < 1e-4f, "Attention各行の和 = 1 (softmax)");

    Tensor3 sal = TinyViT::saliency_map(out);
    float lo = 1e9f, hi = -1e9f;
    for (float v : sal.v) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    CHECK(sal.h == 8 && sal.w == 8 && lo >= 0.0f && hi <= 1.0f, "顕著性マップ 8x8, 値域[0,1]");
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    test_fft_roundtrip();
    test_fft_sine_peak();
    test_stft_istft();
    test_unified_aliasing();
    test_wav_roundtrip();
    test_bmp_roundtrip();
    test_conv_identity();
    test_vit_attention();

    if (g_failed == 0) {
        std::printf("\n全テストPASS\n");
        return 0;
    }
    std::printf("\n%d件のテストがFAIL\n", g_failed);
    return 1;
}
