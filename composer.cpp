// composer.cpp — 音響×視覚融合コンポーザー CLI
//
//   composer.exe see  <in.wav> <out.bmp>              音を「見る」  : WAV→STFT→CNN特徴→画像
//   composer.exe hear <in.bmp> <out.wav>              映像を「聴く」: 画像→スペクトル解釈→ISTFT→WAV
//   composer.exe fuse <in.wav> <in.bmp> <out.wav> <out.bmp>
//                                                     相互変調: ViT AttentionがUnifiedTensorの
//                                                     画像ビュー経由で音のスペクトルを直接書き換える
//   composer.exe demo                                 内蔵の和音+図形で全モードを実行
//
// 核となる仕組み: UnifiedTensor が1本のfloatバッファを
// 「画像 (frames x bins x 2ch)」と「複素スペクトル (frames x bins)」の
// 両方として同時に解釈する。画像ビューへの書き込み＝音の変化。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

constexpr int kSampleRate = 44100;

void normalize_peak(std::vector<float>& s, float peak = 0.85f) {
    float mx = 0.0f;
    for (float v : s) mx = std::max(mx, std::abs(v));
    if (mx > 1e-9f) {
        const float g = peak / mx;
        for (float& v : s) v *= g;
    }
}

// スペクトルビュー → 正規化した対数マグニチュード画像 (h=bins, w=frames, 低域が下)
Tensor3 spectrogram_gray(const SpectralView& spec) {
    const int frames = spec.frames(), bins = spec.bins();
    Tensor3 g(bins, frames, 1);
    float mx = 1e-9f;
    for (int f = 0; f < frames; ++f)
        for (int b = 0; b < bins; ++b) {
            const float m = std::log1p(spec.magnitude(f, b) * 10.0f);
            g.at(bins - 1 - b, f, 0) = m;
            mx = std::max(mx, m);
        }
    for (auto& v : g.v) v /= mx;
    return g;
}

// ---------- see: 音を「見る」 ----------
void mode_see(const std::string& in_wav, const std::string& out_bmp) {
    const AudioBuffer audio = read_wav(in_wav);
    std::printf("[see] %s: %zuサンプル @%dHz\n", in_wav.c_str(), audio.samples.size(), audio.sample_rate);

    StftConfig cfg;
    UnifiedTensor unified = stft(audio.samples, cfg);
    SpectralView spec = unified.as_spectrum(stft_frame_count(audio.samples.size(), cfg), cfg.bins());
    std::printf("[see] STFT: %dフレーム x %dビン (共有バッファ %zu floats)\n",
                spec.frames(), spec.bins(), unified.size());

    Tensor3 gray = spectrogram_gray(spec);

    // CNNで音のテクスチャ特徴を抽出し、色相変調として重ねる
    TinyCNN cnn;
    Tensor3 features = cnn.forward(resize_bilinear(gray, 128, 128));
    std::printf("[see] CNN特徴マップ: %dx%dx%d\n", features.h, features.w, features.c);

    Tensor3 tint = resize_bilinear(features, gray.h, gray.w);
    float fmax[3] = {1e-9f, 1e-9f, 1e-9f};
    for (int y = 0; y < tint.h; ++y)
        for (int x = 0; x < tint.w; ++x)
            for (int ch = 0; ch < 3; ++ch) fmax[ch] = std::max(fmax[ch], tint.at(y, x, ch));

    Tensor3 img(gray.h, gray.w, 3);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) {
            const float m = gray.at(y, x, 0);
            float r, g, b;
            magma_colormap(m, r, g, b);
            img.at(y, x, 0) = 0.75f * r + 0.25f * (tint.at(y, x, 0) / fmax[0]) * m;
            img.at(y, x, 1) = 0.75f * g + 0.25f * (tint.at(y, x, 1) / fmax[1]) * m;
            img.at(y, x, 2) = 0.75f * b + 0.25f * (tint.at(y, x, 2) / fmax[2]) * m;
        }
    write_bmp(out_bmp, img);
    std::printf("[see] 出力: %s (%dx%d)\n", out_bmp.c_str(), img.w, img.h);
}

// ---------- hear: 映像を「聴く」 ----------
void mode_hear(const std::string& in_bmp, const std::string& out_wav) {
    const Tensor3 src = read_bmp(in_bmp);
    std::printf("[hear] %s: %dx%d\n", in_bmp.c_str(), src.w, src.h);

    StftConfig cfg;
    const int bins = cfg.bins();
    const int frames = std::clamp(src.w, 64, 1024);
    const Tensor3 gray = resize_bilinear(to_grayscale(src), bins, frames);

    // 明るさ→マグニチュード、位相はビン周波数に沿って累積（コヒーレントな音色になる）
    // 初期位相はランダム: 全ビン同位相だとインパルス列（クリック音）に退化するため
    UnifiedTensor unified(static_cast<size_t>(frames) * bins * 2);
    SpectralView spec = unified.as_spectrum(frames, bins);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> rand_phase(0.0f, 2.0f * kPi);
    std::vector<float> phase(bins);
    for (auto& p : phase) p = rand_phase(rng);
    for (int f = 0; f < frames; ++f) {
        for (int b = 0; b < bins; ++b) {
            const float lum = gray.at(bins - 1 - b, f, 0); // 画像の上=高音
            const float mag = 40.0f * std::pow(lum, 2.5f);
            spec.at(f, b) = std::polar(mag, phase[b]);
        }
        for (int b = 0; b < bins; ++b)
            phase[b] += 2.0f * kPi * b * cfg.hop() / cfg.fft_size;
    }

    std::vector<float> samples = istft(spec, cfg);
    normalize_peak(samples);
    write_wav(out_wav, {samples, kSampleRate});
    std::printf("[hear] 出力: %s (%.1f秒)\n", out_wav.c_str(),
                static_cast<double>(samples.size()) / kSampleRate);
}

// ---------- fuse: 同一メモリ上で相互変調 ----------
void mode_fuse(const std::string& in_wav, const std::string& in_bmp,
               const std::string& out_wav, const std::string& out_bmp) {
    const AudioBuffer audio = read_wav(in_wav);
    const Tensor3 image = read_bmp(in_bmp);
    std::printf("[fuse] 音: %zuサンプル / 画像: %dx%d\n", audio.samples.size(), image.w, image.h);

    StftConfig cfg;
    UnifiedTensor unified = stft(audio.samples, cfg);
    const int frames = stft_frame_count(audio.samples.size(), cfg);
    const int bins = cfg.bins();

    // ViTのAttention（各パッチの被注目度）→ 顕著性マップ
    TinyViT vit;
    const ViTOutput vout = vit.forward(resize_bilinear(to_grayscale(image), 64, 64));
    Tensor3 saliency = resize_bilinear(TinyViT::saliency_map(vout), bins, frames);
    std::printf("[fuse] ViT: %dトークン, Attention %dx%d → 顕著性マップ\n",
                vout.grid_h * vout.grid_w, vout.attention.r, vout.attention.c);

    // ここが融合の核: 同じメモリを「画像」として開き、ピクセル操作で音のスペクトルを変調する。
    // iv.at(f,b,{0,1}) は spec.at(f,b) の実部・虚部そのもの。
    ImageView iv = unified.as_image(frames, bins, 2);
    for (int f = 0; f < frames; ++f)
        for (int b = 0; b < bins; ++b) {
            const float s = saliency.at(bins - 1 - b, f, 0);
            const float gain = 0.25f + 1.75f * s; // 画像の注目領域ほど音が強く鳴る
            iv.at(f, b, 0) *= gain;
            iv.at(f, b, 1) *= gain;
        }

    // 同じバッファをスペクトルとして読み直して音へ
    SpectralView spec = unified.as_spectrum(frames, bins);
    std::vector<float> samples = istft(spec, cfg);
    normalize_peak(samples);
    write_wav(out_wav, {samples, audio.sample_rate});

    // 変調後のスペクトログラムに元画像をブレンドして視覚側の出力に
    const Tensor3 gray = spectrogram_gray(spec);
    const Tensor3 img_resized = resize_bilinear(image, gray.h, gray.w);
    Tensor3 img(gray.h, gray.w, 3);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) {
            float r, g, b;
            magma_colormap(gray.at(y, x, 0), r, g, b);
            img.at(y, x, 0) = 0.6f * r + 0.4f * img_resized.at(y, x, 0);
            img.at(y, x, 1) = 0.6f * g + 0.4f * img_resized.at(y, x, 1);
            img.at(y, x, 2) = 0.6f * b + 0.4f * img_resized.at(y, x, 2);
        }
    write_bmp(out_bmp, img);
    std::printf("[fuse] 出力: %s / %s\n", out_wav.c_str(), out_bmp.c_str());
}

// ---------- demo素材の生成 ----------
AudioBuffer make_demo_audio() {
    const float notes[] = {220.0f, 277.18f, 329.63f, 440.0f}; // A3, C#4, E4, A4
    const float note_len = 0.375f;
    const int total = static_cast<int>(3.0f * kSampleRate);
    std::vector<float> s(total, 0.0f);
    for (int i = 0; i < total; ++i) {
        const float t = static_cast<float>(i) / kSampleRate;
        const int idx = static_cast<int>(t / note_len) % 4;
        const float nt = std::fmod(t, note_len);
        const float env = std::exp(-4.0f * nt);
        const float f0 = notes[idx];
        s[i] = env * (0.6f * std::sin(2.0f * kPi * f0 * t) +
                      0.25f * std::sin(2.0f * kPi * f0 * 2.0f * t) +
                      0.1f * std::sin(2.0f * kPi * f0 * 3.0f * t));
    }
    normalize_peak(s);
    return {s, kSampleRate};
}

Tensor3 make_demo_image() {
    const int n = 512;
    Tensor3 img(n, n, 3);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            const float dx = (x - n / 2) / static_cast<float>(n / 2);
            const float dy = (y - n / 2) / static_cast<float>(n / 2);
            const float r = std::sqrt(dx * dx + dy * dy);
            const float rings = 0.5f + 0.5f * std::sin(r * 24.0f);           // 同心円
            const float bands = 0.5f + 0.5f * std::sin((dx + dy) * 10.0f);   // 斜めバンド
            const float glow = std::max(0.0f, 1.0f - r);                     // 中心の光
            img.at(y, x, 0) = std::clamp(0.7f * glow + 0.3f * rings, 0.0f, 1.0f);
            img.at(y, x, 1) = std::clamp(0.4f * glow + 0.4f * bands, 0.0f, 1.0f);
            img.at(y, x, 2) = std::clamp(0.5f * rings * bands + 0.3f * glow, 0.0f, 1.0f);
        }
    return img;
}

void mode_demo() {
    std::printf("=== demo: 素材生成 ===\n");
    write_wav("demo_audio.wav", make_demo_audio());
    write_bmp("demo_image.bmp", make_demo_image());
    std::printf("demo_audio.wav (アルペジオ3秒) / demo_image.bmp (512x512) を生成\n\n");

    std::printf("=== 音を「見る」 ===\n");
    mode_see("demo_audio.wav", "demo_see.bmp");
    std::printf("\n=== 映像を「聴く」 ===\n");
    mode_hear("demo_image.bmp", "demo_hear.wav");
    std::printf("\n=== 相互変調 (fuse) ===\n");
    mode_fuse("demo_audio.wav", "demo_image.bmp", "demo_fuse.wav", "demo_fuse.bmp");
}

void usage() {
    std::printf(
        "音響×視覚融合コンポーザー\n"
        "  composer.exe demo\n"
        "  composer.exe see  <in.wav> <out.bmp>\n"
        "  composer.exe hear <in.bmp> <out.wav>\n"
        "  composer.exe fuse <in.wav> <in.bmp> <out.wav> <out.bmp>\n");
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        const std::string mode = argc >= 2 ? argv[1] : "";
        if (mode == "demo" && argc == 2) {
            mode_demo();
        } else if (mode == "see" && argc == 4) {
            mode_see(argv[2], argv[3]);
        } else if (mode == "hear" && argc == 4) {
            mode_hear(argv[2], argv[3]);
        } else if (mode == "fuse" && argc == 6) {
            mode_fuse(argv[2], argv[3], argv[4], argv[5]);
        } else {
            usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "エラー: %s\n", e.what());
        return 1;
    }
    return 0;
}
