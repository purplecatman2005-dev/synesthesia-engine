#pragma once
// fft.hpp — 反復型 Cooley-Tukey FFT / STFT / ISTFT（外部ライブラリなし）
// STFT: Hann分析窓、hop = N/2
// ISTFT: Hann合成窓 + overlap-add、Σw² で点毎正規化（完全再構成）

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

#include "tensor.hpp"

namespace syn {

constexpr float kPi = 3.14159265358979323846f;

inline void fft_inplace(std::vector<std::complex<float>>& a, bool inverse) {
    const size_t n = a.size();
    assert(n > 0 && (n & (n - 1)) == 0); // 2のべき乗のみ
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = 2.0f * kPi / static_cast<float>(len) * (inverse ? 1.0f : -1.0f);
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                const auto u = a[i + k];
                const auto v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const float inv = 1.0f / static_cast<float>(n);
        for (auto& x : a) x *= inv;
    }
}

inline std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * i / n)); // periodic Hann
    return w;
}

struct StftConfig {
    int fft_size = 1024;              // N（2のべき乗）
    int hop() const { return fft_size / 2; }
    int bins() const { return fft_size / 2; } // bin 0..N/2-1 を保存（Nyquistは捨てる）
};

inline int stft_frame_count(size_t num_samples, const StftConfig& cfg) {
    if (num_samples < static_cast<size_t>(cfg.fft_size)) return 1;
    return 1 + static_cast<int>((num_samples - cfg.fft_size) / cfg.hop());
}

// 音声サンプル列 → UnifiedTensor（frames x bins の複素スペクトル）
inline UnifiedTensor stft(const std::vector<float>& samples, const StftConfig& cfg) {
    const int n = cfg.fft_size, hop = cfg.hop(), bins = cfg.bins();
    const int frames = stft_frame_count(samples.size(), cfg);
    const auto win = hann_window(n);

    UnifiedTensor tensor(static_cast<size_t>(frames) * bins * 2);
    SpectralView spec = tensor.as_spectrum(frames, bins);

    std::vector<std::complex<float>> frame(n);
    for (int f = 0; f < frames; ++f) {
        const size_t off = static_cast<size_t>(f) * hop;
        for (int i = 0; i < n; ++i) {
            const size_t idx = off + i;
            const float s = idx < samples.size() ? samples[idx] : 0.0f;
            frame[i] = {s * win[i], 0.0f};
        }
        fft_inplace(frame, false);
        for (int b = 0; b < bins; ++b) spec.at(f, b) = frame[b];
    }
    return tensor;
}

// UnifiedTensor のスペクトルビュー → 音声サンプル列
inline std::vector<float> istft(SpectralView spec, const StftConfig& cfg) {
    const int n = cfg.fft_size, hop = cfg.hop(), bins = cfg.bins();
    const int frames = spec.frames();
    assert(bins == spec.bins());
    if (frames <= 0) return {};
    const auto win = hann_window(n);

    const size_t out_len = static_cast<size_t>(frames - 1) * hop + n;
    std::vector<float> out(out_len, 0.0f), wsum(out_len, 0.0f);

    std::vector<std::complex<float>> frame(n);
    for (int f = 0; f < frames; ++f) {
        // エルミート対称で全スペクトルを復元（実信号）
        frame[0] = {spec.at(f, 0).real(), 0.0f}; // DCは実数
        for (int b = 1; b < bins; ++b) {
            frame[b] = spec.at(f, b);
            frame[n - b] = std::conj(spec.at(f, b));
        }
        frame[bins] = {0.0f, 0.0f}; // Nyquist
        fft_inplace(frame, true);

        const size_t off = static_cast<size_t>(f) * hop;
        for (int i = 0; i < n; ++i) {
            out[off + i] += frame[i].real() * win[i];
            wsum[off + i] += win[i] * win[i];
        }
    }
    // 除数に下限を設ける: 端はwsum→0で除算がノイズを爆発的に増幅するため、
    // 下限以下の領域は自然なフェードイン/アウトとして扱う
    for (size_t i = 0; i < out_len; ++i)
        out[i] /= std::max(wsum[i], 0.05f);
    return out;
}

} // namespace syn
