#pragma once
// audio_io.hpp — PCM16 WAV の読み書き（モノラル書き出し、読み込みはモノ/ステレオ対応）

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace syn {

struct AudioBuffer {
    std::vector<float> samples; // -1.0 .. 1.0, モノラル
    int sample_rate = 44100;
};

inline void write_wav(const std::string& path, const AudioBuffer& audio) {
    const uint16_t channels = 1, bits = 16;
    const uint32_t sr = static_cast<uint32_t>(audio.sample_rate);
    const uint32_t byte_rate = sr * channels * bits / 8;
    const uint16_t block_align = channels * bits / 8;
    const uint32_t data_size = static_cast<uint32_t>(audio.samples.size() * 2);
    const uint32_t riff_size = 36 + data_size;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("WAVを書き込めません: " + path);
    auto put16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto put32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };

    f.write("RIFF", 4); put32(riff_size); f.write("WAVE", 4);
    f.write("fmt ", 4); put32(16); put16(1); put16(channels);
    put32(sr); put32(byte_rate); put16(block_align); put16(bits);
    f.write("data", 4); put32(data_size);

    for (float s : audio.samples) {
        const float c = std::clamp(s, -1.0f, 1.0f);
        const int16_t v = static_cast<int16_t>(c < 0 ? c * 32768.0f : c * 32767.0f);
        f.write(reinterpret_cast<const char*>(&v), 2);
    }
}

inline AudioBuffer read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("WAVを開けません: " + path);

    char riff[4], wave[4];
    uint32_t riff_size = 0;
    f.read(riff, 4);
    f.read(reinterpret_cast<char*>(&riff_size), 4);
    f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0)
        throw std::runtime_error("WAV形式ではありません: " + path);

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    std::vector<int16_t> raw;
    bool got_fmt = false, got_data = false;

    while (f && !(got_fmt && got_data)) {
        char id[4];
        uint32_t size = 0;
        if (!f.read(id, 4)) break;
        if (!f.read(reinterpret_cast<char*>(&size), 4)) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (size < 16) throw std::runtime_error("fmtチャンクが不正です: " + path);
            std::vector<char> buf(size);
            f.read(buf.data(), size);
            std::memcpy(&format, buf.data() + 0, 2);
            std::memcpy(&channels, buf.data() + 2, 2);
            std::memcpy(&sr, buf.data() + 4, 4);
            std::memcpy(&bits, buf.data() + 14, 2);
            got_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            raw.resize(size / 2);
            f.read(reinterpret_cast<char*>(raw.data()), size / 2 * 2);
            got_data = true;
        } else {
            f.seekg(size + (size & 1), std::ios::cur); // チャンクは2バイト境界
        }
    }
    if (!got_fmt || !got_data) throw std::runtime_error("fmt/dataチャンクが見つかりません: " + path);
    if (format != 1 || bits != 16 || channels < 1 || channels > 2)
        throw std::runtime_error("PCM16のモノ/ステレオのみ対応です: " + path);

    AudioBuffer out;
    out.sample_rate = static_cast<int>(sr);
    const size_t n = raw.size() / channels;
    out.samples.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int c = 0; c < channels; ++c) s += raw[i * channels + c] / 32768.0f;
        out.samples[i] = s / channels; // ステレオはモノラルにミックス
    }
    return out;
}

} // namespace syn
