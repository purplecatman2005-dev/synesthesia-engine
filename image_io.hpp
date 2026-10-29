#pragma once
// image_io.hpp — 24bit無圧縮BMPの読み書きと基本画像処理（Tensor3、リサイズ、グレースケール）

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace syn {

// 汎用の実体テンソル（NN・画像処理用。UnifiedTensorのビューとは別物）
struct Tensor3 {
    int h = 0, w = 0, c = 0;
    std::vector<float> v;
    Tensor3() = default;
    Tensor3(int h_, int w_, int c_) : h(h_), w(w_), c(c_), v(static_cast<size_t>(h_) * w_ * c_, 0.0f) {}
    float& at(int y, int x, int ch) { return v[(static_cast<size_t>(y) * w + x) * c + ch]; }
    float at(int y, int x, int ch) const { return v[(static_cast<size_t>(y) * w + x) * c + ch]; }
};

// RGB各0..1のTensor3(h,w,3)をBMPへ（y=0が上）
inline void write_bmp(const std::string& path, const Tensor3& img) {
    if (img.c != 3) throw std::runtime_error("write_bmpは3チャンネルのみ対応");
    const int w = img.w, h = img.h;
    const int row_bytes = (w * 3 + 3) & ~3;
    const uint32_t data_size = static_cast<uint32_t>(row_bytes) * h;
    const uint32_t file_size = 54 + data_size;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("BMPを書き込めません: " + path);
    auto put16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto put32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };

    f.write("BM", 2); put32(file_size); put32(0); put32(54);
    put32(40); put32(static_cast<uint32_t>(w)); put32(static_cast<uint32_t>(h));
    put16(1); put16(24); put32(0); put32(data_size);
    put32(2835); put32(2835); put32(0); put32(0);

    std::vector<uint8_t> row(row_bytes, 0);
    for (int y = h - 1; y >= 0; --y) { // BMPはボトムアップ
        for (int x = 0; x < w; ++x) {
            auto q = [&](int ch) {
                return static_cast<uint8_t>(std::clamp(img.at(y, x, ch), 0.0f, 1.0f) * 255.0f + 0.5f);
            };
            row[x * 3 + 0] = q(2); // B
            row[x * 3 + 1] = q(1); // G
            row[x * 3 + 2] = q(0); // R
        }
        f.write(reinterpret_cast<const char*>(row.data()), row_bytes);
    }
}

inline Tensor3 read_bmp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("BMPを開けません: " + path);
    std::vector<uint8_t> hdr(54);
    f.read(reinterpret_cast<char*>(hdr.data()), 54);
    if (!f || hdr[0] != 'B' || hdr[1] != 'M') throw std::runtime_error("BMP形式ではありません: " + path);

    uint32_t data_offset = 0, header_size = 0;
    int32_t w = 0, h = 0;
    uint16_t bpp = 0;
    uint32_t compression = 0;
    std::memcpy(&data_offset, hdr.data() + 10, 4);
    std::memcpy(&header_size, hdr.data() + 14, 4);
    std::memcpy(&w, hdr.data() + 18, 4);
    std::memcpy(&h, hdr.data() + 22, 4);
    std::memcpy(&bpp, hdr.data() + 28, 2);
    std::memcpy(&compression, hdr.data() + 30, 4);
    if (bpp != 24 || compression != 0) throw std::runtime_error("24bit無圧縮BMPのみ対応です: " + path);

    const bool top_down = h < 0;
    const int height = top_down ? -h : h;
    if (w <= 0 || height <= 0 || w > 1 << 16 || height > 1 << 16)
        throw std::runtime_error("BMPの寸法が不正です: " + path);
    const int row_bytes = (w * 3 + 3) & ~3;

    f.seekg(data_offset, std::ios::beg);
    Tensor3 img(height, w, 3);
    std::vector<uint8_t> row(row_bytes);
    for (int i = 0; i < height; ++i) {
        const int y = top_down ? i : height - 1 - i;
        f.read(reinterpret_cast<char*>(row.data()), row_bytes);
        for (int x = 0; x < w; ++x) {
            img.at(y, x, 0) = row[x * 3 + 2] / 255.0f;
            img.at(y, x, 1) = row[x * 3 + 1] / 255.0f;
            img.at(y, x, 2) = row[x * 3 + 0] / 255.0f;
        }
    }
    return img;
}

inline Tensor3 to_grayscale(const Tensor3& img) {
    Tensor3 g(img.h, img.w, 1);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) {
            if (img.c >= 3)
                g.at(y, x, 0) = 0.299f * img.at(y, x, 0) + 0.587f * img.at(y, x, 1) + 0.114f * img.at(y, x, 2);
            else
                g.at(y, x, 0) = img.at(y, x, 0);
        }
    return g;
}

inline Tensor3 resize_bilinear(const Tensor3& src, int nh, int nw) {
    Tensor3 dst(nh, nw, src.c);
    const float sy = src.h > 1 ? static_cast<float>(src.h - 1) / std::max(1, nh - 1) : 0.0f;
    const float sx = src.w > 1 ? static_cast<float>(src.w - 1) / std::max(1, nw - 1) : 0.0f;
    for (int y = 0; y < nh; ++y) {
        const float fy = y * sy;
        const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, src.h - 1);
        const float ty = fy - y0;
        for (int x = 0; x < nw; ++x) {
            const float fx = x * sx;
            const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, src.w - 1);
            const float tx = fx - x0;
            for (int ch = 0; ch < src.c; ++ch) {
                const float top = src.at(y0, x0, ch) * (1 - tx) + src.at(y0, x1, ch) * tx;
                const float bot = src.at(y1, x0, ch) * (1 - tx) + src.at(y1, x1, ch) * tx;
                dst.at(y, x, ch) = top * (1 - ty) + bot * ty;
            }
        }
    }
    return dst;
}

// マグマ風カラーマップ（0..1 → RGB 0..1）
inline void magma_colormap(float t, float& r, float& g, float& b) {
    static const float anchors[5][3] = {
        {0.00f, 0.00f, 0.02f}, {0.32f, 0.07f, 0.49f}, {0.72f, 0.21f, 0.47f},
        {0.99f, 0.54f, 0.38f}, {0.99f, 0.99f, 0.75f}};
    t = std::clamp(t, 0.0f, 1.0f) * 4.0f;
    const int i = std::min(3, static_cast<int>(t));
    const float u = t - i;
    r = anchors[i][0] * (1 - u) + anchors[i + 1][0] * u;
    g = anchors[i][1] * (1 - u) + anchors[i + 1][1] * u;
    b = anchors[i][2] * (1 - u) + anchors[i + 1][2] * u;
}

} // namespace syn
