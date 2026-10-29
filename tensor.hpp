#pragma once
// tensor.hpp — UnifiedTensor: 単一のfloatバッファを
//   ・画像ビュー   (H x W x C)
//   ・スペクトルビュー (frames x bins の複素数)
// として同時に解釈する。両ビューは同じメモリをエイリアスするため、
// 片方への書き込みはもう片方から即座に見える（= 音と映像の等価データ化の核）。

#include <cassert>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

namespace syn {

class ImageView {
public:
    ImageView(std::shared_ptr<std::vector<float>> buf, int h, int w, int c)
        : buf_(std::move(buf)), h_(h), w_(w), c_(c) {
        assert(static_cast<size_t>(h_) * w_ * c_ <= buf_->size());
    }
    float& at(int y, int x, int ch) { return (*buf_)[(static_cast<size_t>(y) * w_ + x) * c_ + ch]; }
    float at(int y, int x, int ch) const { return (*buf_)[(static_cast<size_t>(y) * w_ + x) * c_ + ch]; }
    int height() const { return h_; }
    int width() const { return w_; }
    int channels() const { return c_; }

private:
    std::shared_ptr<std::vector<float>> buf_;
    int h_, w_, c_;
};

class SpectralView {
public:
    SpectralView(std::shared_ptr<std::vector<float>> buf, int frames, int bins)
        : buf_(std::move(buf)), frames_(frames), bins_(bins) {
        assert(static_cast<size_t>(frames_) * bins_ * 2 <= buf_->size());
    }
    // std::complex<float> は float2つ分のレイアウトが規格で保証されている
    std::complex<float>& at(int frame, int bin) {
        auto* p = reinterpret_cast<std::complex<float>*>(buf_->data());
        return p[static_cast<size_t>(frame) * bins_ + bin];
    }
    const std::complex<float>& at(int frame, int bin) const {
        auto* p = reinterpret_cast<const std::complex<float>*>(buf_->data());
        return p[static_cast<size_t>(frame) * bins_ + bin];
    }
    float magnitude(int frame, int bin) const { return std::abs(at(frame, bin)); }
    int frames() const { return frames_; }
    int bins() const { return bins_; }

private:
    std::shared_ptr<std::vector<float>> buf_;
    int frames_, bins_;
};

class UnifiedTensor {
public:
    explicit UnifiedTensor(size_t n) : buf_(std::make_shared<std::vector<float>>(n, 0.0f)) {}

    size_t size() const { return buf_->size(); }
    float* data() { return buf_->data(); }
    const float* data() const { return buf_->data(); }

    // 画像として見る: pixel(y=frame, x=bin, ch={re,im}) がスペクトルの1セルと一致
    ImageView as_image(int h, int w, int c) {
        assert(static_cast<size_t>(h) * w * c == buf_->size());
        return ImageView(buf_, h, w, c);
    }
    // スペクトルとして見る: frames x bins の複素数配列
    SpectralView as_spectrum(int frames, int bins) {
        assert(static_cast<size_t>(frames) * bins * 2 == buf_->size());
        return SpectralView(buf_, frames, bins);
    }

private:
    std::shared_ptr<std::vector<float>> buf_;
};

} // namespace syn
