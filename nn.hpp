#pragma once
// nn.hpp — フルスクラッチ推論NN
//   TinyCNN : Conv2D→ReLU→MaxPool ×2 の特徴抽出器（音の可視化に使用）
//   TinyViT : 8x8パッチ埋め込み + Multi-Head Self-Attention 1ブロック
//             （画像の顕著性マップ＝Attention受信量を音の変調に使用）
// 重みはシード固定のHe初期化（未学習の決定的特徴写像として動作）

#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "image_io.hpp" // Tensor3

namespace syn {

// ---------- 行列 ----------
struct Mat {
    int r = 0, c = 0;
    std::vector<float> v;
    Mat() = default;
    Mat(int r_, int c_) : r(r_), c(c_), v(static_cast<size_t>(r_) * c_, 0.0f) {}
    float& at(int i, int j) { return v[static_cast<size_t>(i) * c + j]; }
    float at(int i, int j) const { return v[static_cast<size_t>(i) * c + j]; }
};

inline Mat matmul(const Mat& a, const Mat& b) {
    assert(a.c == b.r);
    Mat out(a.r, b.c);
    for (int i = 0; i < a.r; ++i)
        for (int k = 0; k < a.c; ++k) {
            const float av = a.at(i, k);
            if (av == 0.0f) continue;
            for (int j = 0; j < b.c; ++j) out.at(i, j) += av * b.at(k, j);
        }
    return out;
}

inline void he_init(std::vector<float>& w, int fan_in, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
    for (auto& x : w) x = dist(rng);
}

// ---------- CNN ----------
class Conv2D {
public:
    Conv2D(int k, int cin, int cout, std::mt19937& rng)
        : k_(k), cin_(cin), cout_(cout),
          w_(static_cast<size_t>(cout) * k * k * cin), b_(cout, 0.0f) {
        he_init(w_, k * k * cin, rng);
    }
    // stride 1, same padding
    Tensor3 forward(const Tensor3& in) const {
        assert(in.c == cin_);
        const int pad = k_ / 2;
        Tensor3 out(in.h, in.w, cout_);
        for (int y = 0; y < in.h; ++y)
            for (int x = 0; x < in.w; ++x)
                for (int co = 0; co < cout_; ++co) {
                    float acc = b_[co];
                    for (int ky = 0; ky < k_; ++ky) {
                        const int iy = y + ky - pad;
                        if (iy < 0 || iy >= in.h) continue;
                        for (int kx = 0; kx < k_; ++kx) {
                            const int ix = x + kx - pad;
                            if (ix < 0 || ix >= in.w) continue;
                            for (int ci = 0; ci < cin_; ++ci)
                                acc += in.at(iy, ix, ci) * weight(co, ky, kx, ci);
                        }
                    }
                    out.at(y, x, co) = acc;
                }
        return out;
    }
    float& weight(int co, int ky, int kx, int ci) {
        return w_[((static_cast<size_t>(co) * k_ + ky) * k_ + kx) * cin_ + ci];
    }
    float weight(int co, int ky, int kx, int ci) const {
        return w_[((static_cast<size_t>(co) * k_ + ky) * k_ + kx) * cin_ + ci];
    }

private:
    int k_, cin_, cout_;
    std::vector<float> w_, b_;
};

inline void relu_inplace(Tensor3& t) {
    for (auto& x : t.v) x = x > 0.0f ? x : 0.0f;
}

inline Tensor3 maxpool2(const Tensor3& in) {
    Tensor3 out(in.h / 2, in.w / 2, in.c);
    for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.w; ++x)
            for (int ch = 0; ch < in.c; ++ch) {
                float m = in.at(y * 2, x * 2, ch);
                m = std::max(m, in.at(y * 2 + 1, x * 2, ch));
                m = std::max(m, in.at(y * 2, x * 2 + 1, ch));
                m = std::max(m, in.at(y * 2 + 1, x * 2 + 1, ch));
                out.at(y, x, ch) = m;
            }
    return out;
}

class TinyCNN {
public:
    explicit TinyCNN(unsigned seed = 42)
        : rng_(seed), conv1_(3, 1, 8, rng_), conv2_(3, 8, 16, rng_) {}

    // グレースケール(h,w,1) → 特徴マップ(h/4, w/4, 16)
    Tensor3 forward(const Tensor3& gray) const {
        Tensor3 x = conv1_.forward(gray);
        relu_inplace(x);
        x = maxpool2(x);
        x = conv2_.forward(x);
        relu_inplace(x);
        x = maxpool2(x);
        return x;
    }

private:
    std::mt19937 rng_;
    Conv2D conv1_, conv2_;
};

// ---------- ViT ----------
inline void layernorm_rows(Mat& m) {
    for (int i = 0; i < m.r; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < m.c; ++j) mean += m.at(i, j);
        mean /= m.c;
        float var = 0.0f;
        for (int j = 0; j < m.c; ++j) {
            const float d = m.at(i, j) - mean;
            var += d * d;
        }
        var /= m.c;
        const float inv = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < m.c; ++j) m.at(i, j) = (m.at(i, j) - mean) * inv;
    }
}

inline void softmax_rows(Mat& m) {
    for (int i = 0; i < m.r; ++i) {
        float mx = m.at(i, 0);
        for (int j = 1; j < m.c; ++j) mx = std::max(mx, m.at(i, j));
        float sum = 0.0f;
        for (int j = 0; j < m.c; ++j) {
            m.at(i, j) = std::exp(m.at(i, j) - mx);
            sum += m.at(i, j);
        }
        for (int j = 0; j < m.c; ++j) m.at(i, j) /= sum;
    }
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

struct ViTOutput {
    Mat tokens;   // n x dim パッチ埋め込み（Transformer通過後）
    Mat attention; // n x n ヘッド平均のAttention行列
    int grid_h = 0, grid_w = 0;
};

class TinyViT {
public:
    static constexpr int kPatch = 8;
    static constexpr int kDim = 32;
    static constexpr int kHeads = 4;

    explicit TinyViT(unsigned seed = 7) {
        std::mt19937 rng(seed);
        const int in_dim = kPatch * kPatch; // グレースケール前提
        w_embed_ = Mat(in_dim, kDim);
        he_init(w_embed_.v, in_dim, rng);
        for (Mat* m : {&wq_, &wk_, &wv_, &wo_}) {
            *m = Mat(kDim, kDim);
            he_init(m->v, kDim, rng);
        }
        w_mlp1_ = Mat(kDim, kDim * 2);
        he_init(w_mlp1_.v, kDim, rng);
        w_mlp2_ = Mat(kDim * 2, kDim);
        he_init(w_mlp2_.v, kDim * 2, rng);
    }

    // グレースケール画像 → トークン列とAttentionマップ
    ViTOutput forward(const Tensor3& gray) const {
        assert(gray.c == 1);
        const int gh = gray.h / kPatch, gw = gray.w / kPatch;
        const int n = gh * gw;
        assert(n > 0);

        // パッチ埋め込み + サイン波位置エンコーディング
        Mat x(n, kDim);
        Mat patch(1, kPatch * kPatch);
        for (int py = 0; py < gh; ++py)
            for (int px = 0; px < gw; ++px) {
                for (int dy = 0; dy < kPatch; ++dy)
                    for (int dx = 0; dx < kPatch; ++dx)
                        patch.at(0, dy * kPatch + dx) = gray.at(py * kPatch + dy, px * kPatch + dx, 0);
                const Mat emb = matmul(patch, w_embed_);
                const int t = py * gw + px;
                for (int j = 0; j < kDim; ++j) x.at(t, j) = emb.at(0, j) + positional(t, j);
            }

        // Transformer ブロック: LN → MHA → 残差 → LN → MLP → 残差
        Mat normed = x;
        layernorm_rows(normed);
        Mat q = matmul(normed, wq_), k = matmul(normed, wk_), v = matmul(normed, wv_);

        const int dh = kDim / kHeads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        Mat attn_avg(n, n), mha(n, kDim);
        for (int h = 0; h < kHeads; ++h) {
            const int off = h * dh;
            Mat scores(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < dh; ++d) s += q.at(i, off + d) * k.at(j, off + d);
                    scores.at(i, j) = s * scale;
                }
            softmax_rows(scores);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    attn_avg.at(i, j) += scores.at(i, j) / kHeads;
                    for (int d = 0; d < dh; ++d)
                        mha.at(i, off + d) += scores.at(i, j) * v.at(j, off + d);
                }
        }
        Mat attended = matmul(mha, wo_);
        for (size_t i = 0; i < x.v.size(); ++i) attended.v[i] += x.v[i]; // 残差

        Mat normed2 = attended;
        layernorm_rows(normed2);
        Mat hidden = matmul(normed2, w_mlp1_);
        for (auto& e : hidden.v) e = gelu(e);
        Mat mlp_out = matmul(hidden, w_mlp2_);
        for (size_t i = 0; i < mlp_out.v.size(); ++i) mlp_out.v[i] += attended.v[i]; // 残差

        return {std::move(mlp_out), std::move(attn_avg), gh, gw};
    }

    // 各パッチが「どれだけ注目されたか」（Attention列和）→ 0..1 の顕著性マップ
    static Tensor3 saliency_map(const ViTOutput& out) {
        const int n = out.grid_h * out.grid_w;
        std::vector<float> recv(n, 0.0f);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) recv[j] += out.attention.at(i, j);
        float lo = recv[0], hi = recv[0];
        for (float r : recv) {
            lo = std::min(lo, r);
            hi = std::max(hi, r);
        }
        const float range = hi - lo > 1e-9f ? hi - lo : 1.0f;
        Tensor3 map(out.grid_h, out.grid_w, 1);
        for (int y = 0; y < out.grid_h; ++y)
            for (int x = 0; x < out.grid_w; ++x)
                map.at(y, x, 0) = (recv[y * out.grid_w + x] - lo) / range;
        return map;
    }

private:
    static float positional(int pos, int j) {
        const float div = std::pow(10000.0f, static_cast<float>(j / 2 * 2) / kDim);
        return (j % 2 == 0) ? std::sin(pos / div) : std::cos(pos / div);
    }

    Mat w_embed_, wq_, wk_, wv_, wo_, w_mlp1_, w_mlp2_;
};

} // namespace syn
