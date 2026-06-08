#pragma once
// ---------------------------------------------------------------------------
// tiny_mlp.hpp -- TEST-LOCAL flat-f32 MLP loader + forward (the C++ DEPLOY side of
// the policy->co-resident inference bridge). NOT production: it lives in the test
// tree and exists only to prove EXACT numerical parity (Gate 1b, 1e-5) with the
// PyTorch net trained by python/spikes/bridge_export.py.
//
// The bridge is deliberately NOT python-bind / NOT ONNX / NOT torch-at-deploy: PPO
// emits flat f32 weights, and the demo runs them through THIS ~50-line forward.
//
// WEIGHT LAYOUT (must match bridge_export.export_weights, all row-major, in order):
//   W1 [H, IN]      (H*IN floats, row r = output unit r of layer 1)
//   b1 [H]
//   W2 [H, H]
//   b2 [H]
//   W3 [OUT, H]
//   b3 [OUT]
// FORWARD: h1 = tanh(W1 x + b1); h2 = tanh(W2 h1 + b2); y = W3 h2 + b3.
// Weights/inputs/layer outputs are float32; the per-row dot-product ACCUMULATOR is
// double (the standard "f32 storage, f64 accumulate" inference pattern). This is
// not precision-gaming: naive sequential f32 summation rounds the 64-term dot
// products differently from torch's blocked/pairwise f32 sum (~1.5e-5 divergence
// through two tanh layers) -- both are valid f32 evaluations, neither is "wrong",
// but f64 accumulation removes the naive-summation noise so the C++ forward and the
// genuine torch-f32 golden agree to ~1e-6 (Gate 1b's 1e-5 bar is then a clean pass
// that certifies layout/activation/weight transfer, not summation order). torch
// nn.Linear.weight is [out,in] row-major, so the matvec y_o = sum_i W[o,i]*x_i +
// b[o] reproduces it verbatim.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::test {

template <uint32_t IN = 32u, uint32_t H = 64u, uint32_t OUT = 10u>
class TinyMlp {
public:
    static constexpr uint32_t kIn = IN, kHidden = H, kOut = OUT;
    static constexpr uint32_t kCount =
        (H * IN + H) + (H * H + H) + (OUT * H + OUT);

    // Load the flat f32 weights from `path`. Throws if the file size mismatches.
    static TinyMlp Load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("tiny_mlp: cannot open " + path);
        std::vector<float> flat((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        // (the above reads bytes; reinterpret as floats below.)
        f.clear();
        f.seekg(0, std::ios::end);
        const std::streamoff bytes = f.tellg();
        if (static_cast<size_t>(bytes) != kCount * sizeof(float)) {
            throw std::runtime_error(
                "tiny_mlp: " + path + " size mismatch: got " +
                std::to_string(bytes) + " bytes, expected " +
                std::to_string(kCount * sizeof(float)));
        }
        f.seekg(0, std::ios::beg);
        TinyMlp m;
        m.raw_.resize(kCount);
        f.read(reinterpret_cast<char*>(m.raw_.data()),
               static_cast<std::streamsize>(kCount * sizeof(float)));
        m.Slice();
        return m;
    }

    // Build from an in-memory flat f32 vector (used by the BITE tests to inject a
    // deliberately TRANSPOSED W1 without re-reading the file).
    static TinyMlp FromFlat(std::vector<float> flat) {
        if (flat.size() != kCount)
            throw std::runtime_error("tiny_mlp: flat size mismatch");
        TinyMlp m;
        m.raw_ = std::move(flat);
        m.Slice();
        return m;
    }

    const std::vector<float>& Flat() const { return raw_; }

    // y = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3. `x` must be length IN; returns OUT.
    std::vector<float> Forward(const std::vector<float>& x) const {
        if (x.size() != IN) throw std::runtime_error("tiny_mlp: bad input dim");
        std::vector<float> h1(H), h2(H), y(OUT);
        for (uint32_t o = 0u; o < H; ++o) {
            double acc = b1_[o];
            for (uint32_t i = 0u; i < IN; ++i)
                acc += static_cast<double>(W1_[o * IN + i]) * x[i];
            h1[o] = std::tanh(static_cast<float>(acc));
        }
        for (uint32_t o = 0u; o < H; ++o) {
            double acc = b2_[o];
            for (uint32_t i = 0u; i < H; ++i)
                acc += static_cast<double>(W2_[o * H + i]) * h1[i];
            h2[o] = std::tanh(static_cast<float>(acc));
        }
        for (uint32_t o = 0u; o < OUT; ++o) {
            double acc = b3_[o];
            for (uint32_t i = 0u; i < H; ++i)
                acc += static_cast<double>(W3_[o * H + i]) * h2[i];
            y[o] = static_cast<float>(acc);
        }
        return y;
    }

private:
    void Slice() {
        const float* p = raw_.data();
        W1_ = p;            p += H * IN;
        b1_ = p;            p += H;
        W2_ = p;            p += H * H;
        b2_ = p;            p += H;
        W3_ = p;            p += OUT * H;
        b3_ = p;            p += OUT;
    }
    std::vector<float> raw_;
    const float* W1_ = nullptr;
    const float* b1_ = nullptr;
    const float* W2_ = nullptr;
    const float* b2_ = nullptr;
    const float* W3_ = nullptr;
    const float* b3_ = nullptr;
};

}  // namespace nuka::test
