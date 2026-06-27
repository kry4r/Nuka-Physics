#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::inference::MlpPolicy -- a feed-forward actor with input
// RunningMeanStd normalization, loaded from a flat self-describing binary
// (tools/inference/export_go2_policy.py). Reproduces an rl_games actor forward
// (normalize -> [Linear, ELU] x N -> linear head) on host floats with NO libtorch
// dependency, matching the torch reference to < 1e-4. General: the obs/act dims +
// layer sizes come from the file, so any MLP actor loads (no policy-specific code).
//
// HOST-ONLY (no CUDA tokens).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::inference {

class MlpPolicy {
public:
    // Load weights + normalizer from `path`. Returns false (and logs to stderr) on a
    // missing/malformed file; Ready() reflects load success.
    bool Load(const std::string& path);
    bool Ready() const { return ready_; }
    uint32_t ObsDim() const { return obs_dim_; }
    uint32_t ActDim() const { return act_dim_; }

    // Forward one obs vector (length ObsDim) -> action (length ActDim, caller-sized).
    // A no-op when not Ready. The dot products accumulate in double for stability.
    void Forward(const float* obs, float* out) const;

    // Read-only view of one dense layer (weights row-major [out, in]). Pointers stay
    // valid while the policy is loaded; the host arrays own the storage. The device
    // port (GpuMlp) uploads these to reproduce the same forward.
    struct LayerView {
        uint32_t in = 0, out = 0, act = 0;  // act: 1 == ELU, 0 == linear
        const float* w = nullptr;           // [out*in]
        const float* b = nullptr;           // [out]
    };
    uint32_t LayerCount() const { return static_cast<uint32_t>(layers_.size()); }
    LayerView GetLayer(uint32_t i) const {
        const Layer& L = layers_[i];
        return LayerView{L.in, L.out, L.act, L.w.data(), L.b.data()};
    }
    const float* Mean() const { return mean_.data(); }
    const float* Std() const { return std_.data(); }
    float Clip() const { return clip_; }
    float EluAlpha() const { return elu_alpha_; }

private:
    struct Layer {
        uint32_t in = 0, out = 0, act = 0;  // act: 1 == ELU, 0 == linear
        std::vector<float> w;               // row-major [out, in]
        std::vector<float> b;               // [out]
    };

    bool ready_ = false;
    uint32_t obs_dim_ = 0, act_dim_ = 0;
    float clip_ = 5.0f, elu_alpha_ = 1.0f;
    std::vector<float> mean_, std_;         // input RunningMeanStd (f32)
    std::vector<Layer> layers_;
    mutable std::vector<float> buf_a_, buf_b_;  // ping-pong scratch
};

}  // namespace nuka::runtime::inference
