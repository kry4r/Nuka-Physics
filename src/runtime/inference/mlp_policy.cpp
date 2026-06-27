// ---------------------------------------------------------------------------
// MlpPolicy -- flat-binary loader + host MLP forward (see mlp_policy.hpp).
// ---------------------------------------------------------------------------

#include "runtime/inference/mlp_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>

namespace nuka::runtime::inference {

namespace {
constexpr uint32_t kMagic = 0x4E4B4D4Cu;  // "NKML"

template <class T>
bool ReadPod(std::ifstream& f, T* v) {
    f.read(reinterpret_cast<char*>(v), sizeof(T));
    return static_cast<bool>(f);
}
bool ReadFloats(std::ifstream& f, std::vector<float>* v, size_t n) {
    v->resize(n);
    f.read(reinterpret_cast<char*>(v->data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
}
}  // namespace

bool MlpPolicy::Load(const std::string& path) {
    ready_ = false;
    layers_.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[mlp_policy] cannot open %s\n", path.c_str());
        return false;
    }
    uint32_t magic = 0, version = 0;
    if (!ReadPod(f, &magic) || magic != kMagic) {
        std::fprintf(stderr, "[mlp_policy] bad magic in %s\n", path.c_str());
        return false;
    }
    if (!ReadPod(f, &version) || !ReadPod(f, &obs_dim_) || !ReadPod(f, &clip_) ||
        !ReadPod(f, &elu_alpha_)) {
        std::fprintf(stderr, "[mlp_policy] truncated header in %s\n", path.c_str());
        return false;
    }
    if (obs_dim_ == 0u || !ReadFloats(f, &mean_, obs_dim_) ||
        !ReadFloats(f, &std_, obs_dim_)) {
        std::fprintf(stderr, "[mlp_policy] bad normalizer in %s\n", path.c_str());
        return false;
    }
    uint32_t n_layers = 0;
    if (!ReadPod(f, &n_layers) || n_layers == 0u) {
        std::fprintf(stderr, "[mlp_policy] no layers in %s\n", path.c_str());
        return false;
    }
    uint32_t expect_in = obs_dim_;
    for (uint32_t l = 0; l < n_layers; ++l) {
        Layer L;
        if (!ReadPod(f, &L.in) || !ReadPod(f, &L.out) || !ReadPod(f, &L.act) ||
            L.in != expect_in || L.in == 0u || L.out == 0u) {
            std::fprintf(stderr, "[mlp_policy] bad layer %u in %s\n", l, path.c_str());
            return false;
        }
        if (!ReadFloats(f, &L.w, static_cast<size_t>(L.in) * L.out) ||
            !ReadFloats(f, &L.b, L.out)) {
            std::fprintf(stderr, "[mlp_policy] truncated layer %u in %s\n", l,
                         path.c_str());
            return false;
        }
        expect_in = L.out;
        layers_.push_back(std::move(L));
    }
    act_dim_ = expect_in;
    ready_ = true;
    return true;
}

void MlpPolicy::Forward(const float* obs, float* out) const {
    if (!ready_) return;
    // Input RunningMeanStd normalize + clamp (rl_games f32 normalize).
    buf_a_.resize(obs_dim_);
    for (uint32_t i = 0; i < obs_dim_; ++i) {
        float y = (obs[i] - mean_[i]) / std_[i];
        y = y < -clip_ ? -clip_ : (y > clip_ ? clip_ : y);
        buf_a_[i] = y;
    }
    std::vector<float>* in = &buf_a_;
    std::vector<float>* scratch = &buf_b_;
    for (const Layer& L : layers_) {
        scratch->resize(L.out);
        for (uint32_t o = 0; o < L.out; ++o) {
            double acc = static_cast<double>(L.b[o]);
            const float* wr = &L.w[static_cast<size_t>(o) * L.in];
            const float* xi = in->data();
            for (uint32_t i = 0; i < L.in; ++i) {
                acc += static_cast<double>(wr[i]) * static_cast<double>(xi[i]);
            }
            float v = static_cast<float>(acc);
            if (L.act == 1u) {  // ELU
                v = v > 0.0f ? v : elu_alpha_ * (std::exp(v) - 1.0f);
            }
            (*scratch)[o] = v;
        }
        std::swap(in, scratch);
    }
    for (uint32_t i = 0; i < act_dim_; ++i) out[i] = (*in)[i];
}

}  // namespace nuka::runtime::inference
