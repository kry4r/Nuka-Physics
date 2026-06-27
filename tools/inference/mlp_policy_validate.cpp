// ---------------------------------------------------------------------------
// mlp_policy_validate -- load a flat policy binary (export_go2_policy.py) + a
// binary golden of (obs -> mu) pairs computed by torch, run the C++ MlpPolicy
// forward on each obs, and report max|delta| vs the torch reference. Exit 0 iff
// max|delta| < 1e-4 (the C++-vs-Python inference parity bar).
//
// Usage: mlp_policy_validate <weights.bin> <golden.bin>
// ---------------------------------------------------------------------------

#include "runtime/inference/mlp_policy.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <weights.bin> <golden.bin>\n", argv[0]);
        return 2;
    }
    nuka::runtime::inference::MlpPolicy policy;
    if (!policy.Load(argv[1])) return 2;

    std::ifstream g(argv[2], std::ios::binary);
    if (!g) {
        std::fprintf(stderr, "cannot open golden %s\n", argv[2]);
        return 2;
    }
    uint32_t n = 0, obs_dim = 0, act_dim = 0;
    g.read(reinterpret_cast<char*>(&n), sizeof(n));
    g.read(reinterpret_cast<char*>(&obs_dim), sizeof(obs_dim));
    g.read(reinterpret_cast<char*>(&act_dim), sizeof(act_dim));
    if (!g || obs_dim != policy.ObsDim() || act_dim != policy.ActDim()) {
        std::fprintf(stderr, "golden dim mismatch: golden obs=%u act=%u vs policy obs=%u act=%u\n",
                     obs_dim, act_dim, policy.ObsDim(), policy.ActDim());
        return 2;
    }
    std::vector<float> obs(static_cast<size_t>(n) * obs_dim);
    std::vector<float> mu(static_cast<size_t>(n) * act_dim);
    g.read(reinterpret_cast<char*>(obs.data()),
           static_cast<std::streamsize>(obs.size() * sizeof(float)));
    g.read(reinterpret_cast<char*>(mu.data()),
           static_cast<std::streamsize>(mu.size() * sizeof(float)));
    if (!g) {
        std::fprintf(stderr, "truncated golden\n");
        return 2;
    }

    std::vector<float> out(act_dim);
    double max_err = 0.0;
    for (uint32_t s = 0; s < n; ++s) {
        policy.Forward(&obs[static_cast<size_t>(s) * obs_dim], out.data());
        for (uint32_t j = 0; j < act_dim; ++j) {
            const double e = std::fabs(static_cast<double>(out[j]) -
                                       static_cast<double>(mu[static_cast<size_t>(s) * act_dim + j]));
            if (e > max_err) max_err = e;
        }
    }
    std::printf("mlp_policy_validate: samples=%u obs_dim=%u act_dim=%u  max|delta|=%.3e  (tol 1e-4)\n",
                n, obs_dim, act_dim, max_err);
    const bool pass = max_err < 1.0e-4;
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
