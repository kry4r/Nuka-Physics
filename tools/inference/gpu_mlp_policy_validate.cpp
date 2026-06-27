// ---------------------------------------------------------------------------
// gpu_mlp_policy_validate -- load a flat policy binary into BOTH the host MlpPolicy
// (the numeric oracle) and the device GpuMlp, run the same obs through each, and
// report max|delta| GPU-vs-host. With a torch golden (export_go2_policy.py) it also
// reports GPU-vs-torch. Exit 0 iff max|GPU - host| < 1e-4 (the GPU-vs-host bar).
//
// Usage: gpu_mlp_policy_validate <weights.bin> [golden.bin]
// ---------------------------------------------------------------------------

#include "runtime/inference/gpu_policy.hpp"
#include "runtime/inference/mlp_policy.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

namespace inf = nuka::runtime::inference;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights.bin> [golden.bin]\n", argv[0]);
        return 2;
    }
    inf::MlpPolicy host;
    if (!host.Load(argv[1])) return 2;
    inf::GpuMlp gpu;
    if (!gpu.InitFromHost(host)) {
        std::fprintf(stderr, "gpu_mlp_policy_validate: GpuMlp init failed\n");
        return 2;
    }
    const uint32_t obs_dim = host.ObsDim();
    const uint32_t act_dim = host.ActDim();

    // Obs batch: the torch golden's obs if given, else a deterministic random set.
    uint32_t n = 0;
    std::vector<float> obs;
    std::vector<float> torch_mu;
    bool have_golden = false;
    if (argc >= 3) {
        std::ifstream g(argv[2], std::ios::binary);
        uint32_t gn = 0, gobs = 0, gact = 0;
        if (g && g.read(reinterpret_cast<char*>(&gn), 4) &&
            g.read(reinterpret_cast<char*>(&gobs), 4) &&
            g.read(reinterpret_cast<char*>(&gact), 4) && gobs == obs_dim &&
            gact == act_dim) {
            obs.resize(static_cast<size_t>(gn) * obs_dim);
            torch_mu.resize(static_cast<size_t>(gn) * act_dim);
            g.read(reinterpret_cast<char*>(obs.data()),
                   static_cast<std::streamsize>(obs.size() * sizeof(float)));
            g.read(reinterpret_cast<char*>(torch_mu.data()),
                   static_cast<std::streamsize>(torch_mu.size() * sizeof(float)));
            if (g) {
                n = gn;
                have_golden = true;
            }
        }
        if (!have_golden) {
            std::fprintf(stderr, "gpu_mlp_policy_validate: golden unusable, using random obs\n");
        }
    }
    if (n == 0) {
        n = 512;
        obs.resize(static_cast<size_t>(n) * obs_dim);
        std::mt19937 rng(1234u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : obs) v = nd(rng);
    }

    std::vector<float> gpu_out(static_cast<size_t>(n) * act_dim, 0.0f);
    if (!gpu.Forward(obs.data(), gpu_out.data(), n)) {
        std::fprintf(stderr, "gpu_mlp_policy_validate: GPU forward failed\n");
        return 2;
    }
    std::vector<float> host_out(act_dim, 0.0f);
    double max_gpu_host = 0.0, max_gpu_torch = 0.0, max_host_torch = 0.0;
    for (uint32_t s = 0; s < n; ++s) {
        host.Forward(&obs[static_cast<size_t>(s) * obs_dim], host_out.data());
        for (uint32_t j = 0; j < act_dim; ++j) {
            const double g = gpu_out[static_cast<size_t>(s) * act_dim + j];
            const double h = host_out[j];
            max_gpu_host = std::max(max_gpu_host, std::fabs(g - h));
            if (have_golden) {
                const double t = torch_mu[static_cast<size_t>(s) * act_dim + j];
                max_gpu_torch = std::max(max_gpu_torch, std::fabs(g - t));
                max_host_torch = std::max(max_host_torch, std::fabs(h - t));
            }
        }
    }
    std::printf("gpu_mlp_policy_validate: samples=%u obs_dim=%u act_dim=%u  "
                "max|GPU-host|=%.3e  (tol 1e-4)\n",
                n, obs_dim, act_dim, max_gpu_host);
    if (have_golden) {
        std::printf("  golden: max|host-torch|=%.3e  max|GPU-torch|=%.3e\n",
                    max_host_torch, max_gpu_torch);
    }
    const bool pass = max_gpu_host < 1.0e-4;
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
