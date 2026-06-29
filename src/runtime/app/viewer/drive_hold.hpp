#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer -- general PD-hold gain enable.
//
// Uploads uniform PD gains to every ACTUATED joint of every articulation (across all
// envs), skipping each articulation's root/base link, so DRIVE TARGETS become
// effective for policy-free direct joint control (the Drive panel + keyboard teleop).
// General: driven by the cooked articulation topology, never per-robot constants. The
// PD solver additionally skips Fixed joints, so fixed sub-joints get zero torque
// naturally. Force limit is left at the cooked value (the kernel treats 0 as no clamp).
// HOST-ONLY (only Data UploadField behind the phi vtable; no CUDA tokens).
// ---------------------------------------------------------------------------

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nuka::runtime::app::viewer {

// Apply uniform stiffness `kp` + damping `kd` to the actuated joints of `env_count`
// envs. A no-op when the world carries no articulation. Idempotent (persistent fields).
inline void ApplyHoldGains(nk::World& world, uint32_t env_count, float kp, float kd) {
    const nk::Model& m = world.GetModel();
    const uint32_t L = m.articulation.link_count;  // per-env link count
    if (L == 0u) return;
    const uint32_t envs = env_count > 0u ? env_count : 1u;
    const uint32_t K = m.articulation.articulation_count;

    // Per-env template: gains on actuated joints only (skip each articulation's base).
    std::vector<float> kp_env(L, 0.0f), kd_env(L, 0.0f);
    auto mark = [&](uint32_t root, uint32_t count) {
        for (uint32_t l = root + 1u; l < root + count && l < L; ++l) {
            kp_env[l] = kp;
            kd_env[l] = kd;
        }
    };
    if (K > 0u && m.articulation.articulation_link_offset.size() == K &&
        m.articulation.articulation_link_count.size() == K) {
        for (uint32_t k = 0; k < K; ++k)
            mark(m.articulation.articulation_link_offset[k],
                 m.articulation.articulation_link_count[k]);
    } else {
        mark(0u, L);  // single articulation spanning [0, L)
    }

    // Replicate env-major into the persistent drive-gain fields.
    std::vector<float> kp_all(static_cast<size_t>(L) * envs);
    std::vector<float> kd_all(static_cast<size_t>(L) * envs);
    for (uint32_t e = 0; e < envs; ++e) {
        std::copy(kp_env.begin(), kp_env.end(),
                  kp_all.begin() + static_cast<size_t>(e) * L);
        std::copy(kd_env.begin(), kd_env.end(),
                  kd_all.begin() + static_cast<size_t>(e) * L);
    }
    nk::Data& d = world.GetData();
    const uint64_t bytes = static_cast<uint64_t>(kp_all.size()) * sizeof(float);
    d.UploadField(nk::FieldId::DriveStiffness, kp_all.data(), bytes, 0);
    d.UploadField(nk::FieldId::DriveDamping, kd_all.data(), bytes, 0);
}

}  // namespace nuka::runtime::app::viewer
