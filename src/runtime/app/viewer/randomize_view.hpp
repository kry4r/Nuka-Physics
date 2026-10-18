#pragma once
// Domain randomization as the editor keeps it for a document: render jitters, physics ranges and
// factor ranges for the parameter registry, all drawn from one seed. Every draw is a logged input.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// nuka_render_dr_desc_t's jitters, in its order.
inline constexpr const char* kRenderJitters[] = {"color",           "roughness",   "metallic", "light_dir",
                                                 "light_intensity", "light_color", "ambient"};
inline constexpr size_t kJitterCount = sizeof(kRenderJitters) / sizeof(kRenderJitters[0]);
// The physics ranges the engine applies: a mass factor, then armature and gravity offsets.
inline constexpr const char* kPhysicsRanges[] = {"mass", "armature", "gravity"};
inline constexpr size_t kRangeCount = sizeof(kPhysicsRanges) / sizeof(kPhysicsRanges[0]);

// A registry parameter whose every element is scaled by its own factor in [lo, hi].
struct ParamRange {
    std::string key;
    bool on = true;
    float lo = 1.0f, hi = 1.0f;
};

struct RandomizeConfig {
    uint32_t seed = 1u;
    bool render = false;
    std::array<float, kJitterCount> jitter{};
    bool physics = false;
    std::array<float, 2u * kRangeCount> range = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // lo, hi pairs
    std::vector<ParamRange> params;
    bool reset_draws = true;  // a reset draws again with the next seed

    bool Active() const {
        if (render || physics) return true;
        for (const ParamRange& p : params)
            if (p.on) return true;
        return false;
    }
};

struct RandomizeView {
    RandomizeConfig config;
    uint64_t version = 0u;  // bumped whenever `config` changes
    // The world's live per-link float parameters, which ranges can scale.
    std::vector<std::string> param_keys;
    std::vector<std::string> param_units;
    bool camera = false;       // render jitters need a camera sensor
    bool articulated = false;  // physics ranges need an articulation
    bool drawn = false;        // a draw from config.seed is in force
};

}  // namespace nuka::runtime::app::viewer
