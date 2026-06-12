#pragma once
// ---------------------------------------------------------------------------
// nk CookToModel — SceneIR -> {nk::Model, SceneMap} (plan §3.3 / M3).
//
// Drives the EXISTING cook (scene::CookScene for the heavy lifting: V-HACD /
// SDF / filters; runtime::articulation::CookArticulations for the kinematic
// tree) and TRANSCRIBES the resulting CookedBlob + articulation topology into
// the nk::Model field tables. The env template is replicated env_count times
// (env-major). The physics-material bucket table is built from the cooked
// per-shape friction/contact params. SceneMap binds each entity to its cooked
// rows (record order == row order for bodies/joints; shapes expand by V-HACD
// pieces — see the .cpp for the mapping choice).
//
// PURE C++ — zero CUDA tokens. (The Model's device upload happens later, in
// nk::World; CookToModel only fills host tables.)
// ---------------------------------------------------------------------------

#include "nk/model/model.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"

namespace nuka::scene::cook {

struct CookToModelResult {
    nk::Model model;
    SceneMap  scene_map;
};

// Cook a SceneIR into an nk::Model replicated across `env_count` envs, plus the
// EntityId<->row SceneMap. env_count must be >= 1 (clamped to 1 if 0).
CookToModelResult CookToModel(const SceneIR& scene, int env_count);

} // namespace nuka::scene::cook
