#pragma once
// ---------------------------------------------------------------------------
// nuka::scene — authored scene METADATA that is NOT a cook-fidelity record:
// the per-articulation settled IC (initial_state). Stored on the SceneIR
// alongside the records; persisted by the .nks Save/Load.
//
// HOST-ONLY, NO CUDA, NO nk/phi deps (plain data; std + math only).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"

#include <map>
#include <string>
#include <vector>

namespace nuka::scene {

// The settled IC of ONE articulation (the factory's 200-step settle product,
// controller ruling R4: BAKED — never re-derived). `qpos` is the flat PER-LINK
// q array (length == the articulation's device link count; the floating base
// root link's slot carries the base DOFs implicitly, the rest are 1-DOF joints)
// and `root` is the settled base pose. T4 maps qpos -> model.articulation.
// initial_q and root -> the base pose. Keyed in SceneInitialState by the
// articulation's tree node path (e.g. the import attach prefix "h1").
struct ArticulationInitialState {
    std::vector<float> qpos;                          // per-link settled q.
    math::Transform    root = math::Transform::Identity();
};

// initial_state section: node-path -> settled IC. A std::map keeps the keys in a
// deterministic (lexicographic) order so a second Save is byte-identical.
using SceneInitialState = std::map<std::string, ArticulationInitialState>;

}  // namespace nuka::scene
