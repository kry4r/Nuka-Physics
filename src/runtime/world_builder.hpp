#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::BuildWorld – construct runtime world from cooked blob
// ---------------------------------------------------------------------------

#include "runtime/world_template.hpp"
#include "runtime/batch_context.hpp"
#include "scene/cooked_blob.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime {

// Mutable per-instance simulation state populated by BuildWorld (a standalone
// per-instance state container folded into world_builder.hpp during the M0
// cleanup since BuildWorld's BuiltWorld is its only producer).
struct BuiltWorldState {
    uint32_t body_count = 0;

    std::vector<math::Transform> poses;
    std::vector<math::Vec3>      linear_velocities;
    std::vector<math::Vec3>      angular_velocities;
    std::vector<math::Vec3>      forces;
    std::vector<math::Vec3>      torques;
    std::vector<float>           articulation_q;
    std::vector<float>           articulation_qdot;
    std::vector<float>           articulation_qddot;
    std::vector<float>           articulation_tau;
};

struct BuiltWorld {
    WorldTemplate  template_view;
    BuiltWorldState instance;
    BatchContext   batch;
};

/// Build a single-instance world from a cooked scene blob.
BuiltWorld BuildWorld(const scene::CookedBlob& blob);

} // namespace nuka::runtime
