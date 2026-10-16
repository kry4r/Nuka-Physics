#pragma once
// A read-only physics debug layer: collider proxies (green dynamic, magenta static) and
// contact markers in RenderWorld::debug_instances, built from data the physics thread read.

#include "runtime/app/viewer/debug_draw.hpp"
#include "runtime/app/viewer/editor_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::app::viewer {

// One environment's colliders and contact points, read at a tick boundary.
struct OverlayData {
    bool colliders_valid = false;
    std::vector<nuka_collider_t> colliders;
    bool contacts_valid = false;
    std::vector<uint32_t> contact_counts;    // per contact slot
    std::vector<math::Vec3> contact_points;  // four per contact slot
};

// Reads what the enabled layers need through the C ABI; disabled layers come back invalid.
void ReadOverlayData(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                     bool colliders, bool contacts, OverlayData* out);

class DebugOverlay {
public:
    void Reset();

    // Replaces the debug channel with the enabled layers; both off leaves it empty.
    void Rebuild(const OverlayData& data, render::RenderWorld& render_world, bool show_colliders,
                 bool show_contacts);

    uint32_t LastColliderCount() const { return batch_.Report().colliders; }
    uint32_t LastContactCount() const { return batch_.Report().contacts; }
    uint32_t LastSkippedShapes() const { return batch_.Report().unsupported_shapes; }
    bool CollidersAvailable() const { return colliders_available_; }
    bool ContactsAvailable() const { return contacts_available_; }
    const DebugDrawReport& DrawReport() const { return batch_.Report(); }

private:
    // A rebuilt RenderWorld drops the appended materials; they are re-added when stale.
    void EnsureMaterials(render::RenderWorld& render_world);

    uint32_t mat_dynamic_ = 0u;
    uint32_t mat_static_ = 0u;
    uint32_t mat_contact_ = 0u;
    bool materials_ready_ = false;
    bool overflow_logged_ = false;
    bool colliders_available_ = false;
    bool contacts_available_ = false;
    DebugDrawBatch batch_;
};

}  // namespace nuka::runtime::app::viewer
