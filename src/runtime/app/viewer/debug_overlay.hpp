#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::DebugOverlay -- a read-only physics debug-draw
// seam. Each frame it reads the cooked collision shapes + the live solver
// contact buffer and emits RenderWorld debug instances: a primitive proxy per
// collider (green dynamic / magenta static, Isaac-style) posed at the collider's
// live BodyPose, and a small marker at every active contact point.
//
// READ-ONLY: it mutates nothing in the SceneIR / physics and is independent of
// the edit / undo path. Debug instances go into RenderWorld's dedicated
// `debug_instances` channel (cleared each rebuild) with a sentinel entity + a
// non-movable pose source, so the real instance set, picker, and scene tree are
// untouched and a both-toggles-off frame is byte-identical to the real set.
//
// HOST-ONLY / zero-CUDA-token: device reads go only through Data::DownloadField.
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::nk {
class World;
}  // namespace nuka::nk

namespace nuka::runtime::app::viewer {

// One uniform cap on the total debug instances (collider proxies + contact
// markers) appended per frame. Overflow drops the remainder with a loud one-shot
// log -- no magic per-scene number (the project's cap discipline).
inline constexpr uint32_t kMaxDebugOverlayInstances = 8192u;

// Fixed marker radius (metres) for a contact-point sphere.
inline constexpr float kContactMarkerRadius = 0.02f;

class DebugOverlay {
public:
    // Forget the cached debug-material ids (optional). Rebuild is self-healing
    // across re-cooks on its own -- it re-adds materials when they go stale and
    // clears its instances by their sentinel tag -- so this is only a convenience
    // for reusing one overlay across distinct cooked worlds (the data-path gate).
    void Reset();

    // Rebuild the debug layer from the live world for `env_index`. Removes the
    // prior append, then -- when a toggle is on -- appends collider proxies and/or
    // contact markers. A no-op (beyond removing any prior append) when both
    // toggles are off, so the default path leaves the real instance set untouched.
    // `world` is read only (cooked shapes + downloaded fields); it is non-const
    // solely because Data::DownloadField is reached through a non-const accessor.
    void Rebuild(nk::World& world, render::RenderWorld& render_world,
                 uint32_t env_index, bool show_colliders, bool show_contacts);

    // Counts emitted by the most recent Rebuild (the data-path gate reads these).
    uint32_t LastColliderCount() const { return last_colliders_; }
    uint32_t LastContactCount()  const { return last_contacts_; }
    uint32_t LastSkippedShapes() const { return last_skipped_; }

private:
    // Append the three debug materials (dynamic / static / contact) to the
    // RenderWorld once after a Reset, recording their ids. Re-adds after a recook
    // (the rebuilt RenderWorld dropped the prior materials).
    void EnsureMaterials(render::RenderWorld& render_world);

    uint32_t AppendColliders(nk::World& world, render::RenderWorld& render_world,
                             uint32_t env_index, uint32_t& budget);
    uint32_t AppendContacts(nk::World& world, render::RenderWorld& render_world,
                            uint32_t env_index, uint32_t& budget);

    // Material ids into RenderWorld::materials; valid only while materials_ready_.
    uint32_t mat_dynamic_ = 0u;
    uint32_t mat_static_  = 0u;
    uint32_t mat_contact_ = 0u;
    bool     materials_ready_ = false;
    bool     overflow_logged_ = false;

    uint32_t last_colliders_ = 0u;
    uint32_t last_contacts_  = 0u;
    uint32_t last_skipped_   = 0u;

    // Reused host staging for the per-frame field downloads (allocation-free after
    // the first rebuild).
    std::vector<math::Transform> body_poses_;
    std::vector<math::Transform> link_poses_;
    std::vector<float>           body_inv_mass_;
    std::vector<uint32_t>        ucontact_counts_;
    std::vector<math::Vec3>      ucontact_points_;
};

}  // namespace nuka::runtime::app::viewer
