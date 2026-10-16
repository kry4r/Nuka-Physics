#include "runtime/app/viewer/debug_overlay.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/ecs/components.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace nuka::runtime::app::viewer {
namespace {

using nuka::math::Transform;
using nuka::math::Vec3;
constexpr float kContactSig[3] = {1.0f, 0.45f, 0.05f};
constexpr float kDebugOverlayAlpha = 0.5f;

void SetColor(scene::RenderMaterial& mat, float r, float g, float b, float e) {
    mat.base_color[0] = r; mat.base_color[1] = g; mat.base_color[2] = b; mat.base_color[3] = 1.0f;
    mat.emissive[0] = r * e; mat.emissive[1] = g * e; mat.emissive[2] = b * e;
    mat.metallic = 0.0f;
    mat.roughness = 1.0f;
    mat.opacity = kDebugOverlayAlpha;
}

}  // namespace

void DebugOverlay::Reset() {
    materials_ready_ = false;
    overflow_logged_ = false;
    batch_.Reset();
    colliders_available_ = contacts_available_ = false;
}

void DebugOverlay::EnsureMaterials(render::RenderWorld& render_world) {
    // Reuse the cached materials only while they are still the ones we appended; a
    // re-cook rebuilds RenderWorld::materials and drops them (stale id / signature).
    const bool intact =
        materials_ready_ && mat_contact_ < render_world.materials.size() &&
        render_world.materials[mat_contact_].base_color[0] == kContactSig[0] &&
        render_world.materials[mat_contact_].base_color[1] == kContactSig[1] &&
        render_world.materials[mat_contact_].base_color[2] == kContactSig[2];
    if (intact) return;
    mat_dynamic_ = static_cast<uint32_t>(render_world.materials.size());
    render_world.materials.emplace_back();
    SetColor(render_world.materials.back(), 0.15f, 0.85f, 0.25f, 0.45f);  // dynamic: green
    mat_static_ = static_cast<uint32_t>(render_world.materials.size());
    render_world.materials.emplace_back();
    SetColor(render_world.materials.back(), 0.85f, 0.18f, 0.85f, 0.45f);  // static: magenta
    mat_contact_ = static_cast<uint32_t>(render_world.materials.size());
    render_world.materials.emplace_back();
    SetColor(render_world.materials.back(), 1.0f, 0.45f, 0.05f, 0.9f);    // contact: orange
    materials_ready_ = true;
}

void DebugOverlay::AppendColliders(nuka_world_handle world, render::RenderWorld& render_world,
                                    uint32_t env_index) {
    // Each cooked collider comes with its live world pose for this environment.
    uint32_t count = 0u;
    if (nuka_world_get_colliders(world, env_index, nullptr, 0u, &count) != NUKA_RESULT_OK) return;
    colliders_.resize(count);
    if (count > 0u &&
        nuka_world_get_colliders(world, env_index, colliders_.data(), count, &count) !=
            NUKA_RESULT_OK)
        return;
    colliders_available_ = true;
    for (const nuka_collider_t& collider : colliders_) {
        Transform pose;
        std::memcpy(&pose, collider.pose, sizeof(pose));
        if (!IsValidDebugPose(pose)) { batch_.RejectCollider(); continue; }
        const uint32_t mat = collider.is_static != 0u ? mat_static_ : mat_dynamic_;
        batch_.AppendCollider(render_world, collider.kind, collider.params, pose, mat);
    }
}

void DebugOverlay::AppendContacts(nuka_world_handle world, const WorldLayout& layout,
                                   render::RenderWorld& render_world, uint32_t env_index) {
    const uint32_t max_c = layout.contact_slots_per_env;
    if (max_c == 0u) { contacts_available_ = true; return; }
    if (!batch_.ShouldReadContacts(max_c)) return;

    ucontact_counts_.assign(max_c, 0u);
    ucontact_points_.assign(static_cast<size_t>(max_c) * 4u, Vec3{0, 0, 0});
    static_assert(sizeof(Vec3) == 3u * sizeof(float), "UcontactPoint wire layout");
    const size_t slot_base = static_cast<size_t>(env_index % layout.env_count) * max_c;
    if (nuka_world_download_field(world, NUKA_FIELD_UCONTACT_COUNT, ucontact_counts_.data(),
                                  static_cast<size_t>(max_c) * sizeof(uint32_t),
                                  slot_base * sizeof(uint32_t)) != NUKA_RESULT_OK ||
        nuka_world_download_field(world, NUKA_FIELD_UCONTACT_POINT, ucontact_points_.data(),
                                  static_cast<size_t>(max_c) * 4u * sizeof(Vec3),
                                  slot_base * 4u * sizeof(Vec3)) != NUKA_RESULT_OK) {
        return;
    }

    contacts_available_ = true;
    for (uint32_t s = 0; s < max_c; ++s) {
        const uint32_t count = std::min(ucontact_counts_[s], 4u);
        for (uint32_t p = 0; p < count; ++p)
            batch_.AppendContact(render_world, ucontact_points_[static_cast<size_t>(s) * 4u + p],
                                 mat_contact_);
    }
}

void DebugOverlay::Rebuild(nuka_world_handle world, const WorldLayout& layout,
                           render::RenderWorld& render_world, uint32_t env_index,
                           bool show_colliders, bool show_contacts) {
    // The overlay owns the dedicated debug channel outright: clear it each rebuild.
    // The real `instances` set is never touched, so a non-overlay frame is untouched.
    render_world.debug_instances.clear();
    batch_.Reset();
    colliders_available_ = contacts_available_ = false;
    if (!show_colliders && !show_contacts) { overflow_logged_ = false; return; }

    EnsureMaterials(render_world);
    if (show_colliders) AppendColliders(world, render_world, env_index % layout.env_count);
    if (show_contacts) AppendContacts(world, layout, render_world, env_index);

    const auto& report = batch_.Report();
    const bool omitted = report.omitted_instances != 0u || report.contacts_budget_skipped;
    if (omitted && !overflow_logged_) {
        std::fprintf(stderr, "[debug_overlay] omitted=%llu; contacts unread due to budget=%s\n",
                     static_cast<unsigned long long>(report.omitted_instances),
                     report.contacts_budget_skipped ? "yes" : "no");
    }
    overflow_logged_ = omitted;
}

}  // namespace nuka::runtime::app::viewer
