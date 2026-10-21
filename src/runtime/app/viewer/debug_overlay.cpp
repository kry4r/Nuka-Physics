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
constexpr float kStaticSig[3] = {0.85f, 0.18f, 0.85f};
constexpr float kDebugOverlayAlpha = 0.5f;
static_assert(sizeof(Vec3) == 3u * sizeof(float), "UcontactPoint wire layout");

void SetColor(scene::RenderMaterial& mat, float r, float g, float b, float e) {
    mat.base_color[0] = r; mat.base_color[1] = g; mat.base_color[2] = b; mat.base_color[3] = 1.0f;
    mat.emissive[0] = r * e; mat.emissive[1] = g * e; mat.emissive[2] = b * e;
    mat.metallic = 0.0f;
    mat.roughness = 1.0f;
    mat.opacity = kDebugOverlayAlpha;
}

}  // namespace

void ReadOverlayData(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                     bool colliders, bool contacts, OverlayData* out) {
    out->colliders_valid = false;
    out->contacts_valid = false;
    const uint32_t env = env_index % layout.env_count;
    uint32_t count = 0u;
    if (colliders && nuka_world_get_colliders(world, env, nullptr, 0u, &count) == NUKA_RESULT_OK) {
        out->colliders.resize(count);
        out->colliders_valid =
            count == 0u ||
            nuka_world_get_colliders(world, env, out->colliders.data(), count, &count) == NUKA_RESULT_OK;
    }
    if (!contacts) return;
    const uint32_t slots = layout.contact_slots_per_env;
    out->contact_counts.assign(slots, 0u);
    out->contact_points.assign(static_cast<size_t>(slots) * 4u, Vec3{0, 0, 0});
    const size_t base = static_cast<size_t>(env) * slots;
    out->contacts_valid =
        slots == 0u ||
        (nuka_world_download_field(world, NUKA_FIELD_UCONTACT_COUNT, out->contact_counts.data(),
                                   static_cast<size_t>(slots) * sizeof(uint32_t),
                                   base * sizeof(uint32_t)) == NUKA_RESULT_OK &&
         nuka_world_download_field(world, NUKA_FIELD_UCONTACT_POINT, out->contact_points.data(),
                                   static_cast<size_t>(slots) * 4u * sizeof(Vec3),
                                   base * 4u * sizeof(Vec3)) == NUKA_RESULT_OK);
}

void DebugOverlay::Reset() {
    materials_ready_ = false;
    overflow_logged_ = false;
    batch_.Reset();
    colliders_available_ = contacts_available_ = false;
}

void DebugOverlay::EnsureMaterials(render::RenderWorld& render_world) {
    const bool intact =
        materials_ready_ && mat_static_ < render_world.materials.size() &&
        render_world.materials[mat_static_].base_color[0] == kStaticSig[0] &&
        render_world.materials[mat_static_].base_color[1] == kStaticSig[1] &&
        render_world.materials[mat_static_].base_color[2] == kStaticSig[2];
    if (intact) return;
    mat_dynamic_ = static_cast<uint32_t>(render_world.materials.size());
    render_world.materials.emplace_back();
    SetColor(render_world.materials.back(), 0.15f, 0.85f, 0.25f, 0.45f);  // dynamic: green
    mat_static_ = static_cast<uint32_t>(render_world.materials.size());
    render_world.materials.emplace_back();
    SetColor(render_world.materials.back(), kStaticSig[0], kStaticSig[1], kStaticSig[2], 0.45f);  // static: magenta
    materials_ready_ = true;
}

void DebugOverlay::Rebuild(const OverlayData& data, render::RenderWorld& render_world,
                           bool show_colliders, bool show_contacts) {
    render_world.debug_instances.clear();
    batch_.Reset();
    colliders_available_ = contacts_available_ = false;
    if (!show_colliders && !show_contacts) { overflow_logged_ = false; return; }

    EnsureMaterials(render_world);
    if (show_colliders && data.colliders_valid) {
        colliders_available_ = true;
        for (const nuka_collider_t& collider : data.colliders) {
            Transform pose;
            std::memcpy(&pose, collider.pose, sizeof(pose));
            if (!IsValidDebugPose(pose)) { batch_.RejectCollider(); continue; }
            const uint32_t mat = collider.is_static != 0u ? mat_static_ : mat_dynamic_;
            batch_.AppendCollider(render_world, collider.kind, collider.params, pose, mat);
        }
    }
    if (show_contacts && data.contacts_valid) {
        contacts_available_ = true;
        for (size_t s = 0; s < data.contact_counts.size(); ++s)
            for (uint32_t p = 0; p < std::min(data.contact_counts[s], 4u); ++p)
                batch_.CountContact(data.contact_points[s * 4u + p]);
    }

    const auto& report = batch_.Report();
    const bool omitted = report.omitted_instances != 0u;
    if (omitted && !overflow_logged_) {
        const auto count = static_cast<unsigned long long>(report.omitted_instances);
        std::fprintf(stderr, "[debug_overlay] omitted=%llu\n", count);
    }
    overflow_logged_ = omitted;
}

}  // namespace nuka::runtime::app::viewer
