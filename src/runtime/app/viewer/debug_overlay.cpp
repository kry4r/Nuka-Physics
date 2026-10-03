#include "runtime/app/viewer/debug_overlay.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/data/data.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/ecs/components.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>

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

void DebugOverlay::AppendColliders(nk::World& world, render::RenderWorld& render_world,
                                    uint32_t env_index) {
    const nk::Model& model = world.GetModel();
    const auto& shapes = model.shape_table_rows;
    const uint32_t bodies = model.capacities.bodies_per_env;
    const uint32_t env_count = model.capacities.env_count;
    if (shapes.empty()) { colliders_available_ = true; return; }
    const uint32_t env = (env_count > 0u) ? (env_index % env_count) : 0u;

    // A body-attached collider's live frame: an articulation LINK reads
    // LinkPose[link] o link_geom_local[link] (its BodyPose row is unpopulated until a
    // step syncs it); a free body reads BodyPose[i]; implicit ground is world origin.
    // STATIC only if implicit ground or a free body with inv_mass 0 -- a link
    // (body_to_link != ~0u) is dynamic though its body-row inv_mass is 0.
    const auto& body_to_link = model.body_to_link;                    // host; empty w/o articulation
    const auto& link_geom_local = model.articulation.link_geom_local; // host; per template link
    const uint32_t links = model.capacities.links_per_env;
    static_assert(sizeof(Transform) == 7u * sizeof(float), "BodyPose wire layout");
    body_poses_.assign(bodies ? bodies : 1u, Transform::Identity());
    body_inv_mass_.assign(bodies ? bodies : 1u, 0.0f);
    link_poses_.assign(links ? links : 1u, Transform::Identity());
    nk::Data& data = world.GetData();
    if (bodies > 0u) {
        const uint64_t base = static_cast<uint64_t>(env) * bodies;
        if (!data.DownloadField(nk::FieldId::BodyPose, body_poses_.data(),
                                static_cast<uint64_t>(bodies) * sizeof(Transform),
                                base * sizeof(Transform)) ||
            !data.DownloadField(nk::FieldId::BodyInvMass, body_inv_mass_.data(),
                                static_cast<uint64_t>(bodies) * sizeof(float),
                                base * sizeof(float))) {
            return;
        }
    }
    if (links > 0u &&
        !data.DownloadField(nk::FieldId::LinkPose, link_poses_.data(),
                            static_cast<uint64_t>(links) * sizeof(Transform),
                            static_cast<uint64_t>(env) * links * sizeof(Transform))) {
        return;
    }

    colliders_available_ = true;
    for (uint32_t i = 0; i < shapes.size(); ++i) {
        const auto& sh = shapes[i];
        // Resolve the owning body through the row's own body_id (shape rows are
        // NOT 1:1 with body rows); body_id < 0 == static (ground / heightfield).
        const uint32_t bid = (sh.body_id >= 0) ? static_cast<uint32_t>(sh.body_id)
                                               : ~uint32_t(0);
        const bool has_body = bid < bodies;
        if (sh.body_id >= 0 && !has_body) { batch_.RejectCollider(); continue; }
        const bool is_link =
            has_body && bid < body_to_link.size() && body_to_link[bid] != ~uint32_t(0);
        Transform pose = Transform::Identity();
        if (is_link) {
            const uint32_t l = body_to_link[bid];
            if (l >= links || !IsValidDebugPose(link_poses_[l])) { batch_.RejectCollider(); continue; }
            pose = link_poses_[l];
            // A proxy collidable carries its own geom offset; the primary uses the link's.
            const auto& coll_link = model.body_collidable_link;
            if (bid < coll_link.size() && coll_link[bid] != ~uint32_t(0)) {
                if (bid >= model.body_collidable_local.size() ||
                    !IsValidDebugPose(model.body_collidable_local[bid])) { batch_.RejectCollider(); continue; }
                pose = pose * model.body_collidable_local[bid];
            } else if (l < link_geom_local.size()) {
                if (!IsValidDebugPose(link_geom_local[l])) { batch_.RejectCollider(); continue; }
                pose = pose * link_geom_local[l];
            }
        } else if (has_body) {
            pose = body_poses_[bid];
        }
        const bool is_static =
            has_body ? (!is_link && body_inv_mass_[bid] == 0.0f) : true;
        const uint32_t mat = is_static ? mat_static_ : mat_dynamic_;
        batch_.AppendCollider(render_world, sh.kind, sh.params, pose, mat);
    }
}

void DebugOverlay::AppendContacts(nk::World& world, render::RenderWorld& render_world,
                                   uint32_t env_index) {
    const nk::Model& model = world.GetModel();
    const uint32_t max_c = model.capacities.max_contacts_per_env;
    const uint32_t env_count = model.capacities.env_count;
    if (max_c == 0u) { contacts_available_ = true; return; }
    if (!batch_.ShouldReadContacts(max_c)) return;
    const uint32_t env = (env_count > 0u) ? (env_index % env_count) : 0u;

    ucontact_counts_.assign(max_c, 0u);
    ucontact_points_.assign(static_cast<size_t>(max_c) * 4u, Vec3{0, 0, 0});
    static_assert(sizeof(Vec3) == 3u * sizeof(float), "UcontactPoint wire layout");
    nk::Data& data = world.GetData();
    const uint64_t slot_base = static_cast<uint64_t>(env) * max_c;
    if (!data.DownloadField(nk::FieldId::UcontactCount, ucontact_counts_.data(),
                            static_cast<uint64_t>(max_c) * sizeof(uint32_t),
                            slot_base * sizeof(uint32_t)) ||
        !data.DownloadField(nk::FieldId::UcontactPoint, ucontact_points_.data(),
                            static_cast<uint64_t>(max_c) * 4u * sizeof(Vec3),
                            slot_base * 4u * sizeof(Vec3))) {
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

void DebugOverlay::Rebuild(nk::World& world, render::RenderWorld& render_world,
                           uint32_t env_index, bool show_colliders, bool show_contacts) {
    // The overlay owns the dedicated debug channel outright: clear it each rebuild.
    // The real `instances` set is never touched, so a non-overlay frame is untouched.
    render_world.debug_instances.clear();
    batch_.Reset();
    colliders_available_ = contacts_available_ = false;
    if (!show_colliders && !show_contacts) { overflow_logged_ = false; return; }

    EnsureMaterials(render_world);
    if (show_colliders) AppendColliders(world, render_world, env_index);
    if (show_contacts) AppendContacts(world, render_world, env_index);

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
