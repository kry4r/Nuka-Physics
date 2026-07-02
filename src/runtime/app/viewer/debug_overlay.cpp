#include "runtime/app/viewer/debug_overlay.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/data/data.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/ecs/components.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace nuka::runtime::app::viewer {
namespace {

using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::render::MeshGeometry;

constexpr float kPi = 3.14159265358979323846f;

// Sentinel entity tagging a debug instance. index == ~0u is never a live slot (so
// the picker / tree never resolve it) and the gen distinguishes it from
// kInvalidEntity so the per-frame clear pops exactly the overlay's own tail.
constexpr scene::EntityId kDebugEntity{~uint32_t(0), ~uint32_t(0) - 1u};

// The contact material's signature colour; the staleness check re-adds the debug
// materials after a re-cook rebuilt RenderWorld::materials without them.
constexpr float kContactSig[3] = {1.0f, 0.45f, 0.05f};

// See-through alpha for the debug proxies. The present pass's wireframe pipeline
// draws opaque lines (blend off) and ignores this; the translucent fallback uses it.
constexpr float kDebugOverlayAlpha = 0.5f;

// Cooked collision-shape kinds (CollisionShapeComponent::Kind / shape_kind.hpp).
constexpr uint32_t kKindSphere  = 0u;
constexpr uint32_t kKindCapsule = 1u;
constexpr uint32_t kKindBox     = 2u;
constexpr uint32_t kKindPlane   = 3u;

// -- proxy tessellators: positions + flat per-vertex normals, dims baked in (the
// render Transform carries no scale), matching the visual primitive convention.
void PushTri(MeshGeometry& m, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
    Vec3 n{e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z,
           e1.x * e2.y - e1.y * e2.x};
    const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-12f) { n.x /= len; n.y /= len; n.z /= len; }
    const uint32_t base = m.VertexCount();
    const Vec3 v[3] = {a, b, c};
    for (const auto& p : v) {
        m.positions.push_back(p.x); m.positions.push_back(p.y); m.positions.push_back(p.z);
        m.normals.push_back(n.x);   m.normals.push_back(n.y);   m.normals.push_back(n.z);
    }
    m.indices.push_back(base); m.indices.push_back(base + 1); m.indices.push_back(base + 2);
}

void PushQuad(MeshGeometry& m, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    PushTri(m, a, b, c);
    PushTri(m, a, c, d);
}

MeshGeometry MakeBox(float hx, float hy, float hz) {
    MeshGeometry m;
    const Vec3 p[8] = {
        {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
        {-hx, -hy,  hz}, {hx, -hy,  hz}, {hx, hy,  hz}, {-hx, hy,  hz}};
    PushQuad(m, p[0], p[3], p[2], p[1]);
    PushQuad(m, p[4], p[5], p[6], p[7]);
    PushQuad(m, p[0], p[1], p[5], p[4]);
    PushQuad(m, p[3], p[7], p[6], p[2]);
    PushQuad(m, p[0], p[4], p[7], p[3]);
    PushQuad(m, p[1], p[2], p[6], p[5]);
    return m;
}

MeshGeometry MakeSphere(float r, uint32_t stacks = 10u, uint32_t slices = 14u) {
    MeshGeometry m;
    auto at = [&](uint32_t i, uint32_t j) -> Vec3 {
        const float v = kPi * static_cast<float>(i) / static_cast<float>(stacks);
        const float u = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
        return {r * std::sin(v) * std::cos(u), r * std::sin(v) * std::sin(u), r * std::cos(v)};
    };
    for (uint32_t i = 0; i < stacks; ++i)
        for (uint32_t j = 0; j < slices; ++j)
            PushQuad(m, at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
    return m;
}

// Capsule along local Z: a cylinder of half-height hh + two hemispherical caps of
// radius r (the engine's capsule convention).
MeshGeometry MakeCapsule(float r, float hh, uint32_t slices = 14u, uint32_t cap_stacks = 5u) {
    MeshGeometry m;
    for (uint32_t j = 0; j < slices; ++j) {
        const float u0 = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
        const float u1 = 2.0f * kPi * static_cast<float>(j + 1) / static_cast<float>(slices);
        PushQuad(m, {r * std::cos(u0), r * std::sin(u0), -hh},
                 {r * std::cos(u1), r * std::sin(u1), -hh},
                 {r * std::cos(u1), r * std::sin(u1), hh},
                 {r * std::cos(u0), r * std::sin(u0), hh});
    }
    auto cap = [&](float zc, float sign) {
        for (uint32_t i = 0; i < cap_stacks; ++i) {
            const float v0 = 0.5f * kPi * static_cast<float>(i) / static_cast<float>(cap_stacks);
            const float v1 = 0.5f * kPi * static_cast<float>(i + 1) / static_cast<float>(cap_stacks);
            for (uint32_t j = 0; j < slices; ++j) {
                const float u0 = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
                const float u1 = 2.0f * kPi * static_cast<float>(j + 1) / static_cast<float>(slices);
                auto pt = [&](float vv, float uu) -> Vec3 {
                    return {r * std::sin(vv) * std::cos(uu), r * std::sin(vv) * std::sin(uu),
                            zc + sign * r * std::cos(vv)};
                };
                PushQuad(m, pt(v0, u0), pt(v1, u0), pt(v1, u1), pt(v0, u1));
            }
        }
    };
    cap(hh, 1.0f);
    cap(-hh, -1.0f);
    return m;
}

// A plane collidable's finite viz patch: a double-sided flat quad of half-extents
// (hx,hy) in the local Z=0 plane (the physics plane is infinite; this shows where).
MeshGeometry MakePlane(float hx, float hy) {
    MeshGeometry m;
    // An infinite physics plane carries no extent; draw a bounded ground patch.
    constexpr float kGroundPatch = 2.0f;
    if (hx <= 0.0f) hx = kGroundPatch;
    if (hy <= 0.0f) hy = kGroundPatch;
    const Vec3 a{-hx, -hy, 0.0f}, b{hx, -hy, 0.0f}, c{hx, hy, 0.0f}, d{-hx, hy, 0.0f};
    PushQuad(m, a, b, c, d);
    PushQuad(m, a, d, c, b);
    return m;
}

void SetColor(scene::RenderMaterial& mat, float r, float g, float b, float e) {
    mat.base_color[0] = r; mat.base_color[1] = g; mat.base_color[2] = b; mat.base_color[3] = 1.0f;
    mat.emissive[0] = r * e; mat.emissive[1] = g * e; mat.emissive[2] = b * e;
    mat.metallic = 0.0f;
    mat.roughness = 1.0f;
    mat.opacity = kDebugOverlayAlpha;
}

render::RenderInstance MakeDebugInstance(uint32_t mesh_id, uint32_t material_id,
                                         const Transform& pose) {
    render::RenderInstance inst;
    inst.entity = kDebugEntity;  // tag: cleared each rebuild; pick / tree skip it
    inst.mesh_id = mesh_id;
    inst.render_material_id = material_id;
    inst.world_xform = pose;
    inst.cached_visual_local = Transform::Identity();
    inst.pose_source = render::PoseSource{};  // Static: the viewport picker ignores it
    return inst;
}

}  // namespace

void DebugOverlay::Reset() {
    materials_ready_ = false;
    overflow_logged_ = false;
    last_colliders_ = last_contacts_ = last_skipped_ = 0u;
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

uint32_t DebugOverlay::AppendColliders(nk::World& world,
                                       render::RenderWorld& render_world,
                                       uint32_t env_index, uint32_t& budget) {
    const nk::Model& model = world.GetModel();
    const auto& shapes = model.shape_table_rows;
    const uint32_t bodies = model.capacities.bodies_per_env;
    const uint32_t env_count = model.capacities.env_count;
    if (shapes.empty()) return 0u;
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
            return 0u;
        }
    }
    if (links > 0u &&
        !data.DownloadField(nk::FieldId::LinkPose, link_poses_.data(),
                            static_cast<uint64_t>(links) * sizeof(Transform),
                            static_cast<uint64_t>(env) * links * sizeof(Transform))) {
        return 0u;
    }

    uint32_t emitted = 0u;
    char key[96];
    for (uint32_t i = 0; i < shapes.size(); ++i) {
        const auto& sh = shapes[i];
        // Resolve the owning body through the row's own body_id (shape rows are
        // NOT 1:1 with body rows); body_id < 0 == static (ground / heightfield).
        const uint32_t bid = (sh.body_id >= 0) ? static_cast<uint32_t>(sh.body_id)
                                               : ~uint32_t(0);
        const bool has_body = bid < bodies;
        const bool is_link =
            has_body && bid < body_to_link.size() && body_to_link[bid] != ~uint32_t(0);
        Transform pose = Transform::Identity();
        if (is_link) {
            const uint32_t l = body_to_link[bid];
            if (l < link_poses_.size()) pose = link_poses_[l];
            if (l < link_geom_local.size()) pose = pose * link_geom_local[l];
        } else if (has_body) {
            pose = body_poses_[bid];
        }
        const bool is_static =
            has_body ? (!is_link && body_inv_mass_[bid] == 0.0f) : true;
        const uint32_t mat = is_static ? mat_static_ : mat_dynamic_;
        uint32_t mesh_id = render::kNoId;
        switch (sh.kind) {
            case kKindSphere: {
                const float r = sh.params[0];
                std::snprintf(key, sizeof(key), "dbgcol:sph:%.4f", r);
                mesh_id = render_world.meshes.InternPrimitive(
                    key, [&] { return MakeSphere(r); });
                break;
            }
            case kKindCapsule: {
                const float r = sh.params[0], hh = sh.params[1];
                std::snprintf(key, sizeof(key), "dbgcol:cap:%.4f,%.4f", r, hh);
                mesh_id = render_world.meshes.InternPrimitive(
                    key, [&] { return MakeCapsule(r, hh); });
                break;
            }
            case kKindBox: {
                const float hx = sh.params[0], hy = sh.params[1], hz = sh.params[2];
                std::snprintf(key, sizeof(key), "dbgcol:box:%.4f,%.4f,%.4f", hx, hy, hz);
                mesh_id = render_world.meshes.InternPrimitive(
                    key, [&] { return MakeBox(hx, hy, hz); });
                break;
            }
            case kKindPlane: {
                const float hx = sh.params[0], hy = sh.params[1];
                std::snprintf(key, sizeof(key), "dbgcol:pln:%.4f,%.4f", hx, hy);
                mesh_id = render_world.meshes.InternPrimitive(
                    key, [&] { return MakePlane(hx, hy); });
                break;
            }
            default:
                // ConvexHull / SdfMesh / heightfield: no analytic primitive dims (the
                // editor light-cook strips mesh hulls), so they are not drawn.
                ++last_skipped_;
                continue;
        }
        if (budget == 0u) break;
        render_world.debug_instances.push_back(MakeDebugInstance(mesh_id, mat, pose));
        --budget;
        ++emitted;
    }
    return emitted;
}

uint32_t DebugOverlay::AppendContacts(nk::World& world,
                                      render::RenderWorld& render_world,
                                      uint32_t env_index, uint32_t& budget) {
    const nk::Model& model = world.GetModel();
    const uint32_t max_c = model.capacities.max_contacts_per_env;
    const uint32_t env_count = model.capacities.env_count;
    if (max_c == 0u) return 0u;
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
        return 0u;
    }

    const float r = kContactMarkerRadius;
    const uint32_t mesh_id = render_world.meshes.InternPrimitive(
        "dbgcontact:marker", [&] { return MakeSphere(r, 8u, 10u); });
    uint32_t emitted = 0u;
    for (uint32_t s = 0; s < max_c; ++s) {
        uint32_t cnt = ucontact_counts_[s];
        if (cnt > 4u) cnt = 4u;  // elem:4 manifold slot
        for (uint32_t p = 0; p < cnt; ++p) {
            if (budget == 0u) return emitted;
            const Vec3& pt = ucontact_points_[static_cast<size_t>(s) * 4u + p];
            render_world.debug_instances.push_back(
                MakeDebugInstance(mesh_id, mat_contact_, Transform{pt, math::Quat::Identity()}));
            --budget;
            ++emitted;
        }
    }
    return emitted;
}

void DebugOverlay::Rebuild(nk::World& world, render::RenderWorld& render_world,
                           uint32_t env_index, bool show_colliders, bool show_contacts) {
    // The overlay owns the dedicated debug channel outright: clear it each rebuild.
    // The real `instances` set is never touched, so a non-overlay frame is untouched.
    render_world.debug_instances.clear();
    last_colliders_ = last_contacts_ = last_skipped_ = 0u;
    if (!show_colliders && !show_contacts) return;

    EnsureMaterials(render_world);
    uint32_t budget = kMaxDebugOverlayInstances;
    if (show_colliders) last_colliders_ = AppendColliders(world, render_world, env_index, budget);
    if (show_contacts)  last_contacts_  = AppendContacts(world, render_world, env_index, budget);

    if (budget == 0u && !overflow_logged_) {
        std::fprintf(stderr,
                     "[debug_overlay] instance cap %u reached (colliders=%u contacts=%u); "
                     "dropping remainder\n",
                     kMaxDebugOverlayInstances, last_colliders_, last_contacts_);
        overflow_logged_ = true;
    }
}

}  // namespace nuka::runtime::app::viewer
