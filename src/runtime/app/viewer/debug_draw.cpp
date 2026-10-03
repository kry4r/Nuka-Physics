#include "runtime/app/viewer/debug_draw.hpp"

#include <bit>
#include <cmath>
#include <cstdio>

namespace nuka::runtime::app::viewer {
namespace {

using math::Transform;
using math::Vec3;
using render::MeshGeometry;
constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kSphere = static_cast<uint32_t>(scene::CollisionShapeComponent::Kind::Sphere);
constexpr uint32_t kCapsule = static_cast<uint32_t>(scene::CollisionShapeComponent::Kind::Capsule);
constexpr uint32_t kBox = static_cast<uint32_t>(scene::CollisionShapeComponent::Kind::Box);
constexpr uint32_t kPlane = static_cast<uint32_t>(scene::CollisionShapeComponent::Kind::Plane);
constexpr scene::EntityId kDebugEntity{~uint32_t(0), ~uint32_t(0) - 1u};

bool Finite(const Vec3& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

void PushTri(MeshGeometry& m, const Vec3& a, const Vec3& b, const Vec3& c) {
    const double ax = static_cast<double>(b.x) - a.x, ay = static_cast<double>(b.y) - a.y;
    const double az = static_cast<double>(b.z) - a.z, bx = static_cast<double>(c.x) - a.x;
    const double by = static_cast<double>(c.y) - a.y, bz = static_cast<double>(c.z) - a.z;
    const double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const double len = std::hypot(nx, ny, nz);
    const Vec3 n = len > 0.0 ? Vec3{static_cast<float>(nx / len), static_cast<float>(ny / len),
                                  static_cast<float>(nz / len)} : Vec3{};
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
    const Vec3 a{-hx, -hy, 0.0f}, b{hx, -hy, 0.0f}, c{hx, hy, 0.0f}, d{-hx, hy, 0.0f};
    PushQuad(m, a, b, c, d);
    PushQuad(m, a, d, c, b);
    return m;
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

bool IsValidDebugPose(const Transform& pose) {
    const auto& q = pose.rotation;
    // Rigid poses are used unchanged; reject broken rotations rather than repairing physics data.
    const double norm2 = static_cast<double>(q.w) * q.w + static_cast<double>(q.x) * q.x +
                         static_cast<double>(q.y) * q.y + static_cast<double>(q.z) * q.z;
    return Finite(pose.position) && std::isfinite(norm2) && std::abs(norm2 - 1.0) <= 1e-3;
}

bool DebugDrawBatch::Reserve() {
    if (Remaining() != 0u) return true;
    ++report_.omitted_instances;
    return false;
}

bool DebugDrawBatch::ShouldReadContacts(uint32_t capacity) {
    report_.contacts_budget_skipped = capacity != 0u && Remaining() == 0u;
    return capacity != 0u && !report_.contacts_budget_skipped;
}

void DebugDrawBatch::AppendCollider(render::RenderWorld& world, uint32_t kind,
                                    const float* params, const Transform& pose, uint32_t material) {
    if (kind > kPlane) { ++report_.unsupported_shapes; return; }
    const uint32_t count = kind == kSphere ? 1u : (kind == kBox ? 3u : 2u);
    float p[3] = {};
    for (uint32_t i = 0; i < count; ++i) {
        if (!std::isfinite(params[i])) { RejectCollider(); return; }
        p[i] = params[i] == 0.0f ? 0.0f : params[i];
    }
    if (kind == kPlane) {
        // An infinite plane is represented by the existing finite 2 m half-extent patch.
        for (uint32_t i = 0; i < 2u; ++i) if (p[i] <= 0.0f) p[i] = 2.0f;
    } else if (p[0] <= 0.0f || (kind == kCapsule && p[1] < 0.0f) ||
               (kind == kBox && (p[1] <= 0.0f || p[2] <= 0.0f))) {
        RejectCollider(); return;
    }
    if (!IsValidDebugPose(pose)) { RejectCollider(); return; }
    Vec3 extent{p[0], p[0], p[0]};
    if (kind == kCapsule) extent.z += p[1];
    if (kind == kBox) extent = {p[0], p[1], p[2]};
    if (kind == kPlane) extent = {p[0], p[1], 0.0f};
    for (int i = 0; i < 8; ++i) {
        const Vec3 corner{(i & 1) ? extent.x : -extent.x, (i & 2) ? extent.y : -extent.y,
                          (i & 4) ? extent.z : -extent.z};
        if (!Finite(pose.TransformPoint(corner))) { RejectCollider(); return; }
    }
    if (!Reserve()) return;
    char key[80];
    std::snprintf(key, sizeof(key), "dbgcol:%u:%08x:%08x:%08x", kind,
                  std::bit_cast<uint32_t>(p[0]), std::bit_cast<uint32_t>(p[1]),
                  std::bit_cast<uint32_t>(p[2]));
    const uint32_t mesh = world.meshes.InternPrimitive(key, [&] {
        switch (kind) {
            case kSphere: return MakeSphere(p[0]);
            case kCapsule: return MakeCapsule(p[0], p[1]);
            case kBox: return MakeBox(p[0], p[1], p[2]);
            default: return MakePlane(p[0], p[1]);
        }
    });
    world.debug_instances.push_back(MakeDebugInstance(mesh, material, pose));
    ++report_.colliders;
}

void DebugDrawBatch::AppendContact(render::RenderWorld& world, const Vec3& point, uint32_t material) {
    if (!Finite(point)) { ++report_.invalid_contacts; return; }
    if (!Reserve()) return;
    const uint32_t mesh = world.meshes.InternPrimitive(
        "dbgcontact:marker", [] { return MakeSphere(kContactMarkerRadius, 8u, 10u); });
    world.debug_instances.push_back(MakeDebugInstance(mesh, material, {point, math::Quat::Identity()}));
    ++report_.contacts;
}

}  // namespace nuka::runtime::app::viewer
