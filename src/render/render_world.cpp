// ---------------------------------------------------------------------------
// nuka::render::RenderWorld implementation (M8 manifest #1).
//
// BuildRenderWorld walks the ECS Registry for every renderable entity. The
// authored appearance is the VisualMeshComponent's real .nka MESH triangles;
// CollisionShapeComponents are rendered as primitive tessellations ONLY in a
// scene that has no visual meshes at all (a pure-collision / synthetic scene),
// so a mesh-bearing robot is not occluded by its physics proxies (M8.5 beauty
// pass). It resolves each instance's PBR material, its cached visual_local offset
// (the composed node chain from the physics-bound ancestor down to the visual
// node, Decision D1), and its pose source (LinkPose/BodyPose/Static, resolved
// ONCE so the per-frame publisher never re-walks SceneMap).
//
// HOST-ONLY, ZERO CUDA: this file reads the .nka MESH chunk through the pure
// host NkaFile/DecodeMesh path and tessellates primitives in plain C++. No
// triple-chevron kernel launch, no cuda-runtime call, no cuda_runtime include,
// no phi backend_cuda include (enforced by the src/render/** lint red-line).
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/asset/nka.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_map.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace nuka::render {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// -- node-chain helpers ------------------------------------------------------
// The visual entity sits as a node in the SceneGraph; its TransformComponent
// is "relative to the PARENT NODE" (components.hpp:31). We compose the chain of
// TransformComponent.local from a node UP toward (but excluding) a stop node, or
// up to the root. The Registry holds the entity->SceneNode backref (NodeOf).

// Local transform of an entity (identity when it has no TransformComponent).
math::Transform LocalOf(const scene::Registry& reg, scene::EntityId e) {
    if (const auto* t = reg.Get<scene::TransformComponent>(e)) {
        return t->local;
    }
    return math::Transform::Identity();
}

// Compose the world transform of `node` by walking parent links to the root,
// accumulating each node's entity-local transform: world = L_root * ... * L_node.
math::Transform NodeWorldTransform(const scene::Registry& reg,
                                   const std::shared_ptr<scene::SceneNode>& node) {
    // Collect the chain root..node, then compose root-first.
    std::vector<scene::EntityId> chain;
    for (auto n = node; n; n = n->parent.lock()) {
        if (n->entity != scene::kInvalidEntity) {
            chain.push_back(n->entity);
        }
    }
    math::Transform world = math::Transform::Identity();
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        world = world * LocalOf(reg, *it);
    }
    return world;
}

// Compose the transform from `stop` node's frame DOWN to `node`'s frame:
// visual_local = L_(stop.child) * ... * L_node, i.e. the product of every
// entity-local transform strictly BELOW `stop` on the path to `node`. When
// `stop == node` (the visual IS the physics node) this is identity. When `stop`
// is null (no physics-bound ancestor) it degrades to the full world chain.
math::Transform LocalRelativeTo(const scene::Registry& reg,
                                const std::shared_ptr<scene::SceneNode>& node,
                                const std::shared_ptr<scene::SceneNode>& stop) {
    std::vector<scene::EntityId> chain;
    for (auto n = node; n; n = n->parent.lock()) {
        if (stop && n == stop) {
            break;  // stop frame reached: do NOT include the stop node's local
        }
        if (n->entity != scene::kInvalidEntity) {
            chain.push_back(n->entity);
        }
    }
    math::Transform t = math::Transform::Identity();
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        t = t * LocalOf(reg, *it);
    }
    return t;
}

// -- pose-source resolution --------------------------------------------------
// Walk from a visual node up the tree to the NEAREST ancestor entity that the
// cook bound in the SceneMap with a usable pose row. Returns that node (the
// "physics node") + the resolved PoseSource. Link rows win over body rows (an
// articulation link's FK pose is the authoritative driver); a bound body_row is
// the movable-rigid-body fallback; otherwise Static.
struct ResolvedPose {
    PoseSource                       source;       // where the live pose comes from
    std::shared_ptr<scene::SceneNode> physics_node; // the frame visual_local is relative to (may be null)
};

ResolvedPose ResolvePoseSource(const scene::SceneMap& map,
                               const std::shared_ptr<scene::SceneNode>& visual_node) {
    ResolvedPose out;
    // A link ancestor anywhere up the chain drives the pose; a body_row is only
    // the fallback for a free rigid body with no link ancestor.
    bool have_body = false;
    PoseSource body;
    std::shared_ptr<scene::SceneNode> body_node;
    for (auto n = visual_node; n; n = n->parent.lock()) {
        if (n->entity == scene::kInvalidEntity) {
            continue;
        }
        const scene::CookedRef* ref = map.RefOf(n->entity);
        if (!ref) {
            continue;
        }
        if (ref->link_index != scene::SceneMap::kNoRow) {
            out.source.kind = PoseSource::Kind::Link;
            out.source.row  = ref->link_index;
            out.physics_node = n;
            return out;
        }
        if (!have_body && ref->body_row != scene::SceneMap::kNoRow) {
            body.kind = PoseSource::Kind::Body;
            body.row  = ref->body_row;
            body_node = n;
            have_body = true;
        }
    }
    if (have_body) {
        out.source = body;
        out.physics_node = body_node;
        return out;
    }
    // No physics-bound ancestor: static instance pinned at its bind pose.
    out.source.kind = PoseSource::Kind::Static;
    out.source.row  = kNoId;
    out.physics_node = nullptr;
    return out;
}

// -- primitive tessellators (host C++) --------------------------------------
// Each returns a MeshGeometry with positions + flat per-vertex normals + an
// index buffer. Deterministic vertex/index order (D1-safe). UVs are omitted
// (the renderer may synthesize them); the raster/RT consumer needs only
// positions + normals to shade a primitive.

void PushTri(MeshGeometry& m, const math::Vec3& a, const math::Vec3& b, const math::Vec3& c) {
    const math::Vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
    const math::Vec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
    math::Vec3 n{e1.y * e2.z - e1.z * e2.y,
                 e1.z * e2.x - e1.x * e2.z,
                 e1.x * e2.y - e1.y * e2.x};
    const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-12f) { n.x /= len; n.y /= len; n.z /= len; }
    const uint32_t base = m.VertexCount();
    const math::Vec3 verts[3] = {a, b, c};
    for (const auto& v : verts) {
        m.positions.push_back(v.x); m.positions.push_back(v.y); m.positions.push_back(v.z);
        m.normals.push_back(n.x);   m.normals.push_back(n.y);   m.normals.push_back(n.z);
    }
    m.indices.push_back(base); m.indices.push_back(base + 1); m.indices.push_back(base + 2);
}

void PushQuad(MeshGeometry& m, const math::Vec3& a, const math::Vec3& b,
              const math::Vec3& c, const math::Vec3& d) {
    PushTri(m, a, b, c);
    PushTri(m, a, c, d);
}

MeshGeometry MakeBox(float hx, float hy, float hz) {
    MeshGeometry m;
    const math::Vec3 p[8] = {
        {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
        {-hx, -hy,  hz}, {hx, -hy,  hz}, {hx, hy,  hz}, {-hx, hy,  hz},
    };
    PushQuad(m, p[0], p[3], p[2], p[1]);  // -z
    PushQuad(m, p[4], p[5], p[6], p[7]);  // +z
    PushQuad(m, p[0], p[1], p[5], p[4]);  // -y
    PushQuad(m, p[3], p[7], p[6], p[2]);  // +y
    PushQuad(m, p[0], p[4], p[7], p[3]);  // -x
    PushQuad(m, p[1], p[2], p[6], p[5]);  // +x
    return m;
}

MeshGeometry MakeSphere(float r, uint32_t stacks = 12, uint32_t slices = 16) {
    MeshGeometry m;
    auto at = [&](uint32_t i, uint32_t j) -> math::Vec3 {
        const float v = kPi * static_cast<float>(i) / static_cast<float>(stacks);
        const float u = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
        return {r * std::sin(v) * std::cos(u), r * std::sin(v) * std::sin(u), r * std::cos(v)};
    };
    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < slices; ++j) {
            const math::Vec3 a = at(i, j), b = at(i + 1, j),
                             c = at(i + 1, j + 1), d = at(i, j + 1);
            PushQuad(m, a, b, c, d);
        }
    }
    return m;
}

// Capsule along the local Z axis: cylinder of half-height hh + two hemispherical
// caps of radius r (matches the engine's capsule convention).
MeshGeometry MakeCapsule(float r, float hh, uint32_t slices = 16, uint32_t cap_stacks = 6) {
    MeshGeometry m;
    // Cylindrical side.
    for (uint32_t j = 0; j < slices; ++j) {
        const float u0 = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
        const float u1 = 2.0f * kPi * static_cast<float>(j + 1) / static_cast<float>(slices);
        const math::Vec3 a{r * std::cos(u0), r * std::sin(u0), -hh};
        const math::Vec3 b{r * std::cos(u1), r * std::sin(u1), -hh};
        const math::Vec3 c{r * std::cos(u1), r * std::sin(u1),  hh};
        const math::Vec3 d{r * std::cos(u0), r * std::sin(u0),  hh};
        PushQuad(m, a, b, c, d);
    }
    // Hemispherical caps (top at +hh, bottom at -hh).
    auto cap = [&](float zc, float sign) {
        for (uint32_t i = 0; i < cap_stacks; ++i) {
            const float v0 = 0.5f * kPi * static_cast<float>(i) / static_cast<float>(cap_stacks);
            const float v1 = 0.5f * kPi * static_cast<float>(i + 1) / static_cast<float>(cap_stacks);
            for (uint32_t j = 0; j < slices; ++j) {
                const float u0 = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
                const float u1 = 2.0f * kPi * static_cast<float>(j + 1) / static_cast<float>(slices);
                auto pt = [&](float vv, float uu) -> math::Vec3 {
                    return {r * std::sin(vv) * std::cos(uu),
                            r * std::sin(vv) * std::sin(uu),
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

// Area-weighted SMOOTH vertex normals from positions + indices. Used when a
// cooked .nka MESH carries no (or all-zero, T5-flagged) normals -- without this
// the shader's normalize() of a zero normal yields NaN and the surface renders
// black. Smooth (vertex-averaged) normals read better than flat on the curved
// robot shells. Deterministic accumulation order (D1-safe).
std::vector<float> SmoothNormals(const std::vector<float>& positions,
                                 const std::vector<uint32_t>& indices) {
    std::vector<float> normals(positions.size(), 0.0f);
    auto at = [&](uint32_t v, int c) { return positions[static_cast<size_t>(v) * 3 + c]; };
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        const math::Vec3 pa{at(a, 0), at(a, 1), at(a, 2)};
        const math::Vec3 pb{at(b, 0), at(b, 1), at(b, 2)};
        const math::Vec3 pc{at(c, 0), at(c, 1), at(c, 2)};
        // Un-normalized cross => area-weighted contribution (larger faces weigh more).
        const math::Vec3 fn{
            (pb.y - pa.y) * (pc.z - pa.z) - (pb.z - pa.z) * (pc.y - pa.y),
            (pb.z - pa.z) * (pc.x - pa.x) - (pb.x - pa.x) * (pc.z - pa.z),
            (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)};
        for (uint32_t v : {a, b, c}) {
            normals[static_cast<size_t>(v) * 3 + 0] += fn.x;
            normals[static_cast<size_t>(v) * 3 + 1] += fn.y;
            normals[static_cast<size_t>(v) * 3 + 2] += fn.z;
        }
    }
    for (size_t v = 0; v + 2 < normals.size(); v += 3) {
        const float len = std::sqrt(normals[v] * normals[v] +
                                    normals[v + 1] * normals[v + 1] +
                                    normals[v + 2] * normals[v + 2]);
        if (len > 1e-12f) {
            normals[v] /= len; normals[v + 1] /= len; normals[v + 2] /= len;
        } else {
            normals[v] = 0.0f; normals[v + 1] = 0.0f; normals[v + 2] = 1.0f;
        }
    }
    return normals;
}

// True when the normal stream is missing, length-mismatched, or DEGENERATE
// (every component ~0 -- the T5 zero-filled-normals cook gap).
bool NormalsUnusable(const std::vector<float>& positions, const std::vector<float>& normals) {
    if (normals.size() != positions.size() || normals.empty()) return true;
    for (float c : normals) {
        if (std::fabs(c) > 1e-6f) return false;  // a real normal exists -> usable
    }
    return true;  // all-zero
}

// Convert a decoded .nka MESH into our backend-agnostic MeshGeometry. Synthesizes
// smooth normals when the cooked stream is absent or all-zero so downstream
// shading (raster + M11 RT) never normalizes a zero vector.
MeshGeometry FromNkaMesh(const scene::NkaMesh& src) {
    MeshGeometry m;
    m.positions = src.positions;
    m.uvs       = src.uvs;
    m.indices   = src.indices;
    m.normals   = NormalsUnusable(src.positions, src.normals)
                      ? SmoothNormals(src.positions, src.indices)
                      : src.normals;
    return m;
}

// Stable dedup key for a primitive-fallback mesh.
std::string PrimKey(const char* kind, float a, float b, float c) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "prim:%s:%.6g,%.6g,%.6g", kind, a, b, c);
    return std::string(buf);
}

}  // namespace

RenderWorld BuildRenderWorld(const scene::Registry& registry, const scene::SceneMap& map) {
    RenderWorld world;

    // -- material table: copy the Registry's render-material assets verbatim, in
    //    id order, then append a shared default for instances with no material.
    const uint32_t mat_count = static_cast<uint32_t>(registry.RenderMaterialCount());
    world.materials.reserve(mat_count + 1);
    for (uint32_t id = 0; id < mat_count; ++id) {
        if (const scene::RenderMaterial* m = registry.GetRenderMaterial(id)) {
            world.materials.push_back(*m);
        } else {
            world.materials.push_back(scene::RenderMaterial{});
        }
    }
    world.default_material_id = static_cast<uint32_t>(world.materials.size());
    world.materials.push_back(scene::RenderMaterial{});  // engine default grey

    // Cache decoded .nka containers so repeated MESH loads of the same file open
    // it once. Host-only NkaFile; throws on a missing/bad container -> we catch
    // and fall back so a bad ref never blocks the gate (D3).
    std::unordered_map<std::string, std::shared_ptr<scene::NkaFile>> nka_cache;
    auto open_nka = [&](const std::string& path) -> std::shared_ptr<scene::NkaFile> {
        auto it = nka_cache.find(path);
        if (it != nka_cache.end()) {
            return it->second;
        }
        std::shared_ptr<scene::NkaFile> file;
        try {
            file = std::make_shared<scene::NkaFile>(scene::NkaFile::Open(path));
        } catch (const std::exception&) {
            file = nullptr;  // missing/bad container -> fallback at the call site
        }
        nka_cache.emplace(path, file);
        return file;
    };

    auto resolve_material = [&](uint32_t component_material_id) -> uint32_t {
        if (component_material_id == scene::Registry::kNoSlot ||
            component_material_id >= mat_count) {
            return world.default_material_id;
        }
        return component_material_id;
    };

    // Collision primitives render only as a fallback when NO real visual mesh loaded
    // (a non-empty set suppresses them: a mesh's proxy is a redundant wrong-sized occluder).
    std::unordered_set<scene::EntityId, scene::EntityIdHash> has_visual_mesh;

    // -- emit one RenderInstance per renderable entity -----------------------
    auto build_common = [&](scene::EntityId e, uint32_t mesh_id, uint32_t material_id) {
        RenderInstance inst;
        inst.entity             = e;
        inst.mesh_id            = mesh_id;
        inst.render_material_id = material_id;

        const std::shared_ptr<scene::SceneNode> node = registry.NodeOf(e);
        const ResolvedPose rp = ResolvePoseSource(map, node);
        inst.pose_source = rp.source;
        // visual_local = transform from the physics frame down to the visual node
        // (Decision D1). With no physics ancestor, this collapses to identity and
        // the bind-pose world_xform below carries the full placement (Static).
        inst.cached_visual_local =
            (rp.physics_node ? LocalRelativeTo(registry, node, rp.physics_node)
                             : math::Transform::Identity());
        // Bind pose: the full node-chain world transform from the root, so the
        // first frame (before any sync) already shows the rest placement.
        inst.world_xform = NodeWorldTransform(registry, node);
        world.instances.push_back(std::move(inst));
    };

    // VisualMeshComponent entities.
    registry.ForEach<scene::VisualMeshComponent>(
        [&](scene::EntityId e, const scene::VisualMeshComponent& vis) {
            if (vis.mesh.Empty() || vis.mesh.fourcc != scene::NkaTagMesh()) {
                // No bound MESH: tessellate a renderable visual primitive from its
                // params (does NOT mark has_visual_mesh, so pure-collision render holds).
                using PK = scene::VisualMeshComponent::PrimKind;
                if (vis.prim_kind == PK::None) {
                    // Still asset-gated / non-renderable: skip (no misleading cube).
                    return;
                }
                // Slot semantics are kind-specific (see VisualMeshComponent doc):
                // sphere(r) / capsule(r,half_height) / box|plane(hx,hy,hz). Name the
                // slots per-branch so no caller reads the wrong index silently.
                uint32_t prim_mesh_id;
                switch (vis.prim_kind) {
                    case PK::Box:
                    case PK::Plane: {
                        const float hx = vis.prim_params[0], hy = vis.prim_params[1],
                                    hz = vis.prim_params[2];
                        prim_mesh_id = world.meshes.InternPrimitive(
                            PrimKey("vbox", hx, hy, hz), [&]() { return MakeBox(hx, hy, hz); });
                        break;
                    }
                    case PK::Sphere: {
                        const float r = vis.prim_params[0];
                        prim_mesh_id = world.meshes.InternPrimitive(
                            PrimKey("vsphere", r, 0, 0), [&]() { return MakeSphere(r); });
                        break;
                    }
                    case PK::Capsule: {
                        const float r = vis.prim_params[0], half_height = vis.prim_params[1];
                        prim_mesh_id = world.meshes.InternPrimitive(
                            PrimKey("vcapsule", r, half_height, 0),
                            [&]() { return MakeCapsule(r, half_height); });
                        break;
                    }
                    default:
                        return;
                }
                build_common(e, prim_mesh_id, resolve_material(vis.render_material_id));
                return;
            }
            std::shared_ptr<scene::NkaFile> file = open_nka(vis.mesh.nka_path);
            if (!file || file->CountChunks(scene::NkaTagMesh()) <= vis.mesh.index) {
                // The MESH ref points at a missing/short container: skip (no box).
                return;
            }
            const uint32_t mesh_id = world.meshes.InternNkaMesh(vis.mesh, [&]() {
                return FromNkaMesh(scene::DecodeMesh(
                    file->LoadChunk(scene::NkaTagMesh(), vis.mesh.index)));
            });
            has_visual_mesh.insert(e);  // a real mesh loaded -> proxies now redundant
            build_common(e, mesh_id, resolve_material(vis.render_material_id));
        });

    // Collision-proxy fallback (only when no visual mesh loaded): box/sphere/capsule
    // tessellate from params; hull/SDF carry no host geometry and are skipped.
    using CK = scene::CollisionShapeComponent::Kind;
    if (has_visual_mesh.empty())
    registry.ForEach<scene::CollisionShapeComponent>(
        [&](scene::EntityId e, const scene::CollisionShapeComponent& cs) {
            uint32_t mesh_id;
            switch (cs.kind) {
                case CK::Box:
                case CK::Plane: {
                    const float hx = cs.params[0], hy = cs.params[1], hz = cs.params[2];
                    mesh_id = world.meshes.InternPrimitive(
                        PrimKey("box", hx, hy, hz),
                        [&]() { return MakeBox(hx, hy, hz); });
                    break;
                }
                case CK::Sphere: {
                    const float r = cs.params[0];
                    mesh_id = world.meshes.InternPrimitive(
                        PrimKey("sphere", r, 0, 0),
                        [&]() { return MakeSphere(r); });
                    break;
                }
                case CK::Capsule: {
                    const float r = cs.params[0], hh = cs.params[1];
                    mesh_id = world.meshes.InternPrimitive(
                        PrimKey("capsule", r, hh, 0),
                        [&]() { return MakeCapsule(r, hh); });
                    break;
                }
                case CK::ConvexHull:
                case CK::SdfMesh:
                default: {
                    // No host geometry available from the Registry alone: skip
                    // rather than emit a misleading placeholder box (the cooked
                    // hull/SDF triangles bind to .nka MESH in a later asset pass).
                    return;
                }
            }
            // CollisionShapeComponent carries only a PHYSICS material id, not a
            // render material; collision-primitive instances use the shared
            // default render material.
            build_common(e, mesh_id, world.default_material_id);
        });

    // -- cameras -------------------------------------------------------------
    registry.ForEach<scene::CameraComponent>(
        [&](scene::EntityId e, const scene::CameraComponent& cam) {
            RenderCamera rc;
            rc.entity               = e;
            rc.vertical_fov_degrees = cam.vertical_fov_degrees;
            rc.near_clip            = cam.near_clip;
            rc.far_clip             = cam.far_clip;
            const auto node = registry.NodeOf(e);
            const ResolvedPose rp = ResolvePoseSource(map, node);
            rc.pose_source = rp.source;
            // The CameraComponent.local_transform is the camera's offset within
            // its own node; fold it into the visual_local so a body-attached
            // camera tracks the body each frame (downloaded_pose ∘ visual_local).
            const math::Transform node_local =
                (rp.physics_node ? LocalRelativeTo(registry, node, rp.physics_node)
                                 : math::Transform::Identity());
            rc.cached_visual_local = node_local * cam.local_transform;
            rc.world_xform = NodeWorldTransform(registry, node) * cam.local_transform;
            world.cameras.push_back(rc);
        });

    // -- lights --------------------------------------------------------------
    registry.ForEach<scene::LightComponent>(
        [&](scene::EntityId e, const scene::LightComponent& light) {
            RenderLight rl;
            rl.entity    = e;
            rl.type      = light.type;
            rl.color     = light.color;
            rl.intensity = light.intensity;
            const auto node = registry.NodeOf(e);
            const ResolvedPose rp = ResolvePoseSource(map, node);
            rl.pose_source = rp.source;
            const math::Transform node_local =
                (rp.physics_node ? LocalRelativeTo(registry, node, rp.physics_node)
                                 : math::Transform::Identity());
            rl.cached_visual_local = node_local * light.local_transform;
            rl.world_xform = NodeWorldTransform(registry, node) * light.local_transform;
            world.lights.push_back(rl);
        });

    return world;
}

}  // namespace nuka::render
