// ---------------------------------------------------------------------------
// h1_grasp_video.cpp -- M10 Phase A: THE HEADLINE DEMO.
//
// A unified, offscreen C++ video tool that renders the PROVEN 51-DOF H1 cup-
// grasp (approach -> close -> lift -> sustained friction-only hold) to a PBR-
// beautiful mp4 on lavapipe-CPU, with VISUAL meshes (not collision-primitive
// "stick figure" wireframe). The single tool == loaded scene (.nks) + a per-
// frame CONTROL SOURCE (here the union-cook TableTorque choreography, shared
// verbatim with the h1_grasp_lift gate via h1_grasp_driver.hpp).
//
// PIPELINE (m10-recon §2):
//   nks::Load(h1_cup_table.nks)               -- FROZEN scene (read-only)
//     -> CookSceneToUnionTemplate(N=1)        -- the 51-DOF union cook (physics)
//     -> BuildNkUnionModel -> nk::World        -- the device core
//   + a HAND-BUILT RenderWorld (CR-2 option A): the union cook emits NO SceneMap,
//     so we bind each H1 visual geom's triangles to its union LINK row (via
//     gripper_proto.link_body name match) and the cup hull to the cup BODY row.
//   PER FRAME: DriveStep(s) -> publish FK (DownloadField LinkPose/BodyPose/Base
//     -> world_xform = fk * cached_visual_local) -> VulkanRasterRenderer::Render
//     (PBR + studio ground + hero camera) -> WritePpmP6 frame_%06d.ppm.
//
// The grasp PHYSICS is byte-for-byte the gated h1_grasp_lift path (D1-frozen).
// The visual triangles come from the live MJCF/USD import in the FROZEN .nks
// (mesh_vertices populated by nks Load -> ApplyImports -> LoadMjcf/LoadUsd); the
// frozen .nks/.nka are NEVER edited. Built behind NK_BUILD_VULKAN_VALIDATION.
//
// USAGE:
//   h1_grasp_video [--frames N] [--out-dir DIR] [--width W] [--height H]
//                  [--probe]   (probe: load+cook+bind, print diag, render 1 frame)
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "examples/demo/h1_grasp_driver.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/asset/nka.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_nk_model.hpp"
#include "scene/cook/union_scene_constants.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace render = nuka::render;
namespace cook = nuka::scene::cook;
namespace coresident = nuka::runtime::coresident;
namespace demo = nuka::demo;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";
// The COMMITTED self-contained H1 visual asset (owner OD-2: RAW commit for
// fresh-clone reproducibility). Cooked from h1_with_hand.xml via nuka_cook_scene;
// its sidecar .nka carries all 49 visual MESH chunks (no .nuka-assets/ STL dep,
// no `imports` block). When present the demo renders the H1 from THIS file; else
// it falls back to the live .nuka-assets/ STL re-import in the frozen scene.
// PHYSICS always stays on the frozen h1_cup_table.nks union cook (untouched).
constexpr const char* kVisualNksPath = "examples/scenes/h1_visual.nks";

// Strip the import attach prefix ("h1/pelvis" -> "pelvis") so a union link's
// frozen-scene body name matches the committed visual scene's unprefixed name.
std::string StripPrefix(const std::string& n) {
    const auto slash = n.find_last_of('/');
    return (slash == std::string::npos) ? n : n.substr(slash + 1);
}

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t frames = 420u;
    std::string out_dir = "/tmp/h1_grasp_frames";
    uint32_t width = 1280u;
    uint32_t height = 720u;
    bool probe = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](uint32_t def) -> uint32_t {
            return (i + 1 < argc) ? static_cast<uint32_t>(std::atoi(argv[++i])) : def;
        };
        if (s == "--frames") a.frames = next(a.frames);
        else if (s == "--width") a.width = next(a.width);
        else if (s == "--height") a.height = next(a.height);
        else if (s == "--probe") a.probe = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
    }
    return a;
}

// ---- a flat-shaded axis-aligned box (the table top) ------------------------
// Builds a render::MeshGeometry for a box of the given half-extents centered at
// the local origin, with per-face flat normals (so the PBR shader gets crisp
// table edges). Local; positioned in the world via the instance world_xform.
render::MeshGeometry MakeBoxMesh(float hx, float hy, float hz) {
    render::MeshGeometry m;
    const Vec3 p[8] = {
        {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
        {-hx, -hy,  hz}, {hx, -hy,  hz}, {hx, hy,  hz}, {-hx, hy,  hz},
    };
    auto quad = [&](int a, int b, int c, int d, const Vec3& n) {
        const int idx[6] = {a, b, c, a, c, d};
        const uint32_t base = m.VertexCount();
        for (int k : idx) {
            m.positions.push_back(p[k].x);
            m.positions.push_back(p[k].y);
            m.positions.push_back(p[k].z);
            m.normals.push_back(n.x); m.normals.push_back(n.y); m.normals.push_back(n.z);
        }
        for (uint32_t k = 0; k < 6; ++k) m.indices.push_back(base + k);
    };
    quad(0, 3, 2, 1, {0.0f, 0.0f, -1.0f});  // -z
    quad(4, 5, 6, 7, {0.0f, 0.0f,  1.0f});  // +z
    quad(0, 1, 5, 4, {0.0f, -1.0f, 0.0f});  // -y
    quad(3, 7, 6, 2, {0.0f,  1.0f, 0.0f});  // +y
    quad(0, 4, 7, 3, {-1.0f, 0.0f, 0.0f});  // -x
    quad(1, 2, 6, 5, { 1.0f, 0.0f, 0.0f});  // +x
    return m;
}

// ---- visual-mesh geometry from a SceneIR shape record ----------------------
// The H1 STL visual geoms (contype==0 && conaffinity==0) carry their triangles
// in CollisionShapeRecord::mesh_vertices after nks Load -> ApplyImports ->
// LoadMjcf. Build a render::MeshGeometry from those triangles (positions +
// derived smooth normals so the PBR shader never normalizes a zero vector).
render::MeshGeometry MeshFromRecordTris(const std::vector<float>& verts,
                                        const std::vector<uint32_t>& idx) {
    render::MeshGeometry m;
    m.positions = verts;
    m.indices = idx;
    // Area-weighted smooth normals (matches render_world.cpp SmoothNormals).
    m.normals.assign(verts.size(), 0.0f);
    auto at = [&](uint32_t v, int c) { return verts[static_cast<size_t>(v) * 3 + c]; };
    for (size_t i = 0; i + 2 < idx.size(); i += 3) {
        const uint32_t a = idx[i], b = idx[i + 1], c = idx[i + 2];
        const Vec3 pa{at(a, 0), at(a, 1), at(a, 2)};
        const Vec3 pb{at(b, 0), at(b, 1), at(b, 2)};
        const Vec3 pc{at(c, 0), at(c, 1), at(c, 2)};
        const Vec3 fn{(pb.y - pa.y) * (pc.z - pa.z) - (pb.z - pa.z) * (pc.y - pa.y),
                      (pb.z - pa.z) * (pc.x - pa.x) - (pb.x - pa.x) * (pc.z - pa.z),
                      (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)};
        for (uint32_t v : {a, b, c}) {
            m.normals[static_cast<size_t>(v) * 3 + 0] += fn.x;
            m.normals[static_cast<size_t>(v) * 3 + 1] += fn.y;
            m.normals[static_cast<size_t>(v) * 3 + 2] += fn.z;
        }
    }
    for (size_t v = 0; v + 2 < m.normals.size(); v += 3) {
        const float len = std::sqrt(m.normals[v] * m.normals[v] +
                                    m.normals[v + 1] * m.normals[v + 1] +
                                    m.normals[v + 2] * m.normals[v + 2]);
        if (len > 1e-12f) {
            m.normals[v] /= len; m.normals[v + 1] /= len; m.normals[v + 2] /= len;
        } else { m.normals[v] = 0.0f; m.normals[v + 1] = 0.0f; m.normals[v + 2] = 1.0f; }
    }
    return m;
}

// ---- compact deterministic 3D convex-hull triangulation -------------------
// The cup geometry the union cook carries is a POINT CLOUD (CoResidentCup::
// hull_verts, flat xyz, no topology) -- the cup's .nka SAMP sample cloud. To
// render the cup as a solid PBR surface we compute its convex hull faces with a
// simple incremental hull (O(n^2), fine for a few hundred cup samples) and
// triangulate them. Deterministic: same point cloud -> same triangles.
render::MeshGeometry ConvexHullTriangles(const std::vector<float>& verts) {
    render::MeshGeometry m;
    const size_t n = verts.size() / 3;
    if (n < 4) return m;
    auto P = [&](size_t i) {
        return Vec3{verts[i * 3 + 0], verts[i * 3 + 1], verts[i * 3 + 2]};
    };
    // A face is an oriented triangle (i,j,k) whose outward normal points away
    // from the hull interior (the centroid of the first 4 non-degenerate verts).
    struct Face { int a, b, c; Vec3 n; float d; };
    // pick 4 non-coplanar seed points.
    int i0 = 0, i1 = -1, i2 = -1, i3 = -1;
    for (size_t i = 1; i < n; ++i) {
        if ((P(i) - P(i0)).Length() > 1e-6f) { i1 = static_cast<int>(i); break; }
    }
    if (i1 < 0) return m;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 cr{
            (P(i1) - P(i0)).y * (P(i) - P(i0)).z - (P(i1) - P(i0)).z * (P(i) - P(i0)).y,
            (P(i1) - P(i0)).z * (P(i) - P(i0)).x - (P(i1) - P(i0)).x * (P(i) - P(i0)).z,
            (P(i1) - P(i0)).x * (P(i) - P(i0)).y - (P(i1) - P(i0)).y * (P(i) - P(i0)).x};
        if (cr.Length() > 1e-7f) { i2 = static_cast<int>(i); break; }
    }
    if (i2 < 0) return m;
    Vec3 nrm{
        (P(i1) - P(i0)).y * (P(i2) - P(i0)).z - (P(i1) - P(i0)).z * (P(i2) - P(i0)).y,
        (P(i1) - P(i0)).z * (P(i2) - P(i0)).x - (P(i1) - P(i0)).x * (P(i2) - P(i0)).z,
        (P(i1) - P(i0)).x * (P(i2) - P(i0)).y - (P(i1) - P(i0)).y * (P(i2) - P(i0)).x};
    float best = 1e-6f;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 d = P(i) - P(i0);
        const float dist = std::fabs(d.x * nrm.x + d.y * nrm.y + d.z * nrm.z);
        if (dist > best) { best = dist; i3 = static_cast<int>(i); }
    }
    if (i3 < 0) return m;
    const Vec3 centroid{
        (P(i0).x + P(i1).x + P(i2).x + P(i3).x) * 0.25f,
        (P(i0).y + P(i1).y + P(i2).y + P(i3).y) * 0.25f,
        (P(i0).z + P(i1).z + P(i2).z + P(i3).z) * 0.25f};
    auto make_face = [&](int a, int b, int c) -> Face {
        const Vec3 pa = P(a), pb = P(b), pc = P(c);
        Vec3 fn{(pb.y - pa.y) * (pc.z - pa.z) - (pb.z - pa.z) * (pc.y - pa.y),
                (pb.z - pa.z) * (pc.x - pa.x) - (pb.x - pa.x) * (pc.z - pa.z),
                (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)};
        const float fl = fn.Length();
        if (fl > 1e-12f) { fn.x /= fl; fn.y /= fl; fn.z /= fl; }
        // orient outward (away from centroid).
        const Vec3 toc{pa.x - centroid.x, pa.y - centroid.y, pa.z - centroid.z};
        if (fn.x * toc.x + fn.y * toc.y + fn.z * toc.z < 0.0f) {
            std::swap(b, c);
            fn.x = -fn.x; fn.y = -fn.y; fn.z = -fn.z;
        }
        Face f{a, b, c, fn, fn.x * P(a).x + fn.y * P(a).y + fn.z * P(a).z};
        return f;
    };
    std::vector<Face> faces = {make_face(i0, i1, i2), make_face(i0, i1, i3),
                               make_face(i0, i2, i3), make_face(i1, i2, i3)};
    auto above = [&](const Face& f, const Vec3& p) {
        return f.n.x * p.x + f.n.y * p.y + f.n.z * p.z - f.d > 1e-7f;
    };
    for (size_t pi = 0; pi < n; ++pi) {
        const Vec3 p = P(pi);
        std::vector<int> visible;
        for (size_t fi = 0; fi < faces.size(); ++fi)
            if (above(faces[fi], p)) visible.push_back(static_cast<int>(fi));
        if (visible.empty()) continue;
        // collect the horizon edges (edges on exactly one visible face).
        std::unordered_map<long long, int> edge_count;
        auto key = [](int u, int v) {
            int a = std::min(u, v), b = std::max(u, v);
            return static_cast<long long>(a) * 1000000LL + b;
        };
        std::vector<std::pair<int, int>> edges;  // directed, from visible faces
        for (int fi : visible) {
            const Face& f = faces[fi];
            const int tri[3][2] = {{f.a, f.b}, {f.b, f.c}, {f.c, f.a}};
            for (auto& e : tri) {
                ++edge_count[key(e[0], e[1])];
                edges.emplace_back(e[0], e[1]);
            }
        }
        // remove visible faces.
        std::vector<bool> vis_mask(faces.size(), false);
        for (int fi : visible) vis_mask[fi] = true;
        std::vector<Face> kept;
        for (size_t fi = 0; fi < faces.size(); ++fi)
            if (!vis_mask[fi]) kept.push_back(faces[fi]);
        // for each horizon edge (count==1) make a new face to p.
        for (auto& e : edges) {
            if (edge_count[key(e.first, e.second)] != 1) continue;
            kept.push_back(make_face(e.first, e.second, static_cast<int>(pi)));
        }
        faces.swap(kept);
        if (faces.size() > 20000) break;  // safety cap.
    }
    // emit triangles with per-face flat normals.
    for (const Face& f : faces) {
        const Vec3 tri[3] = {P(f.a), P(f.b), P(f.c)};
        const uint32_t base = m.VertexCount();
        for (const Vec3& v : tri) {
            m.positions.push_back(v.x); m.positions.push_back(v.y); m.positions.push_back(v.z);
            m.normals.push_back(f.n.x); m.normals.push_back(f.n.y); m.normals.push_back(f.n.z);
        }
        m.indices.push_back(base); m.indices.push_back(base + 1); m.indices.push_back(base + 2);
    }
    return m;
}

// One render instance + its publish source (which device row drives its pose).
struct DemoInstance {
    uint32_t mesh_id = 0u;
    uint32_t material_id = 0u;
    Transform visual_local = Transform::Identity();  // geom local in its body frame
    render::PoseSource::Kind kind = render::PoseSource::Kind::Static;
    uint32_t row = 0u;  // link row (Link) or body row (Body)
};

// World transform of a SceneIR body (walk parent chain; cup/table are roots).
Transform BodyWorldTransform(const nuka::scene::SceneIR& scene,
                             nuka::scene::BodyId body) {
    Transform out = Transform::Identity();
    nuka::scene::BodyId cur = body;
    while (cur != nuka::scene::kInvalidBody && cur < scene.Bodies().size()) {
        const auto& rec = scene.GetBody(cur);
        out = rec.local_transform * out;
        cur = rec.parent_id;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    if (!std::filesystem::exists(kNksPath)) {
        std::fprintf(stderr, "[h1_grasp_video] missing scene %s\n", kNksPath);
        return 2;
    }

    // ---- the cook (NO factory): Load(.nks) -> cook -> BuildNkUnionModel ------
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookedUnionScene cooked = cook::CookSceneToUnionTemplate(scene, 1);
    const coresident::UnionSceneTemplate& tmpl = cooked.tmpl;
    const uint32_t links = cooked.link_count;
    const uint32_t cup_local = tmpl.cup_local_index;
    const float dt = cook::kH1UnionDt;

    std::printf("[h1_grasp_video] cooked: dof_stride=%u links=%u cup_local=%u "
                "feet=%zu fingertips=%zu cup_mass=%.3f\n",
                cooked.dof_stride, links, cup_local, tmpl.feet.size(),
                tmpl.fingertips.size(), cooked.cup_mass);

    // ---- body NAME -> union device link, via gripper_proto.link_body --------
    // Each device link maps to a SceneIR BodyId; resolve the body name so a
    // visual geom (which owns a shape on a body) can find its driving link row.
    const auto& link_body = tmpl.gripper_proto.link_body;
    std::unordered_map<nuka::scene::BodyId, uint32_t> body_to_link;
    for (uint32_t l = 0; l < link_body.size() && l < links; ++l) {
        const uint32_t b = link_body[l];
        if (b < scene.Bodies().size()) body_to_link.emplace(b, l);
    }
    std::printf("[h1_grasp_video] link_body map: %zu links -> bodies\n",
                body_to_link.size());

    // ---- HAND-BUILD the RenderWorld (CR-2 option A) -------------------------
    render::RenderWorld rw;
    // material table: a tasteful default + a warm cup material.
    nuka::scene::RenderMaterial h1_mat;     // matte dark metal for the robot
    h1_mat.base_color[0] = 0.32f; h1_mat.base_color[1] = 0.34f;
    h1_mat.base_color[2] = 0.38f; h1_mat.base_color[3] = 1.0f;
    h1_mat.metallic = 0.55f; h1_mat.roughness = 0.42f;
    nuka::scene::RenderMaterial cup_mat;    // warm ceramic for the cup
    cup_mat.base_color[0] = 0.85f; cup_mat.base_color[1] = 0.45f;
    cup_mat.base_color[2] = 0.28f; cup_mat.base_color[3] = 1.0f;
    cup_mat.metallic = 0.05f; cup_mat.roughness = 0.55f;
    nuka::scene::RenderMaterial table_mat;  // warm matte wood for the table
    table_mat.base_color[0] = 0.55f; table_mat.base_color[1] = 0.40f;
    table_mat.base_color[2] = 0.27f; table_mat.base_color[3] = 1.0f;
    table_mat.metallic = 0.0f; table_mat.roughness = 0.78f;
    rw.materials.push_back(h1_mat);     // id 0
    rw.materials.push_back(cup_mat);    // id 1
    rw.materials.push_back(table_mat);  // id 2
    rw.default_material_id = 0u;

    std::vector<DemoInstance> sources;  // parallel to rw.instances (publish rows)

    // union link -> stripped body NAME (for matching against the committed visual
    // scene's unprefixed body names).
    std::unordered_map<std::string, uint32_t> name_to_link;
    for (uint32_t l = 0; l < link_body.size() && l < links; ++l) {
        const uint32_t b = link_body[l];
        if (b < scene.Bodies().size())
            name_to_link.emplace(StripPrefix(scene.GetBody(b).name), l);
    }

    auto add_h1_instance = [&](render::MeshGeometry&& geo, uint32_t link,
                               const Transform& vlocal, const std::string& key) {
        if (geo.Empty()) return;
        const uint32_t mesh_id = rw.meshes.Count();
        rw.meshes.InternPrimitive(key, [&]() { return std::move(geo); });
        render::RenderInstance inst;
        inst.mesh_id = mesh_id;
        inst.render_material_id = 0u;  // h1_mat
        inst.pose_source.kind = render::PoseSource::Kind::Link;
        inst.pose_source.row = link;
        inst.cached_visual_local = vlocal;
        rw.instances.push_back(inst);
        DemoInstance src;
        src.mesh_id = mesh_id;
        src.material_id = 0u;
        src.visual_local = vlocal;
        src.kind = render::PoseSource::Kind::Link;
        src.row = link;
        sources.push_back(src);
    };

    uint32_t h1_geoms = 0u, h1_skipped_noLink = 0u;
    const bool use_committed = std::filesystem::exists(kVisualNksPath);
    if (use_committed) {
        // --- REPRODUCIBLE PATH: render H1 meshes from the committed self-contained
        // h1_visual.nks (.nka MESH chunks); NO .nuka-assets/ dependency. ---
        const nuka::scene::SceneIR vscene = nuka::scene::nks::Load(kVisualNksPath);
        for (const auto& s : vscene.Shapes()) {
            const bool is_visual = (s.contype == 0u && s.conaffinity == 0u);
            if (!is_visual) continue;
            if (s.body_id == nuka::scene::kInvalidBody ||
                s.body_id >= vscene.Bodies().size())
                continue;
            const std::string nm = StripPrefix(vscene.GetBody(s.body_id).name);
            const auto it = name_to_link.find(nm);
            if (it == name_to_link.end()) { ++h1_skipped_noLink; continue; }
            // Load triangles: prefer the decoded MESH chunk (self-contained), else
            // the record's mesh_vertices (nks Load already resolved the MESH ref
            // into mesh_vertices when the .nka is present).
            render::MeshGeometry geo;
            if (!s.mesh_vertices.empty() && !s.mesh_indices.empty()) {
                geo = MeshFromRecordTris(s.mesh_vertices, s.mesh_indices);
            }
            if (geo.Empty()) continue;
            add_h1_instance(std::move(geo), it->second, s.local_transform,
                            "h1vis:" + std::to_string(s.id));
            ++h1_geoms;
        }
        std::printf("[h1_grasp_video] H1 meshes from COMMITTED %s: %u (skipped "
                    "no-link: %u)\n", kVisualNksPath, h1_geoms, h1_skipped_noLink);
    } else {
        // --- LOCAL PATH: render H1 from the frozen scene's live STL re-import
        // (mesh_vertices populated by nks Load -> ApplyImports -> LoadMjcf). ---
        for (const auto& s : scene.Shapes()) {
            const bool is_visual = (s.contype == 0u && s.conaffinity == 0u);
            if (!is_visual) continue;
            if (s.mesh_vertices.empty() || s.mesh_indices.empty()) continue;
            const auto it = body_to_link.find(s.body_id);
            if (it == body_to_link.end()) { ++h1_skipped_noLink; continue; }
            render::MeshGeometry geo =
                MeshFromRecordTris(s.mesh_vertices, s.mesh_indices);
            add_h1_instance(std::move(geo), it->second, s.local_transform,
                            "h1geom:" + std::to_string(s.id));
            ++h1_geoms;
        }
        std::printf("[h1_grasp_video] H1 meshes from LIVE .nuka-assets STL: %u "
                    "(skipped no-link: %u)\n", h1_geoms, h1_skipped_noLink);
    }

    // The cup: render its convex hull (the cup geometry the cook carries) driven
    // by the cup BODY row. The cook carries the cup as a COM-centered POINT CLOUD
    // (tmpl.cup.hull_verts, the .nka SAMP samples -- no triangles in the SceneIR
    // shape after Load offloaded them), so we triangulate its convex hull here.
    // The hull verts are mesh-local / COM-centered and the cup BodyPose tracks the
    // cup body origin, so the visual_local is identity.
    uint32_t cup_geoms = 0u;
    {
        const std::vector<float>& hv = tmpl.cup.hull_verts;
        render::MeshGeometry geo = ConvexHullTriangles(hv);
        if (!geo.Empty()) {
            const uint32_t mesh_id = rw.meshes.Count();
            rw.meshes.InternPrimitive("cup_hull", [&]() { return geo; });
            render::RenderInstance inst;
            inst.mesh_id = mesh_id;
            inst.render_material_id = 1u;  // cup_mat
            inst.pose_source.kind = render::PoseSource::Kind::Body;
            inst.pose_source.row = cup_local;
            inst.cached_visual_local = Transform::Identity();
            rw.instances.push_back(inst);
            DemoInstance src;
            src.mesh_id = mesh_id;
            src.material_id = 1u;
            src.visual_local = Transform::Identity();
            src.kind = render::PoseSource::Kind::Body;
            src.row = cup_local;
            sources.push_back(src);
            ++cup_geoms;
        }
        std::printf("[h1_grasp_video] cup hull verts=%zu -> tris=%u\n",
                    hv.size() / 3, geo.TriangleCount());
    }
    std::printf("[h1_grasp_video] cup geoms bound to body row %u: %u\n",
                cup_local, cup_geoms);

    // The TABLE: a STATIC root box body that gives the cup its "sitting on a
    // table" context. It is NOT a device link/body (it lives in the cook only as
    // a contact proxy), so we bind it as a Static render instance at its frozen
    // SceneIR world pose -- it never moves. We render every BOX shape on a root
    // (parent-less) body whose name mentions "table". The table-vs-cup-rest
    // validation below confirms the cup binding sits on this surface.
    uint32_t table_geoms = 0u;
    float table_top_z = 0.0f;
    for (const auto& s : scene.Shapes()) {
        if (s.type != nuka::scene::ShapeType::Box) continue;
        if (s.body_id == nuka::scene::kInvalidBody ||
            s.body_id >= scene.Bodies().size())
            continue;
        const auto& body = scene.GetBody(s.body_id);
        // root body only (the table is a root; H1 links have a parent).
        if (body.parent_id != nuka::scene::kInvalidBody) continue;
        const std::string bname = StripPrefix(body.name);
        const std::string pname = body.name;
        if (pname.find("table") == std::string::npos &&
            bname.find("table") == std::string::npos) {
            // also accept a root box that carries the union table proxy name.
            continue;
        }
        const Transform body_world = BodyWorldTransform(scene, s.body_id);
        const Transform shape_world = body_world * s.local_transform;
        render::MeshGeometry geo =
            MakeBoxMesh(s.half_extents.x, s.half_extents.y, s.half_extents.z);
        const uint32_t mesh_id = rw.meshes.Count();
        rw.meshes.InternPrimitive("table_box:" + std::to_string(s.id),
                                  [&]() { return geo; });
        render::RenderInstance inst;
        inst.mesh_id = mesh_id;
        inst.render_material_id = 2u;  // table_mat
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        inst.world_xform = shape_world;
        inst.cached_visual_local = Transform::Identity();
        rw.instances.push_back(inst);
        DemoInstance src;
        src.mesh_id = mesh_id;
        src.material_id = 2u;
        src.visual_local = shape_world;  // Static: published once as world_xform.
        src.kind = render::PoseSource::Kind::Static;
        src.row = 0u;
        sources.push_back(src);
        table_top_z = shape_world.position.z + s.half_extents.z;
        std::printf("[h1_grasp_video] TABLE '%s' box he=(%.3f,%.3f,%.3f) "
                    "center=(%.3f,%.3f,%.3f) top_z=%.4f\n",
                    body.name.c_str(), s.half_extents.x, s.half_extents.y,
                    s.half_extents.z, shape_world.position.x,
                    shape_world.position.y, shape_world.position.z, table_top_z);
        ++table_geoms;
    }
    std::printf("[h1_grasp_video] table geoms (static): %u\n", table_geoms);

    if (rw.instances.empty()) {
        std::fprintf(stderr, "[h1_grasp_video] FATAL: no render instances bound\n");
        return 3;
    }

    // ---- bring up the device + nk::World ------------------------------------
    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "[h1_grasp_video] no phi device\n"); return 4; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "[h1_grasp_video] no backend\n"); return 4; }

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = cook::kH1UnionGravityZ;
    cfg.vel_iters = 64;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;

    nk::Model model = coresident::BuildNkUnionModel(tmpl, 1u);
    nk::World w(std::move(model), 1u, dev, backend, cfg);
    if (!w.Ready()) { std::fprintf(stderr, "[h1_grasp_video] world !ready\n"); return 5; }

    // ---- the renderer (offscreen lavapipe; throws if no Vulkan) -------------
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[h1_grasp_video] no Vulkan device: %s\n", e.what());
        nphi::BackendFree(backend);
        return 6;
    }
    std::printf("[h1_grasp_video] renderer ICD: %s\n",
                renderer->DeviceName().c_str());

    // Hero camera: a tight 3/4 product shot framed on the MANIPULATION -- the
    // table surface, the cup, the grasping hand and the torso fill the frame; the
    // dangling fixed-base legs (the H1 hangs in the air) are cropped below the
    // bottom edge. hero_framing is OFF so this explicit camera fully controls the
    // shot (use_camera_override). Eye/target tuned below; overridable via env for
    // probe iteration (NK_CAM_EYE="x,y,z", NK_CAM_TGT="x,y,z", NK_CAM_FOV="deg").
    render::RasterOptions opts;
    opts.width = args.width;
    opts.height = args.height;
    opts.draw_ground = true;
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_eye = {1.50f, -1.64f, 1.74f};
    opts.camera_target = {0.41f, -0.15f, 1.16f};
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 33.0f;
    opts.background = {18, 20, 26, 255};
    // -- probe-only camera override from env (lets us iterate the hero shot
    //    without a rebuild; absent -> the tuned defaults above ship). --
    auto env_vec3 = [](const char* name, Vec3* out) {
        const char* v = std::getenv(name);
        if (!v) return;
        float a = 0, b = 0, c = 0;
        if (std::sscanf(v, "%f,%f,%f", &a, &b, &c) == 3) { *out = {a, b, c}; }
    };
    env_vec3("NK_CAM_EYE", &opts.camera_eye);
    env_vec3("NK_CAM_TGT", &opts.camera_target);
    if (const char* f = std::getenv("NK_CAM_FOV")) opts.camera_fov_degrees =
        static_cast<float>(std::atof(f));

    // ---- the per-frame publish: download FK -> compose world_xform ----------
    auto publish = [&]() {
        for (size_t i = 0; i < rw.instances.size(); ++i) {
            const DemoInstance& src = sources[i];
            Transform fk = Transform::Identity();
            if (src.kind == render::PoseSource::Kind::Link) {
                w.GetData().DownloadField(nk::FieldId::LinkPose, &fk, sizeof(Transform),
                                          static_cast<uint64_t>(src.row) * sizeof(Transform));
            } else if (src.kind == render::PoseSource::Kind::Body) {
                w.GetData().DownloadField(nk::FieldId::BodyPose, &fk, sizeof(Transform),
                                          static_cast<uint64_t>(src.row) * sizeof(Transform));
            }
            rw.instances[i].world_xform = fk * src.visual_local;
        }
    };

    auto write_ppm = [&](const render::VulkanOffscreenReport& rep,
                         const std::string& path) -> bool {
        if (rep.pixels.empty() || rep.width == 0u || rep.height == 0u) return false;
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        char hdr[64];
        const int hn = std::snprintf(hdr, sizeof(hdr), "P6\n%u %u\n255\n",
                                     rep.width, rep.height);
        bool ok = (hn > 0) &&
                  (std::fwrite(hdr, 1, static_cast<size_t>(hn), f) == (size_t)hn);
        std::vector<unsigned char> rgb(static_cast<size_t>(rep.width) * 3u);
        for (uint32_t y = 0; ok && y < rep.height; ++y) {
            for (uint32_t x = 0; x < rep.width; ++x) {
                const auto& p = rep.pixels[static_cast<size_t>(y) * rep.width + x];
                rgb[x * 3 + 0] = p.r; rgb[x * 3 + 1] = p.g; rgb[x * 3 + 2] = p.b;
            }
            if (std::fwrite(rgb.data(), 1, rgb.size(), f) != rgb.size()) ok = false;
        }
        if (std::fclose(f) != 0) ok = false;
        return ok;
    };

    // ---- PROBE: bind + publish bind-pose + render 1 frame, dump diag --------
    if (args.probe) {
        publish();  // bind-pose FK before stepping
        render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
        std::filesystem::create_directories(args.out_dir);
        const std::string p = args.out_dir + "/probe_frame.ppm";
        const bool ok = write_ppm(rep, p);
        std::printf("[h1_grasp_video] PROBE: instances=%u meshes=%u non_bg=%zu "
                    "(%ux%u) -> %s (%s)\n",
                    rw.InstanceCount(), rw.meshes.Count(),
                    rep.non_background_pixel_count, rep.width, rep.height,
                    p.c_str(), ok ? "OK" : "WRITE-FAIL");
        nphi::BackendFree(backend);
        return ok ? 0 : 7;
    }

    // ---- THE FUSED DRIVE + RENDER LOOP -------------------------------------
    // Arc: APPROACH/REST 0-100 -> CLOSE 100-180 -> LIFT 180 (table off) ->
    // HOLD 180-frames. Render EVERY step (8x slow-mo). End on the held cup.
    std::filesystem::create_directories(args.out_dir);
    std::vector<float> torque = demo::InitTorque(cooked, links);
    std::vector<float> q(links), qdot(links);
    uint32_t written = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    for (uint32_t s = 0; s < args.frames; ++s) {
        demo::DriveStep(w, cooked, links, s, /*kill_grip=*/false, &torque, &q, &qdot);
        publish();
        // Cup-on-table binding validation: at the pre-grasp rest (frame 5) the cup
        // BodyPose origin (the COM the hull is centered on) should sit just above
        // the rendered table top -- not floating high nor sunk into it.
        if (s == 5u || s == 179u || s == 180u || s == 220u || s == 300u ||
            s == 419u) {
            Transform cup_pose = Transform::Identity();
            w.GetData().DownloadField(nk::FieldId::BodyPose, &cup_pose,
                                      sizeof(Transform),
                                      static_cast<uint64_t>(cup_local) * sizeof(Transform));
            std::printf("[h1_grasp_video] CUP-Z @frame%u: cup_com_z=%.4f "
                        "table_top_z=%.4f cup_com_above_top=%.4f\n",
                        s, cup_pose.position.z, table_top_z,
                        cup_pose.position.z - table_top_z);
        }
        render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", s);
        const std::string path = args.out_dir + "/" + name;
        if (!write_ppm(rep, path)) {
            std::fprintf(stderr, "[h1_grasp_video] write fail @ %u\n", s);
            nphi::BackendFree(backend);
            return 7;
        }
        if (s == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        ++written;
        if ((s % 60u) == 0u)
            std::printf("[h1_grasp_video] frame %u/%u non_bg=%zu\n", s,
                        args.frames, rep.non_background_pixel_count);
    }
    std::printf("[h1_grasp_video] DONE: wrote %u frames to %s (first non_bg=%zu, "
                "last non_bg=%zu)\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg);

    nphi::BackendFree(backend);
    return 0;
}
