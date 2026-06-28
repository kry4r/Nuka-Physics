// ---------------------------------------------------------------------------
// Editor edit -> nks::Save -> reload round-trip gate.
//
// The editor mutates the authoritative SceneIR records (transform + material) and
// persists via nks::Save. This asserts those edits survive a save+reload AND that
// collision mesh geometry is NOT lost -- the LoadLightScene trap a naive Save would
// hit. A negative control documents that trap. The demo scene is also emitted to
// $NUKA_EDIT_DEMO_DIR (when set) for the headless inspector capture.
// ---------------------------------------------------------------------------

#include "scene/canonical_types.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace nuka::scene;
namespace fs = std::filesystem;

namespace {

// FNV-1a over a float field set so "field equality" is one comparable digest.
uint64_t Fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (float f : v) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<uint8_t>(bits >> (b * 8));
            h *= 1099511628211ull;
        }
    }
    return h;
}

std::vector<float> MaterialFields(const MaterialRecord& m) {
    return {m.base_color.x, m.base_color.y, m.base_color.z, m.alpha,
            m.roughness,    m.metallic,
            m.emissive.x,   m.emissive.y,   m.emissive.z,
            m.sheen,        m.transmission, m.ior,
            m.absorption.x, m.absorption.y, m.absorption.z};
}

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/editor_edit_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    fs::path File(const std::string& n) const { return path_ / n; }
private:
    fs::path path_;
};

// A clean editable scene: a vivid VISUAL box (renders + materialed + selectable),
// a grey ground, and a body carrying a collision MESH so the no-data-loss gate has
// real geometry to preserve.
SceneIR BuildEditableDemo() {
    SceneIR s;

    MaterialRecord red;
    red.name = "demo_red";
    red.base_color = {0.85f, 0.13f, 0.13f};
    red.roughness = 0.45f;
    red.metallic = 0.0f;
    const MaterialId red_id = s.AddMaterial(red);

    MaterialRecord gnd;
    gnd.name = "demo_ground";
    gnd.base_color = {0.46f, 0.48f, 0.52f};
    const MaterialId gnd_id = s.AddMaterial(gnd);

    // Ground: static collision box (renders as a grey proxy).
    RigidBodyRecord gb;
    gb.name = "ground";
    gb.is_static = true;
    const BodyId gbid = s.AddRigidBody(gb);
    CollisionShapeRecord gs;
    gs.body_id = gbid;
    gs.type = ShapeType::Box;
    gs.half_extents = {3.0f, 3.0f, 0.1f};
    gs.material_id = gnd_id;
    s.AddCollisionShape(gs);

    // The editable hero: a static body with a VISUAL box (contype/conaffinity 0)
    // bound to demo_red, so it renders red and its node owns the render instance.
    RigidBodyRecord bb;
    bb.name = "redbox";
    bb.is_static = true;
    bb.local_transform.position = {0.0f, 0.0f, 0.6f};
    const BodyId bbid = s.AddRigidBody(bb);
    CollisionShapeRecord vis;
    vis.body_id = bbid;
    vis.name = "skin";
    vis.type = ShapeType::Box;
    vis.half_extents = {0.4f, 0.4f, 0.4f};
    vis.material_id = red_id;
    vis.contype = 0;
    vis.conaffinity = 0;
    s.AddCollisionShape(vis);

    // A body whose collision is a real MESH (a tetra) -- the no-data-loss subject.
    RigidBodyRecord mp;
    mp.name = "meshprop";
    mp.mass = 1.0f;
    mp.local_transform.position = {1.2f, 0.0f, 0.5f};
    const BodyId mpid = s.AddRigidBody(mp);
    CollisionShapeRecord ms;
    ms.body_id = mpid;
    ms.name = "hull";
    ms.type = ShapeType::ConvexHull;
    ms.decompose_mode = DecomposeMode::Skip;
    ms.mesh_vertices = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    ms.mesh_indices = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
    s.AddCollisionShape(ms);

    return s;
}

}  // namespace

// The critical gate: edits persist + collision geometry survives a Save+reload.
TEST(EditorEditRoundtrip, EditsPersistAndCollisionSurvives) {
    SceneIR scene = BuildEditableDemo();

    // Emit the pre-edit scene for the headless inspector capture (opt-in).
    if (const char* dir = std::getenv("NUKA_EDIT_DEMO_DIR")) {
        nks::Save(scene, (fs::path(dir) / "edit_demo.nks").string());
    }

    // Pin the collision-mesh geometry BEFORE the edit (the no-data-loss subject).
    ShapeId hull = scene.ShapeCount();
    for (ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        if (!scene.GetShape(i).mesh_vertices.empty()) { hull = i; break; }
    }
    ASSERT_LT(hull, scene.ShapeCount()) << "no collision-mesh shape in the demo";
    const std::vector<float> mesh_before = scene.GetShape(hull).mesh_vertices;
    ASSERT_FALSE(mesh_before.empty());

    // Apply the SAME record mutations the editor's commit path writes.
    MaterialId red = scene.MaterialCount();
    for (MaterialId i = 0; i < scene.MaterialCount(); ++i) {
        if (scene.GetMaterial(i).name == "demo_red") { red = i; break; }
    }
    ASSERT_LT(red, scene.MaterialCount());
    {
        MaterialRecord& m = scene.GetMaterialMut(red);
        m.base_color = {0.10f, 0.80f, 0.30f};  // recolor to green
        m.roughness = 0.20f;
        m.metallic = 0.90f;
        m.emissive = {0.40f, 0.05f, 0.00f};
        m.transmission = 0.50f;
        m.ior = 1.45f;
    }
    ShapeId skin = scene.ShapeCount();
    for (ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        if (scene.GetShape(i).name == "skin") { skin = i; break; }
    }
    ASSERT_LT(skin, scene.ShapeCount());
    scene.GetShapeMut(skin).local_transform.position = {0.55f, -0.30f, 1.10f};

    const uint64_t mat_fnv = Fnv(MaterialFields(scene.GetMaterial(red)));

    // Save (full fidelity) -> reload.
    TempDir tmp;
    const std::string path = tmp.File("edited.nks").string();
    nks::Save(scene, path);
    const SceneIR loaded = nks::Load(path);

    // Material edits persisted (FNV field-equality).
    MaterialId lred = loaded.MaterialCount();
    for (MaterialId i = 0; i < loaded.MaterialCount(); ++i) {
        if (loaded.GetMaterial(i).name == "demo_red") { lred = i; break; }
    }
    ASSERT_LT(lred, loaded.MaterialCount());
    EXPECT_EQ(Fnv(MaterialFields(loaded.GetMaterial(lred))), mat_fnv)
        << "material edits (color/rough/metal/emissive/transmission/ior) did not persist";
    EXPECT_FLOAT_EQ(loaded.GetMaterial(lred).transmission, 0.50f);
    EXPECT_FLOAT_EQ(loaded.GetMaterial(lred).emissive.x, 0.40f);

    // Transform edit persisted.
    ShapeId lskin = loaded.ShapeCount();
    for (ShapeId i = 0; i < loaded.ShapeCount(); ++i) {
        if (loaded.GetShape(i).name == "skin") { lskin = i; break; }
    }
    ASSERT_LT(lskin, loaded.ShapeCount());
    EXPECT_FLOAT_EQ(loaded.GetShape(lskin).local_transform.position.x, 0.55f);
    EXPECT_FLOAT_EQ(loaded.GetShape(lskin).local_transform.position.z, 1.10f);

    // THE no-data-loss gate: collision mesh geometry survived the save+reload.
    ShapeId lhull = loaded.ShapeCount();
    for (ShapeId i = 0; i < loaded.ShapeCount(); ++i) {
        if (loaded.GetShape(i).name == "hull") { lhull = i; break; }
    }
    ASSERT_LT(lhull, loaded.ShapeCount());
    EXPECT_EQ(loaded.GetShape(lhull).mesh_vertices, mesh_before)
        << "collision mesh geometry was LOST across Save+reload (data-loss trap)";
    EXPECT_FALSE(loaded.GetShape(lhull).mesh_vertices.empty());
}

// Tree-edit record shapes survive Save+reload: a renamed shape, plus an added box
// body + shape + material (the records AddBoxChild emits). Mirrors the editor's
// record-level mutations so the serialization the tree ops depend on stays covered.
TEST(EditorEditRoundtrip, TreeEditsPersist) {
    SceneIR scene = BuildEditableDemo();

    // Rename the "skin" shape -> "crate" (the editor's record-name edit).
    ShapeId skin = scene.ShapeCount();
    for (ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        if (scene.GetShape(i).name == "skin") { skin = i; break; }
    }
    ASSERT_LT(skin, scene.ShapeCount());
    scene.GetShapeMut(skin).name = "crate";

    // Add a movable box body + box shape + material (the AddBoxChild records).
    RigidBodyRecord nb;
    nb.name = "addbox";
    nb.is_static = false;
    nb.local_transform.position = {0.0f, 0.0f, 0.5f};
    const BodyId nbid = scene.AddRigidBody(nb);
    MaterialRecord nm;
    nm.name = "addbox_mat";
    nm.base_color = {0.72f, 0.72f, 0.75f};
    const MaterialId nmid = scene.AddMaterial(nm);
    CollisionShapeRecord ns;
    ns.body_id = nbid;
    ns.type = ShapeType::Box;
    ns.half_extents = {0.25f, 0.25f, 0.25f};
    ns.material_id = nmid;
    ns.name = "addbox_geom";
    scene.AddCollisionShape(ns);

    TempDir tmp;
    const std::string path = tmp.File("treeedit.nks").string();
    nks::Save(scene, path);
    const SceneIR loaded = nks::Load(path);

    // Rename persisted: "crate" present, "skin" gone.
    bool has_crate = false, has_skin = false;
    for (ShapeId i = 0; i < loaded.ShapeCount(); ++i) {
        const std::string& n = loaded.GetShape(i).name;
        if (n == "crate") has_crate = true;
        if (n == "skin")  has_skin = true;
    }
    EXPECT_TRUE(has_crate) << "shape rename did not persist";
    EXPECT_FALSE(has_skin) << "old shape name survived the rename";

    // Add persisted: the new body + its shape + material all survive the round-trip.
    bool has_body = false, has_geom = false, has_mat = false;
    for (BodyId i = 0; i < loaded.RigidBodyCount(); ++i)
        if (loaded.GetBody(i).name == "addbox") has_body = true;
    for (ShapeId i = 0; i < loaded.ShapeCount(); ++i)
        if (loaded.GetShape(i).name == "addbox_geom") has_geom = true;
    for (MaterialId i = 0; i < loaded.MaterialCount(); ++i)
        if (loaded.GetMaterial(i).name == "addbox_mat") has_mat = true;
    EXPECT_TRUE(has_body) << "added body did not persist";
    EXPECT_TRUE(has_geom) << "added shape did not persist";
    EXPECT_TRUE(has_mat) << "added material did not persist";
}

// Negative control: the light-strip path (what the editor must NOT save from)
// loses collision geometry -- the trap the full-fidelity load avoids.
TEST(EditorEditRoundtrip, LightStripWouldLoseCollision) {
    SceneIR scene = BuildEditableDemo();
    for (ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(i);
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    TempDir tmp;
    const std::string path = tmp.File("stripped.nks").string();
    nks::Save(scene, path);
    const SceneIR loaded = nks::Load(path);
    for (ShapeId i = 0; i < loaded.ShapeCount(); ++i) {
        EXPECT_TRUE(loaded.GetShape(i).mesh_vertices.empty())
            << "stripped scene unexpectedly retained geometry";
    }
}
