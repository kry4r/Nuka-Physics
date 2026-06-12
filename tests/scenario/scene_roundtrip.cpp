// ---------------------------------------------------------------------------
// M2c scenario gate: .nks / .nka scene roundtrip fidelity.
// ---------------------------------------------------------------------------
// Import (MJCF/USD) -> Compose -> Save(.nks + .nka) -> Load -> assert the loaded
// SceneIR is structurally + cook-identical to the original, and a second Save is
// byte-identical to the first. This is the strongest honest fidelity check for
// the own scene format: it goes all the way down to byte-equal cooked shape /
// contact-param rows.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "scene/asset/nka.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"
#include "scene/format/json.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace nuka::scene;
namespace fs = std::filesystem;

namespace {

const std::string kMinimalArm = "tests/data/minimal_arm.xml";
const std::string kMinimalScene = "tests/data/minimal_scene.usda";

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupLargeUsd =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model_large.usda";

bool MinimalAssetsAvailable() {
    return fs::exists(kMinimalArm) && fs::exists(kMinimalScene);
}
bool HeavyAssetsAvailable() {
    return fs::exists(kH1Mjcf) && fs::exists(kCupLargeUsd);
}

// A fresh temp dir per run so reruns are hermetic.
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/nks_roundtrip_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path File(const std::string& name) const { return path_ / name; }
    // A fresh sub-directory (created) for saving the same-named scene twice so
    // the embedded sibling .nka BASENAME is identical across the two saves
    // (byte-identity is about determinism, not the on-disk path).
    fs::path SubFile(const std::string& subdir, const std::string& name) const {
        const fs::path d = path_ / subdir;
        std::error_code ec;
        fs::create_directories(d, ec);
        return d / name;
    }
    const fs::path& Path() const { return path_; }

private:
    fs::path path_;
};

std::vector<uint8_t> ReadBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

// -- a small always-on scene: MJCF arm + USD scene, composed --------------
SceneIR BuildMinimalComposed() {
    SceneIR arm = nuka::import::LoadMjcf(kMinimalArm);
    SceneIR usd = nuka::import::LoadUsd(kMinimalScene);
    return Compose(arm, usd, nuka::math::Transform::Identity(), "addon/");
}

// -- tree equality: pre-order names + derived paths -----------------------
// EXACT equality, including autonamed nodes: an unnamed shape/joint record
// projects to a RECORD-identity-derived name ("geom_<id>" / "joint_<id>",
// see SceneIR's StableAutoName), which is a stable scene property that
// round-trips through .nks. No wildcard normalization.
void CollectTree(const SceneGraph& tree, std::vector<std::string>& names,
                 std::vector<std::string>& paths) {
    tree.Traverse(tree.Root(), [&](const std::shared_ptr<SceneNode>& n) {
        names.push_back(n->name);
        paths.push_back(tree.PathOf(n));
    });
}

void ExpectTreeEqual(const SceneIR& a, const SceneIR& b) {
    std::vector<std::string> na, pa, nb, pb;
    CollectTree(a.Tree(), na, pa);
    CollectTree(b.Tree(), nb, pb);
    ASSERT_EQ(na.size(), nb.size()) << "tree node count differs";
    EXPECT_EQ(na, nb) << "pre-order node names differ";
    EXPECT_EQ(pa, pb) << "derived node paths differ";
}

// -- record equality ------------------------------------------------------
void ExpectBodyRecordsEqual(const SceneIR& a, const SceneIR& b) {
    ASSERT_EQ(a.RigidBodyCount(), b.RigidBodyCount());
    for (BodyId i = 0; i < a.RigidBodyCount(); ++i) {
        const auto& ra = a.GetBody(i);
        const auto& rb = b.GetBody(i);
        EXPECT_EQ(ra.name, rb.name);
        EXPECT_EQ(ra.parent_id, rb.parent_id);
        EXPECT_EQ(ra.is_static, rb.is_static);
        EXPECT_FLOAT_EQ(ra.mass, rb.mass);
        EXPECT_FLOAT_EQ(ra.local_transform.position.x, rb.local_transform.position.x);
        EXPECT_FLOAT_EQ(ra.local_transform.position.y, rb.local_transform.position.y);
        EXPECT_FLOAT_EQ(ra.local_transform.position.z, rb.local_transform.position.z);
        EXPECT_FLOAT_EQ(ra.local_transform.rotation.w, rb.local_transform.rotation.w);
    }
}

void ExpectShapeRecordsEqual(const SceneIR& a, const SceneIR& b) {
    ASSERT_EQ(a.ShapeCount(), b.ShapeCount());
    for (ShapeId i = 0; i < a.ShapeCount(); ++i) {
        const auto& sa = a.GetShape(i);
        const auto& sb = b.GetShape(i);
        EXPECT_EQ(sa.name, sb.name);
        EXPECT_EQ(sa.body_id, sb.body_id);
        EXPECT_EQ(sa.material_id, sb.material_id);
        EXPECT_EQ(static_cast<int>(sa.type), static_cast<int>(sb.type));
        // cook-critical contact metadata (spot-assert)
        EXPECT_EQ(sa.contype, sb.contype);
        EXPECT_EQ(sa.conaffinity, sb.conaffinity);
        EXPECT_FLOAT_EQ(sa.solref[0], sb.solref[0]);
        EXPECT_FLOAT_EQ(sa.solref[1], sb.solref[1]);
        EXPECT_FLOAT_EQ(sa.friction_mu, sb.friction_mu);
        EXPECT_EQ(sa.condim, sb.condim);
        EXPECT_EQ(static_cast<int>(sa.decompose_mode), static_cast<int>(sb.decompose_mode));
        EXPECT_EQ(sa.mesh_vertices.size(), sb.mesh_vertices.size());
        EXPECT_EQ(sa.mesh_indices, sb.mesh_indices);
        EXPECT_EQ(sa.mesh_vertices, sb.mesh_vertices);
    }
}

void ExpectJointRecordsEqual(const SceneIR& a, const SceneIR& b) {
    ASSERT_EQ(a.JointCount(), b.JointCount());
    for (JointId i = 0; i < a.JointCount(); ++i) {
        const auto& ja = a.GetJoint(i);
        const auto& jb = b.GetJoint(i);
        EXPECT_EQ(ja.name, jb.name);
        EXPECT_EQ(static_cast<int>(ja.type), static_cast<int>(jb.type));
        EXPECT_EQ(ja.parent_body, jb.parent_body);
        EXPECT_EQ(ja.child_body, jb.child_body);
        EXPECT_FLOAT_EQ(ja.lower_limit, jb.lower_limit);
        EXPECT_FLOAT_EQ(ja.upper_limit, jb.upper_limit);
    }
}

void ExpectMaterialNamesEqual(const SceneIR& a, const SceneIR& b) {
    ASSERT_EQ(a.MaterialCount(), b.MaterialCount());
    for (MaterialId i = 0; i < a.MaterialCount(); ++i) {
        EXPECT_EQ(a.GetMaterial(i).name, b.GetMaterial(i).name);
        EXPECT_FLOAT_EQ(a.GetMaterial(i).friction_mu, b.GetMaterial(i).friction_mu);
    }
}

// -- byte-equal cooked rows (strongest fidelity check) --------------------
void ExpectCookEqual(const SceneIR& a, const SceneIR& b) {
    const CookedBlob ca = CookScene(a);
    const CookedBlob cb = CookScene(b);
    EXPECT_EQ(ca.body_count, cb.body_count);
    EXPECT_EQ(ca.joint_count, cb.joint_count);
    EXPECT_EQ(ca.shape_count, cb.shape_count);
    EXPECT_EQ(ca.material_count, cb.material_count);
    EXPECT_EQ(ca.convex_geometry.Count(), cb.convex_geometry.Count());

    // byte-equal shape rows
    ASSERT_EQ(ca.shapes.types.size(), cb.shapes.types.size());
    for (size_t i = 0; i < ca.shapes.types.size(); ++i) {
        EXPECT_EQ(static_cast<int>(ca.shapes.types[i]), static_cast<int>(cb.shapes.types[i]));
        EXPECT_EQ(ca.shapes.body_ids[i], cb.shapes.body_ids[i]);
        EXPECT_FLOAT_EQ(ca.shapes.radii[i], cb.shapes.radii[i]);
        EXPECT_FLOAT_EQ(ca.shapes.half_heights[i], cb.shapes.half_heights[i]);
        EXPECT_FLOAT_EQ(ca.shapes.half_extents[i].x, cb.shapes.half_extents[i].x);
    }
    // byte-equal contact-param rows
    EXPECT_EQ(ca.contact_params.contypes, cb.contact_params.contypes);
    EXPECT_EQ(ca.contact_params.conaffinities, cb.contact_params.conaffinities);
    EXPECT_EQ(ca.contact_params.frictions, cb.contact_params.frictions);
    EXPECT_EQ(ca.contact_params.solref0, cb.contact_params.solref0);
    EXPECT_EQ(ca.contact_params.solref1, cb.contact_params.solref1);
    EXPECT_EQ(ca.contact_params.solimp, cb.contact_params.solimp);
    EXPECT_EQ(ca.contact_params.condims, cb.contact_params.condims);
}

}  // namespace

// 1. Minimal always-on: full fidelity (tree/components/records/cook).
TEST(SceneRoundtrip, MinimalAlwaysOn) {
    if (!MinimalAssetsAvailable()) GTEST_SKIP() << "minimal tests/data assets missing";
    TempDir tmp;

    const SceneIR orig = BuildMinimalComposed();
    const std::string nks = tmp.File("minimal.nks").string();
    nks::Save(orig, nks);

    // Format-shape pin (§3.7): the saved JSON root must carry a nested "tree"
    // section and the split material sections, and must NOT carry the old flat
    // "bodies" record array (the deviation this change removed).
    {
        std::ifstream in(nks, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        const json::Value root = json::Value::Parse(text);
        EXPECT_TRUE(root.Has("tree")) << "saved .nks lacks the §3.7 \"tree\" section";
        EXPECT_TRUE(root.Find("tree")->IsArray()) << "\"tree\" must be a node array";
        EXPECT_TRUE(root.Has("physics_materials"));
        EXPECT_TRUE(root.Has("render_materials"));
        EXPECT_FALSE(root.Has("bodies")) << "flat \"bodies\" array must be gone";
        EXPECT_FALSE(root.Has("shapes")) << "flat \"shapes\" array must be gone";
        EXPECT_FALSE(root.Has("joints")) << "flat \"joints\" array must be gone";
    }

    const SceneIR loaded = nks::Load(nks);

    ExpectTreeEqual(orig, loaded);
    ExpectBodyRecordsEqual(orig, loaded);
    ExpectShapeRecordsEqual(orig, loaded);
    ExpectJointRecordsEqual(orig, loaded);
    ExpectMaterialNamesEqual(orig, loaded);
    EXPECT_EQ(orig.SensorCount(), loaded.SensorCount());
    EXPECT_EQ(orig.CameraCount(), loaded.CameraCount());
    EXPECT_EQ(orig.LightCount(), loaded.LightCount());
    EXPECT_EQ(orig.ExcludePairs().size(), loaded.ExcludePairs().size());
    EXPECT_EQ(orig.ContactPairs().size(), loaded.ContactPairs().size());
    ExpectCookEqual(orig, loaded);
}

// 2. Second save is byte-identical (.nks + .nka).
TEST(SceneRoundtrip, SecondSaveByteIdentical) {
    if (!MinimalAssetsAvailable()) GTEST_SKIP() << "minimal tests/data assets missing";
    TempDir tmp;

    const SceneIR orig = BuildMinimalComposed();
    // Same filename stem ("scene") in two sub-dirs so the embedded .nka basename
    // matches; only then is byte-identity a pure determinism check.
    const fs::path nks1 = tmp.SubFile("run1", "scene.nks");
    nks::Save(orig, nks1.string());

    const SceneIR loaded = nks::Load(nks1.string());
    const fs::path nks2 = tmp.SubFile("run2", "scene.nks");
    nks::Save(loaded, nks2.string());

    EXPECT_EQ(ReadBytes(nks1), ReadBytes(nks2)) << ".nks not byte-identical on re-save";
    EXPECT_EQ(ReadBytes(tmp.SubFile("run1", "scene.nka")),
              ReadBytes(tmp.SubFile("run2", "scene.nka")))
        << ".nka not byte-identical on re-save";
}

// 3. H1 + cup (mesh-heavy: exercises MESH/CMSH path). Skips without assets.
// Mesh shapes are set to DecomposeMode::Skip so cooking treats each source mesh
// as a single convex hull instead of running the (minutes-long, many-geom)
// V-HACD decomposition. The mesh BYTES still round-trip through the .nka CMSH
// chunks (the point of this case); the cook just stays cheap + deterministic.
TEST(SceneRoundtrip, H1CupRoundtrip) {
    if (!HeavyAssetsAvailable()) GTEST_SKIP() << ".nuka-assets H1/cup missing";
    TempDir tmp;

    SceneIR h1 = nuka::import::LoadMjcf(kH1Mjcf);
    SceneIR cup = nuka::import::LoadUsd(kCupLargeUsd);
    SceneIR orig =
        Compose(h1, cup, nuka::math::Transform::Identity(), "cup/");
    for (ShapeId i = 0; i < orig.ShapeCount(); ++i) {
        auto& s = orig.GetShapeMut(i);
        if (!s.mesh_vertices.empty()) s.decompose_mode = DecomposeMode::Skip;
    }
    // No manual facade rebuild needed: GetShapeMut marks the facade dirty and
    // the next facade read (ExpectTreeEqual below) lazily re-projects.

    // Same stem ("h1cup") in two sub-dirs so the embedded .nka basename matches.
    const fs::path nks1 = tmp.SubFile("run1", "h1cup.nks");
    nks::Save(orig, nks1.string());
    const SceneIR loaded = nks::Load(nks1.string());

    ExpectTreeEqual(orig, loaded);
    ExpectBodyRecordsEqual(orig, loaded);
    ExpectShapeRecordsEqual(orig, loaded);

    const CookedBlob co = CookScene(orig);
    const CookedBlob cl = CookScene(loaded);
    EXPECT_EQ(co.body_count, cl.body_count);
    EXPECT_EQ(co.shape_count, cl.shape_count);
    EXPECT_EQ(co.convex_geometry.Count(), cl.convex_geometry.Count());

    // second-save byte identity (same stem in a sibling dir)
    const fs::path nks2 = tmp.SubFile("run2", "h1cup.nks");
    nks::Save(loaded, nks2.string());
    EXPECT_EQ(ReadBytes(nks1), ReadBytes(nks2));
    EXPECT_EQ(ReadBytes(tmp.SubFile("run1", "h1cup.nka")),
              ReadBytes(tmp.SubFile("run2", "h1cup.nka")));
}

// 4. Override layer: change one transform + one material field; rest untouched.
TEST(SceneRoundtrip, OverrideLayer) {
    if (!MinimalAssetsAvailable()) GTEST_SKIP() << "minimal tests/data assets missing";
    TempDir tmp;

    const SceneIR base = BuildMinimalComposed();
    ASSERT_GT(base.RigidBodyCount(), 0u);
    ASSERT_GT(base.MaterialCount(), 0u);

    const std::string base_nks = tmp.File("base.nks").string();
    nks::Save(base, base_nks);

    // Build an overlay targeting the first body's path and first material name.
    const std::string body_path = base.GetBody(0).name;     // record name == path
    const std::string mat_name = base.GetMaterial(0).name;
    const float orig_x = base.GetBody(0).local_transform.position.x;

    const std::string overlay_text =
        std::string("{\n  \"overrides\": {\n") +
        "    \"" + body_path + "\": { \"transform\": { \"pos\": [42.5, 1.0, 2.0] },"
        " \"rigid_body\": { \"mass\": 7.25 } },\n" +
        "    \"" + mat_name + "\": { \"material\": { \"friction_mu\": 0.123 } }\n" +
        "  }\n}\n";
    const std::string overlay_path = tmp.File("overlay.nks").string();
    {
        std::ofstream out(overlay_path, std::ios::binary | std::ios::trunc);
        out << overlay_text;
    }

    const SceneIR overridden = nks::Load(base_nks, overlay_path);

    EXPECT_FLOAT_EQ(overridden.GetBody(0).local_transform.position.x, 42.5f);
    EXPECT_FLOAT_EQ(overridden.GetBody(0).mass, 7.25f);
    EXPECT_FLOAT_EQ(overridden.GetMaterial(0).friction_mu, 0.123f);
    EXPECT_NE(orig_x, 42.5f);  // sanity: the override actually changed something

    // A non-targeted body keeps its transform.
    if (base.RigidBodyCount() > 1) {
        EXPECT_FLOAT_EQ(overridden.GetBody(1).local_transform.position.x,
                        base.GetBody(1).local_transform.position.x);
    }
}

// 4b. Facade write-through after DIRECT record mutation (pins the M2b
//     divergence fix): mutating a record via Get*Mut must be reflected by the
//     facade on its next read, with no manual rebuild. A collision shape whose
//     contype/conaffinity is flipped to 0/0 must re-project from
//     CollisionShapeComponent to VisualMeshComponent.
TEST(SceneRoundtrip, FacadeResyncsAfterRecordMutation) {
    if (!MinimalAssetsAvailable()) GTEST_SKIP() << "minimal tests/data assets missing";

    SceneIR scene = BuildMinimalComposed();

    // Find a colliding shape (contype/conaffinity not both 0).
    ShapeId target = scene.ShapeCount();
    for (ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        const auto& s = scene.GetShape(i);
        if (s.contype != 0 || s.conaffinity != 0) { target = i; break; }
    }
    ASSERT_LT(target, scene.ShapeCount()) << "no colliding shape in the scene";

    // Facade BEFORE the mutation: a CollisionShapeComponent.
    {
        const EntityId e = scene.EntityOfShape(target);
        ASSERT_NE(e, kInvalidEntity);
        EXPECT_TRUE(scene.Ecs().Has<CollisionShapeComponent>(e));
        EXPECT_FALSE(scene.Ecs().Has<VisualMeshComponent>(e));
    }

    // Mutate the record DIRECTLY (bypasses the Add* write-through).
    auto& rec = scene.GetShapeMut(target);
    rec.contype = 0;
    rec.conaffinity = 0;

    // Facade AFTER: the next facade read must lazily re-project. NOTE the
    // entity must be re-fetched -- re-projection creates fresh entities.
    {
        const EntityId e = scene.EntityOfShape(target);
        ASSERT_NE(e, kInvalidEntity);
        EXPECT_TRUE(scene.Ecs().Has<VisualMeshComponent>(e))
            << "facade did not re-project after a direct record mutation";
        EXPECT_FALSE(scene.Ecs().Has<CollisionShapeComponent>(e));
    }
}

// 5. SceneMap roundtrip: bind entities->rows in record order; close the loop
//    name -> entity -> row -> entity -> PathOf for every body.
TEST(SceneRoundtrip, SceneMapRoundtrip) {
    if (!MinimalAssetsAvailable()) GTEST_SKIP() << "minimal tests/data assets missing";
    TempDir tmp;

    const SceneIR orig = BuildMinimalComposed();
    const std::string nks = tmp.File("map.nks").string();
    nks::Save(orig, nks);
    const SceneIR loaded = nks::Load(nks);

    const CookedBlob blob = CookScene(loaded);
    ASSERT_EQ(blob.body_count, loaded.RigidBodyCount());

    // Bind body rows: cooked body row i corresponds to BodyId i (record order).
    SceneMap map;
    for (BodyId i = 0; i < loaded.RigidBodyCount(); ++i) {
        const EntityId e = loaded.EntityOfBody(i);
        ASSERT_NE(e, kInvalidEntity);
        CookedRef ref;
        ref.body_row = i;
        map.Bind(e, ref);
    }

    // Close the loop for every body.
    for (BodyId i = 0; i < loaded.RigidBodyCount(); ++i) {
        const EntityId e = loaded.EntityOfBody(i);
        const CookedRef* ref = map.RefOf(e);
        ASSERT_NE(ref, nullptr);
        EXPECT_EQ(ref->body_row, i);
        const EntityId back = map.EntityOfBody(i);
        EXPECT_EQ(back, e);
        // entity -> node -> path is well-defined.
        const auto node = loaded.Ecs().NodeOf(e);
        ASSERT_NE(node, nullptr);
        const std::string path = loaded.Tree().PathOf(node);
        EXPECT_FALSE(path.empty()) << "body " << i << " has an empty path";
    }
}
