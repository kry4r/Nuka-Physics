// ---------------------------------------------------------------------------
// M7 T3 gate: examples/scenes/h1_cup_table.nks roundtrip + content exposure.
// ---------------------------------------------------------------------------
// Load the committed union grasp scene (H1 via imports + cup body + table body +
// the grasp config + initial_state + settle), then:
//   (1) Save the LOADED scene twice and assert the two .nks (and the two .nka)
//       are BYTE-IDENTICAL -- the existing roundtrip determinism invariant,
//       extended to the new initial_state/settle/grasp sections (deterministic
//       key order, lossless float round-trip).
//   (2) Assert the loaded SceneIR EXPOSES the cup body / table body / expanded
//       H1 import / grasp config / initial_state / settle.
//
// This does NOT weaken scene_roundtrip's own equality gates; it is an additive,
// scene-specific coverage of the M7 sections on a real authored file.
// ---------------------------------------------------------------------------

#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace nuka::scene;
namespace fs = std::filesystem;

namespace {

const std::string kScene = "examples/scenes/h1_cup_table.nks";
const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";

bool SceneAvailable() {
    return fs::exists(kScene) && fs::exists(kH1Mjcf);
}

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/nks_grasp_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    // A fresh sub-dir so the embedded sibling .nka basename matches across the
    // two saves (byte-identity is determinism, not the on-disk path).
    fs::path SubFile(const std::string& subdir, const std::string& name) const {
        const fs::path d = path_ / subdir;
        std::error_code ec;
        fs::create_directories(d, ec);
        return d / name;
    }

private:
    fs::path path_;
};

std::vector<uint8_t> ReadBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

}  // namespace

// 1. The committed union grasp scene round-trips byte-identically (incl. the
//    initial_state / settle / grasp sections + the cup-hull SAMP in the .nka).
TEST(NksGraspSceneRoundtrip, SecondSaveByteIdentical) {
    if (!SceneAvailable()) GTEST_SKIP() << "h1_cup_table.nks / H1 asset missing";
    TempDir tmp;

    const SceneIR loaded = nks::Load(kScene);
    const fs::path nks1 = tmp.SubFile("run1", "h1_cup_table.nks");
    nks::Save(loaded, nks1.string());
    const fs::path nks2 = tmp.SubFile("run2", "h1_cup_table.nks");
    nks::Save(loaded, nks2.string());

    EXPECT_EQ(ReadBytes(nks1), ReadBytes(nks2)) << ".nks not byte-identical on re-save";
    EXPECT_EQ(ReadBytes(tmp.SubFile("run1", "h1_cup_table.nka")),
              ReadBytes(tmp.SubFile("run2", "h1_cup_table.nka")))
        << ".nka not byte-identical on re-save";
}

// 2. Load -> Save -> Load yields an IDENTICAL grasp/initial_state/settle (the
//    sections survive a full round-trip, not just a deterministic re-emit).
TEST(NksGraspSceneRoundtrip, MetadataSurvivesReload) {
    if (!SceneAvailable()) GTEST_SKIP() << "h1_cup_table.nks / H1 asset missing";
    TempDir tmp;

    const SceneIR a = nks::Load(kScene);
    const fs::path nks1 = tmp.SubFile("run1", "h1_cup_table.nks");
    nks::Save(a, nks1.string());
    const SceneIR b = nks::Load(nks1.string());

    // grasp config
    ASSERT_TRUE(a.Grasp().present);
    ASSERT_TRUE(b.Grasp().present);
    EXPECT_FLOAT_EQ(a.Grasp().finger_mu, b.Grasp().finger_mu);
    EXPECT_FLOAT_EQ(a.Grasp().table_mu, b.Grasp().table_mu);
    EXPECT_FLOAT_EQ(a.Grasp().foot_mu, b.Grasp().foot_mu);
    EXPECT_DOUBLE_EQ(a.Grasp().total_mass, b.Grasp().total_mass);
    EXPECT_EQ(a.Grasp().dof_stride, b.Grasp().dof_stride);
    EXPECT_EQ(a.Grasp().cup_hull_verts.size(), b.Grasp().cup_hull_verts.size());
    EXPECT_EQ(a.Grasp().cup_hull_verts, b.Grasp().cup_hull_verts);
    EXPECT_EQ(a.Grasp().wrap_spheres.size(), b.Grasp().wrap_spheres.size());
    EXPECT_EQ(a.Grasp().foot_spheres.size(), b.Grasp().foot_spheres.size());
    EXPECT_EQ(a.Grasp().grip_links, b.Grasp().grip_links);
    EXPECT_EQ(a.Grasp().drive_close.size(), b.Grasp().drive_close.size());
    EXPECT_EQ(a.Grasp().dof_torque_limit, b.Grasp().dof_torque_limit);

    // initial_state
    ASSERT_EQ(a.InitialState().count("h1"), 1u);
    ASSERT_EQ(b.InitialState().count("h1"), 1u);
    EXPECT_EQ(a.InitialState().at("h1").qpos, b.InitialState().at("h1").qpos);

    // settle
    EXPECT_EQ(a.Settle().steps, b.Settle().steps);
    EXPECT_FLOAT_EQ(a.Settle().dt, b.Settle().dt);
    ASSERT_EQ(a.Settle().holds.size(), b.Settle().holds.size());
    ASSERT_FALSE(a.Settle().holds.empty());
    EXPECT_EQ(a.Settle().holds[0].dof_pattern, b.Settle().holds[0].dof_pattern);
}

// 3. The loaded SceneIR exposes the first-class objects: cup body + table body +
//    the expanded H1 import + the grasp geometry (30 wrap + 4 foot spheres).
TEST(NksGraspSceneRoundtrip, ExposesCupTableH1Grasp) {
    if (!SceneAvailable()) GTEST_SKIP() << "h1_cup_table.nks / H1 asset missing";

    const SceneIR s = nks::Load(kScene);

    // The H1 import expanded (its bodies carry the "h1/" attach prefix).
    bool saw_h1 = false, saw_cup = false, saw_table = false;
    for (BodyId i = 0; i < s.RigidBodyCount(); ++i) {
        const std::string& n = s.GetBody(i).name;
        if (n.rfind("h1/", 0) == 0) saw_h1 = true;
        if (n == "cup/body") saw_cup = true;
        if (n == "table/body") saw_table = true;
    }
    EXPECT_TRUE(saw_h1) << "H1 import did not expand";
    EXPECT_TRUE(saw_cup) << "cup body missing";
    EXPECT_TRUE(saw_table) << "table body missing";

    // The cup + table collision shapes exist.
    bool cup_shape = false, table_shape = false;
    for (ShapeId i = 0; i < s.ShapeCount(); ++i) {
        const std::string& n = s.GetShape(i).name;
        if (n == "cup/collision") cup_shape = true;
        if (n == "table/collision") table_shape = true;
    }
    EXPECT_TRUE(cup_shape);
    EXPECT_TRUE(table_shape);

    // The grasp config carries the full factory geometry.
    ASSERT_TRUE(s.Grasp().present);
    EXPECT_EQ(s.Grasp().wrap_spheres.size(), 30u);
    EXPECT_EQ(s.Grasp().foot_spheres.size(), 4u);
    EXPECT_EQ(s.Grasp().grip_links.size(), 12u);
    EXPECT_EQ(s.Grasp().dof_stride, 51u);
    EXPECT_EQ(s.Grasp().base_dof, 6u);
    EXPECT_GT(s.Grasp().cup_hull_verts.size(), 0u);
    EXPECT_FLOAT_EQ(s.Grasp().finger_mu, 0.8f);
    EXPECT_FLOAT_EQ(s.Grasp().table_mu, 0.6f);
    EXPECT_FLOAT_EQ(s.Grasp().foot_mu, 0.8f);
    EXPECT_FLOAT_EQ(s.Grasp().cup_scale_xy, 1.8f);

    // initial_state + settle present.
    EXPECT_EQ(s.InitialState().count("h1"), 1u);
    EXPECT_GT(s.InitialState().at("h1").qpos.size(), 0u);
    EXPECT_EQ(s.Settle().steps, 200);
    EXPECT_FALSE(s.Settle().holds.empty());
}
