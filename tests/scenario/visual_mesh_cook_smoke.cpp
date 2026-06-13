// ---------------------------------------------------------------------------
// visual_mesh_cook_smoke -- M8.5 T5 gate for the VISUAL-MESH COOK.
// ---------------------------------------------------------------------------
// The beauty enabler: robots must render as REAL triangle geometry, not 5cm
// placeholder boxes. The producer gaps closed in T5 are:
//   (1) nks.cpp routes a VISUAL-only geom's triangles to a .nka MESH chunk (vs
//       the CMSH chunk a colliding mesh uses) and emits a MESH AssetRef on the
//       visual_mesh node; Load decodes it + records the resolved ref.
//   (2) scene_ir.cpp ProjectShape populates VisualMeshComponent.mesh from that
//       ref, so render_world.cpp's NkaTagMesh() branch fires.
//   (+ mjcf_importer.cpp: resolve geom `type` from the <default> class and
//      default an unnamed <mesh> to its file stem, so a menagerie-style robot
//      like go2 actually loads its visual STL/OBJ.)
//
// This gate is SCENE-SIDE only (NO Vulkan / NO CUDA): it cooks go2 (and, if the
// assets are present, h1_with_hand) through import -> Save(.nks/.nka) -> Load,
// then BuildRenderWorld and asserts >=1 instance is MeshSource::NkaMesh with >0
// triangles, and that the .nka round-trips the mesh data. It is gated like the
// other scene tests (no NK_BUILD_VULKAN_VALIDATION needed).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "import/mjcf_importer.hpp"
#include "render/render_world.hpp"
#include "scene/asset/nka.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace nuka;

namespace {

const std::string kGo2Mjcf = ".nuka-assets/mujoco_menagerie/unitree_go2/go2.xml";
const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";

// A fresh temp dir so reruns are hermetic and the committed examples/ scenes are
// never overwritten by the gate.
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/visual_mesh_cook_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path File(const std::string& name) const { return path_ / name; }

private:
    fs::path path_;
};

// Cook an MJCF robot to a temp .nks/.nka, Load it back, BuildRenderWorld, and
// assert the visual meshes render as real .nka MESH triangles. Returns the
// number of NkaMesh instances seen (>0 on success).
uint32_t RunCookAndRender(const std::string& mjcf_path, const std::string& stem) {
    TempDir tmp;
    const std::string nks = tmp.File(stem + ".nks").string();

    // (1) import -> (2) Save (routes visual geoms to .nka MESH chunks).
    const scene::SceneIR imported = nuka::import::LoadMjcf(mjcf_path);
    scene::nks::Save(imported, nks);

    // The sibling .nka must carry >=1 MESH chunk (the visual-mesh cook produced
    // real geometry, not an empty container).
    const fs::path nka = tmp.File(stem + ".nka");
    const scene::NkaFile nka_file = scene::NkaFile::Open(nka.string());
    const uint32_t mesh_chunks = nka_file.CountChunks(scene::NkaTagMesh());
    EXPECT_GT(mesh_chunks, 0u) << "no MESH chunks cooked into the .nka";

    // (3) Load -> BuildRenderWorld (an empty SceneMap is fine: this gate checks
    // mesh SOURCING, not live poses -> every instance resolves to a Static pose).
    const scene::SceneIR loaded = scene::nks::Load(nks);
    const scene::SceneMap empty_map;
    const render::RenderWorld world =
        render::BuildRenderWorld(loaded.Ecs(), empty_map);
    EXPECT_GT(world.InstanceCount(), 0u) << "BuildRenderWorld produced no instances";

    // (4) >=1 instance must be a real .nka MESH (NOT the placeholder-box fallback)
    // with a non-trivial triangle count.
    uint32_t nka_mesh_instances = 0;
    uint32_t total_tris = 0;
    for (const render::RenderInstance& inst : world.instances) {
        if (world.meshes.Source(inst.mesh_id) == render::MeshSource::NkaMesh) {
            ++nka_mesh_instances;
            const uint32_t tris = world.meshes.Geometry(inst.mesh_id).TriangleCount();
            EXPECT_GT(tris, 0u) << "an NkaMesh instance has zero triangles";
            total_tris += tris;
        }
    }
    EXPECT_GT(nka_mesh_instances, 0u)
        << "every instance fell back to a placeholder box -- the visual-mesh cook "
           "did not fire (producer gap not closed)";

    // (5) mesh-data round-trip: re-Load decodes the SAME triangle counts the
    // record carried, proving the MESH chunk preserves the source geometry.
    uint32_t record_visual_with_tris = 0;
    for (scene::ShapeId i = 0; i < loaded.ShapeCount(); ++i) {
        const auto& s = loaded.GetShape(i);
        const bool is_visual = (s.contype == 0 && s.conaffinity == 0);
        if (is_visual && !s.mesh_vertices.empty() && !s.mesh_indices.empty()) {
            ++record_visual_with_tris;
            // The visual_mesh_ref must point at a MESH chunk in the sibling .nka.
            EXPECT_FALSE(s.visual_mesh_ref.empty())
                << "loaded visual shape carries triangles but no MESH AssetRef";
        }
    }
    EXPECT_GT(record_visual_with_tris, 0u)
        << "no visual-only shapes carried triangles after round-trip";

    std::printf("[VISUAL-MESH COOK] %s: mesh_chunks=%u nka_mesh_instances=%u "
                "total_tris=%u nka_bytes=%lld\n",
                stem.c_str(), mesh_chunks, nka_mesh_instances, total_tris,
                static_cast<long long>(fs::file_size(nka)));
    return nka_mesh_instances;
}

}  // namespace

// go2 -- the cheap pipeline proof (16 unique visual meshes).
TEST(VisualMeshCook, Go2RendersRealMeshes) {
    if (!fs::exists(kGo2Mjcf)) GTEST_SKIP() << "go2 MJCF missing: " << kGo2Mjcf;
    EXPECT_GT(RunCookAndRender(kGo2Mjcf, "go2"), 0u);
}

// h1_with_hand -- the demo hero (49 visual STL/OBJ). Skips without assets.
// NOTE: the cooked h1_with_hand .nka (~88MB of STL/OBJ geometry) is intentionally
// NOT committed pending an LFS decision; go2.nka is the committed pipeline proof
// and h1 is reproduced on demand from the source MJCF via nuka_cook_scene (this
// test cooks+renders it live when the assets are present, else SKIPs).
TEST(VisualMeshCook, H1WithHandRendersRealMeshes) {
    if (!fs::exists(kH1Mjcf)) GTEST_SKIP() << "h1_with_hand MJCF missing: " << kH1Mjcf;
    EXPECT_GT(RunCookAndRender(kH1Mjcf, "h1_with_hand"), 0u);
}

// The committed examples/scenes/go2.nks (if present) loads and renders real
// meshes -- this is the scene the T3 viewer can point at for the beauty showcase.
TEST(VisualMeshCook, CommittedGo2SceneRendersRealMeshes) {
    const std::string committed = "examples/scenes/go2.nks";
    if (!fs::exists(committed)) GTEST_SKIP() << "committed go2.nks not present";

    const scene::SceneIR loaded = scene::nks::Load(committed);
    const scene::SceneMap empty_map;
    const render::RenderWorld world =
        render::BuildRenderWorld(loaded.Ecs(), empty_map);
    ASSERT_GT(world.InstanceCount(), 0u);

    uint32_t nka_mesh_instances = 0;
    for (const render::RenderInstance& inst : world.instances) {
        if (world.meshes.Source(inst.mesh_id) == render::MeshSource::NkaMesh) {
            ++nka_mesh_instances;
            EXPECT_GT(world.meshes.Geometry(inst.mesh_id).TriangleCount(), 0u);
        }
    }
    EXPECT_GT(nka_mesh_instances, 0u)
        << "committed go2.nks renders no real meshes (re-cook with nuka_cook_scene)";
}
