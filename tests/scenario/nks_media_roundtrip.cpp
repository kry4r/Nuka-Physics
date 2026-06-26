// ---------------------------------------------------------------------------
// .nks media + rigid-primitive roundtrip fidelity.
// ---------------------------------------------------------------------------
// A SceneIR carrying authored rigid primitives + a cloth / soft-tet MediaRecord
// is Saved to .nks and Loaded back; the loaded records must be field-equal and
// the cloth must cook (BuildClothXpbdInput) to the byte-identical lattice — the
// same FNV-1a the coupled / facade cloth cook produces. A separate case proves a
// media-free scene (examples/scenes/go2.nks) is byte-identical through Save: the
// new media section is additive (emitted only when media is present).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace nuka::scene;
namespace fs = std::filesystem;

namespace {

// FNV-1a over a float position stream (the cook's lattice), the SAME hash the
// coupled-world cook canary pins over the device PARTICLE_POSITION field.
uint64_t FnvPositions(const std::vector<nuka::math::Vec3>& p) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(p.data());
    const size_t n = p.size() * sizeof(nuka::math::Vec3);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ull;
    }
    return h;
}

std::vector<uint8_t> ReadBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/nks_media_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path SubFile(const std::string& subdir, const std::string& name) const {
        const fs::path d = path_ / subdir;
        std::error_code ec;
        fs::create_directories(d, ec);
        return d / name;
    }

private:
    fs::path path_;
};

// A scene with an authored static ground box + a cloth (canary params) + a
// soft-tet medium with a render skin. The cloth uses the go2_cloth_drape demo's
// frozen lattice params so its cook FNV is the pinned canary.
SceneIR BuildMediaScene() {
    SceneIR s;

    MaterialRecord mat;
    mat.name = "jelly";
    mat.base_color = nuka::math::Vec3{0.85f, 0.18f, 0.20f};
    mat.friction_mu = 0.7f;
    const MaterialId mid = s.AddMaterial(std::move(mat));

    RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    ground.local_transform.position = nuka::math::Vec3{0.0f, 0.0f, 0.0f};
    const BodyId gid = s.AddRigidBody(std::move(ground));
    CollisionShapeRecord gshape;
    gshape.name = "ground_box";
    gshape.body_id = gid;
    gshape.type = ShapeType::Box;
    gshape.half_extents = nuka::math::Vec3{2.0f, 2.0f, 0.05f};
    gshape.friction_mu = 0.9f;
    s.AddCollisionShape(std::move(gshape));

    MediaRecord cloth;
    cloth.name = "drape";
    cloth.kind = MediaRecord::Kind::Cloth;
    cloth.method = MediaRecord::Method::Xpbd;
    cloth.cloth_grid.nx = 55u;
    cloth.cloth_grid.ny = 51u;
    cloth.cloth_grid.spacing = 0.024f;
    cloth.cloth_grid.origin = nuka::math::Vec3{0.0f, 0.0f, 0.90f};
    cloth.cloth_grid.free = true;
    cloth.xpbd.particle_mass = 0.012f;
    cloth.xpbd.friction = 1.8f;
    cloth.xpbd.bend_alpha = 0.09f;
    cloth.xpbd.iters = 80u;
    cloth.xpbd.aero_drag_normal = 30.0f;
    cloth.xpbd.aero_drag_tangent = 0.12f;
    cloth.xpbd.aero_drag_max_dv = 0.16f;
    cloth.render_skin.normal_offset = 0.0015f;
    cloth.render_skin.smooth_iters = 4u;
    cloth.render_material_id = mid;
    s.AddMedia(std::move(cloth));

    MediaRecord ball;
    ball.name = "ball";
    ball.kind = MediaRecord::Kind::SoftTet;
    ball.method = MediaRecord::Method::Xpbd;
    ball.tet_sphere.center = nuka::math::Vec3{0.0f, 0.0f, 0.118f};
    ball.tet_sphere.radius = 0.10f;
    ball.tet_sphere.cells = 12u;
    ball.tet_sphere.cell_len = 2.0f * 0.10f / 12.0f;  // the morph's snug default
    ball.xpbd.particle_mass = 0.005f;
    ball.xpbd.friction = 0.6f;
    ball.xpbd.distance_alpha = 3.0e-6f;
    ball.xpbd.volume_alpha = 6.0e-6f;
    ball.xpbd.iters = 44u;
    ball.render_skin.normal_offset = 0.002f;
    ball.render_skin.smooth_iters = 6u;
    ball.render_skin.smooth_lambda = 0.5f;
    ball.render_material_id = mid;
    s.AddMedia(std::move(ball));

    return s;
}

void ExpectMediaEqual(const MediaRecord& a, const MediaRecord& b) {
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(static_cast<int>(a.kind), static_cast<int>(b.kind));
    EXPECT_EQ(static_cast<int>(a.method), static_cast<int>(b.method));
    EXPECT_EQ(a.render_material_id, b.render_material_id);
    EXPECT_EQ(a.baked.nka_path, b.baked.nka_path);
    EXPECT_EQ(a.baked.fourcc, b.baked.fourcc);
    EXPECT_EQ(a.baked.index, b.baked.index);
    // cloth grid
    EXPECT_EQ(a.cloth_grid.nx, b.cloth_grid.nx);
    EXPECT_EQ(a.cloth_grid.ny, b.cloth_grid.ny);
    EXPECT_FLOAT_EQ(a.cloth_grid.spacing, b.cloth_grid.spacing);
    EXPECT_FLOAT_EQ(a.cloth_grid.origin.x, b.cloth_grid.origin.x);
    EXPECT_FLOAT_EQ(a.cloth_grid.origin.y, b.cloth_grid.origin.y);
    EXPECT_FLOAT_EQ(a.cloth_grid.origin.z, b.cloth_grid.origin.z);
    EXPECT_EQ(a.cloth_grid.free, b.cloth_grid.free);
    // tet sphere
    EXPECT_FLOAT_EQ(a.tet_sphere.center.x, b.tet_sphere.center.x);
    EXPECT_FLOAT_EQ(a.tet_sphere.center.z, b.tet_sphere.center.z);
    EXPECT_FLOAT_EQ(a.tet_sphere.radius, b.tet_sphere.radius);
    EXPECT_EQ(a.tet_sphere.cells, b.tet_sphere.cells);
    EXPECT_FLOAT_EQ(a.tet_sphere.cell_len, b.tet_sphere.cell_len);
    // fluid box
    EXPECT_FLOAT_EQ(a.fluid_box.min.x, b.fluid_box.min.x);
    EXPECT_FLOAT_EQ(a.fluid_box.max.z, b.fluid_box.max.z);
    EXPECT_FLOAT_EQ(a.fluid_box.spacing, b.fluid_box.spacing);
    // xpbd material
    EXPECT_FLOAT_EQ(a.xpbd.particle_mass, b.xpbd.particle_mass);
    EXPECT_FLOAT_EQ(a.xpbd.friction, b.xpbd.friction);
    EXPECT_FLOAT_EQ(a.xpbd.distance_alpha, b.xpbd.distance_alpha);
    EXPECT_FLOAT_EQ(a.xpbd.bend_alpha, b.xpbd.bend_alpha);
    EXPECT_FLOAT_EQ(a.xpbd.volume_alpha, b.xpbd.volume_alpha);
    EXPECT_EQ(a.xpbd.iters, b.xpbd.iters);
    EXPECT_FLOAT_EQ(a.xpbd.aero_drag_normal, b.xpbd.aero_drag_normal);
    EXPECT_FLOAT_EQ(a.xpbd.aero_drag_tangent, b.xpbd.aero_drag_tangent);
    EXPECT_FLOAT_EQ(a.xpbd.aero_drag_max_dv, b.xpbd.aero_drag_max_dv);
    // pbf material
    EXPECT_FLOAT_EQ(a.pbf.rest_density, b.pbf.rest_density);
    EXPECT_FLOAT_EQ(a.pbf.support_scale, b.pbf.support_scale);
    EXPECT_EQ(a.pbf.iters, b.pbf.iters);
    EXPECT_FLOAT_EQ(a.pbf.friction, b.pbf.friction);
    EXPECT_EQ(a.pbf.clamp_overdensity, b.pbf.clamp_overdensity);
    EXPECT_EQ(a.pbf.walls_enabled, b.pbf.walls_enabled);
    EXPECT_FLOAT_EQ(a.pbf.floor_z, b.pbf.floor_z);
    EXPECT_EQ(a.pbf.boundary_layers, b.pbf.boundary_layers);
    // mpm material
    EXPECT_FLOAT_EQ(a.mpm.youngs, b.mpm.youngs);
    EXPECT_FLOAT_EQ(a.mpm.poisson, b.mpm.poisson);
    EXPECT_FLOAT_EQ(a.mpm.density, b.mpm.density);
    EXPECT_FLOAT_EQ(a.mpm.model_kind, b.mpm.model_kind);
    EXPECT_FLOAT_EQ(a.mpm.bulk_modulus, b.mpm.bulk_modulus);
    EXPECT_FLOAT_EQ(a.mpm.dx, b.mpm.dx);
    EXPECT_EQ(a.mpm.substeps, b.mpm.substeps);
    EXPECT_FLOAT_EQ(a.mpm.floor_normal.z, b.mpm.floor_normal.z);
    EXPECT_FLOAT_EQ(a.mpm.floor_d, b.mpm.floor_d);
    EXPECT_FLOAT_EQ(a.mpm.floor_friction, b.mpm.floor_friction);
    // render skin
    EXPECT_FLOAT_EQ(a.render_skin.normal_offset, b.render_skin.normal_offset);
    EXPECT_EQ(a.render_skin.smooth_iters, b.render_skin.smooth_iters);
    EXPECT_FLOAT_EQ(a.render_skin.smooth_lambda, b.render_skin.smooth_lambda);
}

}  // namespace

// 1. Media + rigid primitives round-trip field-equal, and the cloth cooks to the
//    SAME lattice (FNV) before/after — proving .nks carries the cook input exactly.
TEST(NksMediaRoundtrip, MediaAndPrimitiveRoundtrip) {
    TempDir tmp;
    const SceneIR orig = BuildMediaScene();
    const fs::path nks = tmp.SubFile("run1", "media.nks");
    nks::Save(orig, nks.string());

    // The saved JSON must carry the additive "media" array.
    {
        std::ifstream in(nks, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        EXPECT_NE(text.find("\"media\""), std::string::npos)
            << "saved .nks lacks the media section";
    }

    const SceneIR loaded = nks::Load(nks.string());

    // rigid primitive round-trips
    ASSERT_EQ(orig.RigidBodyCount(), loaded.RigidBodyCount());
    ASSERT_EQ(orig.ShapeCount(), loaded.ShapeCount());
    EXPECT_EQ(orig.GetBody(0).name, loaded.GetBody(0).name);
    EXPECT_EQ(orig.GetBody(0).is_static, loaded.GetBody(0).is_static);
    EXPECT_EQ(static_cast<int>(orig.GetShape(0).type),
              static_cast<int>(loaded.GetShape(0).type));
    EXPECT_FLOAT_EQ(orig.GetShape(0).half_extents.z, loaded.GetShape(0).half_extents.z);
    EXPECT_FLOAT_EQ(orig.GetShape(0).friction_mu, loaded.GetShape(0).friction_mu);

    // media records round-trip
    ASSERT_EQ(orig.MediaCount(), 2u);
    ASSERT_EQ(loaded.MediaCount(), orig.MediaCount());
    for (MediaId i = 0; i < orig.MediaCount(); ++i) {
        ExpectMediaEqual(orig.GetMedia(i), loaded.GetMedia(i));
    }

    // the cloth cooks to the byte-identical lattice (orig == loaded), and matches
    // the frozen go2_cloth_drape canary FNV the coupled/facade cloth cook pins.
    const cook::XpbdCookInput orig_cloth = cook::BuildClothXpbdInput(orig.GetMedia(0));
    const cook::XpbdCookInput load_cloth = cook::BuildClothXpbdInput(loaded.GetMedia(0));
    ASSERT_FALSE(load_cloth.positions.empty());
    ASSERT_EQ(orig_cloth.positions.size(), load_cloth.positions.size());
    const uint64_t fnv_o = FnvPositions(orig_cloth.positions);
    const uint64_t fnv_l = FnvPositions(load_cloth.positions);
    std::printf("[nks-media] cloth P=%zu fnv_orig=%llu fnv_loaded=%llu\n",
                load_cloth.positions.size(),
                static_cast<unsigned long long>(fnv_o),
                static_cast<unsigned long long>(fnv_l));
    EXPECT_EQ(fnv_o, fnv_l) << "the .nks-loaded cloth cooks to a different lattice";
    EXPECT_EQ(fnv_l, 777503423208024307ull)
        << "the .nks-declared cloth lattice is NOT the frozen canary";

    // the soft-tet cooks to the byte-identical rest lattice before/after.
    const cook::XpbdCookInput orig_ball = cook::BuildSoftTetXpbdInput(orig.GetMedia(1));
    const cook::XpbdCookInput load_ball = cook::BuildSoftTetXpbdInput(loaded.GetMedia(1));
    ASSERT_FALSE(load_ball.positions.empty());
    EXPECT_EQ(FnvPositions(orig_ball.positions), FnvPositions(load_ball.positions))
        << "the .nks-loaded soft-tet cooks to a different rest lattice";
}

// 2. A media-FREE scene is additive + a deterministic Save/Load inverse: the
//    media section is emitted ONLY when present, so examples/scenes/go2.nks
//    re-saves to a byte-identical .nks + .nka on re-load (the §3.7 determinism
//    gate) with no media key, and cooks to the same rigid blob — the Go2 canary
//    path is unaffected.
TEST(NksMediaRoundtrip, Go2NksAdditiveDeterministic) {
    const fs::path go2 = fs::path("examples/scenes/go2.nks");
    if (!fs::exists(go2)) GTEST_SKIP() << "examples/scenes/go2.nks not available";

    const SceneIR loaded = nks::Load(go2.string());
    EXPECT_EQ(loaded.MediaCount(), 0u) << "go2.nks must carry no media";

    // Save -> Load -> Save under the SAME stem; the two saves must be byte-equal
    // (.nks + .nka), the determinism inverse the format guarantees.
    TempDir tmp;
    const fs::path nks1 = tmp.SubFile("run1", "go2.nks");
    nks::Save(loaded, nks1.string());
    const SceneIR reloaded = nks::Load(nks1.string());
    const fs::path nks2 = tmp.SubFile("run2", "go2.nks");
    nks::Save(reloaded, nks2.string());

    EXPECT_EQ(ReadBytes(nks1), ReadBytes(nks2)) << "go2.nks not byte-identical on re-save";
    EXPECT_EQ(ReadBytes(tmp.SubFile("run1", "go2.nka")),
              ReadBytes(tmp.SubFile("run2", "go2.nka")))
        << "go2.nka not byte-identical on re-save";

    // No media key is emitted for a media-free scene (the additive guarantee).
    const std::vector<uint8_t> b1 = ReadBytes(nks1);
    const std::string text1(b1.begin(), b1.end());
    EXPECT_EQ(text1.find("\"media\""), std::string::npos)
        << "a media-free scene must emit no media section";

    // The cooked rigid blob is unchanged (the Go2 canary path is unaffected).
    const CookedBlob a = CookScene(loaded);
    const CookedBlob b = CookScene(reloaded);
    EXPECT_EQ(a.body_count, b.body_count);
    EXPECT_EQ(a.shape_count, b.shape_count);
    EXPECT_EQ(a.contact_params.frictions, b.contact_params.frictions);
    EXPECT_EQ(a.contact_params.solref0, b.contact_params.solref0);
    EXPECT_EQ(a.contact_params.condims, b.contact_params.condims);
}
