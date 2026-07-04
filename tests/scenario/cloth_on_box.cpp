// ---------------------------------------------------------------------------
// Cloth <-> rigid-box contact on the ONE general particle<->body path. A free
// cloth panel dropped onto (a) a STATIC box and (b) a FREE rigid box rests on the
// box top instead of tunnelling through it. The cloth particles are spheres of
// d_min/2 gathered from the SAME body LBVH the rigid narrowphase uses; the box is
// an analytic kKindBox row (amf::SphereBox, no EPA dead band). No cloth-specific
// contact code -- cloth rides path (a) (body<->particle rows -> SolveRowsBlockIsland).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"  // kPairDrivenRowsPerSlot
#include "phi/backend.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace ns = nuka::scene;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;  // collision::ShapeKind::Box
constexpr float kSpacing = 0.05f;  // cloth lattice spacing -> particle radius 0.025.

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;  // pos_iters 4 (default): resting contact pushes out penetration.
}

// A free (all-dynamic) cloth panel centred at (0,0,z), fitting on a box top.
cook::XpbdCookInput MakeClothPanel(uint32_t n, float z) {
    ns::MediaRecord m;
    m.kind = ns::MediaRecord::Kind::Cloth;
    m.method = ns::MediaRecord::Method::Xpbd;
    m.cloth_grid.nx = n; m.cloth_grid.ny = n;
    m.cloth_grid.spacing = kSpacing;
    m.cloth_grid.origin = Vec3{0.0f, 0.0f, z};
    m.cloth_grid.free = true;  // a free drape (no pinned perimeter).
    m.xpbd.particle_mass = 0.01f;
    m.xpbd.iters = 8u;
    m.xpbd.distance_alpha = 1.0e-7f;
    m.xpbd.bend_alpha = 1.0e-4f;
    m.xpbd.friction = 0.6f;
    return cook::BuildClothXpbdInput(m);
}

void AddBox(nk::Model& m, const Vec3& pos, float half, int32_t body_id,
            float inv_mass) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = inv_mass;
    // A solid-box inertia (approx) for the free case; zero for the static case.
    const float ii = inv_mass > 0.0f
                         ? inv_mass / (2.0f / 3.0f * half * half) : 0.0f;
    bi.inv_inertia = Vec3{ii, ii, ii};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// Finish the capacities + cook the cloth on top of an already-populated body set.
void FinishModel(nk::Model& m, const cook::XpbdCookInput& cloth) {
    nk::ModelMaterialBucket mb;
    for (float& v : mb.values) v = 0.0f;
    mb.values[nk::kBucketStaticMu] = 0.8f;
    mb.values[nk::kBucketDynamicMu] = 0.8f;
    mb.values[nk::kBucketDensity] = 1000.0f;
    m.material_buckets = {mb};
    m.body_material_bucket.assign(m.body_init.size(), 0u);
    m.capacities.num_material_buckets = 1u;

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 8u * bodies;  // rigid budget BEFORE the cloth grows it.
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::CookXpbdParticles(m, 1u, cloth);
    m.particles.pp_contact_d_min = kSpacing;  // arm the body<->particle sphere radius.
}

struct Drop {
    float min_z = 0.0f, max_z = 0.0f, mean_z = 0.0f;
    bool finite = true;
};
Drop ProbeCloth(nk::World& w, uint32_t P, uint32_t first, uint32_t count) {
    std::vector<Vec3> pos(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
    Drop d;
    d.min_z = 1e30f; d.max_z = -1e30f;
    double sum = 0.0;
    for (uint32_t i = first; i < first + count; ++i) {
        const Vec3& p = pos[i];
        d.finite = d.finite && std::isfinite(p.x) && std::isfinite(p.y) &&
                   std::isfinite(p.z);
        d.min_z = std::min(d.min_z, p.z);
        d.max_z = std::max(d.max_z, p.z);
        sum += p.z;
    }
    d.mean_z = static_cast<float>(sum / count);
    return d;
}

}  // namespace

// A free cloth panel dropped onto a STATIC box rests on the box top (min z near the
// box top + the particle radius), never tunnelling to a large negative z.
TEST(ClothOnBox, RestsOnStaticBox) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    constexpr float kBoxHalf = 0.20f;      // 0.40 m box; the 0.30 m cloth fits on top.
    constexpr float kBoxTop = 0.10f;       // box centred so its top face is z = 0.10.
    nk::Model m;
    AddBox(m, Vec3{0.0f, 0.0f, kBoxTop - kBoxHalf}, kBoxHalf, /*body_id=*/0, 0.0f);
    AddBox(m, Vec3{5.0f, 0.0f, 0.0f}, 0.05f, /*body_id=*/1, 0.0f);  // far filler.
    const cook::XpbdCookInput cloth = MakeClothPanel(7u, /*z=*/kBoxTop + 0.25f);
    FinishModel(m, cloth);
    const uint32_t P = m.capacities.particles_per_env;
    ASSERT_EQ(P, 49u);

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    Drop d;
    for (uint32_t s = 0; s < 480u; ++s) {
        w.Step();
        if (s % 120u == 0u || s + 1u == 480u) {
            d = ProbeCloth(w, P, 0u, P);
            std::fprintf(stderr, "[cloth-static-box] step=%u min_z=%.4f mean_z=%.4f "
                         "max_z=%.4f finite=%d\n", s, d.min_z, d.mean_z, d.max_z,
                         d.finite ? 1 : 0);
        }
    }
    const float radius = 0.5f * kSpacing;
    ASSERT_TRUE(d.finite) << "cloth went non-finite over the box";
    EXPECT_GT(d.min_z, kBoxTop - 0.02f)
        << "cloth sank into / through the static box (min_z " << d.min_z << ")";
    EXPECT_LT(d.min_z, kBoxTop + radius + 0.05f)
        << "cloth hovered above the box (no contact settled)";
    EXPECT_LT(d.max_z, kBoxTop + 0.20f) << "cloth never fell onto the box";
}

// A free cloth panel dropped onto a FREE (movable) rigid box that itself rests on a
// static ground box. The cloth rests on the free box top (two-way: the box carries
// the cloth load) and nothing tunnels.
TEST(ClothOnBox, RestsOnFreeBox) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    constexpr float kGroundHalf = 0.6f;     // big static ground box, top face z = 0.
    constexpr float kFreeHalf = 0.15f;      // free box on the ground, top face z = 0.30.
    constexpr float kFreeCenter = kFreeHalf;   // bottom face flush on the ground top.
    const float free_top = kFreeCenter + kFreeHalf;

    nk::Model m;
    AddBox(m, Vec3{0.0f, 0.0f, -kGroundHalf}, kGroundHalf, /*body_id=*/0, 0.0f);
    AddBox(m, Vec3{0.0f, 0.0f, kFreeCenter}, kFreeHalf, /*body_id=*/1,
           /*inv_mass=*/1.0f / 2.0f);   // 2 kg free box (heavy vs the 0.5 kg cloth).
    const cook::XpbdCookInput cloth = MakeClothPanel(5u, /*z=*/free_top + 0.25f);
    FinishModel(m, cloth);
    const uint32_t P = m.capacities.particles_per_env;
    ASSERT_EQ(P, 25u);

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    auto box_z = [&] {
        std::vector<Transform> bp(2);
        w.GetData().DownloadField(nk::FieldId::BodyPose, bp.data(),
                                  2u * sizeof(Transform));
        return bp[1].position.z;  // the free box centre.
    };

    Drop d;
    for (uint32_t s = 0; s < 480u; ++s) {
        w.Step();
        if (s % 120u == 0u || s + 1u == 480u) {
            d = ProbeCloth(w, P, 0u, P);
            std::fprintf(stderr, "[cloth-free-box] step=%u cloth_min_z=%.4f "
                         "cloth_mean_z=%.4f box_z=%.4f finite=%d\n", s, d.min_z,
                         d.mean_z, box_z(), d.finite ? 1 : 0);
        }
    }
    const float bz = box_z();
    const float cur_top = bz + kFreeHalf;
    ASSERT_TRUE(d.finite) << "cloth went non-finite over the free box";
    EXPECT_GT(bz, -0.10f) << "the free box tunnelled through the static ground";
    EXPECT_GT(d.min_z, cur_top - 0.03f)
        << "cloth sank into / through the free box (min_z " << d.min_z
        << " box_top " << cur_top << ")";
    EXPECT_LT(d.min_z, cur_top + 0.5f * kSpacing + 0.06f)
        << "cloth hovered above the free box (no contact settled)";
}
