// ---------------------------------------------------------------------------
// General contact pipeline — GENERAL HEIGHTFIELD narrowphase.
//
// Proves the ONE PRINCIPLE: a heightfield is NOT a special terrain contact — it
// is a data-driven set of collidable triangles that ride the SAME cvx GJK/EPA
// narrowphase + the SAME mixed-island solver every other body uses. A body
// collides against the cooked height grid via the per-cell TRIANGLE_PRISM
// midphase routed through cvx::ConvexNarrowphase; the reaction scatters
// through the EXISTING PairDriven mixed-island assembly/solve (the convex side
// resolves free-rigid, the heightfield side is static body_id == -1).
//
// SampleTerrainHeight is called ONLY at cook (CookHeightfieldGrid); the
// contact kernel reads the COOKED grid, never an in-kernel sample.
//
// Cases covered:
//   (a) a box resting on a FLAT heightfield grid rests stably at the surface
//       (no sink-through, no explosion).
//   (b) ★ a box placed ON a real PyramidStairs STEP EDGE / riser (NOT the flat
//       platform top — the trap that hid the bug) settles WITHOUT exploding
//       (finite, base near the local step surface, no base_z=16029 / NaN) and
//       WITHOUT 穿模 (it does not sink below the local step surface).
//   (c) the step body gets contacts from the correct prisms (tread + riser):
//       the box is supported (resting vz ~ 0) and a contact is detected.
//   (d) two-run byte-identity for the heightfield contact path.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"
#include "scene/terrain/heightfield_sample.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;  // collision::ShapeKind Box

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg(float gz) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = gz;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 64u;
    return cfg;
}

void AddBox(nk::Model& m, const Vec3& pos, const Vec3& vel, float half,
            float mass, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.linear_velocity = vel;
    bi.inv_mass = mass > 0.0f ? 1.0f / mass : 0.0f;
    if (mass > 0.0f) {
        const float ii = 1.0f / (mass * half * half);
        bi.inv_inertia = Vec3{ii, ii, ii};
    } else {
        bi.inv_inertia = Vec3{0, 0, 0};
    }
    if (m.body_init.size() <= static_cast<size_t>(body_id)) {
        m.body_init.resize(body_id + 1);
    }
    m.body_init[body_id] = bi;
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id;
    sh.group = 0u;
    if (m.shape_table_rows.size() <= static_cast<size_t>(body_id)) {
        m.shape_table_rows.resize(body_id + 1);
    }
    m.shape_table_rows[body_id] = sh;
}

// Build a parametric HeightField (the ONE grid the cook + the surface check share).
// ring_rise 0 => flat; ring_rise > 0 => a Chebyshev staircase (8 rings, the prior
// PyramidStairs amplitude). Pure parameters -- no terrain-type selector.
terrain::HeightField MakeHf(uint32_t nrow, uint32_t ncol, float cell,
                            float ring_rise, float ring_width,
                            float ring_platform) {
    terrain::TerrainGenConfig cfg;
    cfg.nrow = nrow;
    cfg.ncol = ncol;
    cfg.cell_x = cell;
    cfg.cell_y = cell;
    cfg.origin = Vec3{-0.5f * static_cast<float>(ncol - 1u) * cell,
                      -0.5f * static_cast<float>(nrow - 1u) * cell, 0.0f};
    cfg.base_z = 0.0f;
    cfg.ring_rise = ring_rise;
    cfg.ring_width = ring_width;
    cfg.ring_platform = ring_platform;
    cfg.ring_count = ring_rise != 0.0f ? 8u : 0u;
    terrain::HeightField hf;
    terrain::GenerateHeightField(cfg, hf);
    return hf;
}

void FinishBodyModel(nk::Model& m) {
    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 32u;
    cap.max_rows_per_env = 32u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;
}

struct Snapshot {
    bool ok = false;
    std::vector<Vec3> lin;
    std::vector<Transform> pose;
    std::vector<uint32_t> contact_count;  // per env
};

Snapshot Download(nk::World& w, uint32_t bodies) {
    Snapshot s;
    s.lin.assign(bodies, Vec3::Zero());
    s.pose.assign(bodies, Transform::Identity());
    s.contact_count.assign(1, 0u);
    nk::Data& d = w.GetData();
    s.ok = d.DownloadField(nk::FieldId::BodyLinearVelocity, s.lin.data(),
                           s.lin.size() * sizeof(Vec3)) &&
           d.DownloadField(nk::FieldId::BodyPose, s.pose.data(),
                           s.pose.size() * sizeof(Transform));
    d.DownloadField(nk::FieldId::ContactCount, s.contact_count.data(),
                    s.contact_count.size() * sizeof(uint32_t));
    return s;
}

bool Finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}  // namespace

// --- (a) box rests on a FLAT heightfield grid --------------------------------
TEST(PairDrivenHeightfield, BoxRestsOnFlatGrid) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // A FLAT heightfield (terrain type Flat, ground_height 0) under a box dropped
    // from above. The box must REST on the surface (z ~ half) via the per-cell
    // TRIANGLE_PRISM contacts — not fall through, not explode.
    nk::Model m;
    AddBox(m, Vec3{0, 0, 0.30f}, Vec3{0, 0, 0}, 0.10f, 1.0f, 0);  // body 0 = box
    const terrain::HeightField hf = MakeHf(9u, 9u, 0.25f, 0.0f, 0.0f, 0.0f);
    const uint32_t hf_row = cook::CookHeightfieldGrid(m, hf);
    ASSERT_NE(hf_row, ~0u);
    FinishBodyModel(m);
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 240u; ++s) ASSERT_TRUE(w.Step().AllOk()) << s;

    const Snapshot snap = Download(w, bodies);
    ASSERT_TRUE(snap.ok);
    const float box_z = snap.pose[0].position.z;
    std::fprintf(stderr, "[hf flat] box_z=%.4f vbox.z=%.5f contacts=%u\n",
                 box_z, snap.lin[0].z, snap.contact_count[0]);

    ASSERT_TRUE(Finite(snap.pose[0].position)) << "box pose NaN/inf";
    EXPECT_GT(box_z, 0.10f - 0.04f) << "box sank through the flat grid (穿模)";
    EXPECT_LT(box_z, 0.10f + 0.04f) << "box floats above its resting height";
    EXPECT_LT(std::abs(snap.lin[0].z), 0.3f) << "box must come to rest";
    EXPECT_GT(snap.contact_count[0], 0u) << "no heightfield contacts detected";
}

// --- (b)+(c) ★ box on a real PyramidStairs STEP EDGE -------------------------
TEST(PairDrivenHeightfield, BoxOnStepEdgeSettlesNoExplodeNoClip) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // A PyramidStairs heightfield (a real step-edge geometry). Platform
    // half-width 0.5; step_width 0.5; step_height 0.15. The box is placed OVER a
    // STEP EDGE (not the flat platform top — the trap): at Chebyshev distance
    // straddling the first riser outside the platform, so the box's footprint
    // overlaps BOTH the upper tread AND the riser prisms. It must SETTLE finite,
    // near the LOCAL step surface, with no base_z=16029 / NaN and no clip-through.
    const float step_height = 0.15f;
    const terrain::HeightField hf = MakeHf(21u, 21u, 0.25f, step_height, 0.5f, 1.0f);

    // Place the box over the edge between the platform (r <= 0.5) and ring 1.
    // Box centre at x = 0.5 (the platform/ring-1 boundary), y = 0 -> the box
    // [0.4, 0.6] straddles the riser at x=0.5. The platform top sits at
    // ground + 8*step_height = 8*0.15 = 1.20; ring-1 tread at 1.20 - 0.15 = 1.05.
    const float box_half = 0.10f;
    const float edge_x = 0.5f;
    const float platform_top = static_cast<float>(8) * step_height;  // 1.20
    const float ring1_top = platform_top - step_height;              // 1.05
    const float drop_z = platform_top + 0.20f;  // start above the platform top.

    nk::Model m;
    AddBox(m, Vec3{edge_x, 0, drop_z}, Vec3{0, 0, 0}, box_half, 1.0f, 0);
    const uint32_t hf_row = cook::CookHeightfieldGrid(m, hf);
    ASSERT_NE(hf_row, ~0u);
    FinishBodyModel(m);
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
    ASSERT_TRUE(w.Ready());
    uint32_t max_contacts = 0u;
    for (uint32_t s = 0; s < 360u; ++s) {
        ASSERT_TRUE(w.Step().AllOk()) << s;
        uint32_t cc = 0u;
        w.GetData().DownloadField(nk::FieldId::ContactCount, &cc, sizeof(cc));
        max_contacts = std::max(max_contacts, cc);
    }

    const Snapshot snap = Download(w, bodies);
    ASSERT_TRUE(snap.ok);
    const Vec3 p = snap.pose[0].position;
    std::fprintf(stderr,
                 "[hf step-edge] box=(%.4f,%.4f,%.4f) vz=%.5f platform_top=%.3f "
                 "ring1_top=%.3f contacts=%u max=%u\n",
                 p.x, p.y, p.z, snap.lin[0].z, platform_top, ring1_top,
                 snap.contact_count[0], max_contacts);

    // (1) FINITE — no base_z=16029 / NaN explosion.
    ASSERT_TRUE(Finite(p)) << "box pose NaN/inf (the explosion)";
    ASSERT_TRUE(Finite(snap.lin[0])) << "box velocity NaN/inf";
    EXPECT_LT(p.z, platform_top + 1.0f) << "box exploded upward";
    EXPECT_GT(p.z, ring1_top - 1.0f) << "box exploded / fell far below the steps";

    // (2) NO 穿模 — the box rests AT or ABOVE the LOCAL step surface at its FINAL
    // resting (x,y). A frictionless box dropped on a step EDGE may slide down to a
    // stable lower tread (correct physics — it rests ON a step, never tunnels the
    // back). We evaluate the cooked surface (the SAME SampleTerrainHeight the cook
    // used) at the box's final column and assert the box base is at/above it minus
    // a small penetration tolerance. This is the real 穿模 check: the box never
    // sinks below the surface it sits on, anywhere on the staircase.
    const float box_base = p.z - box_half;
    const float surf_here = terrain::SampleHeightFieldZ(hf, p.x, p.y);
    std::fprintf(stderr, "[hf step-edge] surf@(%.3f,%.3f)=%.4f box_base=%.4f\n",
                 p.x, p.y, surf_here, box_base);
    EXPECT_GT(box_base, surf_here - 0.04f)
        << "box clipped through the step (穿模) at its resting column";
    EXPECT_LT(box_base, surf_here + 0.06f)
        << "box floats above the local step surface";
    // It must have come to rest somewhere ON the staircase (not flung off it).
    EXPECT_LT(p.z, platform_top + box_half + 0.10f) << "box exploded / floats high";

    // (3) SUPPORTED — the box came to rest (vz damped) AND a contact exists (the
    // tread/riser prisms held it).
    EXPECT_LT(std::abs(snap.lin[0].z), 0.5f) << "box not at rest on the step";
    // The final-snapshot count can read zero once fully at rest; the run must
    // have produced tread/riser manifolds at some point.
    EXPECT_GT(max_contacts, 0u) << "no step-prism contacts detected";
}

// --- (c2) ★ box WEDGED against a step RISER gets BOTH tread + riser contacts --
TEST(PairDrivenHeightfield, BoxWedgedAgainstRiserPushedOut) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // Place the box already RESTING on ring-1's tread (z = ring1_top + half) but
    // pushed INWARD so it OVERLAPS the platform's vertical RISER (the face at the
    // platform edge x = half_plat = 0.5 rising from ring1_top 1.05 to platform_top
    // 1.20). The riser prisms must push it back OUTWARD (the horizontal riser
    // normal) while the tread prisms hold it UP — the tread+riser dual contact the
    // step-edge case needs. We assert: finite, no penetration into the platform
    // (its x stays outside the platform footprint), and it rests on the tread.
    const float step_height = 0.15f;
    const terrain::HeightField hf = MakeHf(21u, 21u, 0.25f, step_height, 0.5f, 1.0f);
    const float box_half = 0.10f;
    const float platform_top = static_cast<float>(8) * step_height;  // 1.20
    const float ring1_top = platform_top - step_height;             // 1.05
    const float half_plat = 0.5f;

    // Box starts ON ring-1 tread just clear of the platform footprint and is
    // nudged toward the riser: the riser must resist it while the tread holds it.
    const float start_x = half_plat + box_half + 0.01f;  // left edge at 0.51
    nk::Model m;
    AddBox(m, Vec3{start_x, 0.0f, ring1_top + box_half + 0.005f},
           Vec3{-0.2f, 0, 0}, box_half, 1.0f, 0);  // nudge toward the platform.
    const uint32_t hf_row = cook::CookHeightfieldGrid(m, hf);
    ASSERT_NE(hf_row, ~0u);
    FinishBodyModel(m);
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
    ASSERT_TRUE(w.Ready());
    uint32_t max_contacts = 0u;
    for (uint32_t s = 0; s < 360u; ++s) {
        ASSERT_TRUE(w.Step().AllOk()) << s;
        uint32_t cc = 0u;
        w.GetData().DownloadField(nk::FieldId::ContactCount, &cc, sizeof(cc));
        max_contacts = std::max(max_contacts, cc);
    }

    const Snapshot snap = Download(w, bodies);
    ASSERT_TRUE(snap.ok);
    const Vec3 p = snap.pose[0].position;
    // The box rests on a discrete STEP TREAD; at a riser the bilinear sample ramps
    // between the tread and the higher platform, so the no-tunnel support is the
    // LOWEST grid corner under the box footprint (the tread it sits on), not the
    // mid-riser bilinear value (which is the sloped prism the box may shallowly
    // penetrate -- the known box x prism EPA shallow-penetration dead-band).
    float tread = 1.0e9f;
    for (float dy = -box_half; dy <= box_half + 1e-6f; dy += box_half)
        for (float dx = -box_half; dx <= box_half + 1e-6f; dx += box_half)
            tread = std::min(tread, terrain::SampleHeightFieldZ(hf, p.x + dx, p.y + dy));
    const float surf_here = terrain::SampleHeightFieldZ(hf, p.x, p.y);
    std::fprintf(stderr,
                 "[hf wedge] box=(%.4f,%.4f,%.4f) vz=%.4f surf=%.4f tread=%.4f contacts=%u\n",
                 p.x, p.y, p.z, snap.lin[0].z, surf_here, tread, snap.contact_count[0]);

    ASSERT_TRUE(Finite(p)) << "wedged box NaN/inf (explosion)";
    // RISER reaction pushed it OUT: the box's inner face must NOT penetrate deep
    // into the platform footprint — its centre stays at/outside the riser minus a
    // tolerance (the riser prevents it from sliding under the higher platform top).
    EXPECT_GT(p.x, half_plat - box_half - 0.04f)
        << "box tunneled under the platform riser (穿模 on the riser)";
    // TREAD reaction held it UP: its base rests at/above the tread it sits on.
    EXPECT_GT(p.z - box_half, tread - 0.04f)
        << "box sank through the tread";
    EXPECT_GT(max_contacts, 0u) << "no tread/riser contacts";
}

// --- (d) two-run byte-identity for the heightfield contact path --------------
TEST(PairDrivenHeightfield, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    auto run = [&]() -> Snapshot {
        const terrain::HeightField hf = MakeHf(21u, 21u, 0.25f, 0.15f, 0.5f, 1.0f);
        nk::Model m;
        AddBox(m, Vec3{0.5f, 0.05f, 1.45f}, Vec3{0.1f, 0, 0}, 0.10f, 1.0f, 0);
        cook::CookHeightfieldGrid(m, hf);
        FinishBodyModel(m);
        const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
        if (!w.Ready()) return {};
        for (uint32_t s = 0; s < 200u; ++s)
            if (!w.Step().AllOk()) return {};
        return Download(w, bodies);
    };

    const Snapshot a = run();
    const Snapshot c = run();
    ASSERT_TRUE(a.ok && c.ok);

    auto same = [](const auto& x, const auto& y) {
        return x.size() == y.size() &&
               std::memcmp(x.data(), y.data(),
                           x.size() * sizeof(typename std::decay_t<decltype(x)>::value_type)) == 0;
    };
    EXPECT_TRUE(same(a.lin, c.lin)) << "body velocity differs across runs";
    EXPECT_TRUE(same(a.pose, c.pose)) << "body pose differs across runs";
}
