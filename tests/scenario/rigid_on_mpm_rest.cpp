// ---------------------------------------------------------------------------
// Two-way rigid <-> MLS-MPM coupling gate: a free rigid box rests on an MPM bed.
//
// The MPM medium (sim_method=mlsmpm through the config selector, DATA not a demo
// branch) couples to the dynamic body through the env-private grid: the body's
// SDF is rasterized onto the grid, the node velocity is projected onto the body
// surface velocity, and the equal-and-opposite reaction is deposited into the
// SAME body-side sink the row path writes. Asserts:
//   * the per-substep grid->body reaction summed over a step balances gravity
//     (Sigma reaction.z ~= m*g*dt) and the box is held up (read from the per-body
//     diagnostic mpm_body_reaction, NOT ReadoutContactWrench which sums rows);
//   * the free-fall BITE (disable ONLY the dynamic-body BC; the static-plane BC
//     stays on so the medium still rests on the floor) drops the box at ~g and the
//     reaction is ~0;
//   * two runs are byte-identical (the deterministic per-body gather).
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
#include "nk/solve/nk_row.hpp"   // kPairDrivenRowsPerSlot
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace sdfq = nuka::runtime::sdf;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;

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

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

// Scene geometry. An MPM bed (a low slab of densely-sampled particles) on the
// static floor; a free rigid box rests on its top, bottom face just touching.
constexpr float kFloorZ   = 0.0f;
constexpr float kDx       = 0.02f;
constexpr float kBedHalfXY = 0.10f;   // bed half-extent in x/y.
constexpr float kBedTopZ  = 0.08f;    // bed surface height.
constexpr float kBoxHalf  = 0.05f;    // rigid box half-extent (cube).
constexpr float kBoxMass  = 0.5f;     // free rigid box mass.

// Bottom of the box sits a hair into the bed top so the BC band engages at rest.
constexpr float kBoxCenterZ = kBedTopZ + kBoxHalf - 0.004f;

// An analytic box SDF over a narrow band, cooked into the Model directly (no mesh
// dependency): for each band voxel store the exact box signed distance + gradient.
// origin at the band's lower corner so PackSdfCellKey indices are >= 0 (monotonic).
void AddBoxSdf(nk::Model& m, int32_t body_id, float half) {
    const float vh = kDx;                 // SDF voxel size == grid dx.
    const float band = 3.0f * vh;         // narrow band half-width.
    const float ext = half + band;        // span the box + band each axis.
    const int n = static_cast<int>(std::ceil(ext / vh)) + 1;  // voxels per side from 0.
    // Origin so local coords [-ext, +ext] map to voxel indices [0, 2n].
    const Vec3 origin{-static_cast<float>(n) * vh, -static_cast<float>(n) * vh,
                      -static_cast<float>(n) * vh};
    const uint32_t base = static_cast<uint32_t>(m.sdf_cell_values.size());
    auto box_phi = [&](const Vec3& p, Vec3& grad) -> float {
        // Signed distance to an axis-aligned box of half-extent `half`.
        const Vec3 d{std::fabs(p.x) - half, std::fabs(p.y) - half, std::fabs(p.z) - half};
        const Vec3 dpos{std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f)};
        const float outside = std::sqrt(dpos.LengthSq());
        const float inside = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
        const float phi = outside + inside;
        // Gradient: outside -> normalized dpos with sign; inside -> axis of max d.
        if (outside > 1e-6f) {
            const Vec3 g{dpos.x * (p.x < 0 ? -1.f : 1.f), dpos.y * (p.y < 0 ? -1.f : 1.f),
                         dpos.z * (p.z < 0 ? -1.f : 1.f)};
            const float gl = std::sqrt(g.LengthSq());
            grad = gl > 1e-8f ? g * (1.0f / gl) : Vec3{0, 0, 1};
        } else {
            Vec3 g{0, 0, 0};
            if (d.x >= d.y && d.x >= d.z) g.x = p.x < 0 ? -1.f : 1.f;
            else if (d.y >= d.z) g.y = p.y < 0 ? -1.f : 1.f;
            else g.z = p.z < 0 ? -1.f : 1.f;
            grad = g;
        }
        return phi;
    };
    uint32_t count = 0u;
    for (int i = 0; i <= 2 * n; ++i)
        for (int j = 0; j <= 2 * n; ++j)
            for (int k = 0; k <= 2 * n; ++k) {
                const Vec3 p{origin.x + i * vh, origin.y + j * vh, origin.z + k * vh};
                Vec3 grad{0, 0, 0};
                const float phi = box_phi(p, grad);
                if (std::fabs(phi) > band) continue;  // narrow band only.
                m.sdf_cell_keys.push_back(sdfq::PackSdfCellKey(
                    static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                    static_cast<uint32_t>(k)));
                m.sdf_cell_values.push_back(phi);
                m.sdf_cell_gradients.push_back(grad);
                ++count;
            }
    nk::Model::SdfGrid sg;
    sg.origin = origin;
    sg.voxel_size = vh;
    sg.dims[0] = sg.dims[1] = sg.dims[2] = static_cast<uint32_t>(2 * n + 1);
    sg.cell_offset = base;
    sg.cell_count = count;
    const uint32_t grid_idx = static_cast<uint32_t>(m.sdf_grids.size());
    m.sdf_grids.push_back(sg);

    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u;
    sh.sdf_grid = grid_idx;                 // the body BC samples THIS grid.
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// Densely sample the MPM bed on a lattice at dx/2 (8 particles/cell).
cook::MpmCookInput BuildBedInput() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    const float bed_lo_z = kFloorZ + pdx;          // a hair above the floor.
    for (float x = -kBedHalfXY; x <= kBedHalfXY + 1e-4f; x += pdx)
        for (float y = -kBedHalfXY; y <= kBedHalfXY + 1e-4f; y += pdx)
            for (float z = bed_lo_z; z <= kBedTopZ + 1e-4f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    const float density = 1000.0f;
    in.inv_mass.assign(n, 1.0f / (density * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = 5.0e5f;     // a firm bed (holds the box up).
    in.material.poisson = 0.3f;
    in.material.density = density;
    in.material.model_kind = 0.0f;   // fixed-corotated elastic.
    const float span = kBedHalfXY + 4.0f * kDx;
    in.grid_origin = Vec3{-span, -span, kFloorZ - 4.0f * kDx};
    const float top = kBoxCenterZ + kBoxHalf + 4.0f * kDx;
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * span / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top - in.grid_origin.z) / kDx) + 1u;
    in.dx = kDx;
    in.substeps = 25u;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = kFloorZ;
    in.floor_friction = 0.5f;
    return in;
}

// Free rigid box + a far immovable box (the LBVH/contact-budget filler) on top of
// the MPM bed, cooked via the sim_method=mlsmpm selector. bite => the BITE flag.
// box_x shifts the free box in +x (>0 overhangs the bed +x edge -> a tipping torque).
nk::Model BuildModel(bool bite, float box_x = 0.0f) {
    nk::Model m;
    m.capacities.env_count = 1u;

    // Free rigid box (body 0): inv_mass > 0, an inertia for a solid cube.
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{box_x, 0.0f, kBoxCenterZ};
    bi.inv_mass = 1.0f / kBoxMass;
    const float I = (1.0f / 6.0f) * kBoxMass * (2.0f * kBoxHalf) * (2.0f * kBoxHalf);
    bi.inv_inertia = Vec3{1.0f / I, 1.0f / I, 1.0f / I};
    m.body_init.push_back(bi);
    AddBoxSdf(m, 0, kBoxHalf);

    // A far immovable box so bodies_per_env >= 2 builds the arena LBVH cleanly.
    nk::Model::BodyInit bf;
    bf.pose = Transform::Identity();
    bf.pose.position = Vec3{5.0f, 0.0f, 0.0f};
    bf.inv_mass = 0.0f; bf.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bf);
    AddBoxSdf(m, 1, 0.05f);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_sdf_grids = static_cast<uint32_t>(m.sdf_grids.size());
    cap.max_sdf_cells = static_cast<uint32_t>(m.sdf_cell_values.size());
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    // The MPM bed (sim_method=mlsmpm). CookMpmParticles preserves the rigid contact
    // budget (rigid_base) and grows the particle reserve on top.
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, BuildBedInput());

    m.particles.mpm_body_friction = 0.5f;
    m.particles.mpm_bite_disable_dynamic_bc = bite;
    return m;
}

}  // namespace

// Gate (c): the per-substep grid->body reaction summed over a step balances
// gravity and the box is held up (no sink-through, finite).
TEST(RigidOnMpmRest, RestReactionBalancesGravity) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildModel(/*bite=*/false);
    ASSERT_GT(m.capacities.particles_per_env, 200u) << "the bed must be dense";
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const float dt = Cfg().dt;
    const float weight_impulse = kBoxMass * 9.81f * dt;  // m*g*dt per step.

    // Let the box settle onto the bed, then average the reaction over a window.
    Transform body_pose{};
    auto box_z = [&]() {
        w.GetData().DownloadField(nk::FieldId::BodyPose, &body_pose, sizeof(Transform));
        return body_pose.position.z;
    };
    const float z0 = box_z();
    constexpr uint32_t kSettle = 600u;
    for (uint32_t s = 0; s < kSettle; ++s) w.Step();
    const float z_settled = box_z();

    // Average the per-step reaction impulse over a measurement window at rest.
    constexpr uint32_t kWindow = 240u;
    double sum_rz = 0.0;
    float min_z = 1e30f;
    bool nonfinite = false;
    Vec3 reaction{0, 0, 0};
    Vec3 lin_vel{0, 0, 0};
    for (uint32_t s = 0; s < kWindow; ++s) {
        w.Step();
        w.GetData().DownloadField(nk::FieldId::MpmBodyReaction, &reaction, sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &lin_vel, sizeof(Vec3));
        const float z = box_z();
        min_z = std::min(min_z, z);
        nonfinite = nonfinite || !std::isfinite(z) || !std::isfinite(reaction.z);
        sum_rz += reaction.z;
    }
    const float avg_rz = static_cast<float>(sum_rz / kWindow);
    const float rel_err = std::fabs(avg_rz - weight_impulse) / weight_impulse;

    std::fprintf(stderr,
                 "[rest] z0=%.4f settled=%.4f min_z=%.4f weight_impulse=%.6e "
                 "avg_reaction.z=%.6e rel_err=%.3f lin_vel.z=%.4f\n",
                 z0, z_settled, min_z, weight_impulse, avg_rz, rel_err, lin_vel.z);

    EXPECT_FALSE(nonfinite) << "the rest trajectory must be finite";
    // The box stays settled: it does not sink appreciably below its rest height
    // during the window (a large-sink regression trips this).
    EXPECT_GT(min_z, 0.9f * z_settled) << "the box sank away from its rest height";
    EXPECT_LT(std::fabs(lin_vel.z), 0.5f) << "the box is not at rest (large vz)";
    // The grid reaction balances gravity over the step's substeps.
    EXPECT_GT(avg_rz, 0.0f) << "the reaction must push the box UP";
    EXPECT_LT(rel_err, 0.05f)
        << "Sigma reaction.z over substeps must balance m*g*dt within tolerance";
}

// The BITE: disable ONLY the dynamic-body BC (the static-plane BC stays on, so the
// medium still rests on the floor) => the box free-falls and the reaction is ~0.
TEST(RigidOnMpmRest, FreeFallBiteWhenBcDisabled) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildModel(/*bite=*/true);
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const float dt = Cfg().dt;
    Transform body_pose{};
    auto box_z = [&]() {
        w.GetData().DownloadField(nk::FieldId::BodyPose, &body_pose, sizeof(Transform));
        return body_pose.position.z;
    };
    const float z0 = box_z();

    constexpr uint32_t kSteps = 60u;   // 0.25 s of fall.
    float max_react = 0.0f;
    Vec3 reaction{0, 0, 0};
    for (uint32_t s = 0; s < kSteps; ++s) {
        w.Step();
        w.GetData().DownloadField(nk::FieldId::MpmBodyReaction, &reaction, sizeof(Vec3));
        max_react = std::max(max_react, std::fabs(reaction.z));
    }
    const float z1 = box_z();
    const float t = kSteps * dt;
    const float expected_drop = 0.5f * 9.81f * t * t;  // 1/2 g t^2 free-fall.
    const float drop = z0 - z1;

    std::fprintf(stderr,
                 "[bite] z0=%.4f z1=%.4f drop=%.4f expected_drop=%.4f max_react=%.6e\n",
                 z0, z1, drop, expected_drop, max_react);

    // The dynamic-body BC is off => no grid->body reaction.
    EXPECT_LT(max_react, 1e-6f) << "the BITE must zero the dynamic-body reaction";
    // The box free-falls at ~g (within a band; it starts at rest, no support).
    EXPECT_GT(drop, 0.7f * expected_drop) << "the box must fall ~freely under the BITE";
    EXPECT_LT(drop, 1.3f * expected_drop) << "the box fall must match ~1/2 g t^2";
}

// Two runs of the rest scene produce a byte-identical body trajectory (the
// deterministic per-body gather + the deterministic grid transfer).
TEST(RigidOnMpmRest, RestTwoRunByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    constexpr uint32_t kPart = 8u;  // a few leading particle positions to compare.
    // Drop the box OFF-CENTRE so the angular gather carries non-trivial torque (a
    // sign-flipped/nondeterministic angular accumulation would then diverge here).
    auto run = [&](Transform& pose_out, Vec3& lin_out, Vec3& ang_out,
                   std::vector<Vec3>& part_out) -> bool {
        nk::Model m = BuildModel(/*bite=*/false, /*box_x=*/0.07f);
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return false;
        for (uint32_t s = 0; s < 200u; ++s) w.Step();
        part_out.assign(kPart, Vec3::Zero());
        return w.GetData().DownloadField(nk::FieldId::BodyPose, &pose_out,
                                         sizeof(Transform)) &&
               w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &lin_out,
                                         sizeof(Vec3)) &&
               w.GetData().DownloadField(nk::FieldId::BodyAngularVelocity, &ang_out,
                                         sizeof(Vec3)) &&
               w.GetData().DownloadField(nk::FieldId::ParticlePos, part_out.data(),
                                         kPart * sizeof(Vec3));
    };
    Transform pa{}, pc{};
    Vec3 la{}, lc{}, aa{}, ac{};
    std::vector<Vec3> ra, rc;
    ASSERT_TRUE(run(pa, la, aa, ra));
    ASSERT_TRUE(run(pc, lc, ac, rc));
    EXPECT_EQ(0, std::memcmp(&pa, &pc, sizeof(Transform)))
        << "body pose differs run-to-run (a non-deterministic reaction would)";
    EXPECT_EQ(0, std::memcmp(&la, &lc, sizeof(Vec3)))
        << "body linear velocity differs run-to-run";
    EXPECT_EQ(0, std::memcmp(&aa, &ac, sizeof(Vec3)))
        << "body angular velocity differs run-to-run (sign-flipped angular gather)";
    EXPECT_EQ(0, std::memcmp(ra.data(), rc.data(), kPart * sizeof(Vec3)))
        << "particle positions differ run-to-run";
}

// Off-centre drop: the free box overhangs the bed +x edge, so the supported -x
// side gets all the upward reaction. The unsupported +x side must tip DOWN, i.e.
// the body picks up a positive omega_y (right-hand rule: +omega_y drops +x). The
// sign is derived physically; only its sign + non-triviality are asserted.
TEST(RigidOnMpmRest, OffCentreDropInducesTippingTorque) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildModel(/*bite=*/false, /*box_x=*/0.07f);
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    Vec3 ang_vel{0, 0, 0};
    Vec3 reaction{0, 0, 0};
    double peak_wy = 0.0;
    // Step through the first contact transient; track the peak signed omega_y.
    constexpr uint32_t kSteps = 200u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        w.Step();
        w.GetData().DownloadField(nk::FieldId::BodyAngularVelocity, &ang_vel,
                                  sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::MpmBodyReaction, &reaction,
                                  sizeof(Vec3));
        if (std::fabs(ang_vel.y) > std::fabs(peak_wy)) peak_wy = ang_vel.y;
    }
    std::fprintf(stderr,
                 "[torque] peak_omega_y=%.6e final_omega_y=%.6e reaction.z=%.6e\n",
                 peak_wy, ang_vel.y, reaction.z);

    EXPECT_TRUE(std::isfinite(ang_vel.y)) << "the angular trajectory must be finite";
    // The +x overhang tips the box +x-side down -> a positive omega_y.
    EXPECT_GT(peak_wy, 0.0) << "the box must tip toward the unsupported +x side";
    // Non-trivial: well above the symmetric-drop angular noise floor.
    EXPECT_GT(std::fabs(peak_wy), 1e-3)
        << "the induced torque must be non-trivial in magnitude";
}
