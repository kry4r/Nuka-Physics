// Grid and rigid endpoints share finite-mass contact rows.
// Complete world steps check momentum, reaction balance and contact convergence.

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
#include "nk/solve/point_endpoint.hpp"
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

nk::Model FiniteMassImpact(float mass, uint32_t owners, uint32_t capacity, float gap = 0.0f) {
    nk::Model model;
    auto& cap = model.capacities;
    cap.bodies_per_env = cap.max_bodies_total = owners;
    cap.max_contacts_per_env = owners * 4u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    for (uint32_t i = 0u; i < owners; ++i) {
        nk::Model::BodyInit body;
        body.pose = Transform::Identity();
        body.pose.position = {-2.0f - gap, 0.0f, 0.0f};
        body.inv_mass = 1.0f / mass;
        body.inv_inertia = Vec3{1.0f, 1.0f, 1.0f} * (3.0f / (8.0f * mass));
        model.body_init.push_back(body);
        nk::Model::PairDrivenShape shape;
        shape.kind = kKindBox;
        shape.body_id = static_cast<int32_t>(i);
        shape.params[0] = shape.params[1] = shape.params[2] = 2.0f;
        shape.contype = 1u;
        shape.conaffinity = 0u;
        model.shape_table_rows.push_back(shape);
    }
    cook::MpmCookInput input;
    input.positions = {{0.0f, 0.0f, 0.0f}};
    input.velocities = {{-1.0f, 0.0f, 0.0f}};
    input.inv_mass = {1.0f};
    input.vol0 = {0.001f};
    input.material.youngs = 1000.0f;
    input.material.poisson = 0.3f;
    input.material.density = 1000.0f;
    input.grid_origin = {-0.3f, -0.3f, -0.3f};
    input.grid_dims[0] = input.grid_dims[1] = input.grid_dims[2] = 7u;
    input.dx = 0.1f;
    input.floor_d = -10.0f;
    input.floor_friction = 0.0f;
    input.contact_capacity = capacity;
    cook::CookMpmParticles(model, 1u, input);
    model.particles.mpm_body_friction = 0.0f;
    return model;
}

template <class T>
std::vector<T> ReadValues(nk::World& world, nk::FieldId field) {
    const size_t bytes = world.GetModel().capacities.ElementCount(field) * nk::LayoutOf(field).elem_size;
    std::vector<T> values(bytes / sizeof(T));
    EXPECT_TRUE(world.GetData().DownloadField(field, values.data(), bytes));
    return values;
}

}  // namespace

TEST(RigidOnMpmRest, FiniteMassMultipleOwnersConserveMomentum) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (const float ratio : {1.0f, 10.0f, 100.0f, 1000.0f}) {
        for (const uint32_t owners : {1u, 2u}) {
            for (const float dt : {0.0001f, 0.0002f}) {
                SCOPED_TRACE(::testing::Message() << "mass=" << ratio << " owners=" << owners << " dt=" << dt);
                auto config = Cfg();
                config.dt = dt;
                config.gravity[2] = 0.0f;
                config.vel_iters = 4096u;
                config.pos_iters = 3u;
                nk::World world(FiniteMassImpact(ratio, owners, 2u * nk::kMpmStencilNodes),
                                1u, backend.dev, backend.backend, config);
                ASSERT_TRUE(world.Ready()) << world.CreationError();
                ASSERT_TRUE(world.Step().AllOk());
                const auto& cap = world.GetModel().capacities;
                const auto mass = ReadValues<float>(world, nk::FieldId::GridMass);
                const auto grid = ReadValues<Vec3>(world, nk::FieldId::GridVelocity);
                const auto rigid = ReadValues<Vec3>(world, nk::FieldId::BodyLinearVelocity);
                const auto angular = ReadValues<Vec3>(world, nk::FieldId::BodyAngularVelocity);
                const auto particles = ReadValues<Vec3>(world, nk::FieldId::ParticleVel);
                const auto reaction = ReadValues<Vec3>(world, nk::FieldId::MpmBodyReaction);
                const auto rows = ReadValues<nk::NkRow>(world, nk::FieldId::Urows);
                const auto impulses = ReadValues<float>(world, nk::FieldId::Lambda);
                const auto pseudo = ReadValues<float>(world, nk::FieldId::RowPseudoLambda);
                const auto attempted = ReadValues<uint64_t>(world, nk::FieldId::GridContactAttempted);
                const auto retained = ReadValues<uint32_t>(world, nk::FieldId::GridContactRetained);
                const auto ranges = ReadValues<nk::PointEndpointRange>(world, nk::FieldId::PointEndpointRanges);
                const auto terms = ReadValues<nk::PointEndpointTerm>(world, nk::FieldId::PointEndpointTerms);
                ASSERT_EQ(attempted.size(), 1u);
                EXPECT_GT(attempted[0], 0u);
                EXPECT_EQ(attempted[0], retained[0]);
                EXPECT_EQ(ReadValues<uint64_t>(world, nk::FieldId::GridContactOverflow)[0], 0u);
                EXPECT_EQ(ReadValues<uint32_t>(world, nk::FieldId::EnvStatus)[0], 0u);
                std::vector<bool> touched(mass.size());
                double residual = 0.0;
                uint32_t contacts = 0u;
                for (uint32_t r = 0u; r < rows.size(); ++r) {
                    const auto& row = rows[r];
                    if (!(row.flags & nk::nk_row_flags::kContactNormal) || row.a.kind != nk::kNkSidePointEndpoint) continue;
                    EXPECT_EQ(row.b.kind, nk::kNkSideRigid);
                    EXPECT_NE(row.flags & nk::nk_row_flags::kSpeculative, 0u);
                    EXPECT_FLOAT_EQ(pseudo[r], 0.0f);
                    Vec3 interpolated{};
                    const auto range = ranges[row.a.index];
                    for (uint32_t i = 0u; i < range.count; ++i) {
                        const auto& term = terms[range.first + i];
                        ASSERT_EQ(term.kind, nk::kNkSideGrid);
                        touched[term.index] = true;
                        interpolated += term.Multiply(grid[term.index]);
                    }
                    EXPECT_LT((interpolated - particles[0]).Length(), 2.0e-6f);
                    const double velocity = row.a.jlin.Dot(interpolated) +
                        row.b.jlin.Dot(rigid[row.b.index]) + row.b.jang.Dot(angular[row.b.index]) - row.rhs * dt;
                    residual = std::max(residual, impulses[r] > 1.0e-9f ? std::fabs(velocity) : std::max(-velocity, 0.0));
                    ++contacts;
                }
                EXPECT_EQ(contacts, retained[0]);
                std::printf("[finite-mass] mass=%g owners=%u dt=%g iterations=%u residual=%.9e contacts=%u\n",
                            ratio, owners, dt, config.vel_iters, residual, contacts);
                EXPECT_LT(residual, 2.0e-5);
                double coupled_mass = 0.0, grid_px = 0.0, grid_energy = 0.0;
                Vec3 momentum{}, moment{};
                for (uint32_t n = 0u; n < mass.size(); ++n) {
                    if (touched[n]) coupled_mass += mass[n];
                    grid_px += mass[n] * grid[n].x;
                    grid_energy += 0.5 * mass[n] * grid[n].LengthSq();
                    const Vec3 position{-0.3f + (n % 7u) * 0.1f,
                        -0.3f + ((n / 7u) % 7u) * 0.1f, -0.3f + (n / 49u) * 0.1f};
                    momentum += grid[n] * mass[n];
                    moment += position.Cross(grid[n] * mass[n]);
                }
                const double common = -coupled_mass / (coupled_mass + ratio * owners);
                double particle_px = particles[0].x;
                for (uint32_t b = 0u; b < owners; ++b) {
                    EXPECT_NEAR(rigid[b].x, common, 2.0e-5);
                    EXPECT_LT(angular[b].Length(), 2.0e-5f);
                    EXPECT_LT((reaction[b] - rigid[b] * ratio).Length(), 2.0e-5f);
                    momentum += rigid[b] * ratio;
                    moment += Vec3{-2.0f, 0.0f, 0.0f}.Cross(rigid[b] * ratio) + angular[b] * (8.0f * ratio / 3.0f);
                    grid_px += ratio * rigid[b].x;
                    particle_px += ratio * rigid[b].x;
                    grid_energy += 0.5 * ratio * rigid[b].LengthSq() + (4.0 / 3.0) * ratio * angular[b].LengthSq();
                }
                EXPECT_NEAR(grid_px, -1.0, 2.0e-5);
                EXPECT_NEAR(particle_px, -1.0, 2.0e-5);
                EXPECT_LT((momentum - Vec3{-1.0f, 0.0f, 0.0f}).Length(), 2.0e-5f);
                EXPECT_LT(moment.Length(), 2.0e-5f);
                EXPECT_LE(grid_energy, 0.5 + 1.0e-6);
                EXPECT_EQ(ReadValues<uint64_t>(world, nk::FieldId::GridContactPeak)[0], attempted[0]);
                ASSERT_EQ(world.Reset({0u}), nphi::Status::Ok);
                EXPECT_EQ(ReadValues<uint64_t>(world, nk::FieldId::GridContactPeak)[0], 0u);
            }
        }
    }
    auto separated_config = Cfg();
    separated_config.dt = 0.0002f;
    separated_config.gravity[2] = 0.0f;
    nk::World separated(FiniteMassImpact(1.0f, 2u, 2u * nk::kMpmStencilNodes, 0.025f),
                        1u, backend.dev, backend.backend, separated_config);
    ASSERT_TRUE(separated.Ready());
    ASSERT_TRUE(separated.Step().AllOk());
    for (const auto impulse : ReadValues<Vec3>(separated, nk::FieldId::MpmBodyReaction))
        EXPECT_EQ(impulse.LengthSq(), 0.0f);
    EXPECT_NEAR(ReadValues<Vec3>(separated, nk::FieldId::ParticleVel)[0].x, -1.0f, 2.0e-6f);
    nk::World limited(FiniteMassImpact(1.0f, 2u, 1u), 1u, backend.dev, backend.backend, Cfg());
    ASSERT_TRUE(limited.Ready());
    ASSERT_TRUE(limited.Step().AllOk());
    const auto attempted = ReadValues<uint64_t>(limited, nk::FieldId::GridContactAttempted)[0];
    EXPECT_GT(attempted, 1u);
    EXPECT_EQ(ReadValues<uint32_t>(limited, nk::FieldId::GridContactRetained)[0], 1u);
    EXPECT_EQ(ReadValues<uint64_t>(limited, nk::FieldId::GridContactOverflow)[0], attempted - 1u);
    EXPECT_NE(ReadValues<uint32_t>(limited, nk::FieldId::EnvStatus)[0] & nphi::kEnvStatusGridContactOverflow, 0u);
}

TEST(RigidOnMpmRest, MaterialPointGapAndThinColliderUseFinalVelocity) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (float mass : {0.1f, 1.0f, 1000.0f}) {
        for (float dt : {0.001f, 0.002f}) {
            for (float gap : {-0.001f, 0.0f, 0.004f, 0.03f}) {
                SCOPED_TRACE(::testing::Message() << "mass=" << mass << " dt=" << dt << " gap=" << gap);
                auto model = FiniteMassImpact(mass, 1u, nk::kMpmStencilNodes);
                const float half = 0.002f, band = 0.002f, incoming = -10.0f;
                model.body_init[0].pose.position.x = 0.013f;
                model.shape_table_rows[0].params[0] = half;
                model.particles.mpm_body_band = band;
                const float initial = model.body_init[0].pose.position.x + half + gap;
                model.particles.initial_pos[0].x = initial;
                model.particles.initial_vel[0].x = incoming;
                auto config = Cfg();
                config.dt = dt;
                config.gravity[2] = 0.0f;
                config.pos_iters = 4u;
                config.pos_beta = 1.0f;
                config.pos_slop = 0.0f;
                nk::World world(std::move(model), 1u, backend.dev, backend.backend, config);
                ASSERT_TRUE(world.Ready()) << world.CreationError();
                ASSERT_TRUE(world.Step().AllOk());
                EXPECT_EQ(ReadValues<uint32_t>(world, nk::FieldId::EnvStatus)[0], 0u);
                const auto point = ReadValues<Vec3>(world, nk::FieldId::ParticlePos)[0];
                const auto particle_velocity = ReadValues<Vec3>(world, nk::FieldId::ParticleVel)[0];
                const auto body = ReadValues<Transform>(world, nk::FieldId::BodyPose)[0];
                const auto body_velocity = ReadValues<Vec3>(world, nk::FieldId::BodyLinearVelocity)[0];
                const double impulse = std::max(0.0, (-double(std::max(gap, 0.0f)) / dt - incoming)
                    / (1.0 + 1.0 / mass));
                const double recovery = std::max(-double(gap), 0.0) / (1.0 + 1.0 / mass);
                EXPECT_NEAR(particle_velocity.x, incoming + impulse, 3.0e-5);
                EXPECT_NEAR(body_velocity.x, -impulse / mass, 3.0e-5);
                EXPECT_NEAR(particle_velocity.x + mass * body_velocity.x, incoming, 3.0e-5);
                EXPECT_NEAR(point.x, initial + particle_velocity.x * dt + recovery, 3.0e-7f);
                EXPECT_NEAR(body.position.x, 0.013f + body_velocity.x * dt - recovery / mass, 3.0e-7f);
                EXPECT_LE(particle_velocity.LengthSq() + mass * body_velocity.LengthSq(), incoming * incoming + 1.0e-4f);
                EXPECT_GE(point.x - body.position.x - half, -2.0e-7f);
                EXPECT_NEAR(point.x - body.position.x - half,
                            std::max(0.0f, gap + incoming * dt), 3.0e-7f);
            }
        }
    }
}

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
        ASSERT_TRUE(w.Step().AllOk());
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

TEST(RigidOnMpmRest, SharedIntervalsMatchCompleteWorldSteps) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool planned : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "graph=" << planned);
        auto make_model = [] {
            auto model = BuildModel(false, 0.07f);
            model.body_init[0].inertial_frame.position = {0.004f, -0.003f, 0.002f};
            return model;
        };
        auto interval_model = make_model();
        const uint32_t substeps = interval_model.particles.mpm_substeps;
        interval_model.particles.mpm_substeps = 1u;
        auto config = Cfg();
        auto interval_config = config;
        interval_config.dt /= static_cast<float>(substeps);
        nk::World outer(make_model(), 2u, backend.dev, backend.backend, config);
        nk::World intervals(std::move(interval_model), 2u, backend.dev, backend.backend, interval_config);
        ASSERT_TRUE(outer.Ready()) << outer.CreationError();
        ASSERT_TRUE(intervals.Ready()) << intervals.CreationError();
        const auto& cap = outer.GetModel().capacities;
        const uint32_t bodies = cap.bodies_per_env * cap.env_count;
        const uint32_t particles = cap.particles_per_env * cap.env_count;
        const auto advance = [&](nk::World& world) {
            return planned ? world.StepPlanned() : world.Step().result;
        };
        const auto read_vectors = [&](nk::World& world, nk::FieldId field) {
            std::vector<Vec3> result(bodies);
            EXPECT_TRUE(world.GetData().DownloadField(field, result.data(), result.size() * sizeof(Vec3)));
            return result;
        };
        const auto compare = [&](nk::FieldId field, size_t bytes) {
            std::vector<uint8_t> a(bytes), b(bytes);
            ASSERT_TRUE(outer.GetData().DownloadField(field, a.data(), bytes));
            ASSERT_TRUE(intervals.GetData().DownloadField(field, b.data(), bytes));
            EXPECT_EQ(a, b) << nk::FieldName(field);
        };
        const auto expect_vector = [](Vec3 a, Vec3 b, float tolerance) {
            EXPECT_NEAR(a.x, b.x, tolerance);
            EXPECT_NEAR(a.y, b.y, tolerance);
            EXPECT_NEAR(a.z, b.z, tolerance);
        };
        std::vector<Vec3> forces(bodies), torques(bodies);
        forces[0] = {0.12f, -0.06f, 0.08f};
        torques[0] = {0.001f, 0.002f, -0.001f};
        for (uint32_t step = 0u; step < 4u; ++step) {
            SCOPED_TRACE(step);
            if (step == 2u) {
                ASSERT_EQ(outer.Reset({1u}), nphi::Status::Ok);
                ASSERT_EQ(intervals.Reset({1u}), nphi::Status::Ok);
            }
            const auto before = read_vectors(outer, nk::FieldId::BodyLinearVelocity);
            ASSERT_TRUE(outer.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), bodies * sizeof(Vec3)));
            ASSERT_TRUE(outer.GetData().UploadField(nk::FieldId::BodyTorque, torques.data(), bodies * sizeof(Vec3)));
            ASSERT_EQ(advance(outer), nphi::Status::Ok);
            std::vector<Vec3> impulse(bodies), world_moment(bodies);
            std::vector<uint32_t> combined_status(cap.env_count, 0u);
            for (uint32_t interval = 0u; interval < substeps; ++interval) {
                ASSERT_TRUE(intervals.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), bodies * sizeof(Vec3)));
                ASSERT_TRUE(intervals.GetData().UploadField(nk::FieldId::BodyTorque, torques.data(), bodies * sizeof(Vec3)));
                ASSERT_EQ(advance(intervals), nphi::Status::Ok);
                const auto linear = read_vectors(intervals, nk::FieldId::MpmBodyReaction);
                const auto angular = read_vectors(intervals, nk::FieldId::MpmBodyAngReaction);
                std::vector<Transform> poses(bodies);
                ASSERT_TRUE(intervals.GetData().DownloadField(nk::FieldId::BodyPose, poses.data(), bodies * sizeof(Transform)));
                std::vector<uint32_t> status(cap.env_count);
                ASSERT_TRUE(intervals.GetData().DownloadField(nk::FieldId::EnvStatus, status.data(), status.size() * sizeof(uint32_t)));
                for (uint32_t env = 0u; env < cap.env_count; ++env) combined_status[env] |= status[env];
                for (uint32_t body = 0u; body < bodies; ++body) {
                    const auto& frame = intervals.GetModel().body_init[body % cap.bodies_per_env].inertial_frame;
                    const Vec3 origin = poses[body].TransformPoint(frame.position);
                    impulse[body] += linear[body];
                    world_moment[body] += angular[body] + origin.Cross(linear[body]);
                }
            }
            compare(nk::FieldId::BodyPose, bodies * sizeof(Transform));
            compare(nk::FieldId::BodyLinearVelocity, bodies * sizeof(Vec3));
            compare(nk::FieldId::BodyAngularVelocity, bodies * sizeof(Vec3));
            compare(nk::FieldId::ParticlePos, particles * sizeof(Vec3));
            compare(nk::FieldId::ParticleVel, particles * sizeof(Vec3));
            compare(nk::FieldId::ParticleF, particles * 9u * sizeof(float));
            compare(nk::FieldId::ParticleC, particles * 9u * sizeof(float));
            const auto linear = read_vectors(outer, nk::FieldId::MpmBodyReaction);
            const auto angular = read_vectors(outer, nk::FieldId::MpmBodyAngReaction);
            const auto after = read_vectors(outer, nk::FieldId::BodyLinearVelocity);
            std::vector<Transform> poses(bodies);
            ASSERT_TRUE(outer.GetData().DownloadField(nk::FieldId::BodyPose, poses.data(), bodies * sizeof(Transform)));
            for (uint32_t body = 0u; body < bodies; ++body) {
                const auto& initial = outer.GetModel().body_init[body % cap.bodies_per_env];
                const Vec3 origin = poses[body].TransformPoint(initial.inertial_frame.position);
                expect_vector(linear[body], impulse[body], 1.0e-7f);
                expect_vector(angular[body] + origin.Cross(linear[body]), world_moment[body], 1.0e-7f);
                if (initial.inv_mass > 0.0f) {
                    const Vec3 gravity{config.gravity[0], config.gravity[1], config.gravity[2]};
                    const Vec3 change = (after[body] - before[body]) / initial.inv_mass;
                    const Vec3 external = (forces[body] + gravity / initial.inv_mass) * config.dt;
                    expect_vector(change, external + linear[body], 2.0e-6f);
                }
            }
            EXPECT_GT(linear[0].z, 1.0e-6f);
            EXPECT_EQ(read_vectors(outer, nk::FieldId::BodyForce), std::vector<Vec3>(bodies));
            EXPECT_EQ(read_vectors(outer, nk::FieldId::BodyTorque), std::vector<Vec3>(bodies));
            std::vector<uint32_t> status(cap.env_count);
            ASSERT_TRUE(outer.GetData().DownloadField(nk::FieldId::EnvStatus, status.data(), status.size() * sizeof(uint32_t)));
            EXPECT_EQ(status, combined_status);
        }
    }
}
