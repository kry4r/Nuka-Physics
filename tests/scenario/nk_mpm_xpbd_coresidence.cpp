// ---------------------------------------------------------------------------
// MLS-MPM + XPBD co-residence in ONE nk::Model ([mpm | xpbd] layout). Builds a
// granular column (MPM, slice [0, n_mpm)) and a perimeter-pinned cloth membrane
// (XPBD, slice [n_mpm, P)) co-resident via scene::cook::CookMpmXpbd, then co-steps
// nk::World on the MpmXpbd particle-op path. Asserts:
//
//   1. SUPERSET BYTE-IDENTITY: an MPM-only co-residence cook == CookMpmParticles;
//      an XPBD-only co-residence cook == CookXpbdParticles (single-medium goldens
//      unaffected).
//   2. CO-RESIDENT LAYOUT: mode == MpmXpbd, n_mpm_particles == n_mpm, the XPBD
//      constraint indices land in [n_mpm, P), the slices do not overlap.
//   3. CO-STEP, NO CROSS-CORRUPTION: the granular slice settles (finite, no grid
//      escape) while the cloth slice's pinned corners stay put and its interior
//      sags under gravity -- the MPM transfer never overwrites a cloth particle and
//      XpbdProject never zeroes an MPM velocity. Run with env_count = 2 to catch
//      per-env sub-slice striding.
//   4. D1: the co-resident forward is two-run byte-exact.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "constraint/coulomb_contact.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace ns = nuka::scene;
using nuka::math::Vec3;

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

namespace {

template <class T>
std::vector<T> ReadSurfaceField(nk::World& world, nk::FieldId field) {
    const size_t bytes = world.GetModel().capacities.ElementCount(field) * nk::LayoutOf(field).elem_size;
    std::vector<T> result(bytes / sizeof(T));
    EXPECT_TRUE(world.GetData().DownloadField(field, result.data(), bytes));
    return result;
}

nk::Model SurfaceImpactModel(float surface_mass, float friction, uint32_t surfaces, uint32_t envs) {
    cook::MpmCookInput mpm;
    mpm.positions = {{0.075f, 0.025f, -0.0125f}};
    mpm.velocities = {{-1.0f, 0.4f, 0.3f}};
    mpm.inv_mass = {1.0f};
    mpm.vol0 = {0.001f};
    mpm.material.youngs = 1000.0f;
    mpm.material.poisson = 0.3f;
    mpm.material.density = 1000.0f;
    mpm.grid_origin = {-0.3f, -0.4f, -0.4f};
    mpm.grid_dims[0] = 7u;
    mpm.grid_dims[1] = mpm.grid_dims[2] = 9u;
    mpm.dx = 0.1f;
    mpm.substeps = 1u;
    mpm.floor_d = -10.0f;
    mpm.floor_friction = 0.0f;
    mpm.contact_capacity = surfaces * nk::kMpmStencilNodes;
    cook::XpbdCookInput xpbd;
    xpbd.positions = {{-0.025f, -1.0f, -1.0f}, {-0.025f, 1.0f, -1.0f},
                      {-0.025f, 0.0f, 1.0f}};
    xpbd.surfaces = {{{0u, 1u, 2u}, 0.0f, friction}};
    if (surfaces == 2u) {
        xpbd.positions[2].y = 1.0f;
        xpbd.positions.push_back({-0.025f, -1.0f, 1.0f});
        xpbd.surfaces.push_back({{0u, 2u, 3u}, 0.0f, friction});
    }
    xpbd.inv_mass.assign(xpbd.positions.size(), surface_mass > 0.0f ?
        static_cast<float>(xpbd.positions.size()) / surface_mass : 0.0f);
    xpbd.velocities.assign(xpbd.positions.size(), Vec3{});
    xpbd.solver_iterations = 1u;
    nk::Model model;
    cook::CookMpmXpbd(model, envs, mpm, xpbd);
    model.particles.mpm_body_friction = friction;
    return model;
}

nk::Pipeline::SolverConfig SurfaceConfig(uint16_t iterations = 1024u) {
    nk::Pipeline::SolverConfig config;
    config.dt = 0.001f;
    config.gravity[2] = 0.0f;
    config.vel_iters = iterations;
    config.pos_iters = 0u;
    return config;
}

}  // namespace

TEST(NkMpmXpbdCoResidence, InterpolatedEndpointPreservesRigidMotionAndImpulseWork) {
    const uint32_t indices[3] = {7u, 13u, 29u};
    const Vec3 vertices[3] = {{0.2f, -0.8f, -0.5f}, {0.4f, 0.9f, -0.3f}, {-0.1f, 0.2f, 1.1f}};
    const Vec3 weights{0.2f, 0.3f, 0.5f};
    const Vec3 point = vertices[0] * weights.x + vertices[1] * weights.y +
                       vertices[2] * weights.z + Vec3{0.04f, -0.02f, 0.03f};
    nk::PointEndpointTerm terms[3];
    ASSERT_TRUE(nk::BuildTrianglePointEndpoint(indices, vertices, weights, point, terms));
    const Vec3 translation{0.3f, -0.6f, 0.2f}, omega{-0.4f, 0.2f, 0.9f}, impulse{0.6f, -0.8f, 0.4f};
    Vec3 velocity{}, total_impulse{}, total_moment{};
    double work = 0.0;
    for (uint32_t i = 0u; i < 3u; ++i) {
        const Vec3 v = translation + omega.Cross(vertices[i]);
        const Vec3 j = terms[i].TransposeMultiply(impulse);
        velocity += terms[i].Multiply(v);
        total_impulse += j;
        total_moment += vertices[i].Cross(j);
        work += v.Dot(j);
    }
    EXPECT_LT((velocity - translation - omega.Cross(point)).Length(), 2.0e-6f);
    EXPECT_LT((total_impulse - impulse).Length(), 2.0e-6f);
    EXPECT_LT((total_moment - point.Cross(impulse)).Length(), 2.0e-6f);
    EXPECT_NEAR(work, velocity.Dot(impulse), 2.0e-6);
    const Vec3 singular[3] = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
    EXPECT_FALSE(nk::BuildTrianglePointEndpoint(indices, singular, weights, point, terms));
}

TEST(NkMpmXpbdCoResidence, ContactBlockSatisfiesNormalAndAnisotropicFriction) {
    // Unit inverse mass with r=(0.5,0.7,-0.3) and inverse inertia diag(2,3,4).
    const nuka::math::SymmetricMat3 response{3.23f, 2.18f, 2.73f, -1.4f, 0.45f, 0.42f};
    for (float first : {0.0f, 0.15f, 0.6f}) {
        for (float second : {0.0f, 0.3f, 1.1f}) {
            for (const Vec3 incoming : {Vec3{-1.0f, 1.1f, -0.7f}, Vec3{0.4f, -0.2f, 0.1f}}) {
                Vec3 impulse{};
                for (uint32_t iteration = 0u; iteration < 1024u; ++iteration) {
                    const Vec3 velocity = incoming + response.Multiply(impulse);
                    impulse = nuka::constraint::ProjectedCoulombStep(response, velocity * -1.0f,
                                                                    impulse, first, second);
                }
                const Vec3 velocity = incoming + response.Multiply(impulse);
                EXPECT_GE(impulse.x, 0.0f);
                EXPECT_GE(velocity.x, -3.0e-6f);
                EXPECT_LT(std::abs(velocity.x * impulse.x), 3.0e-6f);
                const float a = first > 0.0f ? impulse.y / first : 0.0f;
                const float b = second > 0.0f ? impulse.z / second : 0.0f;
                EXPECT_LE(std::hypot(a, b), impulse.x + 3.0e-6f);
                if (first == 0.0f) EXPECT_EQ(impulse.y, 0.0f);
                if (second == 0.0f) EXPECT_EQ(impulse.z, 0.0f);
                const float scaled_speed = std::hypot(first * velocity.y, second * velocity.z);
                if (scaled_speed > 3.0e-6f) {
                    EXPECT_NEAR(impulse.y, -impulse.x * first * first * velocity.y / scaled_speed, 3.0e-6f);
                    EXPECT_NEAR(impulse.z, -impulse.x * second * second * velocity.z / scaled_speed, 3.0e-6f);
                }
                EXPECT_LE(impulse.y * velocity.y + impulse.z * velocity.z, 3.0e-6f);
                EXPECT_LE(impulse.Dot(incoming) + 0.5f * impulse.Dot(response.Multiply(impulse)), 3.0e-6f);
            }
        }
    }
}

TEST(NkMpmXpbdCoResidence, SurfaceContactFiniteMassFrictionAndMomentum) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr uint32_t envs = 2u;
    constexpr uint16_t iterations = 32768u;
    for (float mass : {0.0f, 0.1f, 1.0f, 10.0f, 1000.0f}) {
        for (float friction : {0.0f, 0.2f, 0.6f}) {
            for (uint32_t surfaces : {1u, 2u}) {
                SCOPED_TRACE(::testing::Message() << "mass=" << mass << " friction=" << friction << " surfaces=" << surfaces);
                nk::World world(SurfaceImpactModel(mass, friction, surfaces, envs), envs,
                                backend.dev, backend.backend, SurfaceConfig(iterations));
                ASSERT_TRUE(world.Ready()) << world.CreationError();
                ASSERT_NE(world.FieldPtr(nk::FieldId::ContactForce), nullptr);
                const auto initial = ReadSurfaceField<Vec3>(world, nk::FieldId::ParticlePos);
                ASSERT_TRUE(world.Step().AllOk());
                const auto& cap = world.GetModel().capacities;
                const auto positions = ReadSurfaceField<Vec3>(world, nk::FieldId::ParticlePos);
                const auto velocity = ReadSurfaceField<Vec3>(world, nk::FieldId::ParticleVel);
                const auto affine = ReadSurfaceField<float>(world, nk::FieldId::ParticleC);
                const auto grid = ReadSurfaceField<Vec3>(world, nk::FieldId::GridVelocity);
                const auto grid_mass = ReadSurfaceField<float>(world, nk::FieldId::GridMass);
                const auto ranges = ReadSurfaceField<nk::PointEndpointRange>(world, nk::FieldId::PointEndpointRanges);
                const auto terms = ReadSurfaceField<nk::PointEndpointTerm>(world, nk::FieldId::PointEndpointTerms);
                const auto rows = ReadSurfaceField<nk::NkRow>(world, nk::FieldId::Urows);
                const auto lambda = ReadSurfaceField<float>(world, nk::FieldId::Lambda);
                const auto side_kind = ReadSurfaceField<uint32_t>(world, nk::FieldId::ContactSideBKind);
                const auto retained = ReadSurfaceField<uint32_t>(world, nk::FieldId::GridContactRetained);
                for (uint32_t flag : ReadSurfaceField<uint32_t>(world, nk::FieldId::EnvStatus)) EXPECT_EQ(flag, 0u);
                for (uint64_t overflow : ReadSurfaceField<uint64_t>(world, nk::FieldId::GridContactOverflow)) EXPECT_EQ(overflow, 0u);
                for (uint32_t env = 0u; env < envs; ++env) {
                    const uint32_t first = env * cap.particles_per_env;
                    std::vector<Vec3> vertex_impulse(cap.particles_per_env);
                    double residual = 0.0, normal_residual = 0.0, tangent_residual = 0.0;
                    uint32_t contacts = 0u;
                    for (uint32_t r = env * cap.max_rows_per_env; r < (env + 1u) * cap.max_rows_per_env; ++r) {
                        const auto& row = rows[r];
                        if (!(row.flags & nk::nk_row_flags::kActive)) continue;
                        ASSERT_EQ(row.a.kind, nk::kNkSideGrid);
                        ASSERT_EQ(row.b.kind, nk::kNkSidePointEndpoint);
                        ASSERT_GE(row.b.index, env * cap.point_endpoints_per_env);
                        ASSERT_LT(row.b.index, (env + 1u) * cap.point_endpoints_per_env);
                        const auto range = ranges[row.b.index];
                        ASSERT_EQ(range.count, 3u);
                        Vec3 point_velocity{};
                        for (uint32_t i = 0u; i < range.count; ++i) {
                            const auto& term = terms[range.first + i];
                            ASSERT_EQ(term.kind, nk::kNkSideParticle);
                            ASSERT_GT(term.index, first);
                            ASSERT_LT(term.index, first + cap.particles_per_env);
                            vertex_impulse[term.index - first] += term.TransposeMultiply(row.b.jlin) * lambda[r];
                            point_velocity += term.Multiply(velocity[term.index]);
                        }
                        if (!(row.flags & nk::nk_row_flags::kContactNormal)) continue;
                        const double speed = row.a.jlin.Dot(grid[row.a.index]) + row.b.jlin.Dot(point_velocity);
                        const double normal_error = lambda[r] > 1.0e-8f ? std::abs(speed) : std::max(-speed, 0.0);
                        normal_residual = std::max(normal_residual, normal_error);
                        Vec3 tangential_velocity{}, tangential_impulse{};
                        for (uint32_t tangent = 1u; tangent <= 2u; ++tangent) {
                            const uint32_t at = r + tangent * row.group_normal_count;
                            const auto& tr = rows[at];
                            const float value = tr.a.jlin.Dot(grid[row.a.index]) + tr.b.jlin.Dot(point_velocity);
                            if (tangent == 1u) {
                                tangential_velocity.x = value;
                                tangential_impulse.x = lambda[at];
                            } else {
                                tangential_velocity.y = value;
                                tangential_impulse.y = lambda[at];
                            }
                        }
                        EXPECT_LE(tangential_impulse.Length(), friction * lambda[r] + 2.0e-7f);
                        EXPECT_LE(tangential_impulse.Dot(tangential_velocity), 2.0e-7f);
                        if (friction > 0.0f && lambda[r] > 1.0e-8f) {
                            const float sliding_speed = tangential_velocity.Length();
                            const float boundary = friction * lambda[r];
                            if (tangential_impulse.Length() < boundary - 2.0e-7f)
                                tangent_residual = std::max(tangent_residual, double(sliding_speed));
                            else if (sliding_speed > 1.0e-5f)
                                EXPECT_LT((tangential_impulse + tangential_velocity * (boundary / sliding_speed)).Length(), 2.0e-6f);
                        }
                        if (env == 0u && (normal_error > 3.0e-5 ||
                            (friction > 0.0f && lambda[r] > 1.0e-8f &&
                             tangential_impulse.Length() < friction * lambda[r] - 2.0e-7f &&
                             tangential_velocity.Length() > 3.0e-5f)))
                            std::printf("[surface-row] iterations=%u mass=%g mu=%g surfaces=%u row=%u ln=%.9g vn=%.9g lt=(%.9g,%.9g) vt=(%.9g,%.9g) margin=%.9g\n",
                                iterations, mass, friction, surfaces, r, lambda[r], speed,
                                tangential_impulse.x, tangential_impulse.y,
                                tangential_velocity.x, tangential_velocity.y,
                                friction * lambda[r] - tangential_impulse.Length());
                        ++contacts;
                    }
                    EXPECT_GT(contacts, 0u);
                    EXPECT_EQ(contacts, retained[env]);
                    EXPECT_EQ(std::count(side_kind.begin() + env * cap.max_contacts_per_env,
                        side_kind.begin() + (env + 1u) * cap.max_contacts_per_env, nk::kNkSidePointEndpoint), contacts);
                    Vec3 momentum{}, moment{}, transferred{}, reaction_moment{};
                    double energy = 0.0;
                    for (uint32_t local = 0u; local < cap.mpm_grid_nodes_per_env; ++local) {
                        const uint32_t g = env * cap.mpm_grid_nodes_per_env + local;
                        const Vec3 point{-0.3f + float(local % 7u) * 0.1f,
                            -0.4f + float((local / 7u) % 9u) * 0.1f, -0.4f + float(local / 63u) * 0.1f};
                        momentum += grid[g] * grid_mass[g];
                        moment += point.Cross(grid[g] * grid_mass[g]);
                        energy += 0.5 * grid_mass[g] * grid[g].LengthSq();
                    }
                    for (uint32_t local = 1u; local < cap.particles_per_env; ++local) {
                        const uint32_t p = first + local;
                        const Vec3 reaction = vertex_impulse[local];
                        transferred += reaction;
                        reaction_moment += initial[p].Cross(reaction);
                        if (mass == 0.0f) {
                            EXPECT_EQ(std::memcmp(&positions[p], &initial[p], sizeof(Vec3)), 0);
                            EXPECT_EQ(velocity[p].LengthSq(), 0.0f);
                            continue;
                        }
                        const float vertex_mass = mass / (cap.particles_per_env - 1u);
                        EXPECT_LT((velocity[p] * vertex_mass - reaction).Length(), 3.0e-5f);
                        energy += 0.5 * vertex_mass * velocity[p].LengthSq();
                    }
                    const Vec3 incoming{-1.0f, 0.4f, 0.3f};
                    EXPECT_LT((momentum + transferred - incoming).Length(), 3.0e-5f);
                    EXPECT_LT((moment + reaction_moment - initial[first].Cross(incoming)).Length(), 3.0e-5f);
                    EXPECT_LE(energy, 0.5 * incoming.LengthSq() + 3.0e-6);
                    EXPECT_GT(transferred.Length(), 0.001f);
                    Vec3 final_momentum = velocity[first], final_moment = positions[first].Cross(velocity[first]);
                    const float* c = affine.data() + first * 9u;
                    final_moment += Vec3{c[7] - c[5], c[2] - c[6], c[3] - c[1]} * (0.25f * 0.1f * 0.1f);
                    for (uint32_t local = 1u; local < cap.particles_per_env; ++local) {
                        const uint32_t p = first + local;
                        const Vec3 body_momentum = mass == 0.0f ? vertex_impulse[local] :
                            velocity[p] * (mass / (cap.particles_per_env - 1u));
                        final_momentum += body_momentum;
                        final_moment += positions[p].Cross(body_momentum);
                    }
                    EXPECT_LT((final_momentum - incoming).Length(), 3.0e-5f);
                    EXPECT_LT((final_moment - initial[first].Cross(incoming)).Length(), 3.0e-5f);
                    residual = std::max(normal_residual, tangent_residual);
                    std::printf("[surface-impact] iterations=%u mass=%g mu=%g surfaces=%u env=%u contacts=%u residual=%.9e normal=%.9e tangent=%.9e momentum=%.9e angular=%.9e\n",
                        iterations, mass, friction, surfaces, env, contacts, residual, normal_residual, tangent_residual,
                        (final_momentum - incoming).Length(), (final_moment - initial[first].Cross(incoming)).Length());
                    EXPECT_LT(residual, 3.0e-5);
                }
            }
        }
    }
}

TEST(NkMpmXpbdCoResidence, SurfaceContactGraphSubstepsAndMaskedReset) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr uint32_t envs = 3u;
    auto config = SurfaceConfig();
    config.dt = 0.0004f;
    config.substeps = 2u;
    nk::World eager(SurfaceImpactModel(1.0f, 0.6f, 2u, envs), envs, backend.dev, backend.backend, config);
    nk::World graph(SurfaceImpactModel(1.0f, 0.6f, 2u, envs), envs, backend.dev, backend.backend, config);
    config.dt *= 0.5f;
    config.substeps = 1u;
    nk::World reference(SurfaceImpactModel(1.0f, 0.6f, 2u, envs), envs, backend.dev, backend.backend, config);
    ASSERT_TRUE(eager.Ready()) << eager.CreationError();
    ASSERT_TRUE(graph.Ready()) << graph.CreationError();
    ASSERT_TRUE(reference.Ready()) << reference.CreationError();
    for (auto* world : {&eager, &graph, &reference}) ASSERT_NE(world->FieldPtr(nk::FieldId::ContactForce), nullptr);
    ASSERT_EQ(graph.SetExecutionMode(nk::World::ExecutionMode::Graph), nphi::Status::Ok);
    const auto* endpoint_address = graph.DataViewRef().point_endpoint_terms;
    for (uint32_t step = 0u; step < 12u; ++step) {
        ASSERT_TRUE(eager.Step().AllOk());
        ASSERT_EQ(graph.StepConfigured(), nphi::Status::Ok);
        ASSERT_TRUE(reference.Step().AllOk());
        ASSERT_TRUE(reference.Step().AllOk());
        for (auto field : {nk::FieldId::ParticlePos, nk::FieldId::ParticleVel, nk::FieldId::ParticleF,
                           nk::FieldId::ParticleC, nk::FieldId::PointEndpointRanges, nk::FieldId::PointEndpointTerms,
                           nk::FieldId::ContactForce, nk::FieldId::ContactSideBKind, nk::FieldId::ContactSideBIndex}) {
            const auto expected = ReadSurfaceField<uint8_t>(reference, field);
            EXPECT_EQ(ReadSurfaceField<uint8_t>(eager, field), expected) << uint32_t(field);
            EXPECT_EQ(ReadSurfaceField<uint8_t>(graph, field), expected) << uint32_t(field);
        }
        if (step != 5u) continue;
        const auto before = ReadSurfaceField<Vec3>(graph, nk::FieldId::ParticlePos);
        for (auto* world : {&eager, &graph, &reference}) ASSERT_EQ(world->Reset({1u}), nphi::Status::Ok);
        const auto after = ReadSurfaceField<Vec3>(graph, nk::FieldId::ParticlePos);
        const auto& cap = graph.GetModel().capacities;
        EXPECT_EQ(std::memcmp(before.data(), after.data(), cap.particles_per_env * sizeof(Vec3)), 0);
        EXPECT_EQ(std::memcmp(before.data() + 2u * cap.particles_per_env, after.data() + 2u * cap.particles_per_env,
                              cap.particles_per_env * sizeof(Vec3)), 0);
        const auto ranges = ReadSurfaceField<nk::PointEndpointRange>(graph, nk::FieldId::PointEndpointRanges);
        for (uint32_t i = cap.point_endpoints_per_env; i < 2u * cap.point_endpoints_per_env; ++i)
            EXPECT_EQ(ranges[i].count, 0u);
    }
    EXPECT_EQ(graph.DataViewRef().point_endpoint_terms, endpoint_address);
    EXPECT_GT(graph.GraphReplays(), 0u);
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

constexpr float kDx = 0.02f;
constexpr float kColHalf = 0.03f;    // granular column half footprint.
constexpr float kColTop  = 0.06f;    // granular column height.
constexpr float kClothSpacing = 0.03f;
constexpr float kClothZ = 0.30f;     // cloth membrane placed well above the column.

// A compact dry-sand column (MLS-MPM, model_kind 4) resting on the static floor.
cook::MpmCookInput BuildGranularColumn() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    for (float x = -kColHalf; x <= kColHalf + 1e-5f; x += pdx)
        for (float y = -kColHalf; y <= kColHalf + 1e-5f; y += pdx)
            for (float z = pdx; z <= kColTop + 1e-5f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    const float density = 1600.0f, vol0 = pdx * pdx * pdx;
    in.velocities.assign(n, Vec3::Zero());
    in.inv_mass.assign(n, 1.0f / (density * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = 3.0e5f; in.material.poisson = 0.3f;
    in.material.density = density; in.material.model_kind = 4.0f;
    in.material.dp_friction = 35.0f;
    in.grid_origin = Vec3{-kColHalf - 4.0f * kDx, -kColHalf - 4.0f * kDx, -3.0f * kDx};
    in.grid_dims[0] = static_cast<uint32_t>((2.0f * (kColHalf + 4.0f * kDx)) / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((kColTop + 12.0f * kDx) / kDx) + 4u;
    in.dx = kDx; in.substeps = 12u;
    in.floor_normal = Vec3{0, 0, 1}; in.floor_d = 0.0f; in.floor_friction = 0.7f;
    return in;
}

// A perimeter-pinned cloth membrane (XPBD), so the corners are anchored and the
// interior sags -- a pinned particle would move only if the MPM transfer wrongly
// overwrote it, so the anchors double as the no-cross-corruption probe.
cook::XpbdCookInput BuildPinnedMembrane(uint32_t n) {
    ns::MediaRecord m;
    m.kind = ns::MediaRecord::Kind::Cloth;
    m.method = ns::MediaRecord::Method::Xpbd;
    m.cloth_grid.nx = n; m.cloth_grid.ny = n;
    m.cloth_grid.spacing = kClothSpacing;
    m.cloth_grid.origin = Vec3{0.0f, 0.0f, kClothZ};
    m.cloth_grid.free = false;    // pin the perimeter (interior sags).
    m.xpbd.particle_mass = 0.01f;
    m.xpbd.iters = 8u;
    m.xpbd.distance_alpha = 1.0e-7f;
    m.xpbd.bend_alpha = 1.0e-4f;
    m.xpbd.aero_drag_normal = 0.6f;
    m.xpbd.aero_drag_tangent = 0.04f;
    m.xpbd.aero_drag_max_dv = 0.5f;
    return cook::BuildClothXpbdInput(m);
}

}  // namespace

// Assertion 1: the one-medium co-residence cooks == the single-medium cooks.
TEST(NkMpmXpbdCoResidence, SupersetMatchesSingleSystemCook) {
    const cook::MpmCookInput mpm = BuildGranularColumn();
    const cook::XpbdCookInput cloth = BuildPinnedMembrane(5u);

    // MPM-only: CookMpmXpbd(mpm, EMPTY) == CookMpmParticles(mpm).
    {
        nk::Model a, b;
        cook::CookMpmXpbd(a, 1u, mpm, cook::XpbdCookInput{});
        cook::CookMpmParticles(b, 1u, mpm);
        EXPECT_EQ(a.particles.mode, nk::Model::ParticleMode::Mpm);
        EXPECT_EQ(a.capacities.particles_per_env, b.capacities.particles_per_env);
        EXPECT_EQ(a.capacities.mpm_grid_nodes_per_env,
                  b.capacities.mpm_grid_nodes_per_env);
        ASSERT_EQ(a.particles.initial_pos.size(), b.particles.initial_pos.size());
        EXPECT_EQ(std::memcmp(a.particles.initial_pos.data(),
                              b.particles.initial_pos.data(),
                              a.particles.initial_pos.size() * sizeof(Vec3)), 0);
        ASSERT_EQ(a.particles.initial_F.size(), b.particles.initial_F.size());
        EXPECT_EQ(std::memcmp(a.particles.initial_F.data(),
                              b.particles.initial_F.data(),
                              a.particles.initial_F.size() * sizeof(float)), 0);
    }
    // XPBD-only: CookMpmXpbd(EMPTY, cloth) == CookXpbdParticles(cloth).
    {
        nk::Model a, b;
        cook::CookMpmXpbd(a, 1u, cook::MpmCookInput{}, cloth);
        cook::CookXpbdParticles(b, 1u, cloth);
        EXPECT_EQ(a.particles.mode, nk::Model::ParticleMode::Xpbd);
        EXPECT_EQ(a.capacities.particles_per_env, b.capacities.particles_per_env);
        EXPECT_EQ(a.capacities.dist_cons_per_env, b.capacities.dist_cons_per_env);
        ASSERT_EQ(a.particles.dist_a.size(), b.particles.dist_a.size());
        EXPECT_EQ(std::memcmp(a.particles.dist_a.data(), b.particles.dist_a.data(),
                              a.particles.dist_a.size() * sizeof(uint32_t)), 0);
    }
}

// Assertion 2: the co-resident cook lays out [mpm | xpbd] with the split + rebased
// constraint indices.
TEST(NkMpmXpbdCoResidence, CoResidentCookLayout) {
    const cook::MpmCookInput mpm = BuildGranularColumn();
    const cook::XpbdCookInput cloth = BuildPinnedMembrane(5u);
    const uint32_t n_mpm = static_cast<uint32_t>(mpm.positions.size());
    const uint32_t n_xpbd = static_cast<uint32_t>(cloth.positions.size());

    nk::Model m;
    cook::CookMpmXpbd(m, 1u, mpm, cloth);
    const nk::Model::ModelParticles& mp = m.particles;
    EXPECT_EQ(mp.mode, nk::Model::ParticleMode::MpmXpbd);
    EXPECT_EQ(mp.n_mpm_particles, n_mpm);
    EXPECT_EQ(m.capacities.particles_per_env, n_mpm + n_xpbd);
    // The MPM slice keeps the sand positions in [0, n_mpm); the cloth positions
    // land in [n_mpm, P).
    ASSERT_EQ(mp.initial_pos.size(), n_mpm + n_xpbd);
    for (uint32_t i = 0; i < n_mpm; ++i)
        EXPECT_EQ(std::memcmp(&mp.initial_pos[i], &mpm.positions[i], sizeof(Vec3)), 0);
    for (uint32_t i = 0; i < n_xpbd; ++i)
        EXPECT_EQ(std::memcmp(&mp.initial_pos[n_mpm + i], &cloth.positions[i],
                              sizeof(Vec3)), 0);
    // Every XPBD distance constraint references the XPBD slice [n_mpm, P).
    ASSERT_FALSE(mp.dist_a.empty());
    for (uint32_t v : mp.dist_a) { EXPECT_GE(v, n_mpm); EXPECT_LT(v, n_mpm + n_xpbd); }
    for (uint32_t v : mp.dist_b) { EXPECT_GE(v, n_mpm); EXPECT_LT(v, n_mpm + n_xpbd); }
    // The MPM continuum fields stay sized to the MPM slice.
    EXPECT_EQ(mp.initial_vol0.size(), n_mpm);
    EXPECT_EQ(mp.initial_material_id.size(), n_mpm);
}

// Assertion 3/4: the two media co-step in ONE world (env_count = 2) without
// corrupting each other, and the forward is D1.
TEST(NkMpmXpbdCoResidence, CoStepNoCrossCorruptionAndD1) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const cook::MpmCookInput mpm = BuildGranularColumn();
    const cook::XpbdCookInput cloth = BuildPinnedMembrane(5u);
    const uint32_t n_mpm = static_cast<uint32_t>(mpm.positions.size());
    const uint32_t n_xpbd = static_cast<uint32_t>(cloth.positions.size());
    constexpr uint32_t E = 2u;   // catches per-env sub-slice striding bugs.

    // The pinned cloth corners: perimeter particles (BuildClothXpbdInput pins the
    // whole perimeter of the n x n lattice, laid out row-major).
    const uint32_t nx = 5u;

    auto run = [&](std::vector<Vec3>* out) {
        nk::Model m;
        m.capacities.env_count = E;
        cook::CookMpmXpbd(m, E, mpm, cloth);
        EXPECT_EQ(m.particles.mode, nk::Model::ParticleMode::MpmXpbd);
        nk::World w(std::move(m), E, b.dev, b.backend, Cfg());
        EXPECT_TRUE(w.Ready());
        const uint32_t P = w.GetModel().capacities.particles_per_env;
        for (uint32_t s = 0; s < 240u; ++s) w.Step();
        std::vector<Vec3> pos(static_cast<size_t>(P) * E);
        w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                  pos.size() * sizeof(Vec3));
        *out = pos;
        return P;
    };

    std::vector<Vec3> a, bpos;
    const uint32_t P = run(&a);
    run(&bpos);
    ASSERT_EQ(P, n_mpm + n_xpbd);

    // Everything finite; no grid escape.
    for (const Vec3& p : a)
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));

    for (uint32_t e = 0; e < E; ++e) {
        const Vec3* env = a.data() + static_cast<size_t>(e) * P;
        // Granular slice [0, n_mpm): settled on the floor (all above the floor plane,
        // top not risen -- the transfer ran, gravity pulled it down / compacted).
        float g_top0 = -1e30f, g_top = -1e30f, g_minz = 1e30f;
        for (uint32_t i = 0; i < n_mpm; ++i) {
            g_top0 = std::max(g_top0, mpm.positions[i].z);
            g_top = std::max(g_top, env[i].z);
            g_minz = std::min(g_minz, env[i].z);
        }
        EXPECT_GT(g_minz, -kDx) << "granular fell through the floor (env " << e << ")";
        EXPECT_LT(g_top, g_top0 + 3.0f * kDx)
            << "granular column blew up instead of settling (env " << e << ")";
        // Cloth slice [n_mpm, P): the pinned corners held at the initial z (the MPM
        // transfer did NOT overwrite them), while the interior sagged under gravity.
        auto corner = [&](uint32_t ix, uint32_t iy) {
            return n_mpm + iy * nx + ix;
        };
        const uint32_t corners[4] = {corner(0, 0), corner(nx - 1, 0),
                                     corner(0, nx - 1), corner(nx - 1, nx - 1)};
        for (uint32_t c : corners) {
            EXPECT_NEAR(env[c].z, kClothZ, 1.0e-3f)
                << "a pinned cloth corner moved (cross-corruption) env " << e;
        }
        // The interior sags strictly below the pinned corners under gravity (a fully
        // inert cloth would stay exactly at kClothZ). The stiff (alpha 1e-7) small
        // membrane bows only microns, so the margin is small but deterministic (D1).
        const uint32_t center = n_mpm + (nx / 2u) * nx + (nx / 2u);
        EXPECT_LT(env[center].z, kClothZ - 1.0e-5f)
            << "the cloth interior did not sag -- XPBD inert? env " << e
            << " center_z " << env[center].z;
        EXPECT_GT(env[center].z, kClothZ - 0.10f)
            << "the cloth interior collapsed -- overwritten by G2P? env " << e;
    }

    // D1: two-run byte-exact over the FULL [mpm | xpbd] union across both envs.
    ASSERT_EQ(a.size(), bpos.size());
    EXPECT_EQ(std::memcmp(a.data(), bpos.data(), a.size() * sizeof(Vec3)), 0)
        << "MpmXpbd co-resident forward not two-run byte-identical (D1)";

    std::fprintf(stderr, "[mpm-xpbd costep] n_mpm=%u n_xpbd=%u P=%u E=%u OK\n",
                 n_mpm, n_xpbd, P, E);
}
