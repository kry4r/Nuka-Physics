// Exercise cooking, controlled robot/cloth/fluid contact, readout and reset on one World pipeline.
// A separated-media control measures the robot's response to both media.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "../../tools/perf/robot_cloth_fluid_scene.hpp"

#include "collision/shape_kind.hpp"
#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
using nuka::math::Transform;
using nuka::math::Vec3;

using namespace nuka::perf::fixture;

std::filesystem::path Go2ScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_stand.usda";
}

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

using PipelineState = std::vector<std::vector<uint8_t>>;

PipelineState ReadPipelineState(nk::World& world) {
    PipelineState state;
    for (auto field : {nk::FieldId::BasePose, nk::FieldId::Q, nk::FieldId::Qdot,
                      nk::FieldId::LinkVelocity, nk::FieldId::BodyPose,
                      nk::FieldId::BodyLinearVelocity, nk::FieldId::BodyAngularVelocity,
                      nk::FieldId::ParticlePos,
                      nk::FieldId::ParticlePrevPos, nk::FieldId::ParticleVel}) {
        const auto& segments = world.GetData().Segments();
        const auto segment = std::find_if(segments.begin(), segments.end(),
            [field](const auto& value) { return value.field == field; });
        if (segment == segments.end()) {
            ADD_FAILURE() << "missing pipeline state field " << static_cast<uint32_t>(field);
            return {};
        }
        state.emplace_back(segment->bytes);
        EXPECT_TRUE(world.GetData().DownloadField(field, state.back().data(), state.back().size()));
    }
    return state;
}

TEST(RobotClothFluidCoResident, GraphControlsReadoutAndResetMatchEager) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const auto fixture = Prepare(Go2ScenePath(), backend.dev, backend.backend, Cfg());
    constexpr uint32_t envs = 3u;
    nk::World eager(CookPrepared(fixture, envs), envs, backend.dev, backend.backend, Cfg());
    nk::World graph(CookPrepared(fixture, envs), envs, backend.dev, backend.backend, Cfg());
    ASSERT_TRUE(eager.Ready()) << eager.CreationError();
    ASSERT_TRUE(graph.Ready()) << graph.CreationError();
    const auto initial = ReadPipelineState(graph);
    const auto* address = graph.DataViewRef().particle_pos;
    ASSERT_EQ(graph.SetExecutionMode(nk::World::ExecutionMode::Graph), nphi::Status::Ok)
        << graph.LastExecutionError().message;
    EXPECT_EQ(ReadPipelineState(graph), initial);
    EXPECT_EQ(graph.CaptureAttempts(), 1u);
    std::vector<float> targets(graph.GetModel().capacities.links_per_env * envs);
    ASSERT_TRUE(graph.GetData().DownloadField(nk::FieldId::DriveTarget, targets.data(),
                                              targets.size() * sizeof(float)));
    const auto rest = targets;
    std::vector<uint32_t> flags(envs);
    for (uint32_t step = 0u; step < 16u; ++step) {
        for (size_t i = 0; i < targets.size(); ++i)
            targets[i] = rest[i] + 0.004f * std::sin(static_cast<float>(step + i));
        if (step == 4u) {
            ASSERT_NE(eager.FieldPtr(nk::FieldId::ContactForce), nullptr);
            ASSERT_NE(graph.FieldPtr(nk::FieldId::ContactForce), nullptr);
            EXPECT_FALSE(graph.GraphReady());
        }
        if (step == 8u) {
            ASSERT_EQ(eager.Reset({1u, 1u}), nphi::Status::Ok);
            ASSERT_EQ(graph.Reset({1u, 1u}), nphi::Status::Ok);
        }
        for (auto* world : {&eager, &graph}) {
            ASSERT_TRUE(world->GetData().UploadField(nk::FieldId::DriveTarget, targets.data(),
                                                    targets.size() * sizeof(float)));
            ASSERT_EQ(world->StepConfigured(), nphi::Status::Ok) << world->LastExecutionError().message;
            ASSERT_EQ(world->Synchronize(), nphi::Status::Ok);
            ASSERT_TRUE(world->GetData().DownloadField(nk::FieldId::EnvStatus, flags.data(),
                                                      flags.size() * sizeof(uint32_t)));
            EXPECT_EQ(flags, std::vector<uint32_t>(envs));
        }
        EXPECT_EQ(ReadPipelineState(eager), ReadPipelineState(graph));
        std::vector<uint8_t> eager_state, graph_state;
        ASSERT_TRUE(eager.GetData().DownloadPersistent(&eager_state));
        ASSERT_TRUE(graph.GetData().DownloadPersistent(&graph_state));
        EXPECT_EQ(eager_state, graph_state);
    }
    EXPECT_EQ(graph.CaptureAttempts(), 2u);
    EXPECT_EQ(graph.GraphReplays(), 16u);
    ASSERT_EQ(graph.Reset(), nphi::Status::Ok);
    EXPECT_EQ(graph.DataViewRef().particle_pos, address);
    EXPECT_EQ(ReadPipelineState(graph), initial);
    ASSERT_EQ(graph.StepConfigured(), nphi::Status::Ok);
    EXPECT_EQ(graph.CaptureAttempts(), 2u);
}

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(), P * sizeof(Vec3)));
}

// Min particle z over the disc of `reach` about (cx,cy), restricted to a particle
// index slice [lo, hi) -- the pocket a foot presses into one medium's slice.
float MinZUnderFootInSlice(const std::vector<Vec3>& pos, float cx, float cy,
                           float reach, uint32_t lo, uint32_t hi) {
    float min_z = 1.0e9f;
    for (uint32_t i = lo; i < hi && i < pos.size(); ++i) {
        const float dx = pos[i].x - cx, dy = pos[i].y - cy;
        if (dx * dx + dy * dy <= reach * reach) min_z = std::min(min_z, pos[i].z);
    }
    return min_z;
}

// Max particle z over a slice [lo, hi) -- a medium's free-surface height.
float SurfaceMaxZInSlice(const std::vector<Vec3>& pos, uint32_t lo, uint32_t hi) {
    float hi_z = -1.0e9f;
    for (uint32_t i = lo; i < hi && i < pos.size(); ++i) hi_z = std::max(hi_z, pos[i].z);
    return hi_z;
}

}  // namespace

// A Go2 articulation + cloth + fluid co-resident in ONE world: stable co-step +
// two-way coupling to BOTH media through the general body<->particle row solver.
TEST(RobotClothFluidCoResident, Go2StanceCouplesClothAndFluidOnOnePipeline) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(Go2ScenePath()))
        GTEST_SKIP() << "go2_stand.usda not present";
    Backend b = GetBackend();

    const auto prepared = Prepare(Go2ScenePath(), b.dev, b.backend, Cfg());
    auto cook_go2 = [&](bool with_free_body = false) {
        return CookRobot(Go2ScenePath(), with_free_body);
    };
    const auto& front_centre = prepared.front_centre;
    const auto& rear_foot = prepared.rear_foot;
    const auto& link_geom_local = prepared.link_geom_local;
    const auto front_link = prepared.front_link, rear_link = prepared.rear_link;
    const float cloth_z = prepared.cloth_z, pool_floor = prepared.pool_floor;
    const uint32_t L = static_cast<uint32_t>(link_geom_local.size());

    struct Run {
        uint32_t cloth_link_rows = 0u, fluid_link_rows = 0u, two_particle_rows = 0u;
        uint32_t cloth_any_rows = 0u, fluid_any_rows = 0u;  // any body side (diag).
        float cloth_link_lambda = 0.0f, fluid_link_lambda = 0.0f;
        float cloth_min_z = 1e9f, pool_min_z = 1e9f, cloth_base_min_z = 0.0f;
        float pool_base_surface = 0.0f, pool_max_surface = -1e9f, pool_base_pocket = 0.0f;
        float front_foot_z = 0.0f, rear_foot_z = 0.0f;  // run-scene foot z (diag).
        bool finite = true;
        std::vector<float> q, qdot;
    };
    auto run_scene = [&](bool patch_present) -> Run {
        Run out;
        nk::Model m = cook_go2(true);

        cook::XpbdCookInput cloth = BuildCloth(front_centre.x, front_centre.y, cloth_z);
        cook::PbfCookInput pool = BuildPool(rear_foot.x, rear_foot.y, pool_floor);
        if (!patch_present) {
            // Sink both media far out of reach: the SAME robot, no coupling.
            for (Vec3& p : cloth.positions) p.z -= 10.0f;
            for (Vec3& p : pool.positions) p.z -= 10.0f;
            pool.grid_min.z -= 10.0f;
            pool.floor_z -= 10.0f;
        }
        cook::CookSoftFluidParticles(m, 1u, cloth, pool);
        m.particles.pp_contact_d_min = kContactDMin;
        const uint32_t n_soft = m.particles.n_soft_particles;
        const uint32_t P = m.capacities.particles_per_env;
        const uint32_t rows = m.capacities.max_rows_per_env;
        const uint32_t bodies = m.capacities.bodies_per_env;
        const auto free_body = std::find_if(m.body_init.begin(), m.body_init.end(),
            [](const auto& body) { return body.inv_mass > 0.0f; });
        EXPECT_NE(free_body, m.body_init.end());
        if (free_body == m.body_init.end()) { out.finite = false; return out; }
        const uint32_t free_index = static_cast<uint32_t>(free_body - m.body_init.begin());
        const float free_inv_mass = free_body->inv_mass;
        const Vec3 initial_com = free_body->pose.TransformPoint(free_body->inertial_frame.position);
        const Vec3 com_offset = free_body->inertial_frame.position;
        free_body->angular_velocity = {5.0f, -3.0f, 2.0f};
        const Vec3 inv_inertia = free_body->inv_inertia;
        const auto inertial_frame = free_body->inertial_frame;
        const auto principal_start = free_body->pose.rotation * inertial_frame.rotation;
        const Vec3 local_start = principal_start.Conjugate().Rotate(free_body->angular_velocity);
        const Vec3 initial_momentum = principal_start.Rotate(
            {local_start.x / inv_inertia.x, local_start.y / inv_inertia.y, local_start.z / inv_inertia.z});
        const auto config = Cfg();
        const Vec3 gravity{config.gravity[0], config.gravity[1], config.gravity[2]};
        const Vec3 applied_force{0.3f, -0.2f, 0.4f};
        const Vec3 applied_torque{0.01f, -0.02f, 0.03f};

        nk::World w(std::move(m), 1u, b.dev, b.backend, config);
        EXPECT_TRUE(w.Ready());
        if (!w.Ready()) { out.finite = false; return out; }
        nk::Data& d = w.GetData();
        const auto initial_state = ReadPipelineState(w);
        PipelineState first_steps;
        constexpr uint32_t replay_steps = 8u;
        std::vector<float> targets(L);
        EXPECT_TRUE(d.DownloadField(nk::FieldId::DriveTarget, targets.data(), targets.size() * sizeof(float)));
        EXPECT_TRUE(d.UploadField(nk::FieldId::DriveTarget, targets.data(), targets.size() * sizeof(float)));
        auto apply_wrench = [&]() {
            std::vector<Vec3> forces(bodies), torques(bodies);
            forces[free_index] = applied_force;
            torques[free_index] = applied_torque;
            EXPECT_TRUE(d.UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
            EXPECT_TRUE(d.UploadField(nk::FieldId::BodyTorque, torques.data(), torques.size() * sizeof(Vec3)));
        };
        apply_wrench();
        std::vector<nk::NkRow> urows(rows);
        std::vector<float> lambda(rows, 0.0f);
        std::vector<Vec3> p;
        for (uint32_t s = 0; s < kSettleSteps + kHoldSteps; ++s) {
            if (!w.Step().AllOk()) {
                ADD_FAILURE() << "pipeline step " << s << " failed";
                out.finite = false;
                return out;
            }
            uint32_t step_status = 0u;
            EXPECT_TRUE(d.DownloadField(nk::FieldId::EnvStatus, &step_status, sizeof(step_status)));
            if (step_status != 0u) {
                ADD_FAILURE() << "invalid environment at step " << s << ": " << step_status;
                out.finite = false;
                return out;
            }
            if (s + 1u == replay_steps) first_steps = ReadPipelineState(w);
            if (s == 0u) {
                std::vector<Vec3> predicted(P);
                EXPECT_TRUE(d.DownloadField(nk::FieldId::PbfPredictedPos, predicted.data(), P * sizeof(Vec3)));
                for (uint32_t i = 0u; i < n_soft; ++i) {
                    const Vec3 expected = cloth.positions[i] +
                        (cloth.inv_mass[i] > 0.0f ? gravity * (config.dt * config.dt) : Vec3::Zero());
                    EXPECT_NEAR(predicted[i].x, expected.x, 1.0e-6f);
                    EXPECT_NEAR(predicted[i].y, expected.y, 1.0e-6f);
                    EXPECT_NEAR(predicted[i].z, expected.z, 1.0e-6f);
                }
                std::vector<Vec3> forces(bodies), torques(bodies), velocity(bodies), omega(bodies);
                EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyForce, forces.data(), bodies * sizeof(Vec3)));
                EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyTorque, torques.data(), bodies * sizeof(Vec3)));
                EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyLinearVelocity, velocity.data(), bodies * sizeof(Vec3)));
                EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyAngularVelocity, omega.data(), bodies * sizeof(Vec3)));
                EXPECT_EQ(forces, std::vector<Vec3>(bodies));
                EXPECT_EQ(torques, std::vector<Vec3>(bodies));
                const Vec3 expected = (gravity + applied_force * free_inv_mass) * config.dt;
                EXPECT_NEAR(velocity[free_index].x, expected.x, 1.0e-6f);
                EXPECT_NEAR(velocity[free_index].y, expected.y, 1.0e-6f);
                EXPECT_NEAR(velocity[free_index].z, expected.z, 1.0e-6f);
                EXPECT_GT(omega[free_index].LengthSq(), 1.0e-7f);
            }
            if (!patch_present || s < kSettleSteps) continue;
            // Capture the pool free surface / pocket at the first hold step (the
            // settled baseline) then track its max surface over the hold window.
            DownloadParticles(w, &p);
            const float cloth_reach = 0.5f * kContactDMin + 2.0f * kClothSpacing;
            if (s == kSettleSteps) {
                out.pool_base_surface = SurfaceMaxZInSlice(p, n_soft, P);
                out.pool_base_pocket = MinZUnderFootInSlice(
                    p, rear_foot.x, rear_foot.y,
                    0.5f * kContactDMin + 2.0f * kPoolSpacing, n_soft, P);
                out.cloth_base_min_z = MinZUnderFootInSlice(
                    p, front_centre.x, front_centre.y, cloth_reach, 0u, n_soft);
                std::vector<Transform> lp(L);
                EXPECT_TRUE(d.DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform)));
                out.front_foot_z =
                    (lp[front_link] * link_geom_local[front_link]).position.z;
                out.rear_foot_z =
                    (lp[rear_link] * link_geom_local[rear_link]).position.z;
            }
            out.pool_max_surface =
                std::max(out.pool_max_surface, SurfaceMaxZInSlice(p, n_soft, P));
            EXPECT_TRUE(d.DownloadField(nk::FieldId::Urows, urows.data(),
                                        urows.size() * sizeof(nk::NkRow)));
            EXPECT_TRUE(d.DownloadField(nk::FieldId::Lambda, lambda.data(),
                                        lambda.size() * sizeof(float)));
            for (uint32_t row = 0u; row < rows; ++row) {
                const auto& rs = urows[row];
                if ((rs.flags & nk::nk_row_flags::kActive) == 0u) continue;
                const bool a_part = rs.a.kind == nk::kNkSideParticle;
                const bool b_part = rs.b.kind == nk::kNkSideParticle;
                if (!(a_part || b_part)) continue;
                if (a_part && b_part) { ++out.two_particle_rows; continue; }
                const auto& owner = a_part ? rs.b : rs.a;
                const uint32_t part_idx = a_part ? rs.a.index : rs.b.index;
                if (part_idx < n_soft) ++out.cloth_any_rows; else ++out.fluid_any_rows;
                if (owner.kind != nk::kNkSideArtic) continue;
                EXPECT_LT(owner.index, w.GetModel().capacities.articulations_per_env);
                EXPECT_LT(part_idx, P);
                const bool normal = (rs.flags & nk::nk_row_flags::kContactNormal) != 0u;
                if (part_idx < n_soft) {           // the cloth (soft) slice.
                    ++out.cloth_link_rows;
                    if (normal)
                        out.cloth_link_lambda =
                            std::max(out.cloth_link_lambda, lambda[row]);
                } else {                            // the fluid slice.
                    ++out.fluid_link_rows;
                    if (normal)
                        out.fluid_link_lambda =
                            std::max(out.fluid_link_lambda, lambda[row]);
                }
            }
        }
        out.q.assign(L, 0.0f); out.qdot.assign(L, 0.0f);
        EXPECT_TRUE(d.DownloadField(nk::FieldId::Q, out.q.data(), L * sizeof(float)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::Qdot, out.qdot.data(), L * sizeof(float)));
        for (float value : out.q) out.finite = out.finite && std::isfinite(value);
        for (float value : out.qdot) out.finite = out.finite && std::isfinite(value);
        std::vector<Vec3> free_velocity(bodies);
        std::vector<Transform> body_poses(bodies);
        EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyLinearVelocity, free_velocity.data(), bodies * sizeof(Vec3)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyPose, body_poses.data(), bodies * sizeof(Transform)));
        std::vector<Vec3> angular(bodies);
        std::vector<float> residual(bodies);
        std::vector<uint32_t> gyro_status(bodies), iterations(bodies);
        uint32_t env_status = 0u;
        EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyAngularVelocity, angular.data(), bodies * sizeof(Vec3)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyGyroResidual, residual.data(), bodies * sizeof(float)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyGyroStatus, gyro_status.data(), bodies * sizeof(uint32_t)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::BodyGyroIterations, iterations.data(), bodies * sizeof(uint32_t)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::EnvStatus, &env_status, sizeof(env_status)));
        EXPECT_EQ(gyro_status[free_index], 0u);
        EXPECT_EQ(env_status, 0u);
        std::vector<uint32_t> neighbor_attempted(P), neighbor_count(P);
        EXPECT_TRUE(d.DownloadField(nk::FieldId::GridNeighborAttempted, neighbor_attempted.data(), P * sizeof(uint32_t)));
        EXPECT_TRUE(d.DownloadField(nk::FieldId::GridNeighborCount, neighbor_count.data(), P * sizeof(uint32_t)));
        EXPECT_EQ(neighbor_count, neighbor_attempted);
        std::fprintf(stderr, "[particle-neighbors] max=%u attempted=%llu retained=%llu\n",
            *std::max_element(neighbor_attempted.begin(), neighbor_attempted.end()),
            static_cast<unsigned long long>(std::accumulate(neighbor_attempted.begin(), neighbor_attempted.end(), uint64_t{0})),
            static_cast<unsigned long long>(std::accumulate(neighbor_count.begin(), neighbor_count.end(), uint64_t{0})));
        EXPECT_LE(residual[free_index], 1.0e-6f);
        const auto principal_end = body_poses[free_index].rotation * inertial_frame.rotation;
        const Vec3 local_end = principal_end.Conjugate().Rotate(angular[free_index]);
        const Vec3 momentum = principal_end.Rotate(
            {local_end.x / inv_inertia.x, local_end.y / inv_inertia.y, local_end.z / inv_inertia.z});
        const Vec3 expected_momentum = initial_momentum + applied_torque * config.dt;
        EXPECT_LT((momentum - expected_momentum).Length(), 2.0e-4f);
        uint64_t model_bytes = 0u, data_bytes = 0u;
        w.GetModel().ComputeModelSegments(&model_bytes);
        for (const auto& segment : d.Segments()) data_bytes += segment.bytes;
        std::fprintf(stderr, "[dynamics-pipeline] gyro_residual=%g iterations=%u L_error=%g "
            "env_status=%u model_bytes=%llu data_field_bytes=%llu\n", residual[free_index],
            iterations[free_index], (momentum - expected_momentum).Length(), env_status,
            static_cast<unsigned long long>(model_bytes), static_cast<unsigned long long>(data_bytes));
        const float steps = static_cast<float>(kSettleSteps + kHoldSteps);
        const Vec3 expected_velocity = gravity * (steps * config.dt) +
            applied_force * (free_inv_mass * config.dt);
        const Vec3 expected_com = initial_com +
            gravity * (0.5f * steps * (steps + 1.0f) * config.dt * config.dt) +
            applied_force * (free_inv_mass * steps * config.dt * config.dt);
        const Vec3 actual_com = body_poses[free_index].TransformPoint(com_offset);
        EXPECT_NEAR(free_velocity[free_index].x, expected_velocity.x, 2.0e-4f);
        EXPECT_NEAR(free_velocity[free_index].y, expected_velocity.y, 2.0e-4f);
        EXPECT_NEAR(free_velocity[free_index].z, expected_velocity.z, 2.0e-4f);
        EXPECT_NEAR(actual_com.x, expected_com.x, 2.0e-3f);
        EXPECT_NEAR(actual_com.y, expected_com.y, 2.0e-3f);
        EXPECT_NEAR(actual_com.z, expected_com.z, 2.0e-3f);
        DownloadParticles(w, &p);
        for (const Vec3& q : p)
            if (!(std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z)))
                out.finite = false;
        if (patch_present) {
            out.cloth_min_z = MinZUnderFootInSlice(
                p, front_centre.x, front_centre.y,
                0.5f * kContactDMin + 2.0f * kClothSpacing, 0u, n_soft);
            out.pool_min_z = MinZUnderFootInSlice(p, rear_foot.x, rear_foot.y,
                                                  0.5f * kContactDMin + 2.0f * kPoolSpacing,
                                                  n_soft, P);
            for (const auto& selection : std::vector<std::vector<uint32_t>>{{}, {0u}}) {
                EXPECT_EQ(w.Reset(selection), nphi::Status::Ok);
                EXPECT_EQ(ReadPipelineState(w), initial_state);
                std::vector<float> restored_targets(L);
                EXPECT_TRUE(d.DownloadField(nk::FieldId::DriveTarget, restored_targets.data(),
                                            restored_targets.size() * sizeof(float)));
                EXPECT_EQ(restored_targets, targets);
                uint32_t contact_count = ~0u;
                EXPECT_TRUE(d.DownloadField(nk::FieldId::ContactCount, &contact_count, sizeof(contact_count)));
                EXPECT_EQ(contact_count, 0u);
                apply_wrench();
                for (uint32_t s = 0u; s < replay_steps; ++s) EXPECT_TRUE(w.Step().AllOk());
                EXPECT_EQ(ReadPipelineState(w), first_steps);
            }
        }
        return out;
    };

    const Run with_media = run_scene(/*patch_present=*/true);
    const Run control = run_scene(/*patch_present=*/false);

    double qdot_delta = 0.0;
    ASSERT_EQ(with_media.qdot.size(), control.qdot.size());
    for (size_t i = 0; i < with_media.qdot.size(); ++i)
        qdot_delta += std::fabs(static_cast<double>(with_media.qdot[i]) -
                                control.qdot[i]);

    const float cloth_dip = with_media.cloth_base_min_z - with_media.cloth_min_z;
    std::fprintf(stderr,
                 "[robot-coupling] cloth_rows=%u cloth_lambda=%.6f base_min_z=%.4f "
                 "cloth_min_z=%.4f (dip=%.4f, lay=%.4f)\n",
                 with_media.cloth_link_rows, with_media.cloth_link_lambda,
                 with_media.cloth_base_min_z, with_media.cloth_min_z, cloth_dip,
                 cloth_z);
    const float pool_rise = with_media.pool_max_surface - with_media.pool_base_surface;
    const float pocket_drop = with_media.pool_base_pocket - with_media.pool_min_z;
    std::fprintf(stderr,
                 "[robot-coupling] fluid_rows=%u fluid_lambda=%.6f base_surface=%.4f "
                 "max_surface=%.4f (rise=%.4f) base_pocket=%.4f pool_min_z=%.4f "
                 "(drop=%.4f)\n",
                 with_media.fluid_link_rows, with_media.fluid_link_lambda,
                 with_media.pool_base_surface, with_media.pool_max_surface, pool_rise,
                 with_media.pool_base_pocket, with_media.pool_min_z, pocket_drop);
    std::fprintf(stderr,
                 "[robot-coupling] cloth_any_rows=%u fluid_any_rows=%u "
                 "run_front_foot_z=%.4f run_rear_foot_z=%.4f two_particle_rows=%u "
                 "qdot_delta_L1=%.6f\n",
                 with_media.cloth_any_rows, with_media.fluid_any_rows,
                 with_media.front_foot_z, with_media.rear_foot_z,
                 with_media.two_particle_rows, qdot_delta);

    // STABLE: the robot + both media stayed finite over the whole co-step.
    EXPECT_TRUE(with_media.finite) << "the co-resident world produced a NaN/blowup";
    EXPECT_TRUE(control.finite) << "the bare-robot control produced a NaN/blowup";

    // TWO-WAY to CLOTH: a body<->particle row whose body side is an articulation
    // LINK against a cloth (soft) particle, with a body-side normal reaction.
    EXPECT_GT(with_media.cloth_link_rows, 0u)
        << "no robot-link<->cloth coupling rows emitted";
    EXPECT_GT(with_media.cloth_link_lambda, 0.0f)
        << "the robot link produced no body-side reaction on the cloth";
    // The foot indents the membrane below its pinned lay (a stable dimple, not the
    // membrane's negligible free sag): cloth_min_z holds below the lay through the
    // hold and the foot carries a real body-side reaction (cloth_link_lambda above).
    // The dimple is modest because the Go2 hold-PD is soft (the hanging foot floats
    // up to barely rest on the taut membrane rather than pressing through it).
    EXPECT_LT(with_media.cloth_min_z, cloth_z - 0.001f)
        << "the standing foot did not dip the cloth below its lay";

    // TWO-WAY to FLUID: the same, against a fluid-slice particle, pool displaced
    // (the submerged foot pushes fluid up at the edges OR down in the pocket).
    EXPECT_GT(with_media.fluid_link_rows, 0u)
        << "no robot-link<->fluid coupling rows emitted";
    EXPECT_GT(with_media.fluid_link_lambda, 0.0f)
        << "the robot link produced no body-side reaction on the fluid";
    EXPECT_TRUE(pool_rise > 0.003f || pocket_drop > 0.003f)
        << "the foot did not displace the fluid (no surface rise, no pocket drop)";

    // The reaction reaches the articulation: the held stance differs from the
    // no-media control (the impulse flows through the articulated dynamics).
    EXPECT_GT(qdot_delta, 1.0e-4)
        << "the robot dynamics did not respond to the coupled media";

    // Every coupling row had exactly one particle side (the general path).
    EXPECT_EQ(with_media.two_particle_rows, 0u)
        << "a body<->particle slot carried two particle sides (not the general path)";
}
