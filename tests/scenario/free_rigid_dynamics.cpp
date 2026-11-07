#include <gtest/gtest.h>

#include <cmath>
#include <array>
#include <limits>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <vector>

#include "import/usd_importer.hpp"
#include "constraint/coulomb_contact.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/readout/energy_ledger.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "phi/articulation_contract.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace scene = nuka::scene;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

struct Backend {
    phi::Device* device = phi::InitBestDevice();
    phi::Backend* backend = device ? phi::DeviceInitBackend(device, nullptr) : nullptr;
};

Backend& Device() {
    static Backend backend;
    return backend;
}

nk::Model CookFreeBody(const scene::RigidBodyRecord& body, bool with_floor, bool surface_contact = false) {
    scene::SceneIR source;
    if (with_floor) {
        scene::RigidBodyRecord floor;
        floor.name = "floor";
        floor.is_static = true;
        floor.local_transform.position = {0, 0, -0.1f};
        scene::CollisionShapeRecord shape;
        shape.body_id = source.AddRigidBody(floor);
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {2, 2, 0.1f};
        source.AddCollisionShape(shape);
    }
    scene::CollisionShapeRecord shape;
    shape.body_id = source.AddRigidBody(body);
    shape.type = scene::ShapeType::Sphere;
    shape.radius = 0.1f;
    if (surface_contact) {
        shape.type = scene::ShapeType::TriMesh;
        shape.decompose_mode = scene::DecomposeMode::Skip;
        shape.mesh_vertices = {-.1f,-.1f,-.1f, .1f,-.1f,-.1f, .1f,.1f,-.1f, -.1f,.1f,-.1f,
            -.1f,-.1f,.1f, .1f,-.1f,.1f, .1f,.1f,.1f, -.1f,.1f,.1f};
        shape.mesh_indices = {0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
            1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7};
    }
    source.AddCollisionShape(shape);
    scene::cook::CookToModelOptions options;
    options.enable_contacts = with_floor;
    if (surface_contact) return scene::cook::CookSceneToModel(source, 1, options).model;
    return scene::cook::CookToModel(source, 1, options).model;
}

nk::Pipeline::SolverConfig Config() {
    nk::Pipeline::SolverConfig config;
    config.dt = 0.002f;
    config.gravity[0] = config.gravity[1] = config.gravity[2] = 0.0f;
    config.pos_iters = 0u;
    return config;
}

template <typename T>
std::vector<T> Read(nk::World& world, nk::FieldId field, size_t count) {
    std::vector<T> values(count);
    EXPECT_TRUE(world.GetData().DownloadField(field, values.data(), count * sizeof(T)));
    return values;
}

void ExpectVectorNear(Vec3 actual, Vec3 expected, float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

Vec3 AngularMomentum(const Transform& pose, const Transform& frame, Vec3 inertia, Vec3 omega) {
    const Quat principal = pose.rotation * frame.rotation;
    const Vec3 local = principal.Conjugate().Rotate(omega);
    return principal.Rotate({inertia.x * local.x, inertia.y * local.y, inertia.z * local.z});
}

std::array<double, 7> RotationReference(Vec3 inertia, Vec3 omega, Quat principal, double duration) {
    using State = std::array<double, 7>;
    State state{omega.x, omega.y, omega.z, principal.w, principal.x, principal.y, principal.z};
    const double ix = inertia.x, iy = inertia.y, iz = inertia.z;
    auto derivative = [&](const State& s) -> State {
        return {(iy - iz) / ix * s[1] * s[2],
                (iz - ix) / iy * s[2] * s[0],
                (ix - iy) / iz * s[0] * s[1],
                -0.5 * (s[4] * s[0] + s[5] * s[1] + s[6] * s[2]),
                0.5 * (s[3] * s[0] + s[5] * s[2] - s[6] * s[1]),
                0.5 * (s[3] * s[1] + s[6] * s[0] - s[4] * s[2]),
                0.5 * (s[3] * s[2] + s[4] * s[1] - s[5] * s[0])};
    };
    auto offset = [](State a, const State& b, double h) {
        for (size_t i = 0; i < a.size(); ++i) a[i] += h * b[i];
        return a;
    };
    constexpr uint32_t steps = 40000u;
    const double dt = duration / steps;
    for (uint32_t i = 0; i < steps; ++i) {
        const auto k1 = derivative(state);
        const auto k2 = derivative(offset(state, k1, dt * 0.5));
        const auto k3 = derivative(offset(state, k2, dt * 0.5));
        const auto k4 = derivative(offset(state, k3, dt));
        for (size_t j = 0; j < state.size(); ++j)
            state[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
    return state;
}
}  // namespace

TEST(FreeRigidDynamics, MountedImuIncludesFreeFallAndCentripetalAcceleration) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (uint32_t substeps : {1u, 4u}) {
        auto config = Config();
        config.dt = 0.002f;
        config.substeps = substeps;
        config.gravity[2] = -9.81f;
        scene::RigidBodyRecord body;
        body.mass = 2.0f;
        body.inertia = {0.2f, 0.2f, 0.2f};
        body.local_transform.position.z = 3.0f;
        body.inertial_transform.position = {0.12f, -0.07f, 0.03f};
        auto model = CookFreeBody(body, false);
        model.body_init[0].angular_velocity = {0.0f, 0.0f, 3.0f};
        nk::World world(std::move(model), 3u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        nuka::sensor::StateSensorDesc imu;
        imu.mount = nuka::sensor::StateSensorMount::Body;
        imu.index = 0u;
        imu.local_offset.position = body.inertial_transform.position;
        uint32_t center_id, offset_id;
        ASSERT_EQ(world.AttachStateSensor(imu, &center_id), phi::Status::Ok);
        imu.local_offset.position.x += 0.2f;
        imu.local_offset.rotation = Quat::FromAxisAngle({1, 0, 0}, 1.57079632679f);
        ASSERT_EQ(world.AttachStateSensor(imu, &offset_id), phi::Status::Ok);
        ASSERT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), phi::Status::Ok);
        nuka::sensor::StateSensorStamp stamp;
        ASSERT_EQ(world.StateSensors().ReadStamp(center_id, 0u, &stamp), phi::Status::Ok);
        EXPECT_EQ(stamp.acquisitions, 0u);
        uint32_t completed_steps = 0u;
        for (uint32_t step = 0u; step < 40u; ++step) {
            const uint32_t batches[] = {1u, 2u, 7u, 0u};
            uint32_t submitted = ~0u;
            ASSERT_EQ(world.StepConfigured(batches[step % 4u], &submitted), phi::Status::Ok);
            ASSERT_EQ(submitted, batches[step % 4u]);
            completed_steps += submitted;
            EXPECT_EQ(world.GraphReplays(), completed_steps);
            ASSERT_EQ(world.StateSensors().ReadStamp(center_id, 0u, &stamp), phi::Status::Ok);
            EXPECT_EQ(stamp.acquisitions, completed_steps);
            std::vector<float> center(18u), offset(18u);
            ASSERT_EQ(world.StateSensors().Download(center_id, center.data(), center.size() * sizeof(float)), phi::Status::Ok);
            ASSERT_EQ(world.StateSensors().Download(offset_id, offset.data(), offset.size() * sizeof(float)), phi::Status::Ok);
            for (uint32_t env = 0u; env < 3u; ++env) {
                for (uint32_t axis = 0u; axis < 3u; ++axis) EXPECT_NEAR(center[env * 6u + axis], 0.0f, 5.0e-4f);
                EXPECT_NEAR(center[env * 6u + 5u], 3.0f, 1.0e-5f);
                EXPECT_NEAR(offset[env * 6u], -1.8f, 1.0e-3f);
                EXPECT_NEAR(offset[env * 6u + 1u], 0.0f, 5.0e-4f);
                EXPECT_NEAR(offset[env * 6u + 2u], 0.0f, 5.0e-4f);
                EXPECT_NEAR(offset[env * 6u + 4u], 3.0f, 1.0e-5f);
            }
        }
        ASSERT_EQ(world.Reset({0u}), phi::Status::Ok);
        uint32_t submitted = 0u;
        ASSERT_EQ(world.StepConfigured(5u, &submitted), phi::Status::Ok);
        ASSERT_EQ(submitted, 5u);
        for (uint32_t env = 0u; env < 3u; ++env) {
            ASSERT_EQ(world.StateSensors().ReadStamp(center_id, env, &stamp), phi::Status::Ok);
            EXPECT_EQ(stamp.acquisitions, env == 0u ? 5u : completed_steps + 5u);
        }
    }
}

TEST(FreeRigidDynamics, FreeRotationConservesMomentumAndConvergesToTheReference) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (Quat rotation : {Quat::Identity(), Quat::FromAxisAngle({1, -2, 3}, 0.9f)}) {
        scene::RigidBodyRecord body;
        body.mass = 2.0f;
        body.inertia = {0.4f, 0.7f, 0.9f};
        body.local_transform.rotation = rotation * Quat::FromAxisAngle({2, 1, -1}, 0.6f);
        body.inertial_transform.rotation = Quat::FromAxisAngle({3, -1, 2}, 0.7f);
        body.inertial_transform.position = {0.02f, -0.03f, 0.01f};
        const Quat principal = body.local_transform.rotation * body.inertial_transform.rotation;
        const Vec3 local_omega{11.0f, 6.0f, -3.0f};
        const Vec3 initial_omega = principal.Rotate(local_omega);
        const Vec3 initial_momentum = AngularMomentum(body.local_transform,
            body.inertial_transform, body.inertia, initial_omega);
        const float energy = 0.5f * initial_momentum.Dot(initial_omega);
        const auto reference = RotationReference(body.inertia, local_omega, principal, 2.0);
        const Quat reference_principal{float(reference[3]), float(reference[4]),
                                       float(reference[5]), float(reference[6])};
        const Vec3 reference_omega = reference_principal.Rotate(
            {float(reference[0]), float(reference[1]), float(reference[2])});
        const Vec3 reference_axis = reference_principal.Rotate({1, 0, 0});
        float previous_error = 0.0f;
        for (uint32_t steps : {240u, 480u}) {
            auto model = CookFreeBody(body, false);
            model.body_init[0].angular_velocity = initial_omega;
            auto config = Config();
            config.dt = 2.0f / steps;
            nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
            ASSERT_TRUE(world.Ready()) << world.CreationError();
            float max_residual = 0.0f;
            for (uint32_t step = 0u; step < steps; ++step) {
                ASSERT_TRUE(world.Step().AllOk());
                const auto status = Read<uint32_t>(world, nk::FieldId::BodyGyroStatus, 2u);
                ASSERT_EQ(status, std::vector<uint32_t>(2u));
                const auto residual = Read<float>(world, nk::FieldId::BodyGyroResidual, 2u);
                max_residual = std::max(max_residual, residual[0]);
            }
            const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 2u);
            const auto angular = Read<Vec3>(world, nk::FieldId::BodyAngularVelocity, 2u);
            const Vec3 momentum = AngularMomentum(poses[0], body.inertial_transform, body.inertia, angular[0]);
            const float momentum_error = (momentum - initial_momentum).Length() / initial_momentum.Length();
            const float energy_error = std::fabs(0.5f * momentum.Dot(angular[0]) / energy - 1.0f);
            const Quat final_principal = poses[0].rotation * body.inertial_transform.rotation;
            const float error = (angular[0] - reference_omega).Length() +
                                (final_principal.Rotate({1, 0, 0}) - reference_axis).Length();
            EXPECT_LT(momentum_error, 3.0e-4f);
            EXPECT_LT(energy_error, 5.0e-4f);
            EXPECT_LE(max_residual, 1.0e-6f);
            ExpectVectorNear(angular[0], angular[1], 0.0f);
            if (previous_error > 0.0f) EXPECT_GT(previous_error / error, 3.2f);
            previous_error = error;
            std::fprintf(stderr, "[gyro-reference] steps=%u error=%g L_relative=%g energy_relative=%g residual=%g\n",
                steps, error, momentum_error, energy_error, max_residual);

            auto invalid = angular;
            invalid[0].x = std::numeric_limits<float>::infinity();
            ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyAngularVelocity,
                                                   invalid.data(), invalid.size() * sizeof(Vec3)));
            ASSERT_TRUE(world.Step().AllOk());
            const auto status = Read<uint32_t>(world, nk::FieldId::BodyGyroStatus, 2u);
            EXPECT_EQ(status[0], phi::kBodyGyroInvalidInput);
            EXPECT_EQ(status[1], 0u);
            const auto env_status = Read<uint32_t>(world, nk::FieldId::EnvStatus, 2u);
            EXPECT_EQ(env_status[0] & phi::kEnvStatusGyroFailure, phi::kEnvStatusGyroFailure);
            EXPECT_EQ(env_status[1], 0u);
            ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
            EXPECT_EQ(Read<uint32_t>(world, nk::FieldId::BodyGyroStatus, 2u), status);
            ASSERT_EQ(world.Reset({0u}), phi::Status::Ok);
            EXPECT_EQ(Read<uint32_t>(world, nk::FieldId::BodyGyroStatus, 2u), std::vector<uint32_t>(2u));
        }
    }
}

TEST(ParticleAerodynamics, SharedFacesAreDissipativeBoundedAndDeterministic) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (float coefficient : {0.0f, 1.0f, 1.0e6f}) {
        scene::cook::VbdCookInput input;
        input.positions = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        input.velocities = {{5, -3, 20}, {}, {-2, 4, 12}, {3, 1, -4}};
        input.inv_mass = {1.0f, 0.0f, 2.0f, 5.0f};
        input.half_thickness.assign(4u, 0.0f);
        input.friction.assign(4u, 0.6f);
        input.aero_triangles = {{{0, 1, 2}}, {{0, 2, 3}}};
        input.aero_drag_normal = coefficient;
        input.aero_drag_tangent = coefficient * 0.1f;
        input.aero_drag_max_dv = 0.5f;
        nk::Model model;
        scene::cook::CookVbdParticles(model, 3u, input);
        auto config = Config();
        nk::World world(std::move(model), 3u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        phi::AeroDragParams params{config.dt, input.aero_drag_normal, input.aero_drag_tangent,
                                  input.aero_drag_max_dv, 6u, 12u};
        const auto before = Read<Vec3>(world, nk::FieldId::ParticleVel, 12u);
        ASSERT_EQ(world.DispatchOp(phi::NkOp::ParticleAeroDrag, &params), phi::Status::Ok);
        const auto after = Read<Vec3>(world, nk::FieldId::ParticleVel, 12u);
        double work = 0.0, energy_change = 0.0;
        for (uint32_t p = 0u; p < 12u; ++p) {
            const float inverse_mass = input.inv_mass[p % 4u];
            if (inverse_mass == 0.0f || coefficient == 0.0f) EXPECT_EQ(before[p], after[p]);
            if (inverse_mass == 0.0f) continue;
            const auto delta = after[p] - before[p];
            EXPECT_LE(delta.Length(), input.aero_drag_max_dv + 1.0e-5f);
            work += before[p].Dot(delta) / inverse_mass;
            energy_change += (after[p].LengthSq() - before[p].LengthSq()) * 0.5 / inverse_mass;
        }
        EXPECT_LE(work, 1.0e-5);
        EXPECT_LE(energy_change, 1.0e-5);
        for (uint32_t repeat = 0u; repeat < 5u; ++repeat) {
            ASSERT_EQ(world.Reset(), phi::Status::Ok);
            ASSERT_EQ(world.DispatchOp(phi::NkOp::ParticleAeroDrag, &params), phi::Status::Ok);
            EXPECT_EQ(Read<Vec3>(world, nk::FieldId::ParticleVel, 12u), after);
        }
        std::fprintf(stderr, "[aero] coefficient=%g work=%g delta_energy=%g\n", coefficient, work, energy_change);
    }
}

TEST(PipelineContracts, MissingOpsAndDispatchFailureStopAtTheBoundary) {
    auto& real = Device();
    if (!real.backend) GTEST_SKIP() << "no CUDA backend";
    struct DeviceProxy {
        const phi::DeviceI* iface;
        phi::Device* real;
        phi::NkOp missing = phi::NkOp::Count;
        bool allocated = false;
        bool no_buffer = false;
    };
    struct BackendProxy {
        const phi::BackendI* iface;
        phi::Backend* real;
        phi::NkOp failure = phi::NkOp::Count;
        std::vector<phi::NkOp> calls;
        uint32_t captures = 0u;
        bool throw_dispatch = false;
        bool fail_completion = false;
    };
    auto device_iface = *phi::IfaceOf(real.device);
    device_iface.supports_op = [](phi::Device* device, phi::NkOp op) {
        const auto& proxy = *reinterpret_cast<DeviceProxy*>(device);
        return op != proxy.missing && phi::DeviceSupportsOp(proxy.real, op);
    };
    device_iface.get_buffer_type = [](phi::Device* device) {
        auto& proxy = *reinterpret_cast<DeviceProxy*>(device);
        proxy.allocated = true;
        if (proxy.no_buffer) return static_cast<phi::BufferType*>(nullptr);
        return phi::DeviceBufferType(proxy.real);
    };
    auto backend_iface = *phi::IfaceOf(real.backend);
    backend_iface.dispatch = [](phi::Backend* backend, const phi::ModelView& model,
                                const phi::DataView& data, const phi::OpCall& call) {
        auto& proxy = *reinterpret_cast<BackendProxy*>(backend);
        proxy.calls.push_back(call.op);
        if (call.op == proxy.failure) {
            if (proxy.throw_dispatch) throw std::runtime_error("injected operator exception");
            return phi::Status::Failed;
        }
        return phi::BackendDispatch(proxy.real, model, data, call);
    };
    backend_iface.synchronize = [](phi::Backend* backend, phi::ExecutionError* error) {
        const auto& proxy = *reinterpret_cast<BackendProxy*>(backend);
        if (proxy.fail_completion) {
            if (error) {
                *error = {};
                error->status = phi::Status::Failed;
                error->native_code = 719;
                std::snprintf(error->message, sizeof(error->message), "injected completion failure");
            }
            return phi::Status::Failed;
        }
        return phi::BackendSynchronize(proxy.real, error);
    };
    backend_iface.plan_create = [](phi::Backend* backend, const phi::ModelView&, const phi::DataView&,
                                   const phi::OpCall*, int, phi::ExecutionError* error) -> phi::Plan* {
        ++reinterpret_cast<BackendProxy*>(backend)->captures;
        if (error) {
            error->status = phi::Status::Unsupported;
            error->failed_op = phi::NkOp::LbvhBuild;
            std::snprintf(error->message, sizeof(error->message), "injected capture failure");
        }
        return nullptr;
    };
    DeviceProxy device{&device_iface, real.device};
    BackendProxy backend{&backend_iface, real.backend};
    auto* dev = reinterpret_cast<phi::Device*>(&device);
    auto* be = reinterpret_cast<phi::Backend*>(&backend);
    scene::RigidBodyRecord body;
    body.mass = 1.0f;
    body.inertia = {1, 1, 1};
    device.missing = phi::NkOp::IntegratePosition;
    nk::World missing(CookFreeBody(body, false), 1u, dev, be, Config());
    EXPECT_FALSE(missing.Ready());
    EXPECT_EQ(missing.CreationStatus(), phi::Status::Unsupported);
    EXPECT_FALSE(missing.CreationError().empty());
    EXPECT_FALSE(device.allocated);
    EXPECT_TRUE(backend.calls.empty());
    EXPECT_FALSE(missing.Step().AllOk());

    device.missing = phi::NkOp::Count;
    device.no_buffer = true;
    nk::World no_buffer(CookFreeBody(body, false), 1u, dev, be, Config());
    EXPECT_FALSE(no_buffer.Ready());
    EXPECT_EQ(no_buffer.CreationStatus(), phi::Status::Failed);
    EXPECT_EQ(no_buffer.CreationError(), "device has no buffer type");
    EXPECT_TRUE(backend.calls.empty());
    device.no_buffer = false;

    device.missing = phi::NkOp::ReadoutContactWrench;
    nk::World world(CookFreeBody(body, true), 1u, dev, be, Config());
    ASSERT_TRUE(world.Ready()) << world.CreationError();
    const size_t original_count = world.GetPipeline().Size();
    EXPECT_EQ(world.FieldPtr(nk::FieldId::ContactForce), nullptr);
    EXPECT_EQ(world.LastStatus(), phi::Status::Unsupported);
    EXPECT_EQ(world.GetPipeline().Size(), original_count);
    ASSERT_TRUE(world.Step().AllOk());
    device.missing = phi::NkOp::Count;
    backend.failure = phi::NkOp::ReadoutContactWrench;
    EXPECT_EQ(world.FieldPtr(nk::FieldId::ContactForce), nullptr);
    EXPECT_EQ(world.LastStatus(), phi::Status::Failed);
    EXPECT_EQ(world.GetPipeline().Size(), original_count);
    backend.failure = phi::NkOp::Count;
    ASSERT_NE(world.FieldPtr(nk::FieldId::ContactForce), nullptr);
    ASSERT_GT(world.GetPipeline().Size(), original_count);
    backend.calls.clear();
    backend.failure = phi::NkOp::IntegratePosition;
    const auto result = world.Step();
    EXPECT_FALSE(result.AllOk());
    EXPECT_EQ(result.result, phi::Status::Failed);
    EXPECT_EQ(result.failed_op, phi::NkOp::IntegratePosition);
    EXPECT_EQ(backend.calls.back(), phi::NkOp::IntegratePosition);
    EXPECT_EQ(std::count(backend.calls.begin(), backend.calls.end(), phi::NkOp::ReadoutContactWrench), 0);
    backend.calls.clear();
    backend.throw_dispatch = true;
    const auto thrown = world.Step();
    EXPECT_EQ(thrown.result, phi::Status::Failed);
    EXPECT_EQ(thrown.failed_op, phi::NkOp::IntegratePosition);
    EXPECT_EQ(backend.calls.back(), phi::NkOp::IntegratePosition);
    EXPECT_STREQ(world.LastExecutionError().message, "injected operator exception");
    backend.throw_dispatch = false;
    backend.failure = phi::NkOp::Count;
    for (uint32_t attempt = 0u; attempt < 5u; ++attempt)
        EXPECT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), phi::Status::Unsupported);
    EXPECT_EQ(backend.captures, 1u);
    EXPECT_EQ(world.CaptureAttempts(), 1u);
    EXPECT_EQ(world.GetExecutionMode(), nk::World::ExecutionMode::Eager);
    EXPECT_EQ(world.LastExecutionError().failed_op, phi::NkOp::LbvhBuild);
    EXPECT_STREQ(world.LastExecutionError().message, "injected capture failure");
    EXPECT_TRUE(world.Step().AllOk());
    backend.fail_completion = true;
    EXPECT_EQ(world.Synchronize(), phi::Status::Failed);
    EXPECT_EQ(world.LastExecutionError().native_code, 719);
    EXPECT_STREQ(world.LastExecutionError().message, "injected completion failure");
    backend.fail_completion = false;
    EXPECT_EQ(world.Synchronize(), phi::Status::Ok);
    EXPECT_EQ(world.LastExecutionError().status, phi::Status::Ok);
}

TEST(PipelineContracts, BufferFailuresPreserveStatusAndReleaseAllocations) {
    auto& real = Device();
    if (!real.backend) GTEST_SKIP() << "no CUDA backend";
    enum class Transfer { None, Upload, Download, Memset, Copy };
    struct Allocator {
        const phi::BufferTypeI* iface;
        const phi::BufferI* buffer_iface;
        phi::BufferType* real;
        uint32_t allocations = 0u, live = 0u, fail_allocation = 0u;
        Transfer fail_transfer = Transfer::None;
    };
    struct Buffer {
        const phi::BufferI* iface;
        phi::Buffer* real;
        Allocator* owner;
    };
    phi::BufferI buffer_iface{};
    buffer_iface.free = [](phi::Buffer* buffer) {
        auto* proxy = reinterpret_cast<Buffer*>(buffer);
        phi::BufferFree(proxy->real);
        --proxy->owner->live;
        delete proxy;
    };
    buffer_iface.base = [](phi::Buffer* buffer) {
        return phi::BufferBase(reinterpret_cast<Buffer*>(buffer)->real);
    };
    buffer_iface.upload = [](phi::Buffer* buffer, const void* source, size_t offset, size_t bytes) {
        auto& proxy = *reinterpret_cast<Buffer*>(buffer);
        return proxy.owner->fail_transfer == Transfer::Upload ? phi::Status::Failed
            : phi::BufferUpload(proxy.real, source, offset, bytes);
    };
    buffer_iface.download = [](phi::Buffer* buffer, void* target, size_t offset, size_t bytes) {
        auto& proxy = *reinterpret_cast<Buffer*>(buffer);
        return proxy.owner->fail_transfer == Transfer::Download ? phi::Status::Failed
            : phi::BufferDownload(proxy.real, target, offset, bytes);
    };
    buffer_iface.memset = [](phi::Buffer* buffer, uint8_t value, size_t offset, size_t bytes) {
        auto& proxy = *reinterpret_cast<Buffer*>(buffer);
        return proxy.owner->fail_transfer == Transfer::Memset ? phi::Status::Failed
            : phi::BufferMemset(proxy.real, value, offset, bytes);
    };
    buffer_iface.copy_from = [](phi::Buffer* target, phi::Buffer* source,
                                size_t target_offset, size_t source_offset, size_t bytes) {
        auto& proxy = *reinterpret_cast<Buffer*>(target);
        return proxy.owner->fail_transfer == Transfer::Copy ? phi::Status::Failed
            : phi::BufferCopyFrom(proxy.real, reinterpret_cast<Buffer*>(source)->real,
                                  target_offset, source_offset, bytes);
    };
    auto allocator_iface = *phi::IfaceOf(phi::DeviceBufferType(real.device));
    allocator_iface.get_name = [](phi::BufferType* type) {
        return phi::BufferTypeName(reinterpret_cast<Allocator*>(type)->real);
    };
    allocator_iface.alignment = [](phi::BufferType* type) {
        return phi::BufferTypeAlignment(reinterpret_cast<Allocator*>(type)->real);
    };
    allocator_iface.is_host = [](phi::BufferType* type) {
        return phi::BufferTypeIsHost(reinterpret_cast<Allocator*>(type)->real);
    };
    allocator_iface.alloc = [](phi::BufferType* type, size_t bytes, phi::Status* status) -> phi::Buffer* {
        auto& allocator = *reinterpret_cast<Allocator*>(type);
        if (++allocator.allocations == allocator.fail_allocation) {
            if (status) *status = phi::Status::OutOfMemory;
            return nullptr;
        }
        auto* buffer = phi::BufferAlloc(allocator.real, bytes, status);
        if (!buffer) return nullptr;
        ++allocator.live;
        return reinterpret_cast<phi::Buffer*>(new Buffer{allocator.buffer_iface, buffer, &allocator});
    };
    Allocator allocator{&allocator_iface, &buffer_iface, phi::DeviceBufferType(real.device)};
    auto* type = reinterpret_cast<phi::BufferType*>(&allocator);
    nk::ModelCapacities capacities;
    capacities.env_count = 1u;
    capacities.bodies_per_env = 1u;
    phi::DataView view{};
    {
        nk::Data data;
        allocator.fail_allocation = 2u;
        EXPECT_EQ(data.Allocate(type, capacities, &view), phi::Status::OutOfMemory);
        EXPECT_EQ(allocator.live, 0u);
        allocator.fail_allocation = 0u;
        allocator.fail_transfer = Transfer::Memset;
        EXPECT_EQ(data.Allocate(type, capacities, &view), phi::Status::Failed);
        EXPECT_EQ(allocator.live, 0u);
        allocator.fail_transfer = Transfer::None;
        ASSERT_EQ(data.Allocate(type, capacities, &view), phi::Status::Ok);
        Vec3 value{};
        allocator.fail_transfer = Transfer::Upload;
        EXPECT_EQ(data.UploadFieldStatus(nk::FieldId::BodyForce, &value, sizeof(value)), phi::Status::Failed);
        EXPECT_EQ(data.UploadFieldStatus(nk::FieldId::BodyForce, &value, sizeof(value), ~uint64_t{0}),
                  phi::Status::InvalidArgument);
        allocator.fail_transfer = Transfer::Download;
        EXPECT_EQ(data.DownloadFieldStatus(nk::FieldId::BodyForce, &value, sizeof(value)), phi::Status::Failed);
        allocator.fail_transfer = Transfer::Copy;
        EXPECT_EQ(phi::BufferCopyFrom(data.GetArena().ScratchBuffer(), data.GetArena().PersistentBuffer(),
                                      0u, 0u, sizeof(value)), phi::Status::Failed);
        allocator.fail_transfer = Transfer::None;
        EXPECT_EQ(data.UploadFieldStatus(nk::FieldId::BodyForce, &value, sizeof(value)), phi::Status::Ok);
        EXPECT_EQ(data.DownloadFieldStatus(nk::FieldId::BodyForce, &value, sizeof(value)), phi::Status::Ok);
    }
    EXPECT_EQ(allocator.live, 0u);
    scene::RigidBodyRecord body;
    body.mass = 1.0f;
    body.inertia = {1, 1, 1};
    auto model = CookFreeBody(body, false);
    phi::ModelView model_view{};
    allocator.fail_transfer = Transfer::Upload;
    EXPECT_EQ(model.UploadTo(type, &model_view), phi::Status::Failed);
    EXPECT_EQ(allocator.live, 0u);

    using OwnedBuffer = std::unique_ptr<phi::Buffer, decltype(&phi::BufferFree)>;
    std::vector<uint8_t> expected(1u << 20u), actual(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<uint8_t>(i % 251u);
    auto* device_type = phi::BackendDeviceBufferType(real.backend);
    auto* host_type = phi::BackendHostBufferType(real.backend);
    OwnedBuffer device_buffer(phi::BufferAlloc(device_type, expected.size()), &phi::BufferFree);
    OwnedBuffer host_source(phi::BufferAlloc(host_type, expected.size()), &phi::BufferFree);
    OwnedBuffer host_target(phi::BufferAlloc(host_type, expected.size()), &phi::BufferFree);
    ASSERT_TRUE(device_buffer && host_source && host_target);
    ASSERT_EQ(phi::BufferUpload(device_buffer.get(), expected.data(), 0u, expected.size()), phi::Status::Ok);
    ASSERT_EQ(phi::BufferCopyFrom(host_source.get(), device_buffer.get(), 0u, 0u, expected.size()), phi::Status::Ok);
    ASSERT_EQ(phi::BufferCopyFrom(host_target.get(), host_source.get(), 0u, 0u, expected.size()), phi::Status::Ok);
    ASSERT_EQ(phi::BufferDownload(host_target.get(), actual.data(), 0u, actual.size()), phi::Status::Ok);
    EXPECT_EQ(actual, expected);
    OwnedBuffer foreign(phi::BufferAlloc(type, expected.size()), &phi::BufferFree);
    ASSERT_TRUE(foreign);
    EXPECT_EQ(phi::BufferCopyFrom(device_buffer.get(), foreign.get(), 0u, 0u, 1u), phi::Status::Unsupported);
}

TEST(PipelineContracts, TopologyValidationRejectsUnsupportedAndMalformedLayouts) {
    const auto path = std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes";
    if (!std::filesystem::exists(path / "go2_stand.usda")) GTEST_SKIP();
    const auto fixed = nuka::import::LoadUsd((path / "go2_stand.usda").string());
    const auto floating = nuka::import::LoadUsd((path / "go2_float.usda").string());
    scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    const auto same = scene::Compose(fixed, fixed, Transform{{2, 0, 0}, Quat::Identity()}, "second_");
    auto supported = scene::cook::CookToModel(same, 1u, options).model;
    std::string reason;
    ASSERT_EQ(supported.ValidateTopology(&reason), phi::Status::Ok) << reason;
    supported.articulation.articulation_link_offset[1] = 0u;
    EXPECT_EQ(supported.ValidateTopology(&reason), phi::Status::InvalidArgument);
    EXPECT_FALSE(reason.empty());
    const auto mixed = scene::Compose(fixed, floating, Transform{{2, 0, 0}, Quat::Identity()}, "floating_");
    auto unsupported = scene::cook::CookToModel(mixed, 1u, options).model;
    EXPECT_EQ(unsupported.ValidateTopology(&reason), phi::Status::Unsupported);
    EXPECT_FALSE(reason.empty());
    auto single = scene::cook::CookToModel(fixed, 1u, options).model;
    single.articulation.articulation_link_count.clear();
    single.articulation.articulation_link_offset.clear();
    ASSERT_EQ(single.ValidateTopology(&reason), phi::Status::Ok) << reason;
    const auto parent = single.articulation.parent_link[1];
    single.articulation.parent_link[1] = single.capacities.links_per_env;
    EXPECT_EQ(single.ValidateTopology(&reason), phi::Status::InvalidArgument);
    single.articulation.parent_link[1] = parent;
    single.capacities.dofs_per_env = phi::kMaxArticulationDof + 1u;
    EXPECT_EQ(single.ValidateTopology(&reason), phi::Status::Unsupported);
}

TEST(FreeRigidDynamics, WorldAngularVelocityRotatesAboutTheAuthoredCenterOfMass) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (Vec3 offset : {Vec3{}, Vec3{0.025f, -0.01f, 0.026f}}) {
        scene::RigidBodyRecord body;
        body.name = "free_body";
        body.mass = 1.0f;
        body.inertia = {1, 1, 1};
        body.local_transform.position = {0, 0, 2};
        body.local_transform.rotation = Quat::FromAxisAngle({0, 1, 0}, 1.57079632679f);
        body.inertial_transform.position = offset;
        auto model = CookFreeBody(body, false);
        ASSERT_EQ(model.body_init.size(), 1u);
        const Vec3 velocity{0.1f, -0.03f, 0.02f};
        const Vec3 omega{1, 0, 0};
        model.body_init[0].linear_velocity = velocity;
        model.body_init[0].angular_velocity = omega;
        const auto config = Config();
        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        constexpr uint32_t steps = 20u;
        for (uint32_t i = 0; i < steps; ++i) ASSERT_TRUE(world.Step().AllOk());
        const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 2u);
        const float duration = steps * config.dt;
        const Quat expected_rotation =
            Quat::FromAxisAngle(omega, duration) * body.local_transform.rotation;
        const Vec3 expected_com = body.local_transform.position +
            body.local_transform.rotation.Rotate(offset) + velocity * duration;
        for (const auto& pose : poses) {
            ExpectVectorNear(pose.position + pose.rotation.Rotate(offset), expected_com, 3.0e-6f);
            EXPECT_NEAR(pose.rotation.w, expected_rotation.w, 2.0e-6f);
            EXPECT_NEAR(pose.rotation.x, expected_rotation.x, 2.0e-6f);
            EXPECT_NEAR(pose.rotation.y, expected_rotation.y, 2.0e-6f);
            EXPECT_NEAR(pose.rotation.z, expected_rotation.z, 2.0e-6f);
        }
    }
}

TEST(FreeRigidDynamics, GravityCompensationAccountsForWorkAndPreservesOtherBodies) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (float compensation : {0.0f, 0.5f, 1.0f, 1.5f}) {
        scene::RigidBodyRecord body;
        body.name = "weighted_body";
        body.mass = 2.0f;
        body.gravity_compensation = compensation;
        auto model = CookFreeBody(body, false);
        model.body_init[0].linear_velocity = {0.0f, 0.0f, 0.25f};
        auto config = Config();
        config.dt = 0.001f;
        config.gravity[2] = -9.81f;
        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        ASSERT_NE(world.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        const float fractions[2] = {compensation, 0.0f};
        EXPECT_EQ(Read<float>(world, nk::FieldId::BodyGravityCompensation, 2u),
            std::vector<float>(2u, compensation));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyGravityCompensation, fractions, sizeof(fractions)));
        for (uint32_t step = 0u; step < 100u; ++step) {
            const auto before = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
            ASSERT_EQ(world.StepPlanned(), phi::Status::Ok);
            const auto after = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
            const auto ledger = Read<float>(world, nk::FieldId::EnergyLedger, 2u * nk::kEnergyColumnCount);
            for (uint32_t env = 0u; env < 2u; ++env) {
                const float expected = before[env].z - 9.81f * (1.0f - fractions[env]) * config.dt;
                EXPECT_NEAR(after[env].z, expected, 1.0e-7f);
                const double work = 0.5 * body.mass * 9.81 * fractions[env] *
                    (double(before[env].z) + after[env].z) * config.dt;
                EXPECT_NEAR(ledger[env * nk::kEnergyColumnCount + uint32_t(nk::EnergyColumn::ExternalWork)], work, 2.0e-8);
            }
        }
    }
}

TEST(FreeRigidDynamics, WorldWrenchesAndGravityAreCovariantAndConsumedOnce) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (uint32_t substeps : {1u, 5u}) {
      for (bool planned : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "substeps=" << substeps << " graph=" << planned);
    for (Quat rotation : {Quat::Identity(), Quat::FromAxisAngle({1, 2, -3}, 1.1f)}) {
        scene::RigidBodyRecord body;
        body.name = "free_body";
        body.mass = 2.0f;
        body.inertia = {0.4f, 0.7f, 1.2f};
        body.local_transform.position = rotation.Rotate({0.3f, -0.4f, 1.5f});
        body.local_transform.rotation = rotation * Quat::FromAxisAngle({2, 1, 3}, 0.6f);
        body.inertial_transform.position = {0.04f, -0.02f, 0.01f};
        body.inertial_transform.rotation = Quat::FromAxisAngle({3, -2, 1}, 0.7f);
        auto config = Config();
        config.dt = 0.001f;
        config.substeps = substeps;
        const Vec3 gravity = rotation.Rotate({1.7f, -2.3f, -8.1f});
        config.gravity[0] = gravity.x;
        config.gravity[1] = gravity.y;
        config.gravity[2] = gravity.z;
        const Vec3 force = rotation.Rotate({0.7f, -0.9f, 1.1f});
        const Vec3 torque = rotation.Rotate(Vec3{0.2f, -0.3f, 0.1f}.Cross({0.7f, -0.9f, 1.1f}));
        nk::World world(CookFreeBody(body, false), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        ASSERT_NE(world.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        const auto advance = [&]() { return planned ? world.StepPlanned() : world.Step().result; };
        const std::vector<Vec3> forces{force, Vec3{}}, torques{torque, Vec3{}};
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyTorque, torques.data(), torques.size() * sizeof(Vec3)));
        ASSERT_EQ(advance(), phi::Status::Ok);
        auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
        const auto omega = Read<Vec3>(world, nk::FieldId::BodyAngularVelocity, 2u);
        const auto pose = Read<Transform>(world, nk::FieldId::BodyPose, 2u);
        ExpectVectorNear(velocity[0], (gravity + force / body.mass) * config.dt, 1.0e-6f);
        ExpectVectorNear(velocity[1], gravity * config.dt, 1.0e-6f);
        const Quat principal = body.local_transform.rotation * body.inertial_transform.rotation;
        Vec3 principal_torque = principal.Conjugate().Rotate(torque);
        principal_torque.x /= body.inertia.x;
        principal_torque.y /= body.inertia.y;
        principal_torque.z /= body.inertia.z;
        ExpectVectorNear(omega[0], principal.Rotate(principal_torque) * config.dt, 1.0e-6f);
        ExpectVectorNear(omega[1], Vec3{}, 1.0e-7f);
        const auto ledger = Read<float>(world, nk::FieldId::EnergyLedger, 2u * substeps * nk::kEnergyColumnCount);
        const auto coverage = Read<uint32_t>(world, nk::FieldId::EnergyLedgerStatus, 2u * substeps);
        double work = 0.0;
        for (uint32_t step = 0u; step < substeps; ++step) {
            work += ledger[step * nk::kEnergyColumnCount + static_cast<uint32_t>(nk::EnergyColumn::ExternalWork)];
            EXPECT_FLOAT_EQ(ledger[(substeps + step) * nk::kEnergyColumnCount +
                static_cast<uint32_t>(nk::EnergyColumn::ExternalWork)], 0.0f);
        }
        EXPECT_NEAR(work, 0.5 * config.dt * (force.Dot(velocity[0]) + torque.Dot(omega[0])), 1.0e-9);
        EXPECT_EQ(coverage, std::vector<uint32_t>(2u * substeps));
        const Vec3 initial_com = body.local_transform.TransformPoint(body.inertial_transform.position);
        for (uint32_t env = 0; env < 2u; ++env)
            ExpectVectorNear(pose[env].TransformPoint(body.inertial_transform.position),
                             initial_com + velocity[env] *
                                 (config.dt * (0.5f + 0.5f / substeps)), 1.0e-6f);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), std::vector<Vec3>(2u));
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyTorque, 2u), std::vector<Vec3>(2u));
        ASSERT_EQ(advance(), phi::Status::Ok);
        velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
        ExpectVectorNear(velocity[0], gravity * (2.0f * config.dt) + force * (config.dt / body.mass), 1.0e-6f);
        ExpectVectorNear(velocity[1], gravity * (2.0f * config.dt), 1.0e-6f);
        const auto unloaded = Read<float>(world, nk::FieldId::EnergyLedger, 2u * substeps * nk::kEnergyColumnCount);
        for (uint32_t record = 0u; record < 2u * substeps; ++record)
            EXPECT_FLOAT_EQ(unloaded[record * nk::kEnergyColumnCount +
                static_cast<uint32_t>(nk::EnergyColumn::ExternalWork)], 0.0f);
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
        ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), forces);
        ASSERT_EQ(world.Reset({0u}), phi::Status::Ok);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), std::vector<Vec3>(2u));
    }
      }
    }
}

TEST(FreeRigidDynamics, FixedAndFloatingArticulationsRespectRotatedGravity) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const Quat rotation = Quat::FromAxisAngle({1, -3, 2}, 1.1f);
    const Vec3 gravity{2.3f, -4.1f, -8.2f};
    const Vec3 rotated_gravity = rotation.Rotate(gravity);
    for (const char* filename : {"go2_stand.usda", "go2_float.usda"}) {
        SCOPED_TRACE(filename);
        const auto path = std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes" / filename;
        ASSERT_TRUE(std::filesystem::exists(path));
        const auto source = nuka::import::LoadUsd(path.string());
        const auto rotated = scene::Compose(scene::SceneIR{}, source, Transform{{}, rotation}, "rotated_");
        scene::cook::CookToModelOptions options;
        options.enable_contacts = false;
        auto config = Config();
        config.dt = 0.0005f;
        config.gravity[0] = gravity.x;
        config.gravity[1] = gravity.y;
        config.gravity[2] = gravity.z;
        nk::World original(scene::cook::CookToModel(source, 1, options).model,
                           2u, backend.device, backend.backend, config);
        config.gravity[0] = rotated_gravity.x;
        config.gravity[1] = rotated_gravity.y;
        config.gravity[2] = rotated_gravity.z;
        nk::World transformed(scene::cook::CookToModel(rotated, 1, options).model,
                              2u, backend.device, backend.backend, config);
        ASSERT_TRUE(original.Ready());
        ASSERT_TRUE(transformed.Ready());
        ASSERT_NE(original.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        ASSERT_NE(transformed.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        const uint32_t links = original.GetModel().capacities.links_per_env * 2u;
        std::vector<nk::Spatial6> loads(links), rotated_loads(links);
        const Vec3 force{0.7f, -0.4f, 1.3f}, torque{-0.1f, 0.3f, 0.2f};
        const Vec3 rotated_force = rotation.Rotate(force), rotated_torque = rotation.Rotate(torque);
        for (uint32_t link = 0u; link < links / 2u; ++link) {
            loads[link] = {{force.x, force.y, force.z, torque.x, torque.y, torque.z}};
            rotated_loads[link] = {{rotated_force.x, rotated_force.y, rotated_force.z,
                rotated_torque.x, rotated_torque.y, rotated_torque.z}};
        }
        ASSERT_TRUE(original.GetData().UploadField(nk::FieldId::LinkWrench, loads.data(), loads.size() * sizeof(nk::Spatial6)));
        ASSERT_TRUE(transformed.GetData().UploadField(nk::FieldId::LinkWrench, rotated_loads.data(), rotated_loads.size() * sizeof(nk::Spatial6)));
        ASSERT_TRUE(original.Step().AllOk());
        ASSERT_TRUE(transformed.Step().AllOk());
        const auto qdot = Read<float>(original, nk::FieldId::Qdot, links);
        const auto rotated_qdot = Read<float>(transformed, nk::FieldId::Qdot, links);
        for (uint32_t link = 0; link < links; ++link)
            EXPECT_NEAR(rotated_qdot[link], qdot[link], 2.0e-6f) << "link=" << link;
        const auto velocity = Read<nk::Spatial6>(original, nk::FieldId::LinkVelocity, links);
        const auto rotated_velocity = Read<nk::Spatial6>(transformed, nk::FieldId::LinkVelocity, links);
        const auto pose = Read<Transform>(original, nk::FieldId::LinkPose, links);
        std::vector<Transform> inertial_frames(links);
        ASSERT_EQ(original.GetModel().DownloadFieldStatus(nk::FieldId::LinkInertialFrame,
            inertial_frames.data(), inertial_frames.size() * sizeof(Transform)), phi::Status::Ok);
        double external_work = 0.0;
        for (uint32_t link = 0u; link < links / 2u; ++link) {
            const auto& v = velocity[link];
            const Vec3 angular = pose[link].rotation.Rotate({v.v[0], v.v[1], v.v[2]});
            const Vec3 offset = pose[link].rotation.Rotate(inertial_frames[link].position);
            const Vec3 linear = pose[link].rotation.Rotate({v.v[3], v.v[4], v.v[5]}) + angular.Cross(offset);
            external_work += 0.5 * config.dt * (force.Dot(linear) + torque.Dot(angular));
        }
        const auto ledger = Read<float>(original, nk::FieldId::EnergyLedger, 2u * nk::kEnergyColumnCount);
        const auto rotated_ledger = Read<float>(transformed, nk::FieldId::EnergyLedger, 2u * nk::kEnergyColumnCount);
        const uint32_t work_column = static_cast<uint32_t>(nk::EnergyColumn::ExternalWork);
        EXPECT_NEAR(ledger[work_column], external_work, 1.0e-8);
        EXPECT_NEAR(rotated_ledger[work_column], ledger[work_column], 1.0e-8);
        EXPECT_FLOAT_EQ(ledger[nk::kEnergyColumnCount + work_column], 0.0f);
        EXPECT_EQ(Read<uint32_t>(original, nk::FieldId::EnergyLedgerStatus, 2u), std::vector<uint32_t>(2u));
        EXPECT_EQ(Read<uint32_t>(transformed, nk::FieldId::EnergyLedgerStatus, 2u), std::vector<uint32_t>(2u));
        for (uint32_t link = 0; link < links; ++link)
            for (uint32_t axis = 0; axis < 6u; ++axis)
                EXPECT_NEAR(rotated_velocity[link].v[axis], velocity[link].v[axis], 2.0e-6f)
                    << "link=" << link << " axis=" << axis;
    }
}

TEST(FreeRigidDynamics, FrictionMixingAppliesToPrimitiveAndMeshContacts) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool surface : {false, true}) {
        for (bool geometric : {false, true}) {
            scene::RigidBodyRecord body;
            body.mass = 1.0f;
            body.inertia = {0.01f, 0.01f, 0.01f};
            body.local_transform.position = {0, 0, surface ? 0.1001f : 0.099f};
            auto model = CookFreeBody(body, true, surface);
            model.body_init[1].linear_velocity = {1, 0, -1};
            for (uint32_t shape = 0u; shape < model.shape_table_rows.size(); ++shape) {
                const auto profile_index = model.shape_table_rows[shape].contact_profile_index;
                auto profile = model.material_buckets[profile_index].Profile();
                profile.mu1 = profile.mu2 = shape == 0u ? 0.0f : 0.5f;
                if (geometric) profile.flags |= nk::kContactProfileFlagGeometricFriction;
                model.material_buckets[profile_index] = nk::ModelMaterialBucket(profile);
            }
            auto config = Config();
            config.vel_iters = 64u;
            config.velocity_iterations_override = true;
            config.measure_contact_residual = true;
            nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
            ASSERT_TRUE(world.Ready()) << world.CreationError();
            ASSERT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), phi::Status::Ok);
            ASSERT_NE(world.FieldPtr(nk::FieldId::ContactSolveCounts), nullptr);
            ASSERT_TRUE(world.Step().AllOk());
            const auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
            const auto counts = Read<uint32_t>(world, nk::FieldId::ContactSolveCounts,
                2u * nuka::constraint::kContactSolveCountSize);
            for (uint32_t env = 0u; env < 2u; ++env) {
                EXPECT_GT(counts[env * nuka::constraint::kContactSolveCountSize], 0u);
                EXPECT_GT(velocity[2u * env + 1u].z, -0.9f);
                if (geometric) EXPECT_NEAR(velocity[2u * env + 1u].x, 1.0f, 2.0e-5f);
                else EXPECT_LT(velocity[2u * env + 1u].x, 0.99f);
            }
            ASSERT_EQ(world.Reset(), phi::Status::Ok);
            ASSERT_TRUE(world.Step().AllOk());
            const auto repeated = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
            for (uint32_t body_index = 0u; body_index < 4u; ++body_index)
                ExpectVectorNear(repeated[body_index], velocity[body_index], 2.0e-5f);
        }
    }
}

TEST(FreeRigidDynamics, SeparatedMeshStaysStillInsideTheQueryBand) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (float gap : {0.0001f, 0.0008f}) {
        scene::RigidBodyRecord body;
        body.name = "free_mesh";
        body.mass = 0.5f;
        body.inertia = {0.003f, 0.007f, 0.011f};
        body.local_transform.position = {0, 0, 0.1f + gap};
        auto config = Config();
        config.pos_iters = 8u;
        nk::World world(CookFreeBody(body, true, true), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        for (uint32_t step = 0u; step < 4u; ++step) ASSERT_TRUE(world.Step().AllOk());
        const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 4u);
        const auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
        for (uint32_t env = 0u; env < 2u; ++env) {
            ExpectVectorNear(poses[env * 2u + 1u].position, body.local_transform.position, 1.0e-6f);
            ExpectVectorNear(velocity[env * 2u + 1u], {}, 1.0e-6f);
        }
    }
}

TEST(FreeRigidDynamics, ContactImpulseUsesTheCenterOfMassAndWorldInertia) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool surface_contact : {false, true}) {
     for (bool offset_center : {false, true}) {
      for (uint16_t position_iterations : {0u, 4u}) {
        SCOPED_TRACE(::testing::Message() << "offset=" << offset_center << " surface=" << surface_contact);
        scene::RigidBodyRecord body;
        body.name = "free_body";
        body.mass = 0.5f;
        body.inertia = {0.003f, 0.007f, 0.011f};
        body.local_transform.position = {0, 0, surface_contact ? 0.1001f : 0.097f};
        body.local_transform.rotation = Quat::FromAxisAngle(surface_contact ? Vec3{0, 0, 1} : Vec3{1, 2, 3}, 1.05f);
        body.inertial_transform.rotation = Quat::FromAxisAngle({3, 1, 2}, 0.66f);
        if (offset_center) body.inertial_transform.position = {0.025f, -0.01f, 0.015f};
        auto model = CookFreeBody(body, true, surface_contact);
        ASSERT_EQ(model.body_init.size(), 2u);
        const auto capacity = model.capacities;
        const Vec3 incoming{0.2f, -0.1f, -0.2f};
        model.body_init[1].linear_velocity = incoming;
        auto config = Config();
        config.pos_iters = position_iterations;
        config.vel_iters = 32u;
        config.velocity_iterations_override = true;
        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        nuka::sensor::StateSensorDesc contact;
        contact.kind = nuka::sensor::StateSensorKind::ContactWrench;
        contact.mount = nuka::sensor::StateSensorMount::Body;
        contact.index = 1u;
        contact.local_offset = {{0.03f, -0.04f, 0.02f}, Quat::FromAxisAngle({1, 3, -2}, 0.7f)};
        uint32_t body_sensor, plate_sensor;
        ASSERT_EQ(world.AttachStateSensor(contact, &body_sensor), phi::Status::Ok);
        contact.index = 0u;
        ASSERT_EQ(world.AttachStateSensor(contact, &plate_sensor), phi::Status::Ok);
        ASSERT_TRUE(world.Step().AllOk());
        const auto linear = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
        const auto angular = Read<Vec3>(world, nk::FieldId::BodyAngularVelocity, 4u);
        const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 4u);
        const auto rows = Read<nk::NkRow>(world, nk::FieldId::Urows, 2u * capacity.max_rows_per_env);
        const auto impulses = Read<float>(world, nk::FieldId::Lambda, rows.size());
        const auto counts = Read<uint32_t>(world, nk::FieldId::UcontactCount,
                                          2u * capacity.max_contacts_per_env);
        const auto points = Read<Vec3>(world, nk::FieldId::UcontactPoint, 4u * counts.size());
        const auto ranges = Read<uint32_t>(world, nk::FieldId::UcontactRowRange, 2u * counts.size());
        const Vec3 center = body.local_transform.position +
            body.local_transform.rotation.Rotate(body.inertial_transform.position);
        std::vector<float> body_load(12u), plate_load(12u);
        ASSERT_EQ(world.StateSensors().Download(body_sensor, body_load.data(), body_load.size() * sizeof(float)), phi::Status::Ok);
        ASSERT_EQ(world.StateSensors().Download(plate_sensor, plate_load.data(), plate_load.size() * sizeof(float)), phi::Status::Ok);
        for (uint32_t env = 0; env < 2u; ++env) {
            Vec3 impulse{}, torque{}, ground_torque{};
            uint32_t surface_contacts = 0u;
            const auto plate = poses[env * 2u] * contact.local_offset;
            const uint32_t rigid_slots = capacity.max_contacts_per_env - capacity.ogc_contacts_per_env;
            for (uint32_t slot = 0; slot < capacity.max_contacts_per_env; ++slot) {
                const uint32_t contact = env * capacity.max_contacts_per_env + slot;
                for (uint32_t point = 0; point < counts[contact]; ++point) {
                    const bool surface = slot >= rigid_slots && slot - rigid_slots < capacity.ogc_contacts_per_env;
                    surface_contacts += surface;
                    for (uint32_t axis = 0; axis < 3u; ++axis) {
                        const uint32_t row_index = ranges[2u * contact] +
                            axis * ranges[2u * contact + 1u] + point;
                        const auto& row = rows[row_index];
                        for (uint32_t which = 0u; which < 2u; ++which) {
                            const auto& side = which == 0u ? row.a : row.b;
                            const Vec3 point_world = points[4u * contact + point];
                            if (side.kind == nk::kNkSideStatic ||
                                (side.kind == nk::kNkSideRigid && side.index == env * 2u))
                                ground_torque += (point_world - plate.position).Cross(side.jlin * impulses[row_index]);
                            if (side.kind != nk::kNkSideRigid || side.index != env * 2u + 1u) continue;
                            const Vec3 applied = side.jlin * impulses[row_index];
                            impulse += applied;
                            torque += (point_world - center).Cross(applied);
                        }
                    }
                }
            }
            ASSERT_GT(impulse.LengthSq(), 1.0e-6f);
            if (surface_contact) EXPECT_GT(surface_contacts, 0u);
            ExpectVectorNear(linear[env * 2u + 1u], incoming + impulse / body.mass, 2.0e-5f);
            ExpectVectorNear(AngularMomentum(poses[env * 2u + 1u], body.inertial_transform,
                body.inertia, angular[env * 2u + 1u]), torque, 3.0e-6f);
            const auto start = body.local_transform * contact.local_offset;
            const auto end = poses[env * 2u + 1u] * contact.local_offset;
            const auto a = start.rotation, b = end.rotation;
            const auto rotation = Quat{a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z}.Normalized();
            const auto measured = [&](const std::vector<float>& values, uint32_t offset) {
                const uint32_t base = env * 6u + offset;
                return Vec3{values[base], values[base + 1u], values[base + 2u]} * config.dt;
            };
            const auto observed = rotation.Rotate(measured(body_load, 0u));
            const auto moment = rotation.Rotate(measured(body_load, 3u)) +
                ((start.position + end.position) * 0.5f - center).Cross(observed);
            ExpectVectorNear(observed, impulse, 2.0e-6f);
            ExpectVectorNear(moment, torque, 2.0e-6f);
            const auto plate_impulse = plate.rotation.Rotate(measured(plate_load, 0u));
            const auto plate_moment = plate.rotation.Rotate(measured(plate_load, 3u));
            ExpectVectorNear(plate_impulse, -impulse, 2.0e-6f);
            ExpectVectorNear(plate_moment, ground_torque, 2.0e-6f);
            const Vec3 body_momentum = AngularMomentum(poses[env * 2u + 1u], body.inertial_transform,
                body.inertia, angular[env * 2u + 1u]);
            ExpectVectorNear(body_momentum + center.Cross(impulse) + plate_moment +
                plate.position.Cross(plate_impulse), Vec3{}, 3.0e-6f);
        }
      }
     }
    }
}

TEST(FreeRigidDynamics, JointSpringAndArmatureEnergyMatchesStoredState) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool implicit : {false, true}) {
     for (auto type : {scene::JointType::Revolute, scene::JointType::Prismatic, scene::JointType::Spherical}) {
        SCOPED_TRACE(::testing::Message() << "implicit=" << implicit << " type=" << uint32_t(type));
        const bool spherical = type == scene::JointType::Spherical;
        scene::SceneIR source;
        source.SolverOptionsMut().law_source = implicit ? nuka::constraint::LawSource::MuJoCo : nuka::constraint::LawSource::None;
        scene::RigidBodyRecord root, body;
        root.name = "support";
        root.is_static = true;
        body.name = "mass";
        body.mass = 2.0f;
        body.inertia = {0.5f, 0.5f, 0.5f};
        scene::JointRecord joint;
        joint.name = "spring";
        joint.parent_body = source.AddRigidBody(root);
        joint.child_body = source.AddRigidBody(body);
        joint.type = type;
        joint.stiffness = spherical ? 0.0f : 8.0f;
        joint.armature = 0.35f;
        joint.damping = 2.0f;
        joint.spring_reference = -0.1f;
        joint.initial_position = 0.2f;
        source.AddJoint(joint);
        scene::cook::CookToModelOptions options;
        options.enable_contacts = false;
        auto model = scene::cook::CookToModel(source, 1, options).model;
        const auto stiffness = model.articulation.joint_stiffness;
        const auto reference = model.articulation.joint_spring_reference;
        const auto armature = model.articulation.joint_armature;
        const uint32_t links = model.capacities.links_per_env;
        nk::World world(std::move(model), 2u, backend.device, backend.backend, Config());
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        ASSERT_NE(world.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        std::vector<float> speed(2u * links, 0.0f);
        std::vector<Vec3> angular(2u * links);
        for (uint32_t env = 0u; env < 2u; ++env) {
            for (uint32_t link = 0u; link < links; ++link) {
                if (armature[link] == 0.0f) continue;
                const float scale = env == 0u ? 1.0f : 2.0f;
                speed[env * links + link] = spherical ? 0.0f : -0.2f * scale;
                if (spherical) angular[env * links + link] = Vec3{0.2f, 0.3f, -0.4f} * scale;
            }
        }
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::Qdot, speed.data(), speed.size() * sizeof(float)));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::JointAngularVelocity, angular.data(), angular.size() * sizeof(Vec3)));
        const auto before = Read<float>(world, nk::FieldId::Q, 2u * links);
        ASSERT_TRUE(world.Step().AllOk());
        const auto after = Read<float>(world, nk::FieldId::Q, 2u * links);
        const auto end_speed = Read<float>(world, nk::FieldId::Qdot, 2u * links);
        const auto end_angular = Read<Vec3>(world, nk::FieldId::JointAngularVelocity, 2u * links);
        const auto ledger = Read<float>(world, nk::FieldId::EnergyLedger, 2u * nk::kEnergyColumnCount);
        const auto status = Read<uint32_t>(world, nk::FieldId::EnergyLedgerStatus, 2u);
        for (uint32_t env = 0u; env < 2u; ++env) {
            double begin = 0.0, end = 0.0, kinetic_begin = 0.0, kinetic_end = 0.0, passive = 0.0;
            for (uint32_t link = 0u; link < links; ++link) {
                const double x0 = double(before[env * links + link]) - reference[link];
                const double x1 = double(after[env * links + link]) - reference[link];
                begin += 0.5 * stiffness[link] * x0 * x0;
                end += 0.5 * stiffness[link] * x1 * x1;
                if (armature[link] == 0.0f) continue;
                const uint32_t index = env * links + link;
                const double inertia = (type == scene::JointType::Prismatic ? 2.0 : 0.5) + armature[link];
                const double denominator = implicit ? inertia + Config().dt * 2.0 : inertia;
                if (spherical) {
                    const float scale = float((inertia - (implicit ? 0.0 : Config().dt * 2.0)) / denominator);
                    ExpectVectorNear(end_angular[index], angular[index] * scale, 2.0e-6f);
                } else {
                    const double impulse = Config().dt * (stiffness[link] * x0 + (implicit ? 0.0 : 2.0 * speed[index]));
                    EXPECT_NEAR(end_speed[index], (inertia * speed[index] - impulse) / denominator, 2.0e-6);
                }
                kinetic_begin += 0.5 * inertia * (spherical ? angular[index].LengthSq() : double(speed[index]) * speed[index]);
                kinetic_end += 0.5 * inertia * (spherical ? end_angular[index].LengthSq() : double(end_speed[index]) * end_speed[index]);
                passive += 2.0 * (spherical ? Config().dt *
                    (implicit ? end_angular[index] : angular[index]).Dot(end_angular[index]) :
                    (implicit ? double(end_speed[index]) : double(speed[index])) * (double(after[index]) - before[index]));
            }
            if (!spherical) {
                ASSERT_GT(begin, 0.1);
                EXPECT_LT(end, begin);
            }
            EXPECT_EQ(status[env], 0u);
            EXPECT_NEAR(ledger[env * nk::kEnergyColumnCount + uint32_t(nk::EnergyColumn::BeginKinetic)], kinetic_begin, 2e-7);
            EXPECT_NEAR(ledger[env * nk::kEnergyColumnCount + uint32_t(nk::EnergyColumn::EndKinetic)], kinetic_end, 2e-7);
            EXPECT_NEAR(ledger[env * nk::kEnergyColumnCount + uint32_t(nk::EnergyColumn::BeginElastic)], begin, 1e-7);
            EXPECT_NEAR(ledger[env * nk::kEnergyColumnCount + uint32_t(nk::EnergyColumn::EndElastic)], end, 1e-7);
            EXPECT_NEAR(ledger[env * nk::kEnergyColumnCount + uint32_t(nk::EnergyColumn::PassiveLoss)], passive, 1e-7);
        }
     }
    }
}

TEST(FreeRigidDynamics, PairSortingPreservesTheFollowingInertiaField) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (uint32_t environments : {1u, 2u}) {
        for (uint32_t slots : {32u, 512u}) {
            SCOPED_TRACE(::testing::Message() << "envs=" << environments << " slots=" << slots);
            scene::RigidBodyRecord body;
            body.name = "free_body";
            body.mass = 1.0f;
            body.inertia = {0.01f, 0.01f, 0.01f};
            body.local_transform.position = {0, 0, 0.0995f};
            auto model = CookFreeBody(body, true);
            model.capacities.max_contacts_per_env = slots;
            model.capacities.max_rows_per_env = slots * nk::kPairDrivenRowsPerSlot;
            nk::World world(std::move(model), environments, backend.device, backend.backend, Config());
            ASSERT_TRUE(world.Ready());
            const auto required = world.GetModel().capacities.pair_sort_scratch_bytes;
            ASSERT_GT(required, environments);
            uint64_t available = 0u;
            for (const auto& segment : world.GetData().Segments()) {
                if (segment.field == nk::FieldId::PairSortScratch) available = segment.bytes;
            }
            ASSERT_GE(available, required) << "GPU sorting must stay inside its allocated segment";
            for (uint32_t step = 0; step < 3u; ++step) {
                ASSERT_TRUE(world.Step().AllOk());
                const auto inertia = Read<nuka::math::SymmetricMat3>(
                    world, nk::FieldId::BodyWorldInvInertia, environments * 2u);
                for (uint32_t env = 0; env < environments; ++env) {
                    const auto& fixed = inertia[env * 2u];
                    EXPECT_FLOAT_EQ(fixed.xx, 0.0f);
                    EXPECT_FLOAT_EQ(fixed.yy, 0.0f);
                    EXPECT_FLOAT_EQ(fixed.zz, 0.0f);
                    const auto& value = inertia[env * 2u + 1u];
                    EXPECT_NEAR(value.xx, 100.0f, 1.0e-4f);
                    EXPECT_NEAR(value.yy, 100.0f, 1.0e-4f);
                    EXPECT_NEAR(value.zz, 100.0f, 1.0e-4f);
                    EXPECT_NEAR(value.xy, 0.0f, 1.0e-4f);
                    EXPECT_NEAR(value.xz, 0.0f, 1.0e-4f);
                    EXPECT_NEAR(value.yz, 0.0f, 1.0e-4f);
                }
            }
        }
    }
}
