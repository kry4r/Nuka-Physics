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
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
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

class IslandMode {
public:
    explicit IslandMode(bool force_static) {
        const char* previous = std::getenv("NUKA_FORCE_STATIC_ISLANDS");
        had_value_ = previous != nullptr;
        if (previous) previous_ = previous;
        Set(force_static ? "1" : "0");
    }
    ~IslandMode() { Set(had_value_ ? previous_.c_str() : nullptr); }
private:
    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("NUKA_FORCE_STATIC_ISLANDS", value ? value : "");
#else
        if (value) setenv("NUKA_FORCE_STATIC_ISLANDS", value, 1);
        else unsetenv("NUKA_FORCE_STATIC_ISLANDS");
#endif
    }
    bool had_value_ = false;
    std::string previous_;
};

nk::Model CookFreeBody(const scene::RigidBodyRecord& body, bool with_floor) {
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
    source.AddCollisionShape(shape);
    scene::cook::CookToModelOptions options;
    options.enable_contacts = with_floor;
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
        scene::cook::XpbdCookInput input;
        input.positions = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        input.velocities = {{5, -3, 20}, {}, {-2, 4, 12}, {3, 1, -4}};
        input.inv_mass = {1.0f, 0.0f, 2.0f, 5.0f};
        input.aero_triangles = {{{0, 1, 2}}, {{0, 2, 3}}};
        input.aero_drag_normal = coefficient;
        input.aero_drag_tangent = coefficient * 0.1f;
        input.aero_drag_max_dv = 0.5f;
        nk::Model model;
        scene::cook::CookXpbdParticles(model, 3u, input);
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

TEST(ParticleNeighborhood, EnvironmentPoolsMatchBruteForceAndIsolateOverflow) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr uint32_t particles = 40u;
    scene::cook::PbfCookInput input;
    for (uint32_t i = 0u; i < particles; ++i)
        input.positions.push_back({0.001f * i, 0.0f, 0.5f});
    input.velocities.resize(particles);
    input.particle_mass = 0.001f;
    input.rest_density = 1000.0f;
    input.support_radius = 0.1f;
    input.grid_min = {-0.1f, -0.1f, 0.4f};
    input.grid_dims[0] = 80u;
    input.grid_dims[1] = input.grid_dims[2] = 4u;
    for (uint32_t capacity : {particles * (particles - 1u), particles * 20u}) {
        nk::Model model;
        scene::cook::CookPbfParticles(model, 2u, input);
        model.capacities.neighbor_pool_capacity_per_env = capacity;
        nk::World world(std::move(model), 2u, backend.device, backend.backend, Config());
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        auto positions = Read<Vec3>(world, nk::FieldId::ParticlePos, particles * 2u);
        for (uint32_t i = 0u; i < particles; ++i) positions[particles + i].x = 0.2f * i;
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::ParticlePos,
            positions.data(), positions.size() * sizeof(Vec3)));
        phi::ParticleGridBuildParams params{};
        bool found = false;
        for (const auto& call : world.GetPipeline().Calls()) {
            if (call.op == phi::NkOp::ParticleGridBuild) {
                params = *static_cast<const phi::ParticleGridBuildParams*>(call.params);
                found = true;
            }
        }
        ASSERT_TRUE(found);
        params.pos_source = phi::kGridPosSourceParticlePos;
        ASSERT_EQ(world.DispatchOp(phi::NkOp::ParticleGridBuild, &params), phi::Status::Ok);
        const auto attempted = Read<uint32_t>(world, nk::FieldId::GridNeighborAttempted, particles * 2u);
        const auto retained = Read<uint32_t>(world, nk::FieldId::GridNeighborCount, particles * 2u);
        const auto offsets = Read<uint32_t>(world, nk::FieldId::GridNeighborOffset, particles * 2u);
        const auto pool = Read<uint32_t>(world, nk::FieldId::GridNeighborIdx, capacity * 2u);
        const bool complete = capacity == particles * (particles - 1u);
        for (uint32_t i = 0u; i < positions.size(); ++i) {
            std::vector<uint32_t> expected;
            const uint32_t start = i / particles * particles;
            for (uint32_t j = start; j < start + particles; ++j)
                if (i != j && (positions[i] - positions[j]).LengthSq() <= params.query_radius * params.query_radius)
                    expected.push_back(j);
            EXPECT_EQ(attempted[i], expected.size());
            if (complete) EXPECT_EQ(retained[i], expected.size());
            ASSERT_LE(retained[i], expected.size());
            ASSERT_LE(offsets[i] + retained[i], (i / particles + 1u) * capacity);
            for (uint32_t j = 0u; j < retained[i]; ++j) EXPECT_EQ(pool[offsets[i] + j], expected[j]);
        }
        const auto status = Read<uint32_t>(world, nk::FieldId::EnvStatus, 2u);
        EXPECT_EQ(status[0], complete ? 0u : phi::kEnvStatusNeighborOverflow);
        EXPECT_EQ(status[1], 0u);
        ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
        const auto restored = Read<uint32_t>(world, nk::FieldId::GridNeighborAttempted, particles * 2u);
        for (uint32_t i = 0u; i < particles; ++i) {
            EXPECT_EQ(restored[i], attempted[i]);
            EXPECT_EQ(restored[particles + i], 0u);
        }
    }
}

TEST(ParticleNeighborhood, RestAdjacencyPreservesStructureAndKeepsUnrelatedContacts) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr uint32_t count = 6u, envs = 2u;
    nk::Model model;
    auto& p = model.particles;
    auto& cap = model.capacities;
    p.mode = nk::Model::ParticleMode::SoftFluid;
    p.n_soft_particles = count;
    p.initial_pos = {{0, 0, 0.5f}, {0.02f, 0, 0.5f}, {0.2f, 0, 0.5f},
                     {0.22f, 0, 0.5f}, {0.4f, 0, 0.5f}, {0.5f, 0, 0.5f}};
    p.initial_vel.resize(count);
    p.inv_mass = {0, 1, 1, 1, 1, 1};
    p.dist_a = {0u}; p.dist_b = {1u}; p.dist_rest = {0.02f}; p.dist_alpha = {0.0f};
    p.sm_cluster_offset = {0u}; p.sm_cluster_size = {2u}; p.sm_stiffness = {0.5f};
    p.sm_rest_centroid = {{0.45f, 0, 0.5f}};
    p.sm_particles = {4u, 5u}; p.sm_rest_q = {{-0.05f, 0, 0}, {0.05f, 0, 0}}; p.sm_mass = {1, 1};
    p.grid_min = {-0.1f, -0.1f, 0.4f};
    p.grid_dims[0] = 16u; p.grid_dims[1] = p.grid_dims[2] = 4u;
    p.cell_size = p.query_radius = 0.06f;
    p.pp_contact_d_min = 0.03f;
    cap.env_count = envs; cap.particles_per_env = count; cap.dist_cons_per_env = 1u;
    cap.shape_match_slots_per_env = 1u; cap.shape_match_members_per_env = 2u;
    cap.max_grid_cells = p.grid_dims[0] * p.grid_dims[1] * p.grid_dims[2];
    p.dist_b[0] = count;
    std::string reason;
    EXPECT_EQ(model.ValidateTopology(&reason), phi::Status::InvalidArgument);
    p.dist_b[0] = 1u;
    ASSERT_EQ(model.ValidateTopology(&reason), phi::Status::Ok) << reason;
    nk::World world(std::move(model), envs, backend.device, backend.backend, Config());
    ASSERT_TRUE(world.Ready()) << world.CreationError();
    auto positions = Read<Vec3>(world, nk::FieldId::ParticlePos, count * envs);
    const auto initial = positions;
    positions[5].x = 0.42f;
    ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::ParticlePos,
        positions.data(), positions.size() * sizeof(Vec3)));
    // Keep the contact sweep's state transitions and budget; omit material unfolding.
    for (const auto& call : world.GetPipeline().Calls()) {
        switch (call.op) {
            case phi::NkOp::ParticlePredict:
            case phi::NkOp::ParticleGridBuild:
            case phi::NkOp::ParticleParticleContact:
            case phi::NkOp::ParticleProjectionVelocity:
            case phi::NkOp::ParticleFinalize:
                ASSERT_EQ(world.DispatchOp(call.op, call.params), phi::Status::Ok);
                break;
            default:
                break;
        }
    }
    const auto actual = Read<Vec3>(world, nk::FieldId::ParticlePos, count * envs);
    for (uint32_t env = 0u; env < envs; ++env) {
        const uint32_t base = env * count;
        ExpectVectorNear(actual[base], positions[base], 0.0f);
        ExpectVectorNear(actual[base + 1u], positions[base + 1u], 0.0f);
        EXPECT_NEAR(actual[base + 3u].x - actual[base + 2u].x, 0.03f, 1.0e-7f);
        EXPECT_NEAR(actual[base + 3u].x + actual[base + 2u].x,
                    positions[base + 3u].x + positions[base + 2u].x, 1.0e-7f);
    }
    EXPECT_NEAR(actual[5].x - actual[4].x, 0.03f, 1.0e-7f);
    EXPECT_NEAR(actual[5].x + actual[4].x, positions[5].x + positions[4].x, 1.0e-7f);
    ExpectVectorNear(actual[count + 4u], positions[count + 4u], 0.0f);
    ExpectVectorNear(actual[count + 5u], positions[count + 5u], 0.0f);
    ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
    const auto reset = Read<Vec3>(world, nk::FieldId::ParticlePos, count * envs);
    for (uint32_t i = 0u; i < count; ++i) {
        ExpectVectorNear(reset[i], actual[i], 0.0f);
        ExpectVectorNear(reset[count + i], initial[count + i], 0.0f);
    }
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

TEST(FreeRigidDynamics, WorldWrenchesAndGravityAreCovariantAndConsumedOnce) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
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
        const Vec3 gravity = rotation.Rotate({1.7f, -2.3f, -8.1f});
        config.gravity[0] = gravity.x;
        config.gravity[1] = gravity.y;
        config.gravity[2] = gravity.z;
        const Vec3 force = rotation.Rotate({0.7f, -0.9f, 1.1f});
        const Vec3 torque = rotation.Rotate(Vec3{0.2f, -0.3f, 0.1f}.Cross({0.7f, -0.9f, 1.1f}));
        nk::World world(CookFreeBody(body, false), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        const std::vector<Vec3> forces{force, Vec3{}}, torques{torque, Vec3{}};
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyTorque, torques.data(), torques.size() * sizeof(Vec3)));
        ASSERT_TRUE(world.Step().AllOk());
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
        const Vec3 initial_com = body.local_transform.TransformPoint(body.inertial_transform.position);
        for (uint32_t env = 0; env < 2u; ++env)
            ExpectVectorNear(pose[env].TransformPoint(body.inertial_transform.position),
                             initial_com + velocity[env] * config.dt, 1.0e-6f);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), std::vector<Vec3>(2u));
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyTorque, 2u), std::vector<Vec3>(2u));
        ASSERT_TRUE(world.Step().AllOk());
        velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
        ExpectVectorNear(velocity[0], gravity * (2.0f * config.dt) + force * (config.dt / body.mass), 1.0e-6f);
        ExpectVectorNear(velocity[1], gravity * (2.0f * config.dt), 1.0e-6f);
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
        ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), forces);
        ASSERT_EQ(world.Reset({0u}), phi::Status::Ok);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), std::vector<Vec3>(2u));
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
        ASSERT_TRUE(original.Step().AllOk());
        ASSERT_TRUE(transformed.Step().AllOk());
        const uint32_t links = original.GetModel().capacities.links_per_env * 2u;
        const auto qdot = Read<float>(original, nk::FieldId::Qdot, links);
        const auto rotated_qdot = Read<float>(transformed, nk::FieldId::Qdot, links);
        for (uint32_t link = 0; link < links; ++link)
            EXPECT_NEAR(rotated_qdot[link], qdot[link], 2.0e-6f) << "link=" << link;
        const auto velocity = Read<nk::Spatial6>(original, nk::FieldId::LinkVelocity, links);
        const auto rotated_velocity = Read<nk::Spatial6>(transformed, nk::FieldId::LinkVelocity, links);
        for (uint32_t link = 0; link < links; ++link)
            for (uint32_t axis = 0; axis < 6u; ++axis)
                EXPECT_NEAR(rotated_velocity[link].v[axis], velocity[link].v[axis], 2.0e-6f)
                    << "link=" << link << " axis=" << axis;
    }
}

TEST(FreeRigidDynamics, ContactImpulseUsesTheCenterOfMassAndWorldInertia) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool force_static : {false, true}) {
        IslandMode mode(force_static);
        for (bool offset_center : {false, true}) {
          for (uint16_t position_iterations : {0u, 4u}) {
            SCOPED_TRACE(::testing::Message() << "static=" << force_static
                         << " offset=" << offset_center);
            scene::RigidBodyRecord body;
            body.name = "free_body";
            body.mass = 0.5f;
            body.inertia = {0.003f, 0.007f, 0.011f};
            body.local_transform.position = {0, 0, 0.097f};
            body.local_transform.rotation = Quat::FromAxisAngle({1, 2, 3}, 1.05f);
            body.inertial_transform.rotation = Quat::FromAxisAngle({3, 1, 2}, 0.66f);
            if (offset_center) body.inertial_transform.position = {0.025f, -0.01f, 0.015f};
            auto model = CookFreeBody(body, true);
            ASSERT_EQ(model.body_init.size(), 2u);
            const auto capacity = model.capacities;
            const Vec3 incoming{0.2f, -0.1f, -0.2f};
            model.body_init[1].linear_velocity = incoming;
            auto config = Config();
            config.pos_iters = position_iterations;
            nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
            ASSERT_TRUE(world.Ready());
            ASSERT_TRUE(world.Step().AllOk());
            const auto linear = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
            const auto angular = Read<Vec3>(world, nk::FieldId::BodyAngularVelocity, 4u);
            const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 4u);
            const auto rows = Read<nk::NkRow>(world, nk::FieldId::Urows, 2u * capacity.max_rows_per_env);
            const auto impulses = Read<float>(world, nk::FieldId::Lambda, rows.size());
            const auto counts = Read<uint32_t>(world, nk::FieldId::UcontactCount,
                                              2u * capacity.max_contacts_per_env);
            const auto points = Read<Vec3>(world, nk::FieldId::UcontactPoint, 4u * counts.size());
            const Vec3 center = body.local_transform.position +
                body.local_transform.rotation.Rotate(body.inertial_transform.position);
            for (uint32_t env = 0; env < 2u; ++env) {
                Vec3 impulse{}, torque{};
                for (uint32_t slot = 0; slot < capacity.max_contacts_per_env; ++slot) {
                    const uint32_t contact = env * capacity.max_contacts_per_env + slot;
                    for (uint32_t point = 0; point < counts[contact]; ++point) {
                        const Vec3 lever = points[4u * contact + point] - center;
                        for (uint32_t axis = 0; axis < 3u; ++axis) {
                            const uint32_t row_index = env * capacity.max_rows_per_env +
                                slot * nk::kPairDrivenRowsPerSlot +
                                axis * nk::kPairDrivenPtsPerSlot + point;
                            const auto& row = rows[row_index];
                            for (const auto& side : {row.a, row.b}) {
                                if (side.kind != nk::kNkSideRigid || side.index != env * 2u + 1u) continue;
                                const Vec3 applied = side.jlin * impulses[row_index];
                                impulse += applied;
                                torque += lever.Cross(applied);
                            }
                        }
                    }
                }
                ASSERT_GT(impulse.LengthSq(), 1.0e-6f);
                ExpectVectorNear(linear[env * 2u + 1u], incoming + impulse / body.mass, 2.0e-5f);
                ExpectVectorNear(AngularMomentum(poses[env * 2u + 1u], body.inertial_transform,
                    body.inertia, angular[env * 2u + 1u]), torque, 3.0e-6f);
            }
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
