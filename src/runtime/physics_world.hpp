#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::PhysicsWorld - physics-facing compiled scene view
// ---------------------------------------------------------------------------

#include "runtime/world_builder.hpp"
#include "scene/cooked_blob.hpp"

#include <cstdint>
#include <memory>

namespace nuka::core::diagnostics {
struct InvariantConfig;
class InvariantSampler;
class TraceSink;
} // namespace nuka::core::diagnostics

namespace nuka::runtime {

struct PhysicsWorld {
    PhysicsWorld();
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld& other);
    PhysicsWorld& operator=(const PhysicsWorld& other);
    PhysicsWorld(PhysicsWorld&& other) noexcept;
    PhysicsWorld& operator=(PhysicsWorld&& other) noexcept;

    uint32_t body_count = 0;
    uint32_t joint_count = 0;
    uint32_t shape_count = 0;
    uint32_t actuator_count = 0;
    uint32_t sensor_count = 0;

    scene::CookedBodyTable body_table;
    scene::CookedJointTable joint_table;
    scene::CookedShapeTable shape_table;
    scene::CookedActuatorTable actuator_table;
    scene::CookedSensorTable sensor_table;

    BuiltWorld runtime_world;

    void ConfigureInvariantSampling(const core::diagnostics::InvariantConfig& config);
    void SetInvariantTraceSink(core::diagnostics::TraceSink* sink);
    core::diagnostics::InvariantSampler* InvariantSampler();
    const core::diagnostics::InvariantSampler* InvariantSampler() const;
    core::diagnostics::TraceSink* InvariantTraceSink() const;

private:
    std::unique_ptr<core::diagnostics::InvariantSampler> invariant_sampler_;
    core::diagnostics::TraceSink* invariant_trace_sink_ = nullptr;
};

PhysicsWorld BuildPhysicsWorld(const scene::CookedBlob& blob);

} // namespace nuka::runtime
