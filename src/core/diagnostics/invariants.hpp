#pragma once
// ---------------------------------------------------------------------------
// nuka::core::diagnostics - physics invariant monitoring
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cuda_runtime.h>  // cudaStream_t

#include <cstdint>
#include <vector>

namespace nuka::runtime {
struct PhysicsWorld;
struct BuiltWorldState;
struct WorldStepReport;
struct WorldTemplate;
} // namespace nuka::runtime

namespace nuka::core::diagnostics {

enum class Invariant : uint8_t {
    Energy,
    LinearMomentum,
    AngularMomentum,
    ConstraintResidual,
    JointRange,
    LinkLength,
    ParticleCount,
    NanInf,
    VelocityEnvelope,
    PositionEnvelope,
    _Count
};

struct InvariantThresholds {
    float energy_drift_rel = 0.02f;
    float momentum_drift_abs = 1.0e-3f;
    float constraint_residual_abs = 1.0e-3f;
    float joint_range_slop_rad = 1.0e-4f;
    float link_length_drift_rel = 1.0e-3f;
    float velocity_max = 1.0e3f;
    float position_max = 1.0e3f;
};

struct InvariantSample {
    Invariant which = Invariant::Energy;
    uint32_t env_id = 0;
    uint32_t step_index = 0;
    float value = 0.0f;
    float threshold = 0.0f;
    bool violation = false;
};

struct InvariantConfig {
    bool enabled = false;
    uint32_t sample_every_steps = 16;
    InvariantThresholds thresholds = {};
    bool abort_on_nan = true;
    bool trace_violations_only = true;
};

struct InvariantWorldView {
    const runtime::WorldTemplate* world_template = nullptr;
    const runtime::BuiltWorldState* instance = nullptr;
    const runtime::WorldStepReport* step_report = nullptr;
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
};

const char* InvariantName(Invariant invariant);

class InvariantSampler {
public:
    explicit InvariantSampler(const InvariantConfig& cfg);

    void Sample(cudaStream_t stream,
                const runtime::PhysicsWorld& world,
                uint32_t step_index,
                std::vector<InvariantSample>* out_violations);
    void Sample(const InvariantWorldView& view,
                uint32_t step_index,
                std::vector<InvariantSample>* out_violations);
    void Reset();

    const InvariantConfig& Config() const noexcept { return cfg_; }

private:
    InvariantConfig cfg_;
    bool baseline_valid_ = false;
    float baseline_energy_ = 0.0f;
    math::Vec3 baseline_linear_momentum_ = math::Vec3::Zero();
    math::Vec3 baseline_angular_momentum_ = math::Vec3::Zero();
    uint32_t observed_step_count_ = 0;
};

} // namespace nuka::core::diagnostics
