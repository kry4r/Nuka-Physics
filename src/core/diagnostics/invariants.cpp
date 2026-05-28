// ---------------------------------------------------------------------------
// nuka::core::diagnostics - physics invariant monitoring
// ---------------------------------------------------------------------------

#include "core/diagnostics/invariants.hpp"

#include "runtime/physics_world.hpp"
#include "runtime/world_stepper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nuka::core::diagnostics {

namespace {

float Length(math::Vec3 value) {
    return std::sqrt(value.Dot(value));
}

float AbsMax(math::Vec3 value) {
    return std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
}

bool IsFinite(float value) {
    return std::isfinite(value);
}

bool IsFinite(math::Vec3 value) {
    return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

bool IsFinite(math::Quat value) {
    return IsFinite(value.w) && IsFinite(value.x) &&
           IsFinite(value.y) && IsFinite(value.z);
}

float MassFromInverse(float inverse_mass) {
    return inverse_mass > 0.0f ? 1.0f / inverse_mass : 0.0f;
}

float RelativeDrift(float value, float baseline) {
    const float denom = std::max(std::abs(baseline), 1.0f);
    return std::abs(value - baseline) / denom;
}

struct Snapshot {
    float energy = 0.0f;
    math::Vec3 linear_momentum = math::Vec3::Zero();
    math::Vec3 angular_momentum = math::Vec3::Zero();
    uint32_t nan_inf_count = 0;
    float max_speed = 0.0f;
    float max_position_abs = 0.0f;
    float max_joint_excursion = 0.0f;
};

float JointAngleAroundAxis(const runtime::WorldInstance& instance,
                           const runtime::WorldTemplate& world_template,
                           uint32_t joint_index) {
    if (joint_index >= world_template.joint_table.parent_bodies.size() ||
        joint_index >= world_template.joint_table.child_bodies.size()) {
        return 0.0f;
    }

    const scene::BodyId parent = world_template.joint_table.parent_bodies[joint_index];
    const scene::BodyId child = world_template.joint_table.child_bodies[joint_index];
    if (parent >= instance.poses.size() || child >= instance.poses.size()) {
        return 0.0f;
    }

    const math::Vec3 axis = joint_index < world_template.joint_table.axes.size()
        ? world_template.joint_table.axes[joint_index].Normalized()
        : math::Vec3::UnitZ();
    const math::Quat parent_inv = instance.poses[parent].rotation.Conjugate().Normalized();
    const math::Quat relative = (parent_inv * instance.poses[child].rotation).Normalized();
    const float sin_half = relative.x * axis.x + relative.y * axis.y + relative.z * axis.z;
    return 2.0f * std::atan2(sin_half, relative.w);
}

Snapshot ComputeSnapshot(const InvariantWorldView& view) {
    Snapshot snapshot;
    if (view.world_template == nullptr || view.instance == nullptr) {
        ++snapshot.nan_inf_count;
        return snapshot;
    }

    const auto& world_template = *view.world_template;
    const auto& instance = *view.instance;
    const uint32_t body_count = std::min({world_template.body_count,
                                          instance.body_count,
                                          static_cast<uint32_t>(instance.poses.size())});
    for (uint32_t body_index = 0; body_index < body_count; ++body_index) {
        const math::Transform pose = instance.poses[body_index];
        const math::Vec3 linear_velocity =
            body_index < instance.linear_velocities.size()
                ? instance.linear_velocities[body_index]
                : math::Vec3::Zero();
        const math::Vec3 angular_velocity =
            body_index < instance.angular_velocities.size()
                ? instance.angular_velocities[body_index]
                : math::Vec3::Zero();
        const float inv_mass = body_index < world_template.body_table.inv_masses.size()
            ? world_template.body_table.inv_masses[body_index]
            : 0.0f;
        const math::Vec3 inv_inertia =
            body_index < world_template.body_table.inv_inertias.size()
                ? world_template.body_table.inv_inertias[body_index]
                : math::Vec3::Zero();

        if (!IsFinite(pose.position) || !IsFinite(pose.rotation) ||
            !IsFinite(linear_velocity) || !IsFinite(angular_velocity) ||
            !IsFinite(inv_mass) || !IsFinite(inv_inertia)) {
            ++snapshot.nan_inf_count;
            continue;
        }

        snapshot.max_speed = std::max(snapshot.max_speed, Length(linear_velocity));
        snapshot.max_position_abs =
            std::max(snapshot.max_position_abs, AbsMax(pose.position));

        const float mass = MassFromInverse(inv_mass);
        if (mass <= 0.0f) {
            continue;
        }

        const math::Vec3 linear_momentum = linear_velocity * mass;
        snapshot.linear_momentum += linear_momentum;
        snapshot.angular_momentum += pose.position.Cross(linear_momentum);

        snapshot.energy += 0.5f * mass * linear_velocity.Dot(linear_velocity);
        snapshot.energy += -mass * view.gravity.Dot(pose.position);

        if (inv_inertia.x > 0.0f) {
            snapshot.energy +=
                0.5f * angular_velocity.x * angular_velocity.x / inv_inertia.x;
        }
        if (inv_inertia.y > 0.0f) {
            snapshot.energy +=
                0.5f * angular_velocity.y * angular_velocity.y / inv_inertia.y;
        }
        if (inv_inertia.z > 0.0f) {
            snapshot.energy +=
                0.5f * angular_velocity.z * angular_velocity.z / inv_inertia.z;
        }
    }

    for (uint32_t joint_index = 0; joint_index < world_template.joint_count; ++joint_index) {
        if (joint_index >= world_template.joint_table.types.size()) {
            continue;
        }
        if (world_template.joint_table.types[joint_index] == scene::JointType::Free) {
            continue;
        }
        const float lower = joint_index < world_template.joint_table.lower_limits.size()
            ? world_template.joint_table.lower_limits[joint_index]
            : -std::numeric_limits<float>::max();
        const float upper = joint_index < world_template.joint_table.upper_limits.size()
            ? world_template.joint_table.upper_limits[joint_index]
            : std::numeric_limits<float>::max();
        const float angle = JointAngleAroundAxis(instance, world_template, joint_index);
        const float excursion = std::max(lower - angle, angle - upper);
        snapshot.max_joint_excursion =
            std::max(snapshot.max_joint_excursion, std::max(excursion, 0.0f));
    }

    return snapshot;
}

void AppendSample(const InvariantConfig& config,
                  Invariant which,
                  uint32_t step_index,
                  float value,
                  float threshold,
                  bool violation,
                  std::vector<InvariantSample>* out) {
    if (out == nullptr) {
        return;
    }
    if (config.trace_violations_only && !violation) {
        return;
    }
    out->push_back({which, 0u, step_index, value, threshold, violation});
}

} // namespace

const char* InvariantName(Invariant invariant) {
    switch (invariant) {
    case Invariant::Energy:
        return "energy";
    case Invariant::LinearMomentum:
        return "linear_momentum";
    case Invariant::AngularMomentum:
        return "angular_momentum";
    case Invariant::ConstraintResidual:
        return "constraint_residual";
    case Invariant::JointRange:
        return "joint_range";
    case Invariant::LinkLength:
        return "link_length";
    case Invariant::ParticleCount:
        return "particle_count";
    case Invariant::NanInf:
        return "nan_inf";
    case Invariant::VelocityEnvelope:
        return "velocity_envelope";
    case Invariant::PositionEnvelope:
        return "position_envelope";
    case Invariant::_Count:
        break;
    }
    return "unknown";
}

InvariantSampler::InvariantSampler(const InvariantConfig& cfg) : cfg_(cfg) {
    if (cfg_.sample_every_steps == 0u) {
        cfg_.sample_every_steps = 1u;
    }
}

void InvariantSampler::Sample(const phi::DeviceContext&,
                              const runtime::PhysicsWorld& world,
                              uint32_t step_index,
                              std::vector<InvariantSample>* out_violations) {
    const InvariantWorldView view{
        &world.runtime_world.template_view,
        &world.runtime_world.instance,
        nullptr,
        {0.0f, -9.81f, 0.0f}
    };
    Sample(view, step_index, out_violations);
}

void InvariantSampler::Sample(const InvariantWorldView& view,
                              uint32_t,
                              std::vector<InvariantSample>* out_violations) {
    if (!cfg_.enabled) {
        return;
    }
    const uint32_t observed_step = ++observed_step_count_;
    if (observed_step % cfg_.sample_every_steps != 0u) {
        return;
    }

    const Snapshot snapshot = ComputeSnapshot(view);
    if (!baseline_valid_) {
        baseline_valid_ = true;
        baseline_energy_ = snapshot.energy;
        baseline_linear_momentum_ = snapshot.linear_momentum;
        baseline_angular_momentum_ = snapshot.angular_momentum;
    }

    const float energy_drift = RelativeDrift(snapshot.energy, baseline_energy_);
    AppendSample(cfg_,
                 Invariant::Energy,
                 observed_step,
                 energy_drift,
                 cfg_.thresholds.energy_drift_rel,
                 energy_drift > cfg_.thresholds.energy_drift_rel,
                 out_violations);

    const float linear_momentum_drift =
        Length(snapshot.linear_momentum - baseline_linear_momentum_);
    AppendSample(cfg_,
                 Invariant::LinearMomentum,
                 observed_step,
                 linear_momentum_drift,
                 cfg_.thresholds.momentum_drift_abs,
                 linear_momentum_drift > cfg_.thresholds.momentum_drift_abs,
                 out_violations);

    const float angular_momentum_drift =
        Length(snapshot.angular_momentum - baseline_angular_momentum_);
    AppendSample(cfg_,
                 Invariant::AngularMomentum,
                 observed_step,
                 angular_momentum_drift,
                 cfg_.thresholds.momentum_drift_abs,
                 angular_momentum_drift > cfg_.thresholds.momentum_drift_abs,
                 out_violations);

    const float residual =
        view.step_report != nullptr ? view.step_report->max_constraint_error : 0.0f;
    AppendSample(cfg_,
                 Invariant::ConstraintResidual,
                 observed_step,
                 residual,
                 cfg_.thresholds.constraint_residual_abs,
                 residual > cfg_.thresholds.constraint_residual_abs,
                 out_violations);

    AppendSample(cfg_,
                 Invariant::JointRange,
                 observed_step,
                 snapshot.max_joint_excursion,
                 cfg_.thresholds.joint_range_slop_rad,
                 snapshot.max_joint_excursion > cfg_.thresholds.joint_range_slop_rad,
                 out_violations);

    AppendSample(cfg_,
                 Invariant::NanInf,
                 observed_step,
                 static_cast<float>(snapshot.nan_inf_count),
                 0.0f,
                 snapshot.nan_inf_count > 0u,
                 out_violations);

    AppendSample(cfg_,
                 Invariant::VelocityEnvelope,
                 observed_step,
                 snapshot.max_speed,
                 cfg_.thresholds.velocity_max,
                 snapshot.max_speed > cfg_.thresholds.velocity_max,
                 out_violations);

    AppendSample(cfg_,
                 Invariant::PositionEnvelope,
                 observed_step,
                 snapshot.max_position_abs,
                 cfg_.thresholds.position_max,
                 snapshot.max_position_abs > cfg_.thresholds.position_max,
                 out_violations);
}

void InvariantSampler::Reset() {
    baseline_valid_ = false;
    baseline_energy_ = 0.0f;
    baseline_linear_momentum_ = math::Vec3::Zero();
    baseline_angular_momentum_ = math::Vec3::Zero();
    observed_step_count_ = 0;
}

} // namespace nuka::core::diagnostics
