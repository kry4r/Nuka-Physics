// ---------------------------------------------------------------------------
// nuka::solver::DiffTestBridge implementation
// ---------------------------------------------------------------------------

#include "solver/diff_test_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nuka::solver {

namespace {

float Dot(math::Vec3 a, math::Vec3 b) {
    return a.Dot(b);
}

uint32_t BodyForRowBody(const constraint::RowBuffers& rows,
                        const constraint::Row& row,
                        uint32_t local_body_index) {
    if (local_body_index >= row.body_count) {
        return constraint::kInvalidBodyIndex;
    }
    const uint32_t index = row.body_list_offset + local_body_index;
    if (index >= rows.BodyIndexCount()) {
        return constraint::kInvalidBodyIndex;
    }
    return rows.body_indices[index];
}

constraint::RowJacobian6 JacobianForRowBody(const constraint::RowBuffers& rows,
                                            const constraint::Row& row,
                                            uint32_t local_body_index) {
    if (local_body_index >= row.body_count) {
        return {};
    }
    const uint32_t index = row.jacobian_offset + local_body_index;
    if (index >= rows.JacobianDataCount()) {
        return {};
    }
    return rows.jacobian_data[index];
}

bool ValidBody(uint32_t body_index, uint32_t body_count) {
    return body_index != constraint::kInvalidBodyIndex && body_index < body_count;
}

float ComputeJv(const constraint::RowBuffers& rows,
                uint32_t row_index,
                const std::vector<runtime::rigid::BodyState>& bodies) {
    const auto& row = rows.rows[row_index];
    float jv = 0.0f;
    for (uint32_t local = 0u; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, static_cast<uint32_t>(bodies.size()))) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const auto& body = bodies[body_index];
        jv += Dot(jacobian.linear, body.linear_velocity);
        jv += Dot(jacobian.angular, body.angular_velocity);
    }
    return jv;
}

float ComputeEffectiveMass(const constraint::RowBuffers& rows,
                           uint32_t row_index,
                           const std::vector<runtime::rigid::BodyState>& bodies) {
    const auto& row = rows.rows[row_index];
    float diagonal = 0.0f;
    for (uint32_t local = 0u; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, static_cast<uint32_t>(bodies.size()))) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const auto& body = bodies[body_index];
        diagonal += body.inv_mass * Dot(jacobian.linear, jacobian.linear);
        diagonal += jacobian.angular.x * jacobian.angular.x * body.inv_inertia.x;
        diagonal += jacobian.angular.y * jacobian.angular.y * body.inv_inertia.y;
        diagonal += jacobian.angular.z * jacobian.angular.z * body.inv_inertia.z;
    }
    return diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
}

void ApplyVelocityImpulse(const constraint::RowBuffers& rows,
                          uint32_t row_index,
                          float delta_impulse,
                          std::vector<runtime::rigid::BodyState>& bodies) {
    const auto& row = rows.rows[row_index];
    for (uint32_t local = 0u; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, static_cast<uint32_t>(bodies.size()))) {
            continue;
        }
        auto& body = bodies[body_index];
        if (body.inv_mass <= 0.0f) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        body.linear_velocity += jacobian.linear * (body.inv_mass * delta_impulse);
        body.angular_velocity.x += jacobian.angular.x * body.inv_inertia.x * delta_impulse;
        body.angular_velocity.y += jacobian.angular.y * body.inv_inertia.y * delta_impulse;
        body.angular_velocity.z += jacobian.angular.z * body.inv_inertia.z * delta_impulse;
    }
}

void ApplyAngularPositionCorrection(runtime::rigid::BodyState& body,
                                    math::Vec3 angular_jacobian,
                                    float position_impulse) {
    const math::Vec3 angular_delta{
        angular_jacobian.x * body.inv_inertia.x * position_impulse,
        angular_jacobian.y * body.inv_inertia.y * position_impulse,
        angular_jacobian.z * body.inv_inertia.z * position_impulse
    };
    const float angle = angular_delta.Length();
    if (angle <= 1.0e-8f) {
        return;
    }
    const math::Quat dq =
        math::Quat::FromAxisAngle(angular_delta / angle, angle);
    body.orientation = (dq * body.orientation).Normalized();
}

float SolvePositionRow(const constraint::RowBuffers& rows,
                       uint32_t row_index,
                       std::vector<runtime::rigid::BodyState>& bodies,
                       float slop,
                       float baumgarte) {
    const auto& row = rows.rows[row_index];
    const auto& material = rows.materials[row_index];
    float error = 0.0f;
    bool apply_angular = false;
    math::Vec3 angular_a = math::Vec3::Zero();
    math::Vec3 angular_b = math::Vec3::Zero();

    if (constraint::IsContactNormalRow(rows, row_index)) {
        error = std::max(material.position_error, 0.0f);
    } else if (material.kind == constraint::RowKind::Joint) {
        const uint32_t body_a = BodyForRowBody(rows, row, 0u);
        const uint32_t body_b = BodyForRowBody(rows, row, 1u);
        const auto jacobian_a = JacobianForRowBody(rows, row, 0u);
        const auto jacobian_b = JacobianForRowBody(rows, row, 1u);
        math::Vec3 axis = jacobian_b.linear;
        if (axis.Length() <= 1.0e-8f) {
            axis = -jacobian_a.linear;
        }
        if (axis.Length() <= 1.0e-8f) {
            return 0.0f;
        }

        math::Vec3 position_a = math::Vec3::Zero();
        math::Vec3 position_b = math::Vec3::Zero();
        math::Quat orientation_a = math::Quat::Identity();
        math::Quat orientation_b = math::Quat::Identity();
        if (ValidBody(body_a, static_cast<uint32_t>(bodies.size()))) {
            position_a = bodies[body_a].position;
            orientation_a = bodies[body_a].orientation;
        }
        if (ValidBody(body_b, static_cast<uint32_t>(bodies.size()))) {
            position_b = bodies[body_b].position;
            orientation_b = bodies[body_b].orientation;
        }
        const auto& anchor = rows.anchors[row_index];
        const math::Vec3 r_a = orientation_a.Rotate(anchor.local_a);
        const math::Vec3 r_b = orientation_b.Rotate(anchor.local_b);
        error = ((position_a + r_a) - (position_b + r_b)).Dot(axis);
        angular_a = -(r_a.Cross(axis));
        angular_b = r_b.Cross(axis);
        apply_angular = true;
    } else {
        return 0.0f;
    }

    const float correction = baumgarte *
        (error >= 0.0f ? std::max(error - slop, 0.0f) : std::min(error + slop, 0.0f));
    if (std::abs(correction) <= 1.0e-8f) {
        return std::abs(error);
    }

    const float effective_mass = ComputeEffectiveMass(rows, row_index, bodies);
    const float position_impulse = correction * effective_mass;
    for (uint32_t local = 0u; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, static_cast<uint32_t>(bodies.size()))) {
            continue;
        }
        auto& body = bodies[body_index];
        if (body.inv_mass > 0.0f) {
            const auto jacobian = JacobianForRowBody(rows, row, local);
            body.position += jacobian.linear * (body.inv_mass * position_impulse);
        }
        if (apply_angular) {
            ApplyAngularPositionCorrection(body,
                                           local == 0u ? angular_a : angular_b,
                                           position_impulse);
        }
    }
    return std::abs(error);
}

void PrepareVelocityTargets(constraint::RowBuffers& rows,
                            const std::vector<runtime::rigid::BodyState>& bodies) {
    for (uint32_t row_index = 0u; row_index < rows.RowCount(); ++row_index) {
        if (!constraint::IsContactNormalRow(rows, row_index) ||
            rows.materials[row_index].restitution <= 0.0f) {
            continue;
        }
        const float jv = ComputeJv(rows, row_index, bodies);
        if (jv < 0.0f) {
            rows.rows[row_index].rhs =
                std::max(rows.rows[row_index].rhs,
                         -rows.materials[row_index].restitution * jv);
        }
    }
}

void SolveRowsReference(constraint::RowBuffers& rows,
                        std::vector<runtime::rigid::BodyState>& bodies,
                        const gpu::RowSolveConfig& config) {
    PrepareVelocityTargets(rows, bodies);
    for (uint32_t iter = 0u; iter < config.velocity_iterations; ++iter) {
        for (uint32_t row_index = 0u; row_index < rows.RowCount(); ++row_index) {
            auto& row = rows.rows[row_index];
            const float effective_mass = ComputeEffectiveMass(rows, row_index, bodies);
            const float lambda = effective_mass * (row.rhs - ComputeJv(rows, row_index, bodies));
            const float old_impulse = row.lambda;
            float lower = row.lower;
            float upper = row.upper;
            if (constraint::IsFrictionRow(rows, row_index)) {
                const float friction_limit =
                    std::max(rows.materials[row_index].friction, 0.0f) *
                    constraint::TotalNormalLambda(rows, row_index);
                lower = -friction_limit;
                upper = friction_limit;
            }
            row.lambda = std::clamp(old_impulse + lambda, lower, upper);
            const float delta = row.lambda - old_impulse;
            if (std::abs(delta) > 1.0e-12f) {
                ApplyVelocityImpulse(rows, row_index, delta, bodies);
            }
        }
    }

    for (uint32_t iter = 0u; iter < config.position_iterations; ++iter) {
        for (uint32_t row_index = 0u; row_index < rows.RowCount(); ++row_index) {
            (void)SolvePositionRow(rows,
                                   row_index,
                                   bodies,
                                   config.slop,
                                   config.baumgarte);
        }
    }
}

float PositionError(const runtime::rigid::BodyState& lhs,
                    const runtime::rigid::BodyState& rhs) {
    return (lhs.position - rhs.position).Length();
}

float VelocityError(const runtime::rigid::BodyState& lhs,
                    const runtime::rigid::BodyState& rhs) {
    return std::max((lhs.linear_velocity - rhs.linear_velocity).Length(),
                    (lhs.angular_velocity - rhs.angular_velocity).Length());
}

} // namespace

DiffTestBridge::DiffTestBridge(const phi::DeviceContext& context)
    : context_(context) {}

gpu::RowSolveReport DiffTestBridge::SolveAndCompare(
    const constraint::RowBuffers& input_rows,
    std::vector<runtime::rigid::BodyState>& bodies,
    const gpu::RowSolveConfig& config,
    float tolerance_position,
    float tolerance_velocity) {
    auto cuda_rows = input_rows;
    auto reference_rows = input_rows;
    auto cuda_bodies = bodies;
    auto reference_bodies = bodies;

    const auto report = gpu::SolveRows(context_, cuda_rows, cuda_bodies, config);
    SolveRowsReference(reference_rows, reference_bodies, config);

    last_divergence_.clear();
    for (uint32_t body_index = 0u;
         body_index < static_cast<uint32_t>(cuda_bodies.size());
         ++body_index) {
        const float position_error =
            PositionError(cuda_bodies[body_index], reference_bodies[body_index]);
        const float velocity_error =
            VelocityError(cuda_bodies[body_index], reference_bodies[body_index]);
        if (position_error > tolerance_position ||
            velocity_error > tolerance_velocity) {
            last_divergence_.push_back({body_index, position_error, velocity_error});
        }
    }

    if (!last_divergence_.empty()) {
        throw std::runtime_error("CUDA RowSolver diverged from validation reference replay");
    }

    bodies = std::move(cuda_bodies);
    return report;
}

} // namespace nuka::solver
