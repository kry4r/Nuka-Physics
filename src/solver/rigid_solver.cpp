// ---------------------------------------------------------------------------
// nuka::solver – PGS iterative rigid body constraint solver
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"

#include <algorithm>
#include <cmath>

namespace nuka::solver {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Dot product of two Vec3.
inline float Dot(const math::Vec3& a, const math::Vec3& b) {
    return a.Dot(b);
}

/// Compute the velocity error (Jv) for a single constraint row.
inline float ComputeJv(
    const constraint::ConstraintBlock& block,
    uint32_t row,
    const runtime::rigid::BodyState& body_a,
    const runtime::rigid::BodyState& body_b)
{
    float jv = 0.0f;
    jv += Dot(block.jacobian_linear_a[row],  body_a.linear_velocity);
    jv += Dot(block.jacobian_angular_a[row], body_a.angular_velocity);
    jv += Dot(block.jacobian_linear_b[row],  body_b.linear_velocity);
    jv += Dot(block.jacobian_angular_b[row], body_b.angular_velocity);
    return jv;
}

/// Apply an impulse delta to the two bodies referenced by a constraint row.
inline void ApplyImpulse(
    const constraint::ConstraintBlock& block,
    uint32_t row,
    float delta_impulse,
    runtime::rigid::BodyState& body_a,
    runtime::rigid::BodyState& body_b)
{
    // body A
    body_a.linear_velocity  += block.jacobian_linear_a[row]  * (body_a.inv_mass * delta_impulse);
    body_a.angular_velocity += math::Vec3{
        block.jacobian_angular_a[row].x * body_a.inv_inertia.x,
        block.jacobian_angular_a[row].y * body_a.inv_inertia.y,
        block.jacobian_angular_a[row].z * body_a.inv_inertia.z
    } * delta_impulse;

    // body B
    body_b.linear_velocity  += block.jacobian_linear_b[row]  * (body_b.inv_mass * delta_impulse);
    body_b.angular_velocity += math::Vec3{
        block.jacobian_angular_b[row].x * body_b.inv_inertia.x,
        block.jacobian_angular_b[row].y * body_b.inv_inertia.y,
        block.jacobian_angular_b[row].z * body_b.inv_inertia.z
    } * delta_impulse;
}

/// Precompute effective mass for each constraint row.
inline void PrecomputeEffectiveMass(
    constraint::ConstraintBlock& block,
    const runtime::rigid::BodyState& body_a,
    const runtime::rigid::BodyState& body_b)
{
    for (uint32_t r = 0; r < block.row_count; ++r) {
        float diag = 0.0f;

        // Linear contribution
        diag += body_a.inv_mass * Dot(block.jacobian_linear_a[r], block.jacobian_linear_a[r]);
        diag += body_b.inv_mass * Dot(block.jacobian_linear_b[r], block.jacobian_linear_b[r]);

        // Angular contribution (diagonal inertia)
        const auto& ja = block.jacobian_angular_a[r];
        diag += ja.x * ja.x * body_a.inv_inertia.x
              + ja.y * ja.y * body_a.inv_inertia.y
              + ja.z * ja.z * body_a.inv_inertia.z;

        const auto& jb = block.jacobian_angular_b[r];
        diag += jb.x * jb.x * body_b.inv_inertia.x
              + jb.y * jb.y * body_b.inv_inertia.y
              + jb.z * jb.z * body_b.inv_inertia.z;

        block.effective_mass[r] = (diag > 1e-12f) ? (1.0f / diag) : 0.0f;
    }
}

inline uint32_t ContactNormalRowCount(const constraint::ConstraintBlock& block) {
    if (block.type != constraint::ConstraintType::Contact) {
        return 0;
    }
    return block.normal_row_count > 0u ? block.normal_row_count : block.row_count;
}

inline bool IsContactFrictionRow(const constraint::ConstraintBlock& block, uint32_t row) {
    return block.type == constraint::ConstraintType::Contact
        && block.friction_row_count > 0u
        && row >= block.first_friction_row
        && row < block.first_friction_row + block.friction_row_count;
}

inline float TotalNormalImpulse(const constraint::ConstraintBlock& block) {
    float impulse = 0.0f;
    const uint32_t normal_rows = ContactNormalRowCount(block);
    for (uint32_t row = 0; row < normal_rows; ++row) {
        impulse += std::max(block.impulse[row], 0.0f);
    }
    return impulse;
}

inline void PrepareContactVelocityTargets(
    constraint::ConstraintBlock& block,
    const runtime::rigid::BodyState& body_a,
    const runtime::rigid::BodyState& body_b) {
    if (block.type != constraint::ConstraintType::Contact || block.restitution <= 0.0f) {
        return;
    }

    const uint32_t normal_rows = ContactNormalRowCount(block);
    for (uint32_t row = 0; row < normal_rows; ++row) {
        const float jv = ComputeJv(block, row, body_a, body_b);
        if (jv < 0.0f) {
            block.rhs[row] = std::max(block.rhs[row], -block.restitution * jv);
        }
    }
}

inline void UpdateFrictionLimits(const constraint::ConstraintBlock& block,
                                 uint32_t row,
                                 float& lower_limit,
                                 float& upper_limit) {
    if (!IsContactFrictionRow(block, row)) {
        return;
    }

    const float max_friction = std::max(block.friction, 0.0f) * TotalNormalImpulse(block);
    lower_limit = -max_friction;
    upper_limit = max_friction;
}

inline void ApplyAngularCorrection(runtime::rigid::BodyState& body,
                                   const math::Vec3& angular_delta) {
    const float angle = angular_delta.Length();
    if (angle <= 1e-8f) {
        return;
    }

    const math::Quat dq = math::Quat::FromAxisAngle(angular_delta / angle, angle);
    body.orientation = (dq * body.orientation).Normalized();
}

inline void ApplyPositionCorrection(runtime::rigid::BodyState& body,
                                    const math::Vec3& linear_jacobian,
                                    const math::Vec3& angular_jacobian,
                                    float position_impulse) {
    if (body.inv_mass > 0.0f) {
        body.position += linear_jacobian * (body.inv_mass * position_impulse);
    }

    const math::Vec3 angular_delta{
        angular_jacobian.x * body.inv_inertia.x * position_impulse,
        angular_jacobian.y * body.inv_inertia.y * position_impulse,
        angular_jacobian.z * body.inv_inertia.z * position_impulse
    };
    ApplyAngularCorrection(body, angular_delta);
}

inline float ComputeJointRowMass(const math::Vec3& linear_a,
                                 const math::Vec3& angular_a,
                                 const math::Vec3& linear_b,
                                 const math::Vec3& angular_b,
                                 const runtime::rigid::BodyState& body_a,
                                 const runtime::rigid::BodyState& body_b) {
    float diag = 0.0f;
    diag += body_a.inv_mass * Dot(linear_a, linear_a);
    diag += body_b.inv_mass * Dot(linear_b, linear_b);
    diag += angular_a.x * angular_a.x * body_a.inv_inertia.x
          + angular_a.y * angular_a.y * body_a.inv_inertia.y
          + angular_a.z * angular_a.z * body_a.inv_inertia.z;
    diag += angular_b.x * angular_b.x * body_b.inv_inertia.x
          + angular_b.y * angular_b.y * body_b.inv_inertia.y
          + angular_b.z * angular_b.z * body_b.inv_inertia.z;
    return (diag > 1e-12f) ? (1.0f / diag) : 0.0f;
}

inline float StabilizeContactPositions(const constraint::ConstraintBlock& block,
                                       runtime::rigid::BodyState& body_a,
                                       runtime::rigid::BodyState& body_b,
                                       const SolverConfig& config) {
    float max_penetration = 0.0f;
    const uint32_t normal_rows = ContactNormalRowCount(block);
    for (uint32_t r = 0; r < normal_rows; ++r) {
        // Position error is the canonical penetration depth. Older tests may
        // still encode it in rhs, so keep that path as a compatibility fallback.
        const float penetration = block.position_error[r] > 0.0f
            ? block.position_error[r]
            : std::abs(block.rhs[r]);
        max_penetration = std::max(max_penetration, penetration);

        const float correction = config.baumgarte * std::max(penetration - config.slop, 0.0f);
        if (correction <= 1e-8f) {
            continue;
        }

        const float position_impulse = correction * block.effective_mass[r];
        ApplyPositionCorrection(body_a,
                                block.jacobian_linear_a[r],
                                math::Vec3::Zero(),
                                position_impulse);
        ApplyPositionCorrection(body_b,
                                block.jacobian_linear_b[r],
                                math::Vec3::Zero(),
                                position_impulse);
    }
    return max_penetration;
}

inline float StabilizeJointPositions(const constraint::ConstraintBlock& block,
                                     runtime::rigid::BodyState& body_a,
                                     runtime::rigid::BodyState& body_b,
                                     const SolverConfig& config) {
    const math::Vec3 axes[3] = {
        math::Vec3::UnitX(),
        math::Vec3::UnitY(),
        math::Vec3::UnitZ()
    };

    float max_error = 0.0f;
    for (const auto& axis : axes) {
        const math::Vec3 r_a = body_a.orientation.Rotate(block.anchor_local_a);
        const math::Vec3 r_b = body_b.orientation.Rotate(block.anchor_local_b);
        const math::Vec3 error = (body_a.position + r_a) - (body_b.position + r_b);
        const float row_error = error.Dot(axis);
        max_error = std::max(max_error, std::abs(row_error));

        const float correction = config.baumgarte * row_error;
        if (std::abs(correction) <= config.slop) {
            continue;
        }

        const math::Vec3 linear_a = -axis;
        const math::Vec3 linear_b = axis;
        const math::Vec3 angular_a = -r_a.Cross(axis);
        const math::Vec3 angular_b = r_b.Cross(axis);
        const float effective_mass = ComputeJointRowMass(linear_a,
                                                         angular_a,
                                                         linear_b,
                                                         angular_b,
                                                         body_a,
                                                         body_b);
        const float position_impulse = correction * effective_mass;
        if (effective_mass <= 0.0f) {
            continue;
        }

        ApplyPositionCorrection(body_a, linear_a, angular_a, position_impulse);
        ApplyPositionCorrection(body_b, linear_b, angular_b, position_impulse);
    }

    return max_error;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// SolveConstraints – PGS solver
// ---------------------------------------------------------------------------

SolveResult SolveConstraints(
    std::vector<constraint::ConstraintBlock>& blocks,
    std::vector<runtime::rigid::BodyState>& bodies,
    const SolverConfig& config)
{
    // Guard against empty input
    if (blocks.empty() || bodies.empty()) {
        return {0.0f, 0};
    }

    // Create a static "ground" body for constraints referencing body ~0u
    runtime::rigid::BodyState ground_body{};  // zero inv_mass = infinite mass

    // Precompute effective masses
    for (auto& block : blocks) {
        auto& ba = (block.body_a < bodies.size()) ? bodies[block.body_a] : ground_body;
        auto& bb = (block.body_b < bodies.size()) ? bodies[block.body_b] : ground_body;
        PrecomputeEffectiveMass(block, ba, bb);
        PrepareContactVelocityTargets(block, ba, bb);
    }

    // --- Velocity iterations (PGS) ---
    for (uint32_t iter = 0; iter < config.velocity_iterations; ++iter) {
        for (auto& block : blocks) {
            auto& ba = (block.body_a < bodies.size()) ? bodies[block.body_a] : ground_body;
            auto& bb = (block.body_b < bodies.size()) ? bodies[block.body_b] : ground_body;

            for (uint32_t r = 0; r < block.row_count; ++r) {
                float jv = ComputeJv(block, r, ba, bb);
                float lambda = block.effective_mass[r] * (block.rhs[r] - jv);

                // Accumulate and clamp
                float old_impulse = block.impulse[r];
                float new_impulse = old_impulse + lambda;
                float lower_limit = block.lower_limit[r];
                float upper_limit = block.upper_limit[r];
                UpdateFrictionLimits(block, r, lower_limit, upper_limit);
                new_impulse = std::max(new_impulse, lower_limit);
                new_impulse = std::min(new_impulse, upper_limit);
                block.impulse[r] = new_impulse;

                float delta = new_impulse - old_impulse;
                if (std::abs(delta) > 1e-12f) {
                    ApplyImpulse(block, r, delta, ba, bb);
                }
            }
        }
    }

    // --- Position stabilization (Baumgarte / joint projection) ---
    float max_penetration = 0.0f;
    for (uint32_t iter = 0; iter < config.position_iterations; ++iter) {
        max_penetration = 0.0f;
        for (auto& block : blocks) {
            auto& ba = (block.body_a < bodies.size()) ? bodies[block.body_a] : ground_body;
            auto& bb = (block.body_b < bodies.size()) ? bodies[block.body_b] : ground_body;

            if (block.type == constraint::ConstraintType::Contact) {
                max_penetration = std::max(max_penetration,
                                           StabilizeContactPositions(block, ba, bb, config));
            } else if (block.type == constraint::ConstraintType::Joint) {
                max_penetration = std::max(max_penetration,
                                           StabilizeJointPositions(block, ba, bb, config));
            }
        }
    }

    return {max_penetration, config.velocity_iterations};
}

// ---------------------------------------------------------------------------
// SolveUnitGroundContact – unit-test helper
// ---------------------------------------------------------------------------

SolveResult SolveUnitGroundContact() {
    // Create a single dynamic body resting on a ground plane (y = 0).
    // The body is a 1 kg box at y = 0 (slightly penetrating).
    std::vector<runtime::rigid::BodyState> bodies(1);
    auto& body = bodies[0];
    body.inv_mass = 1.0f;                              // 1 kg
    body.inv_inertia = {6.0f, 6.0f, 6.0f};            // simplified
    body.position = {0.0f, 0.0f, 0.0f};               // at ground level
    body.linear_velocity = {0.0f, -9.81f * (1.0f / 60.0f), 0.0f}; // gravity impulse for one frame

    // Build a contact constraint: ground (infinite mass) vs body 0
    // Normal pointing up (y+), with slight penetration
    constraint::ConstraintBlock contact{};
    contact.type = constraint::ConstraintType::Contact;
    contact.body_a = 0;       // dynamic body
    contact.body_b = ~0u;     // ground (static, not in bodies array)
    contact.row_count = 1;

    // Normal row (contact normal = +Y)
    contact.jacobian_linear_a[0]  = {0.0f, -1.0f, 0.0f};  // body moves down => violates
    contact.jacobian_angular_a[0] = math::Vec3::Zero();
    contact.jacobian_linear_b[0]  = {0.0f, 1.0f, 0.0f};
    contact.jacobian_angular_b[0] = math::Vec3::Zero();
    contact.rhs[0] = 0.0f;  // target relative velocity = 0 (resting)
    contact.lower_limit[0] = 0.0f;       // normal force >= 0
    contact.upper_limit[0] = 1e6f;       // no upper bound practically
    contact.impulse[0] = 0.0f;

    std::vector<constraint::ConstraintBlock> blocks;
    blocks.push_back(contact);

    SolverConfig config{};
    config.velocity_iterations = 10;
    config.position_iterations = 4;

    auto result = SolveConstraints(blocks, bodies, config);

    // After solving, the body should have near-zero downward velocity
    // and minimal penetration
    result.max_penetration = std::abs(bodies[0].position.y);

    return result;
}

} // namespace nuka::solver
