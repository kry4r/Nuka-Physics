// ---------------------------------------------------------------------------
// Contact constraint builder
// ---------------------------------------------------------------------------

#include "constraint/contact_builder.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace nuka::constraint {

namespace {

math::Vec3 ChooseTangent(const math::Vec3& normal) {
    if (std::abs(normal.x) < 0.9f) {
        return normal.Cross(math::Vec3::UnitX()).Normalized();
    }
    return normal.Cross(math::Vec3::UnitY()).Normalized();
}

} // namespace

ConstraintBlock BuildContactBlock(uint32_t contact_points) {
    ConstraintBlock block;
    block.type = ConstraintType::Contact;

    // One manifold block stores up to four normal rows plus two shared tangent
    // rows, capped by the six-row block storage.
    block.normal_row_count = std::min(contact_points,
                                      ConstraintBlock::kMaxRows - 2u);
    block.first_friction_row = block.normal_row_count;
    block.friction_row_count = block.normal_row_count > 0u ? 2u : 0u;
    block.row_count = block.normal_row_count + block.friction_row_count;

    // Initialise Jacobians to sensible defaults (unit-Y normal).
    for (uint32_t i = 0; i < block.normal_row_count; ++i) {
        block.jacobian_linear_a[i]  = {0.0f, 1.0f, 0.0f};
        block.jacobian_angular_a[i] = math::Vec3::Zero();
        block.jacobian_linear_b[i]  = {0.0f, -1.0f, 0.0f};
        block.jacobian_angular_b[i] = math::Vec3::Zero();

        block.lower_limit[i] = 0.0f;
        block.upper_limit[i] = FLT_MAX;
    }

    if (block.friction_row_count == 2u) {
        const math::Vec3 tangents[2] = {
            math::Vec3::UnitX(),
            math::Vec3::UnitZ()
        };
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t row = block.first_friction_row + i;
            block.jacobian_linear_a[row]  = tangents[i];
            block.jacobian_angular_a[row] = math::Vec3::Zero();
            block.jacobian_linear_b[row]  = -tangents[i];
            block.jacobian_angular_b[row] = math::Vec3::Zero();
            block.lower_limit[row] = 0.0f;
            block.upper_limit[row] = 0.0f;
        }
    }

    return block;
}

std::vector<ConstraintBlock> BuildContactConstraints(
    const std::vector<ContactManifold>& manifolds) {
    std::vector<ConstraintBlock> blocks;
    blocks.reserve(manifolds.size());

    for (const auto& m : manifolds) {
        if (m.point_count == 0) continue;
        ConstraintBlock block = BuildContactBlock(m.point_count);
        block.body_a = m.body_a;
        block.body_b = m.body_b;
        block.friction = m.friction;
        block.restitution = m.restitution;

        if (m.point_count > 0u) {
            const math::Vec3 normal = m.points[0].normal.Normalized();
            const math::Vec3 tangent0 = ChooseTangent(normal);
            const math::Vec3 tangent1 = normal.Cross(tangent0).Normalized();
            for (uint32_t i = 0; i < block.normal_row_count; ++i) {
                const auto& point = m.points[i];
                const math::Vec3 point_normal = point.normal.Normalized();
                block.jacobian_linear_a[i] = point_normal;
                block.jacobian_linear_b[i] = -point_normal;
                block.rhs[i] = m.restitution;
                block.impulse[i] = point.normal_impulse;
            }
            if (block.friction_row_count == 2u) {
                const uint32_t first = block.first_friction_row;
                block.jacobian_linear_a[first] = tangent0;
                block.jacobian_linear_b[first] = -tangent0;
                block.impulse[first] = m.points[0].friction_impulse_1;
                block.jacobian_linear_a[first + 1u] = tangent1;
                block.jacobian_linear_b[first + 1u] = -tangent1;
                block.impulse[first + 1u] = m.points[0].friction_impulse_2;
            }
        }
        blocks.push_back(block);
    }

    return blocks;
}

} // namespace nuka::constraint
