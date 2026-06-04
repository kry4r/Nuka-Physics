// ---------------------------------------------------------------------------
// nuka::constraint::row_builder implementation
// ---------------------------------------------------------------------------

#include "constraint/row_builder.hpp"

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

Row MakeBaseRow(uint32_t row_class_id,
                float rhs,
                float lower,
                float upper,
                float lambda,
                uint16_t flags) {
    Row row;
    row.row_class_id = row_class_id;
    row.rhs = rhs;
    row.lower = lower;
    row.upper = upper;
    row.lambda = lambda;
    row.flags = flags;
    return row;
}

RowJacobian6 MakeJacobian(math::Vec3 linear, math::Vec3 angular = math::Vec3::Zero()) {
    RowJacobian6 jacobian;
    jacobian.linear = linear;
    jacobian.angular = angular;
    return jacobian;
}

void AppendContactGroup(uint32_t body_a,
                        uint32_t body_b,
                        uint32_t point_count,
                        float friction,
                        float restitution,
                        const ContactPoint* points,
                        RowBuffers* out_rows) {
    if (out_rows == nullptr || point_count == 0u) {
        return;
    }

    const uint32_t normal_row_count = std::min(point_count, 4u);
    const uint32_t first_row = out_rows->RowCount();
    const uint32_t first_friction_row = first_row + normal_row_count;
    const uint32_t friction_row_count = normal_row_count > 0u ? 2u : 0u;

    const math::Vec3 normal = points != nullptr
        ? points[0].normal.Normalized()
        : math::Vec3::UnitY();
    const math::Vec3 tangent0 = ChooseTangent(normal);
    const math::Vec3 tangent1 = normal.Cross(tangent0).Normalized();

    for (uint32_t row_index = 0; row_index < normal_row_count; ++row_index) {
        const ContactPoint& point = points[row_index];
        const math::Vec3 row_normal = point.normal.Normalized();
        RowMaterial material;
        material.kind = RowKind::Contact;
        material.group_id = first_row;
        material.normal_row_count = normal_row_count;
        material.first_friction_row = first_friction_row;
        material.friction_row_count = friction_row_count;
        material.friction = friction;
        material.restitution = restitution;
        material.position_error = point.penetration;

        Row row = MakeBaseRow(kMaximalContactRowClassId,
                              0.0f,
                              0.0f,
                              FLT_MAX,
                              point.normal_impulse,
                              row_flags::Unilateral | row_flags::GradActive);
        out_rows->AddRow(row,
                         {body_a, body_b},
                         MakeJacobian(row_normal),
                         MakeJacobian(-row_normal),
                         material);
    }

    if (friction_row_count == 2u) {
        const math::Vec3 tangents[2] = {tangent0, tangent1};
        const float impulses[2] = {
            points[0].friction_impulse_1,
            points[0].friction_impulse_2
        };
        for (uint32_t i = 0; i < 2u; ++i) {
            RowMaterial material;
            material.kind = RowKind::Contact;
            material.group_id = first_row;
            material.normal_row_count = normal_row_count;
            material.first_friction_row = first_friction_row;
            material.friction_row_count = friction_row_count;
            material.friction = friction;
            material.restitution = restitution;

            Row row = MakeBaseRow(kMaximalContactRowClassId,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  impulses[i],
                                  row_flags::Friction | row_flags::GradActive);
            out_rows->AddRow(row,
                             {body_a, body_b},
                             MakeJacobian(tangents[i]),
                             MakeJacobian(-tangents[i]),
                             material);
        }
    }
}

} // namespace

uint32_t BuildContactRows(uint32_t contact_points, RowBuffers* out_rows) {
    if (out_rows == nullptr || contact_points == 0u) {
        return 0u;
    }
    ContactPoint points[ContactManifold::kMaxPoints] = {};
    const uint32_t point_count = std::min(contact_points, ContactManifold::kMaxPoints);
    for (uint32_t index = 0; index < point_count; ++index) {
        points[index].normal = math::Vec3::UnitY();
    }
    const uint32_t before = out_rows->RowCount();
    AppendContactGroup(kInvalidBodyIndex,
                       kInvalidBodyIndex,
                       point_count,
                       0.5f,
                       0.0f,
                       points,
                       out_rows);
    return out_rows->RowCount() - before;
}

void BuildContactRows(const std::vector<ContactManifold>& manifolds,
                      RowBuffers* out_rows) {
    BuildContactRows(std::span<const ContactManifold>(manifolds.data(),
                                                      manifolds.size()),
                     out_rows);
}

void BuildContactRows(std::span<const ContactManifold> manifolds,
                      RowBuffers* out_rows) {
    if (out_rows == nullptr) {
        return;
    }
    for (const auto& manifold : manifolds) {
        if (manifold.point_count == 0u) {
            continue;
        }
        AppendContactGroup(manifold.a.handle,
                           manifold.b.handle,
                           manifold.point_count,
                           manifold.friction,
                           manifold.restitution,
                           manifold.points,
                           out_rows);
    }
}

void AppendRevoluteRows(RowBuffers* out_rows,
                        uint32_t body_a,
                        uint32_t body_b,
                        math::Vec3 axis,
                        math::Transform parent_frame,
                        math::Transform child_frame,
                        float,
                        float,
                        float) {
    if (out_rows == nullptr) {
        return;
    }
    const math::Vec3 norm_axis = axis.Normalized();
    math::Vec3 perp1;
    if (std::abs(norm_axis.x) < 0.9f) {
        perp1 = norm_axis.Cross(math::Vec3::UnitX()).Normalized();
    } else {
        perp1 = norm_axis.Cross(math::Vec3::UnitY()).Normalized();
    }
    const math::Vec3 perp2 = norm_axis.Cross(perp1).Normalized();
    const math::Vec3 dirs[3] = {
        math::Vec3::UnitX(),
        math::Vec3::UnitY(),
        math::Vec3::UnitZ()
    };
    const math::Vec3 r_a = parent_frame.position;
    const math::Vec3 r_b = child_frame.position;
    RowAnchor anchor;
    anchor.local_a = parent_frame.position;
    anchor.local_b = child_frame.position;

    for (const auto& dir : dirs) {
        Row row = MakeBaseRow(kMaximalJointRowClassId,
                              0.0f,
                              -kRowHugeLimit,
                              kRowHugeLimit,
                              0.0f,
                              row_flags::Equality | row_flags::GradActive);
        RowMaterial material;
        material.kind = RowKind::Joint;
        out_rows->AddRow(row,
                         {body_a, body_b},
                         MakeJacobian(-dir, -(r_a.Cross(dir))),
                         MakeJacobian(dir, r_b.Cross(dir)),
                         material,
                         anchor);
    }

    const math::Vec3 rot_axes[2] = {perp1, perp2};
    for (const auto& rot_axis : rot_axes) {
        Row row = MakeBaseRow(kMaximalJointRowClassId,
                              0.0f,
                              -kRowHugeLimit,
                              kRowHugeLimit,
                              0.0f,
                              row_flags::Equality | row_flags::GradActive);
        RowMaterial material;
        material.kind = RowKind::Joint;
        out_rows->AddRow(row,
                         {body_a, body_b},
                         MakeJacobian(math::Vec3::Zero(), rot_axis),
                         MakeJacobian(math::Vec3::Zero(), -rot_axis),
                         material,
                         anchor);
    }
}

void AppendFixedRows(RowBuffers* out_rows,
                     uint32_t body_a,
                     uint32_t body_b,
                     math::Transform parent_frame,
                     math::Transform child_frame) {
    if (out_rows == nullptr) {
        return;
    }
    AppendRevoluteRows(out_rows,
                       body_a,
                       body_b,
                       math::Vec3::UnitZ(),
                       parent_frame,
                       child_frame,
                       -kRowHugeLimit,
                       kRowHugeLimit,
                       0.0f);

    Row row = MakeBaseRow(kMaximalJointRowClassId,
                          0.0f,
                          -kRowHugeLimit,
                          kRowHugeLimit,
                          0.0f,
                          row_flags::Equality | row_flags::GradActive);
    RowMaterial material;
    material.kind = RowKind::Joint;
    RowAnchor anchor;
    anchor.local_a = parent_frame.position;
    anchor.local_b = child_frame.position;
    out_rows->AddRow(row,
                     {body_a, body_b},
                     MakeJacobian(math::Vec3::Zero(), math::Vec3::UnitZ()),
                     MakeJacobian(math::Vec3::Zero(), -math::Vec3::UnitZ()),
                     material,
                     anchor);
}

void AppendDriveRow(RowBuffers* out_rows,
                    uint32_t body_a,
                    uint32_t body_b,
                    math::Vec3 axis,
                    float target_velocity,
                    float max_force) {
    if (out_rows == nullptr) {
        return;
    }
    const math::Vec3 norm_axis = axis.Normalized();
    const float limit = max_force > 0.0f ? max_force : kRowHugeLimit;
    Row row = MakeBaseRow(kMaximalDriveRowClassId,
                          target_velocity,
                          -limit,
                          limit,
                          0.0f,
                          row_flags::Equality | row_flags::GradActive);
    RowMaterial material;
    material.kind = RowKind::Drive;
    out_rows->AddRow(row,
                     {body_a, body_b},
                     MakeJacobian(math::Vec3::Zero(), norm_axis),
                     MakeJacobian(math::Vec3::Zero(), -norm_axis),
                     material);
}

} // namespace nuka::constraint
