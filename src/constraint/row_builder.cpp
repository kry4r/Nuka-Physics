// ---------------------------------------------------------------------------
// nuka::constraint::row_builder implementation
// ---------------------------------------------------------------------------

#include "constraint/row_builder.hpp"

#include "constraint/solref_solimp.hpp"

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

// ---------------------------------------------------------------------------
// v0.8 C4b: NEW compliant-normal + pyramid-friction emitter. VALIDATED-NOT-WIRED.
// ---------------------------------------------------------------------------
// SPEC DEVIATION (controller + advisor RULED): the C4b spec says "Replace
// AppendContactGroup". It is DELIBERATELY NOT replaced -- the compliance + 4-edge
// pyramid MOVE the contact goldens, and per Appendix C + C4b's own re-baseline
// risk note that re-baseline is scheduled at C5, not C4b. This NEW emitter is
// validated-not-wired (matching the whole spine's "validated-not-wired until C5"
// discipline). AppendContactGroup, both BuildContactRows overloads, and
// runtime/world_stepper.cpp:665 are LEFT BYTE-UNTOUCHED.
//   NAMED DOWNSTREAM CONSUMER = C5: flips world_stepper:665 to this emitter +
//   re-baselines the contact goldens.
//
// PYRAMID STRUCTURE RECONCILIATION: the spec prose ("edges = +-tangent0 +- mu*
// normal-style") is loose/garbled (it names only tangent0, which can't span the
// tangent plane, and "+-mu*normal" coupled into friction Jacobians would
// DOUBLE-COUNT the normal against the mandated SEPARATE normal row and would not
// satisfy the test's "edges sum ~= 0"). The test counts settle it decisively:
// "4 normal + 16 pyramid rows" for a 4-point box can only be the HYBRID
// (separate compliant normal row + pure-tangent friction spokes); a true MuJoCo
// pyramidal CONE (n +- mu*t, no separate normal row) yields 16 rows total, not
// 20, and sums to 4*normal != 0. So the friction Jacobians are unit pure-tangent
// spokes (+t0,-t0,+t1,-t1; fixed order); mu is carried in RowMaterial.friction
// (the cone bound is C5's job), exactly as AppendContactGroup does. The "MuJoCo
// pyramid" reference is about the EDGE COUNT (nmaxpyramid = 2*(condim-1) = 4,
// research-newton-mujocowarp.md:94) and the polygonal-disk friction
// approximation, not about normal-coupling.
void EmitCompliantContactRows(std::span<const ContactManifold> manifolds,
                              const ContactRowComplianceInputs& inputs,
                              RowBuffers* out_rows,
                              std::vector<ContactRowSides>* out_sides) {
    if (out_rows == nullptr || out_sides == nullptr) {
        return;
    }

    // condim=1 -> frictionless; condim>=2 -> 2*(condim-1) pyramid spokes. condim
    // 4/6 (torsional/rolling) is the Q8 v1.0 seam; we honor the formula but the
    // tangent-only spokes here are the v0.8 condim<=3 case. Fixed at world build.
    const uint32_t friction_rows_per_point =
        inputs.condim >= 2u ? 2u * (inputs.condim - 1u) : 0u;

    for (const auto& manifold : manifolds) {
        if (manifold.point_count == 0u) {
            continue;
        }
        const uint32_t point_count =
            std::min(manifold.point_count, ContactManifold::kMaxPoints);

        // Group layout for the 1-normal + N-friction-per-point block (recomputed
        // for the C4b counts -- NOT AppendContactGroup's 2-tangent layout). All
        // normal rows are emitted first, then ALL friction rows, so C5 can locate
        // the friction block by [first_friction_row, +friction_row_count).
        const uint32_t group_first_row = out_rows->RowCount();
        const uint32_t normal_row_count = point_count;
        const uint32_t first_friction_row = group_first_row + normal_row_count;
        const uint32_t total_friction_rows =
            friction_rows_per_point * point_count;

        const uint32_t body_a = manifold.a.handle;
        const uint32_t body_b = manifold.b.handle;

        // --- normal rows (one compliant unilateral row per point) ------------
        for (uint32_t i = 0; i < point_count; ++i) {
            const ContactPoint& point = manifold.points[i];
            const math::Vec3 row_normal = point.normal.Normalized();

            // C4a compliance. pos is SIGNED: the manifold stores penetration as
            // a POSITIVE overlap depth, so pos = -penetration (negative when
            // penetrating). pos_aref == pos_imp == pos for the normal row.
            const float solref[2] = {
                point.solref_timeconst,
                point.solref_dampratio
            };
            const float pos = -point.penetration;
            const CompliantContactRow compliant = ComputeCompliantRow(
                solref,
                manifold.solimp,
                pos,                 // pos_imp
                pos,                 // pos_aref
                inputs.vel,
                inputs.invweight,
                inputs.dt,
                inputs.refsafe);

            RowMaterial material;
            material.kind = RowKind::Contact;
            material.group_id = group_first_row;
            material.normal_row_count = normal_row_count;
            material.first_friction_row = first_friction_row;
            material.friction_row_count = total_friction_rows;
            material.friction = manifold.friction;
            material.restitution = manifold.restitution;
            material.position_error = point.penetration;

            Row row = MakeBaseRow(kMaximalContactRowClassId,
                                  compliant.aref_bias,   // rhs = aref
                                  0.0f,
                                  FLT_MAX,
                                  point.normal_impulse,
                                  row_flags::Unilateral | row_flags::GradActive);
            row.compliance_alpha = compliant.R;          // R = dual regularizer

            out_rows->AddRow(row,
                             {body_a, body_b},
                             MakeJacobian(row_normal),
                             MakeJacobian(-row_normal),
                             material);
            out_sides->push_back({manifold.a, manifold.b});
        }

        // --- pyramid friction rows (pure-tangent spokes, fixed order) --------
        // The 4 (condim=3) spoke directions are +t0,-t0,+t1,-t1 where t0,t1 is
        // the deterministic orthonormal tangent basis of point[0]'s normal. They
        // sum to exactly zero and span the tangent plane symmetrically -> a
        // balanced polygonal (square) approximation of the isotropic Coulomb
        // friction disk. mu is carried in RowMaterial.friction (cone bound = C5).
        if (friction_rows_per_point > 0u) {
            const math::Vec3 base_normal = manifold.points[0].normal.Normalized();
            const math::Vec3 tangent0 = ChooseTangent(base_normal);
            const math::Vec3 tangent1 = base_normal.Cross(tangent0).Normalized();

            for (uint32_t i = 0; i < point_count; ++i) {
                const ContactPoint& point = manifold.points[i];
                // Per-point spoke set. For condim==3 (4 spokes) the FIXED angular
                // order is +t0, -t0, +t1, -t1. For other condims we replicate the
                // same +-t0,+-t1 pattern (v0.8 supports condim<=3; >3 deferred).
                const math::Vec3 spokes[4] = {
                    tangent0, -tangent0, tangent1, -tangent1
                };
                const float warm[4] = {
                    point.friction_impulse_1,
                    point.friction_impulse_1,
                    point.friction_impulse_2,
                    point.friction_impulse_2
                };
                for (uint32_t s = 0; s < friction_rows_per_point; ++s) {
                    const math::Vec3 dir = spokes[s & 3u];
                    RowMaterial material;
                    material.kind = RowKind::Contact;
                    material.group_id = group_first_row;
                    material.normal_row_count = normal_row_count;
                    material.first_friction_row = first_friction_row;
                    material.friction_row_count = total_friction_rows;
                    material.friction = manifold.friction;
                    material.restitution = manifold.restitution;

                    // Friction rows carry NO compliance bias: compliance_alpha=0,
                    // rhs=0 (MakeBaseRow default), direction only (mirrors
                    // AppendContactGroup). Cone bounds are applied by C5.
                    Row row = MakeBaseRow(kMaximalContactRowClassId,
                                          0.0f,
                                          0.0f,
                                          0.0f,
                                          warm[s & 3u],
                                          row_flags::Friction |
                                              row_flags::GradActive);
                    out_rows->AddRow(row,
                                     {body_a, body_b},
                                     MakeJacobian(dir),
                                     MakeJacobian(-dir),
                                     material);
                    out_sides->push_back({manifold.a, manifold.b});
                }
            }
        }
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
