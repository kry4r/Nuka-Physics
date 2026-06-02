// ---------------------------------------------------------------------------
// nuka::collision::sdf_contact -- device emission kernel + host row builder
// ---------------------------------------------------------------------------
// p08-B FORWARD. Two pieces:
//
//   1. find_sdf_contact_kernel: one thread per CANDIDATE PIECE-PAIR, each runs
//      the header-inline find_sdf_contact_newton (the SAME function the host
//      tests call) on the two pieces' SDF views and writes its SdfContactResult
//      to results[candidate_index]. Fixed candidate order, one slot per
//      candidate, NO atomics, NO cross-thread reduction => D1. The candidate
//      list is an INPUT (built by the broadphase / the test harness); wiring the
//      runtime broadphase -> this kernel is tracked separately (p08-B scope per
//      the entry plan accepts an externally-supplied candidate list).
//
//   2. AppendSdfContactGroup (host): mirrors row_builder.cpp::AppendContactGroup
//      -- for one penetrating witness it emits a RigidSDFContactRow normal row +
//      two friction rows into RowBuffers, in fixed order, so the EXISTING PGS
//      row solver (which dispatches on RowKind::Contact, not row_class_id)
//      resolves them with no solver change. The new row_class_id (4) is metadata
//      for the p08-C adjoint dispatch.
//
// DETERMINISM: the kernel is slot-indexed (no append, no atomics); the host
// emission walks candidates in index order. Two runs are byte-identical.
// ---------------------------------------------------------------------------

#include "collision/sdf_contact.hpp"

#include "constraint/row.hpp"
#include "constraint/row_buffers.hpp"

#include <cfloat>
#include <cmath>

namespace nuka::collision {

// One thread per candidate. Writes results[i] for candidate i. Slot-indexed:
// no atomics, no reduction, fixed order -> D1. Declared in sdf_contact.hpp.
__global__ void find_sdf_contact_kernel(
    const SdfContactCandidate* __restrict__ candidates,
    const runtime::sdf::SparseSdfDevice* __restrict__ sdf_views,
    SdfContactResult* __restrict__ results,
    SdfContactParams params,
    uint32_t candidate_count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidate_count) {
        return;
    }
    const SdfContactCandidate c = candidates[i];
    results[i] = find_sdf_contact_newton(sdf_views[c.sdf_index_a], c.frame_a,
                                         sdf_views[c.sdf_index_b], c.frame_b,
                                         c.initial_guess_world, params);
}

namespace {

math::Vec3 ChooseTangent(const math::Vec3& normal) {
    if (std::abs(normal.x) < 0.9f) {
        return normal.Cross(math::Vec3::UnitX()).Normalized();
    }
    return normal.Cross(math::Vec3::UnitY()).Normalized();
}

} // namespace

uint32_t AppendSdfContactGroup(uint32_t body_a,
                               uint32_t body_b,
                               const SdfContactResult& contact,
                               float friction,
                               float restitution,
                               constraint::RowBuffers* out_rows) {
    using namespace nuka::constraint;
    if (out_rows == nullptr || !contact.valid || !contact.penetrating) {
        return 0u;
    }

    const uint32_t first_row = out_rows->RowCount();
    const uint32_t normal_row_count = 1u;
    const uint32_t first_friction_row = first_row + normal_row_count;
    const uint32_t friction_row_count = 2u;

    const math::Vec3 normal = contact.normal.Normalized();
    const math::Vec3 tangent0 = ChooseTangent(normal);
    const math::Vec3 tangent1 = normal.Cross(tangent0).Normalized();

    auto make_jacobian = [](math::Vec3 linear) {
        RowJacobian6 j;
        j.linear = linear;
        j.angular = math::Vec3::Zero();
        return j;
    };
    auto base_material = [&]() {
        RowMaterial m;
        m.kind = RowKind::Contact;
        m.group_id = first_row;
        m.normal_row_count = normal_row_count;
        m.first_friction_row = first_friction_row;
        m.friction_row_count = friction_row_count;
        m.friction = friction;
        m.restitution = restitution;
        return m;
    };

    // --- normal row (RigidSDFContactRow id 4; solved by the RowKind::Contact
    //     PGS path exactly like MaximalContactRow) -----------------------------
    {
        RowMaterial material = base_material();
        material.position_error = contact.penetration;

        Row row;
        row.row_class_id = kRigidSdfContactRowClassId;
        row.rhs = 0.0f;
        row.lower = 0.0f;
        row.upper = FLT_MAX;
        row.lambda = 0.0f;
        row.flags = row_flags::Unilateral | row_flags::GradActive;
        RowAnchor anchor;
        anchor.local_a = contact.point;  // world witness (anchor carries geometry)
        anchor.local_b = contact.point;
        out_rows->AddRow(row, {body_a, body_b}, make_jacobian(normal),
                         make_jacobian(-normal), material, anchor);
    }

    // --- two friction rows -----------------------------------------------------
    const math::Vec3 tangents[2] = {tangent0, tangent1};
    for (uint32_t t = 0; t < 2u; ++t) {
        RowMaterial material = base_material();
        Row row;
        row.row_class_id = kRigidSdfContactRowClassId;
        row.rhs = 0.0f;
        row.lower = 0.0f;
        row.upper = 0.0f;
        row.lambda = 0.0f;
        row.flags = row_flags::Friction | row_flags::GradActive;
        out_rows->AddRow(row, {body_a, body_b}, make_jacobian(tangents[t]),
                         make_jacobian(-tangents[t]), material);
    }

    return out_rows->RowCount() - first_row;
}

uint32_t AppendFeatherstoneSdfContactGroup(uint32_t maximal_body,
                                           uint32_t articulation_body,
                                           const SdfContactResult& contact,
                                           const math::Vec3& chain_jacobian_linear,
                                           const math::Vec3& chain_jacobian_angular,
                                           float friction,
                                           float restitution,
                                           constraint::RowBuffers* out_rows) {
    using namespace nuka::constraint;
    if (out_rows == nullptr || !contact.valid || !contact.penetrating) {
        return 0u;
    }

    const uint32_t first_row = out_rows->RowCount();
    const math::Vec3 normal = contact.normal.Normalized();

    // FeatherstoneSDFContactRow (id 5): the constraint Jacobian is the chain
    // (articulation) Jacobian EVALUATED AT THE CONTACT POINT and HELD for the
    // row -- this matches the IFT-at-convergence model (the contact geometry is
    // frozen at the converged witness, so the chain J is too). p08-B does the
    // minimal correct thing: it stores the supplied per-body chain-Jacobian
    // rows (linear = J_v contribution along the normal, angular = J_w) on the
    // articulation side and the standard maximal normal on the maximal side.
    RowMaterial material;
    material.kind = RowKind::Contact;
    material.group_id = first_row;
    material.normal_row_count = 1u;
    material.first_friction_row = first_row + 1u;
    material.friction_row_count = 0u;
    material.friction = friction;
    material.restitution = restitution;
    material.position_error = contact.penetration;

    Row row;
    row.row_class_id = kFeatherstoneSdfContactRowClassId;
    row.rhs = 0.0f;
    row.lower = 0.0f;
    row.upper = FLT_MAX;
    row.lambda = 0.0f;
    row.flags = row_flags::Unilateral | row_flags::GradActive;

    RowJacobian6 jac_maximal;
    jac_maximal.linear = normal;
    jac_maximal.angular = math::Vec3::Zero();
    RowJacobian6 jac_chain;
    jac_chain.linear = chain_jacobian_linear;
    jac_chain.angular = chain_jacobian_angular;

    RowAnchor anchor;
    anchor.local_a = contact.point;
    anchor.local_b = contact.point;
    out_rows->AddRow(row, {maximal_body, articulation_body}, jac_maximal,
                     jac_chain, material, anchor);

    return out_rows->RowCount() - first_row;
}

} // namespace nuka::collision
