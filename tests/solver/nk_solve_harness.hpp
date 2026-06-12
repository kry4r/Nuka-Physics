#pragma once
// ---------------------------------------------------------------------------
// M4 RE-POINT HARNESS — drive ALREADY-ASSEMBLED compliant contact rows through
// the NEW nk SolveRowsBlockIsland op (the §3.4 device-resident unified solve)
// instead of the legacy UnifiedSolve, so the MJX-parity acceptance layers
// (test_foot_ground_mjx_parity / test_foot_box_mjx_parity) can hold the SAME
// golden gates against the nk path ("oracle 口径重指", plan M4).
//
// The harness converts the test-built legacy inputs (constraint::RowBuffers +
// ContactRowSides + RowArticulationRefs + flat chain-J / M^-1 / qdot) into the
// nk field layout 1:1:
//   * NkRow records (side kinds from sides[].react; rigid jang = the C5c
//     r x j augment from the per-row contact point and the body COM — the
//     value the legacy kernel computed in-kernel each iteration),
//   * per-ROW chain-J rows (re-indexed from chain_jac_slot),
//   * the assembly-hoisted w = M^-1 J^T and m_eff (the SAME inner products,
//     same order, computed host-side here exactly as ops/assemble_rows.cu
//     computes them on device),
//   * a hand-built SolveSchedule::Partition over the rows' real conflict keys
//     (the body_indices the legacy coloring keyed on), written straight into
//     the Model (the schedule harness seam),
// then dispatches ONE SolveRowsBlockIsland through nk::World::DispatchOp and
// reads back qdot / lambdas / body velocities.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "constraint/contact_row_sides.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "nk/solve/schedule.hpp"
#include "runtime/rigid/body_state.hpp"

namespace nk_harness {

struct NkSolveResult {
    std::vector<float> qdot;    // post-solve flat joint velocities
    std::vector<float> lambda;  // post-solve per-row impulses
    bool ok = false;
};

// Solve `rows` on the nk path. `qdot` is the seed (dof floats) for ONE
// articulation (art_index 0); `bodies` (optional) are mutated in place like
// the legacy solve. vel_iters/dt mirror the legacy SolverConfig.
inline NkSolveResult NkSolveRows(
    const nuka::constraint::RowBuffers& rows,
    const std::vector<nuka::constraint::ContactRowSides>& sides,
    const std::vector<nuka::constraint::RowArticulationRefs>& art_refs,
    const std::vector<float>& chain_jacobians,  // [chain_jac_slot * dof]
    const std::vector<float>& minv,             // [dof * dof] (one tile)
    const std::vector<float>& qdot_seed,        // [dof]
    std::vector<nuka::runtime::rigid::BodyState>* bodies,
    uint32_t dof, uint16_t vel_iters, float dt) {
    namespace nk = nuka::nk;
    namespace nphi = nuka::phi;
    namespace constraint = nuka::constraint;
    NkSolveResult out;
    const uint32_t n_rows = rows.RowCount();
    if (n_rows == 0u) return out;
    const uint32_t n_bodies = bodies ? static_cast<uint32_t>(bodies->size()) : 0u;

    // ---- 1. NkRow records + per-row chain-J + w + meff ----------------------
    std::vector<nk::NkRow> nk_rows(n_rows);
    std::vector<float> chainj(static_cast<size_t>(n_rows) * dof, 0.0f);
    std::vector<float> w(static_cast<size_t>(n_rows) * dof, 0.0f);
    std::vector<float> meff(n_rows, 0.0f);
    std::vector<float> lambda(n_rows, 0.0f);
    for (uint32_t r = 0u; r < n_rows; ++r) {
        const constraint::Row& row = rows.rows[r];
        const constraint::RowMaterial& mat = rows.materials[r];
        nk::NkRow& nr = nk_rows[r];
        nr.flags = nk::nk_row_flags::kActive;
        if (row.flags & constraint::row_flags::Friction) {
            nr.flags |= nk::nk_row_flags::kFriction;
        }
        nr.group_first = mat.group_id;
        nr.group_normal_count = mat.normal_row_count;
        nr.env = 0u;
        nr.rhs = row.rhs;
        nr.compliance_alpha = row.compliance_alpha;
        nr.lower = row.lower;
        nr.upper = row.upper;
        nr.mu = mat.friction;
        lambda[r] = row.lambda;
        const auto pair = rows.BodiesForRow(r);
        const uint32_t body_idx[2] = {pair.body_a, pair.body_b};
        nk::NkRowSide* nk_sides[2] = {&nr.a, &nr.b};
        for (int s = 0; s < 2; ++s) {
            const constraint::CollidableRef ref =
                s == 0 ? sides[r].a : sides[r].b;
            const constraint::RowArticulationSide art =
                r < art_refs.size() ? (s == 0 ? art_refs[r].a : art_refs[r].b)
                                    : constraint::RowArticulationSide{};
            nk::NkRowSide& side = *nk_sides[s];
            const constraint::RowJacobian6 j =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(s));
            side.jlin = j.linear;
            side.jang = j.angular;
            switch (ref.react) {
                case constraint::ReactionProviderKind::ArticulationChainJ: {
                    side.kind = nk::kNkSideArtic;
                    side.index = art.art_index != constraint::kInvalidArtIndex
                                     ? art.art_index : 0u;
                    // re-index the chain row by ROW slot.
                    if (art.art_index != constraint::kInvalidArtIndex) {
                        std::memcpy(&chainj[static_cast<size_t>(r) * dof],
                                    &chain_jacobians[static_cast<size_t>(
                                                         art.chain_jac_slot) * dof],
                                    dof * sizeof(float));
                    }
                    break;
                }
                case constraint::ReactionProviderKind::RigidInvMass: {
                    side.kind = nk::kNkSideRigid;
                    side.index = body_idx[s];
                    // the C5c in-kernel angular augment, hoisted: jang =
                    // (contact_point - COM) x jlin.
                    if (bodies != nullptr && body_idx[s] < n_bodies) {
                        const auto& b = (*bodies)[body_idx[s]];
                        const nuka::math::Vec3 rv =
                            sides[r].contact_point - b.position;
                        side.jang = rv.Cross(side.jlin);
                    }
                    break;
                }
                default:
                    side.kind = nk::kNkSideStatic;
                    side.index = ~0u;
                    break;
            }
        }
        // w = M^-1 J^T (the ops/assemble_rows.cu K4a inner product, host-run)
        // + m_eff (K4b: side a + side b + R, recip with the 1e-12 floor).
        float diagonal = 0.0f;
        const nk::NkRowSide* sds[2] = {&nr.a, &nr.b};
        for (int s = 0; s < 2; ++s) {
            const nk::NkRowSide& sd = *sds[s];
            if (sd.kind == nk::kNkSideArtic) {
                const float* J = &chainj[static_cast<size_t>(r) * dof];
                for (uint32_t a = 0u; a < dof; ++a) {
                    float acc = 0.0f;
                    for (uint32_t c = 0u; c < dof; ++c) {
                        acc += minv[static_cast<size_t>(a) * dof + c] * J[c];
                    }
                    w[static_cast<size_t>(r) * dof + a] = acc;
                }
                float denom = 0.0f;
                for (uint32_t a = 0u; a < dof; ++a) {
                    denom += J[a] * w[static_cast<size_t>(r) * dof + a];
                }
                diagonal += denom;
            } else if (sd.kind == nk::kNkSideRigid && bodies != nullptr &&
                       sd.index < n_bodies) {
                const auto& b = (*bodies)[sd.index];
                diagonal += b.inv_mass * sd.jlin.Dot(sd.jlin);
                diagonal += sd.jang.x * sd.jang.x * b.inv_inertia.x;
                diagonal += sd.jang.y * sd.jang.y * b.inv_inertia.y;
                diagonal += sd.jang.z * sd.jang.z * b.inv_inertia.z;
            }
        }
        diagonal += nr.compliance_alpha;
        meff[r] = diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
    }

    // ---- 2. the schedule over the rows' REAL conflict keys ------------------
    std::vector<nk::ScheduleRow> srows(n_rows);
    std::vector<uint32_t> row_env(n_rows, 0u);
    for (uint32_t r = 0u; r < n_rows; ++r) {
        const auto pair = rows.BodiesForRow(r);
        srows[r].key_a = pair.body_a;
        srows[r].key_b = pair.body_b;
        srows[r].group_first = rows.materials[r].group_id;
        if (r < art_refs.size()) {
            if (art_refs[r].a.art_index != constraint::kInvalidArtIndex) {
                srows[r].artic = art_refs[r].a.art_index;
            } else if (art_refs[r].b.art_index != constraint::kInvalidArtIndex) {
                srows[r].artic = art_refs[r].b.art_index;
            }
        }
    }
    nk::SolveScheduleResult sched = nk::SolveSchedule::Partition(srows, row_env);
    if (std::getenv("NK_HARNESS_DEBUG") != nullptr) {
        std::printf("[nk-harness] rows=%u islands=%u segs=%u\n", n_rows,
                    sched.island_count, sched.segment_count);
        for (uint32_t i = 0; i < sched.island_count; ++i) {
            std::printf("[nk-harness] island %u: segs=%u flags=%u env=%u\n", i,
                        sched.islands[4*i+1], sched.islands[4*i+2], sched.islands[4*i+3]);
        }
        for (uint32_t r = 0; r < n_rows; ++r) {
            std::printf("[nk-harness] row %u: aK=%u aI=%u bK=%u bI=%u meff=%g rhs=%g R=%g keyA=%u keyB=%u artic=%u\n",
                        r, nk_rows[r].a.kind, nk_rows[r].a.index, nk_rows[r].b.kind,
                        nk_rows[r].b.index, meff[r], nk_rows[r].rhs,
                        nk_rows[r].compliance_alpha, srows[r].key_a, srows[r].key_b, srows[r].artic);
        }
    }

    // ---- 3. a minimal UnionCsr Model carrying the fields ---------------------
    nk::Model model;
    model.contact_family = nk::ContactFamily::UnionCsr;
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1u;
    cap.dofs_per_env = dof;
    cap.links_per_env = dof;  // identity dof maps (scalar scatter targets)
    cap.bodies_per_env = n_bodies;
    cap.max_contacts_per_env = 1u;
    cap.max_rows_per_env = n_rows;
    model.dof_to_link.resize(dof);
    model.dof_to_component.assign(dof, ~0u);
    for (uint32_t k = 0u; k < dof; ++k) model.dof_to_link[k] = k;
    model.schedule_row_order = std::move(sched.row_order);
    model.schedule_color_segments = std::move(sched.color_segments);
    model.schedule_islands = std::move(sched.islands);
    model.schedule_island_count = sched.island_count;
    model.schedule_segment_count = sched.segment_count;
    if (bodies != nullptr) {
        for (const auto& b : *bodies) {
            nk::Model::BodyInit init;
            init.pose.position = b.position;
            init.pose.rotation = b.orientation;
            init.linear_velocity = b.linear_velocity;
            init.angular_velocity = b.angular_velocity;
            init.inv_mass = b.inv_mass;
            init.inv_inertia = b.inv_inertia;
            model.body_init.push_back(init);
        }
    }

    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) return out;
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (backend == nullptr) return out;
    {
        nk::World world(std::move(model), 1u, dev, backend);
        if (!world.Ready()) {
            nphi::BackendFree(backend);
            return out;
        }
        nk::Data& data = world.GetData();
        bool up = true;
        up = up && data.UploadField(nk::FieldId::Urows, nk_rows.data(),
                                    nk_rows.size() * sizeof(nk::NkRow));
        up = up && data.UploadField(nk::FieldId::Lambda, lambda.data(),
                                    lambda.size() * sizeof(float));
        up = up && data.UploadField(nk::FieldId::ChainJacobian, chainj.data(),
                                    chainj.size() * sizeof(float));
        up = up && data.UploadField(nk::FieldId::RowMinvJt, w.data(),
                                    w.size() * sizeof(float));
        up = up && data.UploadField(nk::FieldId::RowMeff, meff.data(),
                                    meff.size() * sizeof(float));
        up = up && data.UploadField(nk::FieldId::QdotFlat, qdot_seed.data(),
                                    qdot_seed.size() * sizeof(float));
        if (up) {
            nphi::SolveRowsBlockIslandParams p{};
            p.dt = dt;
            p.vel_iters = vel_iters;
            p.pos_iters = 0;
            p.family = nphi::kContactFamilyUnionCsr;
            p.total_islands = world.GetModel().schedule_island_count;
            p.max_dof = dof;
            p.env_count = 1u;
            p.articulation_count = 1u;
            p.rows_per_env = n_rows;
            p.base_link_count = dof;
            p.total_body_count = n_bodies;
            if (world.DispatchOp(nphi::NkOp::SolveRowsBlockIsland, &p) ==
                nphi::Status::Ok) {
                out.qdot.resize(dof);
                out.lambda.resize(n_rows);
                out.ok =
                    data.DownloadField(nk::FieldId::QdotFlat, out.qdot.data(),
                                       dof * sizeof(float)) &&
                    data.DownloadField(nk::FieldId::Lambda, out.lambda.data(),
                                       n_rows * sizeof(float));
                if (out.ok && bodies != nullptr && n_bodies > 0u) {
                    std::vector<nuka::math::Vec3> lin(n_bodies), ang(n_bodies);
                    out.ok = data.DownloadField(nk::FieldId::BodyLinearVelocity,
                                                lin.data(),
                                                n_bodies * sizeof(nuka::math::Vec3)) &&
                             data.DownloadField(nk::FieldId::BodyAngularVelocity,
                                                ang.data(),
                                                n_bodies * sizeof(nuka::math::Vec3));
                    for (uint32_t b = 0u; out.ok && b < n_bodies; ++b) {
                        (*bodies)[b].linear_velocity = lin[b];
                        (*bodies)[b].angular_velocity = ang[b];
                    }
                }
            }
        }
    }
    nphi::BackendFree(backend);
    return out;
}

}  // namespace nk_harness
