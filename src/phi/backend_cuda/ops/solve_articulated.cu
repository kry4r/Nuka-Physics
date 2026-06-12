// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — TRANSITIONAL M3b ops (THIS FILE IS DELETED IN M4):
//   AssembleRows          — interim impl: per-contact chain Jacobians (normal /
//                           t1 / t2) + effective masses + ArticulatedContactRow
//                           assembly (legacy stages 6/8/9).
//   SolveRowsBlockIsland  — interim impl: the legacy fused block-per-
//                           articulation PGS solve `SolveArticulatedContactRows`
//                           (incl. the implicit joint-damping seed).
//
// LOUD ROUTING NOTE (plan §3.2): the op enum has NO transitional slot, so the
// ported legacy fused solver is routed through the SolveRowsBlockIsland
// pipeline slot with its OWN params POD (phi::SolveArticulatedParams). M4's
// real SolveRowsBlockIsland (device-resident island/color schedule, §3.4)
// replaces BOTH interim implementations and deletes this file + that POD.
//
// KERNEL BODIES ARE LINE-BY-LINE PORTS (D1 byte-exact contract) of
//   src/runtime/articulation/articulation_jacobian.cu (chain Jacobian)
//   src/runtime/articulation/articulation_contacts.cu (m_eff / assemble / solve)
// Input wiring only: jac_* / contact_meff_* / rows / lambda / m_inv / drive_
// damping come from the Data arena fields; the dof column order, loop order,
// iteration count (kContactSolverIterations == 48) and launch shapes are
// UNCHANGED. NOTE the legacy pipeline computes the chain Jacobians BEFORE the
// CRBA (stage 6 vs 7); here they run inside AssembleRows AFTER CrbaFactorM —
// the two stages read/write DISJOINT buffers, so the results are byte-equal.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kInvalidLink = ~0u;

namespace mg = ::nuka::math::gpu;
using mg::Cross;
using mg::Dot;
using mg::NormalizeOrUp;
using mg::Sub;

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

__forceinline__ __device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return mg::Cross(a, b);
}

__forceinline__ __device__ math::Vec3 NormalizeOrUpLocal(math::Vec3 normal) {
    return mg::NormalizeOrUp(normal);
}

__device__ math::Vec3 ScaleVec(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

// articulation_jacobian.cu: RotateByQuat = the defensive-normalize variant.
__forceinline__ __device__ math::Vec3 RotateByQuat(math::Quat q, math::Vec3 v) {
    return mg::RotateByQuatNormalized(q, v);
}

__forceinline__ __device__ uint32_t JointDofCount(ArticulationJointType type) {
    return JointDofCountDevice(type);
}

// One thread per contact. Walks the ancestor-joint chain (contact link -> root)
// in fixed order, writing each ancestor DOF's Jacobian entry into the contact's
// own dof_stride-wide output slice. No atomics, no cross-contact aliasing.
__global__ void ComputeContactChainJacobianKernel(
    ArticulationDeviceState state,
    const uint32_t* contact_link_indices,
    const math::Vec3* contact_point_world,
    const math::Vec3* contact_normal_world,
    uint32_t contact_count,
    uint32_t dof_stride,
    float* out_chain_jacobian) {
    const uint32_t contact = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact >= contact_count) {
        return;
    }

    const uint32_t contact_link = contact_link_indices[contact];
    if (contact_link >= state.total_link_count) {
        return;
    }

    const uint32_t articulation = state.link_to_articulation[contact_link];
    const uint32_t offset = state.articulation_link_offset[articulation];

    const math::Vec3 point = contact_point_world[contact];
    const math::Vec3 normal = NormalizeOrUp(contact_normal_world[contact]);
    float* const out_row = out_chain_jacobian + static_cast<size_t>(contact) * dof_stride;

    // Walk from the contact's link up to the root. parent_link is articulation-
    // local; the root's parent is the ~0u sentinel.
    uint32_t link = contact_link;
    while (link != kInvalidLink) {
        const ArticulationJointType type = state.joint_type[link];
        const uint32_t dof_count = JointDofCount(type);
        if (dof_count != 0u) {
            // dof_index(link): base-inclusive prefix sum of per-joint DOF counts
            // across the articulation's links in [offset, link).
            uint32_t dof_index = 0u;
            for (uint32_t k = offset; k < link; ++k) {
                dof_index += JointDofCount(state.joint_type[k]);
            }

            if (type == ArticulationJointType::FloatingBase) {
                // T8b: floating-base root contributes 6 columns (see
                // articulation_jacobian.cu for the frame contract).
                const math::Quat base_rot = state.base_pose[articulation].rotation;
                const math::Vec3 base_origin = state.base_pose[articulation].position;
                const math::Vec3 lever = Sub(point, base_origin);
                const math::Vec3 ex = RotateByQuat(base_rot, {1.0f, 0.0f, 0.0f});
                const math::Vec3 ey = RotateByQuat(base_rot, {0.0f, 1.0f, 0.0f});
                const math::Vec3 ez = RotateByQuat(base_rot, {0.0f, 0.0f, 1.0f});
                const float ang[3] = {Dot(Cross(ex, lever), normal),
                                      Dot(Cross(ey, lever), normal),
                                      Dot(Cross(ez, lever), normal)};
                const float lin[3] = {Dot(ex, normal), Dot(ey, normal),
                                      Dot(ez, normal)};
                for (uint32_t b = 0u; b < 3u; ++b) {
                    if (dof_index + b < dof_stride) {
                        out_row[dof_index + b] = ang[b];
                    }
                    if (dof_index + 3u + b < dof_stride) {
                        out_row[dof_index + 3u + b] = lin[b];
                    }
                }
            } else {
                const math::Vec3 axis_world =
                    RotateByQuat(state.link_pose[link].rotation, state.joint_axis[link]);
                float entry = 0.0f;
                if (type == ArticulationJointType::Prismatic) {
                    entry = Dot(axis_world, normal);
                } else {  // Revolute
                    const math::Vec3 lever = Sub(point, state.link_pose[link].position);
                    entry = Dot(Cross(axis_world, lever), normal);
                }
                if (dof_index < dof_stride) {
                    out_row[dof_index] = entry;
                }
            }
        }

        const uint32_t parent_local = state.parent_link[link];
        link = (parent_local == kInvalidLink) ? kInvalidLink : (offset + parent_local);
    }
}

// (3) Per-contact effective mass m_eff = 1 / (J M^-1 J^T). One thread / contact.
__global__ void ComputeContactEffectiveMassKernel(ArticulationDeviceState state,
                                                  const uint32_t* contact_link_indices,
                                                  const float* chain_jacobian,
                                                  const float* inertia_M_inv,
                                                  uint32_t contact_count,
                                                  uint32_t dof_stride,
                                                  float* out_effective_mass) {
    const uint32_t contact = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact >= contact_count) {
        return;
    }

    const uint32_t link = contact_link_indices[contact];
    if (link >= state.total_link_count) {
        out_effective_mass[contact] = 0.0f;
        return;
    }

    const uint32_t articulation = state.link_to_articulation[link];
    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;
    const float* const J =
        chain_jacobian + static_cast<size_t>(contact) * dof_stride;

    // denom = J M^-1 J^T. Padding columns of J are zero, so iterating the full
    // stride is safe even though M^-1's padding rows/cols are zero.
    float denom = 0.0f;
    for (uint32_t r = 0u; r < dof_stride; ++r) {
        float row = 0.0f;
        for (uint32_t c = 0u; c < dof_stride; ++c) {
            row += Minv[static_cast<size_t>(r) * dof_stride + c] * J[c];
        }
        denom += J[r] * row;
    }

    out_effective_mass[contact] = 1.0f / fmaxf(denom, kEffectiveMassDenomEpsilon);
}

// Branch-stable orthonormal tangent basis (t1,t2) for a unit normal n.
__device__ void TangentBasis(math::Vec3 n, math::Vec3* t1, math::Vec3* t2) {
    math::Vec3 reference = {1.0f, 0.0f, 0.0f};
    if (fabsf(n.x) > 0.9f) {
        reference = {0.0f, 1.0f, 0.0f};
    }
    const float proj = Dot3(reference, n);
    math::Vec3 tangent_a = {reference.x - proj * n.x,
                            reference.y - proj * n.y,
                            reference.z - proj * n.z};
    const float len_sq = Dot3(tangent_a, tangent_a);
    if (len_sq > 1.0e-12f) {
        tangent_a = ScaleVec(tangent_a, rsqrtf(len_sq));
    } else {
        tangent_a = {1.0f, 0.0f, 0.0f};
    }
    const math::Vec3 tangent_b = Cross3(n, tangent_a);
    *t1 = tangent_a;
    *t2 = tangent_b;
}

// One thread per contact slot. Assembles the per-contact row set (normal + the
// two friction tangents) from the detection + m_eff inputs. Inactive slots get a
// kRowInactive row the solver skips.
__global__ void AssembleArticulatedContactRowsKernel(
    ArticulationDeviceState state,
    const uint32_t* contact_link,
    const math::Vec3* contact_normal,
    const float* contact_depth,
    const float* normal_effective_mass,
    const float* tangent1_effective_mass,
    const float* tangent2_effective_mass,
    uint32_t slot_count,
    ArticulatedContactRow* out_rows) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }

    ArticulatedContactRow row;
    const uint32_t link = contact_link[slot];
    if (link == kInvalidLink || link >= state.total_link_count) {
        row.row_type = ContactRowType::kRowInactive;
        row.articulation = kInvalidLink;
        row.contact_link = kInvalidLink;
        out_rows[slot] = row;
        return;
    }

    const math::Vec3 normal = NormalizeOrUpLocal(contact_normal[slot]);
    math::Vec3 t1;
    math::Vec3 t2;
    TangentBasis(normal, &t1, &t2);

    row.row_type = ContactRowType::kRowNormal;
    row.articulation = state.link_to_articulation[link];
    row.contact_link = link;
    row.effective_mass = normal_effective_mass[slot];
    row.effective_mass_t1 = tangent1_effective_mass[slot];
    row.effective_mass_t2 = tangent2_effective_mass[slot];
    row.depth = contact_depth[slot];
    row.normal = normal;
    row.tangent1 = t1;
    row.tangent2 = t2;
    out_rows[slot] = row;
}

// Fused block-per-articulation PGS solve + apply. One block, single lane. (See
// articulation_contacts.cu for the full algorithm + determinism contract.)
__global__ void SolveArticulatedContactRowsKernel(ArticulationDeviceState state,
                                                  const ArticulatedContactRow* rows,
                                                  const float* normal_jacobian,
                                                  const float* tangent1_jacobian,
                                                  const float* tangent2_jacobian,
                                                  const float* inertia_M_inv,
                                                  uint32_t dof_stride,
                                                  float dt,
                                                  float friction_coefficient,
                                                  float baumgarte_max_velocity,
                                                  float* inout_lambda,
                                                  const float* joint_damping) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];

    // dof_to_link[k] / dof_to_component[k] (see articulation_contacts.cu).
    uint32_t dof_to_link[kMaxArticulationDof];
    uint32_t dof_to_component[kMaxArticulationDof];
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count && dof < kMaxArticulationDof; ++local) {
        const uint32_t link = offset + local;
        const ArticulationJointType type = state.joint_type[link];
        if (local == 0u && state.parent_link[link] == kInvalidLink &&
            type == ArticulationJointType::FloatingBase) {
            for (uint32_t b = 0u; b < 6u && dof < kMaxArticulationDof; ++b) {
                dof_to_link[dof] = link;
                dof_to_component[dof] = b;
                ++dof;
            }
            continue;
        }
        if (JointDofCountDevice(type) != 0u) {
            dof_to_link[dof] = link;
            dof_to_component[dof] = kInvalidLink;
            ++dof;
        }
    }
    if (dof == 0u || dof_stride == 0u) {
        return;
    }

    // Working joint-velocity vector. Base DOFs seed from link_velocity[root].v
    // (the omega-first base spatial velocity); scalar joint DOFs from state.qdot.
    float qdot_work[kMaxArticulationDof];
    for (uint32_t k = 0u; k < dof; ++k) {
        if (dof_to_component[k] != kInvalidLink) {
            qdot_work[k] = state.link_velocity[dof_to_link[k]].v[dof_to_component[k]];
        } else {
            qdot_work[k] = state.qdot[dof_to_link[k]];
        }
    }

    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    // -- General implicit joint viscous damping (deferred from the drive). ----
    // (See articulation_contacts.cu for the backward-Euler derivation.)
    if (joint_damping != nullptr && dt > 0.0f) {
        float c_qdot[kMaxArticulationDof];
        for (uint32_t k = 0u; k < dof; ++k) {
            const float c = (dof_to_component[k] == kInvalidLink)
                                ? joint_damping[dof_to_link[k]]
                                : 0.0f;
            c_qdot[k] = c * qdot_work[k];  // C * qdot_half (qdot_half = seeded qdot_work)
        }
        for (uint32_t r = 0u; r < dof; ++r) {
            float acc = 0.0f;
            const float* const minv_row = Minv + static_cast<size_t>(r) * dof_stride;
            for (uint32_t c = 0u; c < dof; ++c) {
                acc += minv_row[c] * c_qdot[c];
            }
            qdot_work[r] -= dt * acc;
        }
    }

    // This articulation's contact slots (env-major; env == articulation in the
    // 1-articulation-per-env design).
    const uint32_t slot_base = articulation * kMaxFootContactsPerEnv;
    float lambda_n[kMaxFootContactsPerEnv];
    float lambda_t1[kMaxFootContactsPerEnv];
    float lambda_t2[kMaxFootContactsPerEnv];
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        float ln = 0.0f;
        float lt1 = 0.0f;
        float lt2 = 0.0f;
        if (inout_lambda != nullptr) {
            const size_t base = static_cast<size_t>(slot_base + s) * 3u;
            ln = inout_lambda[base + 0u];
            lt1 = inout_lambda[base + 1u];
            lt2 = inout_lambda[base + 2u];
        }
        lambda_n[s] = ln;
        lambda_t1[s] = lt1;
        lambda_t2[s] = lt2;
    }

    const float inv_dt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
    const float mu = fmaxf(friction_coefficient, 0.0f);

    // Apply any warm-started impulse so qdot_work reflects the seed lambda before
    // the first sweep (qdot_work += M^-1 J^T * actual).
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        if (row.row_type != ContactRowType::kRowNormal ||
            row.articulation != articulation) {
            continue;
        }
        const size_t row_base = static_cast<size_t>(slot_base + s) * dof_stride;
        const float* const jrows[3] = {normal_jacobian + row_base,
                                       tangent1_jacobian + row_base,
                                       tangent2_jacobian + row_base};
        const float seeds[3] = {lambda_n[s], lambda_t1[s], lambda_t2[s]};
        for (uint32_t which = 0u; which < 3u; ++which) {
            const float actual = seeds[which];
            if (actual == 0.0f) {
                continue;
            }
            const float* const jac_row = jrows[which];
            for (uint32_t r = 0u; r < dof; ++r) {
                float acc = 0.0f;
                const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                for (uint32_t c = 0u; c < dof; ++c) {
                    acc += minv_row[c] * jac_row[c];
                }
                qdot_work[r] += acc * actual;
            }
        }
    }

    for (uint32_t iter = 0u; iter < kContactSolverIterations; ++iter) {
        // -- Normal rows first, fixed slot order. ---------------------------
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const ArticulatedContactRow row = rows[slot_base + s];
            if (row.row_type != ContactRowType::kRowNormal ||
                row.articulation != articulation) {
                continue;
            }
            const float* const jn =
                normal_jacobian + static_cast<size_t>(slot_base + s) * dof_stride;
            const float bias = fminf(
                kBaumgarteBeta * inv_dt * fmaxf(row.depth - kPenetrationSlop, 0.0f),
                baumgarte_max_velocity);
            float jv = 0.0f;
            for (uint32_t k = 0u; k < dof; ++k) {
                jv += jn[k] * qdot_work[k];
            }
            const float delta = row.effective_mass * (bias - jv);
            const float old_lambda = lambda_n[s];
            // Normal impulse: lower = 0, upper = +inf.
            float new_lambda = old_lambda + delta;
            if (new_lambda < 0.0f) {
                new_lambda = 0.0f;
            }
            const float actual = new_lambda - old_lambda;
            lambda_n[s] = new_lambda;
            if (actual != 0.0f) {
                for (uint32_t r = 0u; r < dof; ++r) {
                    float acc = 0.0f;
                    const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                    for (uint32_t c = 0u; c < dof; ++c) {
                        acc += minv_row[c] * jn[c];
                    }
                    qdot_work[r] += acc * actual;
                }
            }
        }
        // -- Friction tangent rows, fixed slot order. -----------------------
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const ArticulatedContactRow row = rows[slot_base + s];
            if (row.row_type != ContactRowType::kRowNormal ||
                row.articulation != articulation) {
                continue;
            }
            const float limit = mu * fmaxf(lambda_n[s], 0.0f);
            const size_t row_base = static_cast<size_t>(slot_base + s) * dof_stride;
            // Tangent 1 then tangent 2, fixed order.
            for (uint32_t which = 0u; which < 2u; ++which) {
                const float* const jt =
                    (which == 0u ? tangent1_jacobian : tangent2_jacobian) + row_base;
                const float meff = (which == 0u) ? row.effective_mass_t1
                                                 : row.effective_mass_t2;
                float jv = 0.0f;
                for (uint32_t k = 0u; k < dof; ++k) {
                    jv += jt[k] * qdot_work[k];
                }
                const float delta = meff * (0.0f - jv);
                const float old_lambda = (which == 0u) ? lambda_t1[s] : lambda_t2[s];
                float new_lambda = old_lambda + delta;
                new_lambda = fminf(fmaxf(new_lambda, -limit), limit);
                const float actual = new_lambda - old_lambda;
                if (which == 0u) {
                    lambda_t1[s] = new_lambda;
                } else {
                    lambda_t2[s] = new_lambda;
                }
                if (actual != 0.0f) {
                    for (uint32_t r = 0u; r < dof; ++r) {
                        float acc = 0.0f;
                        const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                        for (uint32_t c = 0u; c < dof; ++c) {
                            acc += minv_row[c] * jt[c];
                        }
                        qdot_work[r] += acc * actual;
                    }
                }
            }
        }
    }

    // Final normal-row sweep (see articulation_contacts.cu for the rationale).
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        if (row.row_type != ContactRowType::kRowNormal ||
            row.articulation != articulation) {
            continue;
        }
        const float* const jn =
            normal_jacobian + static_cast<size_t>(slot_base + s) * dof_stride;
        const float bias = fminf(
            kBaumgarteBeta * inv_dt * fmaxf(row.depth - kPenetrationSlop, 0.0f),
            baumgarte_max_velocity);
        float jv = 0.0f;
        for (uint32_t k = 0u; k < dof; ++k) {
            jv += jn[k] * qdot_work[k];
        }
        const float delta = row.effective_mass * (bias - jv);
        const float old_lambda = lambda_n[s];
        float new_lambda = old_lambda + delta;
        if (new_lambda < 0.0f) {
            new_lambda = 0.0f;
        }
        const float actual = new_lambda - old_lambda;
        lambda_n[s] = new_lambda;
        if (actual != 0.0f) {
            for (uint32_t r = 0u; r < dof; ++r) {
                float acc = 0.0f;
                const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                for (uint32_t c = 0u; c < dof; ++c) {
                    acc += minv_row[c] * jn[c];
                }
                qdot_work[r] += acc * actual;
            }
        }
    }

    // Write the corrected velocity back for every DOF of the articulation.
    for (uint32_t k = 0u; k < dof; ++k) {
        if (dof_to_component[k] != kInvalidLink) {
            state.link_velocity[dof_to_link[k]].v[dof_to_component[k]] = qdot_work[k];
        } else {
            state.qdot[dof_to_link[k]] = qdot_work[k];
        }
    }

    // Persist this step's impulses for the next step's warm start.
    if (inout_lambda != nullptr) {
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const size_t base = static_cast<size_t>(slot_base + s) * 3u;
            inout_lambda[base + 0u] = lambda_n[s];
            inout_lambda[base + 1u] = lambda_t1[s];
            inout_lambda[base + 2u] = lambda_t2[s];
        }
    }
}

// --- op entry points ---------------------------------------------------------

Status OpAssembleRows(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const AssembleRowsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->slot_count == 0u || p->max_dof == 0u || p->env_count == 0u) {
        return Status::Ok;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (p->slot_count + kBlockSize - 1u) / kBlockSize;

    // Chain Jacobians (normal, t1, t2) — the legacy launcher zeroes the output
    // first so untouched columns are zero; same here (a stream memset, captured
    // fine by the plan path; NOT an allocation).
    const size_t jac_bytes =
        static_cast<size_t>(p->slot_count) * p->max_dof * sizeof(float);
    if (cudaMemsetAsync(data.jac_normal, 0, jac_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(data.jac_tangent1, 0, jac_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(data.jac_tangent2, 0, jac_bytes, stream) != cudaSuccess) {
        return Status::Failed;
    }
    LaunchCuda(ComputeContactChainJacobianKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const math::Vec3*>(data.contact_point),
               static_cast<const math::Vec3*>(data.contact_normal),
               p->slot_count, p->max_dof, data.jac_normal);
    LaunchCuda(ComputeContactChainJacobianKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const math::Vec3*>(data.contact_point),
               static_cast<const math::Vec3*>(data.contact_tangent1),
               p->slot_count, p->max_dof, data.jac_tangent1);
    LaunchCuda(ComputeContactChainJacobianKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const math::Vec3*>(data.contact_point),
               static_cast<const math::Vec3*>(data.contact_tangent2),
               p->slot_count, p->max_dof, data.jac_tangent2);

    // Effective masses for normal + t1 + t2 rows.
    LaunchCuda(ComputeContactEffectiveMassKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const float*>(data.jac_normal),
               static_cast<const float*>(data.m_inv),
               p->slot_count, p->max_dof, data.contact_meff_normal);
    LaunchCuda(ComputeContactEffectiveMassKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const float*>(data.jac_tangent1),
               static_cast<const float*>(data.m_inv),
               p->slot_count, p->max_dof, data.contact_meff_tangent1);
    LaunchCuda(ComputeContactEffectiveMassKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const float*>(data.jac_tangent2),
               static_cast<const float*>(data.m_inv),
               p->slot_count, p->max_dof, data.contact_meff_tangent2);

    // Assemble the per-slot row set.
    LaunchCuda(AssembleArticulatedContactRowsKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream, state,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const math::Vec3*>(data.contact_normal),
               static_cast<const float*>(data.contact_depth),
               static_cast<const float*>(data.contact_meff_normal),
               static_cast<const float*>(data.contact_meff_tangent1),
               static_cast<const float*>(data.contact_meff_tangent2),
               p->slot_count,
               reinterpret_cast<ArticulatedContactRow*>(data.rows));
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpSolveArticulated(const ModelView& model, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SolveArticulatedParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->env_count == 0u || p->max_dof == 0u) {
        return Status::Ok;
    }
    if (p->max_dof > kMaxArticulationDof) {
        return Status::Failed;  // legacy loud-throw guard.
    }
    // total_link_count is not read by the solve kernel (it walks via the
    // articulation offset tables); 0 keeps the state honest about that.
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, /*total_link_count=*/0u, p->articulation_count);
    const float* joint_damping =
        (p->apply_implicit_damping != 0u) ? data.drive_damping : nullptr;
    LaunchCuda(SolveArticulatedContactRowsKernel, dim3(p->articulation_count),
               dim3(32u), 0u, stream, state,
               reinterpret_cast<const ArticulatedContactRow*>(data.rows),
               static_cast<const float*>(data.jac_normal),
               static_cast<const float*>(data.jac_tangent1),
               static_cast<const float*>(data.jac_tangent2),
               static_cast<const float*>(data.m_inv),
               p->max_dof, p->dt, p->friction_coefficient,
               p->baumgarte_max_velocity, data.lambda, joint_damping);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkSolveArticulatedOps() {
    SetCudaOp(NkOp::AssembleRows, &OpAssembleRows);
    // TRANSITIONAL routing (documented at the top of this file): the legacy
    // fused solver occupies the SolveRowsBlockIsland slot until M4.
    SetCudaOp(NkOp::SolveRowsBlockIsland, &OpSolveArticulated);
}

} // namespace nuka::phi
