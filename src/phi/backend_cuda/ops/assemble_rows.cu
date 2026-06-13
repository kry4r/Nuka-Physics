// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M4 AssembleRows (the §3.4 row-assembly op).
//
// TWO contact families behind the ONE op (params->family):
//
//   FUSED (kContactFamilyFusedFoot) — the M3 articulation foot pipeline,
//   MOVED VERBATIM from the deleted transitional ops/solve_articulated.cu:
//   per-contact chain Jacobians (normal/t1/t2) + effective masses +
//   ArticulatedContactRow assembly. Kernel bodies are the same line-by-line
//   ports of articulation_jacobian.cu / articulation_contacts.cu (D1
//   byte-exact contract unchanged — the foot goldens pin it).
//
//   UNION (kContactFamilyUnionCsr) — the legacy coresident union world's row
//   assembly, device-resident on FIXED row slots:
//     K1 PackQdotFlat        — the per-env flat prefix-sum qdot pack (the
//                              legacy host qdot[] pack loop 1:1 via the cooked
//                              dof_to_link/dof_to_component maps).
//     K2 EmitUnionRows       — EmitCompliantContactRows' math per (env x
//                              slot): solref/solimp aref + R via the SAME
//                              constraint/solref_solimp.hpp HD header
//                              (vel = 0, invweight = 1 — the union world's
//                              exact inputs), ChooseTangent spoke basis,
//                              normals-then-spokes group layout, the C5c
//                              r x j rigid angular augment, λ cold-start
//                              (the legacy union rebuilt rows from zeroed
//                              manifold impulses every step — warm-start
//                              stays mechanism-present via the persistent
//                              lambda field, zeroed here to match legacy).
//     K3 chain Jacobians     — the SAME ComputeContactChainJacobianKernel,
//                              launched over ROW slots (link/point/dir
//                              gathered per row by K2), out chain_jacobian.
//     K4a row_minv_jt        — w = M^-1 J^T per articulation row (the
//                              ArticulationApplyImpulse inner product, hoisted
//                              to assembly: same inner loop, same order,
//                              computed once instead of per iteration).
//     K4b row_meff           — ComputeCompliantEffectiveMass hoisted to
//                              assembly (constant across iterations): side-a
//                              + side-b J M^-1 J^T + R, recip with the legacy
//                              1e-12 floor.
//   Entirely arena-resident; ZERO host participation; every launch shape is a
//   fixed function of the capacities -> CUDA-graph capturable.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "constraint/solref_solimp.hpp"  // ComputeCompliantRow (HD)
#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kFltMax = 3.402823466e+38f;

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

// ===========================================================================
// SHARED + FUSED-family kernels — MOVED VERBATIM from the transitional
// ops/solve_articulated.cu (M3b), which this file replaces. Bodies are
// line-by-line ports of articulation_jacobian.cu / articulation_contacts.cu.
// ===========================================================================

// One thread per contact. Walks the ancestor-joint chain (contact link -> root)
// in fixed order, writing each ancestor DOF's Jacobian entry into the contact's
// own dof_stride-wide output slice. No atomics, no cross-contact aliasing.
// (The union family launches this over ROW slots — same kernel, the per-slot
// inputs are gathered per row; an inactive row carries the kInvalidLink
// sentinel and is skipped, leaving its memset-zero J row.)
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

// ===========================================================================
// UNION-family kernels (M4 NEW).
// ===========================================================================

// Vec3 normalize — EXACT host math::Vec3::Normalized expression (len = sqrt,
// 1e-12 guard, component-wise division), so the emitted row directions match
// the legacy host EmitCompliantContactRows at the FP floor. (NOT the fused
// family's NormalizeOrUp — a DIFFERENT legacy function.)
__forceinline__ __device__ math::Vec3 NormalizedHostExpr(math::Vec3 v) {
    const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-12f) return {0.0f, 0.0f, 0.0f};
    return {v.x / len, v.y / len, v.z / len};
}

// row_builder.cpp ChooseTangent, host expressions verbatim.
__forceinline__ __device__ math::Vec3 ChooseTangentDev(math::Vec3 n) {
    if (fabsf(n.x) < 0.9f) {
        return NormalizedHostExpr(n.Cross(math::Vec3{1.0f, 0.0f, 0.0f}));
    }
    return NormalizedHostExpr(n.Cross(math::Vec3{0.0f, 1.0f, 0.0f}));
}

// K1: per-(env x dof) flat prefix-sum qdot pack — the legacy host pack loop
// (link_velocity[root].v[i] for base DOFs, qdot[link] for scalar joints) via
// the cooked dof maps.
__global__ void PackQdotFlatKernel(const Spatial6* __restrict__ link_velocity,
                                   const float* __restrict__ qdot,
                                   const uint32_t* __restrict__ dof_to_link,
                                   const uint32_t* __restrict__ dof_to_component,
                                   uint32_t env_count,
                                   uint32_t dof_stride,
                                   uint32_t base_link_count,
                                   float* __restrict__ qdot_flat) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * dof_stride;
    if (gid >= total) return;
    const uint32_t env = gid / dof_stride;
    const uint32_t k = gid - env * dof_stride;
    const uint32_t link = dof_to_link[gid];
    const uint32_t comp = dof_to_component[gid];
    const size_t gl = static_cast<size_t>(env) * base_link_count + link;
    qdot_flat[gid] = (comp != ~0u) ? link_velocity[gl].v[comp] : qdot[gl];
    (void)k;
}

// K2: one thread per (env x union slot) — EmitCompliantContactRows onto the
// slot's FIXED row span. Worst-case layout: max_pts normal rows first, then
// per point 2*(condim-1) spokes (+t0,-t0,+t1,-t1) — the emitter's exact
// per-manifold order on fixed slots; inactive points/slots write zeroed
// records (flags = 0) + lambda = 0 + the kInvalidLink chain-J sentinel.
__global__ void EmitUnionRowsKernel(const float* __restrict__ union_slots,
                                    const uint32_t* __restrict__ ucontact_count,
                                    const math::Vec3* __restrict__ ucontact_point,
                                    const math::Vec3* __restrict__ ucontact_normal,
                                    const float* __restrict__ ucontact_depth,
                                    const math::Transform* __restrict__ body_pose,
                                    float solref0, float solref1,
                                    float solimp0, float solimp1, float solimp2,
                                    float solimp3, float solimp4,
                                    float dt,
                                    uint32_t env_count,
                                    uint32_t slot_count,
                                    uint32_t rows_per_env,
                                    uint32_t bodies_per_env,
                                    uint32_t base_link_count,
                                    uint32_t particles_per_env,
                                    NkRow* __restrict__ urows,
                                    float* __restrict__ lambda,
                                    uint32_t* __restrict__ row_cj_link,
                                    math::Vec3* __restrict__ row_cj_point,
                                    math::Vec3* __restrict__ row_cj_dir,
                                    uint32_t* __restrict__ row_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_count;
    if (gid >= total) return;
    const uint32_t env = gid / slot_count;
    const uint32_t slot = gid - env * slot_count;

    const UnionSlotDev u = LoadUnionSlot(union_slots, slot);
    if (u.cls == kUSlotInactive) return;

    const uint32_t max_pts = USlotMaxPoints(u);
    const uint32_t rpp = USlotRowsPerPoint(u);
    const uint32_t n_spokes = rpp - 1u;
    const uint32_t base = env * rows_per_env + u.row_base;
    const uint32_t n_active = ucontact_count[gid];

    // Side kinds/indices by class (the legacy wiring loop's classification).
    uint32_t kind_a = kNkSideStatic, idx_a = ~0u;
    uint32_t kind_b = kNkSideStatic, idx_b = ~0u;
    switch (u.cls) {
        case kUSlotFootSpherePlane:
            kind_a = kNkSideArtic;  idx_a = env;
            kind_b = kNkSideStatic; idx_b = ~0u;
            break;
        case kUSlotFingerSphereHull:
            kind_a = kNkSideArtic;  idx_a = env;
            kind_b = kNkSideRigid;  idx_b = env * bodies_per_env + u.body;
            break;
        case kUSlotBodyBoxPlane:
            kind_a = kNkSideRigid;  idx_a = env * bodies_per_env + u.body;
            kind_b = kNkSideStatic; idx_b = ~0u;
            break;
        case kUSlotParticleSpherePlane:
            // side a == the GLOBAL particle row (env*particles_per_env + link);
            // side b == the static +Z plane.
            kind_a = kNkSideParticle; idx_a = env * particles_per_env + u.link;
            kind_b = kNkSideStatic;   idx_b = ~0u;
            break;
        case kUSlotParticleSphereBox:
            kind_a = kNkSideParticle; idx_a = env * particles_per_env + u.link;
            kind_b = kNkSideRigid;    idx_b = env * bodies_per_env + u.body;
            break;
        default:
            break;
    }

    // Spoke basis from point[0]'s normal (EmitCompliantContactRows: ONE basis
    // per manifold).
    math::Vec3 t0{1.0f, 0.0f, 0.0f}, t1v{0.0f, 1.0f, 0.0f};
    if (n_active > 0u && n_spokes > 0u) {
        const math::Vec3 base_normal =
            NormalizedHostExpr(ucontact_normal[static_cast<size_t>(gid) * 4u]);
        t0 = ChooseTangentDev(base_normal);
        t1v = NormalizedHostExpr(base_normal.Cross(t0));
    }

    const float solref[2] = {solref0, solref1};
    const float solimp[5] = {solimp0, solimp1, solimp2, solimp3, solimp4};

    // The rigid-side COM for the r x j angular augment (constant this step).
    math::Vec3 com_a{0.0f, 0.0f, 0.0f}, com_b{0.0f, 0.0f, 0.0f};
    if (kind_a == kNkSideRigid) com_a = body_pose[idx_a].position;
    if (kind_b == kNkSideRigid) com_b = body_pose[idx_b].position;

    uint32_t active_rows = 0u;
    for (uint32_t p = 0u; p < max_pts; ++p) {
        const bool live = p < n_active;
        const size_t mp = static_cast<size_t>(gid) * 4u + p;

        // ---- normal row at base + p ------------------------------------
        {
            const uint32_t rs = base + p;
            NkRow row{};  // zero-init: inactive default.
            lambda[rs] = 0.0f;   // legacy union cold start (impulses==0).
            row_cj_link[rs] = kInvalidLink;
            if (live) {
                const math::Vec3 n = NormalizedHostExpr(ucontact_normal[mp]);
                const math::Vec3 point = ucontact_point[mp];
                const float pos = -ucontact_depth[mp];
                const constraint::CompliantContactRow compliant =
                    constraint::ComputeCompliantRow(solref, solimp, pos, pos,
                                                    /*vel=*/0.0f,
                                                    /*invweight=*/1.0f, dt,
                                                    /*refsafe=*/true);
                row.flags = nk::nk_row_flags::kActive;
                row.group_first = base;
                row.group_normal_count = max_pts;
                row.env = env;
                row.rhs = compliant.aref_bias;
                row.compliance_alpha = compliant.R;
                row.lower = 0.0f;
                row.upper = kFltMax;
                row.mu = u.mu;
                row.a.kind = kind_a; row.a.index = idx_a;
                row.a.jlin = n;
                row.b.kind = kind_b; row.b.index = idx_b;
                row.b.jlin = math::Vec3{-n.x, -n.y, -n.z};
                // C5c rigid angular augment: jang = (point - COM) x jlin.
                if (kind_a == kNkSideRigid) row.a.jang = (point - com_a).Cross(row.a.jlin);
                if (kind_b == kNkSideRigid) row.b.jang = (point - com_b).Cross(row.b.jlin);
                if (kind_a == kNkSideArtic) {
                    row_cj_link[rs] = env * base_link_count + u.link;
                    row_cj_point[rs] = point;
                    row_cj_dir[rs] = row.a.jlin;  // the artic side's row direction
                }
                ++active_rows;
            }
            urows[rs] = row;
        }

        // ---- friction spokes at base + max_pts + p*n_spokes + k ----------
        for (uint32_t k = 0u; k < n_spokes; ++k) {
            const uint32_t rs = base + max_pts + p * n_spokes + k;
            NkRow row{};
            lambda[rs] = 0.0f;
            row_cj_link[rs] = kInvalidLink;
            if (live) {
                const math::Vec3 spokes[4] = {
                    t0, math::Vec3{-t0.x, -t0.y, -t0.z},
                    t1v, math::Vec3{-t1v.x, -t1v.y, -t1v.z}};
                const math::Vec3 dir = spokes[k & 3u];
                const math::Vec3 point = ucontact_point[mp];
                row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kFriction;
                row.group_first = base;
                row.group_normal_count = max_pts;
                row.env = env;
                row.rhs = 0.0f;               // no positional bias (emitter).
                row.compliance_alpha = 0.0f;  // spokes carry R = 0.
                row.lower = 0.0f;
                row.upper = 0.0f;             // overridden in-kernel (pyramid).
                row.mu = u.mu;
                row.a.kind = kind_a; row.a.index = idx_a;
                row.a.jlin = dir;
                row.b.kind = kind_b; row.b.index = idx_b;
                row.b.jlin = math::Vec3{-dir.x, -dir.y, -dir.z};
                if (kind_a == kNkSideRigid) row.a.jang = (point - com_a).Cross(row.a.jlin);
                if (kind_b == kNkSideRigid) row.b.jang = (point - com_b).Cross(row.b.jlin);
                if (kind_a == kNkSideArtic) {
                    row_cj_link[rs] = env * base_link_count + u.link;
                    row_cj_point[rs] = point;
                    row_cj_dir[rs] = row.a.jlin;
                }
                ++active_rows;
            }
            urows[rs] = row;
        }
    }
    if (active_rows > 0u) {
        atomicAdd(&row_count[env], active_rows);  // order-independent sum (D1).
    }
}

// K4a: w = M^-1 J^T per articulation row. One thread per (row, r): the
// ArticulationApplyImpulse / ArticulationEffectiveInvMass inner product
// `acc = sum_c Minv[r*stride+c] * J[c]` with the IDENTICAL ascending-c order,
// hoisted to assembly (the value is constant across iterations).
__global__ void ComputeRowMinvJtKernel(const NkRow* __restrict__ urows,
                                       const float* __restrict__ chain_jacobian,
                                       const float* __restrict__ m_inv,
                                       uint32_t total_rows,
                                       uint32_t dof_stride,
                                       float* __restrict__ row_minv_jt) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = total_rows * dof_stride;
    if (gid >= total) return;
    const uint32_t rs = gid / dof_stride;
    const uint32_t r = gid - rs * dof_stride;
    const NkRow row = urows[rs];
    if (!(row.flags & nk::nk_row_flags::kActive) || row.a.kind != kNkSideArtic) {
        return;  // only articulation rows carry a chain-J / w pair.
    }
    const size_t tile = static_cast<size_t>(row.a.index) * dof_stride * dof_stride;
    const float* const Minv = m_inv + tile + static_cast<size_t>(r) * dof_stride;
    const float* const J = chain_jacobian + static_cast<size_t>(rs) * dof_stride;
    float acc = 0.0f;
    for (uint32_t c = 0u; c < dof_stride; ++c) {
        acc += Minv[c] * J[c];
    }
    row_minv_jt[gid] = acc;
}

// K4b: per-row effective mass — ComputeCompliantEffectiveMass hoisted to
// assembly: diagonal = side-a invMass + side-b invMass + R, recip with the
// legacy 1e-12 floor. Side dispatch mirrors CompliantSideEffectiveInvMass.
__global__ void ComputeRowMeffKernel(const NkRow* __restrict__ urows,
                                     const float* __restrict__ chain_jacobian,
                                     const float* __restrict__ row_minv_jt,
                                     const float* __restrict__ body_inv_mass,
                                     const math::Vec3* __restrict__ body_inv_inertia,
                                     const float* __restrict__ particle_inv_mass,
                                     uint32_t total_rows,
                                     uint32_t dof_stride,
                                     float* __restrict__ row_meff) {
    const uint32_t rs = blockIdx.x * blockDim.x + threadIdx.x;
    if (rs >= total_rows) return;
    const NkRow row = urows[rs];
    if (!(row.flags & nk::nk_row_flags::kActive)) {
        row_meff[rs] = 0.0f;
        return;
    }
    float diagonal = 0.0f;
    const NkRowSide sides[2] = {row.a, row.b};
    for (int s = 0; s < 2; ++s) {
        const NkRowSide& sd = sides[s];
        switch (sd.kind) {
            case kNkSideRigid: {
                // RigidEffectiveInvMass: inv_mass*|jlin|^2 + jang^2 . invI.
                const float im = body_inv_mass[sd.index];
                const math::Vec3 ii = body_inv_inertia[sd.index];
                diagonal += im * Dot3(sd.jlin, sd.jlin);
                diagonal += sd.jang.x * sd.jang.x * ii.x;
                diagonal += sd.jang.y * sd.jang.y * ii.y;
                diagonal += sd.jang.z * sd.jang.z * ii.z;
                break;
            }
            case kNkSideArtic: {
                // ArticulationEffectiveInvMass: denom = sum_r J[r] * w[r] —
                // w[r] is the SAME inner product the legacy computed inline.
                const float* const J =
                    chain_jacobian + static_cast<size_t>(rs) * dof_stride;
                const float* const w =
                    row_minv_jt + static_cast<size_t>(rs) * dof_stride;
                float denom = 0.0f;
                for (uint32_t r = 0u; r < dof_stride; ++r) {
                    denom += J[r] * w[r];
                }
                diagonal += denom;
                break;
            }
            case kNkSideParticle:
                diagonal += particle_inv_mass != nullptr
                                ? particle_inv_mass[sd.index] * Dot3(sd.jlin, sd.jlin)
                                : 0.0f;
                break;
            default:
                break;  // StaticNull: zero reaction.
        }
    }
    diagonal += row.compliance_alpha;  // + R (dual regularizer)
    row_meff[rs] = diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
}

// --- op entry point ---------------------------------------------------------

Status OpAssembleRowsFused(const ModelView& model, const DataView& data,
                           const AssembleRowsParams* p, cudaStream_t stream) {
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

Status OpAssembleRowsUnion(const ModelView& model, const DataView& data,
                           const AssembleRowsParams* p, cudaStream_t stream) {
    if (p->env_count == 0u || p->union_slot_count == 0u || p->rows_per_env == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t total_rows = p->env_count * p->rows_per_env;
    const bool has_artic = p->max_dof > 0u && p->articulation_count > 0u;

    // K1: pack the flat qdot (articulated models only).
    if (has_artic) {
        const uint32_t total = p->env_count * p->max_dof;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(PackQdotFlatKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   reinterpret_cast<const Spatial6*>(data.link_velocity),
                   static_cast<const float*>(data.qdot),
                   static_cast<const uint32_t*>(model.dof_to_link),
                   static_cast<const uint32_t*>(model.dof_to_component),
                   p->env_count, p->max_dof, p->base_link_count,
                   data.qdot_flat);
    }

    // K2: emit the union rows onto the fixed slots.
    {
        const uint32_t total = p->env_count * p->union_slot_count;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(EmitUnionRowsKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   static_cast<const float*>(model.union_slots),
                   static_cast<const uint32_t*>(data.ucontact_count),
                   static_cast<const math::Vec3*>(data.ucontact_point),
                   static_cast<const math::Vec3*>(data.ucontact_normal),
                   static_cast<const float*>(data.ucontact_depth),
                   static_cast<const math::Transform*>(data.body_pose),
                   p->solref[0], p->solref[1],
                   p->solimp[0], p->solimp[1], p->solimp[2], p->solimp[3],
                   p->solimp[4],
                   p->dt,
                   p->env_count, p->union_slot_count, p->rows_per_env,
                   p->bodies_per_env, p->base_link_count, p->particles_per_env,
                   reinterpret_cast<NkRow*>(data.urows),
                   data.lambda,
                   data.row_cj_link, data.row_cj_point, data.row_cj_dir,
                   data.row_count);
    }

    if (has_artic) {
        // K3: chain Jacobians per row slot (the SAME kernel as the fused
        // family, fed the per-row gathered link/point/dir). Zero the output
        // first — the kernel writes only ancestor-chain entries (a stream
        // memset, captured fine; NOT an allocation).
        const size_t jbytes =
            static_cast<size_t>(total_rows) * p->max_dof * sizeof(float);
        if (cudaMemsetAsync(data.chain_jacobian, 0, jbytes, stream) != cudaSuccess) {
            return Status::Failed;
        }
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        const uint32_t blocks = (total_rows + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ComputeContactChainJacobianKernel, dim3(blocks), dim3(kBlockSize),
                   0u, stream, state,
                   static_cast<const uint32_t*>(data.row_cj_link),
                   static_cast<const math::Vec3*>(data.row_cj_point),
                   static_cast<const math::Vec3*>(data.row_cj_dir),
                   total_rows, p->max_dof, data.chain_jacobian);

        // K4a: w = M^-1 J^T per articulation row.
        const uint32_t wtotal = total_rows * p->max_dof;
        const uint32_t wblocks = (wtotal + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ComputeRowMinvJtKernel, dim3(wblocks), dim3(kBlockSize), 0u,
                   stream,
                   reinterpret_cast<const NkRow*>(data.urows),
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.m_inv),
                   total_rows, p->max_dof, data.row_minv_jt);
    }

    // K4b: per-row effective mass.
    {
        const uint32_t blocks = (total_rows + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ComputeRowMeffKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   reinterpret_cast<const NkRow*>(data.urows),
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.row_minv_jt),
                   static_cast<const float*>(data.body_inv_mass),
                   static_cast<const math::Vec3*>(data.body_inv_inertia),
                   static_cast<const float*>(data.particle_inv_mass),
                   total_rows, p->max_dof, data.row_meff);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpAssembleRows(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const AssembleRowsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    return (p->family == kContactFamilyUnionCsr)
               ? OpAssembleRowsUnion(model, data, p, stream)
               : OpAssembleRowsFused(model, data, p, stream);
}

} // namespace

void RegisterNkAssembleRowsOps() {
    SetCudaOp(NkOp::AssembleRows, &OpAssembleRows);
}

} // namespace nuka::phi
