// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M4 AssembleRows (the §3.4 row-assembly op).
//
// Contact families behind the ONE op (params->family):
//
//   FUSED (kContactFamilyFusedFoot) — DEAD as of L1-b: the FUSED contact RUNTIME
//   was deleted (solve_rows.cu / contacts_foot.cu kernels gone, and the pipeline
//   no longer selects this family). The OpAssembleRowsFused branch below is now
//   UNREACHABLE — no cooked model is FusedFoot anymore — and is retained, dead,
//   only until the L1-d enum collapse removes it wholesale. (Its kernels are the
//   line-by-line ports of articulation_jacobian.cu / articulation_contacts.cu.)
//
//   UNION (kContactFamilyUnionCsr) — DELETED in L1-c: the legacy coresident
//   union world's CSR row assembly (EmitUnionRowsKernel / OpAssembleRowsUnion)
//   was removed wholesale. Grasp moved to RL; the ONE general path is PairDriven.
//
//   PAIRDRIVEN (kContactFamilyPairDriven) — the ONE general path: the broadphase
//   -> narrowphase manifolds (ucontact_*) are turned into solver rows (urows) by
//   EmitPairDrivenRowsKernel + the shared chain-J / row_minv_jt / row_meff hoists.
//   Entirely arena-resident; ZERO host participation; every launch shape is a
//   fixed function of the capacities -> CUDA-graph capturable.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "constraint/solref_solimp.hpp"  // ComputeCompliantRow (HD)
#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // LoadPrimShape (PairDriven side resolve)
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kFltMax = 3.402823466e+38f;

// Per-shape material lives in mat_buckets (8 f32/bucket) keyed per body row by
// mat_index. Lane meaning mirrors ModelMaterialBucket (cook side):
//   [0]=static mu [1]=dynamic mu [2]=restitution [3]=solref timeconst
//   [4]=solref dampratio [5]=density [6]=sdf cell [7]=reserved.
constexpr uint32_t kMatBucketStride = 8u;
constexpr uint32_t kMatLaneFriction = 0u;
constexpr uint32_t kMatLaneSolref0 = 3u;
constexpr uint32_t kMatLaneSolref1 = 4u;

// Model-default friction for a side with no authored material (static ground /
// unwired tables). Matches the cook's MuJoCo default so default scenes stay at
// the prior global mu=1.0.
constexpr float kDefaultMaterialFriction = 1.0f;

// Combined per-contact material params (the two shapes' authored materials mixed
// per the MuJoCo solmix=max convention). Defaults = the model-level defaults, so
// a contact with no authored material is numerically unchanged.
struct PairMaterial {
    float mu;
    float solref0;
    float solref1;
};

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
// C3 (general contact pipeline Phase 0) — CONTACT POINT/NORMAL FRAME DECISION.
// The unified contact buffer (the FUSED contact_*, the union ucontact_*, and the
// general PairDriven manifold) stores the contact POINT in WORLD space and the
// contact NORMAL (A->B) in WORLD space. Newton stores points in BODY frame +
// normal in world (contact_data.py); Nuka keeps WORLD point + world normal for
// minimal D1 churn, because this chain-Jacobian kernel ALREADY consumes a world
// point + world normal directly (contact_point_world / contact_normal_world
// below) — re-keying to body-frame storage would require a per-row world-recompose
// with no functional gain in Phase 0. Body-frame storage is deferred to the
// collide-once / reuse-across-substeps optimization (out of Phase-0 scope). This
// is the ONE documented representation feeding AssembleRows for every family.
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

// L1-c: EmitUnionRowsKernel (the UnionCsr K2 row emitter) was DELETED here. The
// ONE general path emits its rows via EmitPairDrivenRowsKernel below.

// K1 (PairDriven, S3): pack the per-ARTICULATION flat qdot tiles. One thread per
// (global artic x DOF). dof_to_link/component are the per:dog TEMPLATE map; for
// co-resident dog a the same template-local (link, component) applies to the
// articulation's links, which live contiguously at a*links_per_dog within the env
// (the multi-dog cook concatenates dogs' links). At K==1 (k_tiles == 1) this is
// EXACTLY the legacy PackQdotFlat (env tile 0).
__global__ void PackQdotFlatMultiKernel(const Spatial6* __restrict__ link_velocity,
                                        const float* __restrict__ qdot,
                                        const uint32_t* __restrict__ dof_to_link,
                                        const uint32_t* __restrict__ dof_to_component,
                                        uint32_t artic_count, uint32_t dof_stride,
                                        uint32_t base_link_count, uint32_t k_tiles,
                                        float* __restrict__ qdot_flat) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = artic_count * dof_stride;
    if (gid >= total) return;
    const uint32_t ag = gid / dof_stride;          // global articulation
    const uint32_t k = gid - ag * dof_stride;      // DOF within tile
    const uint32_t kt = (k_tiles == 0u) ? 1u : k_tiles;
    const uint32_t env = ag / kt;
    const uint32_t a = ag - env * kt;              // env-local artic
    const uint32_t links_per_dog = base_link_count / kt;
    const size_t tmpl = static_cast<size_t>(env) * dof_stride + k;  // per:dof template idx
    const uint32_t tmpl_link = dof_to_link[tmpl];
    const uint32_t comp = dof_to_component[tmpl];
    const uint32_t link = tmpl_link + a * links_per_dog;
    const size_t gl = static_cast<size_t>(env) * base_link_count + link;
    qdot_flat[gid] = (comp != ~0u) ? link_velocity[gl].v[comp] : qdot[gl];
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

// ===========================================================================
// GENERAL CONTACT PIPELINE — Phase 1B PairDriven-family assembly (S1/S2/S5).
//
// The third OpAssembleRows branch (the union/fused arms are LITERALLY unchanged,
// the H1 union golden + go2 FusedFoot golden D1 invariant). Per active unified
// contact slot (the pair-driven narrowphase's ucontact_* manifold + the C1/C2
// (a,b) collidable ids) it resolves each side via the registry (body_to_link /
// body_to_articulation / shape_table body_id) into {kNkSideArtic, real artic id
// (S1), owning link}, {kNkSideRigid, body row}, or {kNkSideStatic}, and fills a
// COUPLED two-sided NkRow + the side-A chain-J gather (row_cj_*) AND the side-B
// chain-J gather (row_cj_*_b, S2). The chain-J kernel + K4a/K4b then run with
// the side-B second passes. The Jv reduction / two-sided apply / mixed-island
// coupling are ALL the existing SolveUnionRowWarp + island solver, unchanged.
// ===========================================================================

// Per-slot layout (FIXED worst case, mirroring the union per-manifold layout) —
// the SHARED constants from nk_row.hpp so the cook (row budget), the schedule, and
// this assembly all agree. kPdPtsPerSlot manifold points, each expanding to 1
// normal row + kPdSpokes friction-spoke rows (+t0,-t0,+t1,-t1).
constexpr uint32_t kPdPtsPerSlot = nk::kPairDrivenPtsPerSlot;   // 4
constexpr uint32_t kPdSpokes     = nk::kPairDrivenSpokesPerPt;  // 4
constexpr uint32_t kPdRowsPerSlot = nk::kPairDrivenRowsPerSlot; // 20

// Resolve a collidable body row -> reaction side. body_id == -1 (from shape_table)
// is static; an articulation-link body row resolves to (real artic id, owning
// global link); a free-rigid body row resolves to the rigid side. Returns the
// side kind; out_artic / out_link valid only for the artic kind; out_body for
// the rigid kind. env-major: body_to_* are TEMPLATE-local (per:body), so the
// global body row == env*bodies_per_env + local; the global link == env*base_link
// + template_link; the global artic == env*artics_per_env + local_artic.
__device__ uint32_t ResolvePairSide(const float* shape_table,
                                    const uint32_t* body_to_link,
                                    const uint32_t* body_to_articulation,
                                    uint32_t env, uint32_t local_body,
                                    uint32_t bodies_per_env, uint32_t base_link_count,
                                    uint32_t artics_per_env,
                                    uint32_t* out_artic, uint32_t* out_link,
                                    uint32_t* out_body) {
    *out_artic = ~0u; *out_link = ~0u; *out_body = ~0u;
    const PrimShapeDev s = LoadPrimShape(shape_table, local_body);
    if (s.body_id < 0) {
        return kNkSideStatic;  // static ground / heightfield collidable.
    }
    const uint32_t tmpl_link = (local_body < bodies_per_env)
                                   ? body_to_link[local_body] : ~0u;
    if (tmpl_link == ~0u) {
        // free-rigid body row (not an articulation link).
        *out_body = env * bodies_per_env + local_body;
        return kNkSideRigid;
    }
    const uint32_t local_artic = (local_body < bodies_per_env)
                                     ? body_to_articulation[local_body] : ~0u;
    *out_artic = env * artics_per_env + (local_artic == ~0u ? 0u : local_artic);
    *out_link = env * base_link_count + tmpl_link;
    return kNkSideArtic;
}

// One side's authored material (friction + solref) from mat_buckets/mat_index, or
// the model defaults when the side has no body-row material (static collidable /
// out-of-range / unwired tables) -- so default-material scenes stay unchanged.
__device__ void ReadSideMaterial(const float* mat_buckets, const uint32_t* mat_index,
                                 uint32_t env, uint32_t local_body,
                                 uint32_t bodies_per_env, int32_t body_id,
                                 float def_mu, float def_solref0, float def_solref1,
                                 float* out_mu, float* out_solref0, float* out_solref1) {
    *out_mu = def_mu; *out_solref0 = def_solref0; *out_solref1 = def_solref1;
    if (mat_buckets == nullptr || mat_index == nullptr || body_id < 0 ||
        local_body >= bodies_per_env) {
        return;  // static / unwired -> model default material.
    }
    const uint32_t bucket = mat_index[env * bodies_per_env + local_body];
    const size_t b = static_cast<size_t>(bucket) * kMatBucketStride;
    *out_mu = mat_buckets[b + kMatLaneFriction];
    *out_solref0 = mat_buckets[b + kMatLaneSolref0];
    *out_solref1 = mat_buckets[b + kMatLaneSolref1];
}

// One thread per (env x candidate slot). Emits the slot's NkRow block from the
// unified manifold. Reuses ComputeCompliantRow + ChooseTangentDev (the union
// math). Side A is the candidate's a-collidable, side B its b-collidable; the
// chain-J gathers (row_cj_*, row_cj_*_b) carry the per-side artic link/point/dir.
__global__ void EmitPairDrivenRowsKernel(
    const uint32_t* __restrict__ ucontact_count,
    const math::Vec3* __restrict__ ucontact_point,
    const math::Vec3* __restrict__ ucontact_normal,
    const float* __restrict__ ucontact_depth,
    const uint32_t* __restrict__ ucontact_a,
    const uint32_t* __restrict__ ucontact_b,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ shape_table,
    const uint32_t* __restrict__ body_to_link,
    const uint32_t* __restrict__ body_to_articulation,
    const float* __restrict__ mat_buckets,
    const uint32_t* __restrict__ mat_index,
    float solref0, float solref1,
    float solimp0, float solimp1, float solimp2, float solimp3, float solimp4,
    float dt,
    uint32_t env_count, uint32_t slot_count, uint32_t rows_per_env,
    uint32_t bodies_per_env, uint32_t base_link_count, uint32_t artics_per_env,
    NkRow* __restrict__ urows, float* __restrict__ lambda,
    uint32_t* __restrict__ row_cj_link, math::Vec3* __restrict__ row_cj_point,
    math::Vec3* __restrict__ row_cj_dir,
    uint32_t* __restrict__ row_cj_link_b, math::Vec3* __restrict__ row_cj_point_b,
    math::Vec3* __restrict__ row_cj_dir_b,
    uint32_t* __restrict__ row_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_count;
    if (gid >= total) return;
    const uint32_t env = gid / slot_count;
    const uint32_t slot = gid - env * slot_count;
    const uint32_t base = env * rows_per_env + slot * kPdRowsPerSlot;

    const uint32_t n_active = ucontact_count[gid];

    // Resolve the two sides ONCE per slot (same a/b for every manifold point).
    uint32_t kind_a = kNkSideStatic, kind_b = kNkSideStatic;
    uint32_t art_a = ~0u, link_a = ~0u, body_a = ~0u;
    uint32_t art_b = ~0u, link_b = ~0u, body_b = ~0u;
    uint32_t local_a = ~0u, local_b = ~0u;
    int32_t bid_a = -1, bid_b = -1;
    if (n_active > 0u) {
        local_a = ucontact_a[static_cast<size_t>(gid) * 4u];
        local_b = ucontact_b[static_cast<size_t>(gid) * 4u];
        bid_a = LoadPrimShape(shape_table, local_a).body_id;
        bid_b = LoadPrimShape(shape_table, local_b).body_id;
        kind_a = ResolvePairSide(shape_table, body_to_link, body_to_articulation,
                                 env, local_a, bodies_per_env, base_link_count,
                                 artics_per_env, &art_a, &link_a, &body_a);
        kind_b = ResolvePairSide(shape_table, body_to_link, body_to_articulation,
                                 env, local_b, bodies_per_env, base_link_count,
                                 artics_per_env, &art_b, &link_b, &body_b);
    }
    const uint32_t idx_a = (kind_a == kNkSideArtic) ? art_a
                          : (kind_a == kNkSideRigid) ? body_a : ~0u;
    const uint32_t idx_b = (kind_b == kNkSideArtic) ? art_b
                          : (kind_b == kNkSideRigid) ? body_b : ~0u;

    math::Vec3 com_a{0, 0, 0}, com_b{0, 0, 0};
    if (kind_a == kNkSideRigid) com_a = body_pose[idx_a].position;
    if (kind_b == kNkSideRigid) com_b = body_pose[idx_b].position;

    // Per-contact material = the two shapes' authored materials, mixed by the
    // MuJoCo solmix=max rule (friction max; solref take the stiffer/min timeconst).
    // Static / unauthored sides fall back to the model defaults below.
    float mu_a, sr0_a, sr1_a, mu_b, sr0_b, sr1_b;
    ReadSideMaterial(mat_buckets, mat_index, env, local_a, bodies_per_env, bid_a,
                     kDefaultMaterialFriction, solref0, solref1, &mu_a, &sr0_a, &sr1_a);
    ReadSideMaterial(mat_buckets, mat_index, env, local_b, bodies_per_env, bid_b,
                     kDefaultMaterialFriction, solref0, solref1, &mu_b, &sr0_b, &sr1_b);
    PairMaterial mat;
    mat.mu = fmaxf(mu_a, mu_b);
    mat.solref0 = fminf(sr0_a, sr0_b);  // smaller timeconst == stiffer
    mat.solref1 = fmaxf(sr1_a, sr1_b);

    const float solref[2] = {mat.solref0, mat.solref1};
    const float solimp[5] = {solimp0, solimp1, solimp2, solimp3, solimp4};
    const float mu = mat.mu;  // per-contact friction coefficient (cone bound).
    const float kFltMaxLocal = kFltMax;

    uint32_t active_rows = 0u;
    for (uint32_t p = 0u; p < kPdPtsPerSlot; ++p) {
        const bool live = p < n_active && p < 4u;
        const size_t mp = static_cast<size_t>(gid) * 4u + p;
        const uint32_t normal_row = base + p;          // pts normal rows first.
        const uint32_t spoke_base = base + kPdPtsPerSlot + p * kPdSpokes;

        math::Vec3 n{0, 0, 1};
        math::Vec3 point{0, 0, 0};
        math::Vec3 t0{1, 0, 0}, t1v{0, 1, 0};
        if (live) {
            n = NormalizedHostExpr(ucontact_normal[mp]);
            point = ucontact_point[mp];
            t0 = ChooseTangentDev(n);
            t1v = NormalizedHostExpr(n.Cross(t0));
        }

        // ---- normal row ------------------------------------------------------
        {
            const uint32_t rs = normal_row;
            NkRow row{};
            lambda[rs] = 0.0f;
            row_cj_link[rs] = kInvalidLink;
            row_cj_link_b[rs] = kInvalidLink;
            if (live) {
                const float pos = -ucontact_depth[mp];
                const constraint::CompliantContactRow compliant =
                    constraint::ComputeCompliantRow(solref, solimp, pos, pos,
                                                    /*vel=*/0.0f, /*invweight=*/1.0f,
                                                    dt, /*refsafe=*/true);
                row.flags = nk::nk_row_flags::kActive;
                row.group_first = base;
                row.group_normal_count = kPdPtsPerSlot;
                row.env = env;
                row.rhs = compliant.aref_bias;
                row.compliance_alpha = compliant.R;
                row.lower = 0.0f;
                row.upper = kFltMaxLocal;
                row.mu = mu;
                row.a.kind = kind_a; row.a.index = idx_a; row.a.jlin = n;
                row.b.kind = kind_b; row.b.index = idx_b;
                row.b.jlin = math::Vec3{-n.x, -n.y, -n.z};
                if (kind_a == kNkSideRigid) row.a.jang = (point - com_a).Cross(row.a.jlin);
                if (kind_b == kNkSideRigid) row.b.jang = (point - com_b).Cross(row.b.jlin);
                if (kind_a == kNkSideArtic) {
                    row_cj_link[rs] = link_a; row_cj_point[rs] = point;
                    row_cj_dir[rs] = row.a.jlin;
                }
                if (kind_b == kNkSideArtic) {
                    row_cj_link_b[rs] = link_b; row_cj_point_b[rs] = point;
                    row_cj_dir_b[rs] = row.b.jlin;
                }
                ++active_rows;
            }
            urows[rs] = row;
        }

        // ---- friction spokes -------------------------------------------------
        for (uint32_t k = 0u; k < kPdSpokes; ++k) {
            const uint32_t rs = spoke_base + k;
            NkRow row{};
            lambda[rs] = 0.0f;
            row_cj_link[rs] = kInvalidLink;
            row_cj_link_b[rs] = kInvalidLink;
            if (live) {
                const math::Vec3 spokes[4] = {
                    t0, math::Vec3{-t0.x, -t0.y, -t0.z},
                    t1v, math::Vec3{-t1v.x, -t1v.y, -t1v.z}};
                const math::Vec3 dir = spokes[k & 3u];
                row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kFriction;
                row.group_first = base;
                row.group_normal_count = kPdPtsPerSlot;
                row.env = env;
                row.rhs = 0.0f;
                row.compliance_alpha = 0.0f;
                row.lower = 0.0f;
                row.upper = 0.0f;  // overridden in-kernel (pyramid bound).
                row.mu = mu;
                row.a.kind = kind_a; row.a.index = idx_a; row.a.jlin = dir;
                row.b.kind = kind_b; row.b.index = idx_b;
                row.b.jlin = math::Vec3{-dir.x, -dir.y, -dir.z};
                if (kind_a == kNkSideRigid) row.a.jang = (point - com_a).Cross(row.a.jlin);
                if (kind_b == kNkSideRigid) row.b.jang = (point - com_b).Cross(row.b.jlin);
                if (kind_a == kNkSideArtic) {
                    row_cj_link[rs] = link_a; row_cj_point[rs] = point;
                    row_cj_dir[rs] = row.a.jlin;
                }
                if (kind_b == kNkSideArtic) {
                    row_cj_link_b[rs] = link_b; row_cj_point_b[rs] = point;
                    row_cj_dir_b[rs] = row.b.jlin;
                }
                ++active_rows;
            }
            urows[rs] = row;
        }
    }
    if (active_rows > 0u) atomicAdd(&row_count[env], active_rows);
}

// K4a-B: w_b = M^-1 J_b^T per articulation SIDE-B row (S2). Mirrors
// ComputeRowMinvJtKernel but gates on row.b.kind and tiles by row.b.index.
__global__ void ComputeRowMinvJtBKernel(const NkRow* __restrict__ urows,
                                        const float* __restrict__ chain_jacobian_b,
                                        const float* __restrict__ m_inv,
                                        uint32_t total_rows, uint32_t dof_stride,
                                        float* __restrict__ row_minv_jt_b) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = total_rows * dof_stride;
    if (gid >= total) return;
    const uint32_t rs = gid / dof_stride;
    const uint32_t r = gid - rs * dof_stride;
    const NkRow row = urows[rs];
    if (!(row.flags & nk::nk_row_flags::kActive) || row.b.kind != kNkSideArtic) {
        return;
    }
    const size_t tile = static_cast<size_t>(row.b.index) * dof_stride * dof_stride;
    const float* const Minv = m_inv + tile + static_cast<size_t>(r) * dof_stride;
    const float* const J = chain_jacobian_b + static_cast<size_t>(rs) * dof_stride;
    float acc = 0.0f;
    for (uint32_t c = 0u; c < dof_stride; ++c) acc += Minv[c] * J[c];
    row_minv_jt_b[gid] = acc;
}

// K4b PairDriven: per-row effective mass over BOTH arms with the CORRECT arrays
// (side A: chain_jacobian/row_minv_jt; side B artic: chain_jacobian_b/row_minv_jt_b).
__global__ void ComputeRowMeffPairDrivenKernel(
    const NkRow* __restrict__ urows,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
    const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ row_minv_jt_b,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    uint32_t total_rows, uint32_t dof_stride, float* __restrict__ row_meff) {
    const uint32_t rs = blockIdx.x * blockDim.x + threadIdx.x;
    if (rs >= total_rows) return;
    const NkRow row = urows[rs];
    if (!(row.flags & nk::nk_row_flags::kActive)) { row_meff[rs] = 0.0f; return; }
    float diagonal = 0.0f;
    for (int s = 0; s < 2; ++s) {
        const NkRowSide& sd = (s == 0) ? row.a : row.b;
        switch (sd.kind) {
            case kNkSideRigid: {
                const float im = body_inv_mass[sd.index];
                const math::Vec3 ii = body_inv_inertia[sd.index];
                diagonal += im * Dot3(sd.jlin, sd.jlin);
                diagonal += sd.jang.x * sd.jang.x * ii.x;
                diagonal += sd.jang.y * sd.jang.y * ii.y;
                diagonal += sd.jang.z * sd.jang.z * ii.z;
                break;
            }
            case kNkSideArtic: {
                const float* const J = ((s == 0) ? chain_jacobian : chain_jacobian_b) +
                                       static_cast<size_t>(rs) * dof_stride;
                const float* const w = ((s == 0) ? row_minv_jt : row_minv_jt_b) +
                                       static_cast<size_t>(rs) * dof_stride;
                float denom = 0.0f;
                for (uint32_t r = 0u; r < dof_stride; ++r) denom += J[r] * w[r];
                diagonal += denom;
                break;
            }
            default: break;
        }
    }
    diagonal += row.compliance_alpha;
    row_meff[rs] = diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
}

Status OpAssembleRowsPairDriven(const ModelView& model, const DataView& data,
                                const AssembleRowsParams* p, cudaStream_t stream) {
    if (p->env_count == 0u || p->union_slot_count == 0u || p->rows_per_env == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t total_rows = p->env_count * p->rows_per_env;
    const bool has_artic = p->max_dof > 0u && p->articulation_count > 0u;
    const uint32_t artics_per_env =
        (p->articulation_count > 0u && p->env_count > 0u)
            ? (p->articulation_count / p->env_count) : 0u;

    // K1: pack the per-articulation flat qdot tiles (S3: one tile per co-resident
    // articulation, qdot_flat[artic_global*max_dof + k]). PackQdotFlatKernel is
    // keyed per env*max_dof -> for K>1 launch over articulation_count*max_dof so
    // every dog's tile is packed. dof_to_link/component are per:dof (TEMPLATE),
    // so the kernel re-uses them per artic via the env stride.
    if (has_artic) {
        const uint32_t total = p->articulation_count * p->max_dof;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(PackQdotFlatMultiKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   reinterpret_cast<const Spatial6*>(data.link_velocity),
                   static_cast<const float*>(data.qdot),
                   static_cast<const uint32_t*>(model.dof_to_link),
                   static_cast<const uint32_t*>(model.dof_to_component),
                   p->articulation_count, p->max_dof, p->base_link_count,
                   artics_per_env, data.qdot_flat);
    }

    // K2: emit the pair-driven rows (S1/S2/S5).
    {
        const uint32_t total = p->env_count * p->union_slot_count;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(EmitPairDrivenRowsKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   static_cast<const uint32_t*>(data.ucontact_count),
                   static_cast<const math::Vec3*>(data.ucontact_point),
                   static_cast<const math::Vec3*>(data.ucontact_normal),
                   static_cast<const float*>(data.ucontact_depth),
                   static_cast<const uint32_t*>(data.ucontact_a),
                   static_cast<const uint32_t*>(data.ucontact_b),
                   static_cast<const math::Transform*>(data.body_pose),
                   static_cast<const float*>(model.shape_table),
                   static_cast<const uint32_t*>(model.body_to_link),
                   static_cast<const uint32_t*>(model.body_to_articulation),
                   static_cast<const float*>(data.mat_buckets),
                   static_cast<const uint32_t*>(data.mat_index),
                   p->solref[0], p->solref[1],
                   p->solimp[0], p->solimp[1], p->solimp[2], p->solimp[3], p->solimp[4],
                   p->dt, p->env_count, p->union_slot_count, p->rows_per_env,
                   p->bodies_per_env, p->base_link_count, artics_per_env,
                   reinterpret_cast<NkRow*>(data.urows), data.lambda,
                   data.row_cj_link, data.row_cj_point, data.row_cj_dir,
                   data.row_cj_link_b, data.row_cj_point_b, data.row_cj_dir_b,
                   data.row_count);
    }

    if (has_artic) {
        // K3: chain Jacobians for side A AND side B (the SAME multi-artic kernel,
        // fed each side's per-row link/point/dir gather). Zero both outputs first.
        const size_t jbytes =
            static_cast<size_t>(total_rows) * p->max_dof * sizeof(float);
        if (cudaMemsetAsync(data.chain_jacobian, 0, jbytes, stream) != cudaSuccess ||
            cudaMemsetAsync(data.chain_jacobian_b, 0, jbytes, stream) != cudaSuccess) {
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
        LaunchCuda(ComputeContactChainJacobianKernel, dim3(blocks), dim3(kBlockSize),
                   0u, stream, state,
                   static_cast<const uint32_t*>(data.row_cj_link_b),
                   static_cast<const math::Vec3*>(data.row_cj_point_b),
                   static_cast<const math::Vec3*>(data.row_cj_dir_b),
                   total_rows, p->max_dof, data.chain_jacobian_b);

        // K4a: w = M^-1 J^T for side A and side B.
        const uint32_t wtotal = total_rows * p->max_dof;
        const uint32_t wblocks = (wtotal + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ComputeRowMinvJtKernel, dim3(wblocks), dim3(kBlockSize), 0u, stream,
                   reinterpret_cast<const NkRow*>(data.urows),
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.m_inv),
                   total_rows, p->max_dof, data.row_minv_jt);
        LaunchCuda(ComputeRowMinvJtBKernel, dim3(wblocks), dim3(kBlockSize), 0u, stream,
                   reinterpret_cast<const NkRow*>(data.urows),
                   static_cast<const float*>(data.chain_jacobian_b),
                   static_cast<const float*>(data.m_inv),
                   total_rows, p->max_dof, data.row_minv_jt_b);
    }

    // K4b: per-row effective mass over both arms (correct arrays per side).
    {
        const uint32_t blocks = (total_rows + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ComputeRowMeffPairDrivenKernel, dim3(blocks), dim3(kBlockSize), 0u,
                   stream, reinterpret_cast<const NkRow*>(data.urows),
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.row_minv_jt),
                   static_cast<const float*>(data.chain_jacobian_b),
                   static_cast<const float*>(data.row_minv_jt_b),
                   static_cast<const float*>(data.body_inv_mass),
                   static_cast<const math::Vec3*>(data.body_inv_inertia),
                   total_rows, p->max_dof, data.row_meff);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
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

// L1-c: OpAssembleRowsUnion (the UnionCsr K1-K4 assembly orchestrator) was
// DELETED here. The ONE general path assembles via OpAssembleRowsPairDriven.

Status OpAssembleRows(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const AssembleRowsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->family == kContactFamilyPairDriven) {
        return OpAssembleRowsPairDriven(model, data, p, stream);
    }
    // L1-b/L1-d dead fallthrough: the FUSED family is unreachable (no cooked
    // model selects it). Retained until the L1-d enum collapse removes it.
    return OpAssembleRowsFused(model, data, p, stream);
}

} // namespace

void RegisterNkAssembleRowsOps() {
    SetCudaOp(NkOp::AssembleRows, &OpAssembleRows);
}

} // namespace nuka::phi
