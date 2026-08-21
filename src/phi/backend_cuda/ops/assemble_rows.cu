// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — AssembleRows (the spec row-assembly op).
//
// Contact families behind the ONE op (params->family):
//
// FUSED (kContactFamilyFusedFoot) — DEAD as of : the FUSED contact RUNTIME
// was deleted (solve_rows.cu / contacts_foot.cu kernels gone, and the pipeline
// no longer selects this family). The OpAssembleRowsFused branch below is now
// UNREACHABLE — no cooked model is FusedFoot anymore — and is retained, dead,
// only until the enum collapse removes it wholesale. (Its kernels are the
// line-by-line ports of articulation_jacobian.cu / articulation_contacts.cu.)
//
// UNION (kContactFamilyUnionCsr) — DELETED in : the legacy coresident
// union world's CSR row assembly (EmitUnionRowsKernel / OpAssembleRowsUnion)
// was removed wholesale. Grasp moved to RL; the ONE general path is PairDriven.
//
// PAIRDRIVEN (kContactFamilyPairDriven) — the ONE general path: the broadphase
// -> narrowphase manifolds (ucontact_*) are turned into solver rows (urows) by
// EmitPairDrivenRowsKernel + the shared chain-J / row_minv_jt / row_meff hoists.
// Entirely arena-resident; ZERO host participation; every launch shape is a
// fixed function of the capacities -> CUDA-graph capturable.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "constraint/solref_solimp.hpp"  // ComputeCompliantRow (HD)
#include "nk/contact/contact_profile.hpp"
#include "scene/contact_filter.hpp"
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

// ContactProfileV1 is a packed mutable table. Cook and world upload validate it;
// assembly consumes the canonical 16-word records through each collidable row.
constexpr uint64_t kContactProfileHashDomain = 0x4e554b4150524f46ull;

__device__ scene::ContactParamsIn DefaultProfile(float mu, float solref0,
                                                  float solref1,
                                                  const float* solimp) {
    scene::ContactParamsIn out;
    out.mu1 = mu;
    out.mu2 = mu;
    out.solref[0] = solref0;
    out.solref[1] = solref1;
    for (uint32_t i = 0u; i < 5u; ++i) out.solimp[i] = solimp[i];
    return out;
}

__device__ scene::ContactParamsIn ReadContactProfile(
    const float* table, uint32_t count, uint32_t index,
    const scene::ContactParamsIn& fallback) {
    if (table == nullptr || index >= count) return fallback;
    const float* w = table + static_cast<size_t>(index) * nk::kContactProfileWordCount;
    if (__float_as_uint(w[nk::kContactProfileSchema]) != nk::kContactProfileSchemaVersion ||
        (__float_as_uint(w[nk::kContactProfileFlags]) & ~nk::kContactProfileSupportedFlags) != 0u)
        return fallback;
    scene::ContactParamsIn out;
    out.mu1 = w[nk::kContactProfileMu1];
    out.mu2 = w[nk::kContactProfileMu2];
    out.solref[0] = w[nk::kContactProfileSolref0];
    out.solref[1] = w[nk::kContactProfileSolref1];
    for (uint32_t i = 0u; i < 5u; ++i) out.solimp[i] = w[nk::kContactProfileSolimp0 + i];
    out.solmix = w[nk::kContactProfileSolmix];
    out.margin = w[nk::kContactProfileMargin];
    out.gap = w[nk::kContactProfileGap];
    out.priority = static_cast<int32_t>(__float_as_uint(w[nk::kContactProfilePriority]));
    out.condim = static_cast<uint8_t>(__float_as_uint(w[nk::kContactProfileCondim]));
    return out;
}

__device__ uint64_t HashContactProfileWord(uint64_t hash, uint32_t word) {
    hash ^= static_cast<uint64_t>(word);
    hash *= 0x100000001b3ull;
    return hash;
}

__device__ uint64_t HashMergedContactProfile(const scene::MergedContactParams& p) {
    uint64_t hash = kContactProfileHashDomain;
    hash = HashContactProfileWord(hash, nk::kContactProfileSchemaVersion);
    hash = HashContactProfileWord(hash, nk::kContactProfileFlagRefsafe);
    hash = HashContactProfileWord(hash, __float_as_uint(p.mu1));
    hash = HashContactProfileWord(hash, __float_as_uint(p.mu2));
    for (uint32_t i = 0u; i < 2u; ++i) hash = HashContactProfileWord(hash, __float_as_uint(p.solref[i]));
    for (uint32_t i = 0u; i < 5u; ++i) hash = HashContactProfileWord(hash, __float_as_uint(p.solimp[i]));
    hash = HashContactProfileWord(hash, __float_as_uint(p.margin));
    hash = HashContactProfileWord(hash, __float_as_uint(p.gap));
    hash = HashContactProfileWord(hash, p.condim);
    return hash == 0u ? 0x9e3779b97f4a7c15ull : hash;
}

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

__forceinline__ __device__ uint32_t JointDofCount(ArticulationJointType type) {
    return JointDofCountDevice(type);
}

// ===========================================================================
// SHARED + FUSED-family kernels — MOVED VERBATIM from the transitional
// ops/solve_articulated.cu , which this file replaces. Bodies are
// line-by-line ports of articulation_jacobian.cu / articulation_contacts.cu.
// ===========================================================================

// One thread per contact. Walks the ancestor-joint chain (contact link -> root)
// in fixed order, writing each ancestor DOF's Jacobian entry into the contact's
// own dof_stride-wide output slice. No atomics, no cross-contact aliasing.
// (The union family launches this over ROW slots — same kernel, the per-slot
// inputs are gathered per row; an inactive row carries the kInvalidLink
// sentinel and is skipped. Consumers gate on the current row side before reading
// J, so only a live articulation row needs initialization.)
// C3 (general contact pipeline) — CONTACT POINT/NORMAL FRAME DECISION.
// The unified contact buffer (the FUSED contact_*, the union ucontact_*, and the
// general PairDriven manifold) stores the contact POINT in WORLD space and the
// contact NORMAL (A->B) in WORLD space. Newton stores points in BODY frame +
// normal in world (contact_data.py); Nuka keeps WORLD point + world normal for
// minimal D1 churn, because this chain-Jacobian kernel ALREADY consumes a world
// point + world normal directly (contact_point_world / contact_normal_world
// below) — re-keying to body-frame storage would require a per-row world-recompose
// with no functional gain before the general path is wired. Body-frame storage is deferred to the
// collide-once / reuse-across-substeps optimization (out of scope for the initial wiring). This
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

    // AccumulateChainJointForce uses +=. Zero exactly the live articulation row
    // here instead of clearing both dense rows_per_env x dof arrays every step;
    // inactive and rigid/particle rows are never consumed by the gated K4 kernels.
    for (uint32_t k = 0u; k < dof_stride; ++k) out_row[k] = 0.0f;

    // Walk from the contact's link up to the root, emitting each ancestor joint's
    // column via the shared chain-J builder. The scalar-direction column is the
    // wrench (f=normal, tau=0) case; the deposit path feeds it a full wrench. The
    // out_row slice is zeroed above, so the helper's += writes once.
    uint32_t link = contact_link;
    while (link != kInvalidLink) {
        if (JointDofCount(state.joint_type[link]) != 0u) {
            const uint32_t dof_index = LocalDofIndexDevice(state, offset, link);
            AccumulateChainJointForce(state, articulation, link, dof_index,
                                      dof_stride, point, normal,
                                      math::Vec3{0.0f, 0.0f, 0.0f}, out_row);
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
// UNION-family kernels (NEW).
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

// EmitUnionRowsKernel (the UnionCsr K2 row emitter) was DELETED here. The
// ONE general path emits its rows via EmitPairDrivenRowsKernel below.

// K1 (PairDriven): pack the per-ARTICULATION flat qdot tiles. One thread per
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
    // Read flags first; the ~99% inactive rows bail on 4 bytes, not the 128-byte
    // struct. Active rows then read only the side fields they use (same values).
    const NkRow* const rowp = urows + rs;
    if (!(rowp->flags & nk::nk_row_flags::kActive) || rowp->a.kind != kNkSideArtic) {
        return;  // only articulation rows carry a chain-J / w pair.
    }
    const size_t tile = static_cast<size_t>(rowp->a.index) * dof_stride * dof_stride;
    const float* const Minv = m_inv + tile + static_cast<size_t>(r) * dof_stride;
    const float* const J = chain_jacobian + static_cast<size_t>(rs) * dof_stride;
    float acc = 0.0f;
    for (uint32_t c = 0u; c < dof_stride; ++c) {
        acc += Minv[c] * J[c];
    }
    row_minv_jt[gid] = acc;
}

// (union-era ComputeRowMeffKernel deleted — never launched; superseded by the
// live ComputeRowMeffPairDrivenKernel and ComputeContactEffectiveMassKernel.)

// ===========================================================================
// GENERAL CONTACT PIPELINE (PairDriven) PairDriven-family assembly .
//
// The third OpAssembleRows branch (the union/fused arms are LITERALLY unchanged,
// the H1 union golden + go2 FusedFoot golden D1 invariant). Per active unified
// contact slot (the pair-driven narrowphase's ucontact_* manifold + the C1/C2
// (a,b) collidable ids) it resolves each side via the registry (body_to_link /
// body_to_articulation / shape_table body_id) into {kNkSideArtic, real artic id
// , owning link}, {kNkSideRigid, body row}, or {kNkSideStatic}, and fills a
// COUPLED two-sided NkRow + the side-A chain-J gather (row_cj_*) AND the side-B
// chain-J gather (row_cj_*_b). The chain-J kernel + K4a/K4b then run with
// the side-B second passes. The Jv reduction / two-sided apply / mixed-island
// coupling are ALL the existing SolveUnionRowWarp + island solver, unchanged.
// ===========================================================================

// Per-slot layout (FIXED worst case, mirroring the union per-manifold layout) —
// the SHARED constants from nk_row.hpp so the cook (row budget), the schedule, and
// this assembly all agree. Each point expands to one block normal and two
// tangent rows; the normal row performs the analytic cone projection.
constexpr uint32_t kPdPtsPerSlot = nk::kPairDrivenPtsPerSlot;   // 4
constexpr uint32_t kPdTangentRows = nk::kPairDrivenTangentRowsPerPt; // 2
constexpr uint32_t kPdRowsPerSlot = nk::kPairDrivenRowsPerSlot; // 12
constexpr uint32_t kPdParticlePtsPerSlot =
    nk::kPairDrivenParticlePtsPerSlot;                           // 1
constexpr uint32_t kPdParticleRowsPerSlot =
    nk::kPairDrivenParticleRowsPerSlot;                          // 3

// Resolve a contact side -> reaction side. The side-kind TAG (nk::kUContactSide*)
// is consulted FIRST: it declares whether `index` is a body-local collidable row
// or a global particle id, so a non-body index never reaches the shape_table
// body-row lookup. A particle channel resolves to the particle side with the
// GLOBAL particle id carried in `index` (the narrowphase wrote it global). For a
// body-local index: body_id < 0 (the caller's already-loaded shape body_id) is
// static; an articulation-link body row resolves to (real artic id, owning global
// link); a free-rigid body row resolves to the rigid side. Returns the side kind;
// out_artic / out_link valid only for the artic kind; out_body for the rigid kind;
// out_particle for particle. env-major: body_to_* are TEMPLATE-local (per:body),
// so the global body row == env*bodies_per_env + local; the global link ==
// env*base_link + template_link; the global artic == env*artics_per_env + local.
__device__ uint32_t ResolvePairSide(uint32_t side_kind,
                                    int32_t body_id,
                                    const uint32_t* body_to_link,
                                    const uint32_t* body_to_articulation,
                                    uint32_t env, uint32_t index,
                                    uint32_t bodies_per_env, uint32_t base_link_count,
                                    uint32_t artics_per_env,
                                    uint32_t* out_artic, uint32_t* out_link,
                                    uint32_t* out_body, uint32_t* out_particle) {
    *out_artic = ~0u; *out_link = ~0u; *out_body = ~0u; *out_particle = ~0u;
    if (side_kind == nk::kUContactSideParticle) {
        // The narrowphase carries the GLOBAL particle id in `index`; the particle
        // side is a point mass (jang stays zero) keyed by that id.
        *out_particle = index;
        return kNkSideParticle;
    }
    if (side_kind != nk::kUContactSideBody) {
        return kNkSideStatic;  // unknown channel: no reaction here.
    }
    const uint32_t local_body = index;
    if (body_id < 0) {
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

// Particle systems have no collidable profile row, so they use the canonical
// model defaults with only their authored per-system isotropic friction replaced.

// One thread per (env x candidate slot). Emits the slot's NkRow block from the
// unified manifold. Uses ComputeCompliantRow and the canonical tangent basis
// shared with warm-start persistence; side A/B are the candidate collidables.
__global__ void EmitPairDrivenRowsKernel(
    const uint32_t* __restrict__ ucontact_count,
    const math::Vec3* __restrict__ ucontact_point,
    const math::Vec3* __restrict__ ucontact_normal,
    const math::Vec3* __restrict__ ucontact_tangent1,
    const math::Vec3* __restrict__ ucontact_tangent2,
    const float* __restrict__ ucontact_depth,
    const uint32_t* __restrict__ ucontact_a,
    const uint32_t* __restrict__ ucontact_b,
    const uint32_t* __restrict__ ucontact_a_kind,
    const uint32_t* __restrict__ ucontact_b_kind,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ shape_table,
    const uint32_t* __restrict__ body_to_link,
    const uint32_t* __restrict__ body_to_articulation,
    const float* __restrict__ mat_buckets,
    uint32_t num_material_buckets,
    uint64_t* __restrict__ contact_material,
    uint32_t n_soft_particles, uint32_t particles_per_env,
    float particle_soft_friction, float particle_fluid_friction,
    float solref0, float solref1,
    float solimp0, float solimp1, float solimp2, float solimp3, float solimp4,
    float dt, float baumgarte_max_velocity,
    uint32_t env_count, uint32_t slot_count, uint32_t rows_per_env,
    uint32_t full_row_slot_count,
    uint32_t bodies_per_env, uint32_t base_link_count, uint32_t artics_per_env,
    NkRow* __restrict__ urows, float* __restrict__ lambda,
    uint32_t* __restrict__ row_cj_link, math::Vec3* __restrict__ row_cj_point,
    math::Vec3* __restrict__ row_cj_dir,
    uint32_t* __restrict__ row_cj_link_b, math::Vec3* __restrict__ row_cj_point_b,
    math::Vec3* __restrict__ row_cj_dir_b,
    uint32_t* __restrict__ row_count,
    float* __restrict__ row_penetration) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_count;
    if (gid >= total) return;
    const uint32_t env = gid / slot_count;
    const uint32_t slot = gid - env * slot_count;
    const bool compact_particle_slot = slot >= full_row_slot_count;
    const uint32_t points_per_slot =
        compact_particle_slot ? kPdParticlePtsPerSlot : kPdPtsPerSlot;
    const uint32_t base = env * rows_per_env +
        (compact_particle_slot
             ? full_row_slot_count * kPdRowsPerSlot +
                   (slot - full_row_slot_count) * kPdParticleRowsPerSlot
             : slot * kPdRowsPerSlot);

    const uint32_t n_active = ucontact_count[gid];

    // Resolve the two sides ONCE per slot (same a/b for every manifold point).
    uint32_t kind_a = kNkSideStatic, kind_b = kNkSideStatic;
    uint32_t art_a = ~0u, link_a = ~0u, body_a = ~0u, part_a = ~0u;
    uint32_t art_b = ~0u, link_b = ~0u, body_b = ~0u, part_b = ~0u;
    uint32_t local_a = ~0u, local_b = ~0u;
    uint32_t side_kind_a = nk::kUContactSideBody, side_kind_b = nk::kUContactSideBody;
    int32_t bid_a = -1, bid_b = -1;
    uint32_t profile_a = 0u, profile_b = 0u;
    if (n_active > 0u) {
        local_a = ucontact_a[static_cast<size_t>(gid) * 4u];
        local_b = ucontact_b[static_cast<size_t>(gid) * 4u];
        side_kind_a = ucontact_a_kind[static_cast<size_t>(gid) * 4u];
        side_kind_b = ucontact_b_kind[static_cast<size_t>(gid) * 4u];
        // body_id only meaningful for a body-local index (material read below).
        if (side_kind_a == nk::kUContactSideBody) {
            const PrimShapeDev shape = LoadPrimShape(shape_table, local_a);
            bid_a = shape.body_id;
            profile_a = shape.contact_profile_index;
        }
        if (side_kind_b == nk::kUContactSideBody) {
            const PrimShapeDev shape = LoadPrimShape(shape_table, local_b);
            bid_b = shape.body_id;
            profile_b = shape.contact_profile_index;
        }
        kind_a = ResolvePairSide(side_kind_a, bid_a, body_to_link,
                                 body_to_articulation, env, local_a, bodies_per_env,
                                 base_link_count, artics_per_env, &art_a, &link_a,
                                 &body_a, &part_a);
        kind_b = ResolvePairSide(side_kind_b, bid_b, body_to_link,
                                 body_to_articulation, env, local_b, bodies_per_env,
                                 base_link_count, artics_per_env, &art_b, &link_b,
                                 &body_b, &part_b);
    }
    const uint32_t idx_a = (kind_a == kNkSideArtic) ? art_a
                          : (kind_a == kNkSideRigid) ? body_a
                          : (kind_a == kNkSideParticle) ? part_a : ~0u;
    const uint32_t idx_b = (kind_b == kNkSideArtic) ? art_b
                          : (kind_b == kNkSideRigid) ? body_b
                          : (kind_b == kNkSideParticle) ? part_b : ~0u;

    math::Vec3 com_a{0, 0, 0}, com_b{0, 0, 0};
    if (kind_a == kNkSideRigid) com_a = body_pose[idx_a].position;
    if (kind_b == kNkSideRigid) com_b = body_pose[idx_b].position;

    const float default_solimp[5] = {solimp0, solimp1, solimp2, solimp3, solimp4};
    const scene::ContactParamsIn default_profile =
        DefaultProfile(1.0f, solref0, solref1, default_solimp);
    scene::ContactParamsIn side_a = ReadContactProfile(
        mat_buckets, num_material_buckets, profile_a, default_profile);
    scene::ContactParamsIn side_b = ReadContactProfile(
        mat_buckets, num_material_buckets, profile_b, default_profile);
    if (kind_a == kNkSideParticle) {
        const uint32_t local = particles_per_env > 0u ? part_a % particles_per_env : part_a;
        const float mu = local < n_soft_particles ? particle_soft_friction
                                                   : particle_fluid_friction;
        side_a = DefaultProfile(mu, solref0, solref1, default_solimp);
    }
    if (kind_b == kNkSideParticle) {
        const uint32_t local = particles_per_env > 0u ? part_b % particles_per_env : part_b;
        const float mu = local < n_soft_particles ? particle_soft_friction
                                                   : particle_fluid_friction;
        side_b = DefaultProfile(mu, solref0, solref1, default_solimp);
    }
    const scene::MergedContactParams merged = scene::MergeContactParams(side_a, side_b);
    const float* solref = merged.solref;
    const float* solimp = merged.solimp;
    const float mu1 = merged.mu1;
    const float mu2 = merged.mu2;
    contact_material[gid] = n_active > 0u ? HashMergedContactProfile(merged) : 0u;
    const float kFltMaxLocal = kFltMax;

    uint32_t active_rows = 0u;
    for (uint32_t p = 0u; p < points_per_slot; ++p) {
        const bool live = p < n_active && p < 4u;
        const size_t mp = static_cast<size_t>(gid) * 4u + p;
        const uint32_t normal_row = base + p;          // pts normal rows first.
        const uint32_t tangent1_row = base + points_per_slot + p;
        const uint32_t tangent2_row = base + 2u * points_per_slot + p;

        math::Vec3 n{0, 0, 1};
        math::Vec3 point{0, 0, 0};
        math::Vec3 t0{1, 0, 0}, t1v{0, 1, 0};
        if (live) {
            n = NormalizedHostExpr(ucontact_normal[mp]);
            point = ucontact_point[mp];
            t0 = ucontact_tangent1[mp];
            t1v = ucontact_tangent2[mp];
        }

        // ---- normal row ------------------------------------------------------
        {
            const uint32_t rs = normal_row;
            NkRow row{};
            lambda[rs] = 0.0f;
            row_penetration[rs] = 0.0f;
            row_cj_link[rs] = kInvalidLink;
            row_cj_link_b[rs] = kInvalidLink;
            if (live) {
                const float pos = -ucontact_depth[mp];
                // Geometric penetration (positive when overlapping) for the
                // split-impulse position pass; the velocity solve uses aref.
                row_penetration[rs] = fmaxf(ucontact_depth[mp], 0.0f);
                const constraint::CompliantContactRow compliant =
                    constraint::ComputeCompliantRow(solref, solimp, pos, pos,
                                                    /*vel=*/0.0f, /*invweight=*/1.0f,
                                                    dt, /*refsafe=*/true);
                row.flags = nk::nk_row_flags::kActive;
                if (merged.condim == 3u) row.flags |= nk::nk_row_flags::kBlockNormal;
                row.group_first = base;
                row.group_normal_count = points_per_slot;
                row.env = env;
                // Cap aref so the target separating velocity aref*dt <=
                // baumgarte_max_velocity (+inf default => byte-identical): bounded recovery.
                row.rhs = fminf(compliant.aref_bias, baumgarte_max_velocity / dt);
                row.compliance_alpha = compliant.R;
                row.lower = 0.0f;
                row.upper = kFltMaxLocal;
                row.mu = mu1;
                row.reserved[0] = mu2;
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

        // ---- tangent rows ----------------------------------------------------
        const uint32_t tangent_rows[2] = {tangent1_row, tangent2_row};
        const math::Vec3 tangent_dirs[2] = {t0, t1v};
        for (uint32_t k = 0u; k < kPdTangentRows; ++k) {
            const uint32_t rs = tangent_rows[k];
            NkRow row{};
            lambda[rs] = 0.0f;
            row_penetration[rs] = 0.0f;
            row_cj_link[rs] = kInvalidLink;
            row_cj_link_b[rs] = kInvalidLink;
            if (live && merged.condim == 3u) {
                const math::Vec3 dir = tangent_dirs[k];
                row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kBlockTangent;
                row.group_first = base;
                row.group_normal_count = points_per_slot;
                row.env = env;
                row.rhs = 0.0f;
                row.compliance_alpha = 0.0f;
                row.lower = -kFltMaxLocal;
                row.upper = kFltMaxLocal;
                row.mu = mu1;
                row.reserved[0] = mu2;
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

// Stable per-link lower/upper slots appended after the contact footprint.
// A scalar joint bound is a one-sided row with J = +/-e_joint.
__global__ void EmitJointLimitRowsKernel(
    ArticulationDeviceState state,
    const float* __restrict__ lower_limits,
    const float* __restrict__ upper_limits,
    const uint8_t* __restrict__ limit_flags,
    float dt, float baumgarte_max_velocity,
    uint32_t env_count, uint32_t base_link_count,
    uint32_t rows_per_env, uint32_t contact_rows_per_env,
    uint32_t dof_stride,
    NkRow* __restrict__ urows,
    float* __restrict__ lambda,
    float* __restrict__ chain_jacobian,
    uint32_t* __restrict__ row_cj_link,
    uint32_t* __restrict__ row_cj_link_b,
    float* __restrict__ row_penetration,
    uint32_t* __restrict__ row_count) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) return;
    const uint32_t env = link / base_link_count;
    if (env >= env_count) return;
    const uint32_t local_link = link - env * base_link_count;
    const bool scalar_joint = JointDofCount(state.joint_type[link]) == 1u;
    const uint8_t authored = limit_flags != nullptr ? limit_flags[link] : 0u;
    const uint32_t articulation = scalar_joint
                                      ? state.link_to_articulation[link] : ~0u;
    uint32_t dof = 0u;
    if (scalar_joint) {
        const uint32_t offset = state.articulation_link_offset[articulation];
        dof = LocalDofIndexDevice(state, offset, link);
    }
    for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t rs = env * rows_per_env + contact_rows_per_env +
                            local_link * 2u + side;
        NkRow row{};
        row_cj_link[rs] = kInvalidLink;
        row_cj_link_b[rs] = kInvalidLink;
        row_penetration[rs] = 0.0f;
        float* const J = chain_jacobian + static_cast<size_t>(rs) * dof_stride;
        for (uint32_t k = 0u; k < dof_stride; ++k) J[k] = 0.0f;
        const uint8_t bit = static_cast<uint8_t>(1u << side);
        if (scalar_joint && (authored & bit) != 0u && dt > 0.0f) {
            const float sign = side == 0u ? 1.0f : -1.0f;
            const float bound = side == 0u ? lower_limits[link] : upper_limits[link];
            const float signed_distance =
                side == 0u ? state.q[link] - bound : bound - state.q[link];
            const float target_velocity = -signed_distance / dt;
            const float capped_velocity =
                fminf(target_velocity, baumgarte_max_velocity);
            row.flags = nk::nk_row_flags::kActive |
                        (side == 0u ? nk::nk_row_flags::kJointLimitLower
                                    : nk::nk_row_flags::kJointLimitUpper);
            row.group_first = rs;
            row.group_normal_count = 1u;
            row.env = env;
            row.rhs = capped_velocity / dt;
            row.lower = 0.0f;
            row.upper = kFltMax;
            row.a.kind = kNkSideArtic;
            row.a.index = articulation;
            J[dof] = sign;
            row_penetration[rs] = fmaxf(-signed_distance, 0.0f);
            atomicAdd(&row_count[env], 1u);
        } else {
            lambda[rs] = 0.0f;
        }
        urows[rs] = row;
    }
}

// K4a-B: w_b = M^-1 J_b^T per articulation SIDE-B row . Mirrors
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
    // Read flags first; inactive rows bail on 4 bytes, not the 128-byte struct.
    const NkRow* const rowp = urows + rs;
    if (!(rowp->flags & nk::nk_row_flags::kActive) || rowp->b.kind != kNkSideArtic) {
        return;
    }
    const size_t tile = static_cast<size_t>(rowp->b.index) * dof_stride * dof_stride;
    const float* const Minv = m_inv + tile + static_cast<size_t>(r) * dof_stride;
    const float* const J = chain_jacobian_b + static_cast<size_t>(rs) * dof_stride;
    float acc = 0.0f;
    for (uint32_t c = 0u; c < dof_stride; ++c) acc += Minv[c] * J[c];
    row_minv_jt_b[gid] = acc;
}

__device__ float PairDrivenSideCoupling(
    const NkRowSide& lhs, uint32_t lhs_side, uint32_t lhs_row,
    const NkRowSide& rhs, uint32_t rhs_side, uint32_t rhs_row,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
    const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ row_minv_jt_b,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass, uint32_t dof_stride) {
    if (lhs.kind != rhs.kind || lhs.index != rhs.index) return 0.0f;
    switch (lhs.kind) {
        case kNkSideRigid: {
            const float im = body_inv_mass[lhs.index];
            const math::Vec3 ii = body_inv_inertia[lhs.index];
            return im * Dot3(lhs.jlin, rhs.jlin) +
                   lhs.jang.x * rhs.jang.x * ii.x +
                   lhs.jang.y * rhs.jang.y * ii.y +
                   lhs.jang.z * rhs.jang.z * ii.z;
        }
        case kNkSideArtic: {
            const float* const J = (lhs_side == 0u ? chain_jacobian
                                                   : chain_jacobian_b) +
                                   static_cast<size_t>(lhs_row) * dof_stride;
            const float* const w = (rhs_side == 0u ? row_minv_jt
                                                   : row_minv_jt_b) +
                                   static_cast<size_t>(rhs_row) * dof_stride;
            float coupling = 0.0f;
            for (uint32_t r = 0u; r < dof_stride; ++r) coupling += J[r] * w[r];
            return coupling;
        }
        case kNkSideParticle:
            return particle_inv_mass != nullptr
                       ? particle_inv_mass[lhs.index] * Dot3(lhs.jlin, rhs.jlin)
                       : 0.0f;
        default:
            return 0.0f;
    }
}

__device__ float PairDrivenRowCoupling(
    const NkRow& lhs, uint32_t lhs_row, const NkRow& rhs, uint32_t rhs_row,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
    const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ row_minv_jt_b,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass, uint32_t dof_stride) {
    const NkRowSide lhs_sides[2] = {lhs.a, lhs.b};
    const NkRowSide rhs_sides[2] = {rhs.a, rhs.b};
    float coupling = 0.0f;
    for (uint32_t i = 0u; i < 2u; ++i) {
        for (uint32_t j = 0u; j < 2u; ++j) {
            coupling += PairDrivenSideCoupling(
                lhs_sides[i], i, lhs_row, rhs_sides[j], j, rhs_row,
                chain_jacobian, row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                body_inv_mass, body_inv_inertia, particle_inv_mass, dof_stride);
        }
    }
    return coupling;
}

__device__ void StoreContactBlockInverse(NkRow* normal, float k00, float k01,
                                         float k02, float k11, float k12,
                                         float k22) {
    const float c00 = k11 * k22 - k12 * k12;
    const float c01 = k02 * k12 - k01 * k22;
    const float c02 = k01 * k12 - k02 * k11;
    const float c11 = k00 * k22 - k02 * k02;
    const float c12 = k01 * k02 - k00 * k12;
    const float c22 = k00 * k11 - k01 * k01;
    const float determinant = k00 * c00 + k01 * c01 + k02 * c02;
    const float scale = fmaxf(fmaxf(k00, k11), k22);
    const float threshold = 1.0e-12f * scale * scale * scale;
    if (!(determinant > threshold) || !isfinite(determinant)) {
        for (uint32_t i = 1u; i < 7u; ++i) normal->reserved[i] = 0.0f;
        return;
    }
    const float inv_det = 1.0f / determinant;
    normal->reserved[1] = c00 * inv_det;
    normal->reserved[2] = c01 * inv_det;
    normal->reserved[3] = c02 * inv_det;
    normal->reserved[4] = c11 * inv_det;
    normal->reserved[5] = c12 * inv_det;
    normal->reserved[6] = c22 * inv_det;
}

// Computes scalar masses for every row and the full symmetric ContactBlock3
// inverse on each normal owner.
__global__ void ComputeRowMeffPairDrivenKernel(
    NkRow* __restrict__ urows,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
    const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ row_minv_jt_b,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass,
    uint32_t total_rows, uint32_t dof_stride, float* __restrict__ row_meff) {
    const uint32_t rs = blockIdx.x * blockDim.x + threadIdx.x;
    if (rs >= total_rows) return;
    if (!(urows[rs].flags & nk::nk_row_flags::kActive)) {
        row_meff[rs] = 0.0f;
        return;
    }
    NkRow& row = urows[rs];
    float diagonal = PairDrivenRowCoupling(
        row, rs, row, rs, chain_jacobian, row_minv_jt, chain_jacobian_b,
        row_minv_jt_b, body_inv_mass, body_inv_inertia, particle_inv_mass,
        dof_stride);
    diagonal += row.compliance_alpha;
    row_meff[rs] = diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
    if (!(row.flags & nk::nk_row_flags::kBlockNormal)) return;

    const uint32_t point = (rs - row.group_first) % row.group_normal_count;
    const uint32_t tangent1_row = row.group_first + row.group_normal_count + point;
    const uint32_t tangent2_row = row.group_first + 2u * row.group_normal_count + point;
    const NkRow& tangent1 = urows[tangent1_row];
    const NkRow& tangent2 = urows[tangent2_row];
    if (!(tangent1.flags & nk::nk_row_flags::kBlockTangent) ||
        !(tangent2.flags & nk::nk_row_flags::kBlockTangent)) {
        for (uint32_t i = 1u; i < 7u; ++i) row.reserved[i] = 0.0f;
        return;
    }
    const float k00 = diagonal;
    const float k11 = PairDrivenRowCoupling(
        tangent1, tangent1_row, tangent1, tangent1_row, chain_jacobian,
        row_minv_jt, chain_jacobian_b, row_minv_jt_b, body_inv_mass,
        body_inv_inertia, particle_inv_mass, dof_stride) + tangent1.compliance_alpha;
    const float k22 = PairDrivenRowCoupling(
        tangent2, tangent2_row, tangent2, tangent2_row, chain_jacobian,
        row_minv_jt, chain_jacobian_b, row_minv_jt_b, body_inv_mass,
        body_inv_inertia, particle_inv_mass, dof_stride) + tangent2.compliance_alpha;
    const float k01 = 0.5f * (
        PairDrivenRowCoupling(row, rs, tangent1, tangent1_row, chain_jacobian,
                              row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                              body_inv_mass, body_inv_inertia, particle_inv_mass,
                              dof_stride) +
        PairDrivenRowCoupling(tangent1, tangent1_row, row, rs, chain_jacobian,
                              row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                              body_inv_mass, body_inv_inertia, particle_inv_mass,
                              dof_stride));
    const float k02 = 0.5f * (
        PairDrivenRowCoupling(row, rs, tangent2, tangent2_row, chain_jacobian,
                              row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                              body_inv_mass, body_inv_inertia, particle_inv_mass,
                              dof_stride) +
        PairDrivenRowCoupling(tangent2, tangent2_row, row, rs, chain_jacobian,
                              row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                              body_inv_mass, body_inv_inertia, particle_inv_mass,
                              dof_stride));
    const float k12 = 0.5f * (
        PairDrivenRowCoupling(tangent1, tangent1_row, tangent2, tangent2_row,
                              chain_jacobian, row_minv_jt, chain_jacobian_b,
                              row_minv_jt_b, body_inv_mass, body_inv_inertia,
                              particle_inv_mass, dof_stride) +
        PairDrivenRowCoupling(tangent2, tangent2_row, tangent1, tangent1_row,
                              chain_jacobian, row_minv_jt, chain_jacobian_b,
                              row_minv_jt_b, body_inv_mass, body_inv_inertia,
                              particle_inv_mass, dof_stride));
    StoreContactBlockInverse(&row, k00, k01, k02, k11, k12, k22);
}

__device__ bool ContactRowOffsets(uint32_t env, uint32_t slot, uint32_t point,
                                  uint32_t rows_per_env,
                                  uint32_t full_row_slot_count,
                                  uint32_t* normal_row, uint32_t* tangent1_row,
                                  uint32_t* tangent2_row) {
    const bool compact = slot >= full_row_slot_count;
    const uint32_t points_per_slot =
        compact ? kPdParticlePtsPerSlot : kPdPtsPerSlot;
    if (point >= points_per_slot) return false;
    const uint32_t base = env * rows_per_env +
        (compact
             ? full_row_slot_count * kPdRowsPerSlot +
                   (slot - full_row_slot_count) * kPdParticleRowsPerSlot
             : slot * kPdRowsPerSlot);
    *normal_row = base + point;
    *tangent1_row = base + points_per_slot + point;
    *tangent2_row = base + 2u * points_per_slot + point;
    return true;
}

__device__ bool IsCurrentContactPoint(
    const uint32_t* __restrict__ ucontact_count, uint32_t point_index);

__global__ void PrepareContactWarmStartKernel(
    const uint32_t* __restrict__ ucontact_count,
    const uint64_t* __restrict__ current_pair,
    const uint64_t* __restrict__ current_feature,
    const math::Vec3* __restrict__ current_normal,
    const math::Vec3* __restrict__ current_tangent1,
    const math::Vec3* __restrict__ current_tangent2,
    const uint64_t* __restrict__ current_material,
    const uint64_t* __restrict__ cache_pair,
    const uint64_t* __restrict__ cache_feature,
    const float* __restrict__ cache_lambda,
    const math::Vec3* __restrict__ cache_normal,
    const math::Vec3* __restrict__ cache_tangent1,
    const math::Vec3* __restrict__ cache_tangent2,
    const uint64_t* __restrict__ cache_material,
    const uint32_t* __restrict__ cache_age,
    uint32_t env_count, uint32_t slot_count, uint32_t rows_per_env,
    uint32_t full_row_slot_count, uint32_t decay_steps,
    float* __restrict__ lambda) {
    const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t points_per_env = slot_count * 4u;
    if (point_index >= env_count * points_per_env) return;
    const uint32_t env = point_index / points_per_env;
    const uint32_t local_point = point_index - env * points_per_env;
    const uint32_t slot = local_point / 4u;
    const uint32_t point = local_point & 3u;
    const uint32_t contact_slot = env * slot_count + slot;
    if (point >= ucontact_count[contact_slot]) return;

    const uint64_t pair = current_pair[point_index];
    const uint64_t feature = current_feature[point_index];
    const uint64_t material = current_material[contact_slot];
    const uint32_t begin = env * points_per_env;
    const uint32_t end = begin + points_per_env;
    for (uint32_t i = begin; i < point_index; ++i) {
        if (!IsCurrentContactPoint(ucontact_count, i)) continue;
        if (current_pair[i] == pair && current_feature[i] == feature &&
            current_material[i / 4u] == material) return;
    }
    uint32_t match = ~0u;
    for (uint32_t i = begin; i < end; ++i) {
        if (cache_pair[i] == pair && cache_feature[i] == feature &&
            cache_material[i] == material && cache_age[i] < decay_steps) {
            match = i;
            break;
        }
    }
    if (match == ~0u) return;
    const math::Vec3 n = current_normal[point_index];
    if (n.Dot(cache_normal[match]) <= 0.25f) return;

    uint32_t normal_row = 0u;
    uint32_t tangent1_row = 0u;
    uint32_t tangent2_row = 0u;
    if (!ContactRowOffsets(env, slot, point, rows_per_env,
                           full_row_slot_count, &normal_row, &tangent1_row,
                           &tangent2_row)) return;
    const float normal_impulse = fmaxf(cache_lambda[match * 3u], 0.0f);
    const math::Vec3 old_tangent_impulse =
        cache_tangent1[match] * cache_lambda[match * 3u + 1u] +
        cache_tangent2[match] * cache_lambda[match * 3u + 2u];
    const float tangent1 = old_tangent_impulse.Dot(current_tangent1[point_index]);
    const float tangent2 = old_tangent_impulse.Dot(current_tangent2[point_index]);
    lambda[normal_row] = normal_impulse;
    lambda[tangent1_row] = tangent1;
    lambda[tangent2_row] = tangent2;
}

__device__ void ClearContactCachePoint(
    uint32_t point_index, uint64_t* cache_pair, uint64_t* cache_feature,
    float* cache_lambda, math::Vec3* cache_normal,
    math::Vec3* cache_tangent1, math::Vec3* cache_tangent2,
    uint64_t* cache_material, uint32_t* cache_age) {
    cache_pair[point_index] = 0u;
    cache_feature[point_index] = 0u;
    cache_lambda[point_index * 3u + 0u] = 0.0f;
    cache_lambda[point_index * 3u + 1u] = 0.0f;
    cache_lambda[point_index * 3u + 2u] = 0.0f;
    cache_normal[point_index] = {0.0f, 0.0f, 0.0f};
    cache_tangent1[point_index] = {0.0f, 0.0f, 0.0f};
    cache_tangent2[point_index] = {0.0f, 0.0f, 0.0f};
    cache_material[point_index] = 0u;
    cache_age[point_index] = 0u;
}

__device__ bool IsCurrentContactPoint(
    const uint32_t* __restrict__ ucontact_count, uint32_t point_index) {
    const uint32_t point = point_index & 3u;
    return point < ucontact_count[point_index / 4u];
}

__global__ void SnapshotContactCacheKernel(
    uint32_t point_count,
    const uint64_t* __restrict__ cache_pair,
    const uint64_t* __restrict__ cache_feature,
    const float* __restrict__ cache_lambda,
    const math::Vec3* __restrict__ cache_normal,
    const math::Vec3* __restrict__ cache_tangent1,
    const math::Vec3* __restrict__ cache_tangent2,
    const uint64_t* __restrict__ cache_material,
    const uint32_t* __restrict__ cache_age,
    uint64_t* __restrict__ snapshot_pair,
    uint64_t* __restrict__ snapshot_feature,
    float* __restrict__ snapshot_lambda,
    math::Vec3* __restrict__ snapshot_normal,
    math::Vec3* __restrict__ snapshot_tangent1,
    math::Vec3* __restrict__ snapshot_tangent2,
    uint64_t* __restrict__ snapshot_material,
    uint32_t* __restrict__ snapshot_age) {
    const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= point_count) return;
    snapshot_pair[point_index] = cache_pair[point_index];
    snapshot_feature[point_index] = cache_feature[point_index];
    for (uint32_t k = 0u; k < 3u; ++k)
        snapshot_lambda[point_index * 3u + k] = cache_lambda[point_index * 3u + k];
    snapshot_normal[point_index] = cache_normal[point_index];
    snapshot_tangent1[point_index] = cache_tangent1[point_index];
    snapshot_tangent2[point_index] = cache_tangent2[point_index];
    snapshot_material[point_index] = cache_material[point_index];
    snapshot_age[point_index] = cache_age[point_index];
}

__device__ bool SameContactKey(uint64_t pair_a, uint64_t feature_a,
                               uint64_t material_a, uint64_t pair_b,
                               uint64_t feature_b, uint64_t material_b) {
    return pair_a == pair_b && feature_a == feature_b && material_a == material_b;
}

__global__ void MarkContactCacheRebuildKernel(
    const uint32_t* __restrict__ ucontact_count,
    const uint64_t* __restrict__ current_pair,
    const uint64_t* __restrict__ current_feature,
    const uint64_t* __restrict__ current_material,
    const uint64_t* __restrict__ snapshot_pair,
    const uint64_t* __restrict__ snapshot_feature,
    const uint64_t* __restrict__ snapshot_material,
    const uint32_t* __restrict__ snapshot_age,
    uint32_t env_count, uint32_t slot_count, uint32_t decay_steps,
    uint32_t* __restrict__ current_owner,
    uint32_t* __restrict__ old_keep) {
    const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t points_per_env = slot_count * 4u;
    if (point_index >= env_count * points_per_env) return;
    const uint32_t begin = (point_index / points_per_env) * points_per_env;
    const uint32_t end = begin + points_per_env;

    bool owner = IsCurrentContactPoint(ucontact_count, point_index);
    const uint64_t material = current_material[point_index / 4u];
    for (uint32_t i = begin; owner && i < point_index; ++i) {
        if (!IsCurrentContactPoint(ucontact_count, i)) continue;
        if (SameContactKey(current_pair[i], current_feature[i],
                           current_material[i / 4u], current_pair[point_index],
                           current_feature[point_index], material)) owner = false;
    }
    current_owner[point_index] = owner ? 1u : 0u;

    bool keep = snapshot_pair[point_index] != 0u && decay_steps != 0u &&
                snapshot_age[point_index] + 1u < decay_steps;
    for (uint32_t i = begin; keep && i < point_index; ++i) {
        if (SameContactKey(snapshot_pair[i], snapshot_feature[i], snapshot_material[i],
                           snapshot_pair[point_index], snapshot_feature[point_index],
                           snapshot_material[point_index])) keep = false;
    }
    for (uint32_t i = begin; keep && i < end; ++i) {
        if (!IsCurrentContactPoint(ucontact_count, i)) continue;
        if (SameContactKey(current_pair[i], current_feature[i],
                           current_material[i / 4u], snapshot_pair[point_index],
                           snapshot_feature[point_index],
                           snapshot_material[point_index])) keep = false;
    }
    old_keep[point_index] = keep ? 1u : 0u;
}

__global__ void RebuildContactCacheKernel(
    const uint64_t* __restrict__ current_pair,
    const uint64_t* __restrict__ current_feature,
    const math::Vec3* __restrict__ current_normal,
    const math::Vec3* __restrict__ current_tangent1,
    const math::Vec3* __restrict__ current_tangent2,
    const uint64_t* __restrict__ current_material,
    const float* __restrict__ lambda,
    const uint64_t* __restrict__ snapshot_pair,
    const uint64_t* __restrict__ snapshot_feature,
    const float* __restrict__ snapshot_lambda,
    const math::Vec3* __restrict__ snapshot_normal,
    const math::Vec3* __restrict__ snapshot_tangent1,
    const math::Vec3* __restrict__ snapshot_tangent2,
    const uint64_t* __restrict__ snapshot_material,
    const uint32_t* __restrict__ snapshot_age,
    const uint32_t* __restrict__ current_owner,
    const uint32_t* __restrict__ old_keep,
    uint32_t env_count, uint32_t slot_count, uint32_t rows_per_env,
    uint32_t full_row_slot_count,
    uint64_t* __restrict__ cache_pair,
    uint64_t* __restrict__ cache_feature,
    float* __restrict__ cache_lambda,
    math::Vec3* __restrict__ cache_normal,
    math::Vec3* __restrict__ cache_tangent1,
    math::Vec3* __restrict__ cache_tangent2,
    uint64_t* __restrict__ cache_material,
    uint32_t* __restrict__ cache_age) {
    const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t points_per_env = slot_count * 4u;
    if (point_index >= env_count * points_per_env) return;
    const uint32_t env = point_index / points_per_env;
    const uint32_t local_point = point_index - env * points_per_env;
    const uint32_t slot = local_point / 4u;
    const uint32_t point = local_point & 3u;
    const uint32_t begin = env * points_per_env;
    const uint32_t end = begin + points_per_env;

    if (current_owner[point_index] != 0u) {
        uint32_t normal_row = 0u, tangent1_row = 0u, tangent2_row = 0u;
        if (ContactRowOffsets(env, slot, point, rows_per_env,
                              full_row_slot_count, &normal_row, &tangent1_row,
                              &tangent2_row)) {
            cache_pair[point_index] = current_pair[point_index];
            cache_feature[point_index] = current_feature[point_index];
            cache_lambda[point_index * 3u] = fmaxf(lambda[normal_row], 0.0f);
            cache_lambda[point_index * 3u + 1u] = lambda[tangent1_row];
            cache_lambda[point_index * 3u + 2u] = lambda[tangent2_row];
            cache_normal[point_index] = current_normal[point_index];
            cache_tangent1[point_index] = current_tangent1[point_index];
            cache_tangent2[point_index] = current_tangent2[point_index];
            cache_material[point_index] = current_material[point_index / 4u];
            cache_age[point_index] = 0u;
            return;
        }
    }

    uint32_t free_rank = 0u;
    for (uint32_t i = begin; i < point_index; ++i)
        if (current_owner[i] == 0u) ++free_rank;
    uint32_t source = ~0u;
    uint32_t old_rank = 0u;
    for (uint32_t i = begin; i < end; ++i) {
        if (old_keep[i] == 0u) continue;
        if (old_rank++ == free_rank) {
            source = i;
            break;
        }
    }
    if (source != ~0u) {
        cache_pair[point_index] = snapshot_pair[source];
        cache_feature[point_index] = snapshot_feature[source];
        for (uint32_t k = 0u; k < 3u; ++k)
            cache_lambda[point_index * 3u + k] = snapshot_lambda[source * 3u + k];
        cache_normal[point_index] = snapshot_normal[source];
        cache_tangent1[point_index] = snapshot_tangent1[source];
        cache_tangent2[point_index] = snapshot_tangent2[source];
        cache_material[point_index] = snapshot_material[source];
        cache_age[point_index] = snapshot_age[source] + 1u;
        return;
    }
    ClearContactCachePoint(point_index, cache_pair, cache_feature, cache_lambda,
                           cache_normal, cache_tangent1, cache_tangent2,
                           cache_material, cache_age);
}

Status OpContactWarmStart(const ModelView& /*model*/, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ContactWarmStartParams*>(params);
    if (p == nullptr) return Status::Failed;
    const uint32_t point_count = p->env_count * p->slot_count * 4u;
    if (point_count == 0u) return Status::Ok;
    constexpr uint32_t kBlock = 128u;
    const uint32_t blocks = (point_count + kBlock - 1u) / kBlock;
    if (p->phase == 0u) {
        LaunchCuda(PrepareContactWarmStartKernel, dim3(blocks), dim3(kBlock), 0u,
                   stream, data.ucontact_count, data.ucontact_id_pair,
                   data.ucontact_id_feature, data.ucontact_normal,
                   data.ucontact_tangent1, data.ucontact_tangent2,
                   data.contact_material, data.contact_cache_pair,
                   data.contact_cache_feature, data.contact_cache_lambda,
                   data.contact_cache_normal, data.contact_cache_tangent1,
                   data.contact_cache_tangent2, data.contact_cache_material,
                   data.contact_cache_age, p->env_count, p->slot_count,
                   p->rows_per_env, p->full_row_slot_count, p->decay_steps,
                   data.lambda);
    } else {
        LaunchCuda(SnapshotContactCacheKernel, dim3(blocks), dim3(kBlock), 0u,
                   stream, point_count, data.contact_cache_pair,
                   data.contact_cache_feature, data.contact_cache_lambda,
                   data.contact_cache_normal, data.contact_cache_tangent1,
                   data.contact_cache_tangent2, data.contact_cache_material,
                   data.contact_cache_age, data.contact_cache_snapshot_pair,
                   data.contact_cache_snapshot_feature,
                   data.contact_cache_snapshot_lambda,
                   data.contact_cache_snapshot_normal,
                   data.contact_cache_snapshot_tangent1,
                   data.contact_cache_snapshot_tangent2,
                   data.contact_cache_snapshot_material,
                   data.contact_cache_snapshot_age);
        LaunchCuda(MarkContactCacheRebuildKernel, dim3(blocks), dim3(kBlock), 0u,
                   stream, data.ucontact_count, data.ucontact_id_pair,
                   data.ucontact_id_feature, data.contact_material,
                   data.contact_cache_snapshot_pair,
                   data.contact_cache_snapshot_feature,
                   data.contact_cache_snapshot_material,
                   data.contact_cache_snapshot_age, p->env_count,
                   p->slot_count, p->decay_steps,
                   data.contact_cache_current_owner,
                   data.contact_cache_old_keep);
        LaunchCuda(RebuildContactCacheKernel, dim3(blocks), dim3(kBlock), 0u,
                   stream, data.ucontact_id_pair, data.ucontact_id_feature,
                   data.ucontact_normal, data.ucontact_tangent1,
                   data.ucontact_tangent2, data.contact_material, data.lambda,
                   data.contact_cache_snapshot_pair,
                   data.contact_cache_snapshot_feature,
                   data.contact_cache_snapshot_lambda,
                   data.contact_cache_snapshot_normal,
                   data.contact_cache_snapshot_tangent1,
                   data.contact_cache_snapshot_tangent2,
                   data.contact_cache_snapshot_material,
                   data.contact_cache_snapshot_age,
                   data.contact_cache_current_owner,
                   data.contact_cache_old_keep, p->env_count, p->slot_count,
                   p->rows_per_env, p->full_row_slot_count,
                   data.contact_cache_pair, data.contact_cache_feature,
                   data.contact_cache_lambda, data.contact_cache_normal,
                   data.contact_cache_tangent1, data.contact_cache_tangent2,
                   data.contact_cache_material, data.contact_cache_age);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpAssembleRowsPairDriven(const ModelView& model, const DataView& data,
                                const AssembleRowsParams* p, cudaStream_t stream) {
    if (p->env_count == 0u || p->rows_per_env == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t total_rows = p->env_count * p->rows_per_env;
    if (cudaMemsetAsync(data.row_count, 0,
                        static_cast<size_t>(p->env_count) * sizeof(uint32_t),
                        stream) != cudaSuccess) {
        return Status::Failed;
    }
    const bool has_artic = p->max_dof > 0u && p->articulation_count > 0u;
    const uint32_t artics_per_env =
        (p->articulation_count > 0u && p->env_count > 0u)
            ? (p->articulation_count / p->env_count) : 0u;

    // K1: pack the per-articulation flat qdot tiles (one tile per co-resident
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

    // K2: emit the pair-driven contact rows.
    if (p->union_slot_count > 0u) {
        const uint32_t total = p->env_count * p->union_slot_count;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(EmitPairDrivenRowsKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   static_cast<const uint32_t*>(data.ucontact_count),
                   static_cast<const math::Vec3*>(data.ucontact_point),
                   static_cast<const math::Vec3*>(data.ucontact_normal),
                   static_cast<const math::Vec3*>(data.ucontact_tangent1),
                   static_cast<const math::Vec3*>(data.ucontact_tangent2),
                   static_cast<const float*>(data.ucontact_depth),
                   static_cast<const uint32_t*>(data.ucontact_a),
                   static_cast<const uint32_t*>(data.ucontact_b),
                   static_cast<const uint32_t*>(data.ucontact_a_kind),
                   static_cast<const uint32_t*>(data.ucontact_b_kind),
                   static_cast<const math::Transform*>(data.body_pose),
                   static_cast<const float*>(model.shape_table),
                   static_cast<const uint32_t*>(model.body_to_link),
                   static_cast<const uint32_t*>(model.body_to_articulation),
                   static_cast<const float*>(data.mat_buckets),
                   p->num_material_buckets, data.contact_material,
                   p->n_soft_particles, p->particles_per_env,
                   p->particle_soft_friction, p->particle_fluid_friction,
                   p->solref[0], p->solref[1],
                   p->solimp[0], p->solimp[1], p->solimp[2], p->solimp[3], p->solimp[4],
                   p->dt, p->baumgarte_max_velocity,
                   p->env_count, p->union_slot_count, p->rows_per_env,
                   p->full_row_slot_count,
                   p->bodies_per_env, p->base_link_count, artics_per_env,
                   reinterpret_cast<NkRow*>(data.urows), data.lambda,
                   data.row_cj_link, data.row_cj_point, data.row_cj_dir,
                   data.row_cj_link_b, data.row_cj_point_b, data.row_cj_dir_b,
                   data.row_count, data.row_penetration);
    }

    if (has_artic &&
        p->rows_per_env >= p->contact_rows_per_env + p->base_link_count * 2u) {
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        const uint32_t limit_blocks =
            (p->total_link_count + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(EmitJointLimitRowsKernel, dim3(limit_blocks), dim3(kBlockSize),
                   0u, stream, state, model.joint_limit_lower,
                   model.joint_limit_upper, model.joint_limit_flags,
                   p->dt, p->baumgarte_max_velocity, p->env_count,
                   p->base_link_count, p->rows_per_env,
                   p->contact_rows_per_env, p->max_dof,
                   reinterpret_cast<NkRow*>(data.urows), data.lambda,
                   data.chain_jacobian, data.row_cj_link,
                   data.row_cj_link_b, data.row_penetration, data.row_count);
    }

    if (has_artic) {
        // K3: chain Jacobians for side A AND side B (the SAME multi-artic kernel,
        // fed each side's per-row link/point/dir gather). Each live articulation
        // row clears its own slice before the fixed-order += walk.
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
                   stream, reinterpret_cast<NkRow*>(data.urows),
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.row_minv_jt),
                   static_cast<const float*>(data.chain_jacobian_b),
                   static_cast<const float*>(data.row_minv_jt_b),
                   static_cast<const float*>(data.body_inv_mass),
                   static_cast<const math::Vec3*>(data.body_inv_inertia),
                   static_cast<const float*>(data.particle_inv_mass),
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

// OpAssembleRowsUnion (the UnionCsr K1-K4 assembly orchestrator) was
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
    // dead fallthrough: the FUSED family is unreachable (no cooked
    // model selects it). Retained until the enum collapse removes it.
    return OpAssembleRowsFused(model, data, p, stream);
}

} // namespace

void RegisterNkAssembleRowsOps() {
    SetCudaOp(NkOp::AssembleRows, &OpAssembleRows);
    SetCudaOp(NkOp::ContactWarmStart, &OpContactWarmStart);
}

} // namespace nuka::phi
