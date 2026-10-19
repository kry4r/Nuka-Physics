// All contact providers assemble solved rows and mass response through one common contract.

#include <cuda_runtime.h>
#include "phi/backend_cuda/ops/articulation_mass.cuh"
#include <cub/block/block_scan.cuh>

#include <algorithm>

#include "phi/backend_cuda/ops/contact_row_layout.cuh"
#include "constraint/solref_solimp.hpp"  // ComputeCompliantRow (HD)
#include "nk/contact/contact_profile.hpp"
#include "nk/solve/collidable_owner.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "scene/contact_filter.hpp"
#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/contact_cache.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // LoadPrimShape (PairDriven side resolve)
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"
#include "phi/backend_cuda/ops/vertex_blocks.cuh"

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


__forceinline__ __device__ uint32_t JointDofCount(ArticulationJointType type) {
    return JointDofCountDevice(type);
}


__global__ void ComputeContactChainJacobianKernel(
    ArticulationDeviceState state,
    const uint32_t* contact_link_indices,
    const math::Vec3* contact_point_world,
    const float* contact_normal_world, uint32_t normal_stride,
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
    const math::Vec3 normal = NormalizeOrUp(
        *reinterpret_cast<const math::Vec3*>(contact_normal_world + size_t{contact} * normal_stride));
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

// Merge repeated articulation owners before inversion to retain exact force cancellation.
__global__ void MergeSharedArticulationJacobianKernel(
    const NkRow* __restrict__ rows, uint32_t count, uint32_t dof_stride,
    float* __restrict__ jacobian, float* __restrict__ jacobian_b) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const NkRow& contact = rows[row];
    if (!(contact.flags & nk::nk_row_flags::kActive) ||
        contact.a.kind != kNkSideArtic || contact.b.kind != kNkSideArtic ||
        contact.a.index != contact.b.index) return;
    const size_t base = static_cast<size_t>(row) * dof_stride;
    for (uint32_t i = 0u; i < dof_stride; ++i) {
        jacobian[base + i] += jacobian_b[base + i];
        jacobian_b[base + i] = 0.0f;
    }
}


// Vec3 normalize — EXACT host math::Vec3::Normalized expression (len = sqrt,
// 1e-12 guard, component-wise division), so the emitted row directions match
// the legacy host EmitCompliantContactRows at the FP floor. (NOT the fused
// family's NormalizeOrUp — a DIFFERENT legacy function.)
__forceinline__ __device__ math::Vec3 NormalizedHostExpr(math::Vec3 v) {
    const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-12f) return {0.0f, 0.0f, 0.0f};
    return {v.x / len, v.y / len, v.z / len};
}

// Each articulation's flat velocity tile follows its actual joint DOF prefix; padding is zero.
__global__ void PackQdotFlatMultiKernel(ArticulationDeviceState state, uint32_t dof_stride,
                                        float* __restrict__ qdot_flat) {
    BeginProgrammaticKernel();
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = state.articulation_count * dof_stride;
    if (gid >= total) return;
    const uint32_t articulation = gid / dof_stride;
    const uint32_t dof = gid - articulation * dof_stride;
    uint32_t link = ~0u, component = ~0u;
    if (!ArticulationDofLocation(state, articulation, dof, &link, &component)) {
        qdot_flat[gid] = 0.0f;
        return;
    }
    qdot_flat[gid] = component != ~0u ? state.link_velocity[link].v[component] : state.qdot[link];
}

// A coupled DOF moves with its root, v_t = s v_root, so each row's Jacobian folds onto the
// root columns, J <- J Z. Root columns are never folded themselves, so the order is fixed.
template <bool side_b>
__global__ void FoldMimicJacobianKernel(const NkRow* __restrict__ urows,
                                        const uint32_t* __restrict__ root_dof,
                                        const float* __restrict__ root_scale,
                                        uint32_t total_rows,
                                        uint32_t dof_stride,
                                        float* __restrict__ chain_jacobian) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows || !(urows[row].flags & nk::nk_row_flags::kActive)) return;
    const NkRowSide& endpoint = side_b ? urows[row].b : urows[row].a;
    if (endpoint.kind != kNkSideArtic) return;
    const size_t tile = size_t{endpoint.index} * dof_stride;
    float* const jacobian = chain_jacobian + size_t{row} * dof_stride;
    for (uint32_t k = 0u; k < dof_stride; ++k) {
        const uint32_t root = root_dof[tile + k];
        if (root == k || root >= dof_stride) continue;
        jacobian[root] += root_scale[tile + k] * jacobian[k];
        jacobian[k] = 0.0f;
    }
}

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
constexpr uint32_t kContactRowsBlock = 128u;
constexpr uint32_t kPdTangentRows = nk::kPairDrivenTangentRowsPerPt;


// Particle indices are global; body indices name environment-local collidables.
// Invalid endpoints return ~0u and never read a state array.
__device__ uint32_t ResolvePairSide(uint32_t side_kind,
                                    int32_t body_id,
                                    const uint32_t* body_to_link,
                                    const uint32_t* body_to_articulation,
                                    const uint32_t* body_collidable_body,
                                    uint32_t env, uint32_t index,
                                    uint32_t bodies_per_env, uint32_t base_link_count,
                                    uint32_t artics_per_env, uint32_t particles_per_env,
                                    uint32_t grid_nodes_per_env, uint32_t endpoints_per_env,
                                    uint32_t terms_per_env, const nk::PointEndpointRange* ranges,
                                    const nk::PointEndpointTerm* terms, uint32_t* out_artic, uint32_t* out_link,
                                    uint32_t* out_body, uint32_t* out_particle) {
    *out_artic = ~0u; *out_link = ~0u; *out_body = ~0u; *out_particle = ~0u;
    if (side_kind == nk::kUContactSidePointEndpoint) {
        const uint64_t first = uint64_t{env} * endpoints_per_env;
        if (!ranges || !terms || index < first || index >= first + endpoints_per_env) return ~0u;
        const auto range = ranges[index];
        const uint64_t begin = uint64_t{env} * terms_per_env;
        if (range.count == 0u || range.first < begin || uint64_t{range.first} + range.count > begin + terms_per_env)
            return ~0u;
        for (uint32_t i = 0u; i < range.count; ++i) {
            const auto& term = terms[range.first + i];
            if (i > 0u && terms[range.first + i - 1u].Key() >= term.Key()) return ~0u;
            if (term.kind != kNkSideParticle && term.kind != kNkSideGrid) return ~0u;
            const uint32_t stride = term.kind == kNkSideGrid ? grid_nodes_per_env : particles_per_env;
            if (term.index < uint64_t{env} * stride || term.index >= uint64_t{env + 1u} * stride) return ~0u;
            for (const auto& column : term.column)
                if (!isfinite(column.x) || !isfinite(column.y) || !isfinite(column.z)) return ~0u;
        }
        *out_particle = index;
        return kNkSidePointEndpoint;
    }
    if (side_kind == nk::kUContactSideParticle) {
        const uint64_t first = static_cast<uint64_t>(env) * particles_per_env;
        if (index < first || index >= first + particles_per_env) return ~0u;
        *out_particle = index;
        return kNkSideParticle;
    }
    if (side_kind == nk::kUContactSideGrid) {
        const uint64_t first = static_cast<uint64_t>(env) * grid_nodes_per_env;
        if (index < first || index >= first + grid_nodes_per_env) return ~0u;
        *out_particle = index;
        return kNkSideGrid;
    }
    if (side_kind == nk::kUContactSideBoundary)
        return index < nk::kMpmBoundaryCount ? kNkSideStatic : ~0u;
    if (side_kind != nk::kUContactSideBody) return ~0u;
    const nk::CollidableOwner owner = nk::ResolveCollidableOwner(
        body_id, env, index, bodies_per_env, base_link_count, artics_per_env,
        body_to_link, body_to_articulation, body_collidable_body);
    *out_artic = owner.articulation;
    *out_link = owner.link;
    *out_body = owner.body;
    return owner.kind;
}

// Particle systems have no collidable profile row, so they use the canonical
// model defaults with only their authored per-system isotropic friction replaced.

// Each environment prefixes rigid contact ranges and advances its occupied row extent.
__global__ void InitializePairDrivenRangesKernel(
    const uint32_t* counts, uint32_t slot_count, uint32_t rigid_slots,
    uint32_t rows_per_env, uint32_t contact_rows, uint32_t* ranges,
    uint32_t* row_count, uint32_t* extent, uint64_t* contact_material) {
    using Scan = cub::BlockScan<uint32_t, kContactRowsBlock>;
    __shared__ typename Scan::TempStorage scratch;
    __shared__ uint32_t carry;
    const uint32_t env = blockIdx.x;
    BeginProgrammaticKernel();
    if (threadIdx.x == 0u) {
        carry = 0u;
        row_count[env] = 0u;
        if (extent != nullptr) {
            extent[2u * env] = extent[2u * env + 1u];
            extent[2u * env + 1u] = 1u;
        }
    }
    __syncthreads();
    const uint32_t bound = ContactRowBound(extent, env, contact_rows);
    for (uint32_t tile = 0u; tile < rigid_slots; tile += blockDim.x) {
        const uint32_t slot = tile + threadIdx.x;
        const uint32_t points = slot < rigid_slots ? counts[env * slot_count + slot] : 0u;
        uint32_t prefix;
        Scan(scratch).ExclusiveSum(points, prefix);
        const uint32_t first = env * rows_per_env + (carry + prefix) * (1u + kPdTangentRows);
        if (slot < rigid_slots) {
            const size_t index = size_t{env * slot_count + slot} * 2u;
            ranges[index] = first;
            ranges[index + 1u] = points;
            if (first - env * rows_per_env < bound) contact_material[env * slot_count + slot] = 0u;
        }
        __syncthreads();
        if (threadIdx.x + 1u == blockDim.x) carry += prefix + points;
        __syncthreads();
    }
}

// Row initialization and fixed particle ranges span the device independently of environment count.
__global__ void ClearPairDrivenRowsKernel(
    uint32_t total_rows, uint32_t rows_per_env, uint32_t contact_rows,
    uint32_t slot_count, uint32_t rigid_slots, uint32_t rigid_rows, const uint32_t* extent,
    uint32_t* ranges, NkRow* urows, float* lambda, uint32_t* row_cj_link, uint32_t* row_cj_link_b,
    float* row_penetration, float* row_damping, uint64_t* contact_material) {
    BeginProgrammaticKernel();
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < total_rows;
         row += gridDim.x * blockDim.x) {
        const uint32_t env = row / rows_per_env;
        const uint32_t local = row % rows_per_env;
        const uint32_t bound = ContactRowBound(extent, env, contact_rows);
        row_damping[row] = 0.0f;
        if (local < bound) {
            lambda[row] = 0.0f;
            row_penetration[row] = 0.0f;
            row_cj_link[row] = kInvalidLink;
            row_cj_link_b[row] = kInvalidLink;
            if (urows[row].flags != 0u) urows[row].flags = 0u;
        }
        if (local < slot_count - rigid_slots) {
            const uint32_t slot = env * slot_count + rigid_slots + local;
            const uint32_t first = env * rows_per_env + rigid_rows +
                local * nk::kPairDrivenParticleRowsPerSlot;
            ranges[size_t{slot} * 2u] = first;
            ranges[size_t{slot} * 2u + 1u] = nk::kPairDrivenParticlePtsPerSlot;
            if (first - env * rows_per_env < bound) contact_material[slot] = 0u;
        }
    }
}

// Each candidate slot emits its manifold rows using ComputeCompliantRow.
// The tangent basis and collidable sides are shared with warm-start persistence.
__global__ void EmitPairDrivenRowsKernel(
    const uint32_t* __restrict__ ucontact_count,
    const uint32_t* __restrict__ row_ranges,
    const math::Vec3* __restrict__ ucontact_point,
    const math::Vec3* __restrict__ ucontact_witness_a,
    const math::Vec3* __restrict__ ucontact_witness_b,
    const math::Vec3* __restrict__ ucontact_normal,
    const float* __restrict__ ucontact_depth,
    const uint32_t* __restrict__ ucontact_a,
    const uint32_t* __restrict__ ucontact_b,
    const uint32_t* __restrict__ ucontact_a_kind,
    const uint32_t* __restrict__ ucontact_b_kind,
    const uint32_t* __restrict__ ucontact_law,
    const float* __restrict__ ucontact_friction,
    const math::Transform* __restrict__ body_pose,
    const math::Transform* __restrict__ body_inertial_frame,
    const float* __restrict__ shape_table,
    const uint32_t* __restrict__ body_to_link,
    const uint32_t* __restrict__ body_to_articulation,
    const uint32_t* __restrict__ body_collidable_body,
    const float* __restrict__ mat_buckets,
    uint32_t num_material_buckets,
    uint64_t* __restrict__ contact_material,
    uint32_t particles_per_env, uint32_t grid_nodes_per_env,
    uint32_t endpoints_per_env, uint32_t terms_per_env,
    const nk::PointEndpointRange* endpoint_ranges, const nk::PointEndpointTerm* endpoint_terms,
    const float* __restrict__ particle_friction,
    float solref0, float solref1,
    float solimp0, float solimp1, float solimp2, float solimp3, float solimp4,
    float dt, float baumgarte_max_velocity, float contact_margin, float overlap_recovery_velocity,
    uint32_t env_count, uint32_t slot_count, uint32_t rows_per_env,
    uint32_t full_row_slot_count, uint32_t ogc_slot_count,
    uint32_t bodies_per_env, uint32_t base_link_count, uint32_t artics_per_env,
    NkRow* __restrict__ urows, float* __restrict__ lambda,
    uint32_t* __restrict__ row_cj_link, math::Vec3* __restrict__ row_cj_point,
    uint32_t* __restrict__ row_cj_link_b, math::Vec3* __restrict__ row_cj_point_b,
    uint32_t* __restrict__ row_count,
    float* __restrict__ row_penetration,
    float* __restrict__ row_damping, uint32_t* __restrict__ env_status,
    uint32_t* __restrict__ contact_row_extent) {
    BeginProgrammaticKernel();
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_count;
    if (gid >= total) return;
    const uint32_t env = gid / slot_count;
    const uint32_t slot = gid - env * slot_count;
    const auto range = nkops::ContactRowRangeFor(row_ranges, gid);
    const uint32_t points_per_slot = range.points;
    const uint32_t base = range.first;

    uint32_t n_active = ucontact_count[gid];
    // The last occupied slot of each environment in the warp raises the extent past its rows.
    if (contact_row_extent != nullptr) {
        const uint32_t active = __activemask();
        const uint32_t filled = __ballot_sync(active, n_active > 0u) & __match_any_sync(active, env);
        if (filled != 0u &&
            (threadIdx.x & 31u) == 31u - static_cast<uint32_t>(__clz(static_cast<int>(filled))))
            atomicMax(contact_row_extent + 2u * env + 1u,
                      base - env * rows_per_env + (1u + kPdTangentRows) * points_per_slot + 1u);
    }
    // Empty slots retain their initialized rows.
    if (n_active == 0u) return;
    const uint32_t law = ucontact_law != nullptr ? ucontact_law[gid] : nk::kContactLawCompliant;
    const bool speculative = law == nk::kContactLawSpeculative;
    const bool velocity_only = law == nk::kContactLawVelocity || speculative;

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
        if (side_kind_a == nk::kUContactSideBody && local_a < bodies_per_env) {
            const PrimShapeDev shape = LoadPrimShape(shape_table, local_a);
            bid_a = shape.body_id;
            profile_a = shape.contact_profile_index;
        }
        if (side_kind_b == nk::kUContactSideBody && local_b < bodies_per_env) {
            const PrimShapeDev shape = LoadPrimShape(shape_table, local_b);
            bid_b = shape.body_id;
            profile_b = shape.contact_profile_index;
        }
        kind_a = ResolvePairSide(side_kind_a, bid_a, body_to_link,
                                 body_to_articulation, body_collidable_body,
                                 env, local_a, bodies_per_env,
                                 base_link_count, artics_per_env, particles_per_env, grid_nodes_per_env,
                                 endpoints_per_env, terms_per_env, endpoint_ranges, endpoint_terms,
                                 &art_a, &link_a,
                                 &body_a, &part_a);
        kind_b = ResolvePairSide(side_kind_b, bid_b, body_to_link,
                                 body_to_articulation, body_collidable_body,
                                 env, local_b, bodies_per_env,
                                 base_link_count, artics_per_env, particles_per_env, grid_nodes_per_env,
                                 endpoints_per_env, terms_per_env, endpoint_ranges, endpoint_terms,
                                 &art_b, &link_b,
                                 &body_b, &part_b);
        if (kind_a == ~0u || kind_b == ~0u) {
            if (env_status != nullptr) atomicOr(&env_status[env], kEnvStatusInvalidEndpoint);
            n_active = 0u;
            kind_a = kind_b = kNkSideStatic;
        }
    }
    const uint32_t idx_a = (kind_a == kNkSideArtic) ? art_a
                          : (kind_a == kNkSideRigid) ? body_a
                          : PointMassView::IsPointSide(kind_a) ? part_a : ~0u;
    const uint32_t idx_b = (kind_b == kNkSideArtic) ? art_b
                          : (kind_b == kNkSideRigid) ? body_b
                          : PointMassView::IsPointSide(kind_b) ? part_b : ~0u;

    math::Vec3 com_a{0, 0, 0}, com_b{0, 0, 0};
    if (kind_a == kNkSideRigid) com_a = BodyCenterOfMass(body_pose[idx_a], body_inertial_frame[idx_a]);
    if (kind_b == kNkSideRigid) com_b = BodyCenterOfMass(body_pose[idx_b], body_inertial_frame[idx_b]);

    const float default_solimp[5] = {solimp0, solimp1, solimp2, solimp3, solimp4};
    const float endpoint_mu = ucontact_friction != nullptr ? ucontact_friction[gid] : 0.0f;
    const float default_mu = kind_a == kNkSideGrid || kind_b == kNkSideGrid ? endpoint_mu : 1.0f;
    const scene::ContactParamsIn default_profile =
        DefaultProfile(default_mu, solref0, solref1, default_solimp);
    scene::ContactParamsIn side_a = ReadContactProfile(
        mat_buckets, num_material_buckets, profile_a, default_profile);
    scene::ContactParamsIn side_b = ReadContactProfile(
        mat_buckets, num_material_buckets, profile_b, default_profile);
    if (kind_a == kNkSideParticle)
        side_a = DefaultProfile(particle_friction[part_a - env * particles_per_env],
                                solref0, solref1, default_solimp);
    if (kind_b == kNkSideParticle)
        side_b = DefaultProfile(particle_friction[part_b - env * particles_per_env],
                                solref0, solref1, default_solimp);
    if (kind_a == kNkSideGrid || kind_a == kNkSidePointEndpoint || side_kind_a == nk::kUContactSideBoundary)
        side_a = DefaultProfile(endpoint_mu, solref0, solref1, default_solimp);
    if (kind_b == kNkSideGrid || kind_b == kNkSidePointEndpoint || side_kind_b == nk::kUContactSideBoundary)
        side_b = DefaultProfile(endpoint_mu, solref0, solref1, default_solimp);
    scene::MergedContactParams merged = scene::MergeContactParams(side_a, side_b);
    if (kind_a == kNkSideParticle && side_kind_b == nk::kUContactSideBody) {
        merged.mu1 = scene::ParticleBodyFriction(side_a.mu1, side_b.mu1);
        merged.mu2 = scene::ParticleBodyFriction(side_a.mu2, side_b.mu2);
    } else if (kind_b == kNkSideParticle && side_kind_a == nk::kUContactSideBody) {
        merged.mu1 = scene::ParticleBodyFriction(side_b.mu1, side_a.mu1);
        merged.mu2 = scene::ParticleBodyFriction(side_b.mu2, side_a.mu2);
    }
    if (slot >= full_row_slot_count &&
        slot - full_row_slot_count < ogc_slot_count && n_active > 0u) {
        merged.mu1 = endpoint_mu;
        merged.mu2 = endpoint_mu;
    }
    const float* solref = merged.solref;
    const float* solimp = merged.solimp;
    const float mu1 = merged.mu1;
    const float mu2 = merged.mu2;
    contact_material[gid] = n_active > 0u ? HashMergedContactProfile(merged) : 0u;
    if (n_active > 0u && velocity_only)
        contact_material[gid] = nk::ContactHashWord(contact_material[gid], law);
    const float kFltMaxLocal = kFltMax;

    uint32_t active_rows = 0u;
    for (uint32_t p = 0u; p < points_per_slot; ++p) {
        // A speculative row closes its emitted gap exactly; the margin gate applies to compliant rows.
        const bool live = p < n_active && p < 4u && (speculative ||
            ucontact_depth[static_cast<size_t>(gid) * 4u + p] >=
                merged.gap - fmaxf(contact_margin, merged.margin));
        const size_t mp = static_cast<size_t>(gid) * 4u + p;
        const uint32_t normal_row = base + p;          // pts normal rows first.
        const uint32_t tangent1_row = base + points_per_slot + p;
        const uint32_t tangent2_row = base + 2u * points_per_slot + p;

        math::Vec3 n{0, 0, 1};
        math::Vec3 point{0, 0, 0};
        math::Vec3 point_a{0, 0, 0}, point_b{0, 0, 0};
        math::Vec3 t0{1, 0, 0}, t1v{0, 1, 0};
        if (live) {
            n = NormalizedHostExpr(ucontact_normal[mp]);
            point = ucontact_point[mp];
            const bool ogc_slot = slot >= full_row_slot_count &&
                slot - full_row_slot_count < ogc_slot_count;
            point_a = ogc_slot ? ucontact_witness_a[mp] : point;
            point_b = ogc_slot ? ucontact_witness_b[mp] : point;
            mg::ContactTangentBasis(ucontact_normal[mp], &t0, &t1v);
        }

        // ---- normal row ------------------------------------------------------
        {
            const uint32_t rs = normal_row;
            NkRow row{};
            lambda[rs] = 0.0f;
            row_penetration[rs] = 0.0f;
            row_damping[rs] = 0.0f;
            row_cj_link[rs] = kInvalidLink;
            row_cj_link_b[rs] = kInvalidLink;
            if (live) {
                const float pos = -ucontact_depth[mp];
                // Geometric penetration (positive when overlapping) for the
                // split-impulse position pass; the velocity solve uses aref.
                row_penetration[rs] = speculative ? ucontact_depth[mp] : fmaxf(ucontact_depth[mp], 0.0f);
                constraint::CompliantContactRow compliant{};
                if (!velocity_only)
                    compliant = constraint::ComputeCompliantRow(solref, solimp, pos, pos,
                        /*vel=*/0.0f, /*invweight=*/1.0f, dt, /*refsafe=*/true);
                row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kContactNormal;
                if (merged.condim == 3u) row.flags |= nk::nk_row_flags::kBlockNormal;
                if (law == nk::kContactLawVelocity) row.flags |= nk::nk_row_flags::kVelocityOnly;
                if (speculative) row.flags |= nk::nk_row_flags::kSpeculative;
                row.group_first = base;
                row.group_normal_count = points_per_slot;
                row.env = env;
                // Cap aref so the target separating velocity aref*dt <=
                // baumgarte_max_velocity (+inf default => byte-identical): bounded recovery.
                row.rhs = fminf(compliant.aref_bias, baumgarte_max_velocity / dt);
                // Gap closure belongs to physical velocity; overlap recovers at most overlap_recovery_velocity.
                if (speculative) row.rhs = (fminf(ucontact_depth[mp], 0.0f) +
                    fminf(fmaxf(ucontact_depth[mp], 0.0f), overlap_recovery_velocity * dt)) / (dt * dt);
                row_damping[rs] = compliant.damping;
                row.compliance_alpha = compliant.R;
                row.lower = 0.0f;
                row.upper = kFltMaxLocal;
                row.mu = mu1;
                row.friction_secondary = mu2;
                row.a.kind = kind_a; row.a.index = idx_a; row.a.jlin = n;
                row.b.kind = kind_b; row.b.index = idx_b;
                row.b.jlin = math::Vec3{-n.x, -n.y, -n.z};
                if (kind_a == kNkSideRigid) row.a.jang = (point_a - com_a).Cross(row.a.jlin);
                if (kind_b == kNkSideRigid) row.b.jang = (point_b - com_b).Cross(row.b.jlin);
                if (kind_a == kNkSideArtic) { row_cj_link[rs] = link_a; row_cj_point[rs] = point_a; }
                if (kind_b == kNkSideArtic) { row_cj_link_b[rs] = link_b; row_cj_point_b[rs] = point_b; }
                ++active_rows;
            }
            if (row.flags & nk::nk_row_flags::kActive) urows[rs] = row;
            else urows[rs].flags = 0u;
        }

        // ---- tangent rows ----------------------------------------------------
        const uint32_t tangent_rows[2] = {tangent1_row, tangent2_row};
        const math::Vec3 tangent_dirs[2] = {t0, t1v};
        for (uint32_t k = 0u; k < kPdTangentRows; ++k) {
            const uint32_t rs = tangent_rows[k];
            NkRow row{};
            lambda[rs] = 0.0f;
            row_penetration[rs] = 0.0f;
            row_damping[rs] = 0.0f;
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
                row.friction_secondary = mu2;
                row.a.kind = kind_a; row.a.index = idx_a; row.a.jlin = dir;
                row.b.kind = kind_b; row.b.index = idx_b;
                row.b.jlin = math::Vec3{-dir.x, -dir.y, -dir.z};
                if (kind_a == kNkSideRigid) row.a.jang = (point_a - com_a).Cross(row.a.jlin);
                if (kind_b == kNkSideRigid) row.b.jang = (point_b - com_b).Cross(row.b.jlin);
                if (kind_a == kNkSideArtic) { row_cj_link[rs] = link_a; row_cj_point[rs] = point_a; }
                if (kind_b == kNkSideArtic) { row_cj_link_b[rs] = link_b; row_cj_point_b[rs] = point_b; }
                ++active_rows;
            }
            if (row.flags & nk::nk_row_flags::kActive) urows[rs] = row;
            else urows[rs].flags = 0u;
        }
    }
    if (active_rows > 0u) atomicAdd(&row_count[env], active_rows);
}

// Stable per-link lower/upper slots appended after the contact footprint.
// A scalar joint bound is a one-sided row with J = +/-e_joint.
__device__ void EmitJointLimitRows(
    uint32_t link, const ArticulationDeviceState& state,
    const float* __restrict__ lower_limits,
    const float* __restrict__ upper_limits,
    const uint8_t* __restrict__ limit_flags,
    float dt, float baumgarte_max_velocity, bool position_pass,
    uint32_t env_count, uint32_t base_link_count,
    uint32_t rows_per_env, uint32_t contact_rows_per_env,
    uint32_t dof_stride,
    NkRow* __restrict__ urows,
    float* __restrict__ lambda,
    float* __restrict__ chain_jacobian,
    uint32_t* __restrict__ row_cj_link,
    uint32_t* __restrict__ row_cj_link_b,
    float* __restrict__ row_penetration,
    float* __restrict__ row_damping,
    uint32_t* __restrict__ row_count) {
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
        row_damping[rs] = 0.0f;
        float* const J = chain_jacobian + static_cast<size_t>(rs) * dof_stride;
        for (uint32_t k = 0u; k < dof_stride; ++k) J[k] = 0.0f;
        const uint8_t bit = static_cast<uint8_t>(1u << side);
        if (scalar_joint && (authored & bit) != 0u && dt > 0.0f) {
            const float sign = side == 0u ? 1.0f : -1.0f;
            const float bound = side == 0u ? lower_limits[link] : upper_limits[link];
            const float signed_distance =
                side == 0u ? state.q[link] - bound : bound - state.q[link];
            // With a position pass the row only stops further violation; that pass removes it.
            const float target_velocity =
                -(position_pass ? fmaxf(signed_distance, 0.0f) : signed_distance) / dt;
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

// Joint dry friction is a bounded impulse in the shared coupled velocity solve.
// Its Jacobian is e_joint, with no geometric position correction.
__device__ void EmitJointFrictionRows(
    uint32_t link, const ArticulationDeviceState& state, float dt,
    uint32_t base_link_count, uint32_t rows_per_env,
    uint32_t first_row_per_env, uint32_t dof_stride,
    NkRow* __restrict__ urows, float* __restrict__ lambda,
    float* __restrict__ chain_jacobian,
    uint32_t* __restrict__ row_cj_link, uint32_t* __restrict__ row_cj_link_b,
    float* __restrict__ row_penetration, float* __restrict__ row_damping,
    uint32_t* __restrict__ row_count) {
    const uint32_t env = link / base_link_count;
    const uint32_t local_link = link - env * base_link_count;
    const uint32_t rs = env * rows_per_env + first_row_per_env + local_link;
    NkRow row{};
    row_cj_link[rs] = kInvalidLink;
    row_cj_link_b[rs] = kInvalidLink;
    row_penetration[rs] = 0.0f;
    row_damping[rs] = 0.0f;
    lambda[rs] = 0.0f;
    float* const J = chain_jacobian + static_cast<size_t>(rs) * dof_stride;
    for (uint32_t k = 0u; k < dof_stride; ++k) J[k] = 0.0f;
    const float friction = state.joint_frictionloss[link];
    if (JointDofCount(state.joint_type[link]) == 1u && friction > 0.0f && dt > 0.0f) {
        const uint32_t articulation = state.link_to_articulation[link];
        const uint32_t offset = state.articulation_link_offset[articulation];
        const uint32_t dof = LocalDofIndexDevice(state, offset, link);
        row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kVelocityOnly;
        row.group_first = rs;
        row.group_normal_count = 1u;
        row.env = env;
        row.lower = -friction * dt;
        row.upper = friction * dt;
        row.a.kind = kNkSideArtic;
        row.a.index = articulation;
        J[dof] = 1.0f;
        atomicAdd(&row_count[env], 1u);
    }
    urows[rs] = row;
}

// An actuator's bounded effort is a fixed impulse with J = e_joint; its implicit
// dissipation is folded into the articulation mass instead.
__device__ void EmitJointDriveRows(
    uint32_t link, const ArticulationDeviceState& state, float dt, uint32_t base_link_count,
    uint32_t rows_per_env, uint32_t first_row, uint32_t dof_stride,
    const float* command, const float* lower, const float* upper,
    NkRow* urows, float* lambda, float* chain_jacobian,
    uint32_t* row_cj_link, uint32_t* row_cj_link_b,
    float* row_penetration, float* row_damping, uint32_t* row_count) {
    const uint32_t env = link / base_link_count;
    const uint32_t local = link - env * base_link_count;
    const uint32_t slot = env * rows_per_env + first_row + local;
    NkRow row{};
    row_cj_link[slot] = row_cj_link_b[slot] = kInvalidLink;
    row_penetration[slot] = row_damping[slot] = lambda[slot] = 0.0f;
    float* J = chain_jacobian + static_cast<size_t>(slot) * dof_stride;
    for (uint32_t d = 0u; d < dof_stride; ++d) J[d] = 0.0f;
    if (JointDofCountDevice(state.joint_type[link]) == 1u &&
        fminf(fmaxf(command[link], lower[link]), upper[link]) != 0.0f) {
        const uint32_t articulation = state.link_to_articulation[link];
        const uint32_t offset = state.articulation_link_offset[articulation];
        const uint32_t dof = LocalDofIndexDevice(state, offset, link);
        row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kVelocityOnly;
        row.group_first = slot;
        row.group_normal_count = 1u;
        row.env = env;
        row.a.kind = kNkSideArtic;
        row.a.index = articulation;
        J[dof] = 1.0f;
        const float effort = fminf(fmaxf(command[link], lower[link]), upper[link]);
        row.lower = row.upper = effort * dt;
        atomicAdd(row_count + env, 1u);
    }
    urows[slot] = row;
}

__global__ void EmitJointRowsKernel(
    ArticulationDeviceState state, AssembleRowsParams p,
    const float* lower_limits, const float* upper_limits, const uint8_t* limit_flags,
    DataView data) {
    BeginProgrammaticKernel();
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) return;
    auto* const urows = reinterpret_cast<NkRow*>(data.urows);
    if (p.joint_limit_rows_per_env >= p.base_link_count * 2u)
        EmitJointLimitRows(link, state, lower_limits, upper_limits, limit_flags,
            p.dt, p.baumgarte_max_velocity, p.position_pass != 0u, p.env_count,
            p.base_link_count, p.rows_per_env, p.contact_rows_per_env, p.max_dof,
            urows, data.lambda, data.chain_jacobian, data.row_cj_link,
            data.row_cj_link_b, data.row_penetration, data.row_damping, data.row_count);
    if (p.joint_friction_rows_per_env >= p.base_link_count && p.base_link_count > 0u)
        EmitJointFrictionRows(link, state, p.dt, p.base_link_count, p.rows_per_env,
            p.contact_rows_per_env + p.joint_limit_rows_per_env, p.max_dof,
            urows, data.lambda, data.chain_jacobian, data.row_cj_link,
            data.row_cj_link_b, data.row_penetration, data.row_damping, data.row_count);
    if (p.joint_drive_rows_per_env >= p.base_link_count && p.base_link_count > 0u)
        EmitJointDriveRows(link, state, p.dt, p.base_link_count, p.rows_per_env,
            p.contact_rows_per_env + p.joint_limit_rows_per_env + p.joint_friction_rows_per_env,
            p.max_dof, data.drive_command, data.drive_lower, data.drive_upper,
            urows, data.lambda, data.chain_jacobian, data.row_cj_link,
            data.row_cj_link_b, data.row_penetration, data.row_damping, data.row_count);
}

// L^-1 J of a row's articulated sides; side 0 reads chain_jacobian, side 1 chain_jacobian_b.
struct RowForward {
    const float* side[2] = {nullptr, nullptr};
};

__device__ RowForward ForwardRowSides(const NkRow& row, uint32_t row_index,
    const float* __restrict__ chain_jacobian, const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ mass_factor, uint32_t dof_stride, float* storage) {
    RowForward forward;
    const NkRowSide sides[2] = {row.a, row.b};
    for (uint32_t s = 0u; s < 2u; ++s) {
        if (sides[s].kind != kNkSideArtic) continue;
        float* const y = storage + size_t{s} * dof_stride;
        ForwardMass(mass_factor + size_t{sides[s].index} * dof_stride * dof_stride, dof_stride,
                    (s == 0u ? chain_jacobian : chain_jacobian_b) + size_t{row_index} * dof_stride, nullptr, y);
        forward.side[s] = y;
    }
    return forward;
}

__device__ float PairDrivenSideCoupling(
    const NkRowSide& lhs, const float* lhs_forward,
    const NkRowSide& rhs, const float* rhs_forward,
    const float* __restrict__ mass_factor,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, uint32_t dof_stride) {
    if (PointMassView::IsPointSide(lhs.kind) && PointMassView::IsPointSide(rhs.kind))
        return point_masses.Coupling(lhs, rhs);
    if (lhs.kind != rhs.kind || lhs.index != rhs.index) return 0.0f;
    switch (lhs.kind) {
        case kNkSideRigid: {
            const float im = body_inv_mass[lhs.index];
            return im * Dot3(lhs.jlin, rhs.jlin) +
                   Dot3(lhs.jang, body_world_inv_inertia[lhs.index].Multiply(rhs.jang));
        }
        case kNkSideArtic: {
            const float* const factor = mass_factor + size_t{lhs.index} * dof_stride * dof_stride;
            return MassDot(factor, dof_stride, lhs_forward, rhs_forward);
        }
        case kNkSideParticle:
        case kNkSideGrid: {
            return point_masses.Quad(lhs.kind, lhs.index, lhs.jlin, rhs.jlin);
        }
        default:
            return 0.0f;
    }
}

// Same-side couplings a caller already summed (bit s for side s) replace their merge.
struct SharedSideCoupling {
    uint32_t known = 0u;
    float value[2] = {0.0f, 0.0f};
};

__device__ float PairDrivenRowCoupling(
    const NkRow& lhs, const RowForward& lhs_forward, const NkRow& rhs, const RowForward& rhs_forward,
    const float* __restrict__ mass_factor,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, uint32_t dof_stride, SharedSideCoupling shared = {}) {
    const NkRowSide lhs_sides[2] = {lhs.a, lhs.b};
    const NkRowSide rhs_sides[2] = {rhs.a, rhs.b};
    float coupling = 0.0f;
    for (uint32_t i = 0u; i < 2u; ++i) {
        for (uint32_t j = 0u; j < 2u; ++j) {
            coupling += i == j && ((shared.known >> i) & 1u) != 0u ? shared.value[i]
                : PairDrivenSideCoupling(
                      lhs_sides[i], lhs_forward.side[i], rhs_sides[j], rhs_forward.side[j],
                      mass_factor, body_inv_mass, body_world_inv_inertia, point_masses, dof_stride);
        }
    }
    return coupling;
}

// Point couplings of a block's rows on the endpoint all three share, in one pass over its terms;
// each sum adds terms in the order a pairwise merge does. Order: 11, 22, 01, 02, 12.
__device__ void SharedPointCouplings(const PointMassView& points, const NkRowSide& side,
                                     const math::Vec3 (&jlin)[3], float (&coupling)[5]) {
    for (uint32_t c = 0u; c < 5u; ++c) coupling[c] = 0.0f;
    const uint32_t count = points.Count(side);
    for (uint32_t term = 0u; term < count; ++term) {
        uint32_t kind = side.kind, index = side.index;
        math::Vec3 u[3] = {jlin[0], jlin[1], jlin[2]};
        if (side.kind == nk::kNkSidePointEndpoint) {
            const auto& entry = points.terms[points.ranges[side.index].first + term];
            kind = entry.kind;
            index = entry.index;
            for (uint32_t r = 0u; r < 3u; ++r) u[r] = entry.TransposeMultiply(jlin[r]);
        }
        if (points.InverseMass(kind) == nullptr) continue;
        coupling[0] += points.Quad(kind, index, u[1], u[1]);
        coupling[1] += points.Quad(kind, index, u[2], u[2]);
        coupling[2] += points.Quad(kind, index, u[0], u[1]);
        coupling[3] += points.Quad(kind, index, u[0], u[2]);
        coupling[4] += points.Quad(kind, index, u[1], u[2]);
    }
}

// Tangent rows a block normal owns.
__device__ bool BlockTangentRows(const NkRow* urows, uint32_t normal_row, const NkRow& normal,
                                 uint32_t (&tangent)[2]) {
    if (!(normal.flags & nk::nk_row_flags::kActive) ||
        !(normal.flags & nk::nk_row_flags::kBlockNormal))
        return false;
    const uint32_t point = (normal_row - normal.group_first) % normal.group_normal_count;
    tangent[0] = normal.group_first + normal.group_normal_count + point;
    tangent[1] = normal.group_first + 2u * normal.group_normal_count + point;
    return (urows[tangent[0]].flags & nk::nk_row_flags::kBlockTangent) &&
           (urows[tangent[1]].flags & nk::nk_row_flags::kBlockTangent);
}

__device__ float ContactReferenceVelocity(
    const NkRow& row, uint32_t rs, uint32_t dof_stride,
    const float* chain_jacobian, const float* chain_jacobian_b,
    const float* step_qdot, const math::Vec3* step_body_linear,
    const math::Vec3* step_body_angular, const math::Vec3* step_particle,
    PointMassView point_masses) {
    float velocity = 0.0f;
    const NkRowSide sides[2] = {row.a, row.b};
    for (uint32_t s = 0; s < 2u; ++s) {
        const NkRowSide& side = sides[s];
        if (side.kind == kNkSideArtic) {
            const float* J = (s == 0u ? chain_jacobian : chain_jacobian_b) +
                             static_cast<size_t>(rs) * dof_stride;
            const float* velocity_tile = step_qdot +
                                         static_cast<size_t>(side.index) * dof_stride;
            for (uint32_t d = 0; d < dof_stride; ++d) velocity += J[d] * velocity_tile[d];
        } else if (side.kind == kNkSideRigid) {
            velocity += Dot3(side.jlin, step_body_linear[side.index]) +
                        Dot3(side.jang, step_body_angular[side.index]);
        } else if (PointMassView::IsPointSide(side.kind)) {
            for (uint32_t i = 0u; i < point_masses.Count(side); ++i) {
                const auto term = point_masses.At(side, i);
                if (term.kind == kNkSideParticle && step_particle != nullptr)
                    velocity += Dot3(term.jacobian, step_particle[term.index]);
            }
        }
    }
    return velocity;
}

// Prepares contact compliance, initial-velocity forcing and diagnostic block responses.
__global__ void PrepareContactRowsKernel(
    NkRow* __restrict__ urows,
    math::SymmetricMat3* __restrict__ contact_response,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ mass_factor,
    const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses,
    const float* __restrict__ row_damping,
    const float* __restrict__ step_qdot,
    const math::Vec3* __restrict__ step_body_linear,
    const math::Vec3* __restrict__ step_body_angular,
    const math::Vec3* __restrict__ step_particle,
    uint32_t total_rows, uint32_t dof_stride, float dt, VertexBlockLayout vertex_blocks,
    uint32_t rows_per_env, uint32_t contact_rows_per_env, uint32_t measure,
    const uint32_t* __restrict__ contact_row_extent) {
    const uint32_t rs = blockIdx.x * blockDim.x + threadIdx.x;
    if (rs >= total_rows) return;
    // Contact rows past the active bound have no work.
    const uint32_t env_row = rs % rows_per_env;
    if (env_row < contact_rows_per_env &&
        env_row >= ContactRowBound(contact_row_extent, rs / rows_per_env, contact_rows_per_env)) return;
    if (!(urows[rs].flags & nk::nk_row_flags::kActive)) {
        return;
    }
    NkRow& row = urows[rs];
    if (TouchesVertexBlock(row.a, point_masses, vertex_blocks) ||
        TouchesVertexBlock(row.b, point_masses, vertex_blocks))
        row.flags |= nk::nk_row_flags::kVertexBlock;
    const bool reference = (row.flags & nk::nk_row_flags::kContactNormal) &&
        !(row.flags & (nk::nk_row_flags::kVelocityOnly | nk::nk_row_flags::kSpeculative));
    const bool respond = measure != 0u && (row.flags & nk::nk_row_flags::kBlockNormal);
    if (!reference && !respond) return;
    uint32_t tangent_rows[2] = {0u, 0u};
    const bool block = respond && BlockTangentRows(urows, rs, row, tangent_rows);
    const uint32_t tangent1_row = tangent_rows[0], tangent2_row = tangent_rows[1];
    // Pairs 11, 22, 01, 02, 12 of the normal and its tangents.
    SharedSideCoupling shared[5];
    if (block) {
        const NkRow& tangent1 = urows[tangent1_row];
        const NkRow& tangent2 = urows[tangent2_row];
        for (uint32_t s = 0u; s < 2u; ++s) {
            const NkRowSide& side = s == 0u ? row.a : row.b;
            const NkRowSide& side1 = s == 0u ? tangent1.a : tangent1.b;
            const NkRowSide& side2 = s == 0u ? tangent2.a : tangent2.b;
            if (!PointMassView::IsPointSide(side.kind) || side1.kind != side.kind ||
                side1.index != side.index || side2.kind != side.kind || side2.index != side.index)
                continue;
            float coupling[5];
            SharedPointCouplings(point_masses, side, {side.jlin, side1.jlin, side2.jlin}, coupling);
            for (uint32_t c = 0u; c < 5u; ++c) {
                shared[c].known |= 1u << s;
                shared[c].value[s] = coupling[c];
            }
        }
    }
    // One forward solve per articulated side of the normal and its tangents; couplings are weighted dots.
    float* const storage = static_cast<float*>(alloca(size_t{6u} * dof_stride * sizeof(float)));
    const RowForward forward = ForwardRowSides(row, rs, chain_jacobian, chain_jacobian_b, mass_factor,
                                               dof_stride, storage);
    float diagonal = PairDrivenRowCoupling(
        row, forward, row, forward, mass_factor, body_inv_mass, body_world_inv_inertia, point_masses,
        dof_stride);
    if (reference) {
        // Scale impedance by J*M^-1*J^T so compliance is independent of mass units.
        row.compliance_alpha *= fmaxf(diagonal, 0.0f);
        const float initial_velocity = ContactReferenceVelocity(
            row, rs, dof_stride, chain_jacobian, chain_jacobian_b,
            step_qdot, step_body_linear, step_body_angular, step_particle, point_masses);
        // (1+b*dt)*v_next + R*impulse = v_start - dt*k*d*position.
        row.rhs += initial_velocity / dt;
    }
    diagonal += row.compliance_alpha;
    if (!respond) return;
    if (!block) {
        contact_response[rs] = {};
        return;
    }
    const NkRow& tangent1 = urows[tangent1_row];
    const NkRow& tangent2 = urows[tangent2_row];
    const RowForward forward1 = ForwardRowSides(tangent1, tangent1_row, chain_jacobian, chain_jacobian_b,
                                                mass_factor, dof_stride, storage + size_t{2u} * dof_stride);
    const RowForward forward2 = ForwardRowSides(tangent2, tangent2_row, chain_jacobian, chain_jacobian_b,
                                                mass_factor, dof_stride, storage + size_t{4u} * dof_stride);
    // Normalize by (1+b*dt) to retain a symmetric block with R/(1+b*dt).
    const float damping_scale = 1.0f / (1.0f + row_damping[rs] * dt);
    const float k00 = diagonal - row.compliance_alpha * (1.0f - damping_scale);
    const float k11 = PairDrivenRowCoupling(
        tangent1, forward1, tangent1, forward1, mass_factor, body_inv_mass,
        body_world_inv_inertia, point_masses, dof_stride, shared[0]) + tangent1.compliance_alpha;
    const float k22 = PairDrivenRowCoupling(
        tangent2, forward2, tangent2, forward2, mass_factor, body_inv_mass,
        body_world_inv_inertia, point_masses, dof_stride, shared[1]) + tangent2.compliance_alpha;
    const float k01 = 0.5f * (
        PairDrivenRowCoupling(row, forward, tangent1, forward1, mass_factor, body_inv_mass,
                              body_world_inv_inertia, point_masses, dof_stride, shared[2]) +
        PairDrivenRowCoupling(tangent1, forward1, row, forward, mass_factor, body_inv_mass,
                              body_world_inv_inertia, point_masses, dof_stride, shared[2]));
    const float k02 = 0.5f * (
        PairDrivenRowCoupling(row, forward, tangent2, forward2, mass_factor, body_inv_mass,
                              body_world_inv_inertia, point_masses, dof_stride, shared[3]) +
        PairDrivenRowCoupling(tangent2, forward2, row, forward, mass_factor, body_inv_mass,
                              body_world_inv_inertia, point_masses, dof_stride, shared[3]));
    const float k12 = 0.5f * (
        PairDrivenRowCoupling(tangent1, forward1, tangent2, forward2, mass_factor, body_inv_mass,
                              body_world_inv_inertia, point_masses, dof_stride, shared[4]) +
        PairDrivenRowCoupling(tangent2, forward2, tangent1, forward1, mass_factor, body_inv_mass,
                              body_world_inv_inertia, point_masses, dof_stride, shared[4]));
    contact_response[rs] = {k00, k11, k22, k01, k02, k12};
}

__device__ bool ContactRowOffsets(uint32_t env, uint32_t slot, uint32_t point,
    uint32_t slots_per_env, const uint32_t* ranges, uint32_t* normal_row,
    uint32_t* tangent1_row, uint32_t* tangent2_row) {
    const auto range = nkops::ContactRowRangeFor(ranges, env * slots_per_env + slot);
    if (point >= range.points) return false;
    *normal_row = range.first + point;
    *tangent1_row = range.first + range.points + point;
    *tangent2_row = range.first + 2u * range.points + point;
    return true;
}


// Only compacted current entries can hold a match; rows of other points keep their lambda.
__global__ void PrepareContactWarmStartKernel(
    const math::Vec3* __restrict__ current_normal,
    const float* __restrict__ cache_lambda,
    const math::Vec3* __restrict__ cache_normal,
    const uint32_t* __restrict__ matches,
    const uint32_t* __restrict__ order,
    const uint32_t* __restrict__ begins,
    const uint32_t* __restrict__ ends,
    uint32_t env_count, uint32_t slot_count, const uint32_t* row_ranges, uint32_t blocks_per_env, float* __restrict__ lambda) {
    BeginProgrammaticKernel();
    const uint32_t points_per_env = slot_count * 4u;
    const uint32_t env = blockIdx.x / blocks_per_env;
    const uint32_t part = blockIdx.x % blocks_per_env;
    for (uint32_t entry = begins[env] + part * blockDim.x + threadIdx.x; entry < ends[env];
         entry += blocks_per_env * blockDim.x) {
        const uint32_t point_index = order[entry];
        if (point_index >= env_count * points_per_env) continue;
        const uint32_t match = matches[point_index];
        if (match == ~0u) continue;
        const uint32_t local_point = point_index - env * points_per_env;
        const uint32_t slot = local_point / 4u;
        const uint32_t point = local_point & 3u;
        const math::Vec3 n = current_normal[point_index];
        if (n.Dot(cache_normal[match]) <= 0.25f) continue;
        uint32_t normal_row = 0u, tangent1_row = 0u, tangent2_row = 0u;
        if (!ContactRowOffsets(env, slot, point, slot_count, row_ranges, &normal_row, &tangent1_row,
                               &tangent2_row)) continue;
        const float normal_impulse = fmaxf(cache_lambda[match * 3u], 0.0f);
        math::Vec3 old_t1, old_t2, t1, t2;
        mg::ContactTangentBasis(cache_normal[match], &old_t1, &old_t2);
        mg::ContactTangentBasis(n, &t1, &t2);
        const math::Vec3 old_tangent_impulse =
            old_t1 * cache_lambda[match * 3u + 1u] + old_t2 * cache_lambda[match * 3u + 2u];
        const float tangent1 = old_tangent_impulse.Dot(t1);
        const float tangent2 = old_tangent_impulse.Dot(t2);
        lambda[normal_row] = normal_impulse;
        lambda[tangent1_row] = tangent1;
        lambda[tangent2_row] = tangent2;
    }
}

__device__ void ClearContactCachePoint(
    uint32_t point_index, uint64_t* cache_pair, uint64_t* cache_feature,
    float* cache_lambda, math::Vec3* cache_normal, uint64_t* cache_material, uint32_t* cache_age) {
    cache_pair[point_index] = 0u;
    cache_feature[point_index] = 0u;
    cache_lambda[point_index * 3u + 0u] = 0.0f;
    cache_lambda[point_index * 3u + 1u] = 0.0f;
    cache_lambda[point_index * 3u + 2u] = 0.0f;
    cache_normal[point_index] = {0.0f, 0.0f, 0.0f};
    cache_material[point_index] = 0u;
    cache_age[point_index] = 0u;
}


// The rebuild reads the snapshot only at retained points, so only those are copied; they all lie
// below the env's previous cache bound.
__global__ void SnapshotContactCacheKernel(
    uint32_t points_per_env, uint32_t env_blocks,
    const uint32_t* __restrict__ extent,
    const uint32_t* __restrict__ old_keep,
    const uint64_t* __restrict__ cache_pair,
    const uint64_t* __restrict__ cache_feature,
    const float* __restrict__ cache_lambda,
    const math::Vec3* __restrict__ cache_normal,
    const uint64_t* __restrict__ cache_material,
    const uint32_t* __restrict__ cache_age,
    uint64_t* __restrict__ snapshot_pair,
    uint64_t* __restrict__ snapshot_feature,
    float* __restrict__ snapshot_lambda,
    math::Vec3* __restrict__ snapshot_normal,
    uint64_t* __restrict__ snapshot_material,
    uint32_t* __restrict__ snapshot_age) {
    const uint32_t env = blockIdx.x / env_blocks;
    const uint32_t part = blockIdx.x - env * env_blocks;
    const uint32_t bound = contact_cache::PreviousBound(extent, env, points_per_env);
    for (uint32_t local = part * blockDim.x + threadIdx.x; local < bound;
         local += env_blocks * blockDim.x) {
        const uint32_t point_index = env * points_per_env + local;
        if (old_keep[point_index] == 0u) continue;
        snapshot_pair[point_index] = cache_pair[point_index];
        snapshot_feature[point_index] = cache_feature[point_index];
        for (uint32_t k = 0u; k < 3u; ++k)
            snapshot_lambda[point_index * 3u + k] = cache_lambda[point_index * 3u + k];
        snapshot_normal[point_index] = cache_normal[point_index];
        snapshot_material[point_index] = cache_material[point_index];
        snapshot_age[point_index] = cache_age[point_index];
    }
}

// Points at or past the env's rebuild bound are already clear, which is all a rebuild would leave there.
__global__ void RebuildContactCacheKernel(
    const uint64_t* __restrict__ current_pair,
    const uint64_t* __restrict__ current_feature,
    const math::Vec3* __restrict__ current_normal,
    const uint64_t* __restrict__ current_material,
    const float* __restrict__ lambda,
    const uint64_t* __restrict__ snapshot_pair,
    const uint64_t* __restrict__ snapshot_feature,
    const float* __restrict__ snapshot_lambda,
    const math::Vec3* __restrict__ snapshot_normal,
    const uint64_t* __restrict__ snapshot_material,
    const uint32_t* __restrict__ snapshot_age,
    const uint32_t* __restrict__ current_owner,
    const uint32_t* __restrict__ old_keep,
    const contact_cache::Ranks* __restrict__ ranks,
    const uint32_t* __restrict__ retained_sources,
    const uint32_t* __restrict__ extent, uint32_t env_blocks,
    uint32_t slot_count, const uint32_t* row_ranges,
    uint64_t* __restrict__ cache_pair,
    uint64_t* __restrict__ cache_feature,
    float* __restrict__ cache_lambda,
    math::Vec3* __restrict__ cache_normal,
    uint64_t* __restrict__ cache_material,
    uint32_t* __restrict__ cache_age) {
    const uint32_t points_per_env = slot_count * 4u;
    const uint32_t env = blockIdx.x / env_blocks;
    const uint32_t part = blockIdx.x - env * env_blocks;
    const uint32_t begin = env * points_per_env;
    const uint32_t end = begin + points_per_env;
    const uint32_t bound = extent != nullptr ? min(extent[2u * env + 1u], points_per_env) : points_per_env;
    for (uint32_t local_point = part * blockDim.x + threadIdx.x; local_point < bound;
         local_point += env_blocks * blockDim.x) {
        const uint32_t point_index = begin + local_point;
        const uint32_t slot = local_point / 4u;
        const uint32_t point = local_point & 3u;
        if (current_owner[point_index] != 0u) {
            uint32_t normal_row = 0u, tangent1_row = 0u, tangent2_row = 0u;
            if (ContactRowOffsets(env, slot, point, slot_count, row_ranges, &normal_row, &tangent1_row,
                                  &tangent2_row)) {
                cache_pair[point_index] = current_pair[point_index];
                cache_feature[point_index] = current_feature[point_index];
                cache_lambda[point_index * 3u] = fmaxf(lambda[normal_row], 0.0f);
                cache_lambda[point_index * 3u + 1u] = lambda[tangent1_row];
                cache_lambda[point_index * 3u + 2u] = lambda[tangent2_row];
                cache_normal[point_index] = current_normal[point_index];
                cache_material[point_index] = current_material[point_index / 4u];
                cache_age[point_index] = 0u;
                continue;
            }
        }

        const uint32_t free_rank = ranks[point_index].free - ranks[begin].free;
        const uint32_t old_count = ranks[end - 1u].old + old_keep[end - 1u] - ranks[begin].old;
        const uint32_t source = free_rank < old_count ? retained_sources[begin + free_rank] : ~0u;
        if (source != ~0u) {
            cache_pair[point_index] = snapshot_pair[source];
            cache_feature[point_index] = snapshot_feature[source];
            for (uint32_t k = 0u; k < 3u; ++k)
                cache_lambda[point_index * 3u + k] = snapshot_lambda[source * 3u + k];
            cache_normal[point_index] = snapshot_normal[source];
            cache_material[point_index] = snapshot_material[source];
            cache_age[point_index] = snapshot_age[source] + 1u;
            continue;
        }
        ClearContactCachePoint(point_index, cache_pair, cache_feature, cache_lambda,
                               cache_normal, cache_material, cache_age);
    }
}

Status OpContactWarmStart(const ModelView& /*model*/, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ContactWarmStartParams*>(params);
    if (p == nullptr || p->phase > 1u) return Status::InvalidArgument;
    if (p->slot_count == 0u || p->env_count == 0u) return Status::Ok;
    const uint64_t records_per_env = uint64_t{p->slot_count} * nk::kPairDrivenPtsPerSlot * 2u;
    if (p->env_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / records_per_env)
        return Status::InvalidArgument;
    const uint64_t count = uint64_t{p->env_count} * p->slot_count * nk::kPairDrivenPtsPerSlot;
    const uint32_t point_count = static_cast<uint32_t>(count);
    if (!data.contact_cache_scratch || !data.contact_warm_start_counts ||
        p->workspace_bytes <= contact_cache::Layout(point_count, p->env_count).temp_offset)
        return Status::InvalidArgument;
    contact_cache::Workspace workspace(data.contact_cache_scratch, p->workspace_bytes, point_count, p->env_count);
    constexpr uint32_t kBlock = 128u;
    if (p->phase == 0u) {
        const auto status = contact_cache::BuildIndex(data, point_count, p->slot_count * nk::kPairDrivenPtsPerSlot,
                                                     p->decay_steps, workspace, stream);
        if (status != cudaSuccess) return Status::Failed;
        const uint32_t env_blocks = contact_cache::EnvBlocks(
            p->slot_count * nk::kPairDrivenPtsPerSlot * 2u, p->env_count);
        LaunchCuda<true>(PrepareContactWarmStartKernel, dim3(p->env_count * env_blocks), dim3(kBlock), 0u,
                   stream, data.ucontact_normal, data.contact_cache_lambda,
                   data.contact_cache_normal, workspace.matches, workspace.order,
                   workspace.begins, workspace.ends, p->env_count, p->slot_count,
                   data.ucontact_row_range, env_blocks, data.lambda);
    } else {
        const uint32_t points_per_env = p->slot_count * nk::kPairDrivenPtsPerSlot;
        const uint32_t point_blocks = contact_cache::EnvBlocks(points_per_env, p->env_count);
        LaunchCuda(SnapshotContactCacheKernel, dim3(p->env_count * point_blocks), dim3(kBlock), 0u,
                   stream, points_per_env, point_blocks, data.contact_cache_extent,
                   data.contact_cache_old_keep, data.contact_cache_pair,
                   data.contact_cache_feature, data.contact_cache_lambda,
                   data.contact_cache_normal, data.contact_cache_material,
                   data.contact_cache_age, data.contact_cache_snapshot_pair,
                   data.contact_cache_snapshot_feature,
                   data.contact_cache_snapshot_lambda,
                   data.contact_cache_snapshot_normal,
                   data.contact_cache_snapshot_material,
                   data.contact_cache_snapshot_age);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        const auto status = contact_cache::BuildRanks(data, point_count, points_per_env, workspace, stream);
        if (status != cudaSuccess) return Status::Failed;
        LaunchCuda(RebuildContactCacheKernel, dim3(p->env_count * point_blocks), dim3(kBlock), 0u,
                   stream, data.ucontact_id_pair, data.ucontact_id_feature,
                   data.ucontact_normal, data.contact_material, data.lambda,
                   data.contact_cache_snapshot_pair,
                   data.contact_cache_snapshot_feature,
                   data.contact_cache_snapshot_lambda,
                   data.contact_cache_snapshot_normal,
                   data.contact_cache_snapshot_material,
                   data.contact_cache_snapshot_age,
                   data.contact_cache_current_owner,
                   data.contact_cache_old_keep, workspace.ranks, workspace.matches,
                   data.contact_cache_extent, point_blocks, p->slot_count,
                   data.ucontact_row_range,
                   data.contact_cache_pair, data.contact_cache_feature,
                   data.contact_cache_lambda, data.contact_cache_normal,
                   data.contact_cache_material, data.contact_cache_age);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpSnapshotStepVelocity(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SnapshotStepVelocityParams*>(params);
    if (p == nullptr || p->env_count == 0u) return Status::Failed;
    if (p->articulation_count > 0u && p->max_dof > 0u) {
        constexpr uint32_t block_size = 128u;
        const uint32_t total = p->articulation_count * p->max_dof;
        const auto state = MakeArticulationDeviceState(
            model, data, p->base_link_count * p->env_count, p->articulation_count);
        LaunchCuda(PackQdotFlatMultiKernel, dim3((total + block_size - 1u) / block_size),
                   dim3(block_size), 0u, stream, state, p->max_dof, data.step_qdot_flat);
    }
    if (p->total_body_count > 0u) {
        const size_t bytes = static_cast<size_t>(p->total_body_count) * sizeof(math::Vec3);
        if (cudaMemcpyAsync(data.step_body_linear_velocity, data.body_linear_velocity,
                            bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(data.step_body_angular_velocity, data.body_angular_velocity,
                            bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return Status::Failed;
        }
    }
    if (p->total_particle_count > 0u &&
        cudaMemcpyAsync(data.step_particle_velocity, data.particle_vel,
                        static_cast<size_t>(p->total_particle_count) * sizeof(math::Vec3),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return Status::Failed;
    }
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpAssembleRowsPairDriven(const ModelView& model, const DataView& data,
                                const AssembleRowsParams* p, cudaStream_t stream) {
    if (p->env_count == 0u || p->rows_per_env == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlockSize = kContactRowsBlock;
    const uint32_t total_rows = p->env_count * p->rows_per_env;
    const uint32_t full_slots = std::min(p->full_row_slot_count, p->union_slot_count);
    const uint32_t rigid_rows = p->contact_rows_per_env -
        (p->union_slot_count - full_slots) * nk::kPairDrivenParticleRowsPerSlot;
    LaunchCuda(InitializePairDrivenRangesKernel, dim3(p->env_count), dim3(kBlockSize), 0u, stream,
               data.ucontact_count, p->union_slot_count, full_slots,
               p->rows_per_env, p->contact_rows_per_env, data.ucontact_row_range,
               data.row_count, data.contact_row_extent, data.contact_material);
    LaunchCuda<true>(ClearPairDrivenRowsKernel,
               dim3((total_rows + kBlockSize - 1u) / kBlockSize), dim3(kBlockSize), 0u, stream,
               total_rows, p->rows_per_env, p->contact_rows_per_env,
               p->union_slot_count, full_slots, rigid_rows, data.contact_row_extent,
               data.ucontact_row_range, reinterpret_cast<NkRow*>(data.urows),
               data.lambda, data.row_cj_link, data.row_cj_link_b, data.row_penetration,
               data.row_damping, data.contact_material);
    const bool has_artic = p->max_dof > 0u && p->articulation_count > 0u;
    const uint32_t artics_per_env =
        (p->articulation_count > 0u && p->env_count > 0u)
            ? (p->articulation_count / p->env_count) : 0u;

    // Pack each articulation's actual DOFs into its padded flat velocity tile.
    if (has_artic) {
        const uint32_t total = p->articulation_count * p->max_dof;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        const auto state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        LaunchCuda<true>(PackQdotFlatMultiKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   state, p->max_dof, data.qdot_flat);
    }

    // Assemble contact rows from the unified manifold.
    if (p->union_slot_count > 0u) {
        const uint32_t total = p->env_count * p->union_slot_count;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda<true>(EmitPairDrivenRowsKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   static_cast<const uint32_t*>(data.ucontact_count), data.ucontact_row_range,
                   static_cast<const math::Vec3*>(data.ucontact_point),
                   static_cast<const math::Vec3*>(data.ucontact_witness_a),
                   static_cast<const math::Vec3*>(data.ucontact_witness_b),
                   static_cast<const math::Vec3*>(data.ucontact_normal),
                   static_cast<const float*>(data.ucontact_depth),
                   static_cast<const uint32_t*>(data.ucontact_a),
                   static_cast<const uint32_t*>(data.ucontact_b),
                   static_cast<const uint32_t*>(data.ucontact_a_kind),
                   static_cast<const uint32_t*>(data.ucontact_b_kind),
                   data.ucontact_law, data.ucontact_friction,
                   static_cast<const math::Transform*>(data.body_pose),
                   static_cast<const math::Transform*>(data.body_inertial_frame),
                   static_cast<const float*>(model.shape_table),
                   static_cast<const uint32_t*>(model.body_to_link),
                   static_cast<const uint32_t*>(model.body_to_articulation),
                   static_cast<const uint32_t*>(model.body_collidable_body),
                   static_cast<const float*>(data.mat_buckets),
                   p->num_material_buckets, data.contact_material,
                   p->particles_per_env, p->grid_nodes_per_env,
                   p->point_endpoints_per_env, p->point_endpoint_terms_per_env,
                   data.point_endpoint_ranges, data.point_endpoint_terms,
                   static_cast<const float*>(model.particle_friction),
                   p->solref[0], p->solref[1],
                   p->solimp[0], p->solimp[1], p->solimp[2], p->solimp[3], p->solimp[4],
                   p->dt, p->baumgarte_max_velocity, p->contact_margin, p->overlap_recovery_velocity,
                   p->env_count, p->union_slot_count, p->rows_per_env,
                   p->full_row_slot_count, p->ogc_slot_count,
                   p->bodies_per_env, p->base_link_count, artics_per_env,
                   reinterpret_cast<NkRow*>(data.urows), data.lambda,
                   data.row_cj_link, data.row_cj_point,
                   data.row_cj_link_b, data.row_cj_point_b,
                   data.row_count, data.row_penetration, data.row_damping, data.env_status,
                   data.contact_row_extent);
    }

    if (has_artic && (p->joint_limit_rows_per_env >= p->base_link_count * 2u ||
        (p->base_link_count > 0u && (p->joint_friction_rows_per_env >= p->base_link_count ||
                                    p->joint_drive_rows_per_env >= p->base_link_count)))) {
        const auto state = MakeArticulationDeviceState(model, data, p->total_link_count, p->articulation_count);
        const uint32_t blocks = (p->total_link_count + kBlockSize - 1u) / kBlockSize;
        LaunchCuda<true>(EmitJointRowsKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   state, *p, model.joint_limit_lower, model.joint_limit_upper,
                   model.joint_limit_flags, data);
    }

    if (has_artic) {
        // K3: chain Jacobians for side A AND side B (the SAME multi-artic kernel,
        // fed each side's per-row link/point gather). Each live articulation
        // row clears its own slice before the fixed-order += walk.
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        // Each side's direction is read in place from the rows, normal_stride floats apart.
        const NkRow* const rows = reinterpret_cast<const NkRow*>(data.urows);
        constexpr uint32_t kRowFloats = sizeof(NkRow) / sizeof(float);
        const uint32_t blocks = (total_rows + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ComputeContactChainJacobianKernel, dim3(blocks), dim3(kBlockSize),
                   0u, stream, state,
                   static_cast<const uint32_t*>(data.row_cj_link),
                   static_cast<const math::Vec3*>(data.row_cj_point),
                   &rows->a.jlin.x, kRowFloats,
                   total_rows, p->max_dof, data.chain_jacobian);
        LaunchCuda(ComputeContactChainJacobianKernel, dim3(blocks), dim3(kBlockSize),
                   0u, stream, state,
                   static_cast<const uint32_t*>(data.row_cj_link_b),
                   static_cast<const math::Vec3*>(data.row_cj_point_b),
                   &rows->b.jlin.x, kRowFloats,
                   total_rows, p->max_dof, data.chain_jacobian_b);

        LaunchCuda(MergeSharedArticulationJacobianKernel, dim3(blocks), dim3(kBlockSize),
                   0u, stream, reinterpret_cast<const NkRow*>(data.urows),
                   total_rows, p->max_dof, data.chain_jacobian, data.chain_jacobian_b);

        // With mimic couplings the rows act on the reduced coordinates MimicReduce prepared.
        const bool mimic = p->mimic_couplings_per_env > 0u;
        if (mimic) {
            LaunchCuda(FoldMimicJacobianKernel<false>, dim3(blocks), dim3(kBlockSize), 0u, stream,
                       reinterpret_cast<const NkRow*>(data.urows),
                       static_cast<const uint32_t*>(data.mimic_root_dof),
                       static_cast<const float*>(data.mimic_root_scale),
                       total_rows, p->max_dof, data.chain_jacobian);
            LaunchCuda(FoldMimicJacobianKernel<true>, dim3(blocks), dim3(kBlockSize), 0u, stream,
                       reinterpret_cast<const NkRow*>(data.urows),
                       static_cast<const uint32_t*>(data.mimic_root_dof),
                       static_cast<const float*>(data.mimic_root_scale),
                       total_rows, p->max_dof, data.chain_jacobian_b);
        }
    }

    // Both endpoints contribute to contact preparation and diagnostic response.
    {
        const uint32_t blocks = (total_rows + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(PrepareContactRowsKernel, dim3(blocks), dim3(kBlockSize), 0u,
                   stream, reinterpret_cast<NkRow*>(data.urows),
                   data.row_contact_response,
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.m_factor),
                   static_cast<const float*>(data.chain_jacobian_b),
                   static_cast<const float*>(data.body_inv_mass),
                   static_cast<const math::SymmetricMat3*>(data.body_world_inv_inertia),
                   PointMasses(data),
                   data.row_damping, data.step_qdot_flat, data.step_body_linear_velocity,
                   data.step_body_angular_velocity, data.step_particle_velocity,
                   total_rows, p->max_dof, p->dt, p->vertex_blocks, p->rows_per_env,
                   p->contact_rows_per_env, p->measure_contact_response, data.contact_row_extent);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

// --- op entry point ---------------------------------------------------------

Status OpAssembleRows(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const AssembleRowsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    return p->family == kContactFamilyPairDriven
        ? OpAssembleRowsPairDriven(model, data, p, stream) : Status::Failed;
}

} // namespace

uint64_t ContactCacheScratchBytes(uint32_t point_count, uint32_t env_count) {
    return contact_cache::ScratchBytes(point_count, env_count);
}

uint64_t ArticulationThreadStackBytes(uint32_t max_dof) {
    // PrepareContactRowsKernel allocates six float rows, MassCoupling two double rows; static frames stay small.
    constexpr uint64_t kStaticFrameBytes = 512u;
    return kStaticFrameBytes + std::max(6u * sizeof(float), 2u * sizeof(double)) * uint64_t{max_dof};
}

void RegisterNkAssembleRowsOps() {
    SetCudaOp(NkOp::SnapshotStepVelocity, &OpSnapshotStepVelocity);
    SetCudaOp(NkOp::AssembleRows, &OpAssembleRows);
    SetCudaOp(NkOp::ContactWarmStart, &OpContactWarmStart);
}

} // namespace nuka::phi
