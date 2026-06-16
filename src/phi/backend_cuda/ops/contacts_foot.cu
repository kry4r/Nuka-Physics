// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M3b first narrowphase batch:
//   NarrowphasePrimitives (foot sphere x ground plane) / ContactTangentBasis.
//
// KERNEL BODIES ARE LINE-BY-LINE PORTS (D1 byte-exact contract) of
// src/runtime/articulation/articulation_contacts.cu
// (DetectFootGroundContactsKernel / ComputeContactTangentBasisKernel +
// TangentBasis). Input wiring only: world poses come from the link_pose field
// (refreshed by FkWorldPoses this step — the same bytes the legacy world_pose_
// scratch held), the foot table from the foot_shape Model field, the contact
// stream into the contact_* Data fields. foot_count == 0 still launches the
// detection so every slot is deterministically zero-filled (count = 0) — the
// contacts-off configuration (the single-env oracle path).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"   // M5: pair-driven dispatch
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"  // M4: union-family dispatch
#include "phi/op_schema.hpp"                      // NarrowphasePrimitivesParams
#include "sensor/terrain/terrain_field.hpp"       // Go2-on-stairs: terrain h(x,y)

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kInvalidLink = ~0u;

namespace mg = ::nuka::math::gpu;

__device__ math::Vec3 AddVec(math::Vec3 a, math::Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ math::Vec3 ScaleVec(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

__forceinline__ __device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return mg::Cross(a, b);
}

__forceinline__ __device__ math::Vec3 NormalizeOrUpLocal(math::Vec3 normal) {
    return mg::NormalizeOrUp(normal);
}

__device__ math::Vec3 RotateByQuat(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv_norm = rsqrtf(norm_sq);
        q.w *= inv_norm;
        q.x *= inv_norm;
        q.y *= inv_norm;
        q.z *= inv_norm;
    }
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = ScaleVec(Cross3(qv, v), 2.0f);
    const math::Vec3 wt = ScaleVec(t, q.w);
    const math::Vec3 qvt = Cross3(qv, t);
    return AddVec(AddVec(v, wt), qvt);
}

// (B) One thread per environment. Walks the env's feet in fixed order, compacts
// penetrating feet into the env's own [env*stride, env*stride+count) slots, and
// writes the env's contact count. No atomics; each env owns its slot block.
__global__ void DetectFootGroundContactsKernel(const math::Transform* world_pose,
                                               const FootShape* feet,
                                               uint32_t foot_count,
                                               uint32_t env_count,
                                               uint32_t base_link_count,
                                               float ground_height,
                                               const uint32_t* env_terrain_type,
                                               const float* env_terrain_difficulty,
                                               ::nuka::terrain::TerrainParams terrain,
                                               uint32_t articulations_per_env,
                                               uint32_t max_foot_contacts,
                                               const uint32_t* link_to_articulation,
                                               uint32_t* out_contact_link,
                                               math::Vec3* out_contact_point,
                                               math::Vec3* out_contact_normal,
                                               float* out_contact_depth,
                                               uint32_t* out_contact_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) {
        return;
    }

    // Go2-on-stairs Phase 1: per-env procedural terrain TYPE. env_terrain_type
    // is seeded 0 (Flat) for every env at construction, so the default contact
    // path is byte-identical to the legacy scalar plane (SampleTerrainHeight
    // returns terrain.ground_height == ground_height when type == 0). A null
    // pointer (defensive) also falls back to the flat plane.
    const uint32_t terrain_type =
        (env_terrain_type != nullptr) ? env_terrain_type[env]
                                      : ::nuka::terrain::kTerrainFlat;

    // Go2-on-stairs Phase 2a: per-env terrain DIFFICULTY scale (curriculum). It
    // scales the terrain's VERTICAL feature magnitude (step_height +
    // grid_height_max) via the SHARED ScaleTerrainDifficulty helper (the same one
    // the render tessellator uses, so the two never drift). DEFAULT 1.0 (seeded by
    // World::SeedInitialState) leaves the terrain unscaled; a null pointer
    // (defensive) also falls back to 1.0. For the type-0 (Flat) default the scale
    // is computed but UNUSED -- SampleTerrainHeight returns ground_height
    // regardless of step/grid -- so the flat contact path stays byte-identical (D1).
    const float terrain_difficulty =
        (env_terrain_difficulty != nullptr) ? env_terrain_difficulty[env] : 1.0f;
    const ::nuka::terrain::TerrainParams env_terrain =
        ::nuka::terrain::ScaleTerrainDifficulty(terrain, terrain_difficulty);

    // Task 1: K co-resident dogs per env. The FUSED per-articulation solver reads
    // each articulation's contacts from the slot block [artic*kMaxFootContacts,
    // +kMaxFootContacts) (slot_base = articulation*stride). At K<=1 articulation ==
    // env, so the LEGACY env-keyed layout (base_slot = env*kMaxFootContactsPerEnv,
    // one per-env count) IS the per-articulation layout -> the K==1 path below is
    // byte-identical to the prior kernel (D1). At K>1 each foot is routed into its
    // OWNING articulation's block via link_to_articulation[global_link], with a
    // PER-ARTICULATION fill counter, so K dogs' 4K feet never overflow one env's 4
    // slots and the dog-dog op can APPEND into the same blocks afterwards.
    const bool multi_artic = articulations_per_env > 1u &&
                             link_to_articulation != nullptr;

    if (!multi_artic) {
        // -- LEGACY env-keyed path (K<=1): byte-identical to the prior kernel. ----
        const uint32_t base_slot = env * kMaxFootContactsPerEnv;
        uint32_t count = 0u;
        for (uint32_t foot = 0u; foot < foot_count; ++foot) {
            const FootShape shape = feet[foot];
            const uint32_t link = env * base_link_count + shape.calf_local_link;
            const math::Transform calf = world_pose[link];
            // Foot sphere center in world = calf_world o local_offset.
            const math::Vec3 center =
                AddVec(calf.position, RotateByQuat(calf.rotation, shape.local_offset));
            // Procedural-terrain support height at the foot's (x,y) column. For the
            // default Flat type this returns EXACTLY terrain.ground_height (==
            // ground_height), so depth == (ground_height + radius) - center.z — the
            // byte-identical legacy expression. Normal stays {0,0,1} (heightfield
            // vertical-support model; gradient normals are a v1 non-goal).
            const float surface = ::nuka::terrain::SampleTerrainHeight(
                terrain_type, center.x, center.y, env_terrain);
            const float depth = (surface + shape.radius) - center.z;
            if (depth > 0.0f) {
                const uint32_t slot = base_slot + count;
                out_contact_link[slot] = link;
                // Contact point: foot center projected onto the local terrain surface.
                out_contact_point[slot] = {center.x, center.y, surface};
                out_contact_normal[slot] = {0.0f, 0.0f, 1.0f};
                out_contact_depth[slot] = depth;
                ++count;
            }
        }
        // Zero-fill the env's unused trailing slots so stale data never leaks.
        for (uint32_t slot = count; slot < kMaxFootContactsPerEnv; ++slot) {
            const uint32_t index = base_slot + slot;
            out_contact_link[index] = kInvalidLink;
            out_contact_point[index] = {0.0f, 0.0f, 0.0f};
            out_contact_normal[index] = {0.0f, 0.0f, 0.0f};
            out_contact_depth[index] = 0.0f;
        }
        out_contact_count[env] = count;
        return;
    }

    // -- MULTI-ARTICULATION path (K>1): per-articulation foot slot re-key. --------
    // The env owns K articulation slot blocks [art0*stride .. (artK-1)*stride+stride).
    // articulation_global = env*K + local; the legacy stride is kMaxFootContacts.
    const uint32_t K = articulations_per_env;
    const uint32_t stride = max_foot_contacts;  // == kMaxFootContactsPerEnv
    constexpr uint32_t kMaxArtPerEnv = 64u;
    if (K > kMaxArtPerEnv) {
        // Honest guard (matches the dog-dog op's G0 discipline): never silently
        // misroute past the on-thread per-artic counter capacity.
        out_contact_count[env] = 0u;
        return;
    }
    // Zero-fill every articulation block of this env first (one thread owns the
    // whole env span, so no atomics). The dog-dog op later APPENDS into the slots
    // these feet leave free -- it counts the active foot slots, not re-zeroes them.
    for (uint32_t a = 0u; a < K; ++a) {
        const uint32_t art_global = env * K + a;
        const uint32_t base = art_global * stride;
        for (uint32_t s = 0u; s < stride; ++s) {
            out_contact_link[base + s] = kInvalidLink;
            out_contact_point[base + s] = {0.0f, 0.0f, 0.0f};
            out_contact_normal[base + s] = {0.0f, 0.0f, 0.0f};
            out_contact_depth[base + s] = 0.0f;
        }
    }
    uint32_t fill[kMaxArtPerEnv];
    for (uint32_t a = 0u; a < K; ++a) fill[a] = 0u;

    uint32_t total = 0u;
    for (uint32_t foot = 0u; foot < foot_count; ++foot) {
        const FootShape shape = feet[foot];
        const uint32_t link = env * base_link_count + shape.calf_local_link;
        const math::Transform calf = world_pose[link];
        const math::Vec3 center =
            AddVec(calf.position, RotateByQuat(calf.rotation, shape.local_offset));
        const float surface = ::nuka::terrain::SampleTerrainHeight(
            terrain_type, center.x, center.y, env_terrain);
        const float depth = (surface + shape.radius) - center.z;
        if (depth <= 0.0f) continue;
        // Route into the foot's OWNING articulation's block. link_to_articulation
        // is staged GLOBAL (env*K + local), so this is the global articulation idx.
        const uint32_t art = link_to_articulation[link];
        const uint32_t local = (art >= env * K && art < env * K + K)
                                   ? (art - env * K) : 0u;
        if (fill[local] >= stride) continue;  // a dog can't exceed its own block.
        const uint32_t slot = art * stride + fill[local];
        out_contact_link[slot] = link;
        out_contact_point[slot] = {center.x, center.y, surface};
        out_contact_normal[slot] = {0.0f, 0.0f, 1.0f};
        out_contact_depth[slot] = depth;
        ++fill[local];
        ++total;
    }
    out_contact_count[env] = total;
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

// One thread per contact slot (env-major, fixed stride). Builds (t1,t2) for each
// active contact's normal; inactive slots get zero vectors.
__global__ void ComputeContactTangentBasisKernel(const uint32_t* contact_link,
                                                 const math::Vec3* contact_normal,
                                                 uint32_t slot_count,
                                                 math::Vec3* out_tangent1,
                                                 math::Vec3* out_tangent2) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }
    if (contact_link[slot] == kInvalidLink) {
        out_tangent1[slot] = {0.0f, 0.0f, 0.0f};
        out_tangent2[slot] = {0.0f, 0.0f, 0.0f};
        return;
    }
    const math::Vec3 normal = NormalizeOrUpLocal(contact_normal[slot]);
    math::Vec3 t1;
    math::Vec3 t2;
    TangentBasis(normal, &t1, &t2);
    out_tangent1[slot] = t1;
    out_tangent2[slot] = t2;
}

// --- op entry points ---------------------------------------------------------

Status OpNarrowphasePrimitives(const ModelView& model, const DataView& data,
                               const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphasePrimitivesParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->family == kContactFamilyUnionCsr) {
        // M4 union family: per-(env x union-slot) analytic detection
        // (contacts_union.cu).
        return LaunchUnionNarrowphase(model, data, *p, stream);
    }
    if (p->family == kContactFamilyPairDriven) {
        // M5 generalized family: candidate_pairs -> analytic prim dispatch
        // (narrowphase_prims.cu). ADDITIVE; never touches the union path.
        return LaunchPairDrivenNarrowphase(model, data, *p, stream);
    }
    if (p->env_count == 0u) {
        return Status::Ok;
    }
    // Foot-budget guard. At K<=1 the legacy per-ENV cap holds (one dog, <=4 feet).
    // At K>1 the cap is PER-ARTICULATION (the kernel routes each foot into its
    // owning dog's 4-slot block), so the env's total budget is K*kMaxFootContacts
    // -- a K-dog scene legitimately carries 4K feet. The kernel itself caps each
    // dog's block at the stride (a >4-foot dog drops the overflow), so this is the
    // honest env-wide ceiling, not a per-dog one.
    const uint32_t K = (p->articulations_per_env == 0u) ? 1u
                                                        : p->articulations_per_env;
    const uint32_t foot_budget =
        (K > 1u) ? (K * kMaxFootContactsPerEnv) : kMaxFootContactsPerEnv;
    if (p->foot_count > foot_budget) {
        return Status::Failed;  // loud-throw guard (per-artic budget for K>1).
    }
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (p->env_count + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(DetectFootGroundContactsKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream,
               static_cast<const math::Transform*>(data.link_pose),
               reinterpret_cast<const FootShape*>(model.foot_shape),
               p->foot_count, p->env_count, p->base_link_count, p->ground_height,
               static_cast<const uint32_t*>(data.env_terrain_type),
               static_cast<const float*>(data.env_terrain_difficulty), p->terrain,
               p->articulations_per_env, p->max_foot_contacts,
               static_cast<const uint32_t*>(model.link_to_articulation),
               data.contact_link, data.contact_point, data.contact_normal,
               data.contact_depth, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpContactTangentBasis(const ModelView& /*model*/, const DataView& data,
                             const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ContactTangentBasisParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->slot_count == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (p->slot_count + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(ComputeContactTangentBasisKernel, dim3(block_count), dim3(kBlockSize),
               0u, stream,
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const math::Vec3*>(data.contact_normal),
               p->slot_count, data.contact_tangent1, data.contact_tangent2);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkContactsFootOps() {
    SetCudaOp(NkOp::NarrowphasePrimitives, &OpNarrowphasePrimitives);
    SetCudaOp(NkOp::ContactTangentBasis, &OpContactTangentBasis);
}

} // namespace nuka::phi
