// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M5 broadphase ops (plan §3.5):
//   BuildAabbs        : per-body world-space AABB from shape_table + body_pose.
//   LbvhBuild         : per-env Karras 2012 LBVH over the env's body AABB set.
//   LbvhQueryPairs    : LBVH overlap query -> candidate_pairs (+ pair_count
//                       watermark), with TAGGED FILTERING folded in (contype/
//                       conaffinity bitmask + sorted excluded-pair binary search
//                       + same-body drop + cross-env gating).
//   ParticleGridBuild : uniform spatial hash grid CSR neighbors (the M6 PBF/
//                       XPBD co-step input).
//
// KERNEL BODIES ARE LINE-BY-LINE PORTS of the standalone collision sources
// (broadphase_lbvh.cu / lbvh_traversal.cuh / particle_uniform_grid.cu) — the
// algorithm + the particle-grid stable radix sort + exclusive scan D1 anchors are
// preserved (cub::DeviceRadixSort::SortPairs + DeviceScan::ExclusiveScan, the very
// primitives thrust dispatched to, drawing from a pre-allocated scratch so the op
// captures into a CUDA graph). Only the I/O wiring changes (inputs are the arena/
// Model device pointers; node/morton/visit + sort scratch are arena-resident; the
// output is the candidate_pairs / grid_* Data fields).
//
// FAMILY GATING (plan §3.5 / op_schema kContactFamily*): the broadphase ops do
// real work ONLY for kContactFamilyPairDriven. The union slot-template family
// (the gate-pinned grasp / NkUnionN1 production path) and the fused-foot family
// run their OWN detection and never read candidate_pairs, so these ops EARLY-
// EXIT for them — the union StepPlanned graph stays bit-identical to M4 (these
// ops enqueue nothing in the captured stream). BuildAabbs writes only the
// body_aabb_* scratch the union path never reads, so it is harmless either way,
// but it too early-exits unless pair-driven to keep the captured graph minimal.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cfloat>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

// cub::Sum was removed in CUDA 13's CCCL; cuda::std::plus replaces it there.
#if __CUDACC_VER_MAJOR__ >= 13
#include <cuda/std/functional>
namespace { using NkScanSumOp = ::cuda::std::plus<>; }
#else
namespace { using NkScanSumOp = ::cub::Sum; }
#endif

#include "collision/lbvh_batched.cuh"   // BuildLbvhBatchedNodes (shared env-build)
#include "collision/lbvh_node.cuh"      // LbvhNode (query traversal)
#include "collision/shape_kind.hpp"     // nuka::collision::ShapeKind (R2: one enum)
#include "collision/particle_grid_traversal.cuh"  // ParticleGridConfigDevice / QueryParticleNeighbors
#include "collision/particle_uniform_grid.hpp"     // kParticleGridMaxNeighbors (canonical)
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // kShapeTableRowStride (shared)
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace {

namespace cg = ::nuka::collision::gpu;
constexpr uint32_t kBlockSize = 128u;

// Round up to the 256B section alignment the Arena lays the scratch out at (so
// every sub-region of grid_sort_scratch is 256B-aligned device memory).
constexpr uint64_t kScratchAlign = 256u;
inline __host__ uint64_t AlignScratch(uint64_t v) {
    return (v + (kScratchAlign - 1u)) & ~(kScratchAlign - 1u);
}

// 256B-aligned partition of grid_sort_scratch [cub temp | keys-out | idx-out].
// Sizes the segment (World construct) and partitions it (the op) identically.
struct GridSortScratchLayout {
    uint64_t temp_bytes = 0u;     // cub temp-storage region size (max of sort/scan).
    uint64_t keys_off   = 0u;     // byte offset of the sorted-keys out buffer.
    uint64_t idx_off    = 0u;     // byte offset of the sorted-idx out buffer.
    uint64_t total      = 0u;     // full segment byte size.
    explicit GridSortScratchLayout(uint32_t particle_count) {
        const int n = static_cast<int>(particle_count);
        size_t sort_bytes = 0u;
        (void)cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
            nullptr, sort_bytes, static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), n);
        size_t scan_bytes = 0u;
        (void)cub::DeviceScan::ExclusiveScan(
            nullptr, scan_bytes, static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), NkScanSumOp{}, 0u, n);
        temp_bytes = sort_bytes > scan_bytes ? sort_bytes : scan_bytes;
        const uint64_t nbytes = static_cast<uint64_t>(particle_count) * sizeof(uint32_t);
        keys_off = AlignScratch(temp_bytes);
        idx_off  = AlignScratch(keys_off + nbytes);
        total    = AlignScratch(idx_off + nbytes);
    }
};

// 256B-aligned partition of pair_sort_scratch [cub temp | keys-in | keys-out |
// slot perms | pair snapshot]. Sizes the segment (World construct) and
// partitions it (the op) identically. The canonicalizing sort turns the
// atomicAdd race order of EnvQueryPairsKernel into a pure function of the pair
// ids, so contact slots, row slots, GS sweep order, and the warm-start cache
// are all deterministic run-to-run.
struct PairSortScratchLayout {
    uint64_t temp_bytes = 0u;   // cub temp-storage region size (max of sort/scan).
    uint64_t keys_in_off = 0u;  // byte offset of the unsorted u64 keys.
    uint64_t keys_out_off = 0u; // byte offset of the sorted u64 keys.
    uint64_t perms_in_off = 0u; // byte offset of the unsorted u32 source slots.
    uint64_t perms_out_off = 0u;// byte offset of the sorted u32 source slots.
    uint64_t snap_off = 0u;     // byte offset of the pair snapshot (2 u32/slot).
    uint64_t clamp_off = 0u;    // byte offset of the clamped per-env counts.
    uint64_t prefix_off = 0u;   // byte offset of the per-env live-count prefix.
    uint64_t total = 0u;        // full segment byte size.
    explicit PairSortScratchLayout(uint32_t sort_slots, uint32_t env_count) {
        const int n = static_cast<int>(sort_slots);
        size_t sort_bytes = 0u;
        (void)cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
            nullptr, sort_bytes, static_cast<const uint64_t*>(nullptr),
            static_cast<uint64_t*>(nullptr), static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), n);
        size_t scan_bytes = 0u;
        (void)cub::DeviceScan::ExclusiveScan(
            nullptr, scan_bytes, static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), NkScanSumOp{}, 0u,
            static_cast<int>(env_count));
        temp_bytes = sort_bytes > scan_bytes ? sort_bytes : scan_bytes;
        const uint64_t kbytes = static_cast<uint64_t>(sort_slots) * sizeof(uint64_t);
        const uint64_t pbytes = static_cast<uint64_t>(sort_slots) * sizeof(uint32_t);
        const uint64_t ebytes = static_cast<uint64_t>(env_count) * sizeof(uint32_t);
        keys_in_off = AlignScratch(temp_bytes);
        keys_out_off = AlignScratch(keys_in_off + kbytes);
        perms_in_off = AlignScratch(keys_out_off + kbytes);
        perms_out_off = AlignScratch(perms_in_off + pbytes);
        snap_off = AlignScratch(perms_out_off + pbytes);
        clamp_off = AlignScratch(snap_off + 2u * kbytes);
        prefix_off = AlignScratch(clamp_off + ebytes);
        total = AlignScratch(prefix_off + ebytes);
    }
};
// env_status diagnostic bits (kEnvStatus*) are shared in op_schema.hpp so the
// broadphase, particle-grid, and CRBA ops agree on the readout layout.
// The per-particle neighbor cap is the canonical cg::kParticleGridMaxNeighbors.

// shape_table record: R1 GREW it 8 -> 10 packed f32 / body row
// (Model::PairDrivenShape). Lanes 0..7 unchanged; lanes 8/9 = body_id (int32) +
// group (uint32), the shape->body indirection. L-RECON-D GREW it 10 -> 12,
// appending the per-shape hull slice in lanes 10/11; the AABB broadphase reads
// only lanes 0..9 (the hull bound radius is baked into params[0] at cook), so it
// just needs the WIDENED stride to index the right row.
struct ShapeDev {
    uint32_t   kind;
    float      p[4];        // sphere r / capsule r,hh / box he.xyz
    uint32_t   contype;
    uint32_t   conaffinity;
    uint32_t   sdf_grid;
    int32_t    body_id;     // owning body row, or -1 == static (R1).
    uint32_t   group;       // signed collision-group filter key (R1).
};
__forceinline__ __device__ ShapeDev LoadShape(const float* table, uint32_t row) {
    const float* q = table + static_cast<size_t>(row) * nkops::kShapeTableRowStride;
    ShapeDev s;
    s.kind = __float_as_uint(q[0]);
    s.p[0] = q[1]; s.p[1] = q[2]; s.p[2] = q[3]; s.p[3] = q[4];
    s.contype = __float_as_uint(q[5]);
    s.conaffinity = __float_as_uint(q[6]);
    s.sdf_grid = __float_as_uint(q[7]);
    s.body_id = static_cast<int32_t>(__float_as_uint(q[8]));
    s.group = __float_as_uint(q[9]);
    return s;
}

// Plane broadphase AABB: a plane is analytically infinite, so its LBVH leaf is a
// large-but-finite slab. kPlaneLateralExtent (1 Mm) is the in-plane half-span — a
// scene wider than this misses plane contacts at its rim (raise it / derive from
// the scene bound). kPlaneThickness (1 mm) is the half-thickness along the plane
// normal so the slab stays thin (Morton quantization stays sane).
constexpr float kPlaneLateralExtent = 1.0e6f;
constexpr float kPlaneThickness     = 1.0e-3f;

// Shape kinds — R2: the ONE shared enum (collision/shape_kind.hpp). These local
// aliases keep the kernel switch text identical while removing the divergent
// copy-pasted sentinel block. Only the extents matter for the AABB.
constexpr uint32_t kKindSphere      = ::nuka::collision::kShapeSphere;
constexpr uint32_t kKindCapsule     = ::nuka::collision::kShapeCapsule;
constexpr uint32_t kKindBox         = ::nuka::collision::kShapeBox;
constexpr uint32_t kKindPlane       = ::nuka::collision::kShapePlane;
constexpr uint32_t kKindConvexHull  = ::nuka::collision::kShapeConvexHull;
constexpr uint32_t kKindSdfMesh     = ::nuka::collision::kShapeSdfMesh;
constexpr uint32_t kKindHeightfield = ::nuka::collision::kShapeHeightfield;

__forceinline__ __device__ math::Vec3 RotAbs(math::Quat q, math::Vec3 v) {
    // |R| * v applied component-wise: the world AABB half-extent of a local box
    // is sum_k |col_k| * he_k; we build it from the absolute rotation columns.
    const math::Vec3 qv{q.x, q.y, q.z};
    auto rot = [&](math::Vec3 e) {
        const math::Vec3 t = 2.0f * qv.Cross(e);
        return e + q.w * t + qv.Cross(t);
    };
    const math::Vec3 cx = rot(math::Vec3{1, 0, 0});
    const math::Vec3 cy = rot(math::Vec3{0, 1, 0});
    const math::Vec3 cz = rot(math::Vec3{0, 0, 1});
    return {fabsf(cx.x) * v.x + fabsf(cy.x) * v.y + fabsf(cz.x) * v.z,
            fabsf(cx.y) * v.x + fabsf(cy.y) * v.y + fabsf(cz.y) * v.z,
            fabsf(cx.z) * v.x + fabsf(cy.z) * v.y + fabsf(cz.z) * v.z};
}

// --- BuildAabbs -------------------------------------------------------------
// One thread per (env x body). Derives the world AABB from the shape primitive
// extents + the body world pose, inflated by `margin`.
__global__ void BuildAabbsKernel(const float* __restrict__ shape_table,
                                 const math::Transform* __restrict__ body_pose,
                                 uint32_t total_bodies,
                                 uint32_t bodies_per_env,
                                 float margin,
                                 math::Vec3* __restrict__ out_lo,
                                 math::Vec3* __restrict__ out_hi) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_bodies) return;
    const uint32_t row = gid - (gid / bodies_per_env) * bodies_per_env;
    const ShapeDev s = LoadShape(shape_table, row);
    const math::Transform xf = body_pose[gid];
    math::Vec3 he{0, 0, 0};
    float r = 0.0f;
    switch (s.kind) {
        case kKindSphere:     r = s.p[0]; he = {r, r, r}; break;
        case kKindCapsule:    r = s.p[0]; he = RotAbs(xf.rotation,
                                  math::Vec3{r, r + s.p[1], r}); break;
        case kKindBox:        he = RotAbs(xf.rotation,
                                  math::Vec3{s.p[0], s.p[1], s.p[2]}); break;
        // Hull: conservative BOUND-RADIUS sphere (p[0] = the cooked max |vertex|
        // — rotation-invariant, so no RotAbs needed).
        case kKindConvexHull: r = s.p[0]; he = {r, r, r}; break;
        // SDF mesh: p[1..3] = the cooked per-axis grid bound (a rotated box);
        // a cook without the stamp falls back to the p[0] bound-radius cube.
        case kKindSdfMesh:
            if (s.p[1] > 0.0f && s.p[2] > 0.0f && s.p[3] > 0.0f) {
                he = RotAbs(xf.rotation, math::Vec3{s.p[1], s.p[2], s.p[3]});
            } else {
                r = s.p[0]; he = {r, r, r};
            }
            break;
        // B3 (general contact pipeline Phase 2): the heightfield static collidable.
        // A big FINITE local box spanning the full grid footprint + its z-range so
        // the field is ONE big leaf in the LBVH and any overlapping body yields a
        // (body, heightfield) candidate pair. The cook stamps the grid half-extents
        // in p[0]/p[1] (== 0.5*(ncol-1)*cell / 0.5*(nrow-1)*cell, the local XY span
        // from the grid CENTER, which is where the body_pose sits) and the LOCAL
        // z-range min/max into p[2]/p[3]. The box is centred on the body origin in
        // XY but z-asymmetric (min_z..max_z), so we build the world AABB from the
        // local corners through RotAbs of the half-extent plus the z-centre offset.
        // Mirror the plane clamp posture (rotate the local box by the body pose).
        case kKindHeightfield: {
            const float hx = s.p[0];           // half X span (grid centre -> edge)
            const float hy = s.p[1];           // half Y span
            const float zmin = s.p[2];         // LOCAL z range
            const float zmax = s.p[3];
            const float hz = 0.5f * (zmax - zmin);
            const float zc = 0.5f * (zmax + zmin);  // local z-centre offset.
            he = RotAbs(xf.rotation, math::Vec3{hx, hy, hz});
            // The box is centred at the grid centre + the LOCAL z-centre (rotated
            // into world by the body pose, like every other kind's extent). Inline
            // the quaternion-vector rotation (HD-clean; mirrors the RotAbs lambda).
            const math::Quat q = xf.rotation;
            const math::Vec3 qv{q.x, q.y, q.z};
            const math::Vec3 zloc{0.0f, 0.0f, zc};
            const math::Vec3 tt = 2.0f * qv.Cross(zloc);
            const math::Vec3 zoff_w = zloc + q.w * tt + qv.Cross(tt);
            const math::Vec3 c2 = xf.position + zoff_w;
            const float m = margin;
            out_lo[gid] = {c2.x - he.x - m, c2.y - he.y - m, c2.z - he.z - m};
            out_hi[gid] = {c2.x + he.x + m, c2.y + he.y + m, c2.z + he.z + m};
            return;
        }
        case kKindPlane: default:
            // A plane has effectively-infinite extent; clamp to a large finite
            // box so the LBVH morton quantization stays sane. The slab is thin
            // along the plane NORMAL = the body's LOCAL +Y (the amf:: plane
            // convention, analytical_manifold.hpp: n = frame.cy) — so the
            // local slab MUST be rotated by the body pose like every other
            // kind (review fix: the unrotated slab was thin in WORLD Y, which
            // missed every contact of a z-up ground plane posed with Y->Z).
            he = RotAbs(xf.rotation, math::Vec3{kPlaneLateralExtent,
                                                kPlaneThickness,
                                                kPlaneLateralExtent});
            break;
    }
    const math::Vec3 c = xf.position;
    const float m = margin;
    out_lo[gid] = {c.x - he.x - m, c.y - he.y - m, c.z - he.z - m};
    out_hi[gid] = {c.x + he.x + m, c.y + he.y + m, c.z + he.z + m};
}

// The per-env Karras build lives in collision/lbvh_batched.cuh
// (BuildLbvhBatchedNodes), shared by OpLbvhBuild below + the render TLAS.

// --- LbvhQueryPairs (tagged) ------------------------------------------------
__device__ __forceinline__ bool Overlaps(const collision::AABB& a,
                                          const collision::AABB& b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

// R4: signed collision-GROUP filter. Port of Newton broad_phase_common.py:132-
// 150 test_group_pair, with ONE Nuka divergence: group == 0 is the UNGROUPED
// default (collide-all), NOT Newton's "never collide" -- the cook fills group 0
// for every cooked + static collidable (cook_to_model.cpp:755,838 "group 0 ==
// the default collide-all group"), so the default scene must collide artic <->
// free-rigid <-> static. Non-zero groups then follow Newton's signed semantics:
//   positive group: collides with the SAME positive group OR any negative group.
//   negative group: collides with everything EXCEPT its own negative counterpart.
__device__ __forceinline__ bool TestGroupPair(int32_t ga, int32_t gb) {
    if (ga == 0 || gb == 0) return true;        // ungrouped default == collide-all
    if (ga > 0) return ga == gb || gb < 0;
    /* ga < 0 */ return ga != gb;
}

// Binary search a key in the ASCENDING excluded_pairs list.
__device__ __forceinline__ bool IsExcluded(const uint64_t* keys, uint32_t n,
                                           uint64_t key) {
    uint32_t lo = 0u, hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1u);
        const uint64_t mk = keys[mid];
        if (mk < key) lo = mid + 1u;
        else if (mk > key) hi = mid;
        else return true;
    }
    return false;
}

// One thread per (env x leaf). Walks the env's tree, emits canonical i<j pairs
// surviving the contype/conaffinity mask + excluded-list + same-body drop. The
// pair is stored env-LOCAL (a,b in [0,N)); the cross-env gate is implicit (the
// traversal only ever touches the env's own node slice). pair_count[env] is the
// per-env watermark.
__global__ void EnvQueryPairsKernel(const cg::LbvhNode* __restrict__ nodes,
                                    const float* __restrict__ shape_table,
                                    const uint64_t* __restrict__ excluded,
                                    uint32_t excluded_count,
                                    uint32_t bodies_per_env,
                                    uint32_t slot_stride,
                                    uint32_t rigid_slot_cap,
                                    uint32_t* __restrict__ out_pairs,   // elem:2 per slot
                                    uint32_t* __restrict__ out_count,
                                    uint32_t* __restrict__ env_status) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = bodies_per_env;
    if (N < 2u) {
        if (blockIdx.x == 0u && threadIdx.x == 0u) out_count[env] = 0u;
        return;
    }
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= N) return;
    const uint32_t internal = N - 1u;
    const uint32_t nbase = env * (2u * N - 1u);
    const cg::LbvhNode myleaf = nodes[nbase + internal + lane];
    const collision::AABB query = myleaf.aabb;
    const int32_t my_body = myleaf.left;
    const ShapeDev sa = LoadShape(shape_table, static_cast<uint32_t>(my_body));

    int32_t stack[64];
    int32_t top = 0;
    stack[top++] = 0;  // env-local root internal node
    while (top > 0) {
        const int32_t ni = stack[--top];
        const cg::LbvhNode node = nodes[nbase + ni];
        const int32_t children[2] = {node.left, node.right};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int32_t child = children[c];
            const bool is_leaf = (static_cast<uint32_t>(child) >= internal);
            const collision::AABB& cbox = nodes[nbase + child].aabb;
            if (!Overlaps(query, cbox)) continue;
            if (is_leaf) {
                const int32_t ob = nodes[nbase + child].left;
                if (my_body < ob) {
                    const ShapeDev sb =
                        LoadShape(shape_table, static_cast<uint32_t>(ob));
                    // Two static collidables (body_id < 0 each) have no reaction
                    // side on either body, so any contact between them is an
                    // unsolvable no-op (MuJoCo/Newton skip static-static). Drop the
                    // pair so static terrain never consumes the candidate/row budget
                    // against another static.
                    if (sa.body_id < 0 && sb.body_id < 0) continue;
                    // contype/conaffinity bitmask: collide iff
                    // (a.contype & b.conaffinity) || (b.contype & a.conaffinity).
                    const bool mask = ((sa.contype & sb.conaffinity) != 0u) ||
                                      ((sb.contype & sa.conaffinity) != 0u);
                    if (!mask) continue;
                    // R4: signed collision-group filter (default group 0 ==
                    // collide-all). The group lane is stored as a u32 bit-pattern;
                    // reinterpret as signed for the Newton signed semantics.
                    if (!TestGroupPair(static_cast<int32_t>(sa.group),
                                       static_cast<int32_t>(sb.group))) {
                        continue;
                    }
                    const uint32_t a32 = static_cast<uint32_t>(my_body);
                    const uint32_t b32 = static_cast<uint32_t>(ob);
                    const uint64_t key =
                        (static_cast<uint64_t>(a32) << 32) | static_cast<uint64_t>(b32);
                    if (excluded_count > 0u &&
                        IsExcluded(excluded, excluded_count, key)) {
                        continue;
                    }
                    const uint32_t slot = atomicAdd(&out_count[env], 1u);
                    // Full slot stride for addressing, but cap emission at rigid_slot_cap
                    // so body<->body never spills into the body<->particle sub-range.
                    if (slot < rigid_slot_cap) {
                        const size_t at =
                            (static_cast<size_t>(env) * slot_stride + slot) * 2u;
                        out_pairs[at + 0] = a32;
                        out_pairs[at + 1] = b32;
                    } else if (env_status != nullptr) {
                        // Capacity miss: this pair was dropped. Surface it (never
                        // silent); the watermark out_count[env] also stays honest.
                        atomicOr(&env_status[env], kEnvStatusPairOverflow);
                    }
                }
            } else if (top < 63) {
                stack[top++] = child;
            }
        }
    }
}

// Canonicalization pass 1: flatten each env's live pair slots into a u64 sort
// key over GLOBAL collidable ids (env-major -> one global sort is env-major),
// snapshot the pairs being sorted, and park dead slots at the key sentinel.
// The perm stores the LINEAR source slot id (env*stride + slot) because the
// global sort compacts live entries across env boundaries.
__global__ void PairSortFillKernel(uint32_t env_count, uint32_t slot_cap,
                                   uint32_t slot_stride,
                                   const uint32_t* __restrict__ counts,
                                   const uint32_t* __restrict__ pairs,
                                   uint32_t bodies_per_env,
                                   uint64_t* __restrict__ keys,
                                   uint32_t* __restrict__ perms,
                                   uint32_t* __restrict__ snap) {
    const uint64_t f = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(env_count) * slot_cap;
    if (f >= total) return;
    const uint32_t env = static_cast<uint32_t>(f / slot_cap);
    const uint32_t slot = static_cast<uint32_t>(f - env * slot_cap);
    const uint32_t live = min(counts[env], slot_cap);
    const size_t at = (static_cast<size_t>(env) * slot_stride + slot) * 2u;
    if (slot >= live) {
        keys[f] = 0xFFFFFFFFFFFFFFFFull;
        perms[f] = 0u;
        return;
    }
    const uint32_t a = pairs[at + 0];
    const uint32_t b = pairs[at + 1];
    snap[at + 0] = a;
    snap[at + 1] = b;
    // Emission guarantees a < b, so the packed key is already canonical; global
    // ids keep every env's segment contiguous and ascending under one sort.
    const uint64_t ga = static_cast<uint64_t>(env) * bodies_per_env + a;
    const uint64_t gb = static_cast<uint64_t>(env) * bodies_per_env + b;
    keys[f] = (ga << 32) | gb;
    perms[f] = static_cast<uint32_t>(at / 2u);
}

// Prefix of the per-env clamped live counts: dst_env's sorted entries occupy
// [prefix[dst_env], prefix[dst_env] + min(count,cap)) of the global order.
__global__ void PairClampCountsKernel(const uint32_t* __restrict__ counts,
                                      uint32_t slot_cap, uint32_t* __restrict__ clamped) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    clamped[e] = min(counts[e], slot_cap);
}

// Canonicalization pass 2: the stable sort ranked every live entry by pair id
// globally; recover the destination env from the key itself and its slot from
// the per-env prefix, then copy through the snapshot (dead slots past the
// watermark stay stale -- never consumed).
__global__ void PairSortScatterKernel(
    uint32_t env_count, uint32_t slot_cap,
    const uint64_t* __restrict__ keys_out, const uint32_t* __restrict__ perms,
    const uint32_t* __restrict__ prefix, uint32_t slot_stride,
    uint32_t bodies_per_env, const uint32_t* __restrict__ snap,
    uint32_t* __restrict__ pairs) {
    const uint64_t f = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= static_cast<uint64_t>(env_count) * slot_cap) return;
    if (keys_out[f] == 0xFFFFFFFFFFFFFFFFull) return;
    const uint64_t ga = keys_out[f] >> 32;
    const uint32_t dst_env = static_cast<uint32_t>(ga / bodies_per_env);
    const uint32_t dst_slot = static_cast<uint32_t>(f) - prefix[dst_env];
    const size_t dst_at = (static_cast<size_t>(dst_env) * slot_stride + dst_slot) * 2u;
    const size_t src_at = static_cast<size_t>(perms[f]) * 2u;
    pairs[dst_at + 0] = snap[src_at + 0];
    pairs[dst_at + 1] = snap[src_at + 1];
}

// --- ParticleGridBuild (CSR neighbors) --------------------------------------
// Cell keys are ENV-OFFSET (key = env*cells_per_env + local key) so each env
// owns a private cell span: env-major replicated particles occupy identical
// coordinates, and a shared grid would have every particle neighbor its own
// clones in other envs (review fix — cross-env coupling). The query kernels
// below offset the cell_start/cell_end base by env*cells_per_env so the LOCAL
// keys QueryParticleNeighbors computes index the env's own span. For
// env_count == 1 the offset is 0 and the behavior is byte-identical to the
// legacy single-world grid.
__global__ void GridCellKeysKernel(uint32_t particle_count,
                                   const math::Vec3* __restrict__ pos,
                                   cg::ParticleGridConfigDevice cfg,
                                   uint32_t particles_per_env,
                                   uint32_t cells_per_env,
                                   uint32_t* __restrict__ keys,
                                   uint32_t* __restrict__ idx) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    const math::Vec3 p = pos[i];
    const uint32_t env = i / particles_per_env;
    keys[i] = env * cells_per_env +
              cg::CellKeyFromPos(make_float3(p.x, p.y, p.z), cfg);
    idx[i] = i;
}

__global__ void GridCellRangesKernel(uint32_t particle_count,
                                     const uint32_t* __restrict__ sorted_keys,
                                     uint32_t* __restrict__ cell_start,
                                     uint32_t* __restrict__ cell_end) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    const uint32_t key = sorted_keys[i];
    if (i == 0u) cell_start[key] = 0u;
    else {
        const uint32_t prev = sorted_keys[i - 1u];
        if (prev != key) { cell_end[prev] = i; cell_start[key] = i; }
    }
    if (i == particle_count - 1u) cell_end[key] = particle_count;
}

__global__ void GridCountKernel(uint32_t particle_count,
                                const math::Vec3* __restrict__ pos, float radius,
                                cg::ParticleGridConfigDevice cfg,
                                uint32_t particles_per_env,
                                uint32_t cells_per_env,
                                const uint32_t* __restrict__ cell_start,
                                const uint32_t* __restrict__ cell_end,
                                const uint32_t* __restrict__ idx_sorted,
                                uint32_t* __restrict__ counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    const math::Vec3 pm = pos[i];
    const size_t cbase = static_cast<size_t>(i / particles_per_env) * cells_per_env;
    uint32_t scratch[cg::kParticleGridMaxNeighbors];
    const uint32_t n = cg::QueryParticleNeighbors(
        make_float3(pm.x, pm.y, pm.z), i, radius, cfg, cell_start + cbase,
        cell_end + cbase, idx_sorted, pos, scratch, cg::kParticleGridMaxNeighbors,
        nullptr);
    counts[i] = n;
}

__global__ void GridFillKernel(uint32_t particle_count,
                               const math::Vec3* __restrict__ pos, float radius,
                               cg::ParticleGridConfigDevice cfg,
                               uint32_t particles_per_env,
                               uint32_t cells_per_env,
                               const uint32_t* __restrict__ cell_start,
                               const uint32_t* __restrict__ cell_end,
                               const uint32_t* __restrict__ idx_sorted,
                               const uint32_t* __restrict__ offsets,
                               uint32_t slot_stride,
                               uint32_t* __restrict__ neighbor_idx,
                               uint32_t* __restrict__ counts,
                               uint32_t* __restrict__ env_status) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    const math::Vec3 pm = pos[i];
    const uint32_t env = i / particles_per_env;
    const size_t cbase = static_cast<size_t>(env) * cells_per_env;
    // Per-particle PRIVATE CSR slice (fixed cap slot_stride per particle); the
    // neighbor query writes them sorted ascending -> D1 by region (no append).
    uint32_t* slice = neighbor_idx + static_cast<size_t>(i) * slot_stride;
    bool overflow = false;
    const uint32_t n = cg::QueryParticleNeighbors(
        make_float3(pm.x, pm.y, pm.z), i, radius, cfg, cell_start + cbase,
        cell_end + cbase, idx_sorted, pos, slice, slot_stride, &overflow);
    counts[i] = n;
    // Surface a dropped neighbor (cap exhausted) per-env -> env_status readout;
    // PBF density would otherwise silently lose interactions in dense regions.
    if (overflow && env_status != nullptr) {
        atomicOr(&env_status[env], kEnvStatusNeighborOverflow);
    }
    (void)offsets;  // grid_neighbor_offset kept for the M6 flat-CSR consumer.
}

__global__ void ZeroU32Kernel(uint32_t* __restrict__ a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0u;
}

// Clear ONLY `bit` in each of the n words. env_status is shared across the
// broadphase + particle-grid ops (different bits), so each clears its own bit
// at start (order-independent) rather than zeroing the whole readout word.
__global__ void ClearBitU32Kernel(uint32_t* __restrict__ a, uint32_t n,
                                  uint32_t bit) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] &= ~bit;
}

// --- op entry points ---------------------------------------------------------

Status OpBuildAabbs(const ModelView& model, const DataView& data,
                    const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const BuildAabbsParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    if (p->env_count == 0u || p->bodies_per_env == 0u) return Status::Ok;
    const uint32_t total = p->env_count * p->bodies_per_env;
    const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(BuildAabbsKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               static_cast<const float*>(model.shape_table),
               static_cast<const math::Transform*>(data.body_pose),
               total, p->bodies_per_env, p->margin,
               data.body_aabb_lo, data.body_aabb_hi);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpLbvhBuild(const ModelView& /*model*/, const DataView& data,
                   const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const LbvhBuildParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    const uint32_t N = p->bodies_per_env;
    const uint32_t E = p->env_count;
    if (E == 0u || N < 2u) return Status::Ok;  // a <2-body env has no pairs.

    auto* nodes = reinterpret_cast<cg::LbvhNode*>(data.lbvh_nodes);
    // Shared batched env-build over the arena's split lo/hi AABBs. Leaf `.left`
    // is the env-LOCAL body index (the query maps it via the env's node slice).
    cg::BuildLbvhBatchedNodes(stream, /*device_id=*/0, data.body_aabb_lo,
                              data.body_aabb_hi, E, N, nodes, data.lbvh_morton,
                              data.lbvh_index, data.lbvh_sortkey, data.lbvh_visit);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpLbvhQueryPairs(const ModelView& model, const DataView& data,
                        const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const LbvhQueryPairsParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    const uint32_t N = p->bodies_per_env;
    const uint32_t E = p->env_count;
    if (E == 0u) return Status::Ok;
    // Zero the per-env pair watermark + clear THIS op's env_status overflow bit.
    {
        const uint32_t b = (E + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.pair_count, E);
        if (data.env_status != nullptr) {
            LaunchCuda(ClearBitU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                       data.env_status, E, kEnvStatusPairOverflow);
        }
    }
    if (N < 2u) return Status::Ok;
    auto* nodes = reinterpret_cast<const cg::LbvhNode*>(data.lbvh_nodes);
    const uint32_t leaf_blocks = (N + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(EnvQueryPairsKernel, dim3(leaf_blocks, E), dim3(kBlockSize), 0u, stream,
               nodes, static_cast<const float*>(model.shape_table),
               static_cast<const uint64_t*>(model.excluded_pairs),
               p->excluded_count, N, p->max_contacts_per_env, p->rigid_slot_cap,
               data.candidate_pairs, data.pair_count, data.env_status);
    // Canonicalize the emitted stream: the atomicAdd slot claim above orders
    // pairs by warp-scheduling race, which would leak into contact slots, row
    // slots, the GS sweep order, and the warm-start match as last-ULP float
    // noise. One stable radix sort over global pair ids makes the stream a pure
    // function of the ids (the repo determinism model: integer-only atomics,
    // radix stable_sort, sorted compact output).
    const uint64_t sort_slots64 =
        static_cast<uint64_t>(E) * p->rigid_slot_cap;
    if (sort_slots64 > 0u) {
        if (sort_slots64 > 0xFFFFFFFFull || data.pair_sort_scratch == nullptr) {
            return Status::Failed;  // LOUD: no silent nondeterministic fallback.
        }
        const uint32_t sort_slots = static_cast<uint32_t>(sort_slots64);
        const PairSortScratchLayout sl(sort_slots, E);
        char* sbase = reinterpret_cast<char*>(data.pair_sort_scratch);
        void* sort_temp = sbase;
        uint64_t* keys_in = reinterpret_cast<uint64_t*>(sbase + sl.keys_in_off);
        uint64_t* keys_out = reinterpret_cast<uint64_t*>(sbase + sl.keys_out_off);
        uint32_t* perms_in = reinterpret_cast<uint32_t*>(sbase + sl.perms_in_off);
        uint32_t* perms_out = reinterpret_cast<uint32_t*>(sbase + sl.perms_out_off);
        uint32_t* snap = reinterpret_cast<uint32_t*>(sbase + sl.snap_off);
        uint32_t* clamped = reinterpret_cast<uint32_t*>(sbase + sl.clamp_off);
        uint32_t* prefix = reinterpret_cast<uint32_t*>(sbase + sl.prefix_off);
        size_t temp_bytes = static_cast<size_t>(sl.temp_bytes);
        const uint32_t blocks = static_cast<uint32_t>(
            (sort_slots64 + kBlockSize - 1u) / kBlockSize);
        LaunchCuda(PairSortFillKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   E, p->rigid_slot_cap, p->max_contacts_per_env, data.pair_count,
                   data.candidate_pairs, N, keys_in, perms_in, snap);
        (void)cub::DeviceRadixSort::SortPairs(
            sort_temp, temp_bytes, keys_in, keys_out, perms_in, perms_out,
            static_cast<int>(sort_slots), 0, 64, stream);
        {
            const uint32_t eb = (E + kBlockSize - 1u) / kBlockSize;
            LaunchCuda(PairClampCountsKernel, dim3(eb), dim3(kBlockSize), 0u,
                       stream, data.pair_count, p->rigid_slot_cap, clamped);
            size_t scan_bytes = static_cast<size_t>(sl.temp_bytes);
            (void)cub::DeviceScan::ExclusiveScan(
                sort_temp, scan_bytes, clamped, prefix, NkScanSumOp{}, 0u,
                static_cast<int>(E), stream);
        }
        LaunchCuda(PairSortScatterKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   E, p->rigid_slot_cap, keys_out, perms_out, prefix,
                   p->max_contacts_per_env, N, snap, data.candidate_pairs);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpParticleGridBuild(const ModelView& /*model*/, const DataView& data,
                           const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleGridBuildParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->particle_count == 0u || p->cell_size <= 0.0f) return Status::Ok;
    cg::ParticleGridConfigDevice cfg;
    cfg.cell_size = make_float3(p->cell_size, p->cell_size, p->cell_size);
    const float inv = 1.0f / p->cell_size;
    cfg.inv_cell_size = make_float3(inv, inv, inv);
    cfg.grid_min = make_float3(p->grid_min[0], p->grid_min[1], p->grid_min[2]);
    cfg.grid_dims = make_uint3(p->grid_dims[0], p->grid_dims[1], p->grid_dims[2]);
    const uint32_t Np = p->particle_count;
    const uint64_t cells64 = static_cast<uint64_t>(p->grid_dims[0]) *
                             p->grid_dims[1] * p->grid_dims[2];
    if (cells64 == 0u) return Status::Ok;
    const uint32_t E = p->env_count == 0u ? 1u : p->env_count;
    const uint32_t Ppe = p->particles_per_env == 0u ? Np : p->particles_per_env;
    // LOUD capacity guards (review fix): the cell-range arrays are sized
    // max_grid_cells x env_count — a dims product beyond the cooked capacity
    // would scatter cell keys past the segment (silent arena corruption), and
    // cells*env_count must fit the u32 env-offset key.
    if (cells64 > p->cells_capacity) return Status::Failed;
    if (cells64 * E > 0xFFFFFFFFull) return Status::Failed;
    const uint32_t cells = static_cast<uint32_t>(cells64);
    const uint32_t blocks = (Np + kBlockSize - 1u) / kBlockSize;
    // M6: PBF builds the neighbor grid on the PREDICTED positions (legacy PBF
    // step order), so pos_source routes the op to pbf_predicted_pos; the M5
    // default (0) keeps the particle_pos source. All downstream PBF density /
    // lambda / correction kernels read pbf_predicted_pos, so the neighbor list
    // and the queries are over the SAME positions (D1 + the legacy semantics).
    const auto* pos =
        (p->pos_source == kGridPosSourcePbfPredicted)
            ? static_cast<const math::Vec3*>(data.pbf_predicted_pos)
            : static_cast<const math::Vec3*>(data.particle_pos);

    // Partition the pre-allocated scratch [cub temp | keys-out | idx-out] so the
    // sort/scan never cudaMalloc/sync mid-capture (the grid op joins the graph).
    const GridSortScratchLayout sl(Np);
    char* sbase = reinterpret_cast<char*>(data.grid_sort_scratch);
    void* sort_temp = sbase;
    uint32_t* keys_out = reinterpret_cast<uint32_t*>(sbase + sl.keys_off);
    uint32_t* idx_out  = reinterpret_cast<uint32_t*>(sbase + sl.idx_off);
    size_t sort_temp_bytes = static_cast<size_t>(sl.temp_bytes);
    LaunchCuda(GridCellKeysKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, pos, cfg, Ppe, cells, data.grid_cell_key,
               data.grid_particle_idx);
    // Stable radix sort (byte-identical to the prior thrust stable_sort_by_key,
    // which dispatched here); out-of-place, then D2D-copied back to keep in-place.
    (void)cub::DeviceRadixSort::SortPairs(
        sort_temp, sort_temp_bytes, data.grid_cell_key, keys_out,
        data.grid_particle_idx, idx_out, static_cast<int>(Np), 0, 32, stream);
    (void)cudaMemcpyAsync(data.grid_cell_key, keys_out, Np * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice, stream);
    (void)cudaMemcpyAsync(data.grid_particle_idx, idx_out, Np * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice, stream);
    {  // zero the per-env cell ranges (cells x env_count entries).
        const uint32_t zn = cells * E;
        const uint32_t b = (zn + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.grid_cell_start, zn);
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.grid_cell_end, zn);
    }
    if (data.env_status != nullptr) {  // clear THIS op's neighbor-overflow bit.
        const uint32_t b = (E + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ClearBitU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.env_status, E, kEnvStatusNeighborOverflow);
    }
    LaunchCuda(GridCellRangesKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, data.grid_cell_key, data.grid_cell_start, data.grid_cell_end);
    LaunchCuda(GridCountKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, pos, p->query_radius, cfg, Ppe, cells, data.grid_cell_start,
               data.grid_cell_end, data.grid_particle_idx, data.grid_neighbor_count);
    // Exclusive scan of the count -> flat CSR offset (byte-identical to the prior
    // thrust exclusive_scan; out-of-place, reusing the capture-safe temp region).
    {
        size_t scan_temp_bytes = static_cast<size_t>(sl.temp_bytes);
        (void)cub::DeviceScan::ExclusiveScan(
            sort_temp, scan_temp_bytes, data.grid_neighbor_count,
            data.grid_neighbor_offset, NkScanSumOp{}, 0u, static_cast<int>(Np), stream);
    }
    LaunchCuda(GridFillKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, pos, p->query_radius, cfg, Ppe, cells, data.grid_cell_start,
               data.grid_cell_end, data.grid_particle_idx, data.grid_neighbor_offset,
               cg::kParticleGridMaxNeighbors, data.grid_neighbor_idx,
               data.grid_neighbor_count, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

uint64_t GridSortScratchBytes(uint32_t particle_count) {
    if (particle_count == 0u) {
        return 0u;
    }
    return GridSortScratchLayout(particle_count).total;
}

uint64_t PairSortScratchBytes(uint32_t total_sort_slots, uint32_t env_count) {
    if (total_sort_slots == 0u || env_count == 0u) {
        return 0u;
    }
    return PairSortScratchLayout(total_sort_slots, env_count).total;
}

void RegisterNkBroadphaseOps() {
    SetCudaOp(NkOp::BuildAabbs, &OpBuildAabbs);
    SetCudaOp(NkOp::LbvhBuild, &OpLbvhBuild);
    SetCudaOp(NkOp::LbvhQueryPairs, &OpLbvhQueryPairs);
    SetCudaOp(NkOp::ParticleGridBuild, &OpParticleGridBuild);
}

}  // namespace nuka::phi
