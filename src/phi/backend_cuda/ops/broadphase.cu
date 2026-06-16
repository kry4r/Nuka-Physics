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
// algorithm + the 2x thrust::stable_sort_by_key D1 anchors are preserved; only
// the I/O wiring changes (inputs are the arena/Model device pointers, the node/
// morton/visit scratch is preallocated arena-resident, the output is the
// candidate_pairs / grid_* Data fields).
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
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "collision/lbvh_node.cuh"      // LbvhNode / LbvhDelta / LbvhMerge
#include "collision/morton_codes.cuh"   // Morton3D30
#include "collision/shape_kind.hpp"     // nuka::collision::ShapeKind (R2: one enum)
#include "collision/particle_grid_traversal.cuh"  // ParticleGridConfigDevice / QueryParticleNeighbors
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace {

namespace cg = ::nuka::collision::gpu;
constexpr uint32_t kBlockSize = 128u;
constexpr uint32_t kParticleGridMaxNeighbors = 32u;  // mirror particle_uniform_grid.

// shape_table record: R1 GREW it 8 -> 10 packed f32 / body row
// (Model::PairDrivenShape). Lanes 0..7 unchanged; lanes 8/9 = body_id (int32) +
// group (uint32), the shape->body indirection (INERT in Phase 0).
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
    const float* q = table + static_cast<size_t>(row) * 10u;
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
        // Hull / SDF mesh: conservative BOUND-RADIUS sphere (p[0] = the cooked
        // max |vertex| — rotation-invariant, so no RotAbs needed).
        case kKindConvexHull:
        case kKindSdfMesh:    r = s.p[0]; he = {r, r, r}; break;
        case kKindPlane: default:
            // A plane has effectively-infinite extent; clamp to a large finite
            // box so the LBVH morton quantization stays sane. The slab is thin
            // along the plane NORMAL = the body's LOCAL +Y (the amf:: plane
            // convention, analytical_manifold.hpp: n = frame.cy) — so the
            // local slab MUST be rotated by the body pose like every other
            // kind (review fix: the unrotated slab was thin in WORLD Y, which
            // missed every contact of a z-up ground plane posed with Y->Z).
            he = RotAbs(xf.rotation, math::Vec3{1.0e6f, 1.0e-3f, 1.0e6f});
            break;
    }
    const math::Vec3 c = xf.position;
    const float m = margin;
    out_lo[gid] = {c.x - he.x - m, c.y - he.y - m, c.z - he.z - m};
    out_hi[gid] = {c.x + he.x + m, c.y + he.y + m, c.z + he.z + m};
}

// --- LBVH per-env build (Karras) --------------------------------------------
// Each env owns a contiguous [env*N, env*N+N) body slice + a 2N-1 node slice in
// the lbvh_nodes arena field + an N-entry morton/index/visit slice. The build
// is the standalone BuildLbvhImpl pipeline run per env (the scene-bound, morton,
// stable_sort, init/build/propagate kernels), but the bound + sort happen per
// env so envs never share a tree (env-major). NOTE: the per-env stable_sort is
// run on the WHOLE morton array with the (env, code) compound key so a single
// thrust call orders all envs without cross-env mixing — the env id is the high
// 16 bits of the sort key (bodies_per_env < 2^16 for every cooked scene).

// AABB center morton (per env, into the env-local [0,1] normalized box). We do a
// simple per-env reduction in a single block per env (bodies_per_env is small —
// the union/grasp scene has O(few) movable bodies; a generalized rigid scene is
// 10s-100s, still one block).
__global__ void EnvMortonKernel(const math::Vec3* __restrict__ lo,
                                const math::Vec3* __restrict__ hi,
                                uint32_t bodies_per_env,
                                uint32_t* __restrict__ out_morton,
                                uint32_t* __restrict__ out_index) {
    const uint32_t env = blockIdx.x;
    const uint32_t base = env * bodies_per_env;
    // Single thread per env folds the env's center bound serially (D1 fixed
    // order — bodies_per_env is small) then writes each leaf's Morton code +
    // env-LOCAL index. The bound is private to the env, so envs never mix.
    if (threadIdx.x != 0u) return;
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < bodies_per_env; ++i) {
        const math::Vec3 a = lo[base + i];
        const math::Vec3 b = hi[base + i];
        const float c[3] = {0.5f * (a.x + b.x), 0.5f * (a.y + b.y),
                            0.5f * (a.z + b.z)};
        for (int k = 0; k < 3; ++k) {
            mn[k] = fminf(mn[k], c[k]);
            mx[k] = fmaxf(mx[k], c[k]);
        }
    }
    const float inv[3] = {
        (mx[0] > mn[0]) ? 1.0f / (mx[0] - mn[0]) : 0.0f,
        (mx[1] > mn[1]) ? 1.0f / (mx[1] - mn[1]) : 0.0f,
        (mx[2] > mn[2]) ? 1.0f / (mx[2] - mn[2]) : 0.0f};
    for (uint32_t i = 0; i < bodies_per_env; ++i) {
        const math::Vec3 a = lo[base + i];
        const math::Vec3 b = hi[base + i];
        const float nx = (0.5f * (a.x + b.x) - mn[0]) * inv[0];
        const float ny = (0.5f * (a.y + b.y) - mn[1]) * inv[1];
        const float nz = (0.5f * (a.z + b.z) - mn[2]) * inv[2];
        out_morton[base + i] = cg::Morton3D30(nx, ny, nz);
        out_index[base + i] = i;  // env-LOCAL leaf index (0..N-1)
    }
}

// Init nodes for one env's tree (node slice base = env*(2N-1)).
__global__ void EnvInitNodesKernel(const math::Vec3* __restrict__ lo,
                                   const math::Vec3* __restrict__ hi,
                                   const uint32_t* __restrict__ sorted_index,
                                   uint32_t bodies_per_env,
                                   cg::LbvhNode* __restrict__ nodes) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = bodies_per_env;
    const uint32_t node_count = 2u * N - 1u;
    const uint32_t internal = N - 1u;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;
    const uint32_t nbase = env * node_count;
    const uint32_t bbase = env * N;
    if (idx < internal) {
        nodes[nbase + idx].parent = -1;
        return;
    }
    const uint32_t lane = idx - internal;
    const uint32_t local_body = sorted_index[bbase + lane];  // env-local id
    cg::LbvhNode leaf;
    leaf.left = static_cast<int32_t>(local_body);
    leaf.right = -1;
    leaf.parent = -1;
    const math::Vec3 a = lo[bbase + local_body];
    const math::Vec3 b = hi[bbase + local_body];
    leaf.aabb.min = {a.x, a.y, a.z};
    leaf.aabb.max = {b.x, b.y, b.z};
    nodes[nbase + idx] = leaf;
}

__global__ void EnvBuildInternalKernel(const uint32_t* __restrict__ morton_sorted,
                                       uint32_t bodies_per_env,
                                       cg::LbvhNode* __restrict__ nodes) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = bodies_per_env;
    const uint32_t internal = N - 1u;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= internal) return;
    const uint32_t nbase = env * (2u * N - 1u);
    const uint32_t mbase = env * N;
    const uint32_t* morton = morton_sorted + mbase;
    const int i = static_cast<int>(idx);

    const int d_l = cg::LbvhDelta(i, i - 1, N, morton);
    const int d_r = cg::LbvhDelta(i, i + 1, N, morton);
    const int d = (d_r >= d_l) ? 1 : -1;
    const int delta_min = cg::LbvhDelta(i, i - d, N, morton);
    int l_max = 2;
    while (cg::LbvhDelta(i, i + l_max * d, N, morton) > delta_min) l_max *= 2;
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (cg::LbvhDelta(i, i + (l + t) * d, N, morton) > delta_min) l += t;
    }
    const int j = i + l * d;
    const int delta_node = cg::LbvhDelta(i, j, N, morton);
    int s = 0;
    int t_div = l;
    do {
        t_div = (t_div + 1) >> 1;
        if (cg::LbvhDelta(i, i + (s + t_div) * d, N, morton) > delta_node) s += t_div;
    } while (t_div > 1);
    const int gamma = i + s * d + min(d, 0);
    const int first = min(i, j);
    const int last = max(i, j);
    const int32_t left_child = (first == gamma)
        ? static_cast<int32_t>(internal + gamma) : static_cast<int32_t>(gamma);
    const int32_t right_child = (last == gamma + 1)
        ? static_cast<int32_t>(internal + gamma + 1) : static_cast<int32_t>(gamma + 1);
    nodes[nbase + i].left = left_child;
    nodes[nbase + i].right = right_child;
    nodes[nbase + left_child].parent = i;
    nodes[nbase + right_child].parent = i;
}

__global__ void EnvPropagateKernel(uint32_t bodies_per_env,
                                   cg::LbvhNode* __restrict__ nodes,
                                   uint32_t* __restrict__ visit) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = bodies_per_env;
    const uint32_t internal = N - 1u;
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= N) return;
    const uint32_t nbase = env * (2u * N - 1u);
    const uint32_t vbase = env * N;  // visit sized per:body (>= internal).
    int32_t node = nodes[nbase + internal + lane].parent;
    while (node >= 0) {
        __threadfence();
        const uint32_t prev = atomicAdd(&visit[vbase + node], 1u);
        if (prev == 0u) return;
        const cg::LbvhNode self = nodes[nbase + node];
        nodes[nbase + node].aabb =
            cg::LbvhMerge(nodes[nbase + self.left].aabb,
                          nodes[nbase + self.right].aabb);
        node = nodes[nbase + node].parent;
    }
}

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
                                    uint32_t* __restrict__ out_pairs,   // elem:2 per slot
                                    uint32_t* __restrict__ out_count) {
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
                    if (slot < slot_stride) {
                        const size_t at =
                            (static_cast<size_t>(env) * slot_stride + slot) * 2u;
                        out_pairs[at + 0] = a32;
                        out_pairs[at + 1] = b32;
                    }
                }
            } else if (top < 63) {
                stack[top++] = child;
            }
        }
    }
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
    uint32_t scratch[kParticleGridMaxNeighbors];
    const uint32_t n = cg::QueryParticleNeighbors(
        make_float3(pm.x, pm.y, pm.z), i, radius, cfg, cell_start + cbase,
        cell_end + cbase, idx_sorted, pos, scratch, kParticleGridMaxNeighbors,
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
                               uint32_t* __restrict__ counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    const math::Vec3 pm = pos[i];
    const size_t cbase = static_cast<size_t>(i / particles_per_env) * cells_per_env;
    // Per-particle PRIVATE CSR slice (fixed cap slot_stride per particle); the
    // neighbor query writes them sorted ascending -> D1 by region (no append).
    uint32_t* slice = neighbor_idx + static_cast<size_t>(i) * slot_stride;
    bool overflow = false;
    const uint32_t n = cg::QueryParticleNeighbors(
        make_float3(pm.x, pm.y, pm.z), i, radius, cfg, cell_start + cbase,
        cell_end + cbase, idx_sorted, pos, slice, slot_stride, &overflow);
    counts[i] = n;
    (void)offsets;  // grid_neighbor_offset kept for the M6 flat-CSR consumer.
}

__global__ void ZeroU32Kernel(uint32_t* __restrict__ a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0u;
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
    auto* morton = data.lbvh_morton;
    auto* index = data.lbvh_index;
    auto* visit = data.lbvh_visit;

    // Zero the visit counters (per:body slice).
    {
        const uint32_t n = E * N;
        const uint32_t b = (n + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream, visit, n);
    }
    // Per-env Morton code + env-local leaf index (one thread per env folds the
    // env's own center bound — envs never mix; the per-segment stable_sort below
    // is the D1 radix anchor).
    LaunchCuda(EnvMortonKernel, dim3(E), dim3(kBlockSize), 0u, stream,
               data.body_aabb_lo, data.body_aabb_hi, N, morton, index);
    // Per-env stable_sort_by_key (morton ascending; index follows). Segment sort
    // keeps envs disjoint and preserves the D1 radix anchor.
    for (uint32_t e = 0u; e < E; ++e) {
        const size_t base = static_cast<size_t>(e) * N;
        thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                                   morton + base, morton + base + N, index + base);
    }
    if (cudaGetLastError() != cudaSuccess) return Status::Failed;

    const uint32_t node_count = 2u * N - 1u;
    const uint32_t node_blocks = (node_count + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(EnvInitNodesKernel, dim3(node_blocks, E), dim3(kBlockSize), 0u, stream,
               data.body_aabb_lo, data.body_aabb_hi, index, N, nodes);
    const uint32_t internal_blocks = ((N - 1u) + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(EnvBuildInternalKernel, dim3(internal_blocks, E), dim3(kBlockSize),
               0u, stream, morton, N, nodes);
    const uint32_t leaf_blocks = (N + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(EnvPropagateKernel, dim3(leaf_blocks, E), dim3(kBlockSize), 0u, stream,
               N, nodes, visit);
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
    // Zero the per-env pair watermark.
    {
        const uint32_t b = (E + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.pair_count, E);
    }
    if (N < 2u) return Status::Ok;
    auto* nodes = reinterpret_cast<const cg::LbvhNode*>(data.lbvh_nodes);
    const uint32_t leaf_blocks = (N + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(EnvQueryPairsKernel, dim3(leaf_blocks, E), dim3(kBlockSize), 0u, stream,
               nodes, static_cast<const float*>(model.shape_table),
               static_cast<const uint64_t*>(model.excluded_pairs),
               p->excluded_count, N, p->max_contacts_per_env,
               data.candidate_pairs, data.pair_count);
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

    LaunchCuda(GridCellKeysKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, pos, cfg, Ppe, cells, data.grid_cell_key,
               data.grid_particle_idx);
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                               data.grid_cell_key, data.grid_cell_key + Np,
                               data.grid_particle_idx);
    {  // zero the per-env cell ranges (cells x env_count entries).
        const uint32_t zn = cells * E;
        const uint32_t b = (zn + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.grid_cell_start, zn);
        LaunchCuda(ZeroU32Kernel, dim3(b), dim3(kBlockSize), 0u, stream,
                   data.grid_cell_end, zn);
    }
    LaunchCuda(GridCellRangesKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, data.grid_cell_key, data.grid_cell_start, data.grid_cell_end);
    LaunchCuda(GridCountKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, pos, p->query_radius, cfg, Ppe, cells, data.grid_cell_start,
               data.grid_cell_end, data.grid_particle_idx, data.grid_neighbor_count);
    // Exclusive scan of the count -> flat CSR offset (M6 flat consumer).
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           data.grid_neighbor_count, data.grid_neighbor_count + Np,
                           data.grid_neighbor_offset);
    LaunchCuda(GridFillKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               Np, pos, p->query_radius, cfg, Ppe, cells, data.grid_cell_start,
               data.grid_cell_end, data.grid_particle_idx, data.grid_neighbor_offset,
               kParticleGridMaxNeighbors, data.grid_neighbor_idx,
               data.grid_neighbor_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkBroadphaseOps() {
    SetCudaOp(NkOp::BuildAabbs, &OpBuildAabbs);
    SetCudaOp(NkOp::LbvhBuild, &OpLbvhBuild);
    SetCudaOp(NkOp::LbvhQueryPairs, &OpLbvhQueryPairs);
    SetCudaOp(NkOp::ParticleGridBuild, &OpParticleGridBuild);
}

}  // namespace nuka::phi
