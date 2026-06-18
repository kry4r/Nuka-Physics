// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M5 NarrowphaseSdf op (plan §3.5: the SDF MAIN PATH for
// arbitrary mesh contact, Genesis-style point-vs-grid SAMPLING — NOT the Newton
// 1-point witness, which stays the CPU tier until M9).
//
// For each candidate pair where the OTHER shape carries a cooked sparse narrow-
// band SDF (shape.sdf_grid != ~0), the SAMPLING shape's cooked sample-point set
// (hull verts + edge midpoints, the .nka SAMP chunk, staged into samp_points /
// samp_ranges) is transformed into the SDF shape's local frame and queried
// against the SDF grid (value/gradient -> contact point/normal/depth). The
// deepest-K (K = max_contacts_per_pair) penetrating samples are selected in a
// DETERMINISTIC fixed order (sample index ascending on a penetration tie) and
// written into the ucontact_* manifold scratch — the SAME layout AssembleRows
// consumes (so AssembleRows is UNCHANGED).
//
// The SDF sampler is the SHARED runtime/sdf header (sparse_sdf_query.cuh,
// trilinear, binary search, fixed accumulation order -> D1). The Model SDF cell
// tables are the SdfDeviceWorld layout moved INTO the Model device buffer
// (sdf_headers / sdf_cell_keys / sdf_cell_values / sdf_cell_gradients).
//
// FAMILY GATING: EARLY-EXITS unless family == kContactFamilyPairDriven (the
// union slot-template family detects fingertip x cup with cvx::SphereHull, NOT
// the SDF path). A standalone test launcher (LaunchNarrowphaseSdf) drives the
// converted GJK/EPA precision oracle (test_gjk_epa_convex.cpp) against analytic
// truth with hand-built device SDF + sample buffers.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // kShapeTableRowStride (shared)
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/sdf_types.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace nkops {

namespace sdf = ::nuka::runtime::sdf;

// SDF header row stride (f32 / grid): origin.xyz, voxel_size, dims.xyz, cell_offset.
// FLAG: model.cpp:91 (ElementCount(SdfHeaders) = max_sdf_grids*8) and model.cpp:364
// (staging stride) hardcode the same 8; the divergence-proof home is sdf_types.cuh
// / sparse_sdf_query.cuh so both TUs share THIS constant.
constexpr uint32_t kSdfHeaderStride = 8u;

// Load one SDF grid view from the Model sdf_* device tables (kSdfHeaderStride f32
// header + flat cell arrays). Mirrors SparseSdfDevice; cell_* point into the buffer.
__device__ sdf::SparseSdfDevice LoadSdfGrid(const float* headers,
                                            const uint32_t* cell_counts,
                                            const uint64_t* keys,
                                            const float* values,
                                            const math::Vec3* grads,
                                            uint32_t grid) {
    const float* h = headers + static_cast<size_t>(grid) * kSdfHeaderStride;
    sdf::SparseSdfDevice s;
    s.origin = {h[0], h[1], h[2]};
    s.voxel_size = h[3];
    s.dims[0] = __float_as_uint(h[4]);
    s.dims[1] = __float_as_uint(h[5]);
    s.dims[2] = __float_as_uint(h[6]);
    const uint32_t cell_offset = __float_as_uint(h[7]);
    s.cell_keys = keys + cell_offset;
    s.cell_values = values + cell_offset;
    s.cell_gradients = grads + cell_offset;
    s.cell_count = cell_counts[grid];
    return s;
}

// Insert one penetrating sample into the per-slot deepest-K manifold buffer,
// keeping the slots ordered by DESCENDING depth with the sample index as the
// tie-break (the deterministic fixed order). out_* are the 4-slot ucontact
// arrays for ONE contact slot (gid).
__device__ void DeepestKInsert(math::Vec3 point, math::Vec3 normal, float depth,
                               uint32_t /*sample_id*/, uint32_t k,
                               math::Vec3* pt, math::Vec3* nm, float* dp,
                               uint32_t* count) {
    uint32_t n = *count;
    // Find the insertion position (descending depth). Equal-depth ties keep the
    // earlier-inserted sample (sample index ascending) — stable insert.
    uint32_t pos = n;
    while (pos > 0u && dp[pos - 1u] < depth) --pos;
    if (pos >= k) return;  // shallower than every kept slot and buffer full.
    // Shift the tail down (drop the last if full).
    uint32_t last = (n < k) ? n : (k - 1u);
    for (uint32_t i = last; i > pos; --i) {
        pt[i] = pt[i - 1u]; nm[i] = nm[i - 1u]; dp[i] = dp[i - 1u];
    }
    pt[pos] = point; nm[pos] = normal; dp[pos] = depth;
    *count = (n < k) ? (n + 1u) : k;
}

}  // namespace nkops

namespace {

using namespace ::nuka::phi::nkops;
namespace sdf = ::nuka::runtime::sdf;

// Read just lane 7 (sdf_grid; ~0u when the shape has no cooked SDF) of a
// shape_table row, using the shared row stride.
__forceinline__ __device__ uint32_t LoadShapeSdfGrid(const float* table,
                                                     uint32_t row) {
    return __float_as_uint(table[static_cast<size_t>(row) * nkops::kShapeTableRowStride + 7u]);
}

// One thread per (env x sampling-body). Samples the body's SAMP slice against
// the target SDF grid and writes the deepest-K manifold into the body's
// ucontact slot. The contact NORMAL is the SDF gradient (separation dir for the
// SAMPLING shape = A), depth = -phi (penetration positive inside the surface).
__global__ void NarrowphaseSdfKernel(const float* __restrict__ samp_points,
                                     const uint32_t* __restrict__ samp_ranges,
                                     const float* __restrict__ shape_table,
                                     const float* __restrict__ sdf_headers,
                                     const uint32_t* __restrict__ sdf_cell_count,
                                     const uint64_t* __restrict__ sdf_keys,
                                     const float* __restrict__ sdf_values,
                                     const math::Vec3* __restrict__ sdf_grads,
                                     const math::Transform* __restrict__ body_pose,
                                     const SdfPairDev* __restrict__ pairs,
                                     uint32_t pair_count,
                                     uint32_t k,
                                     float margin,
                                     uint32_t slot_stride,
                                     uint32_t* __restrict__ ucount,
                                     math::Vec3* __restrict__ upoint,
                                     math::Vec3* __restrict__ unormal,
                                     float* __restrict__ udepth,
                                     uint32_t* __restrict__ contact_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pair_count) return;
    const SdfPairDev pr = pairs[gid];
    const uint32_t env = pr.env;
    const uint32_t slot = pr.slot;          // ucontact slot for this pair.
    const uint32_t out_gid = env * slot_stride + slot;

    // Sampling body world pose + its SAMP slice.
    const math::Transform sxf = body_pose[env * pr.bodies_per_env + pr.sample_body];
    const uint32_t soff = samp_ranges[pr.sample_body * 2u + 0u];
    const uint32_t scnt = samp_ranges[pr.sample_body * 2u + 1u];

    // Target SDF grid (the OTHER shape) world pose -> its local frame.
    const math::Transform txf = body_pose[env * pr.bodies_per_env + pr.target_body];
    const uint32_t grid = pr.sdf_grid;
    const sdf::SparseSdfDevice sdf_grid = LoadSdfGrid(
        sdf_headers, sdf_cell_count, sdf_keys, sdf_values, sdf_grads, grid);

    // Per-slot deepest-K manifold accumulators (k <= 4).
    math::Vec3 pt[4]; math::Vec3 nm[4]; float dp[4];
    uint32_t kept = 0u;

    for (uint32_t s = 0u; s < scnt; ++s) {
        const math::Vec3 local{samp_points[(soff + s) * 3u + 0u],
                               samp_points[(soff + s) * 3u + 1u],
                               samp_points[(soff + s) * 3u + 2u]};
        // sample world = sxf o local; then into the SDF's local frame.
        const math::Vec3 world = SdfTransformPoint(sxf, local);
        const math::Vec3 q = SdfInverseTransformPoint(txf, world);
        math::Vec3 grad{0, 0, 0};
        const float phi = sdf::sparse_sdf_sample(sdf_grid, q, grad);
        if (phi >= sdf::SparseSdfDevice::kOutsideBand) continue;
        const float depth = -phi + margin;   // penetration (positive inside).
        if (depth <= 0.0f) continue;
        // Contact normal: the SDF gradient in WORLD, pointing OUT of the target
        // surface = separation dir for the sampling shape (A). Rotate grad
        // (local) by the target rotation.
        const math::Vec3 gw = SdfRotate(txf.rotation, grad);
        const float gl = sqrtf(gw.x * gw.x + gw.y * gw.y + gw.z * gw.z);
        const math::Vec3 n = (gl > 1.0e-12f)
            ? math::Vec3{gw.x / gl, gw.y / gl, gw.z / gl}
            : math::Vec3{0.0f, 0.0f, 1.0f};
        // Contact point: the sample, pushed back onto the surface along n.
        const math::Vec3 cp{world.x - n.x * depth, world.y - n.y * depth,
                            world.z - n.z * depth};
        DeepestKInsert(cp, n, depth, s, k, pt, nm, dp, &kept);
    }

    ucount[out_gid] = kept;
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(out_gid) * 4u + i;
        if (i < kept) { upoint[at] = pt[i]; unormal[at] = nm[i]; udepth[at] = dp[i]; }
        else { upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f; }
    }
    if (kept > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], kept);
    }
}

// In-pipeline SDF narrowphase: one thread per (env x candidate slot). When the
// slot's broadphase pair has EXACTLY one side carrying a cooked SDF grid, sample
// the OTHER (sampling) shape's SAMP point slice against that grid and write the
// deepest-K manifold into THIS slot's unified ucontact_* (side a == the sampling
// body, side b == the SDF body). A non-SDF slot is LEFT untouched (it belongs to
// NarrowphasePrimitives / NarrowphaseHeightfield, which ran first and wrote it).
// Mirrors the NarrowphaseHeightfield slot-ownership pattern so the three pair-
// driven narrowphases compose over one candidate stream without clobbering.
constexpr uint32_t kNoSdfGrid = ~0u;

__global__ void PairDrivenSdfKernel(const uint32_t* __restrict__ candidate_pairs,
                                    const uint32_t* __restrict__ pair_count,
                                    const float* __restrict__ shape_table,
                                    const float* __restrict__ samp_points,
                                    const uint32_t* __restrict__ samp_ranges,
                                    const float* __restrict__ sdf_headers,
                                    const uint32_t* __restrict__ sdf_cell_count,
                                    const uint64_t* __restrict__ sdf_keys,
                                    const float* __restrict__ sdf_values,
                                    const math::Vec3* __restrict__ sdf_grads,
                                    const math::Transform* __restrict__ body_pose,
                                    uint32_t env_count,
                                    uint32_t bodies_per_env,
                                    uint32_t slot_stride,
                                    uint32_t k,
                                    float margin,
                                    uint32_t* __restrict__ ucount,
                                    math::Vec3* __restrict__ upoint,
                                    math::Vec3* __restrict__ unormal,
                                    float* __restrict__ udepth,
                                    uint32_t* __restrict__ ucontact_a,
                                    uint32_t* __restrict__ ucontact_b,
                                    uint32_t* __restrict__ ucontact_gen,
                                    uint32_t* __restrict__ contact_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_stride;
    if (gid >= total) return;
    const uint32_t env = gid / slot_stride;
    const uint32_t slot = gid - env * slot_stride;
    if (slot >= pair_count[env]) return;  // not a live candidate.

    // Own ONLY slots the analytic/cvx (NarrowphasePrimitives) + heightfield ops
    // left EMPTY: a hull body may ALSO carry an SDF grid (the cook binds sdf_grid
    // on hull bodies), and that pair is already resolved by the cvx hull path.
    // Filling only empty slots covers exactly the "non-cvx side -> empty; the SDF
    // path covers it" gap those ops promise, with no clobber of a valid manifold.
    if (ucount[gid] != 0u) return;

    const uint32_t a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
    const uint32_t b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
    // shape_table lane 7 == sdf_grid (kNoSdfGrid when the shape has no cooked SDF).
    const uint32_t ga = LoadShapeSdfGrid(shape_table, a);
    const uint32_t gb = LoadShapeSdfGrid(shape_table, b);
    const bool a_sdf = (ga != kNoSdfGrid);
    const bool b_sdf = (gb != kNoSdfGrid);
    if (a_sdf == b_sdf) return;  // both or neither -> not an SDF pair; leave it.

    const uint32_t sample_body = a_sdf ? b : a;   // the OTHER shape samples.
    const uint32_t target_body = a_sdf ? a : b;   // the SDF shape is the target.
    const uint32_t grid = a_sdf ? ga : gb;

    const math::Transform sxf = body_pose[env * bodies_per_env + sample_body];
    const math::Transform txf = body_pose[env * bodies_per_env + target_body];
    const uint32_t soff = samp_ranges[sample_body * 2u + 0u];
    const uint32_t scnt = samp_ranges[sample_body * 2u + 1u];
    const sdf::SparseSdfDevice sdf_grid = LoadSdfGrid(
        sdf_headers, sdf_cell_count, sdf_keys, sdf_values, sdf_grads, grid);

    const uint32_t kk = (k > 4u) ? 4u : k;
    math::Vec3 pt[4]; math::Vec3 nm[4]; float dp[4];
    uint32_t kept = 0u;
    for (uint32_t s = 0u; s < scnt; ++s) {
        const math::Vec3 local{samp_points[(soff + s) * 3u + 0u],
                               samp_points[(soff + s) * 3u + 1u],
                               samp_points[(soff + s) * 3u + 2u]};
        const math::Vec3 world = SdfTransformPoint(sxf, local);
        const math::Vec3 q = SdfInverseTransformPoint(txf, world);
        math::Vec3 grad{0, 0, 0};
        const float phi = sdf::sparse_sdf_sample(sdf_grid, q, grad);
        if (phi >= sdf::SparseSdfDevice::kOutsideBand) continue;
        const float depth = -phi + margin;
        if (depth <= 0.0f) continue;
        const math::Vec3 gw = SdfRotate(txf.rotation, grad);
        const float gl = sqrtf(gw.x * gw.x + gw.y * gw.y + gw.z * gw.z);
        const math::Vec3 n = (gl > 1.0e-12f)
            ? math::Vec3{gw.x / gl, gw.y / gl, gw.z / gl}
            : math::Vec3{0.0f, 0.0f, 1.0f};
        const math::Vec3 cp{world.x - n.x * depth, world.y - n.y * depth,
                            world.z - n.z * depth};
        DeepestKInsert(cp, n, depth, s, kk, pt, nm, dp, &kept);
    }

    ucount[gid] = kept;
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < kept) {
            upoint[at] = pt[i]; unormal[at] = nm[i]; udepth[at] = dp[i];
            ucontact_a[at] = sample_body;  // side a = the sampling body (sep dir).
            ucontact_b[at] = target_body;  // side b = the SDF body.
            ucontact_gen[at] = 1u;
        } else {
            upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f;
            ucontact_a[at] = 0u; ucontact_b[at] = 0u; ucontact_gen[at] = 0u;
        }
    }
    if (kept > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], kept);
    }
}

Status OpNarrowphaseSdf(const ModelView& model, const DataView& data,
                        const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphaseSdfParams*>(params);
    if (p == nullptr) return Status::Failed;
    // The union slot-template family does its OWN detection (finger x cup is
    // cvx::SphereHull), so the SDF op only runs for the general PairDriven family;
    // the union StepPlanned graph stays bit-identical (this op enqueues nothing).
    if (p->family != kContactFamilyPairDriven) return Status::Ok;
    if (p->env_count == 0u || p->max_contacts_per_env == 0u ||
        p->bodies_per_env == 0u) {
        return Status::Ok;
    }
    // No cooked SDF grids => no SDF pair can exist; skip (the standalone launcher
    // remains the precision-oracle seam). model.sdf_headers null guards an
    // SDF-less model.
    if (model.sdf_headers == nullptr || model.samp_points == nullptr ||
        model.samp_ranges == nullptr || model.shape_table == nullptr ||
        data.candidate_pairs == nullptr || data.body_pose == nullptr) {
        return Status::Ok;
    }
    const uint32_t total = p->env_count * p->max_contacts_per_env;
    constexpr uint32_t kBlock = 128u;
    const uint32_t blocks = (total + kBlock - 1u) / kBlock;
    LaunchCuda(PairDrivenSdfKernel, dim3(blocks), dim3(kBlock), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table),
               static_cast<const float*>(model.samp_points),
               static_cast<const uint32_t*>(model.samp_ranges),
               static_cast<const float*>(model.sdf_headers),
               static_cast<const uint32_t*>(model.sdf_cell_count),
               static_cast<const uint64_t*>(model.sdf_cell_keys),
               static_cast<const float*>(model.sdf_cell_values),
               static_cast<const math::Vec3*>(model.sdf_cell_gradients),
               static_cast<const math::Transform*>(data.body_pose),
               p->env_count, p->bodies_per_env, p->max_contacts_per_env,
               static_cast<uint32_t>(p->max_contacts_per_pair), p->contact_margin,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_gen, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

// Standalone launcher (the precision oracle seam): drive the SDF sampling
// narrowphase over a hand-built pair list + device SAMP/SDF buffers. Returns
// cudaSuccess-mapped Status. (D1: fixed deepest-K order, shared D1 sampler.)
Status LaunchNarrowphaseSdf(const float* samp_points,
                            const uint32_t* samp_ranges,
                            const float* shape_table,
                            const float* sdf_headers,
                            const uint32_t* sdf_cell_count,
                            const uint64_t* sdf_keys,
                            const float* sdf_values,
                            const math::Vec3* sdf_grads,
                            const math::Transform* body_pose,
                            const nkops::SdfPairDev* pairs,
                            uint32_t pair_count,
                            uint32_t k,
                            float margin,
                            uint32_t slot_stride,
                            uint32_t* ucount,
                            math::Vec3* upoint,
                            math::Vec3* unormal,
                            float* udepth,
                            uint32_t* contact_count,
                            cudaStream_t stream) {
    if (pair_count == 0u) return Status::Ok;
    constexpr uint32_t kBlock = 128u;
    const uint32_t blocks = (pair_count + kBlock - 1u) / kBlock;
    LaunchCuda(NarrowphaseSdfKernel, dim3(blocks), dim3(kBlock), 0u, stream,
               samp_points, samp_ranges, shape_table, sdf_headers, sdf_cell_count,
               sdf_keys, sdf_values, sdf_grads, body_pose, pairs, pair_count,
               (k > 4u ? 4u : k), margin, slot_stride, ucount, upoint, unormal,
               udepth, contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

void RegisterNkNarrowphaseSdfOps() {
    SetCudaOp(NkOp::NarrowphaseSdf, &OpNarrowphaseSdf);
}

}  // namespace nuka::phi
