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
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/sdf_types.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace nkops {

namespace sdf = ::nuka::runtime::sdf;

// Load one SDF grid view from the Model sdf_* device tables (8 f32 header +
// flat cell arrays). Mirrors SparseSdfDevice; cell_* point into the model buffer.
__device__ sdf::SparseSdfDevice LoadSdfGrid(const float* headers,
                                            const uint32_t* cell_counts,
                                            const uint64_t* keys,
                                            const float* values,
                                            const math::Vec3* grads,
                                            uint32_t grid) {
    const float* h = headers + static_cast<size_t>(grid) * 8u;
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

Status OpNarrowphaseSdf(const ModelView& /*model*/, const DataView& /*data*/,
                        const void* params, cudaStream_t /*stream*/) {
    const auto* p = static_cast<const NarrowphaseSdfParams*>(params);
    if (p == nullptr) return Status::Failed;
    // M5: the union slot-template and fused-foot families do NOT use the SDF
    // path (the union finger x cup is cvx::SphereHull). The pair-driven SDF
    // narrowphase is driven through the standalone launcher (the precision
    // oracle); the in-pipeline pair-stream wiring (building SdfPairDev from
    // candidate_pairs + shape_table) lands with the pair-driven scene cook —
    // for now the op is a deterministic no-op for every registered family so
    // the gate paths (union StepPlanned) stay bit-identical to M4.
    if (p->family != kContactFamilyPairDriven) return Status::Ok;
    return Status::Ok;
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
