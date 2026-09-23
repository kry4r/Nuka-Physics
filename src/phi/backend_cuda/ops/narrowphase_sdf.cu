// Surface samples query analytic, triangle or sparse SDF geometry and emit shared manifolds.
// Bounds cull candidates; only the declared collision surface supplies contacts.

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cfloat>
#include <limits>

#include "collision/primitive_surface.hpp"
#include "constraint/contact_manifold.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/solve/nk_row.hpp"                  // kUContactSideBody (side-kind tag)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // kShapeTableRowStride (shared)
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/sdf_types.cuh"
#include "phi/backend_cuda/ops/surface_query.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace nkops {

namespace sdf = ::nuka::runtime::sdf;

// Load one SDF grid view from the Model sdf_* device tables (kSdfHeaderStride f32
// header + flat cell arrays). Mirrors SparseSdfDevice; cell_* point into the buffer.
__device__ sdf::SparseSdfDevice LoadSdfGrid(const float* headers,
                                            const uint32_t* cell_counts,
                                            const uint64_t* keys,
                                            const float* values,
                                            const math::Vec3* grads,
                                            uint32_t grid) {
    const float* h = headers + static_cast<size_t>(grid) * ::nuka::phi::kSdfHeaderStride;
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

// Keep descending depth; equal-depth samples retain their insertion order.
__device__ void DeepestKInsert(math::Vec3 point, math::Vec3 normal, float depth,
                               uint32_t id_a, uint32_t id_b, uint32_t k,
                               math::Vec3* pt, math::Vec3* nm, float* dp,
                               uint32_t* feature_a, uint32_t* feature_b, uint32_t* count) {
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
        feature_a[i] = feature_a[i - 1u];
        feature_b[i] = feature_b[i - 1u];
    }
    pt[pos] = point; nm[pos] = normal; dp[pos] = depth;
    feature_a[pos] = id_a;
    feature_b[pos] = id_b;
    *count = (n < k) ? (n + 1u) : k;
}

}  // namespace nkops

namespace {

using namespace ::nuka::phi::nkops;
namespace sdf = ::nuka::runtime::sdf;

constexpr uint32_t kManifoldPoints = constraint::ContactManifold::kMaxPoints;
constexpr uint32_t kSurfaceQueryThreads = 128u;

struct SampleContact {
    math::Vec3 point;
    math::Vec3 normal;
    float depth;
    uint32_t feature_a;
    uint32_t feature_b;
    uint64_t sequence;
};

struct SampleRank {
    float score;
    uint32_t lane;
    uint64_t sequence;
};

struct BetterSample {
    __device__ SampleRank operator()(const SampleRank& a, const SampleRank& b) const {
        if (a.score != b.score) return a.score > b.score ? a : b;
        return a.sequence <= b.sequence ? a : b;
    }
};

__device__ float SampleSeparation(const SampleContact& a, const SampleContact& b) {
    const math::Vec3 distance = a.point - b.point;
    if (a.sequence == b.sequence || (distance.LengthSq() == 0.0f &&
        (a.normal - b.normal).LengthSq() == 0.0f)) return -1.0f;
    return distance.LengthSq();
}

// Keep the deepest contact and spread the remaining points over the surface patch.
// Duplicate positions with the same normal never consume a manifold slot.
__device__ void InsertSampleContact(const SampleContact& sample, uint32_t capacity,
                                    SampleContact* contacts, uint32_t& count) {
    if (capacity == 0u) return;
    for (uint32_t i = 0u; i < count; ++i) {
        if (SampleSeparation(sample, contacts[i]) >= 0.0f) continue;
        if (sample.depth > contacts[i].depth) contacts[i] = sample;
        return;
    }
    if (count < capacity) { contacts[count++] = sample; return; }
    uint32_t selected[kManifoldPoints];
    uint32_t selected_mask = 0u;
    for (uint32_t kept = 0u; kept < capacity; ++kept) {
        SampleRank best{-1.0f, 0u, ~uint64_t{0u}};
        for (uint32_t i = 0u; i <= count; ++i) {
            if ((selected_mask & (1u << i)) != 0u) continue;
            const auto& candidate = i == count ? sample : contacts[i];
            float score = kept == 0u ? candidate.depth : FLT_MAX;
            for (uint32_t j = 0u; j < kept; ++j) {
                const auto& prior = selected[j] == count ? sample : contacts[selected[j]];
                score = fminf(score, SampleSeparation(candidate, prior));
            }
            best = BetterSample{}(best, {score, i, candidate.sequence});
        }
        selected[kept] = best.lane;
        selected_mask |= 1u << best.lane;
    }
    for (uint32_t i = 0u; i < count; ++i)
        if ((selected_mask & (1u << i)) == 0u) { contacts[i] = sample; return; }
}

// Each pair samples a local surface into an SDF, with normals separating A.
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

    math::Vec3 pt[kManifoldPoints]; math::Vec3 nm[kManifoldPoints]; float dp[kManifoldPoints];
    uint32_t feature_a[kManifoldPoints], feature_b[kManifoldPoints];
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
        // The world SDF gradient separates the sampling shape from the target.
        const math::Vec3 gw = SdfRotate(txf.rotation, grad);
        const float gl = sqrtf(gw.x * gw.x + gw.y * gw.y + gw.z * gw.z);
        if (!isfinite(phi) || !isfinite(gl) || gl < 1.0e-12f) continue;
        const math::Vec3 n = gw / gl;
        // Contact point: the sample, pushed back onto the surface along n.
        const math::Vec3 cp = world - n * phi;
        DeepestKInsert(cp, n, depth, s, nk::kContactFeatureUnavailable,
                       k, pt, nm, dp, feature_a, feature_b, &kept);
    }

    ucount[out_gid] = kept;
    for (uint32_t i = 0u; i < kManifoldPoints; ++i) {
        const size_t at = static_cast<size_t>(out_gid) * kManifoldPoints + i;
        if (i < kept) { upoint[at] = pt[i]; unormal[at] = nm[i]; udepth[at] = dp[i]; }
        else { upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f; }
    }
    if (kept > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], kept);
    }
}

// Query both surfaces into the shared manifold when analytic/convex detection leaves it empty.
__global__ void PairDrivenSdfKernel(const uint32_t* __restrict__ candidate_pairs,
                                    const uint32_t* __restrict__ pair_count,
                                    const float* __restrict__ shape_table,
                                    const float* __restrict__ samp_points,
                                    const uint32_t* __restrict__ samp_ranges,
                                    SurfaceQueryView surfaces,
                                    const math::Transform* __restrict__ body_pose,
                                    uint32_t env_count,
                                    uint32_t bodies_per_env,
                                    uint32_t slot_stride,
                                    uint32_t rigid_slot_cap,
                                    uint32_t pair_slots,
                                    uint32_t sample_point_count,
                                    uint32_t k,
                                    float margin,
                                    uint32_t* __restrict__ ucount,
                                    math::Vec3* __restrict__ upoint,
                                    math::Vec3* __restrict__ unormal,
                                    float* __restrict__ udepth,
                                    uint32_t* __restrict__ ucontact_a,
                                    uint32_t* __restrict__ ucontact_b,
                                    uint32_t* __restrict__ ucontact_a_kind,
                                    uint32_t* __restrict__ ucontact_b_kind,
                                    uint32_t* __restrict__ ucontact_gen,
                                    uint64_t* __restrict__ ucontact_id_pair,
                                    uint64_t* __restrict__ ucontact_id_feature,
                                    uint32_t* __restrict__ contact_count,
                                    uint32_t* __restrict__ env_status) {
    const uint32_t env = blockIdx.x / pair_slots;
    const uint32_t slot = blockIdx.x - env * pair_slots;
    if (env >= env_count) return;
    const uint32_t gid = env * slot_stride + slot;
    // Body<->body candidates fill [0, rigid_slot_cap); slots above belong to the
    // body<->particle narrowphase (== slot_stride when no particles -> identical).
    if (slot >= pair_count[env] || slot >= rigid_slot_cap) return;

    // Analytic, convex and heightfield detection retain any manifold they emitted.
    if (ucount[gid] != 0u) return;

    const uint32_t a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
    const uint32_t b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
    if (a >= bodies_per_env || b >= bodies_per_env) {
        if (env_status && threadIdx.x == 0u) atomicOr(&env_status[env], kEnvStatusInvalidEndpoint);
        return;
    }
    const PrimShapeDev shapes[2] = {LoadPrimShape(shape_table, a), LoadPrimShape(shape_table, b)};
    const math::Transform poses[2] = {body_pose[env * bodies_per_env + a],
                                      body_pose[env * bodies_per_env + b]};
    const uint32_t bodies[2] = {a, b};
    const uint32_t kk = k > kManifoldPoints ? kManifoldPoints : k;
    SampleContact local_contacts[kManifoldPoints];
    uint32_t local_count = 0u;
    bool queried = false;
    uint32_t geometry_status = 0u;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t other = 1u - side;
        const PrimShapeDev& target = shapes[other];
        if (!HasCollidableSurface(surfaces, bodies[other], target)) continue;
        const math::Transform& sxf = poses[side];
        const math::Transform& txf = poses[other];
        if (shapes[side].kind == collision::kShapeSphere) {
            queried = true;
            if (threadIdx.x != 0u) continue;
            const float radius = shapes[side].params[0];
            const auto surface = QueryCollidableSurface(surfaces, bodies[other], target,
                SdfInverseTransformPoint(txf, sxf.position), radius + margin);
            if (!surface.valid) {
                geometry_status |= kEnvStatusContactGeometryUnavailable;
                continue;
            }
            const float depth = radius + margin - surface.distance;
            if (depth > 0.0f) {
                const auto normal = SdfRotate(txf.rotation, surface.normal);
                const uint32_t feature = surface.triangle != ~0u ? surface.triangle : surface.feature;
                InsertSampleContact({SdfTransformPoint(txf, surface.point),
                    side == 0u ? normal : normal * -1.0f, depth,
                    side == 0u ? 0u : feature, side == 0u ? feature : 0u,
                    uint64_t(side) * (uint64_t(sample_point_count) + 1u)},
                    kk, local_contacts, local_count);
            }
            continue;
        }
        if (target.kind == collision::kShapeSphere &&
            HasCollidableSurface(surfaces, bodies[side], shapes[side])) continue;
        const uint32_t soff = samp_ranges[bodies[side] * 2u];
        const uint32_t scnt = samp_ranges[bodies[side] * 2u + 1u];
        if (soff > sample_point_count || scnt > sample_point_count - soff) {
            geometry_status |= kEnvStatusContactGeometryUnavailable;
            continue;
        }
        if (scnt == 0u) continue;
        queried = true;
        for (uint64_t sample = threadIdx.x; sample < scnt; sample += blockDim.x) {
            const auto s = static_cast<uint32_t>(sample);
            const size_t at = static_cast<size_t>(soff + s) * 3u;
            const math::Vec3 local{samp_points[at], samp_points[at + 1u], samp_points[at + 2u]};
            const math::Vec3 world = SdfTransformPoint(sxf, local);
            const math::Vec3 q = SdfInverseTransformPoint(txf, world);
            const auto surface = QueryCollidableSurface(surfaces, bodies[other], target, q, margin);
            if (!surface.valid) {
                geometry_status |= kEnvStatusContactGeometryUnavailable;
                continue;
            }
            const float phi = surface.distance;
            if (phi >= sdf::SparseSdfDevice::kOutsideBand) continue;
            const float depth = -phi + margin;
            if (depth <= 0.0f) continue;
            const math::Vec3 gw = SdfRotate(txf.rotation, surface.normal);
            const float gl = sqrtf(gw.Dot(gw));
            if (!isfinite(phi) || !isfinite(gl) || gl < 1.0e-12f) {
                geometry_status |= kEnvStatusContactGeometryUnavailable;
                continue;
            }
            const math::Vec3 n = gw / gl;
            const math::Vec3 cp = SdfTransformPoint(txf, surface.point);
            const uint32_t target_feature = surface.triangle != ~0u ? surface.triangle : surface.feature;
            InsertSampleContact({cp, side == 0u ? n : n * -1.0f, depth,
                side == 0u ? s : target_feature, side == 0u ? target_feature : s,
                uint64_t(side) * (uint64_t(sample_point_count) + 1u) + s},
                kk, local_contacts, local_count);
        }
    }
    if (!queried && (shapes[0].kind == collision::kShapeSdfMesh ||
                    shapes[1].kind == collision::kShapeSdfMesh))
        geometry_status |= kEnvStatusContactGeometryUnavailable;
    const bool unavailable = __syncthreads_or(geometry_status != 0u) != 0;
    if (unavailable && env_status && threadIdx.x == 0u)
        atomicOr(&env_status[env], kEnvStatusContactGeometryUnavailable);

    using Reduction = cub::BlockReduce<SampleRank, kSurfaceQueryThreads>;
    __shared__ typename Reduction::TempStorage reduction;
    __shared__ uint32_t winner;
    __shared__ SampleContact selected[kManifoldPoints];
    uint32_t kept = 0u;
    for (uint32_t i = 0u; i < kk; ++i) {
        SampleRank rank{-1.0f, threadIdx.x, ~uint64_t{0u}};
        uint32_t local_best = 0u;
        for (uint32_t candidate = 0u; candidate < local_count; ++candidate) {
            const auto& contact = local_contacts[candidate];
            float score = i == 0u ? contact.depth : FLT_MAX;
            for (uint32_t previous = 0u; previous < i; ++previous)
                score = fminf(score, SampleSeparation(contact, selected[previous]));
            if (score > rank.score || (score == rank.score && contact.sequence < rank.sequence)) {
                rank = {score, threadIdx.x, contact.sequence};
                local_best = candidate;
            }
        }
        const auto best = Reduction(reduction).Reduce(rank, BetterSample{});
        if (threadIdx.x == 0u) winner = best.score >= 0.0f ? best.lane : ~0u;
        __syncthreads();
        if (winner == ~0u) break;
        if (threadIdx.x == winner) {
            const auto& contact = local_contacts[local_best];
            selected[i] = contact;
            const size_t at = static_cast<size_t>(gid) * kManifoldPoints + i;
            upoint[at] = contact.point;
            unormal[at] = contact.normal;
            udepth[at] = contact.depth;
            ucontact_a[at] = a;
            ucontact_b[at] = b;
            ucontact_a_kind[at] = nk::kUContactSideBody;
            ucontact_b_kind[at] = nk::kUContactSideBody;
            ucontact_gen[at] = 1u;
            nk::CanonicalContactDescriptor descriptor;
            descriptor.a.handle = a;
            descriptor.b.handle = b;
            descriptor.local_point_a = PrimInverseTransformPoint(poses[0], contact.point);
            descriptor.local_point_b = PrimInverseTransformPoint(poses[1], contact.point);
            descriptor.normal = contact.normal;
            descriptor.feature_a = contact.feature_a;
            descriptor.feature_b = contact.feature_b;
            descriptor.manifold_slot = i;
            const nk::ContactId id = nk::MakeContactId(descriptor);
            ucontact_id_pair[at] = id.pair;
            ucontact_id_feature[at] = id.feature;
        }
        ++kept;
        __syncthreads();
    }
    if (threadIdx.x == 0u) ucount[gid] = kept;
    for (uint32_t i = threadIdx.x; i < kManifoldPoints; i += blockDim.x) {
        const size_t at = static_cast<size_t>(gid) * kManifoldPoints + i;
        if (i < kept) continue;
        upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f;
        ucontact_a[at] = 0u; ucontact_b[at] = 0u; ucontact_gen[at] = 0u;
        ucontact_a_kind[at] = nk::kUContactSideBody;
        ucontact_b_kind[at] = nk::kUContactSideBody;
        ucontact_id_pair[at] = 0u;
        ucontact_id_feature[at] = 0u;
    }
    if (kept > 0u && contact_count != nullptr && threadIdx.x == 0u) {
        atomicAdd(&contact_count[env], kept);
    }
}

Status OpNarrowphaseSdf(const ModelView& model, const DataView& data,
                        const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphaseSdfParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;
    if (p->env_count == 0u || p->max_contacts_per_env == 0u ||
        p->bodies_per_env == 0u) {
        return Status::Ok;
    }
    const auto surfaces = MakeSurfaceQueryView(model, p->bodies_per_env, p->mesh_geometry,
                                               p->sdf_grid_count, p->sdf_cell_total);
    if (!model.samp_ranges || !model.shape_table || !data.candidate_pairs || !data.body_pose ||
        (p->sample_point_count > 0u && !model.samp_points) || !SurfaceQueryStorageValid(surfaces))
        return Status::InvalidArgument;
    const uint32_t pair_slots = std::min(p->max_contacts_per_env, p->rigid_slot_cap);
    if (pair_slots == 0u) return Status::Ok;
    const uint64_t total = uint64_t(p->env_count) * p->max_contacts_per_env;
    if (total > std::numeric_limits<int>::max() || !data.pair_count ||
        !data.ucontact_count || !data.ucontact_point || !data.ucontact_normal ||
        !data.ucontact_depth || !data.ucontact_a || !data.ucontact_b ||
        !data.ucontact_a_kind || !data.ucontact_b_kind || !data.ucontact_gen ||
        !data.ucontact_id_pair || !data.ucontact_id_feature) return Status::InvalidArgument;
    const uint32_t blocks = p->env_count * pair_slots;
    LaunchCuda(PairDrivenSdfKernel, dim3(blocks), dim3(kSurfaceQueryThreads), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table),
               static_cast<const float*>(model.samp_points),
               static_cast<const uint32_t*>(model.samp_ranges),
               surfaces,
               static_cast<const math::Transform*>(data.body_pose),
               p->env_count, p->bodies_per_env, p->max_contacts_per_env,
               p->rigid_slot_cap, pair_slots, p->sample_point_count,
               static_cast<uint32_t>(p->max_contacts_per_pair), p->contact_margin,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_a_kind, data.ucontact_b_kind,
               data.ucontact_gen, data.ucontact_id_pair,
               data.ucontact_id_feature, data.contact_count, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

// The geometry oracle drives the same signed-distance contract with explicit pairs.
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
               (k > kManifoldPoints ? kManifoldPoints : k), margin, slot_stride, ucount, upoint, unormal,
               udepth, contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

void RegisterNkNarrowphaseSdfOps() {
    SetCudaOp(NkOp::NarrowphaseSdf, &OpNarrowphaseSdf);
}

}  // namespace nuka::phi
