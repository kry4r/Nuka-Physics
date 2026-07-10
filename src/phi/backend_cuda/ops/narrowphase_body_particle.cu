// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — body/artic <-> soft/fluid particle narrowphase.
//
// THE ONE PRINCIPLE: a body<->particle contact is the SAME contact problem as a
// body<->body contact. The particle is a SPHERE of its radius on the ONE general
// path, so its manifold rides the SAME ucontact_* buffer the rigid narrowphase
// fills, expands into the SAME NkRow block by AssembleRows, and solves in the SAME
// block-island PGS. No per-medium branch, no special coupler.
//
// FLOW (one thread per (env x env-local particle)):
// 1. Build the particle's query AABB (sphere of particle_radius) at its position.
// 2. Traverse the env's arena LBVH (data.lbvh_nodes, the env*(2N-1) node slice the
// broadphase built) for overlapping collidable bodies, collected into a private
// insertion-sorted candidate list capped at kCrossSystemMaxCandidates (the
// cross_system_query CSR pattern). For a <2-collidable env (no LBVH) it scans
// the body AABBs directly. A particle exceeding the cap ORs kEnvStatusPairOverflow.
// 3. For each candidate body run the sphere-vs-shape manifold: amf::Sphere{Sphere,
// Box,Plane} inline, cvx::SphereHull (the EPA-bypass closest-point query) for a
// convex hull, a face-only handler for the heightfield (the SphereHull/EPA
// shallow-penetration dead band is avoided exactly as the heightfield path does).
// 4. Write the manifold into the particle's RESERVED contact-slot sub-range
// [slot_base + pi*cands_per_particle + k]. The per-particle base is a FIXED
// (non-atomic) function of the particle index, so the body<->particle slot stream
// is bit-D1 by construction AND occupies a deterministic sub-range relative to the
// racy rigid-rigid slots [0, pair_count) — the cross-stream ordering guard. Side A
// == the particle (GLOBAL id + the kUContactSideParticle index-kind tag), side B
// == the body collidable (kUContactSideBody); the manifold normal is the
// separation dir for the particle (push it off the body).
//
// FAMILY GATING (D1): EARLY-EXITS unless family == kContactFamilyPairDriven.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cfloat>

#include "collision/aabb.hpp"                   // collision::AABB
#include "collision/analytical_manifold.hpp"   // amf:: sphere handlers (HD)
#include "collision/convex_narrowphase.hpp"    // cvx::SphereHull (EPA-bypass)
#include "collision/cross_system_query.hpp"    // kCrossSystemMaxCandidates (the cap)
#include "collision/lbvh_node.cuh"             // LbvhNode (arena tree traversal)
#include "collision/shape_kind.hpp"            // nuka::collision::ShapeKind
#include "math/transform.hpp"
#include "nk/model/generated/views.hpp"        // ModelView / DataView
#include "nk/solve/nk_row.hpp"                  // kUContactSide* (side-kind tags)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // PrimShapeDev / LoadPrimShape / PrimRotate
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/sdf_types.cuh"  // SdfRotate / SdfInverseTransformPoint
#include "phi/op_schema.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"     // sparse_sdf_sample (shared D1 query)

namespace nuka::phi {

namespace {

namespace amf = ::nuka::collision::amf;
namespace cvx = ::nuka::collision::cvx;
namespace cg  = ::nuka::collision::gpu;
using ::nuka::constraint::ContactManifold;
using ::nuka::math::Vec3;
using ::nuka::phi::nkops::PrimShapeDev;
using ::nuka::phi::nkops::LoadPrimShape;
using ::nuka::phi::nkops::PrimRotate;

constexpr uint32_t kBlockSize = 128u;
constexpr uint32_t kKindSphere      = ::nuka::collision::kShapeSphere;
constexpr uint32_t kKindCapsule     = ::nuka::collision::kShapeCapsule;
constexpr uint32_t kKindBox         = ::nuka::collision::kShapeBox;
constexpr uint32_t kKindPlane       = ::nuka::collision::kShapePlane;
constexpr uint32_t kKindConvexHull  = ::nuka::collision::kShapeConvexHull;
constexpr uint32_t kKindSdfMesh     = ::nuka::collision::kShapeSdfMesh;
constexpr uint32_t kKindHeightfield = ::nuka::collision::kShapeHeightfield;
namespace sdfq = ::nuka::runtime::sdf;

// Load one SDF grid view from the Model sdf_* device tables (mirrors
// narrowphase_sdf.cu LoadSdfGrid + mpm.cu LoadGrid: kSdfHeaderStride f32 header +
// flat cell arrays). cell_* point into the shared concatenated buffers.
__device__ __forceinline__ sdfq::SparseSdfDevice LoadParticleSdfGrid(
    const float* headers, const uint32_t* counts, const uint64_t* keys,
    const float* values, const Vec3* grads, uint32_t grid) {
    const float* h = headers + static_cast<size_t>(grid) * kSdfHeaderStride;
    sdfq::SparseSdfDevice s;
    s.origin = {h[0], h[1], h[2]};
    s.voxel_size = h[3];
    s.dims[0] = __float_as_uint(h[4]);
    s.dims[1] = __float_as_uint(h[5]);
    s.dims[2] = __float_as_uint(h[6]);
    const uint32_t off = __float_as_uint(h[7]);
    s.cell_keys = keys + off;
    s.cell_values = values + off;
    s.cell_gradients = grads + off;
    s.cell_count = counts[grid];
    return s;
}

// Per-particle candidate cap — the cross_system_query memory bound (the LOWEST
// collidable indices; a deterministic subset). A particle over the cap surfaces
// kEnvStatusPairOverflow (never a silent drop).
constexpr uint32_t kMaxCandidates = cg::kCrossSystemMaxCandidates;

// Build the particle's world AABB (sphere of `radius`).
__device__ __forceinline__ collision::AABB ParticleAabb(Vec3 p, float radius) {
    collision::AABB box;
    box.min = Vec3{p.x - radius, p.y - radius, p.z - radius};
    box.max = Vec3{p.x + radius, p.y + radius, p.z + radius};
    return box;
}

__device__ __forceinline__ bool AabbOverlaps(const collision::AABB& a,
                                             const collision::AABB& b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

// Insertion-sorted, capped collect of an overlapping collidable into a private
// slice (the cross_system_query.cu TraverseRigidLbvh insert: ascending by index,
// keep the lowest on overflow). Returns the new count.
__device__ __forceinline__ uint32_t InsertCandidate(uint32_t* out, uint32_t count,
                                                    uint32_t body, bool* overflow) {
    if (count < kMaxCandidates) {
        uint32_t pos = count;
        while (pos > 0u && out[pos - 1u] > body) { out[pos] = out[pos - 1u]; --pos; }
        out[pos] = body;
        return count + 1u;
    }
    *overflow = true;
    if (body < out[count - 1u]) {
        uint32_t pos = count - 1u;
        while (pos > 0u && out[pos - 1u] > body) { out[pos] = out[pos - 1u]; --pos; }
        out[pos] = body;
    }
    return count;
}

// Gather the env's collidables overlapping `query` into `out` (env-LOCAL body
// rows). Traverses the env's arena LBVH slice when it exists (N >= 2 leaves built
// the env*(2N-1) tree); else scans the env's body AABBs directly. Both are
// deterministic; the LBVH path mirrors EnvQueryPairsKernel's traversal + the
// cross_system_query insertion sort.
__device__ uint32_t GatherOverlaps(const collision::AABB& query,
                                   const cg::LbvhNode* nodes,
                                   const Vec3* body_aabb_lo, const Vec3* body_aabb_hi,
                                   uint32_t env, uint32_t N,
                                   uint32_t* out, bool* overflow) {
    uint32_t count = 0u;
    if (N >= 2u && nodes != nullptr) {
        const uint32_t internal = N - 1u;
        const uint32_t nbase = env * (2u * N - 1u);
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
                const collision::AABB cbox = nodes[nbase + child].aabb;
                if (!AabbOverlaps(query, cbox)) continue;
                if (is_leaf) {
                    count = InsertCandidate(
                        out, count,
                        static_cast<uint32_t>(nodes[nbase + child].left), overflow);
                } else if (top < 63) {
                    stack[top++] = child;
                } else {
                    *overflow = true;  // stack full: subtree dropped (surfaced).
                }
            }
        }
        return count;
    }
    // <2-collidable env (no LBVH built): direct scan over the env's body AABBs.
    const size_t base = static_cast<size_t>(env) * N;
    for (uint32_t b = 0u; b < N; ++b) {
        collision::AABB box; box.min = body_aabb_lo[base + b]; box.max = body_aabb_hi[base + b];
        if (AabbOverlaps(query, box)) count = InsertCandidate(out, count, b, overflow);
    }
    return count;
}

// Build the body side's world PrimParams (mirrors the prim/heightfield narrowphase
// MakePrim).
__device__ amf::PrimParams MakePrim(const PrimShapeDev& s, const math::Transform& xf) {
    amf::PrimParams p;
    p.frame.cx = PrimRotate(xf.rotation, Vec3{1, 0, 0});
    p.frame.cy = PrimRotate(xf.rotation, Vec3{0, 1, 0});
    p.frame.cz = PrimRotate(xf.rotation, Vec3{0, 0, 1});
    p.frame.t = xf.position;
    p.radius = s.params[0];
    p.half_height = s.params[1];
    p.half_extents = {s.params[0], s.params[1], s.params[2]};
    return p;
}

// A particle-as-sphere PrimParams at a world position (identity frame; a sphere is
// rotation-free).
__device__ amf::PrimParams MakeParticleSphere(Vec3 pos, float radius) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1, 0, 0};
    p.frame.cy = Vec3{0, 1, 0};
    p.frame.cz = Vec3{0, 0, 1};
    p.frame.t = pos;
    p.radius = radius;
    p.half_height = 0.0f;
    p.half_extents = {radius, radius, radius};
    return p;
}

// Sphere (the particle, side A) vs a heightfield cell, FACE-ONLY — the cvx EPA
// shallow-penetration dead band on the prism's internal triangulation edge yields a
// spurious near-horizontal normal, so the particle takes the face contact instead
// (the same robustness the heightfield convex path applies to a sphere foot). The
// normal is oriented OUT of the downward-solid prism so a deep particle is pushed up,
// never sideways; depth grows past the dead band. Separation dir for the sphere(=A).
__device__ bool SphereTriangleFace(Vec3 center, float radius, Vec3 a, Vec3 b, Vec3 c,
                                   Vec3 solid_out, Vec3* out_pos, Vec3* out_nrm,
                                   float* out_pen) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    Vec3 n{ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z,
           ab.x * ac.y - ab.y * ac.x};
    const float nl2 = n.Dot(n);
    if (nl2 < 1.0e-20f) return false;
    n = n * (1.0f / sqrtf(nl2));
    if (n.Dot(solid_out) < 0.0f) n = n * -1.0f;
    const float s = (center - a).Dot(n);
    if (s > radius) return false;
    const Vec3 proj = center - n * s;
    const Vec3 v2 = proj - a;
    const float d00 = ab.Dot(ab), d01 = ab.Dot(ac), d11 = ac.Dot(ac);
    const float d20 = v2.Dot(ab), d21 = v2.Dot(ac);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1.0e-20f) return false;
    const float inv = 1.0f / denom;
    const float vv = (d11 * d20 - d01 * d21) * inv;
    const float ww = (d00 * d21 - d01 * d20) * inv;
    const float uu = 1.0f - vv - ww;
    constexpr float kBaryEps = 1.0e-4f;
    if (uu < -kBaryEps || vv < -kBaryEps || ww < -kBaryEps) return false;
    *out_pos = proj;
    *out_nrm = n;
    *out_pen = radius - s;
    return true;
}

// Read a grid corner's LOCAL z from the cooked normalized height grid (mirrors the
// rigid heightfield narrowphase CornerZ). heights stores h in [0,1]; row-major.
__device__ __forceinline__ float CornerZ(const float* heights, uint32_t data_offset,
                                         uint32_t ncol, float min_z, float range,
                                         uint32_t r, uint32_t c) {
    const float h = heights[data_offset + r * ncol + c];
    return min_z + h * range;
}

// Particle (sphere) vs the heightfield collidable: project the particle's world AABB
// into the heightfield-LOCAL frame, clamp to the cell range, and test each overlapped
// cell triangle via the deep-sink-robust face handler oriented out of the downward-
// solid prism (+cz). Keeps the DEEPEST face contact, exactly like the rigid path's
// SphereHeightfieldTri. The descriptor (cell/origin/data_offset) rides the params.
__device__ void ParticleHeightfield(Vec3 center, float radius, float margin,
                                    const math::Transform& xhf,
                                    const NarrowphaseBodyParticleParams& pp,
                                    const float* heights, ContactManifold* m) {
    // Heightfield world frame (its body pose); the grid is in this LOCAL frame.
    amf::PrimFrame hf;
    hf.cx = PrimRotate(xhf.rotation, Vec3{1, 0, 0});
    hf.cy = PrimRotate(xhf.rotation, Vec3{0, 1, 0});
    hf.cz = PrimRotate(xhf.rotation, Vec3{0, 0, 1});
    hf.t = xhf.position;

    // World AABB (center +- radius + margin) -> heightfield-local AABB (transform all
    // 8 corners, take the local min/max), then map to the overlapped cell range.
    const float he = radius + margin;
    Vec3 lo{3.4e38f, 3.4e38f, 3.4e38f};
    Vec3 hi{-3.4e38f, -3.4e38f, -3.4e38f};
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2) {
                const Vec3 wc{center.x + sx * he, center.y + sy * he, center.z + sz * he};
                const Vec3 lc = hf.WorldToLocal(wc);
                lo.x = fminf(lo.x, lc.x); lo.y = fminf(lo.y, lc.y); lo.z = fminf(lo.z, lc.z);
                hi.x = fmaxf(hi.x, lc.x); hi.y = fmaxf(hi.y, lc.y); hi.z = fmaxf(hi.z, lc.z);
            }

    const float origin_x = pp.origin_x, origin_y = pp.origin_y;
    const float cell = pp.cell_size;
    const float min_z = pp.min_z, range = pp.max_z - pp.min_z;
    const uint32_t ncol = pp.ncol, nrow = pp.nrow;

    // cell index = floor((local - origin) / cell); a cell spans corners c..c+1 so the
    // last valid index is n-2. Clamp to [0, n-2] (the rigid path's clamp).
    auto clampi = [](int v, int lo_, int hi_) -> int {
        return v < lo_ ? lo_ : (v > hi_ ? hi_ : v);
    };
    int c_lo = static_cast<int>(floorf((lo.x - origin_x) / cell));
    int c_hi = static_cast<int>(floorf((hi.x - origin_x) / cell));
    int r_lo = static_cast<int>(floorf((lo.y - origin_y) / cell));
    int r_hi = static_cast<int>(floorf((hi.y - origin_y) / cell));
    c_lo = clampi(c_lo, 0, static_cast<int>(ncol) - 2);
    c_hi = clampi(c_hi, 0, static_cast<int>(ncol) - 2);
    r_lo = clampi(r_lo, 0, static_cast<int>(nrow) - 2);
    r_hi = clampi(r_hi, 0, static_cast<int>(nrow) - 2);

    // Walk the overlapped cells; keep the single DEEPEST face contact (the particle
    // gets one contact slot per body — the rest cluster to one surface anyway).
    Vec3 best_pos{0, 0, 0}, best_nrm{0, 0, 0};
    float best_pen = 0.0f;
    bool have = false;
    for (int r = r_lo; r <= r_hi; ++r) {
        for (int c = c_lo; c <= c_hi; ++c) {
            const uint32_t rc = static_cast<uint32_t>(r), cc = static_cast<uint32_t>(c);
            const float x0 = origin_x + static_cast<float>(c) * cell;
            const float x1 = x0 + cell;
            const float y0 = origin_y + static_cast<float>(r) * cell;
            const float y1 = y0 + cell;
            const float z00 = CornerZ(heights, pp.data_offset, ncol, min_z, range, rc, cc);
            const float z10 = CornerZ(heights, pp.data_offset, ncol, min_z, range, rc, cc + 1u);
            const float z01 = CornerZ(heights, pp.data_offset, ncol, min_z, range, rc + 1u, cc);
            const float z11 = CornerZ(heights, pp.data_offset, ncol, min_z, range, rc + 1u, cc + 1u);
            const Vec3 p00 = hf.LocalToWorld(Vec3{x0, y0, z00});
            const Vec3 p10 = hf.LocalToWorld(Vec3{x1, y0, z10});
            const Vec3 p01 = hf.LocalToWorld(Vec3{x0, y1, z01});
            const Vec3 p11 = hf.LocalToWorld(Vec3{x1, y1, z11});
            // The cell's 2 triangles (Newton: tri0=(p00,p10,p11), tri1=(p00,p11,p01)).
            for (int sub = 0; sub < 2; ++sub) {
                Vec3 a, bb, cv;
                if (sub == 0) { a = p00; bb = p10; cv = p11; }
                else          { a = p00; bb = p11; cv = p01; }
                Vec3 cpos, cnrm; float cpen;
                if (SphereTriangleFace(center, radius, a, bb, cv, hf.cz,
                                       &cpos, &cnrm, &cpen) &&
                    (!have || cpen > best_pen)) {
                    best_pos = cpos; best_nrm = cnrm; best_pen = cpen; have = true;
                }
            }
        }
    }
    if (have) {
        ::nuka::constraint::ContactPoint pt;
        pt.position = best_pos;
        pt.normal = best_nrm;      // sep dir for the particle (side A), oriented up.
        pt.penetration = best_pen;
        pt.stable_key = 0ull;
        m->AddPoint(pt);
    }
}

// ONE body<->particle narrowphase, two byte-identical launch shapes selected by the
// cook-time max hull vcount (kWarp):
//   kWarp == true : ONE WARP per particle. The convex-hull branch (SphereHull) scans
//     a giant cooked link hull, so its SupportHull runs WARP-COOPERATIVELY (32 lanes
//     split the vertex pool, exact lowest-index argmax reduce -- bit-identical to the
//     serial scan). All 32 lanes stay converged with identical inputs through every
//     hull call; non-hull work + every store/atomic runs on LANE 0.
//   kWarp == false: ONE THREAD per particle (lane is constexpr 0, no shuffles, serial
//     hull scan) -- for analytic-only collider worlds where no wide hull pays for the
//     31 idle lanes. Identical output: the warp reduce == the serial argmax and every
//     lane-0 gate collapses to "always" when lane == 0.
template <bool kWarp>
__global__ void NarrowphaseBodyParticleKernel(
    const float* __restrict__ shape_table,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ hull_verts,
    const Vec3* __restrict__ particle_pos,
    const Vec3* __restrict__ pbf_predicted_pos,
    const cg::LbvhNode* __restrict__ lbvh_nodes,
    const Vec3* __restrict__ body_aabb_lo,
    const Vec3* __restrict__ body_aabb_hi,
    const float* __restrict__ heights,
    const float* __restrict__ sdf_headers,
    const uint32_t* __restrict__ sdf_cell_count,
    const uint64_t* __restrict__ sdf_keys,
    const float* __restrict__ sdf_values,
    const Vec3* __restrict__ sdf_grads,
    NarrowphaseBodyParticleParams pp,
    uint32_t* __restrict__ ucount,
    Vec3* __restrict__ upoint,
    Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    uint32_t* __restrict__ ucontact_a,
    uint32_t* __restrict__ ucontact_b,
    uint32_t* __restrict__ ucontact_a_kind,
    uint32_t* __restrict__ ucontact_b_kind,
    uint32_t* __restrict__ ucontact_gen,
    uint32_t* __restrict__ contact_count,
    uint32_t* __restrict__ env_status) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = kWarp ? (threadIdx.x & 31u) : 0u;
    const uint32_t gid = kWarp ? (tid >> 5u) : tid;
    const uint32_t total = pp.env_count * pp.particles_per_env;
    if (gid >= total) return;  // uniform per-warp: the whole warp exits together.
    const uint32_t env = gid / pp.particles_per_env;
    const uint32_t pi = gid - env * pp.particles_per_env;  // env-local particle
    const uint32_t N = pp.bodies_per_env;

    // MpmXpbd: the MPM slice [0, particle_row_base) couples via the grid, not rows,
    // and owns NO reserved slots (the cook exempts it from the budget). Uniform per
    // warp (all lanes share pi). 0 for every other mode -> no particle is skipped.
    if (pi < pp.particle_row_base) return;

    // Reserved contact-slot sub-range for THIS row-making particle: a fixed
    // (non-atomic) function of its rank above particle_row_base, so the slot stream
    // is bit-D1 by construction and lands in a deterministic sub-range relative to
    // the rigid slots [0, pair_count). Initialize all reserved slots to inactive.
    const uint32_t slot0 = pp.particle_slot_base +
                           (pi - pp.particle_row_base) * pp.cands_per_particle;
    if (lane == 0u) {
        for (uint32_t k = 0u; k < pp.cands_per_particle; ++k) {
            const uint32_t slot = slot0 + k;
            if (slot >= pp.slot_stride) break;
            const size_t cell = (static_cast<size_t>(env) * pp.slot_stride + slot) * 4u;
            ucount[static_cast<size_t>(env) * pp.slot_stride + slot] = 0u;
            for (uint32_t i = 0u; i < 4u; ++i) {
                upoint[cell + i] = {0, 0, 0}; unormal[cell + i] = {0, 0, 0};
                udepth[cell + i] = 0.0f;
                ucontact_a[cell + i] = 0u; ucontact_b[cell + i] = 0u;
                ucontact_a_kind[cell + i] = ::nuka::nk::kUContactSideBody;
                ucontact_b_kind[cell + i] = ::nuka::nk::kUContactSideBody;
                ucontact_gen[cell + i] = 0u;
            }
        }
    }
    if (N == 0u) return;  // uniform per-warp.

    const uint32_t global_particle = env * pp.particles_per_env + pi;
    // The fluid slice [n_soft, P) under PBF/SoftFluid reads the predicted position
    // (the gravity-integrated pos the density solve uses), consistent with the
    // particle-particle grid; the soft slice + pure-Xpbd/Coupled keep particle_pos
    // (their predict already wrote it).
    const bool fluid_predicted = pp.fluid_pos_source != 0u &&
                                 pi >= pp.n_soft_particles &&
                                 pbf_predicted_pos != nullptr;
    const Vec3 center =
        fluid_predicted ? pbf_predicted_pos[global_particle]
                        : particle_pos[global_particle];
    const float radius = pp.particle_radius;
    const collision::AABB query = ParticleAabb(center, radius + pp.contact_margin);

    // LANE 0 gathers the candidate list (the cand[] + traversal stack are local
    // memory -- holding them on every lane would spill); under kWarp ncand + each
    // cand[ci] are broadcast so all lanes walk the SAME candidates in the SAME order,
    // byte-identical to the serial gather. The overflow flag is raised by lane 0 only.
    uint32_t cand[kMaxCandidates];
    uint32_t ncand = 0u;
    if (lane == 0u) {
        bool overflow = false;
        ncand = GatherOverlaps(query, lbvh_nodes, body_aabb_lo, body_aabb_hi, env, N,
                               cand, &overflow);
        if (overflow && env_status != nullptr) {
            atomicOr(&env_status[env], kEnvStatusPairOverflow);
        }
    }
    if (kWarp) ncand = __shfl_sync(0xffffffffu, ncand, 0);

    const amf::PrimParams ps = MakeParticleSphere(center, radius);
    uint32_t written = 0u;
    uint32_t emitted_points = 0u;
    // Walk all gathered candidates: write the first cands_per_particle (deterministic
    // ascending-body order), flag any excess loud via kEnvStatusPairOverflow.
    for (uint32_t ci = 0u; ci < ncand; ++ci) {
        // Under kWarp broadcast the candidate body so every lane agrees on the
        // shape/pose (the hull branch needs all 32 lanes converged on identical
        // inputs); serial just reads its own cand[ci].
        uint32_t body = (lane == 0u) ? cand[ci] : 0u;
        if (kWarp) body = __shfl_sync(0xffffffffu, body, 0);
        const PrimShapeDev sb = LoadPrimShape(shape_table, body);
        const math::Transform xb = body_pose[env * N + body];
        const amf::PrimParams pb = MakePrim(sb, xb);

        // Sphere(=particle, A) vs the body shape; the manifold normal is the
        // separation dir for the particle (push it off the body). sb.kind is uniform
        // across the warp (same body), so the warp does NOT diverge at the switch.
        ContactManifold m;
        m.Clear();
        switch (sb.kind) {
            // The cheap analytic branches do not scan a hull -> LANE 0 only (the
            // manifold m is meaningful on lane 0, which performs the store below).
            case kKindSphere:  if (lane == 0u) amf::SphereSphere(ps, pb, &m); break;
            case kKindBox:     if (lane == 0u) amf::SphereBox(ps, pb, &m); break;
            case kKindPlane:   if (lane == 0u) amf::SpherePlane(ps, pb, &m); break;
            case kKindCapsule: if (lane == 0u) {
                                   amf::CapsuleSphere(pb, ps, &m);
                                   // CapsuleSphere gives sep dir for the capsule(=A
                                   // here); flip to the particle's separation dir.
                                   for (uint32_t i = 0u; i < m.point_count; ++i)
                                       m.points[i].normal = Vec3{-m.points[i].normal.x,
                                                                 -m.points[i].normal.y,
                                                                 -m.points[i].normal.z};
                               }
                               break;
            case kKindConvexHull: {
                // Under kWarp ALL 32 lanes call SphereHull in full-warp lockstep: the
                // giant-hull SupportHull scan splits across lanes (warp_lockstep) for
                // the win; lane 0 keeps its own (bit-identical) result for the store.
                cvx::ConvexHullView hull;
                hull.verts = hull_verts + static_cast<size_t>(sb.hull_vert_offset) * 3u;
                hull.vcount = sb.hull_vert_count;
                hull.frame = pb.frame;
                hull.warp_lockstep = kWarp;
                cvx::SphereHull(ps, hull, /*sphere_is_a=*/true, &m);
                break;
            }
            case kKindHeightfield:
                // A heightfield body without a wired descriptor cannot be walked;
                // surface the coverage miss loud (never a silent tunnel-through).
                if (lane == 0u) {
                    if (pp.has_heightfield != 0u && heights != nullptr) {
                        ParticleHeightfield(center, radius, pp.contact_margin, xb, pp,
                                            heights, &m);
                    } else if (env_status != nullptr) {
                        atomicOr(&env_status[env], kEnvStatusPairOverflow);
                    }
                }
                break;
            case kKindSdfMesh:
                // The particle (sphere) vs the body's cooked silhouette SDF: one
                // query of the SAME sparse_sdf_sample the rigid SDF narrowphase +
                // the MPM grid BC call (one query, three callers). Point-vs-grid, so
                // LANE 0 only (no wide hull to split). phi = signed distance at the
                // particle center in the body's local frame; penetrating iff phi <
                // radius (+ margin band). The normal is the OUTWARD SDF gradient
                // (separation dir for the particle); depth = radius - phi.
                if (lane == 0u && sdf_headers != nullptr) {
                    const uint32_t grid = sb.sdf_grid;
                    if (grid != ~0u) {
                        const sdfq::SparseSdfDevice sg = LoadParticleSdfGrid(
                            sdf_headers, sdf_cell_count, sdf_keys, sdf_values,
                            sdf_grads, grid);
                        const Vec3 q =
                            ::nuka::phi::nkops::SdfInverseTransformPoint(xb, center);
                        Vec3 grad{0, 0, 0};
                        const float phi = sdfq::sparse_sdf_sample(sg, q, grad);
                        if (phi < sdfq::SparseSdfDevice::kOutsideBand &&
                            phi < radius + pp.contact_margin) {
                            const Vec3 gw =
                                ::nuka::phi::nkops::SdfRotate(xb.rotation, grad);
                            const float gl = sqrtf(gw.x * gw.x + gw.y * gw.y +
                                                   gw.z * gw.z);
                            const Vec3 n = (gl > 1.0e-12f)
                                ? Vec3{gw.x / gl, gw.y / gl, gw.z / gl}
                                : Vec3{0.0f, 0.0f, 1.0f};
                            ::nuka::constraint::ContactPoint pt;
                            pt.position = Vec3{center.x - n.x * radius,
                                               center.y - n.y * radius,
                                               center.z - n.z * radius};
                            pt.normal = n;             // sep dir for the particle.
                            pt.penetration = radius - phi;
                            pt.stable_key = 0ull;
                            m.AddPoint(pt);
                        }
                    }
                }
                break;
            default: break;  // unknown kind: no analytic particle handler.
        }
        // Manifold store + bookkeeping: LANE 0 ONLY (m on the other lanes is unused).
        // The whole warp keeps iterating every candidate to `ncand` so the hull
        // lockstep never desyncs -- so the store side-effects are gated WITHOUT any
        // warp-divergent break/continue (the slot byte-stream is unchanged from the
        // serial path; a stop only suppresses further writes + ORs the loud flag).
        if (lane == 0u && m.point_count != 0u) {
            const uint32_t slot = slot0 + written;
            // More real body contacts than reserved slots, OR the slot ran past the
            // env stride: keep the first cands_per_particle (deterministic order),
            // flag the drop loud (idempotent OR), and stop emitting.
            if (written >= pp.cands_per_particle || slot >= pp.slot_stride) {
                if (env_status != nullptr)
                    atomicOr(&env_status[env], kEnvStatusPairOverflow);
            } else {
                const size_t scell =
                    (static_cast<size_t>(env) * pp.slot_stride + slot) * 4u;
                const uint32_t n = m.point_count;
                ucount[static_cast<size_t>(env) * pp.slot_stride + slot] = n;
                for (uint32_t i = 0u; i < 4u; ++i) {
                    if (i < n) {
                        upoint[scell + i] = m.points[i].position;
                        unormal[scell + i] = m.points[i].normal;
                        udepth[scell + i] = m.points[i].penetration;
                        // Side A == the particle (GLOBAL id + the particle index-kind
                        // tag); side B == the body collidable (env-local row + tag).
                        ucontact_a[scell + i] = global_particle;
                        ucontact_b[scell + i] = body;
                        ucontact_a_kind[scell + i] = ::nuka::nk::kUContactSideParticle;
                        ucontact_b_kind[scell + i] = ::nuka::nk::kUContactSideBody;
                        ucontact_gen[scell + i] = 1u;
                    }
                }
                emitted_points += n;
                ++written;
            }
        }
    }
    if (lane == 0u && emitted_points > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], emitted_points);
    }
}

Status OpNarrowphaseBodyParticle(const ModelView& model, const DataView& data,
                                 const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphaseBodyParticleParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    if (p->env_count == 0u || p->particles_per_env == 0u ||
        p->slot_stride == 0u || p->cands_per_particle == 0u) {
        return Status::Ok;  // no particles or no reserved slots -> nothing to do.
    }
    if (p->particle_radius <= 0.0f) return Status::Ok;  // no collision radius cooked.
    if (data.particle_pos == nullptr || data.body_pose == nullptr ||
        model.shape_table == nullptr) {
        return Status::Ok;
    }
    const uint32_t total = p->env_count * p->particles_per_env;
    // Warp-per-particle (a wide hull collider) launches 32 threads/particle (== the
    // serial block count *32); thread-per-particle launches one thread/particle.
    const bool warp = p->warp_per_particle != 0u;
    const uint32_t threads = warp ? (total * 32u) : total;
    const uint32_t blocks = (threads + kBlockSize - 1u) / kBlockSize;
    auto launch = [&](auto kernel) {
        LaunchCuda(kernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   static_cast<const float*>(model.shape_table),
                   static_cast<const math::Transform*>(data.body_pose),
                   static_cast<const float*>(model.hull_verts),
                   static_cast<const Vec3*>(data.particle_pos),
                   static_cast<const Vec3*>(data.pbf_predicted_pos),
                   reinterpret_cast<const cg::LbvhNode*>(data.lbvh_nodes),
                   static_cast<const Vec3*>(data.body_aabb_lo),
                   static_cast<const Vec3*>(data.body_aabb_hi),
                   static_cast<const float*>(model.heights),
                   static_cast<const float*>(model.sdf_headers),
                   static_cast<const uint32_t*>(model.sdf_cell_count),
                   static_cast<const uint64_t*>(model.sdf_cell_keys),
                   static_cast<const float*>(model.sdf_cell_values),
                   static_cast<const Vec3*>(model.sdf_cell_gradients), *p,
                   data.ucontact_count, data.ucontact_point, data.ucontact_normal,
                   data.ucontact_depth, data.ucontact_a, data.ucontact_b,
                   data.ucontact_a_kind, data.ucontact_b_kind, data.ucontact_gen,
                   data.contact_count, data.env_status);
    };
    if (warp) launch(&NarrowphaseBodyParticleKernel<true>);
    else      launch(&NarrowphaseBodyParticleKernel<false>);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkNarrowphaseBodyParticleOps() {
    SetCudaOp(NkOp::NarrowphaseBodyParticle, &OpNarrowphaseBodyParticle);
}

}  // namespace nuka::phi
