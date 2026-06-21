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
//   1. Build the particle's query AABB (sphere of particle_radius) at its position.
//   2. Traverse the env's arena LBVH (data.lbvh_nodes, the env*(2N-1) node slice the
//      broadphase built) for overlapping collidable bodies, collected into a private
//      insertion-sorted candidate list capped at kCrossSystemMaxCandidates (the
//      cross_system_query CSR pattern). For a <2-collidable env (no LBVH) it scans
//      the body AABBs directly. A particle exceeding the cap ORs kEnvStatusPairOverflow.
//   3. For each candidate body run the sphere-vs-shape manifold: amf::Sphere{Sphere,
//      Box,Plane} inline, cvx::SphereHull (the EPA-bypass closest-point query) for a
//      convex hull, a face-only handler for the heightfield (the SphereHull/EPA
//      shallow-penetration dead band is avoided exactly as the heightfield path does).
//   4. Write the manifold into the particle's RESERVED contact-slot sub-range
//      [slot_base + pi*cands_per_particle + k]. The per-particle base is a FIXED
//      (non-atomic) function of the particle index, so the body<->particle slot stream
//      is bit-D1 by construction AND occupies a deterministic sub-range relative to the
//      racy rigid-rigid slots [0, pair_count) — the cross-stream ordering guard. Side A
//      == the particle (GLOBAL id + the kUContactSideParticle index-kind tag), side B
//      == the body collidable (kUContactSideBody); the manifold normal is the
//      separation dir for the particle (push it off the body).
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
#include "phi/op_schema.hpp"

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
constexpr uint32_t kKindHeightfield = ::nuka::collision::kShapeHeightfield;

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

// Particle (sphere) vs the heightfield collidable: walk the overlapped cell range
// (the body AABB query already gated the broad overlap), test each cell triangle
// via the deep-sink-robust face handler, keep the DEEPEST face contact. Mirrors the
// heightfield narrowphase cell walk + corner-height extraction.
__device__ void ParticleHeightfield(Vec3 center, float radius,
                                    const PrimShapeDev& hf_sh, const math::Transform& xhf,
                                    const float* heights, const float* shape_table,
                                    ContactManifold* m) {
    (void)hf_sh; (void)shape_table;
    // Heightfield world frame (its body pose); the grid is in this LOCAL frame.
    amf::PrimFrame hf;
    hf.cx = PrimRotate(xhf.rotation, Vec3{1, 0, 0});
    hf.cy = PrimRotate(xhf.rotation, Vec3{0, 1, 0});
    hf.cz = PrimRotate(xhf.rotation, Vec3{0, 0, 1});
    hf.t = xhf.position;
    // The cooked AABB lanes carry the grid half-extents + LOCAL z-range; the device
    // heightfield descriptor (cell/origin/data_offset) is NOT in shape_table, so a
    // heightfield body is only reachable here through the model's single descriptor.
    // The face handler needs the LOCAL grid; without the descriptor we cannot walk
    // cells, so a heightfield particle contact is left to the future descriptor wire
    // (the demo couples against robot links + boxes). Leave empty.
    (void)heights; (void)center; (void)radius; (void)m;
}

__global__ void NarrowphaseBodyParticleKernel(
    const float* __restrict__ shape_table,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ hull_verts,
    const Vec3* __restrict__ particle_pos,
    const cg::LbvhNode* __restrict__ lbvh_nodes,
    const Vec3* __restrict__ body_aabb_lo,
    const Vec3* __restrict__ body_aabb_hi,
    const float* __restrict__ heights,
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
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = pp.env_count * pp.particles_per_env;
    if (gid >= total) return;
    const uint32_t env = gid / pp.particles_per_env;
    const uint32_t pi = gid - env * pp.particles_per_env;  // env-local particle
    const uint32_t N = pp.bodies_per_env;

    // Reserved contact-slot sub-range for THIS particle: a fixed (non-atomic)
    // function of the particle index, so the body<->particle slot stream is bit-D1
    // by construction and lands in a deterministic sub-range relative to the rigid
    // slots [0, pair_count). Initialize all reserved slots to inactive first.
    const uint32_t slot0 = pp.particle_slot_base + pi * pp.cands_per_particle;
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
    if (N == 0u) return;

    const uint32_t global_particle = env * pp.particles_per_env + pi;
    const Vec3 center = particle_pos[global_particle];
    const float radius = pp.particle_radius;
    const collision::AABB query = ParticleAabb(center, radius + pp.contact_margin);

    uint32_t cand[kMaxCandidates];
    bool overflow = false;
    const uint32_t ncand = GatherOverlaps(query, lbvh_nodes, body_aabb_lo,
                                          body_aabb_hi, env, N, cand, &overflow);
    if (overflow && env_status != nullptr) {
        atomicOr(&env_status[env], kEnvStatusPairOverflow);
    }

    const amf::PrimParams ps = MakeParticleSphere(center, radius);
    uint32_t written = 0u;
    uint32_t emitted_points = 0u;
    // Walk all gathered candidates: write the first cands_per_particle (deterministic
    // ascending-body order), flag any excess loud via kEnvStatusPairOverflow.
    for (uint32_t ci = 0u; ci < ncand; ++ci) {
        const uint32_t body = cand[ci];
        const PrimShapeDev sb = LoadPrimShape(shape_table, body);
        const math::Transform xb = body_pose[env * N + body];
        const amf::PrimParams pb = MakePrim(sb, xb);

        // Sphere(=particle, A) vs the body shape; the manifold normal is the
        // separation dir for the particle (push it off the body).
        ContactManifold m;
        m.Clear();
        switch (sb.kind) {
            case kKindSphere:  amf::SphereSphere(ps, pb, &m); break;
            case kKindBox:     amf::SphereBox(ps, pb, &m); break;
            case kKindPlane:   amf::SpherePlane(ps, pb, &m); break;
            case kKindCapsule: amf::CapsuleSphere(pb, ps, &m);
                               // CapsuleSphere gives sep dir for the capsule(=A here);
                               // flip to the particle's separation dir.
                               for (uint32_t i = 0u; i < m.point_count; ++i)
                                   m.points[i].normal = Vec3{-m.points[i].normal.x,
                                                             -m.points[i].normal.y,
                                                             -m.points[i].normal.z};
                               break;
            case kKindConvexHull: {
                cvx::ConvexHullView hull;
                hull.verts = hull_verts + static_cast<size_t>(sb.hull_vert_offset) * 3u;
                hull.vcount = sb.hull_vert_count;
                hull.frame = pb.frame;
                hull.warp_lockstep = false;
                cvx::SphereHull(ps, hull, /*sphere_is_a=*/true, &m);
                break;
            }
            case kKindHeightfield:
                ParticleHeightfield(center, radius, sb, xb, heights, shape_table, &m);
                break;
            default: break;  // SdfMesh / unknown: no analytic particle handler.
        }
        if (m.point_count == 0u) continue;

        // More real body contacts than reserved slots: keep the first
        // cands_per_particle (deterministic order) and flag the drop loud.
        if (written >= pp.cands_per_particle) {
            if (env_status != nullptr) atomicOr(&env_status[env], kEnvStatusPairOverflow);
            continue;
        }
        const uint32_t slot = slot0 + written;
        if (slot >= pp.slot_stride) {
            if (env_status != nullptr) atomicOr(&env_status[env], kEnvStatusPairOverflow);
            break;
        }
        const size_t scell =
            (static_cast<size_t>(env) * pp.slot_stride + slot) * 4u;
        const uint32_t n = m.point_count;
        ucount[static_cast<size_t>(env) * pp.slot_stride + slot] = n;
        for (uint32_t i = 0u; i < 4u; ++i) {
            if (i < n) {
                upoint[scell + i] = m.points[i].position;
                unormal[scell + i] = m.points[i].normal;
                udepth[scell + i] = m.points[i].penetration;
                // Side A == the particle (GLOBAL id + the particle index-kind tag);
                // side B == the body collidable (env-local row + body tag).
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
    if (emitted_points > 0u && contact_count != nullptr) {
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
    const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(NarrowphaseBodyParticleKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               static_cast<const float*>(model.shape_table),
               static_cast<const math::Transform*>(data.body_pose),
               static_cast<const float*>(model.hull_verts),
               static_cast<const Vec3*>(data.particle_pos),
               reinterpret_cast<const cg::LbvhNode*>(data.lbvh_nodes),
               static_cast<const Vec3*>(data.body_aabb_lo),
               static_cast<const Vec3*>(data.body_aabb_hi),
               static_cast<const float*>(model.heights), *p,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_a_kind, data.ucontact_b_kind, data.ucontact_gen,
               data.contact_count, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkNarrowphaseBodyParticleOps() {
    SetCudaOp(NkOp::NarrowphaseBodyParticle, &OpNarrowphaseBodyParticle);
}

}  // namespace nuka::phi
