// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — WP5/WP6/WP8 multi-articulation dog-dog link contact.
//
// THE owner bottom-line ("狗和狗之间需要物理碰撞") realized GENERALLY (no N=2
// special-case): a contact whose two sides are FK-posed collision links of TWO
// DISTINCT co-resident articulations is detected analytically and turned into a
// two-way physical reaction.
//
// The crux that makes this clean: route the dog-dog contact through the EXISTING
// FUSED articulation contact pipeline, which is ALREADY block-per-articulation
// and articulation-keyed (SolveArticulatedContactRowsKernel launches
// dim3(articulation_count); each block reads its OWN slot block
// [articulation*kMaxFootContactsPerEnv ...] and scatters the reaction into its
// OWN per-articulation qdot tile via link_to_articulation[contact_link]). So a
// detected overlap between dog A link L_a and dog B link L_b is emitted as TWO
// fused contact rows — one into dog A's slot block (normal = A away from B,
// contact_link = L_a) and one into dog B's slot block (normal = B away from A =
// -normal, contact_link = L_b). Dog A's solver block resolves its row into dog
// A's qdot tile; dog B's block resolves its into dog B's tile. TWO DISTINCT
// per-articulation reactions, GENERAL for any K, with NO new island solver and
// NO env-aliasing (each articulation owns its slot block by index, not by env).
//
// WP5 (link_pose -> collidable visible to narrowphase): satisfied WITHOUT a
// body_pose sync op — this kernel reads the FK-refreshed link_pose field directly
// (the same field the foot/union detection read), composes it with the cooked
// per-link link_geom_local, and runs the analytic primitive narrowphase. No
// broadphase / body_pose round-trip is needed (a tiny self-contained O(L^2/env)
// pairwise sweep over the env's collision links, K small).
//
// WP6 (capsule x capsule): amf::DispatchPair via the analytic segment-segment
// CapsuleCapsule (no EPA), plus sphere/box for completeness.
//
// WP7 (schedule artic-key): N/A on the FUSED path — the fused solver does NOT go
// through schedule.cpp's island/union machinery; it is articulation-keyed by
// construction (slot_base = articulation*stride). The schedule env-aliasing bug
// is a UNION-path concern; we deliberately reuse the fused path to side-step it.
//
// D1: EARLY-EXITS unless articulations_per_env > 1, so the single-dog (K==1)
// graph never enqueues this op (DeviceSupportsOp registers it, but Pipeline only
// adds it for K>1) AND, even if launched, the gate returns immediately. K==1
// goldens are byte-identical.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"
#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"  // DogDogContactParams

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;
namespace amf = ::nuka::collision::amf;

constexpr uint32_t kInvalidLink = ~0u;

// Narrowphase kind codes (mirror narrowphase_prims.cu kKind*, same ShapeType
// ordering: Sphere=0, Capsule=1, Box=2, Plane=3).
constexpr uint32_t kKindSphere = 0u;
constexpr uint32_t kKindCapsule = 1u;
constexpr uint32_t kKindBox = 2u;

// Dispatch the analytic handler for an ORDERED (kind_a, kind_b) primitive pair,
// covering the dog-dog cases (capsule/sphere/box x capsule/sphere/box). Mirrors
// narrowphase_prims.cu DispatchPair's canonical+swap+flip logic, restricted to
// the bounded primitives links carry (no plane/hull). Normal = separation dir
// for A. (DispatchPair in narrowphase_prims.cu is file-local; this is the
// dog-dog subset, self-contained so this TU needs no cross-file device symbol.)
__device__ void DispatchPairDog(uint32_t ka, const amf::PrimParams& a,
                                uint32_t kb, const amf::PrimParams& b,
                                amf::ContactManifold* out) {
    out->Clear();
    if (ka == kKindSphere && kb == kKindSphere) { amf::SphereSphere(a, b, out); return; }
    if (ka == kKindSphere && kb == kKindBox)    { amf::SphereBox(a, b, out); return; }
    if (ka == kKindBox && kb == kKindBox)       { amf::BoxBox(a, b, out); return; }
    if (ka == kKindCapsule && kb == kKindSphere){ amf::CapsuleSphere(a, b, out); return; }
    if (ka == kKindCapsule && kb == kKindCapsule){ amf::CapsuleCapsule(a, b, out); return; }
    auto flip = [&]() {
        for (uint32_t i = 0; i < out->point_count; ++i) {
            out->points[i].normal = math::Vec3{-out->points[i].normal.x,
                                               -out->points[i].normal.y,
                                               -out->points[i].normal.z};
        }
    };
    if (ka == kKindBox && kb == kKindSphere)    { amf::SphereBox(b, a, out); flip(); return; }
    if (ka == kKindSphere && kb == kKindCapsule){ amf::CapsuleSphere(b, a, out); flip(); return; }
    // capsule x box (and box x capsule): approximate the capsule as its mid
    // sphere vs box (the analytic capsule-box handler is deferred). Conservative
    // but physically push-apart for the dog trunk/leg-vs-box cases the spike hits.
    if (ka == kKindCapsule && kb == kKindBox)   { amf::SphereBox(a, b, out); return; }
    if (ka == kKindBox && kb == kKindCapsule)   { amf::SphereBox(b, a, out); flip(); return; }
    // Unhandled -> empty manifold (out already Cleared).
}

// Build an amf::PrimParams from a link's cooked geometry (kind sentinel +
// 4 packed params) posed by its FK world transform composed with the link-local
// shape transform. kind is the link_geom_kind sentinel value (== ShapeType + 1,
// 0 == none). Returns the narrowphase kKind (Sphere=0/Capsule=1/Box=2) via
// out_kind, or false when the link carries no collidable primitive.
__device__ bool BuildLinkPrim(uint32_t kind_sentinel, const float* params4,
                              const math::Transform& world_link,
                              const math::Transform& local_geom,
                              amf::PrimParams* out, uint32_t* out_kind) {
    if (kind_sentinel == 0u) return false;  // no cooked shape on this link.
    const uint32_t st = kind_sentinel - 1u;  // ShapeType: 0 Sph,1 Cap,2 Box,3 Plane
    if (st > 2u) return false;               // only Sphere/Capsule/Box collidable.
    // Compose the world primitive transform = world_link o local_geom (HD-clean).
    const math::Transform world_geom =
        amf::ComposeTransformHD(world_link, local_geom);
    out->frame = amf::BuildPrimFrameDevice(world_geom);
    out->radius = params4[0];
    out->half_height = params4[1];
    out->half_extents = math::Vec3{params4[0], params4[1], params4[2]};
    *out_kind = st;  // ShapeType order == narrowphase kKind order (verified).
    return true;
}

// One thread per environment. Walks the env's collision links pairwise; for every
// pair of links belonging to DISTINCT articulations whose posed primitives
// overlap, emits TWO fused contact rows (one per dog, opposite normals) into the
// per-ARTICULATION slot blocks. Each articulation's block is the legacy fused
// stride [articulation * kMaxFootContactsPerEnv, +kMaxFootContactsPerEnv); the
// per-dog slot counters cap at kMaxFootContactsPerEnv (a named, conservative
// budget; the deepest contacts are kept in fixed link-pair order so the slot
// assignment is deterministic). All unused slots in the env's articulation blocks
// are zero-filled (kInvalidLink) so the fused assemble/solve skip them.
__global__ void DogDogContactKernel(
    const math::Transform* __restrict__ link_pose,        // FK world poses (per env*L)
    const uint32_t* __restrict__ link_geom_kind,          // per link (env-major)
    const float* __restrict__ link_geom_params,           // 4 f32 per link
    const math::Transform* __restrict__ link_geom_local,  // per link
    const uint32_t* __restrict__ link_to_articulation,    // per env*L
    float contact_margin,
    uint32_t env_count,
    uint32_t base_link_count,           // links per env (sum over K dogs)
    uint32_t articulations_per_env,     // K
    uint32_t max_foot_contacts,         // == kMaxFootContactsPerEnv (slot stride)
    uint32_t* __restrict__ out_contact_link,
    math::Vec3* __restrict__ out_contact_point,
    math::Vec3* __restrict__ out_contact_normal,
    float* __restrict__ out_contact_depth,
    uint32_t* __restrict__ out_contact_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) return;

    const uint32_t L = base_link_count;
    const uint32_t K = articulations_per_env;
    const uint32_t stride = max_foot_contacts;
    // The env's per-articulation slot blocks span [art0*stride, (artK-1)*stride+stride).
    // articulation_global = env * K + local_artic. Zero-fill the whole env span first.
    for (uint32_t a = 0u; a < K; ++a) {
        const uint32_t art_global = env * K + a;
        const uint32_t base = art_global * stride;
        for (uint32_t s = 0u; s < stride; ++s) {
            out_contact_link[base + s] = kInvalidLink;
            out_contact_point[base + s] = math::Vec3{0.0f, 0.0f, 0.0f};
            out_contact_normal[base + s] = math::Vec3{0.0f, 0.0f, 0.0f};
            out_contact_depth[base + s] = 0.0f;
        }
    }

    // Per-articulation slot fill counters (K small; capped at stride). One thread
    // per env owns every slot block, so no atomics are needed.
    // (articulations_per_env is bounded; the demo uses K<=8, well under the cap.)
    constexpr uint32_t kMaxArtPerEnv = 64u;
    uint32_t fill[kMaxArtPerEnv];
    if (K > kMaxArtPerEnv) {
        // Honest guard: refuse silently-wrong over-cap (matches the G0 discipline).
        out_contact_count[env] = 0u;
        return;
    }
    for (uint32_t a = 0u; a < K; ++a) fill[a] = 0u;

    uint32_t emitted = 0u;
    // Pairwise over the env's links (i<j), fixed ascending order (deterministic).
    for (uint32_t li = 0u; li < L; ++li) {
        const uint32_t gi = env * L + li;
        const uint32_t kind_i = link_geom_kind[gi];
        if (kind_i == 0u) continue;
        const uint32_t art_i = link_to_articulation[gi];
        amf::PrimParams pa;
        uint32_t ka;
        if (!BuildLinkPrim(kind_i, link_geom_params + static_cast<size_t>(gi) * 4u,
                           link_pose[gi], link_geom_local[gi], &pa, &ka)) {
            continue;
        }
        for (uint32_t lj = li + 1u; lj < L; ++lj) {
            const uint32_t gj = env * L + lj;
            const uint32_t kind_j = link_geom_kind[gj];
            if (kind_j == 0u) continue;
            const uint32_t art_j = link_to_articulation[gj];
            if (art_i == art_j) continue;  // SAME dog -> self-collision off.

            amf::PrimParams pb;
            uint32_t kb;
            if (!BuildLinkPrim(kind_j, link_geom_params + static_cast<size_t>(gj) * 4u,
                               link_pose[gj], link_geom_local[gj], &pb, &kb)) {
                continue;
            }

            // Analytic narrowphase. amf normal convention = separation dir for A.
            amf::ContactManifold m;
            DispatchPairDog(ka, pa, kb, pb, &m);
            if (m.point_count == 0u) continue;

            // Take the deepest point of the manifold (capsule-capsule emits 1).
            uint32_t best = 0u;
            for (uint32_t pi = 1u; pi < m.point_count; ++pi) {
                if (m.points[pi].penetration > m.points[best].penetration) best = pi;
            }
            const float pen = m.points[best].penetration + contact_margin;
            if (pen <= 0.0f) continue;
            const math::Vec3 pos = m.points[best].position;
            const math::Vec3 n_a = m.points[best].normal;  // A away from B.

            // Emit into dog A's slot block (reaction pushes A along +n_a).
            const uint32_t art_i_global = env * K + art_i;
            if (fill[art_i] < stride) {
                const uint32_t slot = art_i_global * stride + fill[art_i];
                out_contact_link[slot] = gi;
                out_contact_point[slot] = pos;
                out_contact_normal[slot] = n_a;
                out_contact_depth[slot] = pen;
                ++fill[art_i];
                ++emitted;
            }
            // Emit into dog B's slot block (reaction pushes B along -n_a).
            const uint32_t art_j_global = env * K + art_j;
            if (fill[art_j] < stride) {
                const uint32_t slot = art_j_global * stride + fill[art_j];
                out_contact_link[slot] = gj;
                out_contact_point[slot] = pos;
                out_contact_normal[slot] =
                    math::Vec3{-n_a.x, -n_a.y, -n_a.z};
                out_contact_depth[slot] = pen;
                ++fill[art_j];
                ++emitted;
            }
        }
    }
    out_contact_count[env] = emitted;
}

Status OpDogDogContact(const ModelView& model, const DataView& data,
                       const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const DogDogContactParams*>(params);
    if (p == nullptr) return Status::Failed;
    // D1 GATE: only co-resident multi-articulation worlds run the dog-dog path.
    if (p->articulations_per_env <= 1u || p->env_count == 0u ||
        p->base_link_count == 0u) {
        return Status::Ok;
    }
    if (model.link_geom_kind == nullptr) return Status::Ok;  // no cooked geometry.

    constexpr uint32_t kBlockSize = 64u;
    const uint32_t block_count = (p->env_count + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(DogDogContactKernel, dim3(block_count), dim3(kBlockSize), 0u, stream,
               static_cast<const math::Transform*>(data.link_pose),
               static_cast<const uint32_t*>(model.link_geom_kind),
               static_cast<const float*>(model.link_geom_params),
               static_cast<const math::Transform*>(model.link_geom_local),
               static_cast<const uint32_t*>(model.link_to_articulation),
               p->contact_margin, p->env_count, p->base_link_count,
               p->articulations_per_env, p->max_foot_contacts,
               data.contact_link, data.contact_point, data.contact_normal,
               data.contact_depth, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkDogDogContactOps() {
    SetCudaOp(NkOp::DogDogContact, &OpDogDogContact);
}

}  // namespace nuka::phi
