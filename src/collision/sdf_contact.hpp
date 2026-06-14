#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::sdf_contact -- Newton-on-summed-SDF contact (v0.7 p08-B)
// ---------------------------------------------------------------------------
// Detects a contact between two SDF-equipped bodies by minimizing the SUMMED
// signed-distance field  f(p) = phi_a(p) + phi_b(p)  over the world point p.
//
// WHY THE SUM. At a contact the two surfaces touch (or overlap); the point that
// minimizes phi_a + phi_b is the point most simultaneously-inside both colliders
// -- the witness of deepest mutual penetration. At a converged minimum the two
// gradients are anti-parallel (grad_a ~= -grad_b): each SDF gradient is the
// outward surface normal, and at the witness the surfaces face each other. The
// contact normal (the SEPARATION direction for body A, matching the engine's
// row convention) is therefore  normalize(grad_b - grad_a)  (== -grad_a when
// the two grads are anti-parallel; the negation of the raw grad_a-grad_b because
// A's surface outward normal points INTO B at the witness, opposite to the way A
// must move to separate). Penetration depth is  -(phi_a + phi_b)  when that sum
// is negative (both phi < 0 => inside both => overlapping). See the sign-
// derivation comment at the normal computation, verified vs SolveUnitGroundContact.
//
// HOST + DEVICE. This is a CUDA-FREE header (uses nuka::math::Vec3, which is
// __host__ __device__). `find_sdf_contact_newton` is a pure NUKA_SDF_HD free
// function so the host accuracy/D1 tests call it directly AND the device
// emission kernel (sdf_contact.cu) wraps it -- ONE code path, validated on the
// host, shipped on the GPU (same handoff discipline as the p07 sampler).
//
// DETERMINISM (D1). Fixed max-iteration count, fixed convergence tolerance, a
// fixed-order arithmetic step, and NO thread-order-dependent reduction. The
// Newton step magnitude is clamped to <= voxel_size (the smaller of the two
// bodies' voxels) so a fast/thin body cannot tunnel a sub-voxel panel (R-E
// thin-shell) and so the iterate never leaps out of the narrow band in one step.
// Two runs of the same query are bit-identical.
//
// p08-C HANDOFF (the analytical IFT-at-convergence adjoint). The output struct
// carries EVERYTHING the adjoint must differentiate WITHOUT re-deriving the
// solve: the converged world point, both world-frame gradients, both phi, and
// the scalar curvature proxy `step_curvature` actually used in the final step.
// The contact point satisfies the stationarity residual r(p; M, J) = grad_a +
// grad_b = 0 at convergence; IFT then gives dp/d(SDF values) and dp/d(pose) from
// dr/dp (the curvature) and dr/d(params). p08-B does NOT implement that -- but
// it structures its output so p08-C does not have to reconstruct the geometry.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_SDF_HD __host__ __device__
#else
#define NUKA_SDF_HD
#endif

namespace nuka::constraint {
struct RowBuffers;  // fwd-decl (host row builders below; defined in row_buffers.hpp)
} // namespace nuka::constraint

namespace nuka::collision {

// A rigid world<->local frame for one SDF body, passed as a 3x3 rotation (row
// vectors) + translation so the device function needs NO non-HD math helpers
// (Quat::Rotate / Vec3::Normalized are `inline` but not __host__ __device__;
// the caller bakes the rotation on the host). Convention:
//   local  = R^T * (world - t)        (world_to_local)
//   world  = R   *  local  + t        (local_to_world)
// where R's COLUMNS are (rot_col0, rot_col1, rot_col2) == the body axes in
// world. We store columns so local->world is a plain column combination and
// world->local is the (orthonormal) transpose -- a dot with each column.
struct SdfBodyFrame {
    math::Vec3 rot_col0{1.0f, 0.0f, 0.0f};  // world-space body X axis
    math::Vec3 rot_col1{0.0f, 1.0f, 0.0f};  // world-space body Y axis
    math::Vec3 rot_col2{0.0f, 0.0f, 1.0f};  // world-space body Z axis
    math::Vec3 translation{0.0f, 0.0f, 0.0f};

    NUKA_SDF_HD math::Vec3 WorldToLocal(math::Vec3 w) const {
        const math::Vec3 d{w.x - translation.x, w.y - translation.y, w.z - translation.z};
        return {d.x * rot_col0.x + d.y * rot_col0.y + d.z * rot_col0.z,
                d.x * rot_col1.x + d.y * rot_col1.y + d.z * rot_col1.z,
                d.x * rot_col2.x + d.y * rot_col2.y + d.z * rot_col2.z};
    }
    NUKA_SDF_HD math::Vec3 LocalToWorld(math::Vec3 l) const {
        return {rot_col0.x * l.x + rot_col1.x * l.y + rot_col2.x * l.z + translation.x,
                rot_col0.y * l.x + rot_col1.y * l.y + rot_col2.y * l.z + translation.y,
                rot_col0.z * l.x + rot_col1.z * l.y + rot_col2.z * l.z + translation.z};
    }
    // Rotate a LOCAL-frame direction (e.g. an SDF gradient) into world.
    NUKA_SDF_HD math::Vec3 DirLocalToWorld(math::Vec3 l) const {
        return {rot_col0.x * l.x + rot_col1.x * l.y + rot_col2.x * l.z,
                rot_col0.y * l.x + rot_col1.y * l.y + rot_col2.y * l.z,
                rot_col0.z * l.x + rot_col1.z * l.y + rot_col2.z * l.z};
    }
};

struct SdfContactParams {
    uint32_t max_iterations = 32u;       // fixed -> D1 (no early-out asymmetry)
    float    grad_tolerance = 1.0e-5f;   // ||grad_a+grad_b|| stationarity tol
    float    step_clamp_scale = 1.0f;    // step <= step_clamp_scale * voxel_size
};

// The converged contact, plus the full state p08-C's IFT adjoint differentiates.
struct SdfContactResult {
    bool       valid = false;            // true iff converged inside both bands
    bool       penetrating = false;      // true iff (phi_a + phi_b) < 0
    math::Vec3 point{};                  // converged WORLD contact point
    math::Vec3 normal{};                 // WORLD contact normal (unit), a<-b sense
    float      penetration = 0.0f;       // depth = max(-(phi_a+phi_b), 0)

    // --- p08-C handoff: the stationarity-residual state at convergence -------
    float      phi_a = 0.0f;             // phi_a(point)
    float      phi_b = 0.0f;             // phi_b(point)
    math::Vec3 grad_a_world{};           // world-frame grad of phi_a at point
    math::Vec3 grad_b_world{};           // world-frame grad of phi_b at point
    float      residual = 0.0f;          // ||grad_a + grad_b|| at convergence
    float      step_curvature = 0.0f;    // kappa used in the final step (dr/dp proxy)
    uint32_t   iterations = 0u;          // iterations actually run (<= max)
};

// Iterate p (WORLD frame) to minimize phi_a(p)+phi_b(p) starting from
// `initial_guess_world`. Damped gradient/Newton descent: step = -(grad_a+grad_b)
// / kappa, with kappa a curvature proxy and the step magnitude clamped to
// `step_clamp_scale * min(voxel_a, voxel_b)`. If either SDF leaves its band
// (sample returns kOutsideBand), the step is rejected (halved) so a garbage 1e30
// gradient is never fed into the update; persistent out-of-band -> valid=false.
//
// NOTE the function is curvature-explicit: it records the kappa it used so the
// p08-C IFT adjoint applies dr/dp = kappa*I (the local quadratic model) without
// re-deriving it. The grad-based descent is the discretized Newton step on the
// summed field whose stationary point IS the contact witness.
//
// Defined inline (header-only HD) for the SAME reason sparse_sdf_sample is: ONE
// definition compiled by BOTH g++ (host accuracy/D1 tests call it directly) and
// nvcc (the emission kernel in sdf_contact.cu wraps it). No non-HD math helper
// is used -- SdfBodyFrame bakes the rotation, all vector ops are Vec3 HD-constexpr
// plus sqrtf (HD).
inline NUKA_SDF_HD SdfContactResult find_sdf_contact_newton(
    const runtime::sdf::SparseSdfDevice& sdf_a,
    const SdfBodyFrame& frame_a,
    const runtime::sdf::SparseSdfDevice& sdf_b,
    const SdfBodyFrame& frame_b,
    math::Vec3 initial_guess_world,
    const SdfContactParams& params = {}) {
    SdfContactResult out;

    // The Newton step is clamped to the SMALLER body's voxel so the thinner
    // panel governs (R-E thin-shell): a body cannot leap across a sub-voxel slab.
    const float voxel_a = sdf_a.voxel_size > 0.0f ? sdf_a.voxel_size : 1.0e30f;
    const float voxel_b = sdf_b.voxel_size > 0.0f ? sdf_b.voxel_size : 1.0e30f;
    const float min_voxel = voxel_a < voxel_b ? voxel_a : voxel_b;
    if (min_voxel >= 1.0e30f) {
        return out;  // neither body has a usable SDF
    }
    const float max_step = params.step_clamp_scale * min_voxel;

    // Curvature proxy for the damped step: SDFs are ~unit-gradient, so the
    // summed-field curvature along the descent direction is O(1/voxel). kappa =
    // 2/min_voxel gives a step ~ min_voxel for an O(1) residual -- naturally near
    // the clamp -- and is recorded for the p08-C IFT (dr/dp ~= kappa * I).
    const float kappa = 2.0f / min_voxel;

    math::Vec3 p = initial_guess_world;
    math::Vec3 last_good_p = initial_guess_world;  // last iterate in BOTH bands
    math::Vec3 grad_a_w{};
    math::Vec3 grad_b_w{};
    float phi_a = runtime::sdf::SparseSdfDevice::kOutsideBand;
    float phi_b = runtime::sdf::SparseSdfDevice::kOutsideBand;
    float residual = 0.0f;
    bool in_band = false;
    uint32_t iter = 0u;

    for (iter = 0u; iter < params.max_iterations; ++iter) {
        math::Vec3 grad_a_local{};
        math::Vec3 grad_b_local{};
        const float fa = runtime::sdf::sparse_sdf_sample(
            sdf_a, frame_a.WorldToLocal(p), grad_a_local);
        const float fb = runtime::sdf::sparse_sdf_sample(
            sdf_b, frame_b.WorldToLocal(p), grad_b_local);

        // Out-of-band guard: the gradient that comes back with kOutsideBand is
        // garbage; never feed it into a step. Retreat halfway toward the last
        // iterate that WAS in both bands and retry (deterministic line search).
        const bool a_ok = fa < runtime::sdf::SparseSdfDevice::kOutsideBand * 0.5f;
        const bool b_ok = fb < runtime::sdf::SparseSdfDevice::kOutsideBand * 0.5f;
        if (!a_ok || !b_ok) {
            if (!in_band) {
                return out;  // initial guess not in both bands -> no contact here
            }
            p = (p + last_good_p) * 0.5f;
            continue;
        }

        phi_a = fa;
        phi_b = fb;
        grad_a_w = frame_a.DirLocalToWorld(grad_a_local);
        grad_b_w = frame_b.DirLocalToWorld(grad_b_local);
        in_band = true;
        last_good_p = p;

        // Stationarity residual r = grad_a + grad_b (== 0 at the witness).
        const math::Vec3 r = grad_a_w + grad_b_w;
        residual = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        if (residual <= params.grad_tolerance) {
            ++iter;
            break;
        }

        // Damped Newton step along -r, clamped to <= max_step (thin-shell).
        math::Vec3 step = r * (-1.0f / kappa);
        const float step_len = sqrtf(step.x * step.x + step.y * step.y + step.z * step.z);
        if (step_len > max_step && step_len > 0.0f) {
            step = step * (max_step / step_len);
        }
        p = p + step;
    }

    if (!in_band) {
        return out;
    }

    out.valid = true;
    out.point = p;
    out.phi_a = phi_a;
    out.phi_b = phi_b;
    out.grad_a_world = grad_a_w;
    out.grad_b_world = grad_b_w;
    out.residual = residual;
    out.step_curvature = kappa;
    out.iterations = iter;

    // Contact normal -- the SEPARATION direction for body A (the engine's row
    // convention: jacobian_a.linear == the direction A must move to separate,
    // i.e. A's OUTWARD-of-B normal). grad_a is A's surface OUTWARD normal, which
    // at the witness points from A's interior INTO B -- the opposite of the
    // separation direction. So the separation normal is -grad_a == grad_b - grad_a
    // (the two grads are anti-parallel at the witness, so grad_b - grad_a ~= 2*grad_b
    // ~= -2*grad_a). VERIFIED against SolveUnitGroundContact: box A on ground B
    // has grad_a = -Y (box bottom outward), grad_b = +Y (ground top outward);
    // grad_b - grad_a = +Y == the +Y jacobian_a that separates the box upward.
    //
    // Dihedral >= 90 deg degenerate case: if the gradients are (near) ALIGNED
    // instead of opposed, grad_b - grad_a can collapse -- fall back to -grad_a.
    math::Vec3 ndir = grad_b_w - grad_a_w;
    float nlen = sqrtf(ndir.x * ndir.x + ndir.y * ndir.y + ndir.z * ndir.z);
    if (nlen < 1.0e-6f) {
        ndir = grad_a_w * -1.0f;  // deterministic fallback: A's separation normal
        nlen = sqrtf(ndir.x * ndir.x + ndir.y * ndir.y + ndir.z * ndir.z);
    }
    if (nlen < 1.0e-12f) {
        out.normal = math::Vec3{0.0f, 1.0f, 0.0f};  // last-resort fixed axis
    } else {
        out.normal = ndir * (1.0f / nlen);
    }

    const float sum = phi_a + phi_b;
    out.penetrating = sum < 0.0f;
    out.penetration = sum < 0.0f ? -sum : 0.0f;
    return out;
}

// --------------------------------------------------------------------------
// Device emission (sdf_contact.cu) + host row builders
// --------------------------------------------------------------------------

// One broadphase candidate piece-pair to test. `sdf_index_*` index a device
// array of SparseSdfDevice views built from the nk::Model sdf_* tables
// (sdf_grids/sdf_cell_{keys,values,gradients}, uploaded by Model::UploadTo --
// the former SdfDeviceWorld upload duties moved INTO Model, COL-5/M11); the
// frames carry each body's world<->local transform, baked on the host.
struct SdfContactCandidate {
    uint32_t body_a = 0u;
    uint32_t body_b = 0u;
    uint32_t sdf_index_a = 0u;
    uint32_t sdf_index_b = 0u;
    SdfBodyFrame frame_a{};
    SdfBodyFrame frame_b{};
    math::Vec3 initial_guess_world{};
};

#if defined(__CUDACC__)
// One thread per candidate; writes results[i] for candidate i (slot-indexed, no
// atomics, fixed order -> D1). Declared only under nvcc.
__global__ void find_sdf_contact_kernel(
    const SdfContactCandidate* __restrict__ candidates,
    const runtime::sdf::SparseSdfDevice* __restrict__ sdf_views,
    SdfContactResult* __restrict__ results,
    SdfContactParams params,
    uint32_t candidate_count);
#endif

// Emit a RigidSDFContactRow (id 4) normal + 2 friction rows for one penetrating
// witness into `out_rows` (mirrors row_builder.cpp::AppendContactGroup). Solved
// by the existing RowKind::Contact PGS path. Returns the number of rows emitted
// (0 if the contact is not a valid penetration). Host-side.
uint32_t AppendSdfContactGroup(uint32_t body_a,
                               uint32_t body_b,
                               const SdfContactResult& contact,
                               float friction,
                               float restitution,
                               constraint::RowBuffers* out_rows);

// Emit a FeatherstoneSDFContactRow (id 5) normal row: the maximal side carries
// the contact normal, the articulation side carries the supplied CHAIN Jacobian
// evaluated at the contact point and held for the row (IFT-at-convergence). The
// minimal correct construction for p08-B; full articulated separation dynamics
// are exercised elsewhere. Returns rows emitted. Host-side.
uint32_t AppendFeatherstoneSdfContactGroup(uint32_t maximal_body,
                                           uint32_t articulation_body,
                                           const SdfContactResult& contact,
                                           const math::Vec3& chain_jacobian_linear,
                                           const math::Vec3& chain_jacobian_angular,
                                           float friction,
                                           float restitution,
                                           constraint::RowBuffers* out_rows);

} // namespace nuka::collision
