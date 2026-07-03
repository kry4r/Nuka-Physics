// ---------------------------------------------------------------------------
// MLS-MPM continuum step (APIC; Hu et al. 2018) — the umbrella MpmStep op.
//
// ONE registered op whose body iterates a substep loop of file-local kernels:
//   clear -> P2G (mass + APIC momentum + constitutive node force) -> grid update
//   (gravity + momentum->velocity normalize + static-plane BC) -> G2P (recover
//   v_p + affine C_p; advect x_p; write the final particle_vel) -> F-update.
// The medium couples to the floor entirely through the static-plane grid BC; the
// G2P velocity IS the particle's final velocity (no dv-compose — the row path's).
//
// Determinism: particles are radix-sorted by env-offset MPM cell key, then each
// grid node sums its 3^3-stencil particle contributions in the sorted order. The
// __fadd_rn pins the accumulation ORDER (run-to-run deterministic); inner products
// stay FMA-contractible (compile-time deterministic). The grid is env-private
// (env-offset node keys) so replicated envs never cross-couple.
//
// The constitutive P(F) (fixed-corotated / Neo-Hookean elastic) is computed from a
// self-written deterministic 3x3 SVD (fixed iteration count, no data-dependent
// branch) so the node force is D1.
// ---------------------------------------------------------------------------

#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"  // chain-J helper + device state
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/sdf_types.cuh"     // SdfRotate / SdfInverseTransformPoint
#include "phi/op_schema.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"       // SparseSdfDevice / sparse_sdf_sample

namespace nuka::phi {

namespace {

namespace m = ::nuka::math;
constexpr uint32_t kBlockSize = 128u;

// Smallest node mass the momentum->velocity divide is float-stable for. A node
// reached only by denormal-tiny stencil-weight products (a particle sitting at an
// exact cell center rounds one axis weight to ~eps^2) otherwise divides by a
// denormal -> inf/NaN velocity that G2P then spreads. Such a node holds no matter.
constexpr float kMinNodeMass = 1.0e-30f;

// MLS-MPM material-table row stride (f32 per row). Must match
// MpmMaterial::kValueCount: youngs poisson density dp_friction dp_cohesion
// model_kind | bulk_modulus tait_gamma viscosity.
constexpr uint32_t kMpmMatStride = 9u;

// Round up to the 256B section alignment the Arena lays scratch out at, so every
// sub-region of mpm_sort_scratch is 256B-aligned device memory.
constexpr uint64_t kScratchAlign = 256u;
inline __host__ uint64_t AlignScratch(uint64_t v) {
    return (v + (kScratchAlign - 1u)) & ~(kScratchAlign - 1u);
}

// 256B-aligned partition of mpm_sort_scratch [cub temp | keys-out | idx-out].
// Sizes the segment (World construct) and partitions it (the op) identically.
struct MpmSortScratchLayout {
    uint64_t temp_bytes = 0u;     // cub radix-sort temp-storage region size.
    uint64_t keys_off   = 0u;     // byte offset of the sorted-keys out buffer.
    uint64_t idx_off    = 0u;     // byte offset of the sorted-idx out buffer.
    uint64_t total      = 0u;     // full segment byte size.
    explicit MpmSortScratchLayout(uint32_t particle_count) {
        const int n = static_cast<int>(particle_count);
        size_t sort_bytes = 0u;
        (void)cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
            nullptr, sort_bytes, static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), n);
        temp_bytes = sort_bytes;
        const uint64_t nbytes = static_cast<uint64_t>(particle_count) * sizeof(uint32_t);
        keys_off = AlignScratch(temp_bytes);
        idx_off  = AlignScratch(keys_off + nbytes);
        total    = AlignScratch(idx_off + nbytes);
    }
};

// ===========================================================================
// File-local 3x3 helpers (row-major float[9]; the particle_F/C packing) and a
// deterministic SVD/polar for the constitutive stress.
// ===========================================================================

// C = A * B (row-major 3x3).
__device__ __forceinline__ void Mat3Mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = __fadd_rn(__fadd_rn(A[r * 3 + 0] * B[0 * 3 + c],
                                               A[r * 3 + 1] * B[1 * 3 + c]),
                                     A[r * 3 + 2] * B[2 * 3 + c]);
}

// C = A * B^T (row-major 3x3).
__device__ __forceinline__ void Mat3MulT(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = __fadd_rn(__fadd_rn(A[r * 3 + 0] * B[c * 3 + 0],
                                               A[r * 3 + 1] * B[c * 3 + 1]),
                                     A[r * 3 + 2] * B[c * 3 + 2]);
}

__device__ __forceinline__ float Mat3Det(const float* F) {
    return F[0] * (F[4] * F[8] - F[5] * F[7]) -
           F[1] * (F[3] * F[8] - F[5] * F[6]) +
           F[2] * (F[3] * F[7] - F[4] * F[6]);
}

// Transposed inverse F^{-T} (row-major). Returns false (and leaves out unset) for a
// near-singular F so the caller can fall back to a zero stress (degenerate cell).
__device__ __forceinline__ bool Mat3InvTranspose(const float* F, float* out, float det) {
    if (fabsf(det) < 1e-12f) return false;
    const float inv = 1.0f / det;
    // Cofactor matrix C; F^{-1} = C^T/det, so F^{-T} = C/det.
    out[0] = (F[4] * F[8] - F[5] * F[7]) * inv;
    out[1] = (F[5] * F[6] - F[3] * F[8]) * inv;
    out[2] = (F[3] * F[7] - F[4] * F[6]) * inv;
    out[3] = (F[2] * F[7] - F[1] * F[8]) * inv;
    out[4] = (F[0] * F[8] - F[2] * F[6]) * inv;
    out[5] = (F[1] * F[6] - F[0] * F[7]) * inv;
    out[6] = (F[1] * F[5] - F[2] * F[4]) * inv;
    out[7] = (F[2] * F[3] - F[0] * F[5]) * inv;
    out[8] = (F[0] * F[4] - F[1] * F[3]) * inv;
    return true;
}

// One symmetric Jacobi rotation zeroing S(p,q): rotate columns p,q of S and
// accumulate into V (the eigenvectors). Fixed-angle, no data-dependent branch on
// magnitude (a sub-tolerance off-diagonal yields a ~identity rotation), so the
// sweep count is constant -> the decomposition is bit-reproducible (D1).
__device__ __forceinline__ void JacobiRotate(float* S, float* V, int p, int q) {
    const float spq = S[p * 3 + q];
    if (spq == 0.0f) return;
    const float spp = S[p * 3 + p], sqq = S[q * 3 + q];
    const float theta = (sqq - spp) / (2.0f * spq);
    const float sign = theta >= 0.0f ? 1.0f : -1.0f;
    const float t = sign / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    const float c = 1.0f / sqrtf(t * t + 1.0f);
    const float s = t * c;
    for (int k = 0; k < 3; ++k) {
        const float sik = S[k * 3 + p], siq = S[k * 3 + q];
        S[k * 3 + p] = c * sik - s * siq;
        S[k * 3 + q] = s * sik + c * siq;
    }
    for (int k = 0; k < 3; ++k) {
        const float skp = S[p * 3 + k], skq = S[q * 3 + k];
        S[p * 3 + k] = c * skp - s * skq;
        S[q * 3 + k] = s * skp + c * skq;
        const float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
        V[k * 3 + p] = c * vkp - s * vkq;
        V[k * 3 + q] = s * vkp + c * vkq;
    }
}

// Deterministic 3x3 SVD F = U * diag(sig) * V^T. Symmetric-eigen on A = F^T F via a
// fixed 8-sweep Jacobi (V = eigenvectors, sig^2 = eigenvalues), then U = F V / sig.
// A near-zero singular value falls back to the corresponding V column so U stays a
// rotation (the cell is degenerate; the stress there is ~0).
__device__ __forceinline__ void Svd3(const float* F, float* U, float* sig, float* V) {
    float A[9];
    // A := F^T F (symmetric); its eigenvectors are the right singular vectors V.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A[r * 3 + c] = __fadd_rn(__fadd_rn(F[0 * 3 + r] * F[0 * 3 + c],
                                               F[1 * 3 + r] * F[1 * 3 + c]),
                                     F[2 * 3 + r] * F[2 * 3 + c]);
    for (int k = 0; k < 9; ++k) V[k] = (k % 4 == 0) ? 1.0f : 0.0f;  // V = I.
    for (int sweep = 0; sweep < 8; ++sweep) {
        JacobiRotate(A, V, 0, 1);
        JacobiRotate(A, V, 0, 2);
        JacobiRotate(A, V, 1, 2);
    }
    float s2[3] = {A[0], A[4], A[8]};
    // Floor the singular values so volumetric stress stays bounded as J -> 0.
    for (int i = 0; i < 3; ++i) sig[i] = fmaxf(sqrtf(fmaxf(s2[i], 0.0f)), 0.05f);
    // U columns = F * V_col / sig (fall back to V_col when sig ~ 0).
    for (int c = 0; c < 3; ++c) {
        float fc[3];
        for (int r = 0; r < 3; ++r)
            fc[r] = F[r * 3 + 0] * V[0 * 3 + c] + F[r * 3 + 1] * V[1 * 3 + c] +
                    F[r * 3 + 2] * V[2 * 3 + c];
        if (sig[c] > 1e-8f) {
            const float inv = 1.0f / sig[c];
            for (int r = 0; r < 3; ++r) U[r * 3 + c] = fc[r] * inv;
        } else {
            for (int r = 0; r < 3; ++r) U[r * 3 + c] = V[r * 3 + c];
        }
    }
    // Reflect (not just rotate): if det(U) < 0 flip the smallest-magnitude singular
    // value so R = U V^T is the closest proper rotation (handles inverted elements).
    if (Mat3Det(U) < 0.0f) {
        int kmin = 0;
        if (fabsf(sig[1]) < fabsf(sig[kmin])) kmin = 1;
        if (fabsf(sig[2]) < fabsf(sig[kmin])) kmin = 2;
        for (int r = 0; r < 3; ++r) U[r * 3 + kmin] = -U[r * 3 + kmin];
        sig[kmin] = -sig[kmin];
    }
}

// First Piola-Kirchhoff stress P(F). Fixed-corotated (Stomakhin 2012):
// P = 2*mu*(F - R) + lambda*(J-1)*J*F^{-T}, R = U V^T from the SVD. Neo-Hookean
// (model_kind==2 elastic alternative): P = mu*(F - F^{-T}) + lambda*log(J)*F^{-T}.
// Row-major float[9]; mu/lambda are the Lame moduli.
__device__ __forceinline__ void FirstPiola(const float* F, float mu, float lambda,
                                           float model_kind, float* P) {
    const float J = Mat3Det(F);
    float FinvT[9];
    const bool ok = Mat3InvTranspose(F, FinvT, J);
    if (!ok) { for (int k = 0; k < 9; ++k) P[k] = 0.0f; return; }
    if (model_kind > 1.5f) {  // Neo-Hookean elastic (kind 2).
        const float lj = logf(fmaxf(J, 1e-8f));
        for (int k = 0; k < 9; ++k)
            P[k] = mu * (F[k] - FinvT[k]) + lambda * lj * FinvT[k];
        return;
    }
    float U[9], sig[3], V[9], R[9];
    Svd3(F, U, sig, V);
    Mat3MulT(U, V, R);  // R = U * V^T (proper rotation).
    const float coef = lambda * (J - 1.0f) * J;
    for (int k = 0; k < 9; ++k)
        P[k] = 2.0f * mu * (F[k] - R[k]) + coef * FinvT[k];
}

// Elastic Hencky-strain cap: sand's recoverable strain is a few percent; anything
// past it is absorbed plastically (grain rearrangement). This BOUNDS the stress a
// degenerate rim/impact F can emit (the fluid J-floor's philosophy), so a
// near-massless boundary node never receives a runaway kick.
constexpr float kSandHenckyCap = 0.15f;

// Granular Kirchhoff stress (Klar et al. 2016, "Drucker-Prager Elastoplasticity for
// Sand Animation"): Hencky (log-strain) St.-Venant-Kirchhoff elasticity on the stored
// elastic F. tau = U diag(2*mu*eps + lambda*tr(eps)) U^T, eps_i = ln(sig_i). This IS
// the P*F^T the MLS node force wants; the reflection sign in U cancels in the product.
__device__ __forceinline__ void GranularKirchhoff(const float* F, float mu,
                                                  float lambda, float* stress) {
    float U[9], sig[3], V[9];
    Svd3(F, U, sig, V);
    float eps[3];
    for (int i = 0; i < 3; ++i) {
        eps[i] = logf(fmaxf(fabsf(sig[i]), 1.0e-6f));
        eps[i] = fminf(fmaxf(eps[i], -kSandHenckyCap), kSandHenckyCap);
    }
    const float tr = eps[0] + eps[1] + eps[2];
    float tau[3];
    for (int i = 0; i < 3; ++i) tau[i] = 2.0f * mu * eps[i] + lambda * tr;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            stress[r * 3 + c] = U[r * 3 + 0] * tau[0] * U[c * 3 + 0] +
                                U[r * 3 + 1] * tau[1] * U[c * 3 + 1] +
                                U[r * 3 + 2] * tau[2] * U[c * 3 + 2];
}

// Drucker-Prager return map on the trial elastic F (Klar et al. 2016, Box 3). Projects
// the principal Hencky strains onto the yield cone: expansion past the cohesion apex ->
// stress-free apex (discards the volumetric plastic strain, the standard sand model, no
// hardening/volume-correction here), shear past the cone -> radial return, inside ->
// elastic. friction_deg = internal friction angle (deg), cohesion = a stress that
// shifts the apex into tension. cohesion 0 recovers the cohesionless sand map.
__device__ __forceinline__ void SandReturnMap(float* F, float mu, float lambda,
                                             float friction_deg, float cohesion) {
    float U[9], sig[3], V[9];
    Svd3(F, U, sig, V);
    float eps[3], sgn[3];
    for (int i = 0; i < 3; ++i) {
        sgn[i] = sig[i] < 0.0f ? -1.0f : 1.0f;
        eps[i] = logf(fmaxf(fabsf(sig[i]), 1.0e-6f));
    }
    const float tr = eps[0] + eps[1] + eps[2];
    const float kappa = 3.0f * lambda + 2.0f * mu;             // d*lambda + 2*mu, d=3.
    const float c0 = kappa > 1.0e-9f ? cohesion / kappa : 0.0f;  // apex tensile strain.
    float dev[3];
    for (int i = 0; i < 3; ++i) dev[i] = eps[i] - tr * (1.0f / 3.0f);
    const float devn = sqrtf(dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]);
    const float sinp = sinf(friction_deg * 0.017453292519943295f);
    const float alpha = 1.632993161855452f * sinp / fmaxf(3.0f - sinp, 1.0e-6f);
    const float tr_shift = tr - c0;
    float en[3];
    if (devn < 1.0e-12f || tr_shift > 0.0f) {
        for (int i = 0; i < 3; ++i) en[i] = c0 * (1.0f / 3.0f);  // return to the apex.
    } else {
        const float dgamma = devn + (kappa / (2.0f * mu)) * tr_shift * alpha;
        if (dgamma <= 0.0f) {
            for (int i = 0; i < 3; ++i) en[i] = eps[i];           // inside the cone.
        } else {
            const float inv = 1.0f / devn;                        // radial return.
            for (int i = 0; i < 3; ++i) en[i] = eps[i] - dgamma * dev[i] * inv;
        }
    }
    float s2[3];
    for (int i = 0; i < 3; ++i) {
        // Cap the STORED elastic strain: the overflow is plastic densification, so
        // the state the next substep stresses can never spiral (bounded restoring).
        en[i] = fminf(fmaxf(en[i], -kSandHenckyCap), kSandHenckyCap);
        s2[i] = sgn[i] * expf(en[i]);
    }
    float US[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) US[r * 3 + c] = U[r * 3 + c] * s2[c];
    Mat3MulT(US, V, F);  // F = US * V^T = U diag(s2) V^T.
}

// Quadratic B-spline base node + the 3 per-axis weights. The particle sits in the
// stencil [base, base+2]; fx is the particle offset from base in cell units.
struct Bspline {
    int32_t base;
    float   w[3];
};
__device__ __forceinline__ Bspline QuadWeights(float gx) {
    // base = floor(gx - 0.5); fx in [0.5, 1.5) is the offset from base.
    Bspline b;
    b.base = static_cast<int32_t>(floorf(gx - 0.5f));
    const float fx = gx - static_cast<float>(b.base);
    b.w[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
    const float d = fx - 1.0f;
    b.w[1] = 0.75f - d * d;
    b.w[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);
    return b;
}

// Per-env grid node id from integer node coords (env-offset for env-private grids).
__device__ __forceinline__ int64_t NodeId(uint32_t env, int32_t ix, int32_t iy,
                                          int32_t iz, const uint32_t dims[3],
                                          uint32_t nodes_per_env) {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= static_cast<int32_t>(dims[0]) ||
        iy >= static_cast<int32_t>(dims[1]) || iz >= static_cast<int32_t>(dims[2])) {
        return -1;
    }
    const uint32_t local = (static_cast<uint32_t>(iz) * dims[1] +
                            static_cast<uint32_t>(iy)) * dims[0] +
                           static_cast<uint32_t>(ix);
    return static_cast<int64_t>(env) * nodes_per_env + local;
}

// --- grid clear -------------------------------------------------------------
__global__ void MpmGridClearKernel(uint32_t total_nodes, float* mass,
                                   m::Vec3* momentum, m::Vec3* velocity,
                                   m::Vec3* force) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;
    mass[i] = 0.0f;
    momentum[i] = m::Vec3::Zero();
    velocity[i] = m::Vec3::Zero();
    force[i] = m::Vec3::Zero();
}

// Clear THIS op's grid-escape status bit per env (order-independent across ops).
__global__ void MpmClearEscapeBitKernel(uint32_t* env_status, uint32_t env_count) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= env_count) return;
    env_status[e] &= ~kEnvStatusMpmGridEscape;
}

// Zero the per-body reaction probes once per step so they hold the impulse summed
// over THIS step's substeps (linear = balance diagnostic; angular = the deposit
// torque the per-articulation M^-1 J^T pass consumes).
__global__ void MpmClearBodyReactionKernel(uint32_t total_bodies, m::Vec3* reaction,
                                          m::Vec3* ang_reaction) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_bodies) return;
    reaction[b] = m::Vec3::Zero();
    if (ang_reaction != nullptr) ang_reaction[b] = m::Vec3::Zero();
}

// --- cell key (the particle's base-node cell, env-offset) -------------------
// Map an MPM-slice thread t in [0, mpm_count) to (env, global particle index). The
// MPM slice is [0, mpm_per_env) inside each Ppe-strided env, so global = env*Ppe +
// (t - env*mpm_per_env). For a lone MPM medium mpm_per_env == Ppe -> global == t
// (byte-identical to the old contiguous [0, Np) launch).
__device__ __forceinline__ uint32_t MpmSliceGlobal(uint32_t t, uint32_t mpm_per_env,
                                                   uint32_t ppe, uint32_t& env_out) {
    const uint32_t env = t / mpm_per_env;
    env_out = env;
    return env * ppe + (t - env * mpm_per_env);
}

__global__ void MpmCellKeysKernel(uint32_t mpm_count,
                                  const m::Vec3* __restrict__ pos,
                                  uint32_t particles_per_env, uint32_t mpm_per_env,
                                  float inv_dx,
                                  m::Vec3 origin, uint32_t dims_x, uint32_t dims_y,
                                  uint32_t dims_z, uint32_t cells_per_env,
                                  uint32_t* __restrict__ keys,
                                  uint32_t* __restrict__ idx,
                                  uint32_t* __restrict__ env_status) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mpm_count) return;
    uint32_t env = 0u;
    const uint32_t p = MpmSliceGlobal(t, mpm_per_env, particles_per_env, env);
    const m::Vec3 xp = pos[p];
    const float gx = (xp.x - origin.x) * inv_dx;
    const float gy = (xp.y - origin.y) * inv_dx;
    const float gz = (xp.z - origin.z) * inv_dx;
    const int32_t bx = static_cast<int32_t>(floorf(gx - 0.5f));
    const int32_t by = static_cast<int32_t>(floorf(gy - 0.5f));
    const int32_t bz = static_cast<int32_t>(floorf(gz - 0.5f));
    // Walled x/y faces + plane floor contain a pressed particle (flag only a base off
    // the grid there); the open +z top flags a stencil clip (base+2 past the ceiling).
    const bool nonfinite = !isfinite(gx) || !isfinite(gy) || !isfinite(gz);
    const int64_t bx64 = bx, by64 = by, bz64 = bz;
    const bool escaped = nonfinite ||
                         bx64 < 0 || bx64 >= static_cast<int64_t>(dims_x) ||
                         by64 < 0 || by64 >= static_cast<int64_t>(dims_y) ||
                         bz64 < 0 || bz64 + 2 >= static_cast<int64_t>(dims_z);
    if (escaped && env_status != nullptr) {
        atomicOr(&env_status[env], kEnvStatusMpmGridEscape);
    }
    const int32_t cx = bx < 0 ? 0 : (bx >= static_cast<int32_t>(dims_x) ?
                                     static_cast<int32_t>(dims_x) - 1 : bx);
    const int32_t cy = by < 0 ? 0 : (by >= static_cast<int32_t>(dims_y) ?
                                     static_cast<int32_t>(dims_y) - 1 : by);
    const int32_t cz = bz < 0 ? 0 : (bz >= static_cast<int32_t>(dims_z) ?
                                     static_cast<int32_t>(dims_z) - 1 : bz);
    const uint32_t local = (static_cast<uint32_t>(cz) * dims_y +
                            static_cast<uint32_t>(cy)) * dims_x +
                           static_cast<uint32_t>(cx);
    keys[t] = env * cells_per_env + local;
    idx[t] = p;
}

// --- P2G deterministic gather (one thread per grid node) --------------------
// Node i sums the contributions of every particle whose 3^3 stencil includes i,
// i.e. particles whose base cell lies in [i-2, i] per axis. The sorted particle
// stream (by env-offset cell key) is scanned per candidate cell in a fixed order
// with __fadd_rn -> bit-reproducible run-to-run (NO atomics). Accumulates mass +
// APIC momentum + the constitutive node force -N_i*V0*(4/dx^2)*P(F)*F^T*(x_i-x_p).
__global__ void MpmP2GGatherKernel(uint32_t total_nodes,
                                   const m::Vec3* __restrict__ pos,
                                   const float* __restrict__ inv_mass,
                                   const m::Vec3* __restrict__ vel,
                                   const float* __restrict__ part_C,
                                   const float* __restrict__ part_F,
                                   const float* __restrict__ part_vol0,
                                   const uint32_t* __restrict__ part_mat,
                                   const float* __restrict__ material_table,
                                   uint32_t material_count,
                                   const uint32_t* __restrict__ sorted_keys,
                                   const uint32_t* __restrict__ sorted_idx,
                                   uint32_t particle_count, uint32_t particles_per_env,
                                   uint32_t nodes_per_env, uint32_t cells_per_env,
                                   uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
                                   float inv_dx, float dx, float dt, m::Vec3 origin,
                                   float* __restrict__ grid_mass,
                                   m::Vec3* __restrict__ grid_momentum) {
    const uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= total_nodes) return;
    const uint32_t env = node / nodes_per_env;
    const uint32_t local = node % nodes_per_env;
    const int32_t nx = static_cast<int32_t>(local % dims_x);
    const int32_t ny = static_cast<int32_t>((local / dims_x) % dims_y);
    const int32_t nz = static_cast<int32_t>(local / (dims_x * dims_y));
    const m::Vec3 xi = m::Vec3{origin.x + nx * dx, origin.y + ny * dx,
                               origin.z + nz * dx};
    const float stress_scale = 4.0f * inv_dx * inv_dx;  // W^{-1} = 4/dx^2 (quadratic).
    float mass = 0.0f;
    m::Vec3 mom = m::Vec3::Zero();
    // Scan the 3^3 base cells that can place a particle's stencil on this node.
    for (int32_t dz = -2; dz <= 0; ++dz) {
        const int32_t cz = nz + dz;
        if (cz < 0 || cz >= static_cast<int32_t>(dims_z)) continue;
        for (int32_t dy = -2; dy <= 0; ++dy) {
            const int32_t cy = ny + dy;
            if (cy < 0 || cy >= static_cast<int32_t>(dims_y)) continue;
            for (int32_t dx_i = -2; dx_i <= 0; ++dx_i) {
                const int32_t cx = nx + dx_i;
                if (cx < 0 || cx >= static_cast<int32_t>(dims_x)) continue;
                const uint32_t ck = (static_cast<uint32_t>(cz) * dims_y +
                                     static_cast<uint32_t>(cy)) * dims_x +
                                    static_cast<uint32_t>(cx);
                const uint32_t key = env * cells_per_env + ck;
                // Binary-search the lower bound of `key` in the sorted stream.
                uint32_t lo = 0u, hi = particle_count;
                while (lo < hi) {
                    const uint32_t mid = lo + ((hi - lo) >> 1);
                    if (sorted_keys[mid] < key) lo = mid + 1u; else hi = mid;
                }
                for (uint32_t s = lo; s < particle_count && sorted_keys[s] == key; ++s) {
                    const uint32_t p = sorted_idx[s];
                    const float wp = (inv_mass[p] > 0.0f) ? (1.0f / inv_mass[p]) : 0.0f;
                    if (wp <= 0.0f) continue;
                    const m::Vec3 xp = pos[p];
                    const Bspline wxs = QuadWeights((xp.x - origin.x) * inv_dx);
                    const Bspline wys = QuadWeights((xp.y - origin.y) * inv_dx);
                    const Bspline wzs = QuadWeights((xp.z - origin.z) * inv_dx);
                    const int32_t ox = nx - wxs.base, oy = ny - wys.base,
                                  oz = nz - wzs.base;
                    if (ox < 0 || ox > 2 || oy < 0 || oy > 2 || oz < 0 || oz > 2) continue;
                    const float w = wxs.w[ox] * wys.w[oy] * wzs.w[oz];
                    const m::Vec3 vp = vel[p];
                    // APIC affine term C_p * (x_i - x_p) (row-major 3x3 in part_C).
                    const m::Vec3 dpos = xi - xp;
                    const float* C = part_C + static_cast<size_t>(p) * 9u;
                    const m::Vec3 cterm = m::Vec3{
                        C[0] * dpos.x + C[1] * dpos.y + C[2] * dpos.z,
                        C[3] * dpos.x + C[4] * dpos.y + C[5] * dpos.z,
                        C[6] * dpos.x + C[7] * dpos.y + C[8] * dpos.z};
                    const float wm = w * wp;
                    mass = __fadd_rn(mass, wm);
                    mom.x = __fadd_rn(mom.x, wm * (vp.x + cterm.x));
                    mom.y = __fadd_rn(mom.y, wm * (vp.y + cterm.y));
                    mom.z = __fadd_rn(mom.z, wm * (vp.z + cterm.z));
                    // Constitutive node-momentum impulse dt * f_i folded into momentum,
                    // f_i = -w * V0 * (4/dx^2) * P(F) F^T (x_i - x_p).
                    if (part_F != nullptr && part_vol0 != nullptr && dt > 0.0f) {
                        const float vol0 = part_vol0[p];
                        if (vol0 > 0.0f) {
                            const uint32_t mid = part_mat != nullptr ? part_mat[p] : 0u;
                            float youngs = 0.0f, poisson = 0.0f, kind = 0.0f;
                            float bulk = 0.0f, tait_gamma = 0.0f, visc = 0.0f;
                            if (material_table != nullptr && mid < material_count) {
                                const float* mr = material_table +
                                                  static_cast<size_t>(mid) * kMpmMatStride;
                                youngs = mr[0]; poisson = mr[1]; kind = mr[5];
                                bulk = mr[6]; tait_gamma = mr[7]; visc = mr[8];
                            }
                            const float* F = part_F + static_cast<size_t>(p) * 9u;
                            float stress[9];
                            if (kind > 3.5f) {  // granular Drucker-Prager (kind 4).
                                // Hencky elastic Kirchhoff stress off the stored elastic
                                // F; the DP plastic return is applied in the F-update.
                                const float denom =
                                    (1.0f + poisson) * (1.0f - 2.0f * poisson);
                                const float mu = youngs / (2.0f * (1.0f + poisson));
                                const float lambda =
                                    (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
                                GranularKirchhoff(F, mu, lambda, stress);
                            } else if (kind > 2.5f) {  // weakly-compressible fluid (kind 3).
                                // Kirchhoff stress J*sigma = -p*J*I, Tait EOS, no tension;
                                // optional viscosity J*mu*(C+C^T) from the live affine C.
                                // J is floored in the F-update, so the unclamped Tait
                                // pressure stays finite and the restoring force is strong.
                                const float J = Mat3Det(F);
                                const float pr = fmaxf(
                                    bulk * (powf(J, -tait_gamma) - 1.0f), 0.0f);
                                const float diag = -pr * J;
                                stress[0] = diag; stress[1] = 0.0f; stress[2] = 0.0f;
                                stress[3] = 0.0f; stress[4] = diag; stress[5] = 0.0f;
                                stress[6] = 0.0f; stress[7] = 0.0f; stress[8] = diag;
                                if (visc > 0.0f) {
                                    const float Jv = J * visc;
                                    stress[0] += Jv * 2.0f * C[0];
                                    stress[4] += Jv * 2.0f * C[4];
                                    stress[8] += Jv * 2.0f * C[8];
                                    const float s01 = Jv * (C[1] + C[3]);
                                    const float s02 = Jv * (C[2] + C[6]);
                                    const float s12 = Jv * (C[5] + C[7]);
                                    stress[1] += s01; stress[3] += s01;
                                    stress[2] += s02; stress[6] += s02;
                                    stress[5] += s12; stress[7] += s12;
                                }
                            } else {  // fixed-corotated / Neo-Hookean elastic.
                                const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
                                const float mu = youngs / (2.0f * (1.0f + poisson));
                                const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
                                float P[9];
                                FirstPiola(F, mu, lambda, kind, P);
                                Mat3MulT(P, F, stress);  // P * F^T (Cauchy-stress-like).
                            }
                            const float coef = -w * vol0 * stress_scale;
                            mom.x = __fadd_rn(mom.x, dt * coef * (stress[0] * dpos.x +
                                              stress[1] * dpos.y + stress[2] * dpos.z));
                            mom.y = __fadd_rn(mom.y, dt * coef * (stress[3] * dpos.x +
                                              stress[4] * dpos.y + stress[5] * dpos.z));
                            mom.z = __fadd_rn(mom.z, dt * coef * (stress[6] * dpos.x +
                                              stress[7] * dpos.y + stress[8] * dpos.z));
                        }
                    }
                }
            }
        }
    }
    grid_mass[node] = mass;
    grid_momentum[node] = mom;
}

// --- grid update: gravity kick + momentum->velocity normalize + static-plane BC -
// One thread per node. m*v_hat = m*v + dt*(m*g) (the constitutive impulse was
// folded into momentum in P2G); normalize; then project velocity against the static
// floor plane (no-penetration normal + Coulomb friction). The plane is static so
// its surface velocity is zero -> no grid->body reaction (the dynamic body is later).
__global__ void MpmGridUpdateKernel(uint32_t total_nodes, uint32_t nodes_per_env,
                                    uint32_t dims_x, uint32_t dims_y, float dx,
                                    m::Vec3 origin, m::Vec3 gravity, float dt,
                                    m::Vec3 plane_n, float plane_d, float plane_mu,
                                    const float* __restrict__ mass,
                                    const m::Vec3* __restrict__ momentum,
                                    m::Vec3* __restrict__ velocity) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;
    const float mi = mass[i];
    if (mi < kMinNodeMass) { velocity[i] = m::Vec3::Zero(); return; }
    m::Vec3 v = momentum[i] * (1.0f / mi) + gravity * dt;  // symplectic Euler kick.
    // Static-plane BC: a node at/below the plane gets a no-penetration normal cut +
    // Coulomb friction on the tangential component against the removed normal impulse.
    const uint32_t local = i % nodes_per_env;
    const int32_t nx = static_cast<int32_t>(local % dims_x);
    const int32_t ny = static_cast<int32_t>((local / dims_x) % dims_y);
    const int32_t nz = static_cast<int32_t>(local / (dims_x * dims_y));
    const m::Vec3 xi{origin.x + nx * dx, origin.y + ny * dx, origin.z + nz * dx};
    const float sd = plane_n.x * xi.x + plane_n.y * xi.y + plane_n.z * xi.z - plane_d;
    if (sd <= 0.0f) {
        const float vn = v.Dot(plane_n);
        if (vn < 0.0f) {  // moving into the plane: remove the inward normal component.
            const m::Vec3 vt = v - plane_n * vn;       // tangential velocity.
            const float vt_len = sqrtf(vt.LengthSq());
            const float coulomb = plane_mu * (-vn);    // friction budget.
            if (vt_len <= coulomb || vt_len < 1e-8f) {
                v = m::Vec3::Zero();                   // sticks (static friction).
            } else {
                v = vt * (1.0f - coulomb / vt_len);    // dynamic friction drag.
            }
        }
    }
    // Separating domain-wall BC on the OUTERMOST x/y boundary node ring: cut only the
    // OUTWARD-normal velocity (a node never pushed past the grid box). A no-op for any
    // scene whose mass stays >= 2 cells off the x/y faces (a massless ring node returns
    // above); the elastic gates' >= 4*dx margin keeps those nodes massless.
    const int32_t hx = static_cast<int32_t>(dims_x) - 1;
    const int32_t hy = static_cast<int32_t>(dims_y) - 1;
    if (nx == 0 && v.x < 0.0f) v.x = 0.0f;
    else if (nx == hx && v.x > 0.0f) v.x = 0.0f;
    if (ny == 0 && v.y < 0.0f) v.y = 0.0f;
    else if (ny == hy && v.y > 0.0f) v.y = 0.0f;
    velocity[i] = v;
}

// Shape-table lane 7 (sdf_grid; ~0u when the shape has no cooked SDF), canonical
// row stride. Mirrors the narrowphase reader so the body BC samples the same grids.
namespace sdfq = ::nuka::runtime::sdf;
__forceinline__ __device__ uint32_t ShapeSdfGrid(const float* table, uint32_t row) {
    return __float_as_uint(
        table[static_cast<size_t>(row) * nuka::phi::kShapeTableRowStride + 7u]);
}
// Shape-table lane 5 (contype; 0 => not a contact collider).
__forceinline__ __device__ uint32_t ShapeContype(const float* table, uint32_t row) {
    return __float_as_uint(
        table[static_cast<size_t>(row) * nuka::phi::kShapeTableRowStride + 5u]);
}
// Load one SDF grid view from the Model sdf_* tables (header floats + flat cells).
__forceinline__ __device__ sdfq::SparseSdfDevice LoadGrid(
    const float* headers, const uint32_t* counts, const uint64_t* keys,
    const float* values, const m::Vec3* grads, uint32_t grid) {
    const float* h = headers + static_cast<size_t>(grid) * nuka::phi::kSdfHeaderStride;
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

// Project the node velocity onto the surface velocity of its deepest-covering
// body; record dp = m*(v_after-v_before) + the owner for the reaction gather.
__global__ void MpmGridBodyProjectKernel(
    uint32_t total_nodes, uint32_t nodes_per_env, uint32_t dims_x, uint32_t dims_y,
    float dx, m::Vec3 origin, uint32_t bodies_per_env, float body_mu, float band,
    const m::Transform* __restrict__ body_pose,
    const m::Vec3* __restrict__ body_lin_vel,
    const m::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ shape_table, const float* __restrict__ sdf_headers,
    const uint32_t* __restrict__ sdf_cell_count,
    const uint64_t* __restrict__ sdf_keys, const float* __restrict__ sdf_values,
    const m::Vec3* __restrict__ sdf_grads, const float* __restrict__ mass,
    m::Vec3* __restrict__ velocity, m::Vec3* __restrict__ body_dp,
    uint32_t* __restrict__ body_owner, uint32_t* __restrict__ env_status) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;
    body_dp[i] = m::Vec3::Zero();
    body_owner[i] = ~0u;
    const float mi = mass[i];
    if (mi < kMinNodeMass) return;
    const uint32_t env = i / nodes_per_env;
    const uint32_t local = i % nodes_per_env;
    const int32_t nx = static_cast<int32_t>(local % dims_x);
    const int32_t ny = static_cast<int32_t>((local / dims_x) % dims_y);
    const int32_t nz = static_cast<int32_t>(local / (dims_x * dims_y));
    const m::Vec3 xi{origin.x + nx * dx, origin.y + ny * dx, origin.z + nz * dx};
    // Pick the body whose SDF places this node deepest inside its band (most
    // negative phi) -> a single deterministic owner per node.
    uint32_t best_body = ~0u;
    float best_phi = band;
    m::Vec3 best_n = m::Vec3::Zero();
    for (uint32_t bl = 0u; bl < bodies_per_env; ++bl) {
        const uint32_t grid = ShapeSdfGrid(shape_table, bl);
        if (grid == ~0u) {
            // Collidable body with no cooked SDF imposes no grid BC: raise a bit.
            if (ShapeContype(shape_table, bl) != 0u && env_status != nullptr)
                atomicOr(&env_status[env], kEnvStatusMpmOneWayBody);
            continue;
        }
        const m::Transform xf = body_pose[env * bodies_per_env + bl];
        const m::Vec3 q = nkops::SdfInverseTransformPoint(xf, xi);
        const sdfq::SparseSdfDevice sg = LoadGrid(sdf_headers, sdf_cell_count,
                                                  sdf_keys, sdf_values, sdf_grads, grid);
        m::Vec3 grad{0.0f, 0.0f, 0.0f};
        const float phi = sdfq::sparse_sdf_sample(sg, q, grad);
        if (phi >= sdfq::SparseSdfDevice::kOutsideBand || phi >= best_phi) continue;
        const m::Vec3 gw = nkops::SdfRotate(xf.rotation, grad);
        const float gl = sqrtf(gw.LengthSq());
        if (gl < 1.0e-8f) continue;
        best_phi = phi;
        best_body = env * bodies_per_env + bl;
        best_n = gw * (1.0f / gl);
    }
    if (best_body == ~0u) return;
    // Body surface velocity at the node: v_b + w_b x (xi - body_origin).
    const m::Vec3 xb = body_pose[best_body].position;
    const m::Vec3 vb = (body_lin_vel != nullptr) ? body_lin_vel[best_body]
                                                  : m::Vec3::Zero();
    const m::Vec3 wb = (body_ang_vel != nullptr) ? body_ang_vel[best_body]
                                                  : m::Vec3::Zero();
    const m::Vec3 r = xi - xb;
    const m::Vec3 v_surf = vb + wb.Cross(r);
    const m::Vec3 v_before = velocity[i];
    m::Vec3 v_rel = v_before - v_surf;
    const float vn = v_rel.Dot(best_n);
    if (vn < 0.0f) {  // approaching the body surface: project + friction.
        v_rel = v_rel - best_n * vn;            // tangential remainder.
        const float vt_len = sqrtf(v_rel.LengthSq());
        const float coulomb = body_mu * (-vn);
        if (vt_len <= coulomb || vt_len < 1.0e-8f) {
            v_rel = m::Vec3::Zero();            // sticks (static friction).
        } else {
            v_rel = v_rel * (1.0f - coulomb / vt_len);
        }
        const m::Vec3 v_after = v_surf + v_rel;
        velocity[i] = v_after;
        body_dp[i] = (v_after - v_before) * mi;
        body_owner[i] = best_body;
    }
}

// Per-body deterministic gather of -dp / -(x-xb)xdp over the nodes it owns. A
// free-rigid body applies the impulse inline; an articulation-link body (cooked
// inv_mass==0) stores the linear+angular reaction for the per-articulation deposit.
__global__ void MpmGridBodyReactKernel(
    uint32_t total_bodies, uint32_t bodies_per_env, uint32_t nodes_per_env,
    uint32_t dims_x, uint32_t dims_y, float dx, m::Vec3 origin,
    const m::Transform* __restrict__ body_pose,
    const float* __restrict__ body_inv_mass,
    const m::Vec3* __restrict__ body_inv_inertia,
    const uint32_t* __restrict__ body_to_link,
    const m::Vec3* __restrict__ body_dp, const uint32_t* __restrict__ body_owner,
    m::Vec3* __restrict__ body_lin_vel, m::Vec3* __restrict__ body_ang_vel,
    m::Vec3* __restrict__ body_reaction, m::Vec3* __restrict__ body_ang_reaction) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_bodies) return;
    const float im = body_inv_mass[b];
    const bool is_link = (body_to_link != nullptr) && (body_to_link[b] != ~0u);
    // A free-rigid immovable body (static ground/wall) imposes no reaction; a link
    // row is cooked inv_mass==0 yet DOES react through its articulation, so it runs.
    if (im <= 0.0f && !is_link) return;
    const uint32_t env = b / bodies_per_env;
    const m::Vec3 xb = body_pose[b].position;
    const uint64_t base = static_cast<uint64_t>(env) * nodes_per_env;
    m::Vec3 dp_sum = m::Vec3::Zero();   // sum of node momentum changes this body caused.
    m::Vec3 tq_sum = m::Vec3::Zero();   // sum of (x-xb) x dp.
    for (uint32_t local = 0u; local < nodes_per_env; ++local) {
        const uint64_t node = base + local;
        if (body_owner[node] != b) continue;
        const m::Vec3 dp = body_dp[node];
        const int32_t nx = static_cast<int32_t>(local % dims_x);
        const int32_t ny = static_cast<int32_t>((local / dims_x) % dims_y);
        const int32_t nz = static_cast<int32_t>(local / (dims_x * dims_y));
        const m::Vec3 xi{origin.x + nx * dx, origin.y + ny * dx, origin.z + nz * dx};
        const m::Vec3 r = xi - xb;
        dp_sum.x = __fadd_rn(dp_sum.x, dp.x);
        dp_sum.y = __fadd_rn(dp_sum.y, dp.y);
        dp_sum.z = __fadd_rn(dp_sum.z, dp.z);
        const m::Vec3 tq = r.Cross(dp);
        tq_sum.x = __fadd_rn(tq_sum.x, tq.x);
        tq_sum.y = __fadd_rn(tq_sum.y, tq.y);
        tq_sum.z = __fadd_rn(tq_sum.z, tq.z);
    }
    // Reaction = -dp (equal-and-opposite); dp is the per-substep momentum, so it
    // IS the substep impulse (no dt scale).
    const m::Vec3 lin_impulse = dp_sum * (-1.0f);
    const m::Vec3 ang_impulse = tq_sum * (-1.0f);
    if (!is_link) {  // the solve_rows.cu free-rigid apply form.
        body_lin_vel[b].x += lin_impulse.x * im;
        body_lin_vel[b].y += lin_impulse.y * im;
        body_lin_vel[b].z += lin_impulse.z * im;
        if (body_ang_vel != nullptr && body_inv_inertia != nullptr) {
            const m::Vec3 ii = body_inv_inertia[b];
            body_ang_vel[b].x += ang_impulse.x * ii.x;
            body_ang_vel[b].y += ang_impulse.y * ii.y;
            body_ang_vel[b].z += ang_impulse.z * ii.z;
        }
    }
    if (body_reaction != nullptr) {  // accumulate the linear impulse (balance probe).
        body_reaction[b].x += lin_impulse.x;
        body_reaction[b].y += lin_impulse.y;
        body_reaction[b].z += lin_impulse.z;
    }
    // A link body's torque impulse feeds the per-articulation M^-1 J^T deposit.
    if (is_link && body_ang_reaction != nullptr) {
        body_ang_reaction[b].x += ang_impulse.x;
        body_ang_reaction[b].y += ang_impulse.y;
        body_ang_reaction[b].z += ang_impulse.z;
    }
}

// One thread per ARTICULATION. Loops its link bodies in ascending global-row order,
// chain-walks each link's grid reaction wrench into the generalized force g, sums,
// applies M^-1 once, and seeds qdot_flat[tile] += M^-1 J^T wrench. Single writer per
// tile (Go2's 4 feet share one tile) -> race-free + deterministic (fixed order).
__global__ void MpmArticReactDepositKernel(
    nkops::ArticulationDeviceState state, uint32_t artic_count, uint32_t artics_per_env,
    uint32_t bodies_per_env, uint32_t base_link_count, uint32_t max_dof,
    const m::Transform* __restrict__ body_pose,
    const uint32_t* __restrict__ body_to_link,
    const m::Vec3* __restrict__ body_reaction,
    const m::Vec3* __restrict__ body_ang_reaction,
    const float* __restrict__ m_inv, float* __restrict__ qdot_flat) {
    const uint32_t ag = blockIdx.x * blockDim.x + threadIdx.x;
    if (ag >= artic_count) return;
    if (max_dof == 0u || max_dof > nkops::kMaxArticulationDof) return;
    const uint32_t ape = (artics_per_env == 0u) ? 1u : artics_per_env;
    const uint32_t env = ag / ape;
    float g[nkops::kMaxArticulationDof];
    for (uint32_t k = 0u; k < max_dof; ++k) g[k] = 0.0f;
    // Sum each owned link's J^T wrench. Ascending body row -> deterministic g.
    const uint32_t row0 = env * bodies_per_env;
    for (uint32_t lb = 0u; lb < bodies_per_env; ++lb) {
        const uint32_t tmpl_link = body_to_link[lb];
        if (tmpl_link == ~0u) continue;                 // free rigid / static row.
        const uint32_t gl_link = env * base_link_count + tmpl_link;
        if (gl_link >= state.total_link_count) continue;
        // Derive offset from the link's OWN articulation (the assemble_rows.cu
        // reference), then skip if it is not this thread's tile -> race-free.
        const uint32_t articulation = state.link_to_articulation[gl_link];
        if (articulation != ag) continue;
        const uint32_t b = row0 + lb;
        const m::Vec3 f = body_reaction[b];
        const m::Vec3 tau = body_ang_reaction[b];
        if (f.x == 0.0f && f.y == 0.0f && f.z == 0.0f &&
            tau.x == 0.0f && tau.y == 0.0f && tau.z == 0.0f) continue;
        const uint32_t offset = state.articulation_link_offset[articulation];
        const m::Vec3 point = body_pose[b].position;  // == the react gather anchor.
        // Bounded root-walk: a chain is at most total_link_count deep, and every link
        // is range-checked, so a malformed parent map can never spin the GPU.
        uint32_t link = gl_link;
        for (uint32_t depth = 0u; link != ~0u && depth < state.total_link_count; ++depth) {
            if (link >= state.total_link_count) break;
            if (nkops::JointDofCountDevice(state.joint_type[link]) != 0u) {
                const uint32_t dof_index =
                    nkops::LocalDofIndexDevice(state, offset, link);
                nkops::AccumulateChainJointForce(state, articulation, link, dof_index,
                                                 max_dof, point, f, tau, g);
            }
            const uint32_t parent_local = state.parent_link[link];
            link = (parent_local == ~0u) ? ~0u : (offset + parent_local);
        }
    }
    // delta-qdot = M^-1 g (ascending c, plain +=); seed the per-artic flat tile.
    const size_t tile = static_cast<size_t>(ag) * max_dof * max_dof;
    for (uint32_t r = 0u; r < max_dof; ++r) {
        const float* Minv = m_inv + tile + static_cast<size_t>(r) * max_dof;
        float acc = 0.0f;
        for (uint32_t c = 0u; c < max_dof; ++c) acc += Minv[c] * g[c];
        qdot_flat[static_cast<size_t>(ag) * max_dof + r] += acc;
    }
}

// --- G2P gather (one thread per particle; race-free, naturally D1) ----------
// v_p = sum_i N_i v_i; C_p = (4/dx^2) sum_i N_i v_i (x_i - x_p)^T (the MLS fit).
// Advects x_p += dt*v_p and writes the recovered v_p DIRECTLY into particle_vel
// (the MPM particle's final velocity — no finalize dv-compose).
__global__ void MpmG2PGatherKernel(uint32_t mpm_count,
                                   uint32_t particles_per_env, uint32_t mpm_per_env,
                                   uint32_t nodes_per_env,
                                   uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
                                   float inv_dx, float dx, float dt, m::Vec3 origin,
                                   const float* __restrict__ inv_mass,
                                   const m::Vec3* __restrict__ grid_velocity,
                                   m::Vec3* __restrict__ part_pos,
                                   m::Vec3* __restrict__ part_vel,
                                   float* __restrict__ part_C) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mpm_count) return;
    uint32_t env = 0u;
    const uint32_t p = MpmSliceGlobal(t, mpm_per_env, particles_per_env, env);
    const m::Vec3 xp = part_pos[p];
    const Bspline wxs = QuadWeights((xp.x - origin.x) * inv_dx);
    const Bspline wys = QuadWeights((xp.y - origin.y) * inv_dx);
    const Bspline wzs = QuadWeights((xp.z - origin.z) * inv_dx);
    m::Vec3 vp = m::Vec3::Zero();
    float C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t dims[3] = {dims_x, dims_y, dims_z};
    for (int32_t a = 0; a < 3; ++a) {
        for (int32_t b = 0; b < 3; ++b) {
            for (int32_t c = 0; c < 3; ++c) {
                const int32_t ix = wxs.base + a, iy = wys.base + b, iz = wzs.base + c;
                const int64_t id = NodeId(env, ix, iy, iz, dims, nodes_per_env);
                if (id < 0) continue;
                const float w = wxs.w[a] * wys.w[b] * wzs.w[c];
                const m::Vec3 vi = grid_velocity[static_cast<size_t>(id)];
                vp.x = __fadd_rn(vp.x, w * vi.x);
                vp.y = __fadd_rn(vp.y, w * vi.y);
                vp.z = __fadd_rn(vp.z, w * vi.z);
                const m::Vec3 xi = m::Vec3{origin.x + ix * dx, origin.y + iy * dx,
                                           origin.z + iz * dx};
                const m::Vec3 d = xi - xp;
                C[0] = __fadd_rn(C[0], w * vi.x * d.x); C[1] = __fadd_rn(C[1], w * vi.x * d.y); C[2] = __fadd_rn(C[2], w * vi.x * d.z);
                C[3] = __fadd_rn(C[3], w * vi.y * d.x); C[4] = __fadd_rn(C[4], w * vi.y * d.y); C[5] = __fadd_rn(C[5], w * vi.y * d.z);
                C[6] = __fadd_rn(C[6], w * vi.z * d.x); C[7] = __fadd_rn(C[7], w * vi.z * d.y); C[8] = __fadd_rn(C[8], w * vi.z * d.z);
            }
        }
    }
    const float scale = 4.0f * inv_dx * inv_dx;
    float* Cd = part_C + static_cast<size_t>(p) * 9u;
    for (int32_t k = 0; k < 9; ++k) Cd[k] = C[k] * scale;
    // Advect only a movable particle (a pinned inv_mass==0 sample holds position).
    if (inv_mass != nullptr && inv_mass[p] <= 0.0f) { part_vel[p] = vp; return; }
    m::Vec3 np = xp + vp * dt;
    // Separating domain walls at the grid-box faces (x/y both sides + z floor; +z stays
    // open). Clamp position and drop the outward velocity so nothing leaks or injects.
    const float xlo = origin.x + dx, xhi = origin.x + (dims_x - 2) * dx;
    const float ylo = origin.y + dx, yhi = origin.y + (dims_y - 2) * dx;
    const float zlo = origin.z + dx;
    if (np.x < xlo) { np.x = xlo; vp.x = fmaxf(vp.x, 0.0f); }
    else if (np.x > xhi) { np.x = xhi; vp.x = fminf(vp.x, 0.0f); }
    if (np.y < ylo) { np.y = ylo; vp.y = fmaxf(vp.y, 0.0f); }
    else if (np.y > yhi) { np.y = yhi; vp.y = fminf(vp.y, 0.0f); }
    if (np.z < zlo) { np.z = zlo; vp.z = fmaxf(vp.z, 0.0f); }
    part_vel[p] = vp;
    part_pos[p] = np;
}

// --- F-update: elastic F^{n+1} = (I + dt*C) F^n; fluid (kind 3) tracks volume off the
// divergence J *= (1 + dt*tr C), keeping F = cbrt(J)*I (no shear-driven inversion);
// granular (kind 4) predicts the elastic F then Drucker-Prager return-maps it (plastic).
__global__ void MpmUpdateFKernel(uint32_t mpm_count, float dt,
                                 const float* __restrict__ part_C,
                                 const uint32_t* __restrict__ part_mat,
                                 const float* __restrict__ material_table,
                                 uint32_t material_count, uint32_t particles_per_env,
                                 uint32_t mpm_per_env,
                                 float* __restrict__ part_F,
                                 uint32_t* __restrict__ env_status) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mpm_count) return;
    uint32_t env_unused = 0u;
    const uint32_t p = MpmSliceGlobal(t, mpm_per_env, particles_per_env, env_unused);
    const float* C = part_C + static_cast<size_t>(p) * 9u;
    float* F = part_F + static_cast<size_t>(p) * 9u;
    float kind = 0.0f, youngs = 0.0f, poisson = 0.0f, dpf = 0.0f, dpc = 0.0f;
    if (part_mat != nullptr && material_table != nullptr) {
        const uint32_t mid = part_mat[p];
        if (mid < material_count) {
            const float* mr = material_table + static_cast<size_t>(mid) * kMpmMatStride;
            youngs = mr[0]; poisson = mr[1]; dpf = mr[3]; dpc = mr[4]; kind = mr[5];
        }
    }
    if (kind > 3.5f) {  // granular: elastic predictor then Drucker-Prager return map.
        float IpdtC[9];
        for (int k = 0; k < 9; ++k) IpdtC[k] = dt * C[k];
        IpdtC[0] += 1.0f; IpdtC[4] += 1.0f; IpdtC[8] += 1.0f;
        float Fn[9];
        for (int k = 0; k < 9; ++k) Fn[k] = F[k];
        Mat3Mul(IpdtC, Fn, F);  // trial elastic F = (I + dt*C) * F^n.
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        SandReturnMap(F, mu, lambda, dpf, dpc);
    } else if (kind > 2.5f) {  // fluid: volume-only update J *= (1 + dt*tr C), F = cbrt(J)*I.
        // Shear leaves a fluid's volume unchanged, so track J off the divergence tr(C);
        // no det of the full affine map => no shear-driven inversion at a hard impact.
        const float Jraw = Mat3Det(F) * (1.0f + dt * (C[0] + C[4] + C[8]));
        if (Jraw <= 0.0f && env_status != nullptr)
            atomicOr(&env_status[p / particles_per_env], kEnvStatusMpmGridEscape);
        // Floor/ceil J so an explicit overshoot stays finite and self-recovers (water
        // is near-incompressible; J leaves [0.83,1.01] only on a transient overshoot).
        const float s = cbrtf(fminf(fmaxf(Jraw, 0.3f), 3.0f));
        for (int k = 0; k < 9; ++k) F[k] = (k % 4 == 0) ? s : 0.0f;
    } else {  // elastic: F^{n+1} = (I + dt*C) F^n.
        float IpdtC[9];
        for (int k = 0; k < 9; ++k) IpdtC[k] = dt * C[k];
        IpdtC[0] += 1.0f; IpdtC[4] += 1.0f; IpdtC[8] += 1.0f;
        float Fn[9];
        for (int k = 0; k < 9; ++k) Fn[k] = F[k];
        Mat3Mul(IpdtC, Fn, F);  // (I + dt*C) * F^n.
    }
}

// Partition the pre-allocated scratch [cub temp | keys-out | idx-out] (host-side,
// so the sort never cudaMalloc/syncs mid-capture; the gather joins the graph).
struct MpmScratch {
    void* sort_temp = nullptr;
    uint32_t* keys_out = nullptr;
    uint32_t* idx_out = nullptr;
    size_t sort_temp_bytes = 0u;
};
inline MpmScratch PartitionScratch(void* base, uint32_t Np) {
    const MpmSortScratchLayout sl(Np);
    char* sbase = reinterpret_cast<char*>(base);
    MpmScratch s;
    s.sort_temp = sbase;
    s.keys_out = reinterpret_cast<uint32_t*>(sbase + sl.keys_off);
    s.idx_out  = reinterpret_cast<uint32_t*>(sbase + sl.idx_off);
    s.sort_temp_bytes = static_cast<size_t>(sl.temp_bytes);
    return s;
}

// One MLS-MPM substep: clear -> P2G(force) -> grid update + BC -> G2P -> F-update,
// each kernel launch a barrier (the launch boundary is the inter-phase barrier).
void LaunchSubstep(const MpmStepParams& p, const ModelView& model,
                   const DataView& data, float dt_sub, uint32_t Np, uint32_t Ppe,
                   uint32_t mpm_pe, uint32_t mpm_count,
                   uint32_t cpe, uint32_t total_nodes, float inv_dx,
                   const m::Vec3& origin, cudaStream_t stream) {
    const uint32_t nblocks = (total_nodes + kBlockSize - 1u) / kBlockSize;
    // The particle-iterating kernels + the sort span the MPM sub-slice only
    // (mpm_count == Np for a lone MPM medium, so this is byte-identical there).
    const uint32_t pblocks = (mpm_count + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(MpmGridClearKernel, dim3(nblocks), dim3(kBlockSize), 0u, stream,
               total_nodes, data.grid_mass, data.grid_momentum, data.grid_velocity,
               data.grid_force);
    LaunchCuda(MpmCellKeysKernel, dim3(pblocks), dim3(kBlockSize), 0u, stream,
               mpm_count, data.particle_pos, Ppe, mpm_pe, inv_dx, origin,
               p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], cpe,
               data.mpm_grid_cell_key, data.mpm_grid_part_idx, data.env_status);
    MpmScratch sc = PartitionScratch(data.mpm_sort_scratch, mpm_count);
    (void)cub::DeviceRadixSort::SortPairs(
        sc.sort_temp, sc.sort_temp_bytes, data.mpm_grid_cell_key, sc.keys_out,
        data.mpm_grid_part_idx, sc.idx_out, static_cast<int>(mpm_count), 0, 32, stream);
    LaunchCuda(MpmP2GGatherKernel, dim3(nblocks), dim3(kBlockSize), 0u, stream,
               total_nodes, data.particle_pos, data.particle_inv_mass,
               data.particle_vel, data.particle_C, data.particle_F,
               data.particle_vol0, data.particle_material_id, data.mpm_material_table,
               p.material_count, sc.keys_out, sc.idx_out, mpm_count, Ppe, p.nodes_per_env,
               cpe, p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], inv_dx, p.dx,
               dt_sub, origin, data.grid_mass, data.grid_momentum);
    const m::Vec3 g{p.gravity[0], p.gravity[1], p.gravity[2]};
    const m::Vec3 pn{p.plane_n[0], p.plane_n[1], p.plane_n[2]};
    LaunchCuda(MpmGridUpdateKernel, dim3(nblocks), dim3(kBlockSize), 0u, stream,
               total_nodes, p.nodes_per_env, p.grid_dims[0], p.grid_dims[1], p.dx,
               origin, g, dt_sub, pn, p.plane_d, p.plane_mu, data.grid_mass,
               data.grid_momentum, data.grid_velocity);
    // Dynamic-body grid BC + two-way reaction (between grid-update and G2P). The
    // BITE disables ONLY these kernels (the static-plane BC above stays on).
    if (p.dynamic_body_bc != 0u && p.bite_disable_dynamic_bc == 0u &&
        p.bodies_per_env > 0u) {
        const uint32_t total_bodies = p.bodies_per_env * p.env_count;
        const uint32_t bblocks = (total_bodies + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(MpmGridBodyProjectKernel, dim3(nblocks), dim3(kBlockSize), 0u,
                   stream, total_nodes, p.nodes_per_env, p.grid_dims[0],
                   p.grid_dims[1], p.dx, origin, p.bodies_per_env, p.body_mu,
                   p.body_band, data.body_pose, data.body_linear_velocity,
                   data.body_angular_velocity, model.shape_table, model.sdf_headers,
                   model.sdf_cell_count, model.sdf_cell_keys, model.sdf_cell_values,
                   model.sdf_cell_gradients, data.grid_mass, data.grid_velocity,
                   data.grid_body_dp, data.grid_body_owner, data.env_status);
        LaunchCuda(MpmGridBodyReactKernel, dim3(bblocks), dim3(kBlockSize), 0u,
                   stream, total_bodies, p.bodies_per_env, p.nodes_per_env,
                   p.grid_dims[0], p.grid_dims[1], p.dx, origin,
                   data.body_pose, data.body_inv_mass, data.body_inv_inertia,
                   model.body_to_link, data.grid_body_dp, data.grid_body_owner,
                   data.body_linear_velocity, data.body_angular_velocity,
                   data.mpm_body_reaction, data.mpm_body_ang_reaction);
    }
    LaunchCuda(MpmG2PGatherKernel, dim3(pblocks), dim3(kBlockSize), 0u, stream,
               mpm_count, Ppe, mpm_pe, p.nodes_per_env, p.grid_dims[0], p.grid_dims[1],
               p.grid_dims[2], inv_dx, p.dx, dt_sub, origin, data.particle_inv_mass,
               data.grid_velocity, data.particle_pos, data.particle_vel,
               data.particle_C);
    LaunchCuda(MpmUpdateFKernel, dim3(pblocks), dim3(kBlockSize), 0u, stream, mpm_count,
               dt_sub, data.particle_C, data.particle_material_id,
               data.mpm_material_table, p.material_count, Ppe, mpm_pe, data.particle_F,
               data.env_status);
}

Status OpMpmStep(const ModelView& model, const DataView& data,
                 const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const MpmStepParams*>(params);
    if (p == nullptr) return Status::Failed;
    // Defensive inertness (the build-time add() gate already keeps this op off a
    // non-MPM op list): nothing to do without MPM particles / a grid / a cell size.
    if ((p->mode != kParticleModeMpm && p->mode != kParticleModeMpmXpbd) ||
        p->particle_count == 0u || p->nodes_per_env == 0u || p->dx <= 0.0f) {
        return Status::Ok;
    }
    const uint64_t total_nodes64 =
        static_cast<uint64_t>(p->nodes_per_env) * p->env_count;
    const uint64_t cells_per_env =
        static_cast<uint64_t>(p->grid_dims[0]) * p->grid_dims[1] * p->grid_dims[2];
    if (cells_per_env == 0u) return Status::Ok;
    // LOUD overflow: the env-offset node/cell key + total-body launch must fit u32.
    if (total_nodes64 > 0xFFFFFFFFull) return Status::Failed;
    if (cells_per_env * p->env_count > 0xFFFFFFFFull) return Status::Failed;
    if (static_cast<uint64_t>(p->bodies_per_env) * p->env_count > 0xFFFFFFFFull)
        return Status::Failed;
    const uint32_t Np = p->particle_count;
    const uint32_t Ppe = p->particles_per_env == 0u ? Np : p->particles_per_env;
    // The MPM sub-slice [0, mpm_pe) per env. 0 => the whole per-env block is MPM (a
    // lone MPM medium; then mpm_count == Np, byte-identical to the contiguous launch).
    const uint32_t mpm_pe =
        (p->mpm_particles_per_env == 0u || p->mpm_particles_per_env > Ppe)
            ? Ppe : p->mpm_particles_per_env;
    const uint32_t mpm_count = mpm_pe * p->env_count;
    // LOUD invariants: the count fits the per-env footprint + cub's int num_items.
    if (static_cast<uint64_t>(Np) >
        static_cast<uint64_t>(Ppe) * p->env_count) return Status::Failed;
    if (static_cast<uint64_t>(mpm_count) > static_cast<uint64_t>(INT_MAX))
        return Status::Failed;
    const uint32_t cpe = static_cast<uint32_t>(cells_per_env);
    const uint32_t total_nodes = static_cast<uint32_t>(total_nodes64);
    const float inv_dx = 1.0f / p->dx;
    const m::Vec3 origin{p->grid_origin[0], p->grid_origin[1], p->grid_origin[2]};
    const uint32_t substeps = p->substeps == 0u ? 1u : p->substeps;
    const float dt_sub = p->dt / static_cast<float>(substeps);
    // Clear THIS op's grid-escape bit per env once (the per-substep cell-key kernel
    // ORs into it without re-clearing, so an escape in any substep stays flagged).
    if (data.env_status != nullptr) {
        const uint32_t e = p->env_count == 0u ? 1u : p->env_count;
        const uint32_t eb = (e + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(MpmClearEscapeBitKernel, dim3(eb), dim3(kBlockSize), 0u, stream,
                   data.env_status, e);
    }
    // Zero the per-step reaction probe so it accumulates only this step's substeps.
    if (p->dynamic_body_bc != 0u && p->bodies_per_env > 0u &&
        data.mpm_body_reaction != nullptr) {
        const uint32_t tb = p->bodies_per_env * p->env_count;
        const uint32_t bb = (tb + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(MpmClearBodyReactionKernel, dim3(bb), dim3(kBlockSize), 0u,
                   stream, tb, data.mpm_body_reaction, data.mpm_body_ang_reaction);
    }
    for (uint32_t s = 0; s < substeps; ++s) {
        LaunchSubstep(*p, model, data, dt_sub, Np, Ppe, mpm_pe, mpm_count, cpe,
                      total_nodes, inv_dx, origin, stream);
    }
    // Articulated-link deposit: the link rows' net grid reaction (summed over the
    // substeps above) becomes delta-qdot via M^-1 J^T into the per-articulation
    // qdot_flat tile, which SolveRowsBlockIsland seeds from. Once per Step, after the
    // substep loop (m_inv + the kinematic frame are constant across substeps, and
    // M^-1 is linear, so the summed-wrench deposit equals the per-substep sum).
    if (p->dynamic_body_bc != 0u && p->bite_disable_dynamic_bc == 0u &&
        p->bodies_per_env > 0u && p->artic_count > 0u && p->max_dof > 0u &&
        data.mpm_body_ang_reaction != nullptr && data.qdot_flat != nullptr &&
        data.m_inv != nullptr) {
        const uint32_t total_links = p->base_link_count * p->env_count;
        const nkops::ArticulationDeviceState state =
            nkops::MakeArticulationDeviceState(model, data, total_links, p->artic_count);
        const uint32_t ablocks = (p->artic_count + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(MpmArticReactDepositKernel, dim3(ablocks), dim3(kBlockSize), 0u,
                   stream, state, p->artic_count, p->artics_per_env, p->bodies_per_env,
                   p->base_link_count, p->max_dof, data.body_pose, model.body_to_link,
                   data.mpm_body_reaction, data.mpm_body_ang_reaction, data.m_inv,
                   data.qdot_flat);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

uint64_t MpmSortScratchBytes(uint32_t particle_count) {
    if (particle_count == 0u) return 0u;
    return MpmSortScratchLayout(particle_count).total;
}

void RegisterNkMpmOps() {
    SetCudaOp(NkOp::MpmStep, &OpMpmStep);
}

}  // namespace nuka::phi
