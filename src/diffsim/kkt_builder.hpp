#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- contact KKT (Delassus) builder  (v0.5 p03 R2)
// ---------------------------------------------------------------------------
//
// Assembles the batched per-articulation dense SPD Delassus operator
//
//      A_b = J_b M_b^-1 J_b^T          (contact space, one block per articulation)
//
// straight from the engine's real contact buffers into the locked
// BatchedDenseSpdSystem the self-written CG backend consumes. A is the SPD
// reduction the IFT contact-gradient path back-props through in ONE sparse
// solve; it is BLOCK-DIAGONAL across articulations (a robot's foot contacts
// couple only within that robot), so each articulation contributes one
// independent dense SPD block of dimension n_b <= kMaxFootContactsPerEnv(4)*3
// (normal + two friction rows) = kMaxBlockDim(12).
//
// This is NOT the indefinite saddle [M J^T; J 0]; it is the SPD Schur/Delassus
// reduction in contact space -- the whole reason CG (which only converges on
// SPD systems) suffices in v0.5.
//
// ACTIVE-SET CRITERION (v0.5):
//   A contact slot is ENGAGED iff its assembled normal row is active, i.e.
//       row_type != kRowInactive  AND  depth > 0
//   (depth > 0 == in penetration/contact, the same activity flag the forward
//   detector + assembler use). This MATCHES AssembleArticulatedContactRowsKernel
//   (articulation_contacts.cu): it sets row_type = kRowNormal and row.depth =
//   contact_depth[slot] (> 0 from DetectFootGroundContacts) for every active slot,
//   and leaves row_type = kRowInactive (depth defaulted 0) for invalid/empty slots,
//   so our gate selects exactly the assembler's active set. We gate on DEPTH,
//   never on `lambda_`: lambda is
//   the warm-start impulse and is all-zero on a cold-start build, so a
//   lambda-gate would yield spuriously empty blocks. For an engaged slot we emit
//   THREE equality rows -- the normal row and its two friction-tangent rows --
//   treating the contact as STICKING (within the friction cone). This is the IFT
//   linearization at a sticking contact: the converged solve's tangential
//   velocity is ~0, so the sticking (equality) Jacobian is the correct local
//   constraint set to back-prop through. Sliding / at-the-cone-boundary contacts
//   (where a tangent row should drop to an inequality) are a v0.7+ refinement;
//   see the limitation note in kkt_builder.cu.
//
// ROW ORDERING (deterministic): slots ascending within the articulation, and
// within a slot normal -> tangent1 -> tangent2. n_b is the engaged-row count
// (<= 12). This fixed order makes the gather, the host cross-check, and the D1
// byte-exact gate all agree.
//
// D1: one warp per articulation, fixed-order fp64 accumulation, fp32 store with
// explicit upper-triangle-then-mirror so A is BIT-symmetric, zero float atomics.
// Same input + same GPU -> byte-identical `values`/`block_dim`.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"

#include <cstddef>
#include <cstdint>

namespace nuka::diffsim {

// Alias mirroring the other diffsim headers (step_backward.hpp etc.) so the
// engine's contact row type resolves as `articulation::ArticulatedContactRow`.
namespace articulation = nuka::runtime::articulation;

// Owns the two device buffers a BatchedDenseSpdSystem points at. The returned
// BatchedDenseSpdSystem's raw pointers are borrowed from these buffers, so this
// MUST outlive any use of the system (the buffers must not be moved/destroyed
// while the system is live -- the classic raw-pointer-into-owner contract).
//
// R8: the *_bytes fields track each Buffer*'s allocated size so
// BuildContactDelassusSystem keeps its allocate-once / reuse-if-large-enough
// contract (the opaque v2 Buffer has no Size()). RAII frees the v2 buffers; the
// struct is move-only (it owns device memory), like the legacy RAII-Buffer form.
struct ContactDelassusBuffers {
    phi::Buffer* values = nullptr;     // float[block_count * kMaxBlockDim^2], row-major blocks
    phi::Buffer* block_dim = nullptr;  // uint32_t[block_count], n_b per articulation
    std::size_t values_bytes = 0u;     // allocated size of `values` (R8 reuse guard)
    std::size_t block_dim_bytes = 0u;  // allocated size of `block_dim`

    ContactDelassusBuffers() = default;
    ~ContactDelassusBuffers();
    ContactDelassusBuffers(ContactDelassusBuffers&&) noexcept;
    ContactDelassusBuffers& operator=(ContactDelassusBuffers&&) noexcept;
    ContactDelassusBuffers(const ContactDelassusBuffers&) = delete;
    ContactDelassusBuffers& operator=(const ContactDelassusBuffers&) = delete;
};

// Build the batched dense SPD Delassus operator A_b = J_b M_b^-1 J_b^T from the
// engine's contact buffers.
//
//   rows            : ArticulatedContactRow[block_count * kMaxFootContactsPerEnv]
//                     (env-major, fixed stride kMaxFootContactsPerEnv; block b ==
//                      articulation b owns slots [b*stride, b*stride+stride)).
//   jac_normal      : float[block_count * kMaxFootContactsPerEnv * dof_stride]
//   jac_tangent1    : float[...] (same layout) -- friction-tangent-1 chain rows
//   jac_tangent2    : float[...] (same layout) -- friction-tangent-2 chain rows
//                     Each slot's dof-wide row lives at [slot * dof_stride .. +dof_stride).
//   m_inv           : float[block_count * dof_stride * dof_stride] -- per-
//                     articulation M^-1 tile, row-major, stride dof_stride^2.
//   block_count     : number of articulations / SPD blocks (== env_count).
//   dof_stride      : the M^-1 tile / chain-Jacobian stride (== max_dof,
//                     <= kMaxContactSolverDof). NOTE: this is generally NOT
//                     kMaxBlockDim -- the contraction runs over dof_stride
//                     columns while the OUTPUT block is stored at stride 12.
//
// On return `out_buffers` owns freshly allocated `values` (zeroed in full,
// including all padding and empty blocks, for the D1 memcmp gate) and `block_dim`,
// and `out_system` points at them with out_system.block_count == block_count.
// Enqueues on the context stream; the caller synchronizes (mirrors diffsim).
void BuildContactDelassusSystem(
    const phi::DeviceContext& context,
    const articulation::ArticulatedContactRow* rows,
    const float* jac_normal,
    const float* jac_tangent1,
    const float* jac_tangent2,
    const float* m_inv,
    uint32_t block_count,
    uint32_t dof_stride,
    BatchedDenseSpdSystem& out_system,
    ContactDelassusBuffers& out_buffers);

}  // namespace nuka::diffsim
