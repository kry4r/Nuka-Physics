#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — the op-function dispatch table.
//
// ggml analogy: this is the per-backend op-implementation registry the backend
// `dispatch` consults. Each physics op (M3+) registers an `OpFn` into the slot
// indexed by its NkOp; the CUDA backend's dispatch looks the slot up and calls
// it on the backend stream, returning Status::Unsupported for null slots.
//
// In M1 ALL slots are null (no real op is wired). The smoke gate registers a
// single test op at runtime through SetCudaOp(...) to exercise the dispatch +
// plan + event paths end-to-end.
//
// OpFn signature: ops receive the (forward-declared) ModelView/DataView, the
// op's POD params, and the stream to launch on. In M1 ops never dereference the
// views — only `params` and `stream` are meaningful.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "phi/backend.hpp"     // Status, ModelView, DataView, NkOp
#include "phi/op_schema.hpp"   // NkOp

namespace nuka::phi {

using OpFn = Status (*)(const ModelView&, const DataView&, const void* params, cudaStream_t stream);

// The global op table, one slot per NkOp. Defined in registry.cu.
extern OpFn g_ops[static_cast<int>(NkOp::Count)];

// Look up a slot (nullptr if unregistered).
OpFn GetCudaOp(NkOp op);

// Register / override an op fn at runtime. Returns the previous fn (or nullptr).
// This is the public hook the smoke gate uses to plant its test op; it is also
// the runtime-override seam real ops can use. Idempotent registration of the
// SAME fn is a no-op.
OpFn SetCudaOp(NkOp op, OpFn fn);

// Compile-time registration macro for statically-known op fns. Expands to a
// file-scope initializer that plants `fn` into the `op` slot at static-init.
// (Kept dlopen-friendly: a TU using this still needs to be referenced, see the
// nuka_phi2 self-registration de-risk in src/CMakeLists.txt.)
// The registration variable is named from __COUNTER__ (not from `op`, which is
// a qualified enumerator like ::nuka::phi::NkOp::ApplyDrives and would not
// token-paste into a valid identifier).
#define NK_OP_REG_CONCAT_INNER(a, b) a##b
#define NK_OP_REG_CONCAT(a, b) NK_OP_REG_CONCAT_INNER(a, b)
#define REGISTER_NK_OP(op, fn)                                            \
    namespace {                                                          \
    [[maybe_unused]] const ::nuka::phi::OpFn                             \
        NK_OP_REG_CONCAT(nk_reg_op_, __COUNTER__) =                      \
        (::nuka::phi::SetCudaOp((op), (fn)), (fn));                     \
    } // namespace

} // namespace nuka::phi
