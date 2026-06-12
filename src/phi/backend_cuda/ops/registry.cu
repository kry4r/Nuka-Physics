// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — op-function table definition.
//
// All slots start null (M1: no real op wired). GetCudaOp returns the slot;
// SetCudaOp registers/overrides at runtime (the smoke gate's hook). The CUDA
// backend's dispatch (cuda_backend.cu) consults this table and returns
// Status::Unsupported for a null slot.
//
// NOTE: This TU must NOT allocate (lint scope `hot_path_cuda_malloc` covers
// phi/backend_cuda/ops/**). It is pure table bookkeeping — no CUDA allocation.
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/ops/registry.cuh"

namespace nuka::phi {

// Zero-initialized: every slot is null until an op registers itself.
OpFn g_ops[static_cast<int>(NkOp::Count)] = {};

OpFn GetCudaOp(NkOp op) {
    const int idx = static_cast<int>(op);
    if (idx < 0 || idx >= static_cast<int>(NkOp::Count)) {
        return nullptr;
    }
    return g_ops[idx];
}

OpFn SetCudaOp(NkOp op, OpFn fn) {
    const int idx = static_cast<int>(op);
    if (idx < 0 || idx >= static_cast<int>(NkOp::Count)) {
        return nullptr;
    }
    OpFn prev = g_ops[idx];
    g_ops[idx] = fn;
    return prev;
}

} // namespace nuka::phi
