#pragma once
// ---------------------------------------------------------------------------
// PHI v2 — Plan opaque handle (ggml_backend graph-plan analogue).
//
// A Plan is a backend-captured, replayable execution of an OpCall list. For the
// CUDA backend (phi/backend_cuda/cuda_backend.cu) it wraps an instantiated
// cudaGraphExec captured via stream-capture; replaying the same Plan is the D1
// determinism vehicle (identical kernel launches in identical order).
//
// The struct is OPAQUE here — its definition lives in the backend. It is
// created/executed/freed through the Backend vtable (BackendI::plan_create /
// plan_execute / plan_free) in phi/backend.hpp; this header just declares the
// handle so headers can name it without pulling in the backend.
//
// PURE C++ — zero CUDA types.
// ---------------------------------------------------------------------------

namespace nuka::phi {

struct Plan;  // opaque; defined in the backend implementation

} // namespace nuka::phi
