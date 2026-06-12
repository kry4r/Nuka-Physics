#pragma once
// ---------------------------------------------------------------------------
// PHI v2 — Event opaque handle (ggml_backend_event analogue).
//
// An Event is a backend synchronization marker (a cudaEvent for the CUDA
// backend). Created/recorded/waited/freed through the Backend vtable
// (BackendI::event_new / event_record / event_wait / event_free) in
// phi/backend.hpp. The struct is OPAQUE here; its definition lives in the
// backend implementation (phi/backend_cuda/cuda_backend.cu).
//
// PURE C++ — zero CUDA types.
// ---------------------------------------------------------------------------

namespace nuka::phi {

struct Event;  // opaque; defined in the backend implementation

} // namespace nuka::phi
