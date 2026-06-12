#pragma once
// ---------------------------------------------------------------------------
// PHI v2 smoke gate — shared test-side declarations.
//
// The VecAdd op's POD params (3 device pointers + element count) and the
// registration entry point, shared between the test driver (phi2_smoke.cpp) and
// the op TU (phi2_smoke_ops.cu).
// ---------------------------------------------------------------------------

namespace nuka::phi_smoke {

// POD params for the test vector-add op: c[i] = a[i] + b[i], i in [0, n).
// Trivially copyable; the device pointers are raw bases obtained from
// phi::BufferBase on the allocated device buffers.
struct VecAddParams {
    const float* a;
    const float* b;
    float*       c;
    int          n;
};

// Plant the VecAdd op into the CUDA backend's op table (NkOp::ApplyDrives slot).
void RegisterVecAddOp();

} // namespace nuka::phi_smoke
