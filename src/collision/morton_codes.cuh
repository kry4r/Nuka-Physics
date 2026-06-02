#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- 30-bit Morton code generation (Karras 2012 / LBVH).
//
// Device-only helpers. Pure functions of their inputs (no atomics, no
// thread-order dependence) so they are D1-safe by construction. Included only
// from .cu translation units.
// ---------------------------------------------------------------------------

#include <cstdint>

#include <cuda_runtime.h>

namespace nuka::collision::gpu {

// Expand a 10-bit integer into 30 bits by inserting two zeros between each bit
// (the classic "spread bits by 3" used for 3D Morton interleaving).
__device__ __forceinline__ uint32_t ExpandBits10(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// 30-bit Morton code (10 bits/axis) from a normalized position in [0,1]^3.
// Inputs are clamped to [0,1] then quantized to [0,1023]; matches the Karras
// reference. Deterministic: identical floats -> identical code.
__device__ __forceinline__ uint32_t Morton3D30(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    const uint32_t xx = ExpandBits10(static_cast<uint32_t>(x));
    const uint32_t yy = ExpandBits10(static_cast<uint32_t>(y));
    const uint32_t zz = ExpandBits10(static_cast<uint32_t>(z));
    return (xx << 2) | (yy << 1) | zz;
}

} // namespace nuka::collision::gpu
