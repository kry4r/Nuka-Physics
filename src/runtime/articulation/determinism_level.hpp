#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu -- runtime determinism level (p01-W4)
// ---------------------------------------------------------------------------
//
// The determinism level selected once at world / solver creation:
//   Strong (D1, the default): BIT-EXACT across runs and across replicas (same
//     input + same GPU + same driver -> byte-identical output). No float
//     atomics; fixed loop/grid order.
//   Weak   (D2): the reserved escape hatch for a future atomic fast-path. There
//     is currently NO atomic-beneficial hotspot, so D2 today selects the SAME
//     kernels as D1 and is behaviorally identical. D2 is NOT held to the D1
//     bit-exact bar -- a future atomic variant is free to differ.
//
// The plain uint8_t underlying type matches the C ABI nuka_world_desc_t
// .determinism, so a zero-initialized desc maps to the Strong/default path.
//
// Kept in the nuka::runtime::gpu namespace (its historical home) so the
// engine-wide `using nuka::runtime::gpu::DeterminismLevel;` re-exports (e.g.
// diffsim/sparse_solver_backend.hpp) and every transitive consumer
// (gmres/minres/sparse-CG backends, ift_runner) resolve unchanged.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::runtime::gpu {

enum class DeterminismLevel : uint8_t { Strong = 0, Weak = 1 };

} // namespace nuka::runtime::gpu
