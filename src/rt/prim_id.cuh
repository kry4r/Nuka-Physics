#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- global prim-id packing for the two-level TLAS/BLAS tracer (v0.7
// G1-A). The single-level p13 renderer used a FLAT prim id (the LBVH leaf's
// original index). The two-level tracer hits a (instance, local-prim) PAIR, so
// the prim_id AOV packs both into one uint32.
//
// LAYOUT: instance in the HIGH bits, local prim in the LOW bits. This is the
// D1 HINGE: with instance high, the closest-hit tie-break (RtClosestHitUpdate:
// `t < best || (t == best && packed < best_packed)`) is a TOTAL ORDER ==
// (lowest instance, then lowest local prim). So on an equal-t cross-instance
// tie the WINNER is deterministic regardless of TLAS visit order -> the pixel
// value is order-INDEPENDENT -> D1 by construction (no atomics, one ray/pixel).
//
//   kInstanceBits = 12  -> up to 4096 instances
//   kPrimBits     = 20  -> up to 1,048,576 primitives per unique mesh (BLAS)
//   12 + 20 = 32         -> exactly fills a uint32, no waste.
//
// HD (NUKA_RT_HD): the device kernel AND the host oracle pack/unpack with the
// SAME function so the prim_id AOV is byte-exact host==device.
//
// CAP / GATE-DEBT: BuildTwoLevelScene host-asserts instance_count <= 1<<12 and
// every BLAS prim count <= 1<<20. If p16's real USD mesh exceeds 2^20 prims (or
// the scene exceeds 4096 instances) this is a ONE-CONSTANT rebalance (e.g.
// 11/21). NAMED CONSUMER for the rebalance: p16 (H1 + cup + table + USD scene).
// Sentinel edge (remote): PackPrimId(4095, 0xFFFFF) == 0xFFFFFFFF == kNoPrim;
// unreachable in G1-A test scenes, flagged as gate-debt with consumer p16.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

inline constexpr uint32_t kInstanceBits = 12u;
inline constexpr uint32_t kPrimBits = 20u;
inline constexpr uint32_t kMaxInstances = 1u << kInstanceBits;   // 4096
inline constexpr uint32_t kMaxBlasPrims = 1u << kPrimBits;       // 1,048,576
inline constexpr uint32_t kPrimMask = (1u << kPrimBits) - 1u;    // 0x000FFFFF

// Pack (instance, local prim) into one uint32, instance in the HIGH bits.
NUKA_RT_HD inline uint32_t PackPrimId(uint32_t instance, uint32_t local_prim) {
    return (instance << kPrimBits) | (local_prim & kPrimMask);
}

// Unpack a packed prim id back into (instance, local prim).
NUKA_RT_HD inline void UnpackPrimId(uint32_t packed, uint32_t* instance,
                                    uint32_t* local_prim) {
    *instance = packed >> kPrimBits;
    *local_prim = packed & kPrimMask;
}

} // namespace nuka::rt

#undef NUKA_RT_HD
