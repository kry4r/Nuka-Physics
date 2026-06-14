#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::CandidatePair / CandidatePairStream -- the unified broadphase
// OUTPUT type (v0.8 C2a)
// ---------------------------------------------------------------------------
// A CandidatePair names two collidables (by type-tagged CollidableRef) that the
// broadphase says MIGHT touch. It is the SINGLE class-blind hand-off from the
// (rigid LBVH / particle grid / static) broadphase sources to the narrowphase:
// every source emits CandidatePairs, the unified narrowphase consumes them, so
// adding a collidable type does not add a broadphase->narrowphase code path.
//
// CANONICAL STABLE KEY (the deterministic sort key).
//   Each side packs into a 32-bit sideKey:
//       sideKey = (uint32_t(type) << 28) | (handle & 0x0FFFFFFF)
//     - 4 high bits = CollidableType  (so up to 16 types; v0.8 has 4, v0.9 +4)
//     - 28 low bits = handle          (so handle MUST be < 2^28 = 268,435,456)
//   The pair key orders the two sides canonically (min<<32 | max):
//       stable_key = (uint64(min(sideA,sideB)) << 32) | uint64(max(sideA,sideB))
//   Putting TYPE in the high bits of each sideKey means keys NEVER collide
//   across types: a (Particle, 0) sideKey (0x2000_0000) can never equal a
//   (RigidBody, large) sideKey (<= 0x0FFF_FFFF). Canonical ordering makes
//   key(a,b) == key(b,a). The packing is PURE INTEGER -> deterministic (D1).
//
//   HANDLE LIMIT: handle < 2^28 (268,435,456). Larger ids would alias the type
//   bits; the v0.8/v0.9 worlds are far below this. (NDEBUG-free asserts left to
//   the C2b/C2c fill sites that mint the handles.)
//
// THE STREAM (CandidatePairStream): a D1 device container of CandidatePairs
// SORTED ASCENDING by stable_key. Built WITHOUT any append atomics: C2a ships a
// reusable builder that takes an UNSORTED device CandidatePair array and produces
// the sorted stream via thrust::stable_sort (radix on the packed key ->
// deterministic, byte-exact run-to-run). C2b/C2c populate the unsorted input
// from the real broadphase sources via private-slice fill (the count->scan->fill
// posture of cross_system_query.hpp -- NO global append atomic), then call this
// same builder. So the sort/determinism contract lives in ONE place.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define NUKA_CANDIDATE_HD __host__ __device__
#else
#define NUKA_CANDIDATE_HD
#endif

namespace nuka::collision {

using constraint::CollidableRef;
using constraint::CollidableType;

// handle field width in a packed sideKey (low bits); the top (32 - this) bits
// carry the CollidableType. 28 -> up to 16 types, handles < 2^28.
inline constexpr uint32_t kCandidateHandleBits = 28u;
inline constexpr uint32_t kCandidateHandleMask = (1u << kCandidateHandleBits) - 1u;  // 0x0FFFFFFF

// Pack one collidable side into a 32-bit key: type in the high 4 bits, handle in
// the low 28. Pure integer -> deterministic. (Handle is masked to 28 bits; the
// >2^28 guard belongs at the handle-minting site, see the header limit note.)
NUKA_CANDIDATE_HD inline uint32_t PackSideKey(const CollidableRef& ref) {
    return (static_cast<uint32_t>(ref.type) << kCandidateHandleBits) |
           (ref.handle & kCandidateHandleMask);
}

// Canonical 64-bit sort key for an unordered pair: (min<<32) | max of the two
// sideKeys. key(a,b) == key(b,a); distinct (type,handle) pairs get distinct
// keys; type-in-high-bits means cross-type keys never collide.
NUKA_CANDIDATE_HD inline uint64_t MakeStableKey(const CollidableRef& a, const CollidableRef& b) {
    const uint32_t ka = PackSideKey(a);
    const uint32_t kb = PackSideKey(b);
    const uint32_t lo = ka < kb ? ka : kb;
    const uint32_t hi = ka < kb ? kb : ka;
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

// One broadphase candidate pair. `stable_key` is the canonical sort key; the
// builder recomputes it from (a,b) so a caller may leave it 0.
struct CandidatePair {
    CollidableRef a;
    CollidableRef b;
    uint64_t      stable_key = 0ull;
};

// A D1 device container of CandidatePairs, SORTED ASCENDING by stable_key.
// Built by BuildCandidatePairStream from an unsorted CandidatePair array (no
// append atomics; thrust::stable_sort -> deterministic). C2b/C2c fill the
// unsorted input from the real broadphase sources and call the same builder.
class CandidatePairStream {
public:
    CandidatePairStream() = default;
    CandidatePairStream(uint32_t count, phi::Buffer* pairs);
    ~CandidatePairStream();

    CandidatePairStream(const CandidatePairStream&) = delete;
    CandidatePairStream& operator=(const CandidatePairStream&) = delete;
    CandidatePairStream(CandidatePairStream&& other) noexcept;
    CandidatePairStream& operator=(CandidatePairStream&& other) noexcept;

    uint32_t Count() const { return count_; }

    // Device pointer to the sorted CandidatePair array (count_ elements).
    const CandidatePair* DevicePairs() const;

    // Copy the sorted pairs to host (count_ elements).
    std::vector<CandidatePair> DownloadPairs() const;

private:
    uint32_t     count_ = 0u;
    phi::Buffer* pairs_ = nullptr;  // phi v2 opaque handle; freed in the dtor.
};

// Build a sorted CandidatePairStream from an UNSORTED host-provided pair set.
// Computes each pair's stable_key (canonical), uploads, and thrust::stable_sorts
// ascending by stable_key -> a D1 byte-exact stream. The reusable builder C2b/C2c
// call after their private-slice broadphase fill. `count == 0` -> empty stream.
CandidatePairStream BuildCandidatePairStream(const phi::DeviceContext& context,
                                             const std::vector<CandidatePair>& unsorted_pairs);

// Default-context overload (mirrors the cross_system_query.hpp pair).
CandidatePairStream BuildCandidatePairStream(const std::vector<CandidatePair>& unsorted_pairs);

} // namespace nuka::collision
