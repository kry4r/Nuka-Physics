// ---------------------------------------------------------------------------
// nuka::collision::BuildRigidCandidatePairs implementation (v0.8 C2b)
// ---------------------------------------------------------------------------
// The RIGID<->RIGID filtered candidate-pair stream. See rigid_candidate_pairs.hpp
// for the full design (pipeline, filter precedence, handle=body-id model).
//
// D1 (determinism) MECHANISM:
//   * The LBVH (broadphase_lbvh.cu) is already D1: thrust::stable_sort on Morton
//     codes + a thrust::stable_sort of the compact output, integer-only atomics.
//     Its DownloadPairs() list is byte-exact run-to-run.
//   * The FILTER compacts survivors with the cross_system_query.hpp posture --
//     count -> thrust::exclusive_scan -> scatter into a PRIVATE pre-computed slot
//     -- so there is NO append/global atomic anywhere in this TU (the lint glob
//     src/collision/**/*.cu would flag any atomicAdd; we have none). The keep
//     test (bitmask + binary-search exclude) is pure integer arithmetic. The
//     scatter writes each kept pair to flag_scan[i], a deterministic index.
//   * The host explicit-<pair> merge appends to a std::vector in a fixed order
//     (the policy's sorted explicit_pairs), then BuildCandidatePairStream (C2a)
//     stamps the canonical stable_key and thrust::stable_sorts -> the final
//     stream is byte-exact regardless of the survivor/merge order.
// ---------------------------------------------------------------------------

#include "collision/rigid_candidate_pairs.hpp"

#include "collision/broadphase_lbvh.hpp"
#include "collision/dynamic_broadphase.hpp"  // collision::CollisionPair
#include "phi/buffer_transfer.hpp"
#include "scene/contact_filter.hpp"           // PassesContactBitmask

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision {

// candidate_pair.hpp re-exports CollidableRef/CollidableType into this namespace;
// the reaction-provider kind enum lives in nuka::constraint, pull it in too.
using constraint::ReactionProviderKind;

namespace {

constexpr uint32_t kBlockSize = 128u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// Device binary search of the sorted canonical excluded_body_pairs list. The list
// is uploaded as two parallel arrays (lo[], hi[]) with lo <= hi per entry and the
// list ascending by (lo, hi). Returns true iff (a,b) (in any order) is excluded.
// Pure integer comparisons -> deterministic, no atomics.
__device__ __forceinline__ bool DeviceIsExcluded(uint32_t a, uint32_t b,
                                                 const uint32_t* __restrict__ excl_lo,
                                                 const uint32_t* __restrict__ excl_hi,
                                                 uint32_t excl_count) {
    if (excl_count == 0u) {
        return false;
    }
    uint32_t key_lo = a < b ? a : b;
    uint32_t key_hi = a < b ? b : a;
    uint32_t left = 0u;
    uint32_t right = excl_count;  // half-open [left, right)
    while (left < right) {
        const uint32_t mid = left + ((right - left) >> 1u);
        const uint32_t mlo = excl_lo[mid];
        const uint32_t mhi = excl_hi[mid];
        // Lexicographic compare of (mlo, mhi) vs (key_lo, key_hi).
        if (mlo < key_lo || (mlo == key_lo && mhi < key_hi)) {
            left = mid + 1u;
        } else if (mlo == key_lo && mhi == key_hi) {
            return true;
        } else {
            right = mid;
        }
    }
    return false;
}

// --- Pass 1: per-shape-pair keep flag (bitmask AND exclude) -----------------
// One thread per LBVH shape-pair. Writes 1 (keep) or 0 (drop). NO atomics; the
// explicit-<pair> force-include is applied later on the host (it can only ADD
// pairs the bitmask dropped, never remove a kept one).
__global__ void KeepFlagKernel(uint32_t pair_count,
                               const collision::CollisionPair* __restrict__ shape_pairs,
                               const uint32_t* __restrict__ shape_body_ids,
                               const uint32_t* __restrict__ shape_contypes,
                               const uint32_t* __restrict__ shape_conaffinities,
                               const uint32_t* __restrict__ excl_lo,
                               const uint32_t* __restrict__ excl_hi,
                               uint32_t excl_count,
                               uint32_t* __restrict__ out_flags) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pair_count) {
        return;
    }
    const collision::CollisionPair sp = shape_pairs[i];
    const uint32_t sa = sp.body_a;  // SHAPE indices (LBVH leaf order)
    const uint32_t sb = sp.body_b;

    // 1. contype/conaffinity bitmask (MuJoCo two-way AND/OR).
    const bool bitmask_ok = scene::PassesContactBitmask(
        shape_contypes[sa], shape_conaffinities[sa],
        shape_contypes[sb], shape_conaffinities[sb]);

    // 2. body-level exclude (authored <exclude> UNION parent-child auto-exclude).
    const uint32_t ba = shape_body_ids[sa];
    const uint32_t bb = shape_body_ids[sb];
    const bool excluded = DeviceIsExcluded(ba, bb, excl_lo, excl_hi, excl_count);

    // 3. drop a same-body pair (two shapes of ONE multi-shape body) -- a body
    //    never collides with itself. (For v0.8's 1-shape/body this never fires,
    //    since distinct shapes have distinct bodies; it makes the multi-shape
    //    path correct so CompactKernel never emits a (X,X) self-body-pair.)
    out_flags[i] = (bitmask_ok && !excluded && ba != bb) ? 1u : 0u;
}

// --- Pass 2: compact kept shape-pairs into emitted body-pair CandidatePairs --
// Scatter into the PRIVATE slot flag_scan[i] (exclusive scan of the flags). No
// atomics: each kept pair owns a distinct output index. Self-pairs (sa==sb) can
// never appear (LBVH emits canonical a<b); same-body multi-shape pairs (ba==bb)
// were already dropped by KeepFlagKernel (flag==0), so every pair reaching here
// has two distinct bodies.
__global__ void CompactKernel(uint32_t pair_count,
                              const collision::CollisionPair* __restrict__ shape_pairs,
                              const uint32_t* __restrict__ shape_body_ids,
                              const uint32_t* __restrict__ flags,
                              const uint32_t* __restrict__ flag_scan,
                              ReactionProviderKind react,
                              CandidatePair* __restrict__ out_pairs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pair_count) {
        return;
    }
    if (flags[i] == 0u) {
        return;
    }
    const collision::CollisionPair sp = shape_pairs[i];
    const uint32_t ba = shape_body_ids[sp.body_a];
    const uint32_t bb = shape_body_ids[sp.body_b];

    CandidatePair out;
    out.a.type = CollidableType::RigidBody;
    out.a.react = react;
    out.a.handle = ba;
    out.b.type = CollidableType::RigidBody;
    out.b.react = react;
    out.b.handle = bb;
    out.stable_key = 0ull;  // BuildCandidatePairStream recomputes the canonical key.

    out_pairs[flag_scan[i]] = out;
}

} // namespace

CandidatePairStream BuildRigidCandidatePairs(
    const phi::DeviceContext& context,
    const collision::AABB* device_shape_aabbs, uint32_t shape_count,
    const uint32_t* shape_body_ids,
    const uint32_t* shape_contypes,
    const uint32_t* shape_conaffinities,
    const scene::CookedFilterPolicy& policy) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    if (shape_count == 0u) {
        return BuildCandidatePairStream(context, {});
    }

    // The RigidBody reaction provider kind, read off the generated registry so
    // the emit stays in sync with the collidable metadata (no hard-coded enum).
    const ReactionProviderKind rigid_react =
        constraint::GetCollidableTypeInfo(CollidableType::RigidBody).react;

    // --- LBVH over the per-shape world AABBs (already D1) --------------------
    // RETAIN nothing extra; we only need the sorted compact shape-pair list.
    auto lbvh = collision::gpu::BuildLbvhBroadphase(context, device_shape_aabbs,
                                                    shape_count);
    const uint32_t pair_count = lbvh.PairCount();

    // Upload the per-shape metadata the kernels index by SHAPE id.
    std::vector<uint32_t> body_ids(shape_body_ids, shape_body_ids + shape_count);
    std::vector<uint32_t> contypes(shape_contypes, shape_contypes + shape_count);
    std::vector<uint32_t> conaff(shape_conaffinities,
                                 shape_conaffinities + shape_count);
    phi::Buffer d_body_ids = phi::UploadVector(body_ids);
    phi::Buffer d_contypes = phi::UploadVector(contypes);
    phi::Buffer d_conaff = phi::UploadVector(conaff);

    // Upload the sorted canonical exclude list as two parallel uint32 arrays so
    // the device binary search needs no std::pair. (min,max) per entry, ascending.
    const uint32_t excl_count =
        static_cast<uint32_t>(policy.excluded_body_pairs.size());
    std::vector<uint32_t> excl_lo(excl_count);
    std::vector<uint32_t> excl_hi(excl_count);
    for (uint32_t i = 0u; i < excl_count; ++i) {
        excl_lo[i] = policy.excluded_body_pairs[i].first;
        excl_hi[i] = policy.excluded_body_pairs[i].second;
    }
    // Allocate >=1 element so Data() is a valid (unused-when-count-0) pointer.
    phi::Buffer d_excl_lo(std::max<uint32_t>(excl_count, 1u) * sizeof(uint32_t),
                          phi::MemoryKind::Device);
    phi::Buffer d_excl_hi(std::max<uint32_t>(excl_count, 1u) * sizeof(uint32_t),
                          phi::MemoryKind::Device);
    if (excl_count > 0u) {
        d_excl_lo.CopyFromHost(excl_lo.data(), excl_count * sizeof(uint32_t));
        d_excl_hi.CopyFromHost(excl_hi.data(), excl_count * sizeof(uint32_t));
    }

    // Survivors of the on-device bitmask+exclude filter (body-pairs).
    std::vector<CandidatePair> survivors;

    if (pair_count > 0u) {
        const uint32_t blocks = (pair_count + kBlockSize - 1u) / kBlockSize;

        // --- Pass 1: keep flags ---------------------------------------------
        phi::Buffer d_flags(pair_count * sizeof(uint32_t), phi::MemoryKind::Device);
        KeepFlagKernel<<<blocks, kBlockSize, 0, stream>>>(
            pair_count, lbvh.DevicePairs(),
            static_cast<const uint32_t*>(d_body_ids.Data()),
            static_cast<const uint32_t*>(d_contypes.Data()),
            static_cast<const uint32_t*>(d_conaff.Data()),
            static_cast<const uint32_t*>(d_excl_lo.Data()),
            static_cast<const uint32_t*>(d_excl_hi.Data()),
            excl_count,
            static_cast<uint32_t*>(d_flags.Data()));
        CheckCuda(cudaGetLastError(), "KeepFlagKernel launch");

        // --- Exclusive scan -> per-pair output slot -------------------------
        phi::Buffer d_scan(pair_count * sizeof(uint32_t), phi::MemoryKind::Device);
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            static_cast<const uint32_t*>(d_flags.Data()),
            static_cast<const uint32_t*>(d_flags.Data()) + pair_count,
            static_cast<uint32_t*>(d_scan.Data()));
        CheckCuda(cudaGetLastError(), "exclusive_scan keep flags");

        context.stream.Synchronize();
        uint32_t last_scan = 0u;
        uint32_t last_flag = 0u;
        {
            const auto* scan_dev = static_cast<const uint32_t*>(d_scan.Data());
            const auto* flag_dev = static_cast<const uint32_t*>(d_flags.Data());
            CheckCuda(cudaMemcpy(&last_scan, scan_dev + (pair_count - 1u),
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost),
                      "copy last scan");
            CheckCuda(cudaMemcpy(&last_flag, flag_dev + (pair_count - 1u),
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost),
                      "copy last flag");
        }
        const uint32_t survivor_count = last_scan + last_flag;

        if (survivor_count > 0u) {
            // --- Pass 2: compact into body-pair CandidatePairs --------------
            phi::Buffer d_out(survivor_count * sizeof(CandidatePair),
                              phi::MemoryKind::Device);
            CompactKernel<<<blocks, kBlockSize, 0, stream>>>(
                pair_count, lbvh.DevicePairs(),
                static_cast<const uint32_t*>(d_body_ids.Data()),
                static_cast<const uint32_t*>(d_flags.Data()),
                static_cast<const uint32_t*>(d_scan.Data()),
                rigid_react,
                static_cast<CandidatePair*>(d_out.Data()));
            CheckCuda(cudaGetLastError(), "CompactKernel launch");
            context.stream.Synchronize();

            survivors.resize(survivor_count);
            d_out.CopyToHost(survivors.data(),
                             survivor_count * sizeof(CandidatePair));
        } else {
            context.stream.Synchronize();
        }
    }

    // --- Host explicit-<pair> force-include merge ---------------------------
    // MuJoCo: an authored <contact><pair> ALWAYS generates, even if the bitmask
    // would reject it. The policy stores explicit pairs in SOURCE-shape space and
    // v0.8 has NO mesh decomposition (cooked-row id == source-shape id, 1:1), so
    // an explicit pair's (geom1, geom2) are directly the cooked SHAPE ids we map
    // to body ids via shape_body_ids. (The SOURCE-shape -> COOKED-row bridge for a
    // V-HACD-decomposed mesh -- one source shape -> N cooked rows -- is a C3/later
    // concern; this builder treats them 1:1 as v0.8 reality dictates.)
    //
    // We force-include by appending each explicit pair's body-pair to the survivor
    // set. A pair the bitmask already kept becomes a duplicate body-pair; the
    // post-sort dedup below collapses it, so the union is idempotent. We only add
    // pairs whose source shapes are in range (defensive) and whose bodies differ.
    for (const auto& ep : policy.explicit_pairs) {
        const scene::ShapeId g1 = ep.geom1;
        const scene::ShapeId g2 = ep.geom2;
        if (g1 >= shape_count || g2 >= shape_count) {
            continue;  // out-of-range source shape (no 1:1 cooked row) -> skip.
        }
        const uint32_t ba = shape_body_ids[g1];
        const uint32_t bb = shape_body_ids[g2];
        if (ba == bb) {
            continue;  // a body never collides with itself.
        }
        CandidatePair out;
        out.a.type = CollidableType::RigidBody;
        out.a.react = rigid_react;
        out.a.handle = ba;
        out.b.type = CollidableType::RigidBody;
        out.b.react = rigid_react;
        out.b.handle = bb;
        out.stable_key = 0ull;
        survivors.push_back(out);
    }

    // --- Build the sorted D1 stream (C2a stamps stable_key + stable_sort) ----
    CandidatePairStream stream_out =
        BuildCandidatePairStream(context, survivors);

    // --- Dedup adjacent identical body-pairs --------------------------------
    // After the C2a sort, identical body-pairs (a multi-shape body's per-shape-pair
    // candidates mapping to the same body-pair, OR an explicit pair duplicating a
    // bitmask survivor) are ADJACENT (same stable_key). Collapse them so the stream
    // carries each body-pair once (C3 narrowphase re-expands the body's shapes).
    // For v0.8's 1-shape/body with no duplicate explicit pair this is a no-op.
    const auto sorted = stream_out.DownloadPairs();
    bool has_dup = false;
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i].stable_key == sorted[i - 1].stable_key) {
            has_dup = true;
            break;
        }
    }
    if (!has_dup) {
        return stream_out;
    }
    std::vector<CandidatePair> unique_pairs;
    unique_pairs.reserve(sorted.size());
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (i == 0 || sorted[i].stable_key != sorted[i - 1].stable_key) {
            unique_pairs.push_back(sorted[i]);
        }
    }
    return BuildCandidatePairStream(context, unique_pairs);
}

CandidatePairStream BuildRigidCandidatePairs(
    const collision::AABB* device_shape_aabbs, uint32_t shape_count,
    const uint32_t* shape_body_ids,
    const uint32_t* shape_contypes,
    const uint32_t* shape_conaffinities,
    const scene::CookedFilterPolicy& policy) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildRigidCandidatePairs(context, device_shape_aabbs, shape_count,
                                    shape_body_ids, shape_contypes,
                                    shape_conaffinities, policy);
}

} // namespace nuka::collision
