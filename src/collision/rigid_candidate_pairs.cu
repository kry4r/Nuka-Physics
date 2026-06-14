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
#include "phi/backend.hpp"                    // InitBestDevice, DeviceBufferType
#include "phi/buffer.hpp"                     // Buffer*, BufferBase/Free
#include "phi/buffer_transfer_v2.hpp"         // UploadVectorV2
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

// File-local RAII over a phi v2 opaque Buffer*. Stream-0 device buffer type:
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy. Frees on every path (incl. CheckCuda throw paths). Base()/Handle()
// mirror the legacy Data()/handle.
class OwnedBuffer {
public:
    OwnedBuffer() = default;
    explicit OwnedBuffer(size_t bytes) {
        buf_ = phi::BufferAlloc(phi::DeviceBufferType(phi::InitBestDevice()), bytes);
    }
    ~OwnedBuffer() { if (buf_ != nullptr) phi::BufferFree(buf_); }
    OwnedBuffer(OwnedBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    OwnedBuffer& operator=(OwnedBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) phi::BufferFree(buf_);
            buf_ = o.buf_; o.buf_ = nullptr;
        }
        return *this;
    }
    OwnedBuffer(const OwnedBuffer&) = delete;
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;
    // Adopt an already-allocated handle (e.g. from UploadVectorV2).
    static OwnedBuffer Adopt(phi::Buffer* b) { OwnedBuffer o; o.buf_ = b; return o; }
    void* Base() const { return buf_ != nullptr ? phi::BufferBase(buf_) : nullptr; }
    phi::Buffer* Handle() const { return buf_; }
private:
    phi::Buffer* buf_ = nullptr;
};

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

// --- Pass 1: per-AABB-pair keep flag (bitmask AND exclude) ------------------
// One thread per LBVH AABB-pair. Writes 1 (keep) or 0 (drop). NO atomics; the
// explicit-<pair> force-include is applied later on the host (it can only ADD
// pairs the bitmask dropped, never remove a kept one). The contype/conaffinity
// and body-id arrays are indexed by the AABB's ORIGINAL input index (LBVH leaf
// order == input order). These tags are per-AABB and type-agnostic (commit 2):
// the body-id is for the same-body drop + the exclude search ONLY, distinct from
// the emit handle (see CompactKernel).
__global__ void KeepFlagKernel(uint32_t pair_count,
                               const collision::CollisionPair* __restrict__ aabb_pairs,
                               const uint32_t* __restrict__ aabb_types,
                               const uint32_t* __restrict__ aabb_body_ids,
                               const uint32_t* __restrict__ aabb_contypes,
                               const uint32_t* __restrict__ aabb_conaffinities,
                               const uint32_t* __restrict__ excl_lo,
                               const uint32_t* __restrict__ excl_hi,
                               uint32_t excl_count,
                               uint32_t* __restrict__ out_flags) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pair_count) {
        return;
    }
    const collision::CollisionPair sp = aabb_pairs[i];
    const uint32_t sa = sp.body_a;  // AABB indices (LBVH leaf == input order)
    const uint32_t sb = sp.body_b;

    // 1. contype/conaffinity bitmask (MuJoCo two-way AND/OR).
    const bool bitmask_ok = scene::PassesContactBitmask(
        aabb_contypes[sa], aabb_conaffinities[sa],
        aabb_contypes[sb], aabb_conaffinities[sb]);

    // 2. body-level exclude (authored <exclude> UNION parent-child auto-exclude).
    const uint32_t ba = aabb_body_ids[sa];
    const uint32_t bb = aabb_body_ids[sb];
    const bool excluded = DeviceIsExcluded(ba, bb, excl_lo, excl_hi, excl_count);

    // 3. drop a same-body pair (two shapes of ONE multi-shape body) -- a body
    //    never collides with itself. (For v0.8's 1-shape/body this never fires,
    //    since distinct shapes have distinct bodies; it makes the multi-shape
    //    path correct so CompactKernel never emits a (X,X) self-body-pair.)
    //
    //    CROSS-TYPE NOTE: a body id only identifies "the same physical body" WITHIN
    //    one collidable type. A RigidBody and an ArticulationLink could carry the
    //    same numeric body id yet are never the same body, so the same-body drop
    //    must fire ONLY when the two sides share a type. We gate on type-equality:
    //    a same-body-id pair is dropped iff both sides are the SAME type. For the
    //    all-rigid path every AABB is RigidBody, so the type-equality is always
    //    true and this reduces EXACTLY to C2b's `ba != bb` -> byte-identical rigid
    //    output (the regression guard).
    const bool same_actual_body = (ba == bb) && (aabb_types[sa] == aabb_types[sb]);
    out_flags[i] = (bitmask_ok && !excluded && !same_actual_body) ? 1u : 0u;
}

// --- Pass 2: compact kept AABB-pairs into emitted CandidatePairs ------------
// Scatter into the PRIVATE slot flag_scan[i] (exclusive scan of the flags). No
// atomics: each kept pair owns a distinct output index. Self-pairs (sa==sb) can
// never appear (LBVH emits canonical a<b); same-(body,type) pairs were already
// dropped by KeepFlagKernel (flag==0). Each emitted side's (type, react, handle)
// is stamped from the per-AABB tag arrays -- NOT hardcoded RigidBody -- so one
// pass emits rigid<->rigid, link<->link, AND link<->rigid pairs. The emit HANDLE
// comes from aabb_handles (body-id for rigid, link-index for articulation),
// DISTINCT from the body_id the filter used.
__global__ void CompactKernel(uint32_t pair_count,
                              const collision::CollisionPair* __restrict__ aabb_pairs,
                              const uint32_t* __restrict__ aabb_types,
                              const uint32_t* __restrict__ aabb_reacts,
                              const uint32_t* __restrict__ aabb_handles,
                              const uint32_t* __restrict__ flags,
                              const uint32_t* __restrict__ flag_scan,
                              CandidatePair* __restrict__ out_pairs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pair_count) {
        return;
    }
    if (flags[i] == 0u) {
        return;
    }
    const collision::CollisionPair sp = aabb_pairs[i];
    const uint32_t ia = sp.body_a;  // AABB indices (LBVH leaf == input order)
    const uint32_t ib = sp.body_b;

    CandidatePair out;
    out.a.type   = static_cast<CollidableType>(aabb_types[ia]);
    out.a.react  = static_cast<ReactionProviderKind>(aabb_reacts[ia]);
    out.a.handle = aabb_handles[ia];
    out.b.type   = static_cast<CollidableType>(aabb_types[ib]);
    out.b.react  = static_cast<ReactionProviderKind>(aabb_reacts[ib]);
    out.b.handle = aabb_handles[ib];
    out.stable_key = 0ull;  // BuildCandidatePairStream recomputes the canonical key.

    out_pairs[flag_scan[i]] = out;
}

} // namespace

// ---------------------------------------------------------------------------
// The TYPE-AGNOSTIC tagged core (commit 2). LBVH over the tagged AABBs ->
// bitmask+exclude+same-(body,type) filter -> compact (stamp tags) -> union the
// caller-supplied extra survivors -> sorted D1 stream -> adjacent dedup.
// ---------------------------------------------------------------------------
CandidatePairStream BuildCandidatePairsTagged(
    const phi::DeviceContext& context,
    const collision::AABB* device_aabbs, uint32_t count,
    const CandidateAabbTags& tags,
    const std::vector<std::pair<uint32_t, uint32_t>>& excluded_body_pairs,
    const std::vector<CandidatePair>& extra_survivors) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    // --- Host-side handle-guard (C2-named-debt consumer) --------------------
    // A handle >= 2^28 would alias the type bits in PackSideKey (candidate_pair.hpp
    // packs (type<<28)|(handle & 0x0FFFFFFF)), silently corrupting the stable_key.
    // This is the mint site for every handle that enters the stream (rigid body id
    // or articulation link index), so the guard lives here once.
    for (uint32_t i = 0u; i < count; ++i) {
        if (tags.handles[i] > kCandidateHandleMask) {
            throw std::runtime_error(
                "BuildCandidatePairsTagged: collidable handle " +
                std::to_string(tags.handles[i]) + " exceeds the 2^28 limit (" +
                std::to_string(kCandidateHandleMask) +
                ") and would alias the CollidableType bits in the stable_key");
        }
    }

    // Survivors of the on-device bitmask+exclude filter. The caller's extras are
    // appended AFTER the device survivors (below), NOT before: BuildCandidatePairStream
    // uses thrust::STABLE_sort and does NOT swap a/b to canonical, so when an extra
    // (e.g. an explicit <pair> authored in descending geom order -> stored (max,min))
    // duplicates a device-detected pair (always stored (min,max)) the adjacent dedup
    // keeps the FIRST of the equal-key run. Device-first preserves the device's
    // canonical (min,max) byte layout in the deduped output -- the byte-identity the
    // rigid wrapper requires vs the pre-refactor C2b builder.
    std::vector<CandidatePair> survivors;

    if (count > 0u) {
        // --- LBVH over the tagged world AABBs (already D1) -------------------
        // RETAIN nothing extra; we only need the sorted compact AABB-pair list.
        // The LBVH emits ORIGINAL-input-order indices (proven C2), so each tag
        // array indexes straight by the pair's body_a / body_b.
        auto lbvh = collision::gpu::BuildLbvhBroadphase(context, device_aabbs, count);
        const uint32_t pair_count = lbvh.PairCount();

        // Upload the per-AABB tag arrays the kernels index by input id.
        std::vector<uint32_t> types(count);
        std::vector<uint32_t> reacts(count);
        for (uint32_t i = 0u; i < count; ++i) {
            types[i]  = static_cast<uint32_t>(tags.types[i]);
            reacts[i] = static_cast<uint32_t>(tags.reacts[i]);
        }
        std::vector<uint32_t> handles(tags.handles, tags.handles + count);
        std::vector<uint32_t> body_ids(tags.body_ids, tags.body_ids + count);
        std::vector<uint32_t> contypes(tags.contypes, tags.contypes + count);
        std::vector<uint32_t> conaff(tags.conaffinities, tags.conaffinities + count);
        // Stream-0 device buffer type: NULL-stream transfers are byte+ordering
        // identical to the legacy synchronous memcpy.
        phi::BufferType* bt = phi::DeviceBufferType(phi::InitBestDevice());
        OwnedBuffer d_types    = OwnedBuffer::Adopt(phi::UploadVectorV2(bt, types));
        OwnedBuffer d_reacts   = OwnedBuffer::Adopt(phi::UploadVectorV2(bt, reacts));
        OwnedBuffer d_handles  = OwnedBuffer::Adopt(phi::UploadVectorV2(bt, handles));
        OwnedBuffer d_body_ids = OwnedBuffer::Adopt(phi::UploadVectorV2(bt, body_ids));
        OwnedBuffer d_contypes = OwnedBuffer::Adopt(phi::UploadVectorV2(bt, contypes));
        OwnedBuffer d_conaff   = OwnedBuffer::Adopt(phi::UploadVectorV2(bt, conaff));

        // Upload the sorted canonical exclude list as two parallel uint32 arrays
        // so the device binary search needs no std::pair ((min,max), ascending).
        const uint32_t excl_count = static_cast<uint32_t>(excluded_body_pairs.size());
        std::vector<uint32_t> excl_lo(excl_count);
        std::vector<uint32_t> excl_hi(excl_count);
        for (uint32_t i = 0u; i < excl_count; ++i) {
            excl_lo[i] = excluded_body_pairs[i].first;
            excl_hi[i] = excluded_body_pairs[i].second;
        }
        // Allocate >=1 element so Base() is a valid (unused-when-count-0) pointer.
        OwnedBuffer d_excl_lo(std::max<uint32_t>(excl_count, 1u) * sizeof(uint32_t));
        OwnedBuffer d_excl_hi(std::max<uint32_t>(excl_count, 1u) * sizeof(uint32_t));
        if (excl_count > 0u) {
            phi::BufferUpload(d_excl_lo.Handle(), excl_lo.data(), 0,
                              excl_count * sizeof(uint32_t));
            phi::BufferUpload(d_excl_hi.Handle(), excl_hi.data(), 0,
                              excl_count * sizeof(uint32_t));
        }

        if (pair_count > 0u) {
            const uint32_t blocks = (pair_count + kBlockSize - 1u) / kBlockSize;

            // --- Pass 1: keep flags -----------------------------------------
            OwnedBuffer d_flags(pair_count * sizeof(uint32_t));
            KeepFlagKernel<<<blocks, kBlockSize, 0, stream>>>(
                pair_count, lbvh.DevicePairs(),
                static_cast<const uint32_t*>(d_types.Base()),
                static_cast<const uint32_t*>(d_body_ids.Base()),
                static_cast<const uint32_t*>(d_contypes.Base()),
                static_cast<const uint32_t*>(d_conaff.Base()),
                static_cast<const uint32_t*>(d_excl_lo.Base()),
                static_cast<const uint32_t*>(d_excl_hi.Base()),
                excl_count,
                static_cast<uint32_t*>(d_flags.Base()));
            CheckCuda(cudaGetLastError(), "KeepFlagKernel launch");

            // --- Exclusive scan -> per-pair output slot ---------------------
            OwnedBuffer d_scan(pair_count * sizeof(uint32_t));
            thrust::exclusive_scan(
                thrust::cuda::par.on(stream),
                static_cast<const uint32_t*>(d_flags.Base()),
                static_cast<const uint32_t*>(d_flags.Base()) + pair_count,
                static_cast<uint32_t*>(d_scan.Base()));
            CheckCuda(cudaGetLastError(), "exclusive_scan keep flags");

            context.stream.Synchronize();
            uint32_t last_scan = 0u;
            uint32_t last_flag = 0u;
            {
                const auto* scan_dev = static_cast<const uint32_t*>(d_scan.Base());
                const auto* flag_dev = static_cast<const uint32_t*>(d_flags.Base());
                CheckCuda(cudaMemcpy(&last_scan, scan_dev + (pair_count - 1u),
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost),
                          "copy last scan");
                CheckCuda(cudaMemcpy(&last_flag, flag_dev + (pair_count - 1u),
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost),
                          "copy last flag");
            }
            const uint32_t survivor_count = last_scan + last_flag;

            if (survivor_count > 0u) {
                // --- Pass 2: compact into tagged CandidatePairs -------------
                OwnedBuffer d_out(survivor_count * sizeof(CandidatePair));
                CompactKernel<<<blocks, kBlockSize, 0, stream>>>(
                    pair_count, lbvh.DevicePairs(),
                    static_cast<const uint32_t*>(d_types.Base()),
                    static_cast<const uint32_t*>(d_reacts.Base()),
                    static_cast<const uint32_t*>(d_handles.Base()),
                    static_cast<const uint32_t*>(d_flags.Base()),
                    static_cast<const uint32_t*>(d_scan.Base()),
                    static_cast<CandidatePair*>(d_out.Base()));
                CheckCuda(cudaGetLastError(), "CompactKernel launch");
                context.stream.Synchronize();

                const size_t base = survivors.size();
                survivors.resize(base + survivor_count);
                phi::BufferDownload(d_out.Handle(), survivors.data() + base, 0,
                                    survivor_count * sizeof(CandidatePair));
            } else {
                context.stream.Synchronize();
            }
        }
    }

    // Append the caller's extras AFTER the device survivors (see the survivors
    // declaration: device-first preserves the device's canonical (min,max) byte
    // layout as the dedup winner -> byte-identity vs the pre-refactor C2b builder).
    survivors.insert(survivors.end(), extra_survivors.begin(), extra_survivors.end());

    // --- Build the sorted D1 stream (C2a stamps stable_key + stable_sort) ----
    CandidatePairStream stream_out = BuildCandidatePairStream(context, survivors);

    // --- Dedup adjacent identical pairs -------------------------------------
    // After the C2a sort, identical pairs (a multi-shape collidable's per-shape-pair
    // candidates mapping to the same (a,b), OR an extra-survivor duplicating a
    // bitmask survivor) are ADJACENT (same stable_key). Collapse them so the stream
    // carries each pair once (C3 narrowphase re-expands shapes). For 1-shape/body
    // with no duplicate extra this is a no-op.
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

// ---------------------------------------------------------------------------
// The RIGID<->RIGID wrapper -- a thin all-RigidBody tag build over the tagged
// core, BYTE-IDENTICAL to the C2b builder (the rigid regression guard). The
// explicit-<pair> force-include (rigid-source-shape-specific) is computed HERE
// and passed as extra survivors; cross-type explicit pairs are out of scope.
// ---------------------------------------------------------------------------
CandidatePairStream BuildRigidCandidatePairs(
    const phi::DeviceContext& context,
    const collision::AABB* device_shape_aabbs, uint32_t shape_count,
    const uint32_t* shape_body_ids,
    const uint32_t* shape_contypes,
    const uint32_t* shape_conaffinities,
    const scene::CookedFilterPolicy& policy) {
    if (shape_count == 0u && policy.explicit_pairs.empty()) {
        return BuildCandidatePairStream(context, {});
    }

    // The RigidBody reaction provider kind, read off the generated registry so
    // the emit stays in sync with the collidable metadata (no hard-coded enum).
    const ReactionProviderKind rigid_react =
        constraint::GetCollidableTypeInfo(CollidableType::RigidBody).react;

    // All-RigidBody tags: type=RigidBody, react=rigid_react, handle=body_id,
    // body_id=body_id -> the handle/body_id split is a no-op so this reduces to
    // C2b's emit exactly.
    std::vector<CollidableType>     types(shape_count, CollidableType::RigidBody);
    std::vector<ReactionProviderKind> reacts(shape_count, rigid_react);
    CandidateAabbTags tags;
    tags.types         = types.data();
    tags.reacts        = reacts.data();
    tags.handles       = shape_body_ids;   // handle == body id for rigid.
    tags.body_ids      = shape_body_ids;
    tags.contypes      = shape_contypes;
    tags.conaffinities = shape_conaffinities;

    // --- Host explicit-<pair> force-include (rigid only) --------------------
    // MuJoCo: an authored <contact><pair> ALWAYS generates, even if the bitmask
    // would reject it. The policy stores explicit pairs in SOURCE-shape space and
    // v0.8 has NO mesh decomposition (cooked-row id == source-shape id, 1:1), so
    // an explicit pair's (geom1, geom2) are directly the cooked SHAPE ids we map
    // to body ids via shape_body_ids. We append each as an extra survivor (a
    // duplicate of a bitmask survivor collapses in the core's adjacent dedup).
    std::vector<CandidatePair> extra;
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
        extra.push_back(out);
    }

    return BuildCandidatePairsTagged(context, device_shape_aabbs, shape_count,
                                     tags, policy.excluded_body_pairs, extra);
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

// ---------------------------------------------------------------------------
// The ARTICULATION-LINK <-> RIGID mixed entry (commit 3's gate consumer).
// Concatenate [rigid AABBs | link AABBs] with parallel tag arrays (rigid side
// tagged RigidBody/handle=body-id; link side tagged ArticulationLink/handle=link
// index) and run the tagged core. No extra survivors (cross-type explicit pairs
// are out of scope).
// ---------------------------------------------------------------------------
CandidatePairStream BuildArticulationRigidCandidatePairs(
    const phi::DeviceContext& context,
    const collision::AABB* rigid_aabbs, uint32_t rigid_count,
    const uint32_t* rigid_body_ids,
    const uint32_t* rigid_contypes,
    const uint32_t* rigid_conaffinities,
    const LinkShapeAabbs& links,
    const uint32_t* link_contypes,
    const uint32_t* link_conaffinities,
    const std::vector<std::pair<uint32_t, uint32_t>>& excluded_body_pairs) {
    const uint32_t link_count = static_cast<uint32_t>(links.aabbs.size());
    const uint32_t total = rigid_count + link_count;
    if (total == 0u) {
        return BuildCandidatePairStream(context, {});
    }

    const ReactionProviderKind rigid_react =
        constraint::GetCollidableTypeInfo(CollidableType::RigidBody).react;
    const ReactionProviderKind link_react =
        constraint::GetCollidableTypeInfo(CollidableType::ArticulationLink).react;

    // Concatenate the AABBs: [rigid... | link...]. The tag arrays below are in the
    // SAME concatenated order (the LBVH emits input-order indices, proven C2).
    std::vector<collision::AABB> aabbs;
    aabbs.reserve(total);
    {
        // The rigid AABBs live in DEVICE memory; copy them to host to concatenate
        // with the link AABBs (host), then re-upload the unified array. (This entry
        // is the validated-not-wired gate path; the unified upload keeps the tag
        // arrays trivially parallel to one contiguous device buffer.)
        std::vector<collision::AABB> rigid_host(rigid_count);
        if (rigid_count > 0u) {
            // rigid_aabbs is already device-resident; copy device->host directly.
            CheckCuda(cudaMemcpy(rigid_host.data(), rigid_aabbs,
                                 rigid_count * sizeof(collision::AABB),
                                 cudaMemcpyDeviceToHost),
                      "copy rigid AABBs to host");
        }
        for (uint32_t i = 0u; i < rigid_count; ++i) aabbs.push_back(rigid_host[i]);
        for (uint32_t i = 0u; i < link_count; ++i) aabbs.push_back(links.aabbs[i]);
    }
    // Stream-0 device buffer type: NULL-stream transfers are byte+ordering
    // identical to the legacy synchronous memcpy.
    OwnedBuffer d_aabbs = OwnedBuffer::Adopt(phi::UploadVectorV2(
        phi::DeviceBufferType(phi::InitBestDevice()), aabbs));

    // Parallel tag arrays over the concatenated order.
    std::vector<CollidableType>       types(total);
    std::vector<ReactionProviderKind> reacts(total);
    std::vector<uint32_t>             handles(total);
    std::vector<uint32_t>             body_ids(total);
    std::vector<uint32_t>             contypes(total);
    std::vector<uint32_t>             conaff(total);
    for (uint32_t i = 0u; i < rigid_count; ++i) {
        types[i]    = CollidableType::RigidBody;
        reacts[i]   = rigid_react;
        handles[i]  = rigid_body_ids[i];   // rigid handle == body id.
        body_ids[i] = rigid_body_ids[i];
        contypes[i] = rigid_contypes[i];
        conaff[i]   = rigid_conaffinities[i];
    }
    for (uint32_t i = 0u; i < link_count; ++i) {
        const uint32_t k = rigid_count + i;
        types[k]    = CollidableType::ArticulationLink;
        reacts[k]   = link_react;
        handles[k]  = links.link_indices[i];   // articulation handle == link index.
        body_ids[k] = links.shape_body_ids[i];
        contypes[k] = link_contypes[i];
        conaff[k]   = link_conaffinities[i];
    }

    CandidateAabbTags tags;
    tags.types         = types.data();
    tags.reacts        = reacts.data();
    tags.handles       = handles.data();
    tags.body_ids      = body_ids.data();
    tags.contypes      = contypes.data();
    tags.conaffinities = conaff.data();

    return BuildCandidatePairsTagged(
        context, static_cast<const collision::AABB*>(d_aabbs.Base()), total,
        tags, excluded_body_pairs, {});
}

} // namespace nuka::collision
