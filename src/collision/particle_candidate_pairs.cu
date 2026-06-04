// ---------------------------------------------------------------------------
// nuka::collision -- particle<->rigid + particle<->particle candidate-pair
// builders (v0.8 C2c) implementation. See particle_candidate_pairs.hpp for the
// full design (CSR reframe, handle model, SystemKind gate seam).
// ---------------------------------------------------------------------------
// D1 (determinism) MECHANISM:
//   * Both source CSRs (cross_system_query.cu, particle_uniform_grid.cu) are
//     already D1: count -> exclusive_scan -> private-slice fill, lowest-index
//     truncation, integer-only atomics; their downloaded CSR is byte-exact
//     run-to-run.
//   * The reframe kernels here write each CSR entry to its CSR offset slot
//     (out[offsets[p] + k]); every slot is written exactly ONCE -> there is NO
//     append/global atomic and NO scan in this TU (the lint glob src/collision/
//     **/*.cu would flag any atomicAdd; we have none). The slot is a pure-integer
//     deterministic index.
//   * BuildCandidatePairStream (C2a) stamps the canonical stable_key and
//     thrust::stable_sorts -> the final stream is byte-exact regardless of warp
//     order. Adjacent identical pairs (multi-shape body -> same body-pair; grid
//     i->j vs j->i canonicalized to the same pair) are collapsed exactly as C2b.
// ---------------------------------------------------------------------------

#include "collision/particle_candidate_pairs.hpp"

#include "collision/cross_system_query.hpp"
#include "constraint/collidable.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::collision {

// candidate_pair.hpp re-exports CollidableRef/CollidableType into this namespace;
// pull the reaction-provider kind enum in from nuka::constraint.
using constraint::ReactionProviderKind;

namespace {

constexpr uint32_t kBlockSize = 128u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// --- particle<->rigid reframe kernel ----------------------------------------
// One thread per particle. For particle p with CSR slice [off, off+count): each
// candidate SHAPE index s = cand_indices[off + k] maps to body shape_body_ids[s];
// emit CandidatePair{ a=(Particle, p), b=(RigidBody, body), key=0 } at out[off+k].
// Pure-integer slot -> no atomic, no scan. Self-pairs are impossible (cross-type).
__global__ void ReframeParticleRigidKernel(
    uint32_t particle_count,
    const uint32_t* __restrict__ cand_offsets,
    const uint32_t* __restrict__ cand_counts,
    const uint32_t* __restrict__ cand_indices,   // RIGID SHAPE indices (leaf order)
    const uint32_t* __restrict__ shape_body_ids, // shape -> body id
    ReactionProviderKind particle_react,
    ReactionProviderKind rigid_react,
    CandidatePair* __restrict__ out_pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count) {
        return;
    }
    const uint32_t off = cand_offsets[p];
    const uint32_t cnt = cand_counts[p];
    for (uint32_t k = 0u; k < cnt; ++k) {
        const uint32_t shape = cand_indices[off + k];
        const uint32_t body = shape_body_ids[shape];

        CandidatePair out;
        out.a.type = CollidableType::Particle;
        out.a.react = particle_react;
        out.a.handle = p;
        out.b.type = CollidableType::RigidBody;
        out.b.react = rigid_react;
        out.b.handle = body;
        out.stable_key = 0ull;  // BuildCandidatePairStream recomputes the key.

        out_pairs[off + k] = out;
    }
}

// --- particle<->particle reframe kernel -------------------------------------
// One thread per particle. For particle i with neighbor slice [off, off+count):
// each neighbor j = nbr_indices[off + k]; emit a CANONICAL pair (min,max) so the
// grid's i->j and j->i listings produce byte-identical CandidatePairs (collapsed
// by the post-sort adjacent dedup -> one i<j pair). Slot = off + k -> no atomic.
__global__ void ReframeParticleParticleKernel(
    uint32_t particle_count,
    const uint32_t* __restrict__ nbr_offsets,
    const uint32_t* __restrict__ nbr_counts,
    const uint32_t* __restrict__ nbr_indices,
    ReactionProviderKind particle_react,
    CandidatePair* __restrict__ out_pairs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const uint32_t off = nbr_offsets[i];
    const uint32_t cnt = nbr_counts[i];
    for (uint32_t k = 0u; k < cnt; ++k) {
        const uint32_t j = nbr_indices[off + k];
        const uint32_t lo = i < j ? i : j;
        const uint32_t hi = i < j ? j : i;

        CandidatePair out;
        out.a.type = CollidableType::Particle;
        out.a.react = particle_react;
        out.a.handle = lo;
        out.b.type = CollidableType::Particle;
        out.b.react = particle_react;
        out.b.handle = hi;
        out.stable_key = 0ull;

        out_pairs[off + k] = out;
    }
}

// Build the C2a stream from `survivors`, then collapse adjacent identical pairs
// (same stable_key after the canonical sort). Verbatim the C2b dedup posture: a
// no-op when there are no duplicates. Centralizes the download->dedup tail of
// both builders.
CandidatePairStream BuildAndDedup(const phi::DeviceContext& context,
                                  const std::vector<CandidatePair>& survivors) {
    CandidatePairStream stream_out = BuildCandidatePairStream(context, survivors);

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

} // namespace

CandidatePairStream BuildParticleRigidCandidatePairs(
    const phi::DeviceContext& context,
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::LbvhBroadphaseResult& rigid_tree,
    const uint32_t* rigid_shape_body_ids,
    float query_radius,
    const SystemPairMatrix& matrix) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    // --- SystemPairMatrix gate (testable seam) ------------------------------
    if (!scene::IsSystemPairEnabled(matrix, particle_system, SystemKind::Rigid)) {
        return BuildCandidatePairStream(context, {});
    }
    if (particle_count == 0u) {
        return BuildCandidatePairStream(context, {});
    }

    // --- raw cross-system query (already D1 CSR) ----------------------------
    auto query = gpu::QueryParticlesAgainstRigidLbvh(
        context, particle_positions, particle_count, query_radius, rigid_tree);
    const uint32_t total = query.TotalCandidates();
    if (total == 0u) {
        return BuildCandidatePairStream(context, {});
    }

    // Reaction-provider kinds read off the generated registry (no hard-coded enum).
    const ReactionProviderKind particle_react =
        constraint::GetCollidableTypeInfo(CollidableType::Particle).react;
    const ReactionProviderKind rigid_react =
        constraint::GetCollidableTypeInfo(CollidableType::RigidBody).react;

    // Upload the shape->body map (length == rigid shape count == LBVH leaf count).
    const uint32_t shape_count = rigid_tree.LeafCount();
    std::vector<uint32_t> body_ids(rigid_shape_body_ids,
                                   rigid_shape_body_ids + shape_count);
    phi::Buffer d_body_ids = phi::UploadVector(body_ids);

    // --- reframe every CSR entry into its offset slot (NO scan, NO atomic) ---
    phi::Buffer d_out(total * sizeof(CandidatePair), phi::MemoryKind::Device);
    const uint32_t blocks = (particle_count + kBlockSize - 1u) / kBlockSize;
    ReframeParticleRigidKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count,
        query.DeviceCandidateOffsets(),
        query.DeviceCandidateCounts(),
        query.DeviceCandidateIndices(),
        static_cast<const uint32_t*>(d_body_ids.Data()),
        particle_react, rigid_react,
        static_cast<CandidatePair*>(d_out.Data()));
    CheckCuda(cudaGetLastError(), "ReframeParticleRigidKernel launch");
    context.stream.Synchronize();

    std::vector<CandidatePair> survivors(total);
    d_out.CopyToHost(survivors.data(), total * sizeof(CandidatePair));

    return BuildAndDedup(context, survivors);
}

CandidatePairStream BuildParticleRigidCandidatePairs(
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::LbvhBroadphaseResult& rigid_tree,
    const uint32_t* rigid_shape_body_ids,
    float query_radius,
    const SystemPairMatrix& matrix) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildParticleRigidCandidatePairs(
        context, particle_positions, particle_count, particle_system, rigid_tree,
        rigid_shape_body_ids, query_radius, matrix);
}

CandidatePairStream BuildParticleParticleCandidatePairs(
    const phi::DeviceContext& context,
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::ParticleGridConfig& config,
    float query_radius,
    const SystemPairMatrix& matrix) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    // --- SystemPairMatrix gate (same-system, e.g. Fluid<->Fluid) ------------
    if (!scene::IsSystemPairEnabled(matrix, particle_system, particle_system)) {
        return BuildCandidatePairStream(context, {});
    }
    if (particle_count == 0u) {
        return BuildCandidatePairStream(context, {});
    }

    // --- raw uniform-grid neighbor query (already D1 CSR) --------------------
    auto grid = gpu::BuildParticleUniformGrid(
        context, particle_positions, particle_count, config, query_radius);
    const uint32_t total = grid.TotalNeighbors();
    if (total == 0u) {
        return BuildCandidatePairStream(context, {});
    }

    const ReactionProviderKind particle_react =
        constraint::GetCollidableTypeInfo(CollidableType::Particle).react;

    // --- reframe every neighbor (canonical min,max) into its offset slot -----
    phi::Buffer d_out(total * sizeof(CandidatePair), phi::MemoryKind::Device);
    const uint32_t blocks = (particle_count + kBlockSize - 1u) / kBlockSize;
    ReframeParticleParticleKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count,
        grid.DeviceNeighborOffsets(),
        grid.DeviceNeighborCounts(),
        grid.DeviceNeighborIndices(),
        particle_react,
        static_cast<CandidatePair*>(d_out.Data()));
    CheckCuda(cudaGetLastError(), "ReframeParticleParticleKernel launch");
    context.stream.Synchronize();

    std::vector<CandidatePair> survivors(total);
    d_out.CopyToHost(survivors.data(), total * sizeof(CandidatePair));

    // Canonical (i<j) emit means i->j and j->i are byte-identical pairs; the
    // adjacent dedup in BuildAndDedup collapses each double-listing to one i<j.
    return BuildAndDedup(context, survivors);
}

CandidatePairStream BuildParticleParticleCandidatePairs(
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::ParticleGridConfig& config,
    float query_radius,
    const SystemPairMatrix& matrix) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildParticleParticleCandidatePairs(
        context, particle_positions, particle_count, particle_system, config,
        query_radius, matrix);
}

} // namespace nuka::collision
