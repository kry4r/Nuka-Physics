#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu::CrossSystemQuery -- particle -> rigid candidate query
// (v0.7 p05, Task 7.5.3). The keystone of K2 SDF coupling.
//
// For each particle: build a world AABB (sphere of query_radius around the
// particle position), traverse the RIGID LBVH (built by BuildLbvhForQuery in
// p04+p05) for AABB overlap, and emit the candidate rigid body indices. Output
// is CSR-like: per-particle offset + flat candidate list. Used by p08 (SDF
// contact) to know which rigid SDFs to query per particle.
//
// 16-candidate cap (memory R-C, spec 7.5.5): at most kCrossSystemMaxCandidates
// candidates per particle (the LOWEST rigid-body indices; deterministic subset).
// A particle exceeding the cap increments TruncatedParticleCount().
//
// D1 (HARD): no global append atomic -- a two-pass count + exclusive-scan +
// private-slice fill produces per-particle slices that are sorted ascending by
// rigid body index. The only atomic is a uint32 truncation counter. Outputs are
// byte-identical run-to-run.
//
// HOST header: included from g++ TUs. The rigid tree is consumed via the p04
// LbvhBroadphaseResult (forward-declared LbvhNode pointer); no CUDA types here.
// ---------------------------------------------------------------------------

#include "collision/broadphase_lbvh.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace nuka::collision::gpu {

// Per-particle candidate cap (16, spec 7.5.5 memory R-C).
inline constexpr uint32_t kCrossSystemMaxCandidates = 16u;

// Result of a cross-system particle->rigid query. CSR-like: particle i's
// candidate rigid bodies occupy [CandidateOffsets()[i],
// CandidateOffsets()[i] + CandidateCounts()[i]) in the flat CandidateIndices()
// buffer, sorted ascending by rigid body index.
class CrossSystemQueryResult {
public:
    CrossSystemQueryResult() = default;
    CrossSystemQueryResult(uint32_t particle_count,
                           uint32_t candidate_capacity,
                           uint32_t total_candidates,
                           uint32_t truncated_particle_count,
                           phi::Buffer* candidate_offsets,
                           phi::Buffer* candidate_counts,
                           phi::Buffer* candidate_indices);
    ~CrossSystemQueryResult();

    CrossSystemQueryResult(const CrossSystemQueryResult&) = delete;
    CrossSystemQueryResult& operator=(const CrossSystemQueryResult&) = delete;
    CrossSystemQueryResult(CrossSystemQueryResult&& other) noexcept;
    CrossSystemQueryResult& operator=(CrossSystemQueryResult&& other) noexcept;

    uint32_t ParticleCount() const { return particle_count_; }
    uint32_t CandidateCapacity() const { return candidate_capacity_; }
    uint32_t TotalCandidates() const { return total_candidates_; }
    uint32_t TruncatedParticleCount() const { return truncated_particle_count_; }

    const uint32_t* DeviceCandidateOffsets() const;
    const uint32_t* DeviceCandidateCounts() const;
    const uint32_t* DeviceCandidateIndices() const;

    std::vector<uint32_t> DownloadCandidateOffsets() const;
    std::vector<uint32_t> DownloadCandidateCounts() const;
    std::vector<uint32_t> DownloadCandidateIndices() const;

private:
    uint32_t particle_count_ = 0;
    uint32_t candidate_capacity_ = 0;
    uint32_t total_candidates_ = 0;
    uint32_t truncated_particle_count_ = 0;
    phi::Buffer* candidate_offsets_ = nullptr;  // phi v2 handles; freed in the dtor.
    phi::Buffer* candidate_counts_ = nullptr;
    phi::Buffer* candidate_indices_ = nullptr;
};

// Query each of `particle_count` device-resident particle positions against the
// rigid LBVH in `rigid_tree` (which MUST have been built with retain_nodes, i.e.
// via BuildLbvhForQuery; rigid_tree.HasNodes() must be true). Each particle's
// query AABB is its position expanded by `query_radius`. Emits candidate rigid
// body indices (original body ids), capped + sorted per particle.
CrossSystemQueryResult QueryParticlesAgainstRigidLbvh(
    cudaStream_t stream, int device_id,
    const math::Vec3* particle_positions,
    uint32_t particle_count,
    float query_radius,
    const LbvhBroadphaseResult& rigid_tree);

CrossSystemQueryResult QueryParticlesAgainstRigidLbvh(
    const math::Vec3* particle_positions,
    uint32_t particle_count,
    float query_radius,
    const LbvhBroadphaseResult& rigid_tree);

} // namespace nuka::collision::gpu
