#pragma once

__global__ void InitializeArticulationBlocksKernel(
    ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= p.articulation_count || threadIdx.x != 0u) return;
    const size_t base = size_t{articulation} * p.max_dof;
    uint32_t dimension = 0u;
    for (uint32_t dof = 0u; dof < p.max_dof; ++dof) {
        uint32_t link, component;
        if (!ArticulationDofLocation(state, articulation, dof, &link, &component)) continue;
        if (p.mimic_couplings != 0u && data.mimic_root_dof[base + dof] != dof) continue;
        s.articulation_roots[base + dimension++] = dof;
    }
    s.articulation_dimensions[articulation] = dimension;
    for (uint32_t compact = dimension; compact < p.max_dof; ++compact)
        s.articulation_roots[base + compact] = ~0u;
}

// An articulation block in its free (mimic root) coordinates.
struct ArticulationBlock {
    static constexpr uint32_t kMaxDof = kMaxArticulationDof;
    uint32_t index, dimension, begin, end, stride;
    const uint32_t* roots;
    const float* mass;
    const float* snapshot;
    const float* free;
    float* velocity;
    float* matrix;
    float* force;
    float* direction;
    float* diagonal;
    __device__ float Mass(uint32_t i, uint32_t j) const { return mass[size_t{roots[i]} * stride + roots[j]]; }
    __device__ float Snapshot(uint32_t i) const { return snapshot[roots[i]]; }
    __device__ float Free(uint32_t i) const { return free[roots[i]]; }
    __device__ float Jacobian(DataView data, BlockDescentSolveParams p, uint32_t row_slot, uint32_t i) const {
        return ArticulationJacobian(data, p, row_slot, index, roots[i]);
    }
};
static_assert(kMaxArticulationDof <= kThreads);

__global__ void DescendArticulationsKernel(
    ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= p.articulation_count) return;
    const uint32_t owner = p.total_particle_count + p.total_body_count + articulation;
    const size_t base = size_t{articulation} * p.max_dof;
    ArticulationBlock b{};
    b.index = articulation;
    b.dimension = s.articulation_dimensions[articulation];
    b.begin = s.offsets[owner];
    b.end = s.offsets[owner + 1u];
    b.stride = p.max_dof;
    b.roots = s.articulation_roots + base;
    b.mass = (p.mimic_couplings != 0u ? data.m_reduced : data.m) + base * p.max_dof;
    b.snapshot = s.articulation_snapshot + base;
    b.free = s.articulation_free + base;
    b.velocity = s.articulation_velocity + base;
    b.matrix = s.articulation_matrix + base * p.max_dof;
    b.force = s.articulation_force + base;
    b.direction = s.articulation_direction + base;
    b.diagonal = s.articulation_diagonal + base;
    if (b.dimension == 0u) return;
    const nk::BlockSolveFailure failure = DescendDenseBlock(b, data, p, s);

    const uint32_t dof = threadIdx.x;
    uint32_t link = 0u, component = 0u;
    const bool located = dof < p.max_dof && ArticulationDofLocation(state, articulation, dof, &link, &component);
    double value = 0.0;
    if (located) {
        const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + dof] : dof;
        for (uint32_t compact = 0u; compact < b.dimension; ++compact) {
            if (b.roots[compact] == root) {
                value = b.velocity[compact];
                break;
            }
        }
        if (p.mimic_couplings != 0u) value *= data.mimic_root_scale[base + dof];
    }
    if (__syncthreads_or(!(value >= -double(FLT_MAX) && value <= double(FLT_MAX)))) {
        if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner,
            failure != nk::BlockSolveFailure::None ? failure : nk::BlockSolveFailure::InvalidCandidate);
        return;
    }
    if (failure != nk::BlockSolveFailure::None && threadIdx.x == 0u)
        RecordBlockFailure(data, p, s, owner, failure);
    if (dof >= p.max_dof) return;
    const float next = static_cast<float>(value);
    data.qdot_flat[base + dof] = next;
    if (!located) return;
    if (component == ~0u) state.qdot[link] = next;
    else state.link_velocity[link].v[component] = next;
}

__global__ void WriteBlockJointLimitImpulsesKernel(DataView data, BlockDescentSolveParams p) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint64_t total_links = uint64_t{p.base_link_count} * p.env_count;
    for (uint64_t link = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; link < total_links;
         link += uint64_t{gridDim.x} * blockDim.x) {
        const uint32_t env = static_cast<uint32_t>(link / p.base_link_count);
        const uint32_t local_link = static_cast<uint32_t>(link % p.base_link_count);
        const size_t lower = size_t{env} * p.rows_per_env + p.contact_rows_per_env + size_t{local_link} * 2u;
        data.joint_limit_impulse[2u * link] =
            (rows[lower].flags & nk::nk_row_flags::kActive) != 0u ? data.lambda[lower] : 0.0f;
        data.joint_limit_impulse[2u * link + 1u] =
            (rows[lower + 1u].flags & nk::nk_row_flags::kActive) != 0u ? data.lambda[lower + 1u] : 0.0f;
    }
}
