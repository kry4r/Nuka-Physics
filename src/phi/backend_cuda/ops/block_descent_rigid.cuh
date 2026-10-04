#pragma once

__global__ void InitializeRigidBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t body = blockIdx.x;
    if (body >= p.total_body_count || threadIdx.x != 0u) return;
    const size_t base = size_t{body} * kRigidBlockDof;
    const Vec3 linear = data.body_linear_velocity[body];
    const Vec3 angular = data.body_angular_velocity[body];
    s.rigid_free[base] = s.rigid_snapshot[base] = linear.x;
    s.rigid_free[base + 1u] = s.rigid_snapshot[base + 1u] = linear.y;
    s.rigid_free[base + 2u] = s.rigid_snapshot[base + 2u] = linear.z;
    s.rigid_free[base + 3u] = s.rigid_snapshot[base + 3u] = angular.x;
    s.rigid_free[base + 4u] = s.rigid_snapshot[base + 4u] = angular.y;
    s.rigid_free[base + 5u] = s.rigid_snapshot[base + 5u] = angular.z;
    s.rigid_dimensions[body] = 0u;
    const float inverse_mass = data.body_inv_mass[body];
    const uint32_t owner = p.total_particle_count + body;
    if (!Finite(linear) || !Finite(angular)) {
        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidMass);
        return;
    }
    if (inverse_mass == 0.0f) return;
    const double mass = 1.0 / static_cast<double>(inverse_mass);
    SymmetricMat3 inertia{};
    const bool valid = inverse_mass > 0.0f && inverse_mass <= FLT_MAX &&
        mass > 0.0 && mass <= double(FLT_MAX) &&
        nk::vbd::Invert(data.body_world_inv_inertia[body], 0.0f, &inertia) &&
        inertia.xx > 0.0f && inertia.yy > 0.0f && inertia.zz > 0.0f &&
        double(inertia.xx) * inertia.yy > double(inertia.xy) * inertia.xy &&
        Finite({inertia.xx, inertia.yy, inertia.zz}) &&
        Finite({inertia.xy, inertia.xz, inertia.yz});
    if (!valid) {
        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidMass);
        return;
    }
    const float rotational[3][3] = {
        {inertia.xx, inertia.xy, inertia.xz},
        {inertia.xy, inertia.yy, inertia.yz},
        {inertia.xz, inertia.yz, inertia.zz}};
    for (uint32_t row = 0u; row < kRigidBlockDof; ++row) {
        for (uint32_t column = 0u; column < kRigidBlockDof; ++column) {
            float value = 0.0f;
            if (row < 3u && column < 3u && row == column) value = static_cast<float>(mass);
            if (row >= 3u && column >= 3u) value = rotational[row - 3u][column - 3u];
            s.rigid_mass[base * kRigidBlockDof + size_t{row} * kRigidBlockDof + column] = value;
        }
    }
    s.rigid_dimensions[body] = kRigidBlockDof;
}

struct RigidBlock {
    static constexpr uint32_t kMaxDof = kRigidBlockDof;
    uint32_t index, dimension, begin, end, stride;
    const float* mass;
    const float* snapshot;
    const float* free;
    float* velocity;
    float* matrix;
    float* force;
    float* direction;
    float* diagonal;
    __device__ float Mass(uint32_t i, uint32_t j) const { return mass[i * kRigidBlockDof + j]; }
    __device__ float Snapshot(uint32_t i) const { return snapshot[i]; }
    __device__ float Free(uint32_t i) const { return free[i]; }
    __device__ float Jacobian(DataView data, BlockDescentSolveParams, uint32_t row_slot, uint32_t i) const {
        return RigidJacobian(data, row_slot, index, i);
    }
};

__global__ void DescendRigidBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t body = blockIdx.x;
    if (body >= p.total_body_count || s.rigid_dimensions[body] == 0u) return;
    const uint32_t owner = p.total_particle_count + body;
    const size_t base = size_t{body} * kRigidBlockDof;
    RigidBlock b{};
    b.index = body;
    b.dimension = s.rigid_dimensions[body];
    b.begin = s.offsets[owner];
    b.end = s.offsets[owner + 1u];
    b.stride = kRigidBlockDof;
    b.mass = s.rigid_mass + base * kRigidBlockDof;
    b.snapshot = s.rigid_snapshot + base;
    b.free = s.rigid_free + base;
    b.velocity = s.rigid_velocity + base;
    b.matrix = s.rigid_matrix + base * kRigidBlockDof;
    b.force = s.rigid_force + base;
    b.direction = s.rigid_direction + base;
    b.diagonal = s.rigid_diagonal + base;
    const nk::BlockSolveFailure failure = DescendDenseBlock(b, data, p, s);
    if (threadIdx.x != 0u) return;
    if (failure != nk::BlockSolveFailure::None) RecordBlockFailure(data, p, s, owner, failure);
    data.body_linear_velocity[body] = {b.velocity[0], b.velocity[1], b.velocity[2]};
    data.body_angular_velocity[body] = {b.velocity[3], b.velocity[4], b.velocity[5]};
}
