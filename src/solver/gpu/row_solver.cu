// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_solver implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/row_solver.cuh"

#include "math/cuda_vec_ops.cuh"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "solver/gpu/row_scheduler.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::solver::gpu {

namespace {

constexpr uint32_t kRowSolverBlockSize = 128u;

struct DeviceRowBuffers {
    constraint::Row* rows = nullptr;
    uint32_t* body_indices = nullptr;
    constraint::RowJacobian6* jacobian_data = nullptr;
    constraint::RowMaterial* materials = nullptr;
    constraint::RowAnchor* anchors = nullptr;
    uint32_t row_count = 0u;
    uint32_t body_index_count = 0u;
    uint32_t jacobian_data_count = 0u;
};

// Small-vector / quaternion primitives now come from the shared device math
// library (math/cuda_vec_ops.cuh). Bodies are bit-identical to the former local
// copies. The former local quaternion `Normalize` maps to
// QuatNormalizeSqrtLe(., 1e-12f); `Multiply` -> QuatMul; `Rotate` -> RotateShort
// (renamed at call sites). The former forward-declared local `Dot` is dropped;
// the shared Dot is used instead. UploadVector comes from
// phi/buffer_transfer.hpp.
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::Dot;
using mg::Length;
using mg::MakeQuat;
using mg::MakeVec3;
using mg::QuatMul;
using mg::RotateShort;
using mg::Scale;
using mg::Sub;

__device__ void ApplyAngularPositionCorrection(runtime::rigid::BodyState& body,
                                               math::Vec3 angular_jacobian,
                                               float position_impulse) {
    const math::Vec3 angular_delta =
        MakeVec3(angular_jacobian.x * body.inv_inertia.x * position_impulse,
                 angular_jacobian.y * body.inv_inertia.y * position_impulse,
                 angular_jacobian.z * body.inv_inertia.z * position_impulse);
    const float angle = Length(angular_delta);
    if (angle <= 1.0e-8f) {
        return;
    }

    const math::Vec3 axis = Scale(angular_delta, 1.0f / angle);
    const float half_angle = 0.5f * angle;
    const float sine = sinf(half_angle);
    const math::Quat dq =
        MakeQuat(cosf(half_angle), axis.x * sine, axis.y * sine, axis.z * sine);
    body.orientation = mg::QuatNormalizeSqrtLe(QuatMul(dq, body.orientation), 1.0e-12f);
}

__device__ uint32_t UMin(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

__device__ uint32_t UMax(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

__device__ bool ValidBody(uint32_t body, uint32_t body_count) {
    return body != constraint::kInvalidBodyIndex && body < body_count;
}

__device__ constraint::RowJacobian6 JacobianForRowBody(const DeviceRowBuffers& rows,
                                                       const constraint::Row& row,
                                                       uint32_t local_body_index) {
    constraint::RowJacobian6 result;
    result.linear = MakeVec3(0.0f, 0.0f, 0.0f);
    result.angular = MakeVec3(0.0f, 0.0f, 0.0f);
    if (local_body_index >= row.body_count) {
        return result;
    }
    const uint32_t index = row.jacobian_offset + local_body_index;
    if (index >= rows.jacobian_data_count) {
        return result;
    }
    return rows.jacobian_data[index];
}

__device__ uint32_t BodyForRowBody(const DeviceRowBuffers& rows,
                                   const constraint::Row& row,
                                   uint32_t local_body_index) {
    if (local_body_index >= row.body_count) {
        return constraint::kInvalidBodyIndex;
    }
    const uint32_t index = row.body_list_offset + local_body_index;
    if (index >= rows.body_index_count) {
        return constraint::kInvalidBodyIndex;
    }
    return rows.body_indices[index];
}

__device__ float ComputeJv(const DeviceRowBuffers& rows,
                           uint32_t row_index,
                           const runtime::rigid::BodyState* bodies,
                           uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    float jv = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const auto& body = bodies[body_index];
        jv += Dot(jacobian.linear, body.linear_velocity);
        jv += Dot(jacobian.angular, body.angular_velocity);
    }
    return jv;
}

__device__ float ComputeEffectiveMass(const DeviceRowBuffers& rows,
                                      uint32_t row_index,
                                      const runtime::rigid::BodyState* bodies,
                                      uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    float diagonal = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const auto& body = bodies[body_index];
        diagonal += body.inv_mass * Dot(jacobian.linear, jacobian.linear);
        diagonal += jacobian.angular.x * jacobian.angular.x * body.inv_inertia.x;
        diagonal += jacobian.angular.y * jacobian.angular.y * body.inv_inertia.y;
        diagonal += jacobian.angular.z * jacobian.angular.z * body.inv_inertia.z;
    }
    return diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
}

__device__ void ApplyVelocityImpulse(const DeviceRowBuffers& rows,
                                     uint32_t row_index,
                                     float delta_impulse,
                                     runtime::rigid::BodyState* bodies,
                                     uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        auto& body = bodies[body_index];
        if (body.inv_mass <= 0.0f) {
            continue;
        }
        body.linear_velocity = Add(
            body.linear_velocity,
            Scale(jacobian.linear, body.inv_mass * delta_impulse));
        body.angular_velocity.x += jacobian.angular.x * body.inv_inertia.x * delta_impulse;
        body.angular_velocity.y += jacobian.angular.y * body.inv_inertia.y * delta_impulse;
        body.angular_velocity.z += jacobian.angular.z * body.inv_inertia.z * delta_impulse;
    }
}

__device__ bool IsFrictionRow(const DeviceRowBuffers& rows, uint32_t row_index) {
    const auto& material = rows.materials[row_index];
    return material.kind == constraint::RowKind::Contact &&
           material.friction_row_count > 0u &&
           row_index >= material.first_friction_row &&
           row_index < material.first_friction_row + material.friction_row_count;
}

__device__ bool IsContactNormalRow(const DeviceRowBuffers& rows, uint32_t row_index) {
    const auto& material = rows.materials[row_index];
    return material.kind == constraint::RowKind::Contact &&
           row_index >= material.group_id &&
           row_index < material.group_id + UMax(material.normal_row_count, 1u);
}

__device__ float TotalNormalLambda(const DeviceRowBuffers& rows, uint32_t row_index) {
    const auto& material = rows.materials[row_index];
    if (material.kind != constraint::RowKind::Contact ||
        material.normal_row_count == 0u) {
        return 0.0f;
    }
    float total = 0.0f;
    const uint32_t end = UMin(material.group_id + material.normal_row_count,
                              rows.row_count);
    for (uint32_t row = material.group_id; row < end; ++row) {
        total += fmaxf(rows.rows[row].lambda, 0.0f);
    }
    return total;
}

__device__ void PrepareVelocityTargetRow(DeviceRowBuffers rows,
                                         uint32_t row_index,
                                         const runtime::rigid::BodyState* bodies,
                                         uint32_t body_count) {
    if (row_index >= rows.row_count || !IsContactNormalRow(rows, row_index)) {
        return;
    }

    const auto& material = rows.materials[row_index];
    if (material.restitution <= 0.0f) {
        return;
    }

    const float jv = ComputeJv(rows, row_index, bodies, body_count);
    if (jv < 0.0f) {
        rows.rows[row_index].rhs = fmaxf(rows.rows[row_index].rhs,
                                         -material.restitution * jv);
    }
}

__device__ void SolveVelocityRow(DeviceRowBuffers rows,
                                 uint32_t row_index,
                                 runtime::rigid::BodyState* bodies,
                                 uint32_t body_count) {
    constraint::Row& row = rows.rows[row_index];
    const float effective_mass = ComputeEffectiveMass(rows, row_index, bodies, body_count);
    const float jv = ComputeJv(rows, row_index, bodies, body_count);
    const float lambda = effective_mass * (row.rhs - jv);
    const float old_impulse = row.lambda;
    float lower = row.lower;
    float upper = row.upper;
    if (IsFrictionRow(rows, row_index)) {
        const float friction_limit =
            fmaxf(rows.materials[row_index].friction, 0.0f) *
            TotalNormalLambda(rows, row_index);
        lower = -friction_limit;
        upper = friction_limit;
    }

    const float new_impulse = fminf(fmaxf(old_impulse + lambda, lower), upper);
    row.lambda = new_impulse;
    const float delta = new_impulse - old_impulse;
    if (fabsf(delta) > 1.0e-12f) {
        ApplyVelocityImpulse(rows, row_index, delta, bodies, body_count);
    }
}

__device__ float SolvePositionRow(DeviceRowBuffers rows,
                                  uint32_t row_index,
                                  runtime::rigid::BodyState* bodies,
                                  uint32_t body_count,
                                  float slop,
                                  float baumgarte) {
    const constraint::Row& row = rows.rows[row_index];
    const auto& material = rows.materials[row_index];
    float error = 0.0f;
    bool apply_angular = false;
    math::Vec3 angular_a = MakeVec3(0.0f, 0.0f, 0.0f);
    math::Vec3 angular_b = MakeVec3(0.0f, 0.0f, 0.0f);

    if (IsContactNormalRow(rows, row_index)) {
        error = fmaxf(material.position_error, 0.0f);
    } else if (material.kind == constraint::RowKind::Joint) {
        const uint32_t body_a = BodyForRowBody(rows, row, 0u);
        const uint32_t body_b = BodyForRowBody(rows, row, 1u);
        const auto jacobian_a = JacobianForRowBody(rows, row, 0u);
        const auto jacobian_b = JacobianForRowBody(rows, row, 1u);
        math::Vec3 axis = jacobian_b.linear;
        if (Length(axis) <= 1.0e-8f) {
            axis = Scale(jacobian_a.linear, -1.0f);
        }
        if (Length(axis) <= 1.0e-8f) {
            return 0.0f;
        }

        math::Vec3 position_a = MakeVec3(0.0f, 0.0f, 0.0f);
        math::Vec3 position_b = MakeVec3(0.0f, 0.0f, 0.0f);
        math::Quat orientation_a = MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
        math::Quat orientation_b = MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
        if (ValidBody(body_a, body_count)) {
            position_a = bodies[body_a].position;
            orientation_a = bodies[body_a].orientation;
        }
        if (ValidBody(body_b, body_count)) {
            position_b = bodies[body_b].position;
            orientation_b = bodies[body_b].orientation;
        }
        const auto& anchor = rows.anchors[row_index];
        const math::Vec3 r_a = RotateShort(orientation_a, anchor.local_a);
        const math::Vec3 r_b = RotateShort(orientation_b, anchor.local_b);
        error = Dot(Sub(Add(position_a, r_a), Add(position_b, r_b)), axis);
        angular_a = Scale(Cross(r_a, axis), -1.0f);
        angular_b = Cross(r_b, axis);
        apply_angular = true;
    } else {
        return 0.0f;
    }

    const float correction = baumgarte *
        (error >= 0.0f ? fmaxf(error - slop, 0.0f) : fminf(error + slop, 0.0f));
    if (fabsf(correction) <= 1.0e-8f) {
        return fabsf(error);
    }

    const float effective_mass = ComputeEffectiveMass(rows, row_index, bodies, body_count);
    const float position_impulse = correction * effective_mass;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        auto& body = bodies[body_index];
        if (body.inv_mass <= 0.0f) {
            if (apply_angular) {
                const math::Vec3 angular = local == 0u ? angular_a : angular_b;
                ApplyAngularPositionCorrection(body, angular, position_impulse);
            }
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        body.position = Add(body.position,
                            Scale(jacobian.linear,
                                  body.inv_mass * position_impulse));
        if (apply_angular) {
            const math::Vec3 angular = local == 0u ? angular_a : angular_b;
            ApplyAngularPositionCorrection(body, angular, position_impulse);
        }
    }

    return fabsf(error);
}

__global__ void SolveRowsSweepKernel(DeviceRowBuffers rows,
                                     DeviceRowColorPartitions partitions,
                                     runtime::rigid::BodyState* bodies,
                                     uint32_t body_count,
                                     uint32_t velocity_iterations,
                                     uint32_t position_iterations,
                                     float slop,
                                     float baumgarte,
                                     float* max_error_out) {
    const uint32_t lane = threadIdx.x;

    for (uint32_t row_index = lane;
         row_index < rows.row_count;
         row_index += blockDim.x) {
        PrepareVelocityTargetRow(rows, row_index, bodies, body_count);
    }
    __syncthreads();

    for (uint32_t iter = 0u; iter < velocity_iterations; ++iter) {
        for (uint32_t color = 0u; color < partitions.color_count; ++color) {
            const RowColorRange range = partitions.color_ranges[color];
            for (uint32_t local = lane;
                 local < range.row_count;
                 local += blockDim.x) {
                SolveVelocityRow(rows,
                                 partitions.row_indices[range.row_offset + local],
                                 bodies,
                                 body_count);
            }
            __syncthreads();
        }
    }

    float local_error = 0.0f;
    for (uint32_t iter = 0u; iter < position_iterations; ++iter) {
        for (uint32_t color = 0u; color < partitions.color_count; ++color) {
            const RowColorRange range = partitions.color_ranges[color];
            for (uint32_t local = lane;
                 local < range.row_count;
                 local += blockDim.x) {
                local_error = fmaxf(
                    local_error,
                    SolvePositionRow(rows,
                                     partitions.row_indices[range.row_offset + local],
                                     bodies,
                                     body_count,
                                     slop,
                                     baumgarte));
            }
            __syncthreads();
        }
    }

    if (lane == 0u) {
        float max_error = local_error;
        for (uint32_t iter = 0u; iter < position_iterations; ++iter) {
            for (uint32_t color = 0u; color < partitions.color_count; ++color) {
                const RowColorRange range = partitions.color_ranges[color];
                for (uint32_t local = 0u; local < range.row_count; ++local) {
                    const uint32_t row_index =
                        partitions.row_indices[range.row_offset + local];
                    const auto& row = rows.rows[row_index];
                    const auto& material = rows.materials[row_index];
                    if (IsContactNormalRow(rows, row_index)) {
                        max_error = fmaxf(max_error,
                                          fmaxf(material.position_error, 0.0f));
                    } else if (material.kind == constraint::RowKind::Joint) {
                        const uint32_t body_a = BodyForRowBody(rows, row, 0u);
                        const uint32_t body_b = BodyForRowBody(rows, row, 1u);
                        const auto jacobian_a = JacobianForRowBody(rows, row, 0u);
                        const auto jacobian_b = JacobianForRowBody(rows, row, 1u);
                        math::Vec3 axis = jacobian_b.linear;
                        if (Length(axis) <= 1.0e-8f) {
                            axis = Scale(jacobian_a.linear, -1.0f);
                        }
                        if (Length(axis) <= 1.0e-8f) {
                            continue;
                        }
                        math::Vec3 position_a = MakeVec3(0.0f, 0.0f, 0.0f);
                        math::Vec3 position_b = MakeVec3(0.0f, 0.0f, 0.0f);
                        math::Quat orientation_a =
                            MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
                        math::Quat orientation_b =
                            MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
                        if (ValidBody(body_a, body_count)) {
                            position_a = bodies[body_a].position;
                            orientation_a = bodies[body_a].orientation;
                        }
                        if (ValidBody(body_b, body_count)) {
                            position_b = bodies[body_b].position;
                            orientation_b = bodies[body_b].orientation;
                        }
                        const auto& anchor = rows.anchors[row_index];
                        const math::Vec3 r_a =
                            RotateShort(orientation_a, anchor.local_a);
                        const math::Vec3 r_b =
                            RotateShort(orientation_b, anchor.local_b);
                        max_error = fmaxf(
                            max_error,
                            fabsf(Dot(Sub(Add(position_a, r_a),
                                         Add(position_b, r_b)),
                                      axis)));
                    }
                }
            }
        }
        *max_error_out = max_error;
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) +
                                 " failed: " +
                                 cudaGetErrorString(result));
    }
}

// UploadVector now comes from the shared host buffer-transfer header
// (phi/buffer_transfer.hpp); the former local copy was byte-identical.
// (The separate UploadToScratch helper below is out of scope and stays local.)
using ::nuka::phi::UploadVector;

struct RowSolverScratch {
    int device_id = -1;
    phi::Buffer rows;
    phi::Buffer body_indices;
    phi::Buffer jacobians;
    phi::Buffer materials;
    phi::Buffer anchors;
    phi::Buffer color_rows;
    phi::Buffer color_ranges;
    phi::Buffer bodies;
    phi::Buffer max_error;
    size_t rows_bytes = 0u;
    size_t body_indices_bytes = 0u;
    size_t jacobians_bytes = 0u;
    size_t materials_bytes = 0u;
    size_t anchors_bytes = 0u;
    size_t color_rows_bytes = 0u;
    size_t color_ranges_bytes = 0u;
    size_t bodies_bytes = 0u;
    size_t max_error_bytes = 0u;
};

RowSolverScratch& ThreadScratchForDevice(int device_id) {
    thread_local RowSolverScratch scratch;
    if (scratch.device_id != device_id) {
        scratch = RowSolverScratch{};
        scratch.device_id = device_id;
    }
    return scratch;
}

void EnsureScratchBuffer(phi::Buffer& buffer,
                         size_t& capacity_bytes,
                         size_t required_bytes) {
    if (capacity_bytes >= required_bytes) {
        return;
    }
    buffer = phi::Buffer(required_bytes, phi::MemoryKind::Device);
    capacity_bytes = required_bytes;
}

template <typename T>
void UploadToScratch(phi::Buffer& buffer,
                     size_t& capacity_bytes,
                     const std::vector<T>& values) {
    const size_t required_bytes = values.size() * sizeof(T);
    EnsureScratchBuffer(buffer, capacity_bytes, required_bytes);
    if (required_bytes > 0u) {
        buffer.CopyFromHost(values.data(), required_bytes);
    }
}

runtime::gpu::CudaConstraintRowSchedulerReport MakeSchedulerReport(
    uint32_t row_count,
    uint32_t color_count,
    const RowSolveConfig& config) {
    runtime::gpu::CudaConstraintRowBufferView view;
    view.kind = runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr;
    view.layout = runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr;
    view.schedule_mode =
        runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep;
    view.row_count = row_count;
    view.owner_count = color_count;
    view.rows_per_owner = 0u;
    view.row_stride_bytes = sizeof(constraint::Row);

    runtime::gpu::CudaConstraintRowSchedulerConfig scheduler_config;
    scheduler_config.iterations = config.velocity_iterations;
    return runtime::gpu::MakeCudaConstraintRowSchedulerReport(view,
                                                              scheduler_config);
}

} // namespace

RowSolveReport SolveRows(const phi::DeviceContext& context,
                         constraint::RowBuffers& rows,
                         std::vector<runtime::rigid::BodyState>& bodies,
                         const RowSolveConfig& config) {
    return SolveRows(context,
                     rows,
                     bodies.data(),
                     static_cast<uint32_t>(bodies.size()),
                     config);
}

RowSolveReport SolveRows(const phi::DeviceContext& context,
                         constraint::RowBuffers& rows,
                         runtime::rigid::BodyState* bodies,
                         uint32_t body_count,
                         const RowSolveConfig& config) {
    RowSolveReport report;
    report.row_count = rows.RowCount();
    report.velocity_iterations = config.velocity_iterations;
    report.position_iterations = config.position_iterations;
    if (rows.RowCount() == 0u || body_count == 0u || bodies == nullptr) {
        return report;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const RowColorPartitions partitions = BuildRowColorPartitions(rows);
    if (!ValidateNoSharedBodiesPerColor(rows, partitions)) {
        throw std::runtime_error("row scheduler produced a conflicting color partition");
    }

    report.color_count = partitions.ColorCount();
    report.row_scheduler_report =
        MakeSchedulerReport(rows.RowCount(), partitions.ColorCount(), config);

    auto& scratch = ThreadScratchForDevice(context.device_id);
    UploadToScratch(scratch.rows, scratch.rows_bytes, rows.rows);
    UploadToScratch(scratch.body_indices,
                    scratch.body_indices_bytes,
                    rows.body_indices);
    UploadToScratch(scratch.jacobians, scratch.jacobians_bytes, rows.jacobian_data);
    UploadToScratch(scratch.materials, scratch.materials_bytes, rows.materials);
    UploadToScratch(scratch.anchors, scratch.anchors_bytes, rows.anchors);
    UploadToScratch(scratch.color_rows,
                    scratch.color_rows_bytes,
                    partitions.row_indices);
    UploadToScratch(scratch.color_ranges,
                    scratch.color_ranges_bytes,
                    partitions.color_ranges);
    const size_t bodies_bytes =
        body_count * sizeof(runtime::rigid::BodyState);
    EnsureScratchBuffer(scratch.bodies, scratch.bodies_bytes, bodies_bytes);
    scratch.bodies.CopyFromHost(bodies, bodies_bytes);
    EnsureScratchBuffer(scratch.max_error,
                        scratch.max_error_bytes,
                        sizeof(float));
    float zero = 0.0f;
    scratch.max_error.CopyFromHost(&zero, sizeof(float));

    DeviceRowBuffers device_view;
    device_view.rows = static_cast<constraint::Row*>(scratch.rows.Data());
    device_view.body_indices =
        static_cast<uint32_t*>(scratch.body_indices.Data());
    device_view.jacobian_data =
        static_cast<constraint::RowJacobian6*>(scratch.jacobians.Data());
    device_view.materials =
        static_cast<constraint::RowMaterial*>(scratch.materials.Data());
    device_view.anchors =
        static_cast<constraint::RowAnchor*>(scratch.anchors.Data());
    device_view.row_count = rows.RowCount();
    device_view.body_index_count = rows.BodyIndexCount();
    device_view.jacobian_data_count = rows.JacobianDataCount();

    DeviceRowColorPartitions device_partitions;
    device_partitions.row_indices =
        static_cast<const uint32_t*>(scratch.color_rows.Data());
    device_partitions.color_ranges =
        static_cast<const RowColorRange*>(scratch.color_ranges.Data());
    device_partitions.color_count = partitions.ColorCount();

    constexpr uint32_t kBlockSize = kRowSolverBlockSize;
    const cudaStream_t stream = context.stream.Native();
    SolveRowsSweepKernel<<<1u, kBlockSize, 0, stream>>>(
        device_view,
        device_partitions,
        static_cast<runtime::rigid::BodyState*>(scratch.bodies.Data()),
        body_count,
        config.velocity_iterations,
        config.position_iterations,
        config.slop,
        config.baumgarte,
        static_cast<float*>(scratch.max_error.Data()));
    CheckCuda(cudaGetLastError(), "SolveRowsSweepKernel launch");
    ++report.row_scheduler_report.solver_launch_count;

    CheckCuda(cudaStreamSynchronize(stream), "RowSolver stream synchronize");
    scratch.bodies.CopyToHost(bodies, bodies_bytes);
    scratch.rows.CopyToHost(rows.rows.data(),
                            rows.rows.size() * sizeof(constraint::Row));
    scratch.max_error.CopyToHost(&report.max_position_error, sizeof(float));
    report.row_scheduler_report.executed_iterations = config.velocity_iterations;
    report.row_scheduler_report.active_row_count = rows.RowCount();
    report.row_scheduler_report.normal_impulse_count = rows.RowCount();
    return report;
}

} // namespace nuka::solver::gpu
