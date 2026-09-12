#include <cuda_runtime.h>

#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "sensor/noise/measurement_error.hpp"

namespace nuka::phi {
namespace {

constexpr uint32_t kObservationBlockSize = 256u;

__global__ void SampleObservationKernel(SampleObservationParams params) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.env_count * params.values_per_env) return;
    const uint32_t env = index / params.values_per_env;
    params.values[index] = sensor::noise::MeasureValue(params.source[index], params.config,
        params.noise_state + index, index, params.channel, params.stamps[env].sequence,
        params.sample_interval, params.temperature);
}

__global__ void AdvanceObservationKernel(SampleObservationParams params) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= params.env_count) return;
    auto& stamp = params.stamps[env];
    ++stamp.sequence;
    stamp.elapsed_time += params.sample_interval;
    stamp.valid = 1u;
}

__global__ void ResetObservationKernel(ResetObservationParams params) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.selected_count * params.values_per_env) return;
    const uint32_t local = index % params.values_per_env;
    const uint32_t env = params.env_ids[index / params.values_per_env];
    const uint32_t offset = env * params.values_per_env + local;
    params.values[offset] = 0.0f;
    params.noise_state[offset] = {};
    if (local == 0u) params.stamps[env] = {};
}

Status OpSampleObservation(const ModelView&, const DataView&, const void* arguments, cudaStream_t stream) {
    const auto* p = static_cast<const SampleObservationParams*>(arguments);
    if (!p || !p->source || !p->values || !p->noise_state || !p->stamps ||
        !p->env_count || !p->values_per_env || !(p->sample_interval > 0.0))
        return Status::InvalidArgument;
    const uint64_t count = uint64_t{p->env_count} * p->values_per_env;
    if (count > UINT32_MAX) return Status::InvalidArgument;
    const uint32_t blocks = static_cast<uint32_t>((count + kObservationBlockSize - 1u) / kObservationBlockSize);
    LaunchCuda(SampleObservationKernel, dim3(blocks), dim3(kObservationBlockSize), 0u, stream, *p);
    LaunchCuda(AdvanceObservationKernel,
        dim3(static_cast<uint32_t>((uint64_t{p->env_count} + kObservationBlockSize - 1u) / kObservationBlockSize)),
        dim3(kObservationBlockSize), 0u, stream, *p);
    return cudaPeekAtLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpResetObservation(const ModelView&, const DataView&, const void* arguments, cudaStream_t stream) {
    const auto* p = static_cast<const ResetObservationParams*>(arguments);
    if (!p || !p->values || !p->noise_state || !p->stamps || !p->env_ids || !p->values_per_env)
        return Status::InvalidArgument;
    const uint64_t count = uint64_t{p->selected_count} * p->values_per_env;
    if (count > UINT32_MAX) return Status::InvalidArgument;
    if (!count) return Status::Ok;
    const uint32_t blocks = static_cast<uint32_t>((count + kObservationBlockSize - 1u) / kObservationBlockSize);
    LaunchCuda(ResetObservationKernel, dim3(blocks), dim3(kObservationBlockSize), 0u, stream, *p);
    return cudaPeekAtLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkSensorOps() {
    SetCudaOp(NkOp::SampleObservation, &OpSampleObservation);
    SetCudaOp(NkOp::ResetObservation, &OpResetObservation);
}

}  // namespace nuka::phi
