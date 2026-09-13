#include <cuda_runtime.h>

#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/kinematics.cuh"
#include "sensor/noise/measurement_error.hpp"

namespace nuka::phi {
namespace {

constexpr uint32_t kObservationBlockSize = 256u;

namespace mg = math::gpu;

__device__ math::Quat InverseRotation(math::Quat q) { return mg::MakeQuat(q.w, -q.x, -q.y, -q.z); }

__device__ math::Quat RotationMidpoint(math::Quat a, math::Quat b) {
    const float sign = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z < 0.0f ? -1.0f : 1.0f;
    return mg::QuatNormalizeRsqrt(mg::MakeQuat(a.w + sign * b.w, a.x + sign * b.x,
        a.y + sign * b.y, a.z + sign * b.z), 1.0e-12f);
}

__device__ math::Vec3 RotationLog(math::Quat q) {
    if (q.w < 0.0f) q = mg::MakeQuat(-q.w, -q.x, -q.y, -q.z);
    const float sine = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
    const float factor = sine > 1.0e-8f ? 2.0f * atan2f(sine, q.w) / sine : 2.0f;
    return mg::MakeVec3(q.x * factor, q.y * factor, q.z * factor);
}

__device__ math::Quat RotationExp(math::Vec3 vector) {
    const float angle = sqrtf(mg::Dot(vector, vector));
    const float factor = angle > 1.0e-8f ? sinf(0.5f * angle) / angle : 0.5f;
    return mg::MakeQuat(cosf(0.5f * angle), vector.x * factor, vector.y * factor, vector.z * factor);
}

__global__ void ReadoutLinkMotionKernel(nkops::ArticulationDeviceState state, ReadoutMotionParams params) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= state.articulation_count || threadIdx.x != 0u) return;
    const uint32_t env = articulation / params.articulations_per_env;
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t frame_offset = env * (params.links_per_env + params.bodies_per_env);
    for (uint32_t local = 0u; local < state.articulation_link_count[articulation]; ++local) {
        const uint32_t link = offset + local;
        const uint32_t parent = state.parent_link[link];
        const auto type = state.joint_type[link];
        sensor::MotionFrame frame{};
        if (parent == ~0u && type == ArticulationJointType::FloatingBase) {
            frame.pose = state.base_pose[articulation];
            const auto v = state.link_velocity[link];
            frame.angular_velocity = mg::RotateByQuatNormalized(frame.pose.rotation, {v.v[0], v.v[1], v.v[2]});
            frame.linear_velocity = mg::RotateByQuatNormalized(frame.pose.rotation, {v.v[3], v.v[4], v.v[5]});
        } else {
            const auto relative = nkops::JointRelativeFrame(type, state.joint_axis[link],
                state.link_local_pose[link], state.parent_offset[link], state.q[link]);
            if (parent == ~0u) frame.pose = relative;
            else {
                const auto p = params.frames[frame_offset + offset + parent - env * params.links_per_env];
                frame.pose = nkops::ComposeFrame(p.pose, relative);
                frame.angular_velocity = p.angular_velocity;
                frame.linear_velocity = mg::Add(p.linear_velocity,
                    mg::Cross(p.angular_velocity, mg::Sub(frame.pose.position, p.pose.position)));
            }
            const auto axis = mg::RotateByQuatNormalized(frame.pose.rotation, nkops::JointUnitAxis(state.joint_axis[link]));
            if (type == ArticulationJointType::Revolute)
                frame.angular_velocity = mg::Add(frame.angular_velocity, mg::Scale(axis, state.qdot[link]));
            if (type == ArticulationJointType::Prismatic)
                frame.linear_velocity = mg::Add(frame.linear_velocity, mg::Scale(axis, state.qdot[link]));
        }
        params.frames[frame_offset + link - env * params.links_per_env] = frame;
    }
}

__global__ void ReadoutBodyMotionKernel(DataView data, ReadoutMotionParams params) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.env_count * params.bodies_per_env) return;
    const uint32_t env = index / params.bodies_per_env;
    sensor::MotionFrame frame;
    frame.pose = data.body_pose[index];
    frame.angular_velocity = data.body_angular_velocity[index];
    const auto center_offset = mg::RotateByQuatNormalized(frame.pose.rotation, data.body_inertial_frame[index].position);
    frame.linear_velocity = mg::Sub(data.body_linear_velocity[index], mg::Cross(frame.angular_velocity, center_offset));
    params.frames[env * (params.links_per_env + params.bodies_per_env) + params.links_per_env +
        index % params.bodies_per_env] = frame;
}

__device__ sensor::MotionFrame MountedFrame(sensor::MotionFrame frame, math::Transform offset) {
    const auto arm = mg::RotateByQuatNormalized(frame.pose.rotation, offset.position);
    frame.linear_velocity = mg::Add(frame.linear_velocity, mg::Cross(frame.angular_velocity, arm));
    frame.pose = nkops::ComposeFrame(frame.pose, offset);
    return frame;
}

__device__ double SensorUniform(uint64_t seed, uint32_t env, uint64_t sequence, uint32_t cause) {
    auto counter = sensor::noise::MakeCounter(env, sequence);
    counter.v[3] = cause;
    const auto random = sensor::noise::Philox4x32_10(counter, sensor::noise::SplitSeed(seed));
    return (static_cast<double>(random.v[0]) + 0.5) * 0x1p-32;
}

__global__ void SampleStateSensorKernel(DataView data, SampleStateSensorParams p) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= p.env_count) return;
    auto& state = p.runtime[env];
    const auto& desc = p.desc;
    const double start_time = p.times[env];
    const double time = start_time + p.interval;
    if (!state.initialized) {
        state.next_sample_time = start_time + desc.sample_period;
        state.initialized = 1u;
    }
    const uint32_t frame_index = env * p.frames_per_env + desc.index +
        (desc.mount == sensor::StateSensorMount::Body ? p.links_per_env : 0u);
    const auto before = MountedFrame(p.before[frame_index], desc.local_offset);
    const auto after = MountedFrame(p.after[frame_index], desc.local_offset);
    if (desc.kind == sensor::StateSensorKind::Imu) {
        const auto orientation = RotationMidpoint(before.pose.rotation, after.pose.rotation);
        const auto impulse = mg::Sub(mg::Sub(after.linear_velocity, before.linear_velocity),
                                     mg::Scale(p.gravity, static_cast<float>(p.interval)));
        const auto specific_impulse = mg::RotateByQuatNormalized(InverseRotation(orientation), impulse);
        const auto angular_integral = mg::RotateByQuatNormalized(InverseRotation(orientation),
            mg::Scale(mg::Add(before.angular_velocity, after.angular_velocity), static_cast<float>(0.5 * p.interval)));
        const float values[] = {specific_impulse.x, specific_impulse.y, specific_impulse.z,
            angular_integral.x, angular_integral.y, angular_integral.z};
        for (uint32_t component = 0u; component < sensor::kStateSensorErrorChannels; ++component)
            state.integral[component] += values[component];
    }
    state.exposure += p.interval;
    const double tolerance = 1.0e-10 * fmax(1.0, fabs(time));
    if (time + tolerance >= state.next_sample_time) {
        sensor::StateSensorPacket packet{};
        float truth[sensor::kStateSensorErrorChannels] = {};
        if (desc.kind == sensor::StateSensorKind::Imu)
            for (uint32_t c = 0u; c < sensor::kStateSensorErrorChannels; ++c)
                truth[c] = static_cast<float>(state.integral[c] / state.exposure);
        if (desc.kind == sensor::StateSensorKind::FramePose) {
            truth[0] = after.pose.position.x;
            truth[1] = after.pose.position.y;
            truth[2] = after.pose.position.z;
        }
        if (desc.kind == sensor::StateSensorKind::JointState) {
            truth[0] = data.q[env * p.links_per_env + desc.index];
            truth[1] = data.qdot[env * p.links_per_env + desc.index];
        }
        if (desc.kind == sensor::StateSensorKind::LinearVelocity) {
            const auto velocity = mg::RotateByQuatNormalized(InverseRotation(after.pose.rotation), after.linear_velocity);
            truth[0] = velocity.x; truth[1] = velocity.y; truth[2] = velocity.z;
        }
        const uint32_t components = p.value_count < sensor::kStateSensorErrorChannels ? p.value_count : sensor::kStateSensorErrorChannels;
        const uint64_t sequence = state.stamp.acquisitions;
        const uint64_t seed = desc.seed ^ (0xa0761d6478bd642full * (uint64_t{p.channel} + 1u));
        for (uint32_t c = 0u; c < components; ++c) {
            auto config = desc.errors[c];
            config.noise.seed ^= seed;
            packet.values[c] = sensor::noise::MeasureValue(truth[c], config,
                p.noise + env * sensor::kStateSensorErrorChannels + c,
                env, c, sequence, state.exposure, desc.temperature);
        }
        if (desc.kind == sensor::StateSensorKind::FramePose) {
            math::Quat filtered = after.pose.rotation;
            if (sequence != 0u) {
                const auto delta = RotationLog(mg::QuatMul(InverseRotation(state.filtered_orientation), filtered));
                const float d[] = {delta.x, delta.y, delta.z};
                float increment[3];
                bool response = false;
                for (uint32_t c = 0u; c < 3u; ++c) {
                    const float tau = desc.errors[c + 3u].error.response_time;
                    response = response || tau > 0.0f;
                    increment[c] = d[c] * (tau > 0.0f ? static_cast<float>(-expm1(-state.exposure / tau)) : 1.0f);
                }
                if (response) filtered = mg::QuatNormalizeRsqrt(mg::QuatMul(state.filtered_orientation,
                    RotationExp({increment[0], increment[1], increment[2]})), 1.0e-12f);
            }
            state.filtered_orientation = filtered;
            const auto error = mg::MakeVec3(packet.values[3], packet.values[4], packet.values[5]);
            if (mg::Dot(error, error) != 0.0f)
                filtered = mg::QuatNormalizeRsqrt(mg::QuatMul(filtered, RotationExp(error)), 1.0e-12f);
            packet.values[3] = filtered.w;
            packet.values[4] = filtered.x;
            packet.values[5] = filtered.y;
            packet.values[6] = filtered.z;
        }
        ++state.stamp.acquisitions;
        packet.sequence = state.stamp.acquisitions;
        packet.sample_time = time;
        const double delay = fmax(0.0, desc.latency + desc.latency_jitter *
            (2.0 * SensorUniform(seed, env, sequence, 11u) - 1.0));
        packet.available_time = fmax(time + delay, state.last_available_time);
        if (SensorUniform(seed, env, sequence, 12u) < desc.dropout_probability) ++state.stamp.dropped;
        else if (state.queue_size < p.queue_capacity) {
            const uint32_t slot = (state.queue_begin + state.queue_size) % p.queue_capacity;
            p.queue[env * p.queue_capacity + slot] = packet;
            ++state.queue_size;
            state.last_available_time = packet.available_time;
        } else atomicOr(data.env_status + env, kEnvStatusSensorQueueOverflow);
        const double periods = fmax(1.0, floor((time + tolerance - state.next_sample_time) / desc.sample_period) + 1.0);
        state.next_sample_time += periods * desc.sample_period;
        state.exposure = 0.0;
        for (double& integral : state.integral) integral = 0.0;
    }
    while (state.queue_size) {
        const auto& packet = p.queue[env * p.queue_capacity + state.queue_begin];
        if (packet.available_time > time + tolerance) break;
        for (uint32_t c = 0u; c < p.value_count; ++c) p.values[env * p.value_count + c] = packet.values[c];
        state.stamp.sequence = packet.sequence;
        state.stamp.sample_time = packet.sample_time;
        state.stamp.delivery_time = time;
        state.stamp.valid = 1u;
        state.queue_begin = (state.queue_begin + 1u) % p.queue_capacity;
        --state.queue_size;
    }
}

__global__ void AdvanceSensorTimeKernel(AdvanceSensorTimeParams p) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < p.env_count) p.times[env] += p.interval;
}

Status OpReadoutMotion(const ModelView& model, const DataView& data, const void* arguments, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutMotionParams*>(arguments);
    if (!p || !p->frames || !p->env_count) return Status::InvalidArgument;
    if (p->links_per_env) {
        auto state = nkops::MakeArticulationDeviceState(model, data, p->env_count * p->links_per_env,
            p->env_count * p->articulations_per_env);
        LaunchCuda(ReadoutLinkMotionKernel, dim3(state.articulation_count), dim3(32u), 0u, stream, state, *p);
    }
    if (p->bodies_per_env)
        LaunchCuda(ReadoutBodyMotionKernel, dim3((uint64_t{p->env_count} * p->bodies_per_env + kObservationBlockSize - 1u) /
            kObservationBlockSize), dim3(kObservationBlockSize), 0u, stream, data, *p);
    return cudaPeekAtLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpSampleStateSensor(const ModelView&, const DataView& data, const void* arguments, cudaStream_t stream) {
    const auto* p = static_cast<const SampleStateSensorParams*>(arguments);
    if (!p || !p->before || !p->after || !p->times || !p->values || !p->runtime || !p->noise ||
        !p->queue || !p->queue_capacity || !p->env_count) return Status::InvalidArgument;
    LaunchCuda(SampleStateSensorKernel, dim3((uint64_t{p->env_count} + kObservationBlockSize - 1u) /
        kObservationBlockSize), dim3(kObservationBlockSize), 0u, stream, data, *p);
    return cudaPeekAtLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpAdvanceSensorTime(const ModelView&, const DataView&, const void* arguments, cudaStream_t stream) {
    const auto* p = static_cast<const AdvanceSensorTimeParams*>(arguments);
    if (!p || !p->times || !p->env_count) return Status::InvalidArgument;
    LaunchCuda(AdvanceSensorTimeKernel, dim3((uint64_t{p->env_count} + kObservationBlockSize - 1u) /
        kObservationBlockSize), dim3(kObservationBlockSize), 0u, stream, *p);
    return cudaPeekAtLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

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
    SetCudaOp(NkOp::ReadoutMotion, &OpReadoutMotion);
    SetCudaOp(NkOp::SampleStateSensor, &OpSampleStateSensor);
    SetCudaOp(NkOp::AdvanceSensorTime, &OpAdvanceSensorTime);
}

}  // namespace nuka::phi
