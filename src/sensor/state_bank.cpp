#include "sensor/state_bank.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "nk/model/model.hpp"
#include "nk/model/generated/views.hpp"
#include "phi/articulation_contract.hpp"

namespace nuka::sensor {

uint32_t StateSensorValueCount(StateSensorKind kind) {
    switch (kind) {
        case StateSensorKind::Imu: return 6u;
        case StateSensorKind::FramePose: return 7u;
        case StateSensorKind::JointState: return 2u;
        case StateSensorKind::LinearVelocity: return 3u;
    }
    return 0u;
}

bool ValidStateSensorDesc(const StateSensorDesc& desc) {
    const auto& p = desc.local_offset.position;
    const auto& q = desc.local_offset.rotation;
    const double norm = double{q.w} * q.w + double{q.x} * q.x + double{q.y} * q.y + double{q.z} * q.z;
    if (!StateSensorValueCount(desc.kind) || !desc.update_period || !std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(p.z) || !std::isfinite(norm) || norm < 1.0e-12 ||
        !std::isfinite(desc.sample_period) || desc.sample_period < 0.0 ||
        !std::isfinite(desc.latency) || desc.latency < 0.0 ||
        !std::isfinite(desc.latency_jitter) || desc.latency_jitter < 0.0 ||
        !std::isfinite(desc.dropout_probability) || desc.dropout_probability < 0.0f ||
        desc.dropout_probability > 1.0f || !std::isfinite(desc.temperature)) return false;
    for (uint32_t channel = 0u; channel < kStateSensorErrorChannels; ++channel) {
        if (!ValidObservationConfig(desc.errors[channel])) return false;
        if (desc.kind == StateSensorKind::FramePose && channel >= 3u &&
            desc.errors[channel].error.scale_error != 0.0f) return false;
    }
    return true;
}

struct StateSensorBank::Channel {
    phi::Buffer* storage = nullptr;
    size_t bytes = 0u;
    size_t runtime_offset = 0u;
    size_t noise_offset = 0u;
    size_t queue_offset = 0u;
    bool active = true;
    phi::SampleStateSensorParams params;
    ~Channel() { if (storage) phi::BufferFree(storage); }
};

StateSensorBank::StateSensorBank() = default;
StateSensorBank::~StateSensorBank() {
    if (motion_storage_) phi::BufferFree(motion_storage_);
    if (clocks_) phi::BufferFree(clocks_);
}

void StateSensorBank::Initialize(phi::Backend* backend, uint32_t env_count, double outer_dt,
    double interval, uint32_t links, uint32_t bodies, uint32_t articulations) {
    backend_ = backend;
    env_count_ = env_count;
    outer_dt_ = outer_dt;
    interval_ = interval;
    reset_steps_.assign(env_count, 0u);
    motion_before_.env_count = env_count;
    motion_before_.links_per_env = links;
    motion_before_.bodies_per_env = bodies;
    motion_before_.articulations_per_env = articulations;
    motion_after_ = motion_before_;
    advance_.env_count = env_count;
    advance_.interval = interval;
}

phi::Status StateSensorBank::UploadTimes() {
    std::vector<double> times(env_count_);
    for (uint32_t env = 0u; env < env_count_; ++env)
        times[env] = static_cast<double>(step_ - reset_steps_[env]) * outer_dt_;
    const auto status = phi::BufferUpload(clocks_, times.data(), 0u, times.size() * sizeof(double));
    return status == phi::Status::Ok ? phi::BackendSynchronize(backend_) : status;
}

phi::Status StateSensorBank::AllocateMotion() {
    if (motion_storage_) return channels_.empty() ? UploadTimes() : phi::Status::Ok;
    const uint64_t count = uint64_t{env_count_} *
        (uint64_t{motion_before_.links_per_env} + motion_before_.bodies_per_env);
    if (!count || count > UINT32_MAX) return phi::Status::InvalidArgument;
    auto* type = phi::BackendDeviceBufferType(backend_);
    phi::Status status;
    motion_storage_ = phi::BufferAlloc(type, size_t{count} * 2u * sizeof(MotionFrame), &status);
    if (!motion_storage_) return status;
    clocks_ = phi::BufferAlloc(type, size_t{env_count_} * sizeof(double), &status);
    if (!clocks_) {
        phi::BufferFree(motion_storage_);
        motion_storage_ = nullptr;
        return status;
    }
    motion_before_.frames = static_cast<MotionFrame*>(phi::BufferBase(motion_storage_));
    motion_after_.frames = motion_before_.frames + count;
    advance_.times = static_cast<double*>(phi::BufferBase(clocks_));
    status = UploadTimes();
    if (status != phi::Status::Ok) {
        phi::BufferFree(motion_storage_);
        phi::BufferFree(clocks_);
        motion_storage_ = clocks_ = nullptr;
        motion_before_.frames = motion_after_.frames = nullptr;
        advance_.times = nullptr;
    }
    return status;
}

phi::Status StateSensorBank::Add(const nk::Model& model, const StateSensorDesc& requested, uint32_t* id) {
    if (!id || !backend_ || !env_count_ || !(interval_ > 0.0) || !std::isfinite(interval_) ||
        !ValidStateSensorDesc(requested) || channels_.size() >= UINT32_MAX ||
        uint64_t{env_count_} * kStateSensorMaxValues > UINT32_MAX)
        return phi::Status::InvalidArgument;
    StateSensorDesc desc = requested;
    if (desc.sample_period == 0.0) desc.sample_period = outer_dt_ * desc.update_period;
    if (desc.sample_period < interval_ * (1.0 - 1.0e-6)) return phi::Status::InvalidArgument;
    desc.sample_period = std::max(desc.sample_period, interval_);
    if (desc.mount == StateSensorMount::Base) {
        if (desc.index >= model.articulation.articulation_link_offset.size()) return phi::Status::InvalidArgument;
        desc.index = model.articulation.articulation_link_offset[desc.index];
        desc.mount = StateSensorMount::Link;
    }
    if (desc.mount == StateSensorMount::Link) {
        if (desc.index >= model.capacities.links_per_env) return phi::Status::InvalidArgument;
    } else if (desc.mount == StateSensorMount::Body) {
        if (desc.index >= model.capacities.bodies_per_env) return phi::Status::InvalidArgument;
        if (desc.index < model.body_to_link.size() && model.body_to_link[desc.index] != ~0u) {
            desc.index = model.body_to_link[desc.index];
            desc.mount = StateSensorMount::Link;
        } else if (desc.index < model.body_collidable_body.size() &&
                   model.body_collidable_body[desc.index] != ~0u &&
                   model.body_collidable_body[desc.index] != desc.index) {
            return phi::Status::InvalidArgument;
        }
    } else return phi::Status::InvalidArgument;
    if (desc.kind == StateSensorKind::JointState) {
        if (desc.mount != StateSensorMount::Link || desc.index >= model.articulation.joint_type.size())
            return phi::Status::InvalidArgument;
        const auto type = static_cast<phi::ArticulationJointType>(model.articulation.joint_type[desc.index]);
        if (type != phi::ArticulationJointType::Revolute && type != phi::ArticulationJointType::Prismatic)
            return phi::Status::InvalidArgument;
    }
    const auto rotation = desc.local_offset.rotation;
    const double norm = std::sqrt(double{rotation.w} * rotation.w + double{rotation.x} * rotation.x +
        double{rotation.y} * rotation.y + double{rotation.z} * rotation.z);
    desc.local_offset.rotation = {static_cast<float>(rotation.w / norm), static_cast<float>(rotation.x / norm),
        static_cast<float>(rotation.y / norm), static_cast<float>(rotation.z / norm)};
    const double queue_count = std::ceil((desc.latency + desc.latency_jitter + interval_) /
                                         desc.sample_period) + 2.0;
    if (!std::isfinite(queue_count) || queue_count > static_cast<double>(UINT32_MAX / env_count_))
        return phi::Status::InvalidArgument;
    channels_.reserve(channels_.size() + 1u);
    before_.reserve(1u);
    after_.reserve(channels_.size() + 3u);
    auto status = AllocateMotion();
    if (status != phi::Status::Ok) return status;
    auto channel = std::make_unique<Channel>();
    auto& params = channel->params;
    params.desc = desc;
    params.env_count = env_count_;
    params.links_per_env = motion_before_.links_per_env;
    params.frames_per_env = motion_before_.links_per_env + motion_before_.bodies_per_env;
    params.channel = static_cast<uint32_t>(channels_.size());
    params.value_count = StateSensorValueCount(desc.kind);
    params.queue_capacity = static_cast<uint32_t>(queue_count);
    params.before = motion_before_.frames;
    params.after = motion_after_.frames;
    params.times = advance_.times;
    params.interval = interval_;
    params.gravity = gravity_;
    const auto align = [](size_t bytes) { return (bytes + alignof(StateSensorRuntime) - 1u) /
        alignof(StateSensorRuntime) * alignof(StateSensorRuntime); };
    channel->runtime_offset = align(size_t{env_count_} * params.value_count * sizeof(float));
    channel->noise_offset = channel->runtime_offset + size_t{env_count_} * sizeof(StateSensorRuntime);
    channel->queue_offset = align(channel->noise_offset + size_t{env_count_} *
        kStateSensorErrorChannels * sizeof(ObservationNoiseState));
    channel->bytes = channel->queue_offset + size_t{env_count_} * params.queue_capacity * sizeof(StateSensorPacket);
    channel->storage = phi::BufferAlloc(phi::BackendDeviceBufferType(backend_), channel->bytes, &status);
    if (!channel->storage) return status;
    status = phi::BufferMemset(channel->storage, 0u, 0u, channel->bytes);
    if (status != phi::Status::Ok) return status;
    auto* base = static_cast<uint8_t*>(phi::BufferBase(channel->storage));
    params.values = reinterpret_cast<float*>(base);
    params.runtime = reinterpret_cast<StateSensorRuntime*>(base + channel->runtime_offset);
    params.noise = reinterpret_cast<ObservationNoiseState*>(base + channel->noise_offset);
    params.queue = reinterpret_cast<StateSensorPacket*>(base + channel->queue_offset);
    channels_.push_back(std::move(channel));
    BuildCalls();
    *id = params.channel;
    return phi::Status::Ok;
}

void StateSensorBank::RemoveLast(uint32_t id) {
    if (channels_.empty() || id != channels_.size() - 1u) return;
    channels_.pop_back();
    BuildCalls();
}

bool StateSensorBank::HasActive() const {
    return std::any_of(channels_.begin(), channels_.end(),
        [](const auto& channel) { return channel->active; });
}

void StateSensorBank::BuildCalls() {
    before_.clear();
    after_.clear();
    if (HasActive()) {
        before_.push_back({phi::NkOp::ReadoutMotion, &motion_before_});
        after_.push_back({phi::NkOp::ReadoutMotion, &motion_after_});
        for (const auto& channel : channels_)
            if (channel->active) after_.push_back({phi::NkOp::SampleStateSensor, &channel->params});
    }
    if (clocks_) after_.push_back({phi::NkOp::AdvanceSensorTime, &advance_});
}

phi::Status StateSensorBank::ConfigureError(uint32_t id, uint32_t component, const ObservationConfig& config) {
    if (id >= Count() || component >= kStateSensorErrorChannels || !ValidObservationConfig(config) ||
        (component >= ValueCount(id) && channels_[id]->params.desc.kind != StateSensorKind::FramePose))
        return phi::Status::InvalidArgument;
    auto& channel = *channels_[id];
    if (channel.params.desc.kind == StateSensorKind::FramePose && component >= 3u &&
        config.error.scale_error != 0.0f) return phi::Status::InvalidArgument;
    const auto status = phi::BufferMemset(channel.storage, 0u, 0u, channel.bytes);
    if (status != phi::Status::Ok) return status;
    channel.params.desc.errors[component] = config;
    channel.active = true;
    BuildCalls();
    return phi::Status::Ok;
}

void StateSensorBank::SetGravity(math::Vec3 gravity) {
    gravity_ = gravity;
    for (auto& channel : channels_) channel->params.gravity = gravity;
}

phi::Status StateSensorBank::Reset(const std::vector<uint32_t>& env_ids) {
    std::vector<uint32_t> selected = env_ids;
    if (selected.empty()) for (uint32_t env = 0u; env < env_count_; ++env) selected.push_back(env);
    for (uint32_t env : selected) if (env >= env_count_) return phi::Status::InvalidArgument;
    const double zero = 0.0;
    for (uint32_t env : selected) {
        reset_steps_[env] = step_;
        if (clocks_) {
            const auto status = phi::BufferUpload(clocks_, &zero, size_t{env} * sizeof(double), sizeof(double));
            if (status != phi::Status::Ok) return status;
        }
        for (const auto& channel : channels_) {
            const auto& p = channel->params;
            const size_t strides[] = {size_t{p.value_count} * sizeof(float), sizeof(StateSensorRuntime),
                kStateSensorErrorChannels * sizeof(ObservationNoiseState), size_t{p.queue_capacity} * sizeof(StateSensorPacket)};
            const size_t offsets[] = {0u, channel->runtime_offset, channel->noise_offset, channel->queue_offset};
            for (uint32_t segment = 0u; segment < 4u; ++segment) {
                const auto status = phi::BufferMemset(channel->storage, 0u,
                    offsets[segment] + size_t{env} * strides[segment], strides[segment]);
                if (status != phi::Status::Ok) return status;
            }
        }
    }
    return clocks_ ? phi::BackendSynchronize(backend_) : phi::Status::Ok;
}

uint32_t StateSensorBank::Count() const { return static_cast<uint32_t>(channels_.size()); }
bool StateSensorBank::Active(uint32_t id) const { return id < Count() && channels_[id]->active; }
uint32_t StateSensorBank::ValueCount(uint32_t id) const { return id < Count() ? channels_[id]->params.value_count : 0u; }
float* StateSensorBank::Values(uint32_t id) const { return Active(id) ? channels_[id]->params.values : nullptr; }
const StateSensorDesc* StateSensorBank::Descriptor(uint32_t id) const { return id < Count() ? &channels_[id]->params.desc : nullptr; }

size_t StateSensorBank::StorageBytes() const {
    size_t bytes = clocks_ ? size_t{env_count_} * sizeof(double) + size_t{env_count_} *
        (motion_before_.links_per_env + motion_before_.bodies_per_env) * 2u * sizeof(MotionFrame) : 0u;
    for (const auto& channel : channels_) bytes += channel->bytes;
    return bytes;
}

phi::Status StateSensorBank::Download(uint32_t id, void* destination, size_t bytes, size_t offset) const {
    if (!Active(id)) return phi::Status::InvalidArgument;
    const size_t available = size_t{env_count_} * ValueCount(id) * sizeof(float);
    if ((bytes && !destination) || offset > available || bytes > available - offset)
        return phi::Status::InvalidArgument;
    const auto status = phi::BufferDownload(channels_[id]->storage, destination, offset, bytes);
    return status == phi::Status::Ok ? phi::BackendSynchronize(backend_) : status;
}

phi::Status StateSensorBank::ReadStamp(uint32_t id, uint32_t env, StateSensorStamp* out) const {
    if (!Active(id) || env >= env_count_ || !out) return phi::Status::InvalidArgument;
    const auto& channel = *channels_[id];
    const auto status = phi::BufferDownload(channel.storage, out,
        channel.runtime_offset + size_t{env} * sizeof(StateSensorRuntime), sizeof(*out));
    return status == phi::Status::Ok ? phi::BackendSynchronize(backend_) : status;
}

phi::Status StateSensorBank::Capture(StateSensorBankSnapshot* snapshot) const {
    if (!snapshot) return phi::Status::InvalidArgument;
    snapshot->step = step_;
    snapshot->reset_steps = reset_steps_;
    snapshot->times.resize(clocks_ ? env_count_ : 0u);
    if (clocks_) {
        const auto status = phi::BufferDownload(clocks_, snapshot->times.data(), 0u, size_t{env_count_} * sizeof(double));
        if (status != phi::Status::Ok) return status;
    }
    snapshot->sensors.resize(channels_.size());
    for (uint32_t id = 0u; id < Count(); ++id) {
        auto& saved = snapshot->sensors[id];
        saved.desc = channels_[id]->params.desc;
        saved.active = channels_[id]->active;
        saved.bytes.resize(channels_[id]->bytes);
        const auto status = phi::BufferDownload(channels_[id]->storage, saved.bytes.data(), 0u, saved.bytes.size());
        if (status != phi::Status::Ok) return status;
    }
    return phi::BackendSynchronize(backend_);
}

bool StateSensorBank::Compatible(const StateSensorBankSnapshot& snapshot) const {
    if (snapshot.reset_steps.size() != env_count_ || snapshot.sensors.size() > channels_.size() ||
        (!snapshot.times.empty() && snapshot.times.size() != env_count_)) return false;
    for (uint64_t reset : snapshot.reset_steps) if (reset > snapshot.step) return false;
    for (uint32_t id = 0u; id < snapshot.sensors.size(); ++id) {
        const auto& saved = snapshot.sensors[id];
        const auto& current = channels_[id]->params.desc;
        if (!ValidStateSensorDesc(saved.desc) || saved.bytes.size() != channels_[id]->bytes ||
            saved.desc.kind != current.kind || saved.desc.mount != current.mount || saved.desc.index != current.index ||
            saved.desc.sample_period != current.sample_period || saved.desc.latency != current.latency ||
            saved.desc.latency_jitter != current.latency_jitter) return false;
    }
    return true;
}

phi::Status StateSensorBank::Restore(const StateSensorBankSnapshot& snapshot) {
    if (!Compatible(snapshot)) return phi::Status::InvalidArgument;
    step_ = snapshot.step;
    reset_steps_ = snapshot.reset_steps;
    if (clocks_) {
        const auto status = snapshot.times.empty() ? UploadTimes() :
            phi::BufferUpload(clocks_, snapshot.times.data(), 0u, size_t{env_count_} * sizeof(double));
        if (status != phi::Status::Ok) return status;
    }
    for (uint32_t id = 0u; id < Count(); ++id) {
        auto& channel = *channels_[id];
        phi::Status status;
        if (id < snapshot.sensors.size()) {
            const auto& saved = snapshot.sensors[id];
            status = phi::BufferUpload(channel.storage, saved.bytes.data(), 0u, saved.bytes.size());
            channel.params.desc = saved.desc;
            channel.active = saved.active != 0u;
        } else {
            status = phi::BufferMemset(channel.storage, 0u, 0u, channel.bytes);
            channel.active = false;
        }
        if (status != phi::Status::Ok) return status;
    }
    BuildCalls();
    return phi::BackendSynchronize(backend_);
}

}  // namespace nuka::sensor
