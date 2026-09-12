#include "sensor/observation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "nk/model/generated/views.hpp"

namespace nuka::sensor {

bool ValidObservationConfig(const ObservationConfig& config) {
    const auto& n = config.noise;
    if (!std::isfinite(n.param1) || !std::isfinite(n.param2)) return false;
    switch (n.kind) {
        case noise::NoiseKind::None: break;
        case noise::NoiseKind::Gaussian: if (n.param2 < 0.0f) return false; break;
        case noise::NoiseKind::Poisson:
            if (n.param1 < 0.0f || n.param1 > 1.0e8f) return false;
            break;
        default: return false;
    }
    const auto& e = config.error;
    const float values[] = {e.bias, e.scale_error, e.noise_density, e.initial_bias_stddev,
        e.bias_random_walk, e.correlated_bias_stddev, e.correlation_time, e.quantization,
        e.minimum, e.maximum, e.response_time, e.temperature_coefficient,
        e.reference_temperature};
    for (float value : values) if (!std::isfinite(value)) return false;
    return e.noise_density >= 0.0f && e.initial_bias_stddev >= 0.0f &&
        e.bias_random_walk >= 0.0f && e.correlated_bias_stddev >= 0.0f &&
        e.correlation_time >= 0.0f && e.quantization >= 0.0f && e.response_time >= 0.0f &&
        (e.correlated_bias_stddev == 0.0f || e.correlation_time > 0.0f) &&
        e.saturation_enabled <= 1u && (!e.saturation_enabled || e.minimum <= e.maximum);
}

Observation::~Observation() {
    if (storage_) phi::BufferFree(storage_);
    if (selection_) phi::BufferFree(selection_);
}

phi::Status Observation::Initialize(phi::Backend* backend, uint32_t env_count,
                                    uint32_t values_per_env, uint32_t channel) {
    if (storage_ || !backend || !env_count || !values_per_env ||
        uint64_t{env_count} * values_per_env > std::numeric_limits<uint32_t>::max())
        return phi::Status::InvalidArgument;
    const size_t count = size_t{env_count} * values_per_env;
    const size_t value_bytes = count * sizeof(float);
    const size_t states_end = value_bytes + count * sizeof(ObservationNoiseState);
    const size_t alignment = alignof(ObservationStamp);
    const size_t stamp_offset = (states_end + alignment - 1u) / alignment * alignment;
    const size_t storage_bytes = stamp_offset + size_t{env_count} * sizeof(ObservationStamp);
    phi::Status status = phi::Status::Ok;
    auto* type = phi::BackendDeviceBufferType(backend);
    auto* storage = phi::BufferAlloc(type, storage_bytes, &status);
    if (!storage) return status;
    auto* selection = phi::BufferAlloc(type, size_t{env_count} * sizeof(uint32_t), &status);
    if (!selection) { phi::BufferFree(storage); return status; }
    status = phi::BufferMemset(storage, 0u, 0u, storage_bytes);
    if (status != phi::Status::Ok) {
        phi::BufferFree(storage);
        phi::BufferFree(selection);
        return status;
    }
    backend_ = backend;
    storage_ = storage;
    selection_ = selection;
    env_count_ = env_count;
    values_per_env_ = values_per_env;
    channel_ = channel;
    stamp_offset_ = stamp_offset;
    storage_bytes_ = storage_bytes;
    auto* base = static_cast<uint8_t*>(phi::BufferBase(storage));
    values_ = reinterpret_cast<float*>(base);
    noise_state_ = reinterpret_cast<ObservationNoiseState*>(base + value_bytes);
    stamps_ = reinterpret_cast<ObservationStamp*>(base + stamp_offset);
    return phi::Status::Ok;
}

phi::Status Observation::Configure(const ObservationConfig& config) {
    if (!ValidObservationConfig(config)) return phi::Status::InvalidArgument;
    const auto status = Reset();
    if (status == phi::Status::Ok) { config_ = config; active_ = true; }
    return status;
}

phi::Status Observation::Deactivate() {
    const auto status = Configure({});
    if (status == phi::Status::Ok) active_ = false;
    return status;
}

phi::Status Observation::Sample(const float* source, double sample_interval, float temperature) {
    if (!storage_ || !source || !(sample_interval > 0.0) ||
        !std::isfinite(sample_interval) || sample_interval > std::numeric_limits<float>::max() ||
        !std::isfinite(temperature)) return phi::Status::InvalidArgument;
    phi::SampleObservationParams params{};
    params.source = source;
    params.values = values_;
    params.noise_state = noise_state_;
    params.stamps = stamps_;
    params.env_count = env_count_;
    params.values_per_env = values_per_env_;
    params.channel = channel_;
    params.sample_interval = sample_interval;
    params.temperature = temperature;
    params.config = config_;
    const auto status = phi::BackendDispatch(backend_, {}, {}, {phi::NkOp::SampleObservation, &params});
    if (status == phi::Status::Ok) active_ = true;
    return status;
}

phi::Status Observation::Reset(const std::vector<uint32_t>& env_ids) {
    if (!storage_) return phi::Status::InvalidArgument;
    for (uint32_t env : env_ids) if (env >= env_count_) return phi::Status::InvalidArgument;
    if (env_ids.empty()) return phi::BufferMemset(storage_, 0u, 0u, storage_bytes_);
    auto selected = env_ids;
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    auto status = phi::BufferUpload(selection_, selected.data(), 0u,
                                    selected.size() * sizeof(uint32_t));
    if (status != phi::Status::Ok) return status;
    phi::ResetObservationParams params{};
    params.values = values_;
    params.noise_state = noise_state_;
    params.stamps = stamps_;
    params.env_ids = static_cast<const uint32_t*>(phi::BufferBase(selection_));
    params.selected_count = static_cast<uint32_t>(selected.size());
    params.values_per_env = values_per_env_;
    status = phi::BackendDispatch(backend_, {}, {}, {phi::NkOp::ResetObservation, &params});
    if (status != phi::Status::Ok) return status;
    return phi::BackendSynchronize(backend_);
}

phi::Status Observation::Download(void* destination, size_t bytes, size_t offset) const {
    if (!storage_ || (bytes && !destination) || offset > ValueBytes() ||
        bytes > ValueBytes() - offset) return phi::Status::InvalidArgument;
    const auto status = phi::BufferDownload(storage_, destination, offset, bytes);
    return status == phi::Status::Ok ? phi::BackendSynchronize(backend_) : status;
}

phi::Status Observation::ReadStamp(uint32_t env, ObservationStamp* stamp) const {
    if (!storage_ || !stamp || env >= env_count_) return phi::Status::InvalidArgument;
    const auto status = phi::BufferDownload(storage_, stamp,
        stamp_offset_ + size_t{env} * sizeof(ObservationStamp), sizeof(ObservationStamp));
    return status == phi::Status::Ok ? phi::BackendSynchronize(backend_) : status;
}

phi::Status Observation::Capture(ObservationSnapshot* snapshot) const {
    if (!storage_ || !snapshot) return phi::Status::InvalidArgument;
    snapshot->config = config_;
    snapshot->env_count = env_count_;
    snapshot->values_per_env = values_per_env_;
    snapshot->bytes.resize(storage_bytes_);
    const auto status = phi::BufferDownload(storage_, snapshot->bytes.data(), 0u, storage_bytes_);
    return status == phi::Status::Ok ? phi::BackendSynchronize(backend_) : status;
}

bool Observation::Compatible(const ObservationSnapshot& snapshot) const {
    return storage_ && snapshot.env_count == env_count_ &&
        snapshot.values_per_env == values_per_env_ && snapshot.bytes.size() == storage_bytes_ &&
        ValidObservationConfig(snapshot.config);
}

phi::Status Observation::Restore(const ObservationSnapshot& snapshot) {
    if (!Compatible(snapshot)) return phi::Status::InvalidArgument;
    const auto status = phi::BufferUpload(storage_, snapshot.bytes.data(), 0u, storage_bytes_);
    if (status != phi::Status::Ok) return status;
    config_ = snapshot.config;
    active_ = true;
    return phi::BackendSynchronize(backend_);
}

}  // namespace nuka::sensor
