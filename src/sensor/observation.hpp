#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "phi/backend.hpp"
#include "sensor/observation_types.hpp"

namespace nuka::sensor {

struct ObservationSnapshot {
    ObservationConfig config;
    uint32_t env_count = 0u;
    uint32_t values_per_env = 0u;
    std::vector<uint8_t> bytes;
};

// Observation storage never aliases its source and keeps its addresses until destruction.
class Observation {
public:
    Observation() = default;
    ~Observation();
    Observation(const Observation&) = delete;
    Observation& operator=(const Observation&) = delete;

    phi::Status Initialize(phi::Backend* backend, uint32_t env_count,
                           uint32_t values_per_env, uint32_t channel);
    phi::Status Configure(const ObservationConfig& config);
    phi::Status Deactivate();
    phi::Status Sample(const float* source, double sample_interval, float temperature);
    phi::Status Reset(const std::vector<uint32_t>& env_ids = {});
    phi::Status Download(void* destination, size_t bytes, size_t offset = 0u) const;
    phi::Status ReadStamp(uint32_t env, ObservationStamp* stamp) const;
    phi::Status Capture(ObservationSnapshot* snapshot) const;
    phi::Status Restore(const ObservationSnapshot& snapshot);
    bool Compatible(const ObservationSnapshot& snapshot) const;

    float* Values() const { return values_; }
    uint32_t EnvCount() const { return env_count_; }
    uint32_t ValuesPerEnv() const { return values_per_env_; }
    size_t ValueBytes() const { return size_t{env_count_} * values_per_env_ * sizeof(float); }
    size_t StorageBytes() const { return storage_bytes_; }
    const ObservationConfig& Config() const { return config_; }
    bool Active() const { return active_; }

private:
    phi::Backend* backend_ = nullptr;
    phi::Buffer* storage_ = nullptr;
    phi::Buffer* selection_ = nullptr;
    float* values_ = nullptr;
    ObservationNoiseState* noise_state_ = nullptr;
    ObservationStamp* stamps_ = nullptr;
    uint32_t env_count_ = 0u;
    uint32_t values_per_env_ = 0u;
    uint32_t channel_ = 0u;
    size_t stamp_offset_ = 0u;
    size_t storage_bytes_ = 0u;
    ObservationConfig config_;
    bool active_ = true;
};

}  // namespace nuka::sensor
