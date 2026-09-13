#pragma once

#include <memory>
#include <string>
#include <vector>

#include "phi/backend.hpp"
#include "sensor/state_types.hpp"

namespace nuka::nk { class Model; }

namespace nuka::sensor {

struct StateSensorSnapshot {
    StateSensorDesc desc;
    uint32_t active = 0u;
    std::vector<uint8_t> bytes;
};

struct StateSensorBankSnapshot {
    uint64_t step = 0u;
    std::vector<uint64_t> reset_steps;
    std::vector<double> times;
    std::vector<StateSensorSnapshot> sensors;
};

class StateSensorBank {
public:
    StateSensorBank();
    ~StateSensorBank();
    StateSensorBank(const StateSensorBank&) = delete;
    StateSensorBank& operator=(const StateSensorBank&) = delete;

    void Initialize(phi::Backend* backend, uint32_t env_count, double outer_dt,
                    double interval, uint32_t links, uint32_t bodies, uint32_t articulations);
    phi::Status Add(const nk::Model& model, const StateSensorDesc& desc, uint32_t* id);
    void RemoveLast(uint32_t id);
    phi::Status ConfigureError(uint32_t id, uint32_t channel, const ObservationConfig& config);
    void SetGravity(math::Vec3 gravity);
    void StepCompleted() { ++step_; }
    phi::Status Reset(const std::vector<uint32_t>& env_ids);
    phi::Status Capture(StateSensorBankSnapshot* snapshot) const;
    bool Compatible(const StateSensorBankSnapshot& snapshot) const;
    phi::Status Restore(const StateSensorBankSnapshot& snapshot);

    uint32_t Count() const;
    bool HasActive() const;
    uint32_t ValueCount(uint32_t id) const;
    bool Active(uint32_t id) const;
    float* Values(uint32_t id) const;
    const StateSensorDesc* Descriptor(uint32_t id) const;
    phi::Status Download(uint32_t id, void* destination, size_t bytes, size_t offset = 0u) const;
    phi::Status ReadStamp(uint32_t id, uint32_t env, StateSensorStamp* out) const;
    size_t StorageBytes() const;
    const std::vector<phi::OpCall>& Before() const { return before_; }
    const std::vector<phi::OpCall>& After() const { return after_; }

private:
    struct Channel;
    phi::Status AllocateMotion();
    void BuildCalls();
    phi::Status UploadTimes();
    phi::Backend* backend_ = nullptr;
    phi::Buffer* motion_storage_ = nullptr;
    phi::Buffer* clocks_ = nullptr;
    uint32_t env_count_ = 0u;
    double outer_dt_ = 0.0;
    double interval_ = 0.0;
    uint64_t step_ = 0u;
    std::vector<uint64_t> reset_steps_;
    std::vector<std::unique_ptr<Channel>> channels_;
    phi::ReadoutMotionParams motion_before_{};
    phi::ReadoutMotionParams motion_after_{};
    phi::AdvanceSensorTimeParams advance_{};
    std::vector<phi::OpCall> before_;
    std::vector<phi::OpCall> after_;
    math::Vec3 gravity_;
};

}  // namespace nuka::sensor
