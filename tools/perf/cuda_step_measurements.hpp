#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "nk/pipeline/world.hpp"
#include "phi/backend_cuda/cuda_internal.cuh"
#include "scene/format/json.hpp"

namespace nuka::perf {

class CudaStepMeasurements {
public:
    CudaStepMeasurements(nk::World& world, bool enabled, uint32_t steps_per_interval,
                         uint32_t intervals)
        : world_(world), steps_per_interval_(steps_per_interval), intervals_(intervals) {
        if (!enabled) return;
        if (!steps_per_interval || !intervals)
            throw std::invalid_argument("Completion timing requires nonempty intervals");
        if (std::strcmp(phi::BackendName(world.Backend()), "cuda") != 0)
            throw std::invalid_argument("Completion timing requires the CUDA backend");
        auto* backend = reinterpret_cast<phi::CudaBackend*>(world.Backend());
        CheckCuda(cudaSetDevice(backend->device_id));
        CheckCuda(cudaGetDeviceProperties(&device_, backend->device_id));
        CheckCuda(cudaDriverGetVersion(&driver_version_));
        CheckCuda(cudaRuntimeGetVersion(&runtime_version_));
        stream_ = phi::CudaBackendMainStream(backend);
        const uint64_t samples = uint64_t{steps_per_interval} * intervals;
        if (samples > gpu_ms_.max_size()) throw std::length_error("Too many timing samples");
        gpu_ms_.reserve(static_cast<size_t>(samples));
        host_ms_.reserve(static_cast<size_t>(samples));
        interval_ms_.reserve(intervals);
        events_.resize(steps_per_interval);
        CheckCuda(cudaMemGetInfo(&free_start_, &total_bytes_));
        free_min_ = free_end_ = free_start_;
        try {
            for (auto& pair : events_)
                for (auto& event : pair) CheckCuda(cudaEventCreate(&event));
        } catch (...) {
            DestroyEvents();
            throw;
        }
    }

    ~CudaStepMeasurements() { DestroyEvents(); }
    CudaStepMeasurements(const CudaStepMeasurements&) = delete;
    CudaStepMeasurements& operator=(const CudaStepMeasurements&) = delete;

    void BeginInterval() {
        if (!Enabled()) return;
        if (active_ || interval_ms_.size() == intervals_)
            throw std::logic_error("Unexpected timing interval");
        active_ = true;
        pending_ = 0u;
        interval_start_ = Clock::now();
    }

    void Step() {
        if (!Enabled()) {
            CheckStep(world_.StepConfigured());
            return;
        }
        if (!active_ || pending_ == events_.size())
            throw std::logic_error("Step exceeds timing interval capacity");
        CheckCuda(cudaEventRecord(events_[pending_][0], stream_));
        const auto start = Clock::now();
        const auto status = world_.StepConfigured();
        const double host_ms = Milliseconds(start);
        CheckCuda(cudaEventRecord(events_[pending_][1], stream_));
        CheckStep(status);
        host_ms_.push_back(host_ms);
        ++pending_;
    }

    // Collect after the caller's existing completion boundary; never synchronize each step.
    void EndInterval() {
        if (!Enabled()) return;
        const double wall_ms = Milliseconds(interval_start_);
        if (!active_ || pending_ != events_.size())
            throw std::logic_error("Incomplete timing interval");
        CheckCuda(cudaEventQuery(events_[pending_ - 1u][1]));
        interval_ms_.push_back(wall_ms);
        for (size_t i = 0u; i < pending_; ++i) {
            float elapsed = 0.0f;
            CheckCuda(cudaEventElapsedTime(&elapsed, events_[i][0], events_[i][1]));
            gpu_ms_.push_back(elapsed);
        }
        size_t total = 0u;
        CheckCuda(cudaMemGetInfo(&free_end_, &total));
        free_min_ = std::min(free_min_, free_end_);
        active_ = false;
    }

    void Write(const std::filesystem::path& path, double dt) const {
        if (!Enabled()) return;
        if (active_ || interval_ms_.size() != intervals_ || gpu_ms_.size() != host_ms_.size())
            throw std::logic_error("Incomplete completion measurements");
        Json report = Json::Object();
        report.Set("schema_version", Json::Int(1));
        report.Set("gpu_boundary", Json::Str("CUDA events around each World::StepConfigured on its main stream; not kernel busy time"));
        report.Set("wall_boundary", Json::Str("Controlled interval with existing uploads, reaction/status readout and completion; particle analysis, capture and rendering excluded"));
        Json config = Json::Object();
        const auto& model = world_.GetModel();
        config.Set("execution", Json::Str(world_.GetExecutionMode() == nk::World::ExecutionMode::Graph ? "graph" : "eager"));
        config.Set("dt", Json::Float(static_cast<float>(dt)));
        config.Set("envs", Json::Int(model.capacities.env_count));
        config.Set("particles_per_env", Json::Int(model.capacities.particles_per_env));
        config.Set("steps_per_interval", Json::Int(steps_per_interval_));
        config.Set("intervals", Json::Int(intervals_));
        config.Set("steps", Json::Int(gpu_ms_.size()));
        report.Set("config", std::move(config));
        Json hardware = Json::Object();
        hardware.Set("gpu", Json::Str(device_.name));
        hardware.Set("cuda_driver_version", Json::Int(driver_version_));
        hardware.Set("cuda_runtime_version", Json::Int(runtime_version_));
        report.Set("hardware", std::move(hardware));
        for (unsigned group = 0u; group < 2u; ++group) {
            const size_t begin = group == 0u ? 0u : steps_per_interval_;
            const size_t end = group == 0u ? steps_per_interval_ : gpu_ms_.size();
            Json timing = Json::Object();
            timing.Set("gpu_step_completion", Distribution(gpu_ms_, begin, end));
            timing.Set("host_step_call", Distribution(host_ms_, begin, end));
            timing.Set("controlled_interval_wall", Distribution(interval_ms_, group == 0u ? 0u : 1u,
                group == 0u ? 1u : interval_ms_.size()));
            report.Set(group == 0u ? "first_interval" : "steady", std::move(timing));
        }
        uint64_t model_bytes = 0u, arena_bytes[3]{};
        model.ComputeModelSegments(&model_bytes);
        nk::Arena::ComputeSegments(model.capacities, arena_bytes);
        Json memory = Json::Object();
        memory.Set("model_bytes", Json::Int(model_bytes));
        memory.Set("persistent_bytes", Json::Int(arena_bytes[0]));
        memory.Set("workspace_bytes", Json::Int(arena_bytes[1]));
        memory.Set("tape_bytes", Json::Int(arena_bytes[2]));
        memory.Set("data_bytes", Json::Int(arena_bytes[0] + arena_bytes[1] + arena_bytes[2]));
        memory.Set("timing_event_count", Json::Int(events_.size() * 2u));
        memory.Set("timing_event_device_bytes", Json::Null());
        memory.Set("timing_event_device_bytes_unavailable", Json::Str("CUDA does not expose event allocation sizes"));
        memory.Set("timing_sample_host_bytes", Json::Int(sizeof(double) *
            (gpu_ms_.capacity() + host_ms_.capacity() + interval_ms_.capacity())));
        memory.Set("device_total_bytes", Json::Int(total_bytes_));
        memory.Set("device_free_bytes_start", Json::Int(free_start_));
        memory.Set("device_free_bytes_min_observed", Json::Int(free_min_));
        memory.Set("device_free_bytes_end", Json::Int(free_end_));
        memory.Set("device_memory_scope", Json::Str("Whole-device observations at creation and interval completion; includes other processes"));
        report.Set("memory", std::move(memory));
        Json intervals = Json::Array();
        for (size_t i = 0u; i < interval_ms_.size(); ++i) {
            const size_t begin = i * steps_per_interval_, end = begin + steps_per_interval_;
            Json interval = Json::Object();
            interval.Set("index", Json::Int(i));
            interval.Set("gpu_step_sum_ms", Json::Float(Sum(gpu_ms_, begin, end)));
            interval.Set("host_step_sum_ms", Json::Float(Sum(host_ms_, begin, end)));
            interval.Set("controlled_wall_ms", Json::Float(interval_ms_[i]));
            intervals.PushBack(std::move(interval));
        }
        report.Set("intervals", std::move(intervals));
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path);
        output << report.Dump() << '\n';
        if (!output) throw std::runtime_error("Could not write " + path.string());
    }

private:
    using Clock = std::chrono::steady_clock;
    using Json = scene::json::Value;

    bool Enabled() const { return !events_.empty(); }
    static double Milliseconds(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    static void CheckCuda(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void CheckStep(phi::Status status) const {
        if (status != phi::Status::Ok)
            throw std::runtime_error(std::string("Simulation step failed: ") + world_.LastExecutionError().message);
    }
    void DestroyEvents() {
        for (const auto& pair : events_)
            for (auto event : pair) if (event) cudaEventDestroy(event);
    }
    static double Sum(const std::vector<double>& values, size_t begin, size_t end) {
        return std::accumulate(values.begin() + begin, values.begin() + end, 0.0);
    }
    static Json Distribution(const std::vector<double>& values, size_t begin, size_t end) {
        if (begin == end) return Json::Null();
        std::vector<double> sorted(values.begin() + begin, values.begin() + end);
        std::sort(sorted.begin(), sorted.end());
        const auto quantile = [&](double q) { return sorted[size_t(q * (sorted.size() - 1u))]; };
        Json result = Json::Object();
        result.Set("count", Json::Int(sorted.size()));
        result.Set("sum_ms", Json::Float(Sum(values, begin, end)));
        result.Set("mean_ms", Json::Float(Sum(values, begin, end) / sorted.size()));
        result.Set("p50_ms", Json::Float(quantile(0.50)));
        result.Set("p95_ms", Json::Float(quantile(0.95)));
        result.Set("p99_ms", Json::Float(quantile(0.99)));
        result.Set("min_ms", Json::Float(sorted.front()));
        result.Set("max_ms", Json::Float(sorted.back()));
        return result;
    }

    nk::World& world_;
    uint32_t steps_per_interval_, intervals_;
    cudaStream_t stream_ = nullptr;
    cudaDeviceProp device_{};
    int driver_version_ = 0, runtime_version_ = 0;
    std::vector<std::array<cudaEvent_t, 2>> events_;
    std::vector<double> gpu_ms_, host_ms_, interval_ms_;
    Clock::time_point interval_start_{};
    size_t pending_ = 0u, free_start_ = 0u, free_end_ = 0u, free_min_ = 0u, total_bytes_ = 0u;
    bool active_ = false;
};

}  // namespace nuka::perf
