#include "robot_cloth_fluid_scene.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "phi/backend_cuda/cuda_internal.cuh"
#include "scene/format/json.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace fixture = nuka::perf::fixture;
using Json = nuka::scene::json::Value;
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Options {
    std::string scene = "robot-cloth-fluid", execution = "eager", output = "-", state_output, wrench_output;
    uint32_t envs = 1u, steps = 200u, warmup = 250u, seed = 20260908u;
    float dt = 1.0f / 240.0f;
    uint32_t capacity_scale = 1u;
};

uint32_t ParseU32(const std::string& value) {
    size_t consumed = 0u;
    if (value.empty() || value.front() == '-') throw std::invalid_argument("invalid unsigned integer");
    const auto number = std::stoull(value, &consumed);
    if (consumed != value.size() || number > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("unsigned integer out of range");
    return static_cast<uint32_t>(number);
}

Options Parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (++i == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value(argv[i]);
        if (flag == "--scene") options.scene = value;
        else if (flag == "--envs") options.envs = ParseU32(value);
        else if (flag == "--steps") options.steps = ParseU32(value);
        else if (flag == "--warmup") options.warmup = ParseU32(value);
        else if (flag == "--seed") options.seed = ParseU32(value);
        else if (flag == "--dt") {
            size_t consumed = 0u;
            options.dt = std::stof(value, &consumed);
            if (consumed != value.size()) throw std::invalid_argument("invalid timestep");
        }
        else if (flag == "--execution") options.execution = value;
        else if (flag == "--perf-json") options.output = value;
        else if (flag == "--state-output") options.state_output = value;
        else if (flag == "--wrench-output") options.wrench_output = value;
        else if (flag == "--capacity-scale") options.capacity_scale = ParseU32(value);
        else throw std::invalid_argument("unknown option " + flag);
    }
    if (options.envs == 0u || options.steps == 0u || options.capacity_scale == 0u ||
        !(options.dt > 0.0f) || !std::isfinite(options.dt) ||
        uint64_t{options.steps} + options.warmup > std::numeric_limits<uint32_t>::max() ||
        (options.execution != "eager" && options.execution != "graph"))
        throw std::invalid_argument("invalid benchmark configuration");
    return options;
}

struct BackendOwner {
    phi::Backend* backend = nullptr;
    ~BackendOwner() { if (backend) phi::BackendFree(backend); }
};

struct Events {
    std::vector<cudaEvent_t> events;
    explicit Events(size_t count) : events(count, nullptr) {
        try {
            for (auto& event : events) CheckCuda(cudaEventCreate(&event));
        } catch (...) {
            for (auto event : events) if (event) cudaEventDestroy(event);
            throw;
        }
    }
    ~Events() { for (auto event : events) if (event) cudaEventDestroy(event); }
};

void Step(nk::World& world, const Options& options) {
    const auto status = options.execution == "graph" ? world.StepPlanned() : world.Step().result;
    if (status != phi::Status::Ok)
        throw std::runtime_error(options.execution + " step failed with status " +
                                 std::to_string(static_cast<unsigned>(status)));
}

constexpr std::array<nk::FieldId, 10> kPhysicalFields{
    nk::FieldId::BasePose, nk::FieldId::Q, nk::FieldId::Qdot,
    nk::FieldId::LinkVelocity, nk::FieldId::BodyPose,
    nk::FieldId::BodyLinearVelocity, nk::FieldId::BodyAngularVelocity,
    nk::FieldId::ParticlePos, nk::FieldId::ParticlePrevPos, nk::FieldId::ParticleVel};

std::vector<uint8_t> State(nk::World& world, bool cache = true) {
    std::vector<nk::FieldId> fields(kPhysicalFields.begin(), kPhysicalFields.end());
    if (cache) {
        for (const auto field : {nk::FieldId::ContactCachePair, nk::FieldId::ContactCacheFeature,
                                nk::FieldId::ContactCacheMaterial, nk::FieldId::ContactCacheAge,
                                nk::FieldId::ContactCacheNormal, nk::FieldId::ContactCacheTangent1,
                                nk::FieldId::ContactCacheTangent2, nk::FieldId::ContactCacheLambda,
                                nk::FieldId::LinkContactWrench})
            fields.push_back(field);
    }
    std::vector<uint8_t> result;
    for (const auto field : fields) {
        const auto count = world.GetModel().capacities.ElementCount(field);
        const auto bytes = count * nk::LayoutOf(field).elem_size;
        if (!bytes) continue;
        const size_t offset = result.size();
        result.resize(offset + bytes);
        fixture::Require(world.GetData().DownloadField(field, result.data() + offset, bytes),
                         "state download failed");
    }
    return result;
}

std::vector<uint32_t> MismatchedReplicas(nk::World& world, const std::vector<uint8_t>& state) {
    const auto& capacities = world.GetModel().capacities;
    std::vector<bool> mismatch(capacities.env_count, false);
    size_t offset = 0u;
    for (const auto field : kPhysicalFields) {
        const size_t bytes = capacities.ElementCount(field) * nk::LayoutOf(field).elem_size;
        fixture::Require(bytes % capacities.env_count == 0u && bytes <= state.size() - offset,
                         "invalid environment state extent");
        const size_t stride = bytes / capacities.env_count;
        if (stride != 0u) {
            for (uint32_t env = 1u; env < capacities.env_count; ++env)
                mismatch[env] = mismatch[env] ||
                    std::memcmp(state.data() + offset, state.data() + offset + env * stride, stride) != 0;
        }
        offset += bytes;
    }
    fixture::Require(offset == state.size(), "incomplete environment state comparison");
    std::vector<uint32_t> result;
    for (uint32_t env = 1u; env < capacities.env_count; ++env)
        if (mismatch[env]) result.push_back(env);
    return result;
}

uint64_t UpdateDigest(uint64_t hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0u; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
    return hash;
}

std::string FormatDigest(uint64_t hash) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::string Digest(const std::vector<uint8_t>& bytes) {
    return FormatDigest(UpdateDigest(14695981039346656037ull, bytes.data(), bytes.size()));
}

Json Distribution(std::vector<double> values) {
    Json output = Json::Object();
    std::sort(values.begin(), values.end());
    const auto quantile = [&](double q) {
        const double rank = q * static_cast<double>(values.size() - 1u);
        const auto low = static_cast<size_t>(rank);
        return values[low] + (values[std::min(low + 1u, values.size() - 1u)] - values[low]) * (rank - static_cast<double>(low));
    };
    output.Set("count", Json::Int(values.size()));
    double sum = 0.0;
    for (double value : values) sum += value;
    output.Set("mean_us", Json::Float(sum / static_cast<double>(values.size())));
    output.Set("p50_us", Json::Float(quantile(0.5)));
    output.Set("p95_us", Json::Float(quantile(0.95)));
    output.Set("p99_us", Json::Float(quantile(0.99)));
    output.Set("min_us", Json::Float(values.front()));
    output.Set("max_us", Json::Float(values.back()));
    return output;
}

Json Memory(nk::World& world) {
    Json memory = Json::Object(), fields = Json::Array();
    uint64_t model_bytes = 0u, arena_bytes[3]{};
    world.GetModel().ComputeModelSegments(&model_bytes);
    nk::Arena::ComputeSegments(world.GetModel().capacities, arena_bytes);
    memory.Set("model_bytes", Json::Int(model_bytes));
    memory.Set("persistent_bytes", Json::Int(arena_bytes[0]));
    memory.Set("workspace_bytes", Json::Int(arena_bytes[1]));
    memory.Set("tape_bytes", Json::Int(arena_bytes[2]));
    memory.Set("data_bytes", Json::Int(arena_bytes[0] + arena_bytes[1] + arena_bytes[2]));
    for (const auto& segment : world.GetData().Segments()) {
        Json item = Json::Object();
        item.Set("field_id", Json::Int(static_cast<uint32_t>(segment.field)));
        item.Set("name", Json::Str(nk::FieldName(segment.field)));
        item.Set("arena", Json::Int(segment.arena));
        item.Set("bytes", Json::Int(segment.bytes));
        item.Set("offset", Json::Int(segment.offset));
        fields.PushBack(std::move(item));
    }
    memory.Set("fields", std::move(fields));
    return memory;
}

Json Run(const Options& options) {
    auto* device = phi::InitBestDevice();
    fixture::Require(device != nullptr, "no physics device");
    BackendOwner owner{phi::DeviceInitBackend(device, nullptr)};
    fixture::Require(owner.backend != nullptr && std::strcmp(phi::BackendName(owner.backend), "cuda") == 0,
                     "CUDA completion events require a CUDA backend");
    auto* backend = reinterpret_cast<phi::CudaBackend*>(owner.backend);
    CheckCuda(cudaSetDevice(backend->device_id));
    const auto stream = phi::CudaBackendMainStream(backend);
    auto config = fixture::Cfg();
    config.dt = options.dt;
    const auto scene_path = options.scene == "robot-cloth-fluid"
        ? std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes/go2_stand.usda"
        : std::filesystem::path(options.scene);
    const auto prepare_start = Clock::now();
    const auto prepared = fixture::Prepare(scene_path, device, owner.backend, config);
    const double preparation_ms = Milliseconds(prepare_start);
    const auto cook_start = Clock::now();
    auto model = fixture::CookPrepared(prepared, options.envs);
    const auto slots = uint64_t{model.capacities.max_contacts_per_env} * options.capacity_scale;
    const auto rows = slots * nk::kPairDrivenRowsPerSlot;
    fixture::Require(rows * options.envs <= std::numeric_limits<int>::max(), "contact capacity exceeds index range");
    if (options.capacity_scale != 1u) {
        model.capacities.max_contacts_per_env = static_cast<uint32_t>(slots);
        model.capacities.max_rows_per_env = static_cast<uint32_t>(rows);
    }
    const double cook_ms = Milliseconds(cook_start);
    const auto create_start = Clock::now();
    nk::World world(std::move(model), options.envs, device, owner.backend, config);
    fixture::Require(world.Ready(), world.CreationError());
    fixture::Require(world.FieldPtr(nk::FieldId::ContactForce) != nullptr, "contact readout unavailable");
    CheckCuda(cudaStreamSynchronize(stream));
    const double creation_ms = Milliseconds(create_start);
    const auto initial = State(world, false);
    fixture::Require(MismatchedReplicas(world, initial).empty(), "benchmark inputs are not identical replicas");

    double capture_ms = 0.0;
    if (options.execution == "graph") {
        const auto capture_start = Clock::now();
        Step(world, options);
        CheckCuda(cudaStreamSynchronize(stream));
        capture_ms = Milliseconds(capture_start);
        fixture::Require(world.Reset() == phi::Status::Ok, "reset after capture failed");
    }
    const auto warmup_start = Clock::now();
    for (uint32_t i = 0; i < options.warmup; ++i) Step(world, options);
    CheckCuda(cudaStreamSynchronize(stream));
    const double warmup_ms = Milliseconds(warmup_start);

    Events events(size_t{options.steps} * 2u + 2u);
    std::vector<double> host_us(options.steps), gpu_us(options.steps);
    const auto timed_start = Clock::now();
    CheckCuda(cudaEventRecord(events.events.front(), stream));
    const auto submit_start = Clock::now();
    for (uint32_t i = 0; i < options.steps; ++i) {
        CheckCuda(cudaEventRecord(events.events[size_t{i} * 2u + 1u], stream));
        const auto host_start = Clock::now();
        Step(world, options);
        host_us[i] = Milliseconds(host_start) * 1000.0;
        CheckCuda(cudaEventRecord(events.events[size_t{i} * 2u + 2u], stream));
    }
    CheckCuda(cudaEventRecord(events.events.back(), stream));
    const double submission_ms = Milliseconds(submit_start);
    CheckCuda(cudaEventSynchronize(events.events.back()));
    const double wall_ms = Milliseconds(timed_start);
    float gpu_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&gpu_ms, events.events.front(), events.events.back()));
    for (uint32_t i = 0; i < options.steps; ++i) {
        float elapsed = 0.0f;
        CheckCuda(cudaEventElapsedTime(&elapsed, events.events[size_t{i} * 2u + 1u],
                                      events.events[size_t{i} * 2u + 2u]));
        gpu_us[i] = static_cast<double>(elapsed) * 1000.0;
    }
    const auto timed_state = State(world);

    const auto quality_start = Clock::now();
    fixture::Require(world.Reset() == phi::Status::Ok, "quality replay reset failed");
    const bool reset_equal = State(world, false) == initial;
    std::vector<uint32_t> status(options.envs);
    uint32_t status_union = 0u;
    const auto& caps = world.GetModel().capacities;
    std::vector<nk::NkRow> rows_host(size_t{caps.max_rows_per_env} * options.envs);
    const size_t wrench_bytes = caps.ElementCount(nk::FieldId::LinkContactWrench) *
                                nk::LayoutOf(nk::FieldId::LinkContactWrench).elem_size;
    std::vector<float> wrench(wrench_bytes / sizeof(float));
    std::ofstream wrench_output;
    if (!options.wrench_output.empty()) {
        wrench_output.open(options.wrench_output, std::ios::binary);
        fixture::Require(wrench_output.is_open(), "cannot open wrench output");
    }
    uint64_t wrench_hash = 14695981039346656037ull;
    bool wrench_finite = true;
    uint64_t cloth_rows = 0u, fluid_rows = 0u;
    for (uint32_t i = 0; i < options.warmup + options.steps; ++i) {
        Step(world, options);
        fixture::Require(world.GetData().DownloadField(nk::FieldId::EnvStatus, status.data(),
                         status.size() * sizeof(uint32_t)), "status download failed");
        for (auto flags : status) status_union |= flags;
        fixture::Require(world.GetData().DownloadField(nk::FieldId::LinkContactWrench,
                         wrench.data(), wrench_bytes), "link wrench download failed");
        for (float value : wrench) wrench_finite &= std::isfinite(value);
        wrench_hash = UpdateDigest(wrench_hash, wrench.data(), wrench_bytes);
        if (wrench_output.is_open()) {
            wrench_output.write(reinterpret_cast<const char*>(wrench.data()), wrench_bytes);
            fixture::Require(wrench_output.good(), "cannot write wrench output");
        }
        if (i >= options.warmup && (i % 25u == 0u || i + 1u == options.warmup + options.steps)) {
            fixture::Require(world.GetData().DownloadField(nk::FieldId::Urows, rows_host.data(),
                             rows_host.size() * sizeof(nk::NkRow)), "row download failed");
            for (const auto& row : rows_host) {
                if (!(row.flags & nk::nk_row_flags::kActive)) continue;
                const auto& particle = row.a.kind == nk::kNkSideParticle ? row.a : row.b;
                const auto& rigid = row.a.kind == nk::kNkSideParticle ? row.b : row.a;
                if (particle.kind != nk::kNkSideParticle || rigid.kind != nk::kNkSideArtic) continue;
                if (particle.index % caps.particles_per_env < world.GetModel().particles.n_soft_particles) ++cloth_rows;
                else ++fluid_rows;
            }
        }
    }
    const auto replay_state = State(world);
    bool finite = true;
    const auto physical_state = State(world, false);
    const auto mismatched_envs = MismatchedReplicas(world, physical_state);
    for (size_t offset = 0; offset + sizeof(float) <= physical_state.size(); offset += sizeof(float)) {
        float value = 0.0f;
        std::memcpy(&value, physical_state.data() + offset, sizeof(float));
        finite &= std::isfinite(value);
    }
    if (!options.state_output.empty()) {
        std::ofstream output(options.state_output, std::ios::binary);
        output.write(reinterpret_cast<const char*>(replay_state.data()), replay_state.size());
        fixture::Require(output.good(), "cannot write state output");
    }
    const double quality_ms = Milliseconds(quality_start);

    Json result = Json::Object(), execution = Json::Object(), timing = Json::Object();
    result.Set("schema_version", Json::Int(3));
    result.Set("scene", Json::Str(options.scene));
    Json configuration = Json::Object();
    configuration.Set("envs", Json::Int(options.envs));
    configuration.Set("dt", Json::Float(options.dt));
    configuration.Set("seed", Json::Int(options.seed));
    configuration.Set("seed_usage", Json::Str("deterministic held control; no randomization"));
    configuration.Set("vel_iters", Json::Int(config.vel_iters));
    configuration.Set("cloth_iters", Json::Int(fixture::kClothIters));
    configuration.Set("capacity_scale", Json::Int(options.capacity_scale));
    configuration.Set("contact_slots_per_env", Json::Int(caps.max_contacts_per_env));
    configuration.Set("rows_per_env", Json::Int(caps.max_rows_per_env));
    configuration.Set("particles_per_env", Json::Int(caps.particles_per_env));
    configuration.Set("initial_state_fnv1a64", Json::Str(Digest(initial)));
    result.Set("config", std::move(configuration));
    execution.Set("mode", Json::Str(options.execution));
    execution.Set("warmup", Json::Int(options.warmup));
    execution.Set("steps", Json::Int(options.steps));
    execution.Set("boundary", Json::Str("full physics pipeline with contact readout; held control; per-step GPU events; no timed D2H"));
    result.Set("execution", std::move(execution));
    timing.Set("scene_preparation_ms", Json::Float(preparation_ms));
    timing.Set("cook_ms", Json::Float(cook_ms));
    timing.Set("create_upload_ms", Json::Float(creation_ms));
    timing.Set("capture_first_execute_ms", Json::Float(capture_ms));
    timing.Set("warmup_ms", Json::Float(warmup_ms));
    timing.Set("host_submission_ms", Json::Float(submission_ms));
    timing.Set("gpu_completion_ms", Json::Float(gpu_ms));
    timing.Set("synchronized_wall_ms", Json::Float(wall_ms));
    timing.Set("batch_step_ms", Json::Float(static_cast<double>(gpu_ms) / options.steps));
    timing.Set("env_steps_per_s", Json::Float(static_cast<double>(options.envs) * options.steps * 1000.0 / wall_ms));
    timing.Set("amortized_env_step_us", Json::Float(gpu_ms * 1000.0 / options.steps / options.envs));
    timing.Set("rtf", Json::Float(static_cast<double>(options.dt) * options.steps * 1000.0 / gpu_ms));
    timing.Set("quality_replay_ms", Json::Float(quality_ms));
    result.Set("timing", std::move(timing));
    Json per_tag = Json::Object();
    per_tag.Set("gpu_step", Distribution(std::move(gpu_us)));
    per_tag.Set("host_step_submission", Distribution(std::move(host_us)));
    result.Set("per_tag", std::move(per_tag));
    result.Set("memory", Memory(world));
    Json quality = Json::Object();
    quality.Set("finite", Json::Bool(finite));
    quality.Set("link_wrench_finite", Json::Bool(wrench_finite));
    quality.Set("link_wrench_trace_fnv1a64", Json::Str(FormatDigest(wrench_hash)));
    quality.Set("link_wrench_bytes_per_step", Json::Int(wrench_bytes));
    quality.Set("state_layout", Json::Str("physical fields, contact cache, link wrench"));
    quality.Set("env_status_union", Json::Int(status_union));
    quality.Set("reset_state_equal", Json::Bool(reset_equal));
    quality.Set("timed_replay_bit_equal", Json::Bool(timed_state == replay_state));
    quality.Set("state_fnv1a64", Json::Str(Digest(replay_state)));
    quality.Set("authoritative_state_fnv1a64", Json::Str(Digest(
        std::vector<uint8_t>(replay_state.begin(), replay_state.begin() + initial.size()))));
    quality.Set("authoritative_state_bytes", Json::Int(initial.size()));
    quality.Set("replica_state_equal", Json::Bool(mismatched_envs.empty()));
    Json replica_errors = Json::Array();
    for (uint32_t env : mismatched_envs) replica_errors.PushBack(Json::Int(env));
    quality.Set("replica_mismatch_envs", std::move(replica_errors));
    quality.Set("cloth_rows_sampled", Json::Int(cloth_rows));
    quality.Set("fluid_rows_sampled", Json::Int(fluid_rows));
    result.Set("quality", std::move(quality));
    Json hardware = Json::Object();
    cudaDeviceProp properties{};
    CheckCuda(cudaGetDeviceProperties(&properties, backend->device_id));
    int driver = 0, runtime = 0;
    CheckCuda(cudaDriverGetVersion(&driver));
    CheckCuda(cudaRuntimeGetVersion(&runtime));
    hardware.Set("gpu", Json::Str(properties.name));
    hardware.Set("cuda_driver_api", Json::Int(driver));
    hardware.Set("cuda_runtime", Json::Int(runtime));
    hardware.Set("total_device_bytes", Json::Int(properties.totalGlobalMem));
    result.Set("hardware", std::move(hardware));
    Json validity = Json::Object(), unavailable = Json::Array();
    const bool valid = finite && wrench_finite && status_union == 0u && reset_equal && timed_state == replay_state &&
                       cloth_rows > 0u && fluid_rows > 0u && mismatched_envs.empty();
    validity.Set("valid", Json::Bool(valid));
    unavailable.PushBack(Json::Str("GPU clocks, source/binary SHA256 and process identity are collected by the sweep runner"));
    unavailable.PushBack(Json::Str("residual, complementarity and detailed penetration observability remain incomplete"));
    validity.Set("unavailable", std::move(unavailable));
    result.Set("status", std::move(validity));
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    Options options;
    Json result;
    int exit_code = 0;
    try {
        options = Parse(argc, argv);
        result = Run(options);
        if (!result.At("status").At("valid").AsBool()) exit_code = 2;
    } catch (const std::exception& error) {
        result = Json::Object();
        result.Set("schema_version", Json::Int(3));
        Json status = Json::Object();
        status.Set("valid", Json::Bool(false));
        status.Set("error", Json::Str(error.what()));
        result.Set("status", std::move(status));
        std::cerr << error.what() << '\n';
        exit_code = 1;
    }
    if (options.output == "-") std::cout << result.Dump() << '\n';
    else {
        std::ofstream output(options.output);
        output << result.Dump() << '\n';
        if (!output.good()) { std::cerr << "cannot write perf JSON\n"; return 1; }
    }
    return exit_code;
}
