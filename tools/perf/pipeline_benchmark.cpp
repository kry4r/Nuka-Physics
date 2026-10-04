
#include "robot_cloth_fluid_scene.hpp"
#include "measurement_clock.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include "phi/backend_cuda/cuda_internal.cuh"
#include "core/checked_size.hpp"
#include "phi/articulation_contract.hpp"
#include "nk/solve/block_row_schedule.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "nk/solve/vertex_block.hpp"
#include "render/render_world.hpp"
#include "render/sensor_backend.hpp"
#include "render/rt_adapter.hpp"
#include "scene/format/json.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace fixture = nuka::perf::fixture;
using Json = nuka::scene::json::Value;
using Clock = nuka::perf::MeasurementClock;

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
    uint32_t substeps = 1u;
    uint32_t mpm_implicit_stress = 0u;
    uint32_t velocity_iterations = 48u;
    bool velocity_iterations_explicit = false;
    uint32_t state_sensors = 0u;
    uint32_t tactile_grid = 0u;
    uint32_t cloth_nx = fixture::kClothNx;
    uint32_t render_sensors = 0u, render_width = 256u, render_height = 256u;
    uint32_t render_samples = 4u, render_shadows = 4u, render_ao = 3u, render_warmup = 32u;
    uint32_t imaging_models = 0u;
    std::string render_output;
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
        else if (flag == "--substeps") options.substeps = ParseU32(value);
        else if (flag == "--mpm-implicit-stress") options.mpm_implicit_stress = ParseU32(value);
        else if (flag == "--velocity-iterations") {
            options.velocity_iterations = ParseU32(value);
            options.velocity_iterations_explicit = true;
        }
        else if (flag == "--state-sensors") options.state_sensors = ParseU32(value);
        else if (flag == "--tactile-grid") options.tactile_grid = ParseU32(value);
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
        else if (flag == "--cloth-grid") options.cloth_nx = ParseU32(value);
        else if (flag == "--render-sensors") options.render_sensors = ParseU32(value);
        else if (flag == "--render-width") options.render_width = ParseU32(value);
        else if (flag == "--render-height") options.render_height = ParseU32(value);
        else if (flag == "--render-samples") options.render_samples = ParseU32(value);
        else if (flag == "--render-shadows") options.render_shadows = ParseU32(value);
        else if (flag == "--render-ao") options.render_ao = ParseU32(value);
        else if (flag == "--render-warmup") options.render_warmup = ParseU32(value);
        else if (flag == "--imaging-models") options.imaging_models = ParseU32(value);
        else if (flag == "--render-output") options.render_output = value;
        else throw std::invalid_argument("unknown option " + flag);
    }
    if (options.envs == 0u || options.steps == 0u || options.capacity_scale == 0u || options.substeps == 0u ||
        options.velocity_iterations == 0u || options.velocity_iterations > UINT16_MAX ||
        (options.scene == "robot-rigid-mpm-cloth" && options.cloth_nx != fixture::kClothNx) ||
        options.state_sensors > 2u || options.imaging_models > 1u || options.mpm_implicit_stress > 1u ||
        uint64_t{options.tactile_grid} * options.tactile_grid + 1u > UINT32_MAX ||
        !(options.dt > 0.0f) || !std::isfinite(options.dt) ||
        uint64_t{options.steps} + options.warmup > std::numeric_limits<uint32_t>::max() ||
        (options.execution != "eager" && options.execution != "graph"))
        throw std::invalid_argument("invalid benchmark configuration");
    if (options.render_sensors != 0u && (options.render_width == 0u || options.render_height == 0u ||
        options.render_width > 65535u || options.render_height > 65535u || options.render_samples == 0u ||
        uint64_t{options.envs} * options.render_sensors >
            std::numeric_limits<uint32_t>::max() / options.render_width / options.render_height))
        throw std::invalid_argument("invalid render configuration");
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
    if (status != phi::Status::Ok) {
        const auto& error = world.LastExecutionError();
        throw std::runtime_error(options.execution + " step failed with status " +
                                 std::to_string(static_cast<unsigned>(status)) +
                                 ", op " + std::to_string(static_cast<unsigned>(error.failed_op)) +
                                 ", native " + std::to_string(error.native_code) + ": " + error.message);
    }
}

void AttachStateSensors(nk::World& world, const Options& options) {
    if (!options.state_sensors) return;
    const auto& topology = world.GetModel().articulation;
    for (size_t articulation = 0u; articulation < topology.articulation_link_offset.size(); ++articulation) {
        nuka::sensor::StateSensorDesc desc;
        desc.mount = nuka::sensor::StateSensorMount::Link;
        desc.index = topology.articulation_link_offset[articulation];
        desc.latency = double{options.dt} * 2.0;
        desc.latency_jitter = double{options.dt} * 0.5;
        desc.dropout_probability = 0.1f;
        desc.seed = options.seed;
        for (auto& error : desc.errors) {
            error.error.noise_density = 0.01f;
            error.error.initial_bias_stddev = 0.02f;
            error.error.bias_random_walk = 0.002f;
            error.error.correlated_bias_stddev = 0.005f;
            error.error.correlation_time = 0.2f;
        }
        uint32_t id;
        for (auto kind : {nuka::sensor::StateSensorKind::Imu, nuka::sensor::StateSensorKind::FramePose,
                          nuka::sensor::StateSensorKind::LinearVelocity}) {
            desc.kind = kind;
            fixture::Require(world.AttachStateSensor(desc, &id) == phi::Status::Ok, "state sensor attachment failed");
        }
        const auto end = desc.index + topology.articulation_link_count[articulation];
        for (uint32_t link = desc.index; link < end; ++link) {
            const auto kind = static_cast<phi::ArticulationJointType>(topology.joint_type[link]);
            if (kind != phi::ArticulationJointType::Revolute && kind != phi::ArticulationJointType::Prismatic) continue;
            desc.kind = nuka::sensor::StateSensorKind::JointState;
            desc.index = link;
            fixture::Require(world.AttachStateSensor(desc, &id) == phi::Status::Ok, "joint sensor attachment failed");
            if (options.state_sensors >= 2u) {
                desc.kind = nuka::sensor::StateSensorKind::ForceTorque;
                fixture::Require(world.AttachStateSensor(desc, &id) == phi::Status::Ok, "load sensor attachment failed");
                desc.kind = nuka::sensor::StateSensorKind::ContactWrench;
                fixture::Require(world.AttachStateSensor(desc, &id) == phi::Status::Ok, "contact sensor attachment failed");
            }
            break;
        }
    }
}

void AttachTactileSensors(nk::World& world, const Options& options, const fixture::PreparedScene& prepared) {
    if (!options.tactile_grid) return;
    std::vector<nuka::math::Transform> poses(world.GetModel().capacities.links_per_env);
    fixture::Require(world.GetData().DownloadField(nk::FieldId::LinkPose, poses.data(),
        poses.size() * sizeof(poses[0])), "tactile mounting pose download failed");
    const auto inverse = poses.at(prepared.front_link).rotation.Conjugate();
    nuka::sensor::StateSensorDesc desc;
    desc.kind = nuka::sensor::StateSensorKind::Touch;
    desc.mount = nuka::sensor::StateSensorMount::Link;
    desc.index = prepared.front_link;
    desc.local_offset.position = prepared.link_geom_local.at(prepared.front_link).position;
    desc.local_offset.rotation = inverse;
    desc.tactile.size = {0.1f, 0.1f, 0.1f};
    desc.tactile.hysteresis_strength = 0.2f;
    desc.tactile.hysteresis_time = 0.03f;
    desc.latency = double{options.dt} * 2.0;
    desc.latency_jitter = double{options.dt} * 0.5;
    desc.dropout_probability = 0.1f;
    desc.seed = options.seed;
    for (auto& error : desc.errors) {
        error.error.noise_density = 0.01f;
        error.error.initial_bias_stddev = 0.02f;
        error.error.bias_random_walk = 0.002f;
        error.error.correlated_bias_stddev = 0.005f;
        error.error.correlation_time = 0.2f;
    }
    uint32_t id;
    fixture::Require(world.AttachStateSensor(desc, &id) == phi::Status::Ok, "touch volume attachment failed");
    desc.kind = nuka::sensor::StateSensorKind::Tactile;
    const auto origin = desc.local_offset.position;
    const float pitch = 0.1f / static_cast<float>(options.tactile_grid);
    desc.tactile.size = {0.5f * pitch, 0.5f * pitch, 0.1f};
    desc.tactile.spread_fraction = 0.3f;
    desc.tactile.spread_sigma = 0.005f;
    for (uint32_t y = 0u; y < options.tactile_grid; ++y) {
        for (uint32_t x = 0u; x < options.tactile_grid; ++x) {
            desc.local_offset.position = origin + inverse.Rotate({(static_cast<float>(x) + 0.5f) * pitch - 0.05f,
                (static_cast<float>(y) + 0.5f) * pitch - 0.05f, 0.0f});
            fixture::Require(world.AttachStateSensor(desc, &id) == phi::Status::Ok, "taxel attachment failed");
        }
    }
}

std::vector<uint8_t> StateSensorBytes(nk::World& world, bool include_tactile = true) {
    std::vector<uint8_t> bytes;
    if (!world.StateSensors().HasActive()) return bytes;
    nuka::sensor::StateSensorBankSnapshot snapshot;
    fixture::Require(world.StateSensors().Capture(&snapshot) == phi::Status::Ok, "sensor snapshot failed");
    for (const auto& channel : snapshot.sensors)
        if (channel.active && (include_tactile || !nuka::sensor::IsContactRegionSensor(channel.desc.kind)))
            bytes.insert(bytes.end(), channel.bytes.begin(), channel.bytes.end());
    const auto* times = reinterpret_cast<const uint8_t*>(snapshot.times.data());
    bytes.insert(bytes.end(), times, times + snapshot.times.size() * sizeof(double));
    return bytes;
}

constexpr std::array<nk::FieldId, 14> kPhysicalFields{
    nk::FieldId::BasePose, nk::FieldId::Q, nk::FieldId::Qdot,
    nk::FieldId::LinkVelocity, nk::FieldId::BodyPose,
    nk::FieldId::BodyLinearVelocity, nk::FieldId::BodyAngularVelocity,
    nk::FieldId::ParticlePos, nk::FieldId::ParticlePrevPos, nk::FieldId::ParticleVel,
    nk::FieldId::ParticleF, nk::FieldId::ParticleC, nk::FieldId::ParticlePlastic, nk::FieldId::ParticlePlasticF};

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
    memory.Set("state_sensor_bytes", Json::Int(world.StateSensors().StorageBytes()));
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

Json Histogram(const std::map<uint32_t, uint64_t>& counts) {
    Json result = Json::Array();
    for (const auto& [size, count] : counts) {
        Json item = Json::Object();
        item.Set("size", Json::Int(size));
        item.Set("count", Json::Int(count));
        result.PushBack(std::move(item));
    }
    return result;
}

Json XpbdWorkload(const nk::Model& model) {
    Json result = Json::Object();
    const auto family = [&](const char* name, const std::vector<uint32_t>& segments,
                            uint32_t constraints) {
        Json item = Json::Object(), counts = Json::Array();
        uint32_t maximum = 0u;
        uint64_t total = 0u;
        fixture::Require(segments.size() % 2u == 0u, "invalid XPBD color table");
        for (size_t i = 0u; i < segments.size(); i += 2u) {
            fixture::Require(segments[i] == total, "non-contiguous XPBD colors");
            const auto count = segments[i + 1u];
            counts.PushBack(Json::Int(count));
            maximum = std::max(maximum, count);
            total += count;
        }
        fixture::Require(total == constraints, "XPBD colors do not cover constraints");
        item.Set("constraints_per_env", Json::Int(constraints));
        item.Set("constraints_total", Json::Int(total * model.capacities.env_count));
        item.Set("colors", Json::Int(segments.size() / 2u));
        item.Set("constraints_per_color_per_env", std::move(counts));
        item.Set("largest_color_total", Json::Int(uint64_t{maximum} * model.capacities.env_count));
        item.Set("iterations", Json::Int(std::max<uint16_t>(model.particles.xpbd_iters, 1u)));
        result.Set(name, std::move(item));
    };
    const auto& caps = model.capacities;
    family("distance", model.dist_color_segments, caps.dist_cons_per_env);
    family("volume", model.vol_color_segments, caps.vol_cons_per_env);
    family("shape_match", model.sm_color_segments, caps.shape_match_slots_per_env);
    return result;
}

Json VbdWorkload(const nk::Model& model) {
    const auto& caps = model.capacities;
    const auto& particles = model.particles;
    const uint64_t total_elements = nuka::CheckedProduct({caps.vbd_elements_per_env, caps.env_count});
    fixture::Require(total_elements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                     "VBD element count exceeds JSON integer range");
    std::array<uint64_t, 5> counts{};
    for (const nk::VbdElement& element : particles.vbd_elements) {
        switch (element.kind) {
            case nk::kVbdTriangle: ++counts[0]; break;
            case nk::kVbdHinge: ++counts[1]; break;
            case nk::kVbdSpring: ++counts[2]; break;
            case nk::kVbdRodBend: ++counts[3]; break;
            default: ++counts[4]; break;
        }
    }
    Json kinds = Json::Object();
    const char* names[] = {"triangle", "hinge", "spring", "rod_bend", "unknown"};
    for (size_t i = 0u; i < counts.size(); ++i) kinds.Set(names[i], Json::Int(counts[i]));
    fixture::Require(particles.vbd_color_segments.size() == size_t{caps.vbd_colors} * 2u,
                     "invalid VBD color table");
    Json colors = Json::Array();
    uint64_t colored_vertices = 0u;
    for (uint32_t color = 0u; color < caps.vbd_colors; ++color) {
        const uint32_t count = particles.vbd_color_segments[size_t{color} * 2u + 1u];
        colors.PushBack(Json::Int(count));
        colored_vertices = nuka::CheckedAdd(colored_vertices, count);
    }
    Json result = Json::Object();
    result.Set("vertices_per_env", Json::Int(caps.vbd_vertices_per_env));
    result.Set("particle_begin_per_env", Json::Int(caps.vbd_particle_begin));
    result.Set("elements_per_env", Json::Int(caps.vbd_elements_per_env));
    result.Set("total_elements", Json::Int(static_cast<int64_t>(total_elements)));
    result.Set("counts_by_kind", std::move(kinds));
    result.Set("counts_by_kind_scope", Json::Str("per environment"));
    result.Set("static_colors", Json::Int(caps.vbd_colors));
    result.Set("colored_vertices_per_env", Json::Int(colored_vertices));
    result.Set("vertices_per_color_per_env", std::move(colors));
    result.Set("particle_mode", Json::Int(static_cast<uint32_t>(particles.mode)));
    result.Set("coupled_internal", Json::Int(static_cast<uint32_t>(particles.coupled_internal)));
    return result;
}

Json XpbdQuality(nk::World& world, uint32_t step) {
    const auto& model = world.GetModel();
    const auto& particles = model.particles;
    const auto& caps = model.capacities;
    fixture::Require(particles.inv_mass.size() >= caps.particles_per_env &&
                     particles.initial_pos.size() >= caps.particles_per_env &&
                     particles.dist_a.size() >= caps.dist_cons_per_env &&
                     particles.dist_b.size() >= caps.dist_cons_per_env &&
                     particles.dist_rest.size() >= caps.dist_cons_per_env,
                     "particle quality input extent is invalid");
    for (uint32_t i = 0u; i < caps.particles_per_env; ++i) {
        const auto& initial = particles.initial_pos[i];
        fixture::Require(std::isfinite(particles.inv_mass[i]) && particles.inv_mass[i] >= 0.0f &&
                         std::isfinite(initial.x) && std::isfinite(initial.y) && std::isfinite(initial.z),
                         "particle quality input is nonfinite or has invalid inverse mass");
    }
    std::vector<nuka::math::Vec3> positions(size_t{caps.env_count} * caps.particles_per_env);
    fixture::Require(world.GetData().DownloadField(nk::FieldId::ParticlePos, positions.data(),
                     positions.size() * sizeof(nuka::math::Vec3)), "XPBD position download failed");
    double distance_max = 0.0, distance_squared = 0.0;
    double pinned_max = 0.0, distance_rms_max_env = 0.0;
    uint64_t distance_count = 0u, defined_distance_count = 0u;
    uint64_t hard_distance_constraint_samples = 0u, vbd_edge_samples = 0u;
    bool finite = true, pinned_finite = true, strain_defined = true;
    std::vector<uint32_t> nonfinite_count(caps.env_count, 0u), first_nonfinite_index(caps.env_count, 0u);
    const auto point_finite = [](const nuka::math::Vec3& point) {
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
    };
    for (uint32_t env = 0u; env < caps.env_count; ++env) {
        const auto* points = positions.data() + size_t{env} * caps.particles_per_env;
        double env_distance_squared = 0.0;
        bool env_strain_defined = true;
        for (uint32_t i = 0u; i < caps.particles_per_env; ++i) {
            const auto& point = points[i];
            const bool valid_point = point_finite(point);
            if (!valid_point) {
                finite = false;
                if (nonfinite_count[env] == 0u) first_nonfinite_index[env] = i;
                ++nonfinite_count[env];
            }
            if (particles.inv_mass[i] != 0.0f) continue;
            if (!valid_point) {
                pinned_finite = false;
                continue;
            }
            const auto& initial = particles.initial_pos[i];
            const double dx = double{point.x} - initial.x;
            const double dy = double{point.y} - initial.y;
            const double dz = double{point.z} - initial.z;
            pinned_max = std::max(pinned_max, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        uint64_t env_distance_count = 0u;
        const auto strain = [&](uint32_t ia, uint32_t ib, double rest) {
            fixture::Require(ia < caps.particles_per_env && ib < caps.particles_per_env &&
                             rest > 0.0 && std::isfinite(rest), "invalid distance strain indices or rest length");
            ++distance_count;
            ++env_distance_count;
            const auto& a = points[ia];
            const auto& b = points[ib];
            if (!point_finite(a) || !point_finite(b)) {
                strain_defined = false;
                env_strain_defined = false;
                return;
            }
            const double dx = double{a.x} - b.x, dy = double{a.y} - b.y, dz = double{a.z} - b.z;
            const double error = std::abs(std::sqrt(dx * dx + dy * dy + dz * dz) / rest - 1.0);
            if (!std::isfinite(error) || !std::isfinite(error * error)) {
                strain_defined = false;
                env_strain_defined = false;
                return;
            }
            distance_max = std::max(distance_max, error);
            distance_squared += error * error;
            env_distance_squared += error * error;
            ++defined_distance_count;
        };
        for (uint32_t i = 0u; i < caps.dist_cons_per_env; ++i) {
            strain(particles.dist_a[i], particles.dist_b[i], particles.dist_rest[i]);
            ++hard_distance_constraint_samples;
        }
        // Vertex-block edges measure strain against the cooked rest shape.
        for (const nk::VbdElement& element : particles.vbd_elements) {
            const uint32_t n = element.kind == nk::kVbdTriangle ? 3u
                : element.kind == nk::kVbdSpring ? 2u : 0u;
            for (uint32_t j = 0u; j < n && (n == 3u || j == 0u); ++j) {
                const uint64_t a64 = uint64_t{caps.vbd_particle_begin} + element.vertex[j];
                const uint64_t b64 = uint64_t{caps.vbd_particle_begin} + element.vertex[(j + 1u) % n];
                fixture::Require(a64 < caps.particles_per_env && b64 < caps.particles_per_env,
                                 "VBD quality edge index exceeds the particle extent");
                const uint32_t a = static_cast<uint32_t>(a64), b = static_cast<uint32_t>(b64);
                const auto d = particles.initial_pos[a] - particles.initial_pos[b];
                strain(a, b, std::sqrt(double{d.x} * d.x + double{d.y} * d.y + double{d.z} * d.z));
                ++vbd_edge_samples;
            }
        }
        if (env_distance_count > 0u && env_strain_defined)
            distance_rms_max_env = std::max(distance_rms_max_env,
                std::sqrt(env_distance_squared / static_cast<double>(env_distance_count)));
    }
    strain_defined = strain_defined && std::isfinite(distance_squared) && std::isfinite(distance_rms_max_env);
    Json result = Json::Object(), nonfinite_counts = Json::Array(), nonfinite_indices = Json::Array();
    for (uint32_t env = 0u; env < caps.env_count; ++env) {
        nonfinite_counts.PushBack(Json::Int(nonfinite_count[env]));
        nonfinite_indices.PushBack(nonfinite_count[env] != 0u ? Json::Int(first_nonfinite_index[env]) : Json::Null());
    }
    result.Set("step", Json::Int(step));
    result.Set("finite", Json::Bool(finite));
    result.Set("nonfinite_particle_count_by_env", std::move(nonfinite_counts));
    result.Set("nonfinite_particle_first_index_by_env", std::move(nonfinite_indices));
    result.Set("nonfinite_particle_scope", Json::Str("position coordinates; first indices are environment-local, null when none"));
    result.Set("pinned_valid", Json::Bool(pinned_finite && pinned_max == 0.0));
    result.Set("strain_defined", Json::Bool(strain_defined));
    result.Set("distance_count", Json::Int(distance_count));
    result.Set("defined_distance_count", Json::Int(defined_distance_count));
    result.Set("hard_distance_constraint_samples", Json::Int(hard_distance_constraint_samples));
    result.Set("vbd_edge_samples", Json::Int(vbd_edge_samples));
    result.Set("distance_count_scope", Json::Str("hard_distance_constraint_samples + vbd_edge_samples, including undefined samples"));
    result.Set("distance_strain_max", strain_defined ? Json::Float(distance_max) : Json::Null());
    result.Set("distance_strain_rms", strain_defined ? Json::Float(distance_count ?
        std::sqrt(distance_squared / static_cast<double>(distance_count)) : 0.0) : Json::Null());
    result.Set("distance_strain_rms_max_env", strain_defined ? Json::Float(distance_rms_max_env) : Json::Null());
    result.Set("pinned_displacement_max_m", pinned_finite ? Json::Float(pinned_max) : Json::Null());
    return result;
}

struct XpbdAcceptance {
    static constexpr double max_strain_limit = 0.05, rms_strain_limit = 0.01;
    double max_strain = 0.0, max_rms_strain = 0.0;
    uint32_t steps_checked = 0u, first_failed_step = 0u, first_nonfinite_step = 0u;
    uint32_t first_pinned_failure_step = 0u, first_undefined_strain_step = 0u, first_strain_failure_step = 0u;
    bool strain_coverage = true;
    Json first_nonfinite_sample = Json::Null(), first_pinned_failure_sample = Json::Null();

    void Observe(const Json& sample) {
        const uint32_t step = static_cast<uint32_t>(sample.At("step").AsInt());
        const bool finite = sample.At("finite").AsBool(), pinned_valid = sample.At("pinned_valid").AsBool();
        const bool measured = sample.At("strain_defined").AsBool() &&
                              !sample.At("distance_strain_max").IsNull() && !sample.At("distance_strain_rms_max_env").IsNull();
        bool strain_failed = false;
        if (measured) {
            const double strain = sample.At("distance_strain_max").AsDouble();
            const double rms = sample.At("distance_strain_rms_max_env").AsDouble();
            fixture::Require(std::isfinite(strain) && std::isfinite(rms), "defined strain report contains a nonfinite value");
            max_strain = std::max(max_strain, strain);
            max_rms_strain = std::max(max_rms_strain, rms);
            strain_failed = strain > max_strain_limit || rms > rms_strain_limit;
            if (strain_failed && first_strain_failure_step == 0u) first_strain_failure_step = step;
        } else {
            strain_coverage = false;
            if (first_undefined_strain_step == 0u) first_undefined_strain_step = step;
        }
        if (!finite && first_nonfinite_step == 0u) {
            first_nonfinite_step = step;
            first_nonfinite_sample = sample;
        }
        if (!pinned_valid && first_pinned_failure_step == 0u) {
            first_pinned_failure_step = step;
            first_pinned_failure_sample = sample;
        }
        ++steps_checked;
        if (first_failed_step == 0u && (!finite || !pinned_valid || !measured || strain_failed))
            first_failed_step = step;
    }

    bool Valid() const { return steps_checked > 0u && first_failed_step == 0u; }

    Json Report() const {
        Json result = Json::Object();
        result.Set("scope", Json::Str("all replay steps and environments, including warmup; finite positions, fixed pins and edge strain against cooked rest shape"));
        result.Set("budget", Json::Str("declared deformation budget: 5% maximum edge strain and 1% per-environment RMS"));
        result.Set("distance_strain_max_limit", Json::Float(max_strain_limit));
        result.Set("distance_strain_rms_limit", Json::Float(rms_strain_limit));
        result.Set("distance_strain_max", strain_coverage ? Json::Float(max_strain) : Json::Null());
        result.Set("distance_strain_rms_max_env", strain_coverage ? Json::Float(max_rms_strain) : Json::Null());
        result.Set("strain_coverage_complete", Json::Bool(strain_coverage));
        result.Set("finite", Json::Bool(first_nonfinite_step == 0u));
        result.Set("pinned_valid", Json::Bool(first_pinned_failure_step == 0u));
        result.Set("steps_checked", Json::Int(steps_checked));
        result.Set("first_failed_step", first_failed_step > 0u ? Json::Int(first_failed_step) : Json::Null());
        result.Set("first_nonfinite_step", first_nonfinite_step > 0u ? Json::Int(first_nonfinite_step) : Json::Null());
        result.Set("first_nonfinite_sample", first_nonfinite_sample);
        result.Set("first_pinned_failure_step", first_pinned_failure_step > 0u ? Json::Int(first_pinned_failure_step) : Json::Null());
        result.Set("first_pinned_failure_sample", first_pinned_failure_sample);
        result.Set("first_undefined_strain_step", first_undefined_strain_step > 0u ? Json::Int(first_undefined_strain_step) : Json::Null());
        result.Set("first_strain_failure_step", first_strain_failure_step > 0u ? Json::Int(first_strain_failure_step) : Json::Null());
        result.Set("valid", Json::Bool(Valid()));
        return result;
    }
};

Json IslandWorkload(nk::World& world, const std::vector<nk::NkRow>& rows, uint32_t step) {
    const auto& caps = world.GetModel().capacities;
    uint32_t island_count = 0u, active_count = 0u;
    uint64_t articulation_sides = 0u;
    for (const auto& row : rows) {
        if (!(row.flags & nk::nk_row_flags::kActive)) continue;
        ++active_count;
        articulation_sides += (row.a.kind == nk::kNkSideArtic) + (row.b.kind == nk::kNkSideArtic);
    }
    fixture::Require(world.GetData().DownloadField(nk::FieldId::IslandCount,
                     &island_count, sizeof(island_count)), "island count download failed");
    fixture::Require(island_count <= active_count, "island count exceeds active rows");
    std::vector<std::array<uint32_t, 4>> islands(island_count);
    std::vector<uint32_t> order(active_count);
    if (island_count != 0u)
        fixture::Require(world.GetData().DownloadField(nk::FieldId::IslandQuads, islands.data(),
                         islands.size() * sizeof(islands.front())), "island span download failed");
    if (active_count != 0u)
        fixture::Require(world.GetData().DownloadField(nk::FieldId::IslandRows, order.data(),
                         order.size() * sizeof(uint32_t)), "island rows download failed");
    std::vector<bool> visited(rows.size(), false);
    uint32_t visited_count = 0u, scheduled_count = 0u, articulation_islands = 0u;
    std::map<uint32_t, uint64_t> row_histogram, articulation_row_histogram, tree_histogram;
    for (const auto& island : islands) {
        const auto [offset, count, flags, env] = island;
        fixture::Require(env < caps.env_count && count != 0u &&
                         uint64_t{offset} + count <= order.size(), "invalid island span");
        std::set<uint32_t> trees;
        const auto visit = [&](uint32_t id) {
            fixture::Require(id < rows.size() && id / caps.max_rows_per_env == env && !visited[id] &&
                             (rows[id].flags & nk::nk_row_flags::kActive), "invalid island row ownership");
            visited[id] = true;
            ++visited_count;
            for (const auto& side : {rows[id].a, rows[id].b}) {
                if (side.kind == nk::kNkSideArtic) trees.insert(side.index);
            }
        };
        for (uint32_t i = 0u; i < count; ++i) {
            const uint32_t id = order[offset + i];
            fixture::Require(id < rows.size() && !(rows[id].flags & nk::nk_row_flags::kBlockTangent) &&
                             (i == 0u || order[offset + i - 1u] < id), "invalid island row ownership or order");
            visit(id);
            ++scheduled_count;
            if (rows[id].flags & nk::nk_row_flags::kBlockNormal) {
                for (uint32_t tangent = 1u; tangent <= 2u; ++tangent) {
                    const uint32_t spoke = id + tangent * rows[id].group_normal_count;
                    fixture::Require(spoke < rows.size() &&
                        (rows[spoke].flags & nk::nk_row_flags::kBlockTangent), "invalid friction block ownership");
                    visit(spoke);
                }
            }
        }
        fixture::Require(((flags & 1u) != 0u) == !trees.empty(), "invalid island articulation flag");
        ++row_histogram[count];
        ++tree_histogram[static_cast<uint32_t>(trees.size())];
        if (!trees.empty()) { ++articulation_islands; ++articulation_row_histogram[count]; }
    }
    fixture::Require(visited_count == active_count, "active rows missing from island schedule");
    std::sort(islands.begin(), islands.end(), [&](const auto& a, const auto& b) {
        return order[a[0]] < order[b[0]];
    });
    uint64_t hash = 14695981039346656037ull;
    for (const auto& island : islands) {
        hash = UpdateDigest(hash, island.data() + 1u, 3u * sizeof(uint32_t));
        hash = UpdateDigest(hash, order.data() + island[0], island[1] * sizeof(uint32_t));
    }
    Json result = Json::Object();
    result.Set("step", Json::Int(step));
    result.Set("schedule_type", Json::Str("island"));
    result.Set("active_rows", Json::Int(active_count));
    result.Set("scheduled_contact_blocks_and_rows", Json::Int(scheduled_count));
    result.Set("articulation_sides", Json::Int(articulation_sides));
    result.Set("islands", Json::Int(island_count));
    result.Set("articulation_islands", Json::Int(articulation_islands));
    result.Set("rows_per_island", Histogram(row_histogram));
    result.Set("rows_per_articulation_island", Histogram(articulation_row_histogram));
    result.Set("trees_per_island", Histogram(tree_histogram));
    result.Set("canonical_schedule_fnv1a64", Json::Str(FormatDigest(hash)));
    return result;
}

Json BlockWorkload(nk::World& world, const std::vector<nk::NkRow>& rows, uint32_t step,
                   const phi::BlockDescentSolveParams& p) {
    const auto& model = world.GetModel();
    const auto& caps = model.capacities;
    fixture::Require(p.env_count != 0u && p.env_count == caps.env_count &&
                     p.rows_per_env == caps.max_rows_per_env,
                     "invalid block schedule environment or row extent");
    fixture::Require(p.total_particle_count == nuka::CheckedProduct({caps.particles_per_env, p.env_count}) &&
                     p.total_body_count == nuka::CheckedProduct({caps.bodies_per_env, p.env_count}) &&
                     p.articulation_count == nuka::CheckedProduct({model.articulation.articulation_count, p.env_count}) &&
                     p.total_grid_count == nuka::CheckedProduct({caps.mpm_grid_nodes_per_env, p.env_count}),
                     "block schedule owner extents disagree with the model");
    fixture::Require(p.material_cells_per_env == caps.mpm_stress_cells_per_env,
                     "block material cell extent disagrees with the model");
    const uint64_t material_rows = nuka::CheckedProduct({p.material_cells_per_env, nk::kMpmStressRowsPerCell});
    fixture::Require(material_rows <= p.rows_per_env, "material reserved rows exceed the environment row extent");
    const uint64_t material_begin = p.rows_per_env - material_rows;
    const uint64_t owners = nuka::CheckedAdd(
        nuka::CheckedAdd(p.total_particle_count, p.total_body_count),
        nuka::CheckedAdd(p.articulation_count, p.total_grid_count));
    const uint64_t row_count = nuka::CheckedProduct({p.rows_per_env, p.env_count});
    fixture::Require(owners <= UINT32_MAX && row_count <= UINT32_MAX && row_count == rows.size(),
                     "block schedule exceeds row or owner index extent");
    const auto layout = nk::MakeBlockRowScheduleLayout(owners, row_count, p.max_point_terms);
    fixture::Require(layout.Words() <= caps.ElementCount(nk::FieldId::BlockDescentScratch),
                     "block schedule prefix exceeds scratch storage");
    const auto download = [&](uint64_t word, void* destination, uint64_t count, const char* message) {
        if (count != 0u)
            fixture::Require(world.GetData().DownloadField(nk::FieldId::BlockDescentScratch, destination,
                nuka::CheckedProduct({count, sizeof(uint32_t)}),
                nuka::CheckedProduct({word, sizeof(uint32_t)})), message);
    };
    std::vector<uint32_t> offsets(static_cast<size_t>(owners + 1u)), counts(offsets.size());
    uint32_t scheduled_count = 0u;
    download(layout.OffsetsWord(), offsets.data(), offsets.size(), "block offsets download failed");
    download(layout.CountsWord(), counts.data(), counts.size(), "block counts download failed");
    download(layout.ActiveCountWord(), &scheduled_count, 1u, "active block rows count download failed");
    fixture::Require(offsets.front() == 0u && scheduled_count <= row_count,
                     "invalid block schedule origin or active row count");
    for (uint32_t owner = 0u; owner < owners; ++owner) {
        fixture::Require(offsets[owner] <= offsets[owner + 1u] &&
                         offsets[owner + 1u] <= layout.incidence_capacity &&
                         counts[owner] == offsets[owner + 1u] - offsets[owner],
                         "invalid block CSR offset or count");
    }
    fixture::Require(offsets.back() <= layout.incidence_capacity && counts.back() == 0u,
                     "invalid block CSR terminal offset or sentinel count");
    std::vector<uint32_t> failure_reasons(static_cast<size_t>(owners)), failure_rows(failure_reasons.size());
    std::vector<uint32_t> failure_substeps(failure_reasons.size());
    download(layout.FailureReasonsWord(), failure_reasons.data(), failure_reasons.size(),
             "block failure reasons download failed");
    download(layout.FailureRowsWord(), failure_rows.data(), failure_rows.size(),
             "block failure witness rows download failed");
    download(layout.FailureSubstepsWord(), failure_substeps.data(), failure_substeps.size(),
             "block failure substeps download failed");
    Json solver_failure_owners = Json::Array(), solver_failure_equations = Json::Array();
    for (uint32_t owner = 0u; owner < owners; ++owner) {
        if (failure_reasons[owner] == 0u) continue;
        Json failure = Json::Array();
        failure.PushBack(Json::Int(owner));
        failure.PushBack(Json::Int(failure_reasons[owner]));
        failure.PushBack(Json::Int(failure_rows[owner]));
        failure.PushBack(Json::Int(failure_substeps[owner]));
        solver_failure_owners.PushBack(std::move(failure));
        std::array<double, nk::kBlockSolveFailureEquationColumnCount> equation;
        const uint64_t equation_word = nuka::CheckedAdd(layout.FailureEquationsWord(),
            nuka::CheckedProduct({owner, nk::kBlockSolveFailureEquationColumnCount, 2u}));
        download(equation_word, equation.data(), uint64_t{equation.size()} * 2u,
                 "block failure equation download failed");
        Json record = Json::Array();
        record.PushBack(Json::Int(owner));
        for (double value : equation) {
            if (!std::isfinite(value)) {
                record.PushBack(Json::Str(std::isnan(value) ? "NaN" : value < 0.0 ? "-Infinity" : "Infinity"));
            } else {
                std::ostringstream encoded;
                encoded << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
                record.PushBack(Json::Str(encoded.str()));
            }
        }
        solver_failure_equations.PushBack(std::move(record));
    }
    std::vector<uint32_t> active(scheduled_count), incidence(offsets.back());
    download(layout.ActiveRowsWord(), active.data(), active.size(), "active block rows download failed");
    download(layout.IncidenceWord(), incidence.data(), incidence.size(), "block incidence download failed");

    static_assert(sizeof(float) == sizeof(uint32_t), "cached penalties require one scratch word");
    fixture::Require(nuka::CheckedAdd(layout.Words(), row_count) <=
                     caps.ElementCount(nk::FieldId::BlockDescentScratch),
                     "cached block penalties exceed scratch storage");
    std::vector<float> penalties(static_cast<size_t>(row_count));
    download(layout.Words(), penalties.data(), row_count, "cached block penalties download failed");
    struct PenaltyRange {
        uint64_t active_count = 0u, finite_count = 0u;
        double min = 0.0, max = 0.0;
        uint32_t max_row_id = 0u;
        void Observe(float value, uint32_t row) {
            ++active_count;
            if (!std::isfinite(value)) return;
            if (finite_count == 0u) { min = max = value; max_row_id = row; }
            else {
                min = std::min(min, double{value});
                if (value > max || (value == max && row < max_row_id)) { max = value; max_row_id = row; }
            }
            ++finite_count;
        }
        Json Report() const {
            Json result = Json::Object();
            result.Set("active_count", Json::Int(active_count));
            result.Set("finite_count", Json::Int(finite_count));
            result.Set("finite", Json::Bool(finite_count == active_count));
            result.Set("min", finite_count != 0u ? Json::Float(min) : Json::Null());
            result.Set("max", finite_count != 0u ? Json::Float(max) : Json::Null());
            result.Set("max_row_id", finite_count != 0u ? Json::Int(max_row_id) : Json::Null());
            return result;
        }
    };
    PenaltyRange penalty_range;
    std::map<std::pair<uint32_t, uint32_t>, PenaltyRange> penalty_by_endpoint_kind;

    std::vector<nk::PointEndpointRange> ranges(caps.ElementCount(nk::FieldId::PointEndpointRanges));
    std::vector<nk::PointEndpointTerm> terms(caps.ElementCount(nk::FieldId::PointEndpointTerms));
    if (!ranges.empty())
        fixture::Require(world.GetData().DownloadField(nk::FieldId::PointEndpointRanges,
            ranges.data(), ranges.size() * sizeof(ranges[0])), "block endpoint range download failed");
    if (!terms.empty())
        fixture::Require(world.GetData().DownloadField(nk::FieldId::PointEndpointTerms,
            terms.data(), terms.size() * sizeof(terms[0])), "block endpoint term download failed");
    const auto append_owner = [&](uint32_t kind, uint32_t index, uint32_t env,
                                  std::vector<uint32_t>& result) {
        if (kind == nk::kNkSideStatic) return;
        uint32_t per_env = 0u, total = 0u;
        uint64_t first = 0u;
        if (kind == nk::kNkSideParticle) {
            per_env = caps.particles_per_env;
            total = p.total_particle_count;
        } else if (kind == nk::kNkSideRigid) {
            per_env = caps.bodies_per_env;
            total = p.total_body_count;
            first = p.total_particle_count;
        } else if (kind == nk::kNkSideArtic) {
            per_env = caps.articulations_per_env;
            total = p.articulation_count;
            first = uint64_t{p.total_particle_count} + p.total_body_count;
        } else if (kind == nk::kNkSideGrid) {
            per_env = caps.mpm_grid_nodes_per_env;
            total = p.total_grid_count;
            first = uint64_t{p.total_particle_count} + p.total_body_count + p.articulation_count;
        } else {
            fixture::Require(false, "invalid block endpoint kind");
        }
        fixture::Require(per_env != 0u && index < total && index / per_env == env &&
                         first + index < owners, "block endpoint index or environment is invalid");
        result.push_back(static_cast<uint32_t>(first + index));
    };
    const auto row_owners = [&](const nk::NkRow& row, uint32_t env) {
        std::vector<uint32_t> result;
        for (const auto& side : {row.a, row.b}) {
            if (side.kind != nk::kNkSidePointEndpoint) {
                if (row.flags & nk::nk_row_flags::kMaterialBlock)
                    fixture::Require(side.kind == nk::kNkSideStatic, "material block requires one interpolated endpoint");
                append_owner(side.kind, side.index, env, result);
                continue;
            }
            fixture::Require(caps.point_endpoints_per_env != 0u && side.index < ranges.size() &&
                             side.index / caps.point_endpoints_per_env == env,
                             "block interpolated endpoint index or environment is invalid");
            const auto range = ranges[side.index];
            const uint64_t first = nuka::CheckedProduct({env, caps.point_endpoint_terms_per_env});
            const uint64_t end = nuka::CheckedAdd(first, caps.point_endpoint_terms_per_env);
            fixture::Require(range.count != 0u && range.first <= terms.size() &&
                             range.count <= terms.size() - range.first &&
                             range.first >= first && uint64_t{range.first} + range.count <= end,
                             "block interpolated endpoint range crosses storage or environment");
            for (uint32_t term = 0u; term < range.count; ++term) {
                const auto& entry = terms[range.first + term];
                if (row.flags & nk::nk_row_flags::kMaterialBlock)
                    fixture::Require(entry.kind == nk::kNkSideParticle || entry.kind == nk::kNkSideGrid,
                                     "material endpoint term is not a particle or grid owner");
                append_owner(entry.kind, entry.index, env, result);
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    };

    uint64_t grid_lattice_nodes = 0u;
    if (p.total_grid_count != 0u) {
        fixture::Require(p.grid_dims[0] != 0u && p.grid_dims[1] != 0u && p.grid_dims[2] != 0u,
                         "grid color coverage requires actual nonzero grid dimensions");
        grid_lattice_nodes = nuka::CheckedProduct({p.grid_dims[0], p.grid_dims[1], p.grid_dims[2]});
        fixture::Require(nuka::CheckedProduct({grid_lattice_nodes, nk::kMpmLattices}) == caps.mpm_grid_nodes_per_env,
                         "grid color topology disagrees with the owner extent");
    }
    const uint64_t grid_owner_begin =
        uint64_t{p.total_particle_count} + p.total_body_count + p.articulation_count;
    std::vector<std::set<uint32_t>> occupied_cells(p.env_count), expected_cells(p.env_count);
    std::vector<std::set<uint32_t>> material_heads(p.env_count);
    if (p.material_cells_per_env != 0u) {
        fixture::Require(p.grid_particles_per_env != 0u &&
                         p.grid_particles_per_env == model.MpmParticlesPerEnv() &&
                         p.grid_particles_per_env <= caps.particles_per_env &&
                         p.grid_dims[0] != 0u && p.grid_dims[1] != 0u && p.grid_dims[2] != 0u,
                         "material coverage requires valid grid dimensions and MPM particle extent");
        const uint64_t lattice_nodes = nuka::CheckedProduct({p.grid_dims[0], p.grid_dims[1], p.grid_dims[2]});
        const uint64_t half_cells_per_env = nuka::CheckedProduct({lattice_nodes, nk::kMpmHalfCellsPerLatticeNode});
        const uint64_t total_half_cells = nuka::CheckedProduct({half_cells_per_env, p.env_count});
        fixture::Require(nuka::CheckedProduct({lattice_nodes, nk::kMpmLattices}) == caps.mpm_grid_nodes_per_env &&
                         total_half_cells <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                         "material coverage grid or half-cell key extent is invalid or overflowing");
        const uint64_t mpm_count = nuka::CheckedProduct({p.grid_particles_per_env, p.env_count});
        fixture::Require(mpm_count <= caps.ElementCount(nk::FieldId::MpmGridCellKey) &&
                         mpm_count <= caps.ElementCount(nk::FieldId::MpmGridPartIdx),
                         "material coverage key prefix exceeds storage");
        std::vector<uint32_t> keys(static_cast<size_t>(mpm_count)), particles(keys.size());
        std::vector<float> volume(p.total_particle_count);
        fixture::Require(world.GetData().DownloadField(nk::FieldId::MpmGridCellKey, keys.data(),
                         keys.size() * sizeof(keys[0])), "material Predict half-cell keys download failed");
        fixture::Require(world.GetData().DownloadField(nk::FieldId::MpmGridPartIdx, particles.data(),
                         particles.size() * sizeof(particles[0])), "material Predict particle indices download failed");
        fixture::Require(world.GetData().DownloadField(nk::FieldId::ParticleVol0, volume.data(),
                         volume.size() * sizeof(volume[0])), "material reference volume download failed");
        std::vector<bool> particle_seen(p.total_particle_count, false);
        for (size_t item = 0u; item < keys.size(); ++item) {
            const uint32_t particle = particles[item];
            const uint32_t env = static_cast<uint32_t>(item / p.grid_particles_per_env);
            fixture::Require(particle < volume.size() && !particle_seen[particle] &&
                             particle / caps.particles_per_env == env &&
                             particle % caps.particles_per_env < p.grid_particles_per_env,
                             "material Predict index is repeated, invalid or crosses environments");
            particle_seen[particle] = true;
            fixture::Require(keys[item] < total_half_cells && keys[item] / half_cells_per_env == env,
                             "material Predict half-cell key is invalid or crosses environments");
            fixture::Require(std::isfinite(volume[particle]), "material reference volume is nonfinite");
            const uint32_t cell = static_cast<uint32_t>((keys[item] % half_cells_per_env) /
                                                       nk::kMpmHalfCellsPerLatticeNode);
            occupied_cells[env].insert(cell);
            if (volume[particle] > 0.0f) expected_cells[env].insert(cell);
        }
        for (uint32_t env = 0u; env < p.env_count; ++env)
            fixture::Require(occupied_cells[env].size() <= p.material_cells_per_env,
                             "occupied material cells exceed the reserved per-environment capacity");
    }

    std::vector<bool> scheduled(rows.size(), false), visited(rows.size(), false);
    std::vector<std::vector<uint32_t>> expected(static_cast<size_t>(owners));
    std::map<uint32_t, uint64_t> rows_per_block, blocks_per_row;
    uint64_t expected_incidence_count = 0u, material_blocks = 0u;
    uint64_t grid_color_conflict_count = 0u, grid_color_heads_checked = 0u;
    for (uint32_t id : active) {
        fixture::Require(id < rows.size() && !scheduled[id] &&
                         (rows[id].flags & nk::nk_row_flags::kActive) &&
                         !(rows[id].flags & nk::nk_row_flags::kBlockTangent),
                         "invalid or repeated active block row");
        scheduled[id] = true;
        penalty_range.Observe(penalties[id], id);
        penalty_by_endpoint_kind[{std::min(rows[id].a.kind, rows[id].b.kind),
                                  std::max(rows[id].a.kind, rows[id].b.kind)}].Observe(penalties[id], id);
        const uint32_t env = id / p.rows_per_env;
        fixture::Require(((rows[id].flags & nk::nk_row_flags::kMaterialBlock) != 0u) ==
                         (id % p.rows_per_env >= material_begin),
                         "active row disagrees with the material reserved region");
        const auto visit = [&](uint64_t slot) {
            fixture::Require(slot < rows.size() && slot / p.rows_per_env == env && !visited[slot] &&
                             (rows[slot].flags & nk::nk_row_flags::kActive),
                             "invalid, repeated or cross-environment block row coverage");
            visited[slot] = true;
        };
        visit(id);
        if (rows[id].flags & nk::nk_row_flags::kMaterialBlock) {
            const auto& head = rows[id];
            const uint64_t local = id % p.rows_per_env;
            fixture::Require(local >= material_begin &&
                             (local - material_begin) % nk::kMpmStressRowsPerCell == 0u &&
                             local + nk::kMpmStressRowsPerCell <= p.rows_per_env &&
                             uint64_t{id} + nk::kMpmStressRowsPerCell <= rows.size() &&
                             head.group_first == id && head.group_normal_count == 1u && head.env == env &&
                             !(head.flags & nk::nk_row_flags::kBlockNormal) &&
                             head.a.kind == nk::kNkSidePointEndpoint && std::isfinite(head.compliance_alpha),
                             "invalid material block head or reserved row alignment");
            for (uint32_t axis = 1u; axis < nk::kMpmStressRowsPerCell; ++axis) {
                const auto& payload = rows[id + axis];
                fixture::Require(!(payload.flags & nk::nk_row_flags::kActive) &&
                                 payload.group_first == id && payload.group_normal_count == 1u &&
                                 payload.env == head.env && payload.a.kind == head.a.kind &&
                                 payload.a.index == head.a.index && std::isfinite(payload.compliance_alpha),
                                 "invalid inactive material payload row");
            }
            material_heads[env].insert(id);
            ++material_blocks;
        } else if (rows[id].flags & nk::nk_row_flags::kBlockNormal) {
            fixture::Require(rows[id].group_normal_count != 0u, "block normal has zero tangent stride");
            for (uint32_t axis = 1u; axis <= 2u; ++axis) {
                const uint64_t tangent = uint64_t{id} + uint64_t{axis} * rows[id].group_normal_count;
                fixture::Require(tangent < rows.size() &&
                                 (rows[tangent].flags & nk::nk_row_flags::kBlockTangent) &&
                                 !(rows[tangent].flags & (nk::nk_row_flags::kBlockNormal | nk::nk_row_flags::kMaterialBlock)),
                                 "invalid block tangent ownership");
                visit(tangent);
            }
        }
        const auto associated = row_owners(rows[id], env);
        fixture::Require(!associated.empty(), "active block row has no physical owner");
        fixture::Require(associated.size() <= UINT32_MAX, "block row owner count exceeds index extent");
        std::map<uint32_t, uint32_t> color_counts;
        for (uint32_t owner : associated) {
            if (owner < grid_owner_begin) continue;
            fixture::Require(grid_lattice_nodes != 0u && owner - grid_owner_begin < p.total_grid_count,
                             "grid color owner lacks a measured topology");
            const uint64_t node = (owner - grid_owner_begin) % caps.mpm_grid_nodes_per_env;
            const uint32_t lattice = static_cast<uint32_t>(node / grid_lattice_nodes);
            fixture::Require(lattice < nk::kMpmLattices, "unsupported grid lattice in color coverage");
            const uint64_t spatial = node % grid_lattice_nodes;
            const uint32_t x = static_cast<uint32_t>(spatial % p.grid_dims[0]);
            const uint32_t y = static_cast<uint32_t>((spatial / p.grid_dims[0]) % p.grid_dims[1]);
            const uint32_t z = static_cast<uint32_t>(spatial / (uint64_t{p.grid_dims[0]} * p.grid_dims[1]));
            const uint32_t color = lattice == 0u ? x % 2u + 2u * (y % 2u + 2u * (z % 2u)) :
                nk::kMpmLatticeStencilNodes + x % 3u + 3u * (y % 3u + 3u * (z % 3u));
            fixture::Require(color < nk::kMpmCellStencilNodes, "grid color exceeds the shared stencil topology");
            grid_color_conflict_count += color_counts[color]++;
        }
        ++grid_color_heads_checked;
        ++blocks_per_row[static_cast<uint32_t>(associated.size())];
        expected_incidence_count = nuka::CheckedAdd(expected_incidence_count, associated.size());
        for (uint32_t owner : associated) expected[owner].push_back(id);
    }

    uint64_t active_count = 0u, articulation_sides = 0u;
    for (size_t id = 0u; id < rows.size(); ++id) {
        const bool is_active = (rows[id].flags & nk::nk_row_flags::kActive) != 0u;
        fixture::Require(visited[id] == is_active, "active rows missing from block schedule");
        if (!is_active) continue;
        ++active_count;
        articulation_sides += (rows[id].a.kind == nk::kNkSideArtic) + (rows[id].b.kind == nk::kNkSideArtic);
    }
    if (p.material_cells_per_env != 0u) {
        for (uint32_t env = 0u; env < p.env_count; ++env) {
            std::set<uint32_t> expected_heads;
            uint32_t ordinal = 0u;
            for (uint32_t cell : occupied_cells[env]) {
                if (expected_cells[env].count(cell) != 0u) {
                    const uint64_t head = nuka::CheckedAdd(nuka::CheckedProduct({env, p.rows_per_env}),
                        nuka::CheckedAdd(material_begin, nuka::CheckedProduct({ordinal, nk::kMpmStressRowsPerCell})));
                    fixture::Require(head < rows.size(), "expected material head exceeds reserved row storage");
                    expected_heads.insert(static_cast<uint32_t>(head));
                }
                ++ordinal;
            }
            fixture::Require(material_heads[env].size() == expected_cells[env].size() &&
                             material_heads[env] == expected_heads,
                             "active material heads do not cover positive-volume occupied cells in the reserved tail");
        }
    }
    fixture::Require(expected_incidence_count == incidence.size(), "block CSR incidence total is incomplete or excessive");
    uint64_t hash = 14695981039346656037ull, owners_with_rows = 0u;
    hash = UpdateDigest(hash, &owners, sizeof(owners));
    for (uint32_t owner = 0u; owner < owners; ++owner) {
        auto& required = expected[owner];
        std::sort(required.begin(), required.end());
        std::vector<uint32_t> actual(incidence.begin() + offsets[owner], incidence.begin() + offsets[owner + 1u]);
        for (uint32_t id : actual)
            fixture::Require(id < rows.size() && scheduled[id] &&
                             std::binary_search(required.begin(), required.end(), id),
                             "block CSR row is inactive or belongs to another owner");
        std::sort(actual.begin(), actual.end());
        fixture::Require(std::adjacent_find(actual.begin(), actual.end()) == actual.end(),
                         "duplicate owner-row incidence in block CSR");
        fixture::Require(actual == required, "expected owner-row incidence is missing from block CSR");
        const uint32_t count = static_cast<uint32_t>(actual.size());
        ++rows_per_block[count];
        owners_with_rows += count != 0u;
        hash = UpdateDigest(hash, &owner, sizeof(owner));
        hash = UpdateDigest(hash, &count, sizeof(count));
        if (count != 0u) hash = UpdateDigest(hash, actual.data(), actual.size() * sizeof(uint32_t));
    }
    Json result = Json::Object();
    result.Set("step", Json::Int(step));
    result.Set("schedule_type", Json::Str("block_csr"));
    result.Set("active_rows", Json::Int(active_count));
    result.Set("scheduled_contact_blocks_and_rows", Json::Int(scheduled_count));
    Json cached_penalty = penalty_range.Report(), penalty_kinds = Json::Array();
    const auto endpoint_kind_name = [](uint32_t kind) -> const char* {
        switch (kind) {
            case nk::kNkSideRigid: return "Rigid";
            case nk::kNkSideArtic: return "Articulation";
            case nk::kNkSideParticle: return "Particle";
            case nk::kNkSideStatic: return "Static";
            case nk::kNkSideGrid: return "Grid";
            case nk::kNkSidePointEndpoint: return "PointEndpoint";
            default: fixture::Require(false, "unknown cached penalty endpoint kind"); return "";
        }
    };
    for (const auto& entry : penalty_by_endpoint_kind) {
        Json range = entry.second.Report();
        range.Set("a_kind", Json::Str(endpoint_kind_name(entry.first.first)));
        range.Set("b_kind", Json::Str(endpoint_kind_name(entry.first.second)));
        penalty_kinds.PushBack(std::move(range));
    }
    cached_penalty.Set("by_endpoint_kind", std::move(penalty_kinds));
    cached_penalty.Set("scope", Json::Str("current last-substep schedule; cached rho for active heads only, not earlier-substep first failures"));
    cached_penalty.Set("range_scope", Json::Str("min/max over finite cached values; max_row_id uses the smallest global row ID on ties"));
    cached_penalty.Set("endpoint_kind_scope", Json::Str("unordered pairs of original NkRow side kinds; PointEndpoint is not expanded"));
    result.Set("cached_penalty", std::move(cached_penalty));
    result.Set("penalty_scale", Json::Float(p.penalty_scale));
    result.Set("penalty_definition", Json::Str("rho = penalty_scale / full-row inertial response over deduplicated owners, including cross terms between both sides sharing an owner; rho is a solver numerical parameter; physical compliance, friction mu and rhs are unchanged"));
    result.Set("material_blocks", Json::Int(material_blocks));
    result.Set("material_cells_per_env", Json::Int(p.material_cells_per_env));
    const bool grid_color_coverage = grid_color_heads_checked == scheduled_count;
    result.Set("grid_color_conflict_count", Json::Int(grid_color_conflict_count));
    result.Set("grid_color_coverage", Json::Bool(grid_color_coverage));
    result.Set("grid_color_heads_checked", Json::Int(grid_color_heads_checked));
    result.Set("grid_color_scope",
               Json::Str("all active heads; deduplicated Grid owners from actual row endpoints; lattice-0 modulo 2 and lattice-1 modulo 3"));
    result.Set("valid", Json::Bool(grid_color_coverage && grid_color_conflict_count == 0u));
    Json grid_dimensions = Json::Array(), material_counts = Json::Array(), expected_material_counts = Json::Array();
    for (uint32_t dimension : p.grid_dims) grid_dimensions.PushBack(Json::Int(dimension));
    for (uint32_t env = 0u; env < p.env_count; ++env) {
        material_counts.PushBack(Json::Int(material_heads[env].size()));
        if (p.material_cells_per_env != 0u)
            expected_material_counts.PushBack(Json::Int(expected_cells[env].size()));
    }
    result.Set("grid_dims", std::move(grid_dimensions));
    result.Set("material_blocks_by_env", std::move(material_counts));
    result.Set("expected_occupied_stress_cells_by_env",
               p.material_cells_per_env != 0u ? std::move(expected_material_counts) : Json::Null());
    result.Set("material_coverage_scope", Json::Str(p.material_cells_per_env != 0u ?
        "last substep Predict Data MpmGridCellKey/MpmGridPartIdx and positive ParticleVol0; per-env stress cells and Exchange reserved heads" :
        "unmeasured: implicit material stress is disabled"));
    result.Set("material_key_boundary_rule",
               Json::Str("Predict clamps each half-cell coordinate to [0, 2*grid_dim-1]; grid escape remains an environment failure"));
    result.Set("material_head_order",
               Json::Str("per-env ordinal in all occupied stress cells sorted by half-cell key / kMpmHalfCellsPerLatticeNode"));
    result.Set("solver_failure_owners", std::move(solver_failure_owners));
    result.Set("solver_failure_equations", std::move(solver_failure_equations));
    constexpr std::array<const char*, nk::kBlockSolveFailureEquationColumnCount> equation_names{
        "mass", "force_x", "force_y", "force_z", "hessian_xx", "hessian_yy", "hessian_zz",
        "hessian_xy", "hessian_xz", "hessian_yz", "snapshot_x", "snapshot_y", "snapshot_z",
        "free_x", "free_y", "free_z", "iteration", "dominant_row", "dominant_penalty",
        "dominant_response_a", "dominant_response_b", "dominant_residual_normal",
        "dominant_dual_normal", "dominant_curvature_trace"};
    static_assert(equation_names.back() != nullptr, "failure equation columns require declared names");
    Json equation_columns = Json::Array();
    for (const char* column : equation_names) equation_columns.PushBack(Json::Str(column));
    result.Set("solver_failure_equations_columns", std::move(equation_columns));
    result.Set("solver_failure_equations_column_count", Json::Int(nk::kBlockSolveFailureEquationColumnCount));
    result.Set("solver_failure_equations_record_layout", Json::Str("owner followed by the declared equation columns"));
    result.Set("solver_failure_equations_encoding",
               Json::Str("binary64 max_digits10 decimal strings; nonfinite values are NaN, Infinity or -Infinity strings"));
    Json failure_columns = Json::Array();
    for (const char* column : {"owner", "reason", "row", "substep"}) failure_columns.PushBack(Json::Str(column));
    result.Set("solver_failure_owners_columns", std::move(failure_columns));
    result.Set("solver_failure_scope", Json::Str("first failure per owner within the policy step; cleared at its first solve; zero-based substep"));
    result.Set("owners", Json::Int(owners));
    result.Set("owners_with_rows", Json::Int(owners_with_rows));
    result.Set("block_incidence_count", Json::Int(incidence.size()));
    result.Set("rows_per_block", Histogram(rows_per_block));
    result.Set("blocks_per_row", Histogram(blocks_per_row));
    result.Set("articulation_sides", Json::Int(articulation_sides));
    result.Set("canonical_schedule_fnv1a64", Json::Str(FormatDigest(hash)));
    return result;
}

Json SolverWorkload(nk::World& world, const std::vector<nk::NkRow>& rows, uint32_t step) {
    for (const auto& call : world.GetPipeline().Calls()) {
        if (call.op != phi::NkOp::BlockDescentSolve) continue;
        fixture::Require(call.params != nullptr, "block solve parameters are missing");
        return BlockWorkload(world, rows, step, *static_cast<const phi::BlockDescentSolveParams*>(call.params));
    }
    return IslandWorkload(world, rows, step);
}

enum class ContactSystem : uint32_t { Rigid, Articulation, Xpbd, Mpm, Pbf, Fixed, Count };
constexpr uint32_t kContactSystems = static_cast<uint32_t>(ContactSystem::Count);
constexpr std::array<const char*, kContactSystems> kContactSystemNames{
    "rigid", "articulation", "xpbd", "mpm", "pbf", "fixed"};

struct CouplingAcceptance {
    struct Pair {
        double impulse = 0.0;
        double timed_impulse = 0.0;
        uint64_t timed_rows = 0u;
        uint32_t first_step = 0u, last_step = 0u;
    };
    std::vector<std::array<Pair, kContactSystems * kContactSystems>> pairs;
    bool four_systems;

    CouplingAcceptance(uint32_t envs, bool four) : pairs(envs), four_systems(four) {}

    static uint32_t PairIndex(ContactSystem a, ContactSystem b) {
        const auto first = static_cast<uint32_t>(a), second = static_cast<uint32_t>(b);
        return std::min(first, second) * kContactSystems + std::max(first, second);
    }

    bool Required(uint32_t a, uint32_t b) const {
        return four_systems ? a < 4u && b < 4u && a < b :
            a == static_cast<uint32_t>(ContactSystem::Articulation) &&
            (b == static_cast<uint32_t>(ContactSystem::Xpbd) || b == static_cast<uint32_t>(ContactSystem::Pbf));
    }

    void Observe(nk::World& world, const std::vector<nk::NkRow>& rows,
                 const std::vector<float>& impulses, uint32_t step, bool timed) {
        const auto& model = world.GetModel();
        const auto& caps = model.capacities;
        std::vector<nk::PointEndpointRange> ranges(caps.ElementCount(nk::FieldId::PointEndpointRanges));
        std::vector<nk::PointEndpointTerm> terms(caps.ElementCount(nk::FieldId::PointEndpointTerms));
        if (!ranges.empty()) {
            fixture::Require(world.GetData().DownloadField(nk::FieldId::PointEndpointRanges,
                ranges.data(), ranges.size() * sizeof(ranges[0])), "endpoint range download failed");
            fixture::Require(world.GetData().DownloadField(nk::FieldId::PointEndpointTerms,
                terms.data(), terms.size() * sizeof(terms[0])), "endpoint term download failed");
        }
        const auto classify = [&](uint32_t kind, uint32_t index, uint32_t env) {
            if (kind == nk::kNkSideStatic) return ContactSystem::Fixed;
            if (kind == nk::kNkSideRigid) {
                fixture::Require(caps.bodies_per_env && index / caps.bodies_per_env == env,
                    "rigid endpoint crosses environments");
                return model.body_init.at(index % caps.bodies_per_env).inv_mass > 0.0f ?
                    ContactSystem::Rigid : ContactSystem::Fixed;
            }
            if (kind == nk::kNkSideArtic) {
                fixture::Require(caps.articulations_per_env && index / caps.articulations_per_env == env,
                    "articulation endpoint crosses environments");
                return ContactSystem::Articulation;
            }
            if (kind == nk::kNkSideGrid) {
                fixture::Require(caps.mpm_grid_nodes_per_env && index / caps.mpm_grid_nodes_per_env == env,
                    "grid endpoint crosses environments");
                return ContactSystem::Mpm;
            }
            fixture::Require(kind == nk::kNkSideParticle && caps.particles_per_env &&
                index / caps.particles_per_env == env, "invalid particle endpoint");
            const uint32_t local = index % caps.particles_per_env;
            if (local < model.MpmParticlesPerEnv()) return ContactSystem::Mpm;
            if (model.particles.mode == nk::Model::ParticleMode::MpmXpbd ||
                model.particles.mode == nk::Model::ParticleMode::Xpbd ||
                (model.particles.mode == nk::Model::ParticleMode::SoftFluid && local < model.particles.n_soft_particles))
                return ContactSystem::Xpbd;
            fixture::Require(model.particles.mode == nk::Model::ParticleMode::Pbf ||
                model.particles.mode == nk::Model::ParticleMode::SoftFluid, "unknown particle system in coupling report");
            return ContactSystem::Pbf;
        };
        const auto category = [&](const nk::NkRowSide& side, uint32_t env) {
            if (side.kind != nk::kNkSidePointEndpoint) return classify(side.kind, side.index, env);
            fixture::Require(caps.point_endpoints_per_env && side.index / caps.point_endpoints_per_env == env &&
                side.index < ranges.size(), "invalid interpolated endpoint");
            const auto range = ranges[side.index];
            fixture::Require(range.count && range.first <= terms.size() && range.count <= terms.size() - range.first,
                "invalid interpolated endpoint range");
            const auto system = classify(terms[range.first].kind, terms[range.first].index, env);
            for (uint32_t i = 1u; i < range.count; ++i)
                fixture::Require(classify(terms[range.first + i].kind, terms[range.first + i].index, env) == system,
                    "interpolated endpoint spans different physical systems");
            return system;
        };
        for (size_t index = 0u; index < rows.size(); ++index) {
            const auto& row = rows[index];
            if (!(row.flags & nk::nk_row_flags::kActive) || !(row.flags & nk::nk_row_flags::kContactNormal)) continue;
            fixture::Require(std::isfinite(impulses[index]) && impulses[index] >= 0.0f,
                "invalid unilateral contact impulse");
            const uint32_t env = static_cast<uint32_t>(index / caps.max_rows_per_env);
            const auto a = category(row.a, env), b = category(row.b, env);
            auto& sample = pairs.at(env)[PairIndex(a, b)];
            if (timed) {
                ++sample.timed_rows;
                sample.timed_impulse += impulses[index];
            }
            sample.impulse += impulses[index];
            if (impulses[index] > 0.0f) {
                if (!sample.first_step) sample.first_step = step;
                sample.last_step = step;
            }
        }
    }

    bool Valid() const {
        for (const auto& env : pairs)
            for (uint32_t a = 0u; a < kContactSystems; ++a)
                for (uint32_t b = a + 1u; b < kContactSystems; ++b)
                    if (Required(a, b)) {
                        const auto& sample = env[a * kContactSystems + b];
                        if (!(sample.impulse > 0.0) || sample.timed_rows == 0u ||
                            !(sample.timed_impulse > 0.0)) return false;
                    }
        return true;
    }

    Json Report() const {
        Json result = Json::Object(), observations = Json::Array();
        bool timed_coverage = true;
        for (uint32_t a = 0u; a < kContactSystems; ++a) {
            for (uint32_t b = a; b < kContactSystems; ++b) {
                Json pair = Json::Object(), impulses = Json::Array(), rows = Json::Array();
                Json timed_impulses = Json::Array();
                Json first = Json::Array(), last = Json::Array();
                bool observed = false;
                for (const auto& env : pairs) {
                    const auto& sample = env[a * kContactSystems + b];
                    observed |= sample.first_step > 0u || sample.timed_rows > 0u;
                    impulses.PushBack(Json::Float(sample.impulse));
                    timed_impulses.PushBack(Json::Float(sample.timed_impulse));
                    rows.PushBack(Json::Int(sample.timed_rows));
                    first.PushBack(sample.first_step ? Json::Int(sample.first_step) : Json::Null());
                    last.PushBack(sample.last_step ? Json::Int(sample.last_step) : Json::Null());
                    if (Required(a, b))
                        timed_coverage &= sample.timed_rows > 0u && sample.timed_impulse > 0.0;
                }
                if (!observed && !Required(a, b)) continue;
                pair.Set("a", Json::Str(kContactSystemNames[a]));
                pair.Set("b", Json::Str(kContactSystemNames[b]));
                pair.Set("required", Json::Bool(Required(a, b)));
                pair.Set("sampled_normal_impulse_by_env_Ns", std::move(impulses));
                pair.Set("timed_sampled_normal_impulse_by_env_Ns", std::move(timed_impulses));
                pair.Set("timed_normal_rows_by_env", std::move(rows));
                pair.Set("first_impulse_step_by_env", std::move(first));
                pair.Set("last_impulse_step_by_env", std::move(last));
                observations.PushBack(std::move(pair));
            }
        }
        result.Set("scope", Json::Str("final substep of every replay step, including warmup; sampled impulses are not time-integrated totals"));
        result.Set("pairs", std::move(observations));
        result.Set("timed_samples_cover_required_pairs", Json::Bool(timed_coverage));
        result.Set("valid", Json::Bool(Valid()));
        return result;
    }
};

constexpr nuka::math::Vec3 kClothAlbedo{0.06f, 0.48f, 0.34f}, kMpmAlbedo{0.75f, 0.22f, 0.06f};

class PipelineSensor {
public:
    PipelineSensor(nk::World& physics, const fixture::SceneVisuals& visuals,
                    const Options& options) : options_(options) {
        const auto world = nuka::render::BuildRenderWorld(visuals.scene.Ecs(), visuals.scene_map);
        nuka::render::SensorSceneDesc desc;
        desc.scene = nuka::render::RenderWorldToTwoLevelScene(world);
        fixture::Require(!world.instances.empty(), "no sensor geometry");
        for (const auto& instance : world.instances) {
            phi::InstanceScatterRow row;
            row.kind = static_cast<uint32_t>(instance.pose_source.kind);
            row.row = instance.pose_source.row;
            const auto& pose = instance.cached_visual_local;
            const float values[] = {pose.position.x, pose.position.y, pose.position.z,
                                    pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z};
            std::copy(std::begin(values), std::end(values), row.cached_visual_local);
            desc.rows.push_back(row);
            desc.blas_id.push_back(instance.mesh_id);
            desc.material_id.push_back(instance.render_material_id < world.materials.size()
                ? instance.render_material_id : static_cast<uint32_t>(desc.scene.materials.size() - 1u));
        }
        desc.particles = {static_cast<const nuka::math::Vec3*>(physics.FieldPtr(nk::FieldId::ParticlePos)),
            physics.GetModel().capacities.particles_per_env, physics.EnvCount()};
        const auto& particles = physics.GetModel().particles;
        const auto append_material = [&](nuka::math::Vec3 color) {
            nuka::rt::Material material;
            material.albedo = color;
            const auto id = static_cast<uint32_t>(desc.scene.materials.size());
            desc.scene.materials.push_back(material);
            return id;
        };
        const auto cloth_material = append_material(kClothAlbedo), mpm_material = append_material(kMpmAlbedo);
        for (const auto& info : particles.surface_info) {
            nuka::rt::ParticleSurfaceBinding surface;
            const auto first = particles.surface_triangles.begin() + 3u * info.triangle_offset;
            surface.triangle_particles.assign(first, first + 3u * info.triangle_count);
            for (auto& vertex : surface.triangle_particles) vertex += info.vertex_offset;
            nuka::render::AppendParticleSurface(desc, std::move(surface), cloth_material);
        }
        for (const auto& info : visuals.material_surfaces) {
            nuka::rt::ParticleSurfaceBinding surface;
            surface.triangle_particles = info.triangles;
            surface.normal_offset = info.normal_offset;
            surface.smooth_iters = info.smooth_iters;
            surface.smooth_lambda = info.smooth_lambda;
            nuka::render::AppendParticleSurface(desc, std::move(surface), mpm_material);
        }
        for (uint32_t camera = 0u; camera < options_.render_sensors; ++camera) {
            nuka::scene::SensorDesc sensor;
            sensor.type = nuka::scene::SensorType::Camera;
            sensor.mount = nuka::scene::MountFrame::Base;
            const auto yaw = nuka::math::Quat::FromAxisAngle({0, 0, 1},
                6.28318530718f * static_cast<float>(camera) / static_cast<float>(options_.render_sensors));
            sensor.local_offset.position = yaw.Rotate({0.0f, -1.25f, 0.65f});
            sensor.local_offset.rotation = yaw * nuka::math::Quat::FromAxisAngle({1, 0, 0}, std::atan2(1.25f, 0.65f));
            sensor.cam.width = static_cast<uint16_t>(options_.render_width);
            sensor.cam.height = static_cast<uint16_t>(options_.render_height);
            sensor.cam.vfov_degrees = 50.0f;
            desc.sensors.push_back(sensor);
        }
        backend_ = nuka::render::CreateCudaSensorBackend();
        fixture::Require(backend_ != nullptr, "sensor backend unavailable");
        handle_ = backend_->BuildSensorScene(desc);
        fixture::Require(handle_ != nullptr, "sensor scene build failed");
        nuka::rt::SensorFidelityConfig fidelity;
        fidelity.spp = options_.render_samples;
        fidelity.shadow_samples = options_.render_shadows;
        fidelity.ao_samples = options_.render_ao;
        fidelity.ao_enabled = options_.render_ao != 0u;
        fidelity.seed = options_.seed;
        try {
            backend_->SetSensorFidelity(handle_, fidelity);
            if (options_.imaging_models) for (uint32_t camera = 0u; camera < options_.render_sensors; ++camera) {
                nuka::sensor::CameraResponse color;
                color.enabled = 1u;
                color.read_noise_electrons = 3.0f;
                color.row_noise_electrons = 1.0f;
                color.pixel_gain_stddev = 0.01f;
                color.dark_current = 25.0f;
                color.dark_doubling_temperature = 6.0f;
                color.temperature = 35.0f;
                color.seed = options_.seed;
                backend_->SetCameraResponse(handle_, camera, color);
                nuka::sensor::RangeResponse depth;
                depth.enabled = 1u;
                depth.distance_stddev = 0.0005f;
                depth.quadratic_stddev = 0.0001f;
                depth.quantization = 0.0005f;
                depth.return_photons = 1500.0f;
                depth.precision = 0.02f;
                depth.dropout_probability = 0.001f;
                depth.seed = options_.seed;
                backend_->SetRangeResponse(handle_, false, camera, depth);
            }
        } catch (...) {
            backend_->FreeSensorScene(handle_);
            handle_ = nullptr;
            throw;
        }
    }
    ~PipelineSensor() { if (handle_) backend_->FreeSensorScene(handle_); }
    void Render(nk::World& world) {
        phi::ScatterFkSource fk;
        fk.link_pose = world.FieldPtr(nk::FieldId::LinkPose);
        fk.body_pose = world.FieldPtr(nk::FieldId::BodyPose);
        fk.base_pose = world.FieldPtr(nk::FieldId::BasePose);
        fk.links_per_env = world.GetModel().capacities.links_per_env;
        fk.bodies_per_env = world.GetModel().capacities.bodies_per_env;
        fk.world_backend = world.Backend();
        std::vector<double> times(options_.envs);
        for (uint32_t env = 0u; env < options_.envs; ++env) times[env] = world.StateSensors().SimulationTime(env);
        backend_->SetSensorSampleTimes(handle_, times);
        backend_->RenderSensors(handle_, fk, options_.envs, options_.render_width, options_.render_height);
    }
    nuka::rt::SensorStateSnapshot Capture() const { return backend_->CaptureSensorState(handle_); }
    void Restore(const nuka::rt::SensorStateSnapshot& snapshot) { backend_->RestoreSensorState(handle_, snapshot); }
    std::vector<uint8_t> Output() const {
        const size_t pixels = size_t{options_.envs} * options_.render_sensors * options_.render_width * options_.render_height;
        std::vector<uint8_t> output;
        auto append = [&](const void* source, size_t bytes) {
            fixture::Require(source != nullptr, "missing sensor output");
            const size_t offset = output.size();
            output.resize(offset + bytes);
            CheckCuda(cudaMemcpy(output.data() + offset, source, bytes, cudaMemcpyDeviceToHost));
        };
        append(backend_->SensorColorDevice(handle_), pixels * 3u * sizeof(float));
        append(backend_->SensorDepthDevice(handle_), pixels * sizeof(float));
        append(backend_->SensorNormalDevice(handle_), pixels * 3u * sizeof(float));
        append(backend_->SensorAlbedoDevice(handle_), pixels * 3u * sizeof(float));
        append(backend_->SensorPrimDevice(handle_), pixels * sizeof(uint32_t));
        return output;
    }
private:
    const Options& options_;
    std::unique_ptr<nuka::render::SensorBackendI> backend_;
    nuka::render::SensorSceneHandle* handle_ = nullptr;
};

Json RenderMeasurements(nk::World& world, const fixture::SceneVisuals& visuals,
                        const Options& options, const std::vector<uint8_t>& expected_state) {
    struct ActiveBackendScope {
        phi::Backend* previous = phi::ActiveBackend();
        ~ActiveBackendScope() { phi::SetActiveBackend(previous); }
    } active_scope;
    phi::SetActiveBackend(world.Backend());
    const auto stream = phi::CudaBackendMainStream(reinterpret_cast<phi::CudaBackend*>(world.Backend()));
    size_t free_before = 0u, total_bytes = 0u;
    CheckCuda(cudaMemGetInfo(&free_before, &total_bytes));
    const auto create_start = Clock::now();
    PipelineSensor renderer(world, visuals, options);
    const double creation_ms = Milliseconds(create_start);
    fixture::Require(world.Reset() == phi::Status::Ok, "render replay reset failed");
    for (uint32_t i = 0u; i < options.warmup; ++i) Step(world, options);
    CheckCuda(cudaStreamSynchronize(stream));
    for (uint32_t i = 0u; i < options.render_warmup; ++i) renderer.Render(world);
    const auto measure = [&](bool coupled) {
        Events events(size_t{options.steps} * 3u);
        std::vector<double> physics, render, completion, wall;
        CheckCuda(cudaStreamSynchronize(stream));
        const auto batch_start = Clock::now();
        for (uint32_t i = 0u; i < options.steps; ++i) {
            const auto start = Clock::now();
            CheckCuda(cudaEventRecord(events.events[size_t{i} * 3u], stream));
            if (coupled) Step(world, options);
            CheckCuda(cudaEventRecord(events.events[size_t{i} * 3u + 1u], stream));
            renderer.Render(world);
            CheckCuda(cudaEventRecord(events.events[size_t{i} * 3u + 2u], stream));
            wall.push_back(Milliseconds(start) * 1000.0);
        }
        CheckCuda(cudaEventSynchronize(events.events.back()));
        const double wall_ms = Milliseconds(batch_start);
        for (uint32_t i = 0u; i < options.steps; ++i) {
            float p = 0.0f, r = 0.0f, c = 0.0f;
            CheckCuda(cudaEventElapsedTime(&p, events.events[size_t{i} * 3u], events.events[size_t{i} * 3u + 1u]));
            CheckCuda(cudaEventElapsedTime(&r, events.events[size_t{i} * 3u + 1u], events.events[size_t{i} * 3u + 2u]));
            CheckCuda(cudaEventElapsedTime(&c, events.events[size_t{i} * 3u], events.events[size_t{i} * 3u + 2u]));
            physics.push_back(p * 1000.0); render.push_back(r * 1000.0); completion.push_back(c * 1000.0);
        }
        Json result = Json::Object();
        if (coupled) result.Set("physics_gpu", Distribution(std::move(physics)));
        result.Set("render_gpu", Distribution(std::move(render)));
        result.Set("completion_gpu", Distribution(std::move(completion)));
        result.Set("host_submission", Distribution(std::move(wall)));
        result.Set("synchronized_wall_ms", Json::Float(wall_ms));
        result.Set("host_clock", Json::Str(Clock::Name()));
        result.Set("synchronized_wall_mean_us", Json::Float(wall_ms * 1000.0 / options.steps));
        return result;
    };
    Json result = Json::Object(), config = Json::Object();
    result.Set("creation_ms", Json::Float(creation_ms));
    result.Set("physics_to_sensor", measure(true));
    const bool coupled_state_equal = State(world) == expected_state;
    const auto coupled_output = renderer.Output();
    const auto replay_start = renderer.Capture();
    for (uint32_t i = 0u; i < options.render_warmup; ++i) renderer.Render(world);
    result.Set("render_only", measure(false));
    const auto output = renderer.Output();
    const bool repeated_output_equal = output == coupled_output;
    const bool render_state_equal = State(world) == expected_state;
    renderer.Restore(replay_start);
    for (uint32_t i = 0u; i < options.render_warmup + options.steps; ++i) renderer.Render(world);
    const bool replay_output_equal = renderer.Output() == output;
    const size_t pixels = size_t{options.envs} * options.render_sensors * options.render_width * options.render_height;
    const size_t geometry_offset = pixels * 4u * sizeof(float);
    const bool geometry_output_equal = std::equal(output.begin() + geometry_offset, output.end(),
                                                   coupled_output.begin() + geometry_offset);
    uint64_t hits = 0u, cloth_pixels = 0u, mpm_pixels = 0u;
    bool finite = true;
    for (size_t i = 0u; i < pixels * 10u; ++i) {
        float value;
        std::memcpy(&value, output.data() + i * sizeof(float), sizeof(float));
        if (i >= pixels * 3u && i < pixels * 4u) finite &= std::isfinite(value) || value == std::numeric_limits<float>::infinity();
        else finite &= std::isfinite(value);
    }
    for (size_t i = 0u; i < pixels; ++i) {
        uint32_t prim;
        std::memcpy(&prim, output.data() + (pixels * 10u + i) * sizeof(uint32_t), sizeof(uint32_t));
        if (prim != ~0u) ++hits;
        nuka::math::Vec3 albedo;
        std::memcpy(&albedo, output.data() + (pixels * 7u + i * 3u) * sizeof(float), sizeof(albedo));
        if ((albedo - kClothAlbedo).LengthSq() < 1.0e-10f) ++cloth_pixels;
        if ((albedo - kMpmAlbedo).LengthSq() < 1.0e-10f) ++mpm_pixels;
    }
    if (!options.render_output.empty()) {
        std::ofstream file(options.render_output, std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), output.size());
        fixture::Require(file.good(), "cannot write sensor output");
    }
    size_t free_after = 0u;
    CheckCuda(cudaMemGetInfo(&free_after, &total_bytes));
    result.Set("resident_device_delta_bytes", Json::Int(static_cast<int64_t>(free_before) - static_cast<int64_t>(free_after)));
    result.Set("memory_scope", Json::Str("cudaMemGetInfo resident delta; allocation peak requires separate profiling"));
    config.Set("cameras_per_env", Json::Int(options.render_sensors));
    config.Set("width", Json::Int(options.render_width));
    config.Set("height", Json::Int(options.render_height));
    config.Set("samples", Json::Int(options.render_samples));
    config.Set("shadow_samples", Json::Int(options.render_shadows));
    config.Set("ao_samples", Json::Int(options.render_ao));
    config.Set("seed", Json::Int(options.seed));
    config.Set("warmup_frames", Json::Int(options.render_warmup));
    config.Set("imaging_models", Json::Int(options.imaging_models));
    result.Set("config", std::move(config));
    result.Set("geometry_scope", Json::Str("same cook's SceneIR and SceneMap; particle collision surfaces and an MPM lattice boundary skin follow live particles on the public batched sensor path"));
    result.Set("boundary", Json::Str("physics-to-sensor uses live per-step poses and particle positions; render-only repeats its final world; both complete all AOVs on device; output download is untimed"));
    result.Set("output_layout", Json::Str("env-camera-major color f32x3, depth f32, normal f32x3, albedo f32x3, prim u32"));
    result.Set("output_fnv1a64", Json::Str(Digest(output)));
    result.Set("hit_pixels", Json::Int(hits));
    result.Set("xpbd_visible_pixels", Json::Int(cloth_pixels));
    result.Set("mpm_visible_pixels", Json::Int(mpm_pixels));
    const bool material_visible = options.scene != "robot-rigid-mpm-cloth" || (cloth_pixels > 0u && mpm_pixels > 0u);
    result.Set("material_visible", Json::Bool(material_visible));
    result.Set("finite", Json::Bool(finite));
    result.Set("coupled_state_equal", Json::Bool(coupled_state_equal));
    result.Set("render_state_equal", Json::Bool(render_state_equal));
    result.Set("repeated_output_equal", Json::Bool(repeated_output_equal));
    result.Set("replay_output_equal", Json::Bool(replay_output_equal));
    result.Set("geometry_output_equal", Json::Bool(geometry_output_equal));
    const bool expected_frame_change = options.imaging_models ? !repeated_output_equal : repeated_output_equal;
    result.Set("valid", Json::Bool(finite && hits > 0u && material_visible && coupled_state_equal && render_state_equal &&
        replay_output_equal && geometry_output_equal && expected_frame_change));
    return result;
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
    config.substeps = options.substeps;
    config.vel_iters = static_cast<uint16_t>(options.velocity_iterations);
    const bool four_systems = options.scene == "robot-rigid-mpm-cloth";
    const auto scene_path = options.scene == "robot-cloth-fluid" || four_systems
        ? std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes/go2_stand.usda"
        : std::filesystem::path(options.scene);
    const auto prepare_start = Clock::now();
    const auto fixture_config = fixture::Cfg();
    const auto prepared = fixture::Prepare(scene_path, device, owner.backend, fixture_config);
    const double preparation_ms = Milliseconds(prepare_start);
    const auto cook_start = Clock::now();
    fixture::SceneVisuals visuals;
    auto model = four_systems ? fixture::CookMpmPrepared(prepared, options.envs, &visuals, options.mpm_implicit_stress != 0u)
                             : fixture::CookPrepared(prepared, options.envs, true, options.cloth_nx, &visuals);
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
    AttachStateSensors(world, options);
    AttachTactileSensors(world, options, prepared);
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
    const auto timed_sensors = StateSensorBytes(world);

    const auto quality_start = Clock::now();
    fixture::Require(world.Reset() == phi::Status::Ok, "quality replay reset failed");
    const bool reset_equal = State(world, false) == initial;
    std::vector<uint32_t> status(options.envs);
    uint32_t status_union = 0u;
    std::vector<uint32_t> status_first_failure_step_by_env(options.envs, 0u);
    std::vector<uint32_t> status_first_failure_flags_by_env(options.envs, 0u);
    const auto& caps = world.GetModel().capacities;
    std::vector<nk::NkRow> rows_host(size_t{caps.max_rows_per_env} * options.envs);
    std::vector<float> impulses(rows_host.size());
    CouplingAcceptance coupling_acceptance(options.envs, four_systems);
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
    Json workload_samples = Json::Array(), xpbd_samples = Json::Array();
    bool solver_workload_valid = true;
    XpbdAcceptance xpbd_acceptance;
    for (uint32_t i = 0; i < options.warmup + options.steps; ++i) {
        Step(world, options);
        fixture::Require(world.GetData().DownloadField(nk::FieldId::EnvStatus, status.data(),
                         status.size() * sizeof(uint32_t)), "status download failed");
        bool first_failure_step = false;
        for (uint32_t env = 0u; env < status.size(); ++env) {
            const uint32_t flags = status[env];
            status_union |= flags;
            if (flags != 0u && status_first_failure_step_by_env[env] == 0u) {
                status_first_failure_step_by_env[env] = i + 1u;
                status_first_failure_flags_by_env[env] = flags;
                first_failure_step = true;
            }
        }
        fixture::Require(world.GetData().DownloadField(nk::FieldId::LinkContactWrench,
                         wrench.data(), wrench_bytes), "link wrench download failed");
        for (float value : wrench) wrench_finite &= std::isfinite(value);
        wrench_hash = UpdateDigest(wrench_hash, wrench.data(), wrench_bytes);
        if (wrench_output.is_open()) {
            wrench_output.write(reinterpret_cast<const char*>(wrench.data()), wrench_bytes);
            fixture::Require(wrench_output.good(), "cannot write wrench output");
        }
        auto xpbd_quality = XpbdQuality(world, i + 1u);
        xpbd_acceptance.Observe(xpbd_quality);
        fixture::Require(world.GetData().DownloadField(nk::FieldId::Urows, rows_host.data(),
                         rows_host.size() * sizeof(nk::NkRow)), "row download failed");
        fixture::Require(world.GetData().DownloadField(nk::FieldId::Lambda, impulses.data(),
                         impulses.size() * sizeof(float)), "impulse download failed");
        coupling_acceptance.Observe(world, rows_host, impulses, i + 1u, i >= options.warmup);
        const bool periodic_sample =
            i >= options.warmup && (i % 25u == 0u || i + 1u == options.warmup + options.steps);
        if (first_failure_step || periodic_sample) {
            auto sample = SolverWorkload(world, rows_host, i + 1u);
            if (sample.At("schedule_type").AsString() == "block_csr") {
                fixture::Require(sample.Has("valid"), "block workload validity coverage is missing");
                solver_workload_valid &= sample.At("valid").AsBool();
            }
            workload_samples.PushBack(std::move(sample));
        }
        if (periodic_sample) xpbd_samples.PushBack(std::move(xpbd_quality));
    }
    const auto replay_state = State(world);
    const auto replay_sensors = StateSensorBytes(world);
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
    result.Set("schema_version", Json::Int(4));
    result.Set("scene", Json::Str(options.scene));
    Json configuration = Json::Object();
    configuration.Set("envs", Json::Int(options.envs));
    configuration.Set("dt", Json::Float(options.dt));
    configuration.Set("substeps", Json::Int(options.substeps));
    configuration.Set("mpm_substeps", Json::Int(world.GetModel().particles.mpm_substeps));
    configuration.Set("mpm_particles_per_env", Json::Int(world.GetModel().MpmParticlesPerEnv()));
    Json mpm_stress = Json::Object();
    mpm_stress.Set("requested", Json::Bool(options.mpm_implicit_stress != 0u));
    mpm_stress.Set("actual", Json::Bool(caps.mpm_stress_cells_per_env != 0u));
    mpm_stress.Set("capacity", Json::Int(caps.mpm_stress_cells_per_env));
    mpm_stress.Set("capacity_scope", Json::Str("reserved material stress cells per environment"));
    configuration.Set("mpm_implicit_stress", std::move(mpm_stress));
    configuration.Set("state_sensors", Json::Int(options.state_sensors));
    configuration.Set("tactile_grid", Json::Int(options.tactile_grid));
    configuration.Set("fixture_dt", Json::Float(fixture_config.dt));
    configuration.Set("fixture_substeps", Json::Int(fixture_config.substeps));
    configuration.Set("fixture_vel_iters", Json::Int(fixture_config.vel_iters));
    configuration.Set("fixture_velocity_iterations_override", Json::Bool(fixture_config.velocity_iterations_override));
    configuration.Set("fixture_actual_iterations_status", Json::Str("unmeasured"));
    configuration.Set("seed", Json::Int(options.seed));
    configuration.Set("seed_usage", Json::Str("deterministic held control; optional sensor error and delivery randomization"));
    configuration.Set("vel_iters", Json::Int(config.vel_iters));
    configuration.Set("scene_source_path", Json::Str(std::filesystem::absolute(scene_path).string()));
    configuration.Set("actual_substeps", Json::Int(nk::Pipeline::SubstepCount(world.GetModel(), config)));
    Json solver = Json::Object(), solver_environment = Json::Object(), solve_calls = Json::Array();
    solver.Set("requested_velocity_iterations", Json::Int(options.velocity_iterations));
    solver.Set("request_explicit", Json::Bool(options.velocity_iterations_explicit));
    solver.Set("configured_velocity_iterations", Json::Int(config.vel_iters));
    solver.Set("velocity_iterations_override", Json::Bool(config.velocity_iterations_override));
    solver.Set("actual_velocity_iterations", Json::Int(world.GetPipeline().VelocityIterations(config.vel_iters)));
    solver.Set("actual_budget_scope", Json::Str("built pipeline budget per interval; individual solve calls are listed separately"));
    solver.Set("executed_iterations_status", Json::Str("unmeasured"));
    bool block_descent_solver = false;
    uint32_t solve_call_count = 0u;
    for (const auto& call : world.GetPipeline().Calls()) {
        Json solve = Json::Object();
        if (call.op == phi::NkOp::BlockDescentSolve) {
            const auto& params = *static_cast<const phi::BlockDescentSolveParams*>(call.params);
            block_descent_solver = true;
            solve.Set("solver", Json::Str("block_descent"));
            solve.Set("iterations", Json::Int(params.iterations));
            solve.Set("substep", Json::Int(params.substep_index));
            solve.Set("dt", Json::Float(params.dt));
            solve.Set("spectral_radius", Json::Float(params.acceleration_spectral_radius));
            solve.Set("velocity_tolerance", Json::Null());
        } else if (call.op == phi::NkOp::SolveRowsBlockIsland) {
            const auto& params = *static_cast<const phi::SolveRowsBlockIslandParams*>(call.params);
            solve.Set("solver", Json::Str("row_island"));
            solve.Set("iterations", Json::Int(params.vel_iters));
            solve.Set("dt", Json::Float(params.dt));
            solve.Set("velocity_tolerance", std::isfinite(params.vel_tolerance) ? Json::Float(params.vel_tolerance) : Json::Null());
            solve.Set("verify_idle", Json::Bool(params.verify_idle != 0u));
            solve.Set("continue_impulses", Json::Bool(params.continue_impulses != 0u));
        } else continue;
        solve_calls.PushBack(std::move(solve));
        ++solve_call_count;
    }
    solver.Set("mode", Json::Str(solve_call_count == 0u ? "no_solve_calls" : block_descent_solver ? "block_descent" : "row_island"));
    solver.Set("solve_call_count", Json::Int(solve_call_count));
    solver.Set("budget_priority", Json::Str(solve_call_count == 0u ? "no scheduled solve calls" :
        !block_descent_solver ? "configured row budget" :
        config.velocity_iterations_override ? "explicit world override" :
        std::getenv("NUKA_BLOCK_DESCENT_ITERATIONS") != nullptr ? "solver environment" : "solver default"));
    solver.Set("solve_calls", std::move(solve_calls));
    for (const char* key : {"NUKA_BLOCK_DESCENT", "NUKA_BLOCK_DESCENT_ITERATIONS",
                            "NUKA_BLOCK_DESCENT_SPECTRAL_RADIUS", "NUKA_SOLVER_VEL_TOLERANCE"}) {
        const char* value = std::getenv(key);
        solver_environment.Set(key, value != nullptr ? Json::Str(value) : Json::Null());
    }
    solver.Set("environment", std::move(solver_environment));
    configuration.Set("solver", std::move(solver));
    configuration.Set("cloth_iters", Json::Int(fixture::kClothIters));
    configuration.Set("cloth_grid", Json::Int(options.cloth_nx));
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
    timing.Set("host_clock", Json::Str(Clock::Name()));
    timing.Set("cook_ms", Json::Float(cook_ms));
    timing.Set("create_upload_ms", Json::Float(creation_ms));
    timing.Set("capture_first_execute_ms", Json::Float(capture_ms));
    timing.Set("warmup_ms", Json::Float(warmup_ms));
    timing.Set("host_submission_ms", Json::Float(submission_ms));
    timing.Set("gpu_completion_ms", Json::Float(gpu_ms));
    timing.Set("synchronized_wall_ms", Json::Float(wall_ms));
    timing.Set("gpu_to_host_duration_ratio", Json::Float(gpu_ms / wall_ms));
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
    Json sensors = Json::Object();
    sensors.Set("count", Json::Int(world.StateSensors().Count()));
    sensors.Set("timed_replay_bit_equal", Json::Bool(timed_sensors == replay_sensors));
    sensors.Set("state_fnv1a64", Json::Str(Digest(replay_sensors)));
    sensors.Set("existing_state_fnv1a64", Json::Str(Digest(StateSensorBytes(world, false))));
    sensors.Set("tactile_count", Json::Int(options.tactile_grid ? 1u + options.tactile_grid * options.tactile_grid : 0u));
    sensors.Set("scope", Json::Str(options.state_sensors >= 2u ?
        "IMU, pose and velocity per base; encoder, contact wrench and F/T on each articulation's first scalar joint" :
        "IMU, pose and velocity per base; encoder on each articulation's first scalar joint"));
    sensors.Set("update_period", Json::Int(1));
    sensors.Set("noise_density", Json::Float(0.01));
    sensors.Set("initial_bias_stddev", Json::Float(0.02));
    sensors.Set("bias_random_walk", Json::Float(0.002));
    sensors.Set("correlated_bias_stddev", Json::Float(0.005));
    sensors.Set("correlation_time", Json::Float(0.2));
    sensors.Set("latency", Json::Float(double{options.dt} * 2.0));
    sensors.Set("latency_jitter", Json::Float(double{options.dt} * 0.5));
    sensors.Set("dropout_probability", Json::Float(0.1));
    result.Set("state_sensors", std::move(sensors));
    Json workload = Json::Object();
    workload.Set("scope", Json::Str("untimed quality replay; actual solver schedule and owner coverage checked"));
    workload.Set("samples", std::move(workload_samples));
    workload.Set("valid", Json::Bool(solver_workload_valid));
    workload.Set("xpbd", XpbdWorkload(world.GetModel()));
    auto vbd_workload = VbdWorkload(world.GetModel());
    vbd_workload.Set("integrator", Json::Null());
    for (const auto& call : world.GetPipeline().Calls()) {
        if (call.op != phi::NkOp::ClothPredict) continue;
        const auto& params = *static_cast<const phi::ClothStepParams*>(call.params);
        vbd_workload.Set("integrator", Json::Int(params.integrator));
        break;
    }
    workload.Set("vbd", std::move(vbd_workload));
    result.Set("workload", std::move(workload));
    Json quality = Json::Object();
    quality.Set("finite", Json::Bool(finite));
    quality.Set("link_wrench_finite", Json::Bool(wrench_finite));
    quality.Set("link_wrench_trace_fnv1a64", Json::Str(FormatDigest(wrench_hash)));
    quality.Set("link_wrench_bytes_per_step", Json::Int(wrench_bytes));
    quality.Set("state_layout", Json::Str("physical fields including MPM F/C/plastic history, contact cache, link wrench"));
    quality.Set("env_status_union", Json::Int(status_union));
    Json first_failure_steps = Json::Array(), first_failure_flags = Json::Array();
    for (uint32_t env = 0u; env < options.envs; ++env) {
        first_failure_steps.PushBack(Json::Int(status_first_failure_step_by_env[env]));
        first_failure_flags.PushBack(Json::Int(status_first_failure_flags_by_env[env]));
    }
    quality.Set("status_first_failure_step_by_env", std::move(first_failure_steps));
    quality.Set("status_first_failure_flags_by_env", std::move(first_failure_flags));
    quality.Set("status_first_failure_scope",
                Json::Str("untimed quality replay including warmup; 1-based steps, zero step/flags means no observed failure"));
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
    const bool coupling_valid = coupling_acceptance.Valid();
    quality.Set("coupling_acceptance", coupling_acceptance.Report());
    quality.Set("xpbd_samples", std::move(xpbd_samples));
    quality.Set("xpbd_acceptance", xpbd_acceptance.Report());
    result.Set("quality", std::move(quality));
    bool render_valid = true;
    if (options.render_sensors != 0u) {
        auto render = RenderMeasurements(world, visuals, options, timed_state);
        render_valid = render.At("valid").AsBool();
        result.Set("render", std::move(render));
    }
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
                       solver_workload_valid &&
                       coupling_valid && mismatched_envs.empty() && xpbd_acceptance.Valid() && render_valid &&
                       timed_sensors == replay_sensors;
    validity.Set("valid", Json::Bool(valid));
    validity.Set("valid_scope", Json::Str("lifecycle/state/render/deformation/coupling workload contracts"));
    validity.Set("passes_full_physics_acceptance", Json::Bool(false));
    validity.Set("physical_acceptance_status", Json::Str("unmeasured"));
    validity.Set("physical_acceptance_reason", Json::Str("full-system energy ledger, analytic checks, reference comparisons and independent CCD are not integrated into this program"));
    Json failures = Json::Array();
    if (status_union != 0u) failures.PushBack(Json::Str("nonzero environment failure flags"));
    if (!solver_workload_valid) failures.PushBack(Json::Str("grid block coloring conflicts or lacks owner coverage"));
    if (xpbd_acceptance.first_nonfinite_step != 0u) failures.PushBack(Json::Str("nonfinite particles in quality replay"));
    if (xpbd_acceptance.first_pinned_failure_step != 0u) failures.PushBack(Json::Str("pinned particles moved or became nonfinite in quality replay"));
    if (xpbd_acceptance.first_undefined_strain_step != 0u) failures.PushBack(Json::Str("cloth strain is undefined in quality replay"));
    if (xpbd_acceptance.first_strain_failure_step != 0u) failures.PushBack(Json::Str("cloth length error exceeds the physical quality budget"));
    if (!coupling_valid) failures.PushBack(Json::Str("missing rows or positive impulse for a required system pair in the timed window of an environment"));
    if (!render_valid) failures.PushBack(Json::Str("sensor lifecycle, output or physics parity failed"));
    if (timed_sensors != replay_sensors) failures.PushBack(Json::Str("mounted sensor replay mismatch"));
    validity.Set("failures", std::move(failures));
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
        result.Set("schema_version", Json::Int(4));
        Json status = Json::Object();
        status.Set("valid", Json::Bool(false));
        status.Set("valid_scope", Json::Str("lifecycle/state/render/deformation/coupling workload contracts"));
        status.Set("passes_full_physics_acceptance", Json::Bool(false));
        status.Set("physical_acceptance_status", Json::Str("unmeasured"));
        status.Set("physical_acceptance_reason", Json::Str("full-system energy ledger, analytic checks, reference comparisons and independent CCD are not integrated into this program"));
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
