#include "robot_cloth_fluid_scene.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include "phi/backend_cuda/cuda_internal.cuh"
#include "phi/articulation_contract.hpp"
#include "render/render_world.hpp"
#include "render/sensor_backend.hpp"
#include "render/rt_adapter.hpp"
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
    uint32_t substeps = 1u;
    uint32_t state_sensors = 0u;
    uint32_t cloth_nx = fixture::kClothNx;
    uint32_t render_sensors = 0u, render_width = 256u, render_height = 256u;
    uint32_t render_samples = 4u, render_shadows = 4u, render_ao = 3u, render_warmup = 32u;
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
        else if (flag == "--state-sensors") options.state_sensors = ParseU32(value);
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
        else if (flag == "--render-output") options.render_output = value;
        else throw std::invalid_argument("unknown option " + flag);
    }
    if (options.envs == 0u || options.steps == 0u || options.capacity_scale == 0u || options.substeps == 0u ||
        options.state_sensors > 2u ||
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

std::vector<uint8_t> StateSensorBytes(nk::World& world) {
    std::vector<uint8_t> bytes;
    if (!world.StateSensors().HasActive()) return bytes;
    nuka::sensor::StateSensorBankSnapshot snapshot;
    fixture::Require(world.StateSensors().Capture(&snapshot) == phi::Status::Ok, "sensor snapshot failed");
    for (const auto& channel : snapshot.sensors)
        if (channel.active) bytes.insert(bytes.end(), channel.bytes.begin(), channel.bytes.end());
    const auto* times = reinterpret_cast<const uint8_t*>(snapshot.times.data());
    bytes.insert(bytes.end(), times, times + snapshot.times.size() * sizeof(double));
    return bytes;
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
    family("bend", model.bend_color_segments, caps.bend_cons_per_env);
    family("volume", model.vol_color_segments, caps.vol_cons_per_env);
    family("shape_match", model.sm_color_segments, caps.shape_match_slots_per_env);
    return result;
}

Json XpbdQuality(nk::World& world, uint32_t step) {
    const auto& model = world.GetModel();
    const auto& particles = model.particles;
    const auto& caps = model.capacities;
    std::vector<nuka::math::Vec3> positions(size_t{caps.env_count} * caps.particles_per_env);
    fixture::Require(world.GetData().DownloadField(nk::FieldId::ParticlePos, positions.data(),
                     positions.size() * sizeof(positions.front())), "XPBD position download failed");
    double distance_max = 0.0, distance_squared = 0.0, bend_max = 0.0, bend_squared = 0.0;
    double pinned_max = 0.0, distance_rms_max_env = 0.0;
    uint64_t distance_count = 0u, bend_count = 0u;
    for (uint32_t env = 0u; env < caps.env_count; ++env) {
        const auto* points = positions.data() + size_t{env} * caps.particles_per_env;
        double env_distance_squared = 0.0;
        for (uint32_t i = 0u; i < caps.particles_per_env; ++i) {
            const auto& point = points[i];
            fixture::Require(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z),
                             "non-finite particle in quality replay");
            if (particles.inv_mass[i] != 0.0f) continue;
            const auto& initial = particles.initial_pos[i];
            const double dx = double{point.x} - initial.x;
            const double dy = double{point.y} - initial.y;
            const double dz = double{point.z} - initial.z;
            pinned_max = std::max(pinned_max, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        for (uint32_t i = 0u; i < caps.dist_cons_per_env; ++i) {
            const auto& a = points[particles.dist_a[i]];
            const auto& b = points[particles.dist_b[i]];
            const double dx = double{a.x} - b.x, dy = double{a.y} - b.y, dz = double{a.z} - b.z;
            const double rest = particles.dist_rest[i];
            fixture::Require(rest > 0.0, "distance strain requires positive rest length");
            const double error = std::abs(std::sqrt(dx * dx + dy * dy + dz * dz) / rest - 1.0);
            distance_max = std::max(distance_max, error);
            distance_squared += error * error;
            env_distance_squared += error * error;
            ++distance_count;
        }
        if (caps.dist_cons_per_env > 0u)
            distance_rms_max_env = std::max(distance_rms_max_env,
                std::sqrt(env_distance_squared / caps.dist_cons_per_env));
        for (uint32_t i = 0u; i < caps.bend_cons_per_env; ++i) {
            double constraint = 0.0;
            for (uint32_t j = 0u; j < 4u; ++j) {
                const auto& gradient = particles.bend_gradients[size_t{i} * 4u + j];
                const auto& point = points[particles.bend_particles[size_t{i} * 4u + j]];
                constraint += double{gradient.x} * point.x + double{gradient.y} * point.y +
                              double{gradient.z} * point.z;
            }
            bend_max = std::max(bend_max, std::abs(constraint));
            bend_squared += constraint * constraint;
            ++bend_count;
        }
    }
    fixture::Require(pinned_max == 0.0, "pinned particle moved in quality replay");
    Json result = Json::Object();
    result.Set("step", Json::Int(step));
    result.Set("distance_count", Json::Int(distance_count));
    result.Set("distance_strain_max", Json::Float(distance_max));
    result.Set("distance_strain_rms", Json::Float(distance_count ?
        std::sqrt(distance_squared / static_cast<double>(distance_count)) : 0.0));
    result.Set("distance_strain_rms_max_env", Json::Float(distance_rms_max_env));
    result.Set("bend_count", Json::Int(bend_count));
    result.Set("bend_constraint_max", Json::Float(bend_max));
    result.Set("bend_constraint_rms", Json::Float(bend_count ?
        std::sqrt(bend_squared / static_cast<double>(bend_count)) : 0.0));
    result.Set("pinned_displacement_max_m", Json::Float(pinned_max));
    return result;
}

struct XpbdAcceptance {
    static constexpr double max_strain_limit = 0.05, rms_strain_limit = 0.01;
    double max_strain = 0.0, max_rms_strain = 0.0;
    uint32_t steps_checked = 0u, first_failed_step = 0u;

    void Observe(const Json& sample) {
        const double strain = sample.At("distance_strain_max").AsDouble();
        const double rms = sample.At("distance_strain_rms_max_env").AsDouble();
        max_strain = std::max(max_strain, strain);
        max_rms_strain = std::max(max_rms_strain, rms);
        ++steps_checked;
        if (first_failed_step == 0u && (strain > max_strain_limit || rms > rms_strain_limit))
            first_failed_step = static_cast<uint32_t>(sample.At("step").AsInt());
    }

    bool Valid() const { return steps_checked > 0u && first_failed_step == 0u; }

    Json Report() const {
        Json result = Json::Object();
        result.Set("scope", Json::Str("all replay steps and environments, including warmup; hard distance constraints"));
        result.Set("budget", Json::Str("inextensible cloth: 5% maximum edge length error and 1% per-environment RMS"));
        result.Set("distance_strain_max_limit", Json::Float(max_strain_limit));
        result.Set("distance_strain_rms_limit", Json::Float(rms_strain_limit));
        result.Set("distance_strain_max", Json::Float(max_strain));
        result.Set("distance_strain_rms_max_env", Json::Float(max_rms_strain));
        result.Set("steps_checked", Json::Int(steps_checked));
        result.Set("first_failed_step", first_failed_step > 0u ? Json::Int(first_failed_step) : Json::Null());
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
    uint32_t visited_count = 0u, articulation_islands = 0u;
    std::map<uint32_t, uint64_t> row_histogram, articulation_row_histogram, tree_histogram;
    for (const auto& island : islands) {
        const auto [offset, count, flags, env] = island;
        fixture::Require(env < caps.env_count && count != 0u &&
                         uint64_t{offset} + count <= order.size(), "invalid island span");
        std::set<uint32_t> trees;
        for (uint32_t i = 0u; i < count; ++i) {
            const uint32_t id = order[offset + i];
            fixture::Require(id < rows.size() && id / caps.max_rows_per_env == env && !visited[id] &&
                             (rows[id].flags & nk::nk_row_flags::kActive) &&
                             (i == 0u || order[offset + i - 1u] < id), "invalid island row ownership or order");
            visited[id] = true;
            ++visited_count;
            for (const auto& side : {rows[id].a, rows[id].b}) {
                if (side.kind == nk::kNkSideArtic) trees.insert(side.index);
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
    result.Set("active_rows", Json::Int(active_count));
    result.Set("articulation_sides", Json::Int(articulation_sides));
    result.Set("islands", Json::Int(island_count));
    result.Set("articulation_islands", Json::Int(articulation_islands));
    result.Set("rows_per_island", Histogram(row_histogram));
    result.Set("rows_per_articulation_island", Histogram(articulation_row_histogram));
    result.Set("trees_per_island", Histogram(tree_histogram));
    result.Set("canonical_schedule_fnv1a64", Json::Str(FormatDigest(hash)));
    return result;
}

class PipelineSensor {
public:
    PipelineSensor(const fixture::PreparedScene& prepared, const Options& options) : options_(options) {
        auto scene = fixture::RobotScene(prepared.path, true);
        nuka::scene::cook::CookToModelOptions cook_options;
        cook_options.contact_family = nuka::scene::cook::CookContactFamily::PairDriven;
        const auto cooked = nuka::scene::cook::CookToModel(scene, 1, cook_options);
        const auto world = nuka::render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
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
        backend_->RenderSensors(handle_, fk, options_.envs, options_.render_width, options_.render_height);
    }
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

Json RenderMeasurements(nk::World& world, const fixture::PreparedScene& prepared,
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
    PipelineSensor renderer(prepared, options);
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
        result.Set("synchronized_wall_mean_us", Json::Float(wall_ms * 1000.0 / options.steps));
        return result;
    };
    Json result = Json::Object(), config = Json::Object();
    result.Set("creation_ms", Json::Float(creation_ms));
    result.Set("physics_to_sensor", measure(true));
    const bool coupled_state_equal = State(world) == expected_state;
    const auto coupled_output = renderer.Output();
    for (uint32_t i = 0u; i < options.render_warmup; ++i) renderer.Render(world);
    result.Set("render_only", measure(false));
    const auto output = renderer.Output();
    const bool repeated_output_equal = output == coupled_output;
    const bool render_state_equal = State(world) == expected_state;
    const size_t pixels = size_t{options.envs} * options.render_sensors * options.render_width * options.render_height;
    uint64_t hits = 0u;
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
    result.Set("config", std::move(config));
    result.Set("geometry_scope", Json::Str("authored rigid visuals on the public batched sensor path; deforming particle surfaces are not supported by this path"));
    result.Set("boundary", Json::Str("physics-to-sensor uses live per-step poses; render-only repeats its final world; both complete all AOVs on device; output download is untimed"));
    result.Set("output_layout", Json::Str("env-camera-major color f32x3, depth f32, normal f32x3, albedo f32x3, prim u32"));
    result.Set("output_fnv1a64", Json::Str(Digest(output)));
    result.Set("hit_pixels", Json::Int(hits));
    result.Set("finite", Json::Bool(finite));
    result.Set("coupled_state_equal", Json::Bool(coupled_state_equal));
    result.Set("render_state_equal", Json::Bool(render_state_equal));
    result.Set("repeated_output_equal", Json::Bool(repeated_output_equal));
    result.Set("valid", Json::Bool(finite && hits > 0u && coupled_state_equal && render_state_equal && repeated_output_equal));
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
    const auto scene_path = options.scene == "robot-cloth-fluid"
        ? std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes/go2_stand.usda"
        : std::filesystem::path(options.scene);
    const auto prepare_start = Clock::now();
    const auto fixture_config = fixture::Cfg();
    const auto prepared = fixture::Prepare(scene_path, device, owner.backend, fixture_config);
    const double preparation_ms = Milliseconds(prepare_start);
    const auto cook_start = Clock::now();
    auto model = fixture::CookPrepared(prepared, options.envs, true, options.cloth_nx);
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
    const auto& caps = world.GetModel().capacities;
    std::vector<nk::NkRow> rows_host(size_t{caps.max_rows_per_env} * options.envs);
    std::vector<float> impulses(rows_host.size());
    std::vector<double> cloth_impulses(options.envs), fluid_impulses(options.envs);
    uint32_t last_cloth_contact_step = 0u, last_fluid_contact_step = 0u;
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
    Json workload_samples = Json::Array(), xpbd_samples = Json::Array();
    XpbdAcceptance xpbd_acceptance;
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
        auto xpbd_quality = XpbdQuality(world, i + 1u);
        xpbd_acceptance.Observe(xpbd_quality);
        fixture::Require(world.GetData().DownloadField(nk::FieldId::Urows, rows_host.data(),
                         rows_host.size() * sizeof(nk::NkRow)), "row download failed");
        fixture::Require(world.GetData().DownloadField(nk::FieldId::Lambda, impulses.data(),
                         impulses.size() * sizeof(float)), "impulse download failed");
        for (size_t index = 0u; index < rows_host.size(); ++index) {
            const auto& row = rows_host[index];
            if (!(row.flags & nk::nk_row_flags::kActive) ||
                !(row.flags & nk::nk_row_flags::kContactNormal) || impulses[index] <= 0.0f) continue;
            const auto& particle = row.a.kind == nk::kNkSideParticle ? row.a : row.b;
            const auto& rigid = row.a.kind == nk::kNkSideParticle ? row.b : row.a;
            if (particle.kind != nk::kNkSideParticle || rigid.kind != nk::kNkSideArtic) continue;
            const uint32_t env = particle.index / caps.particles_per_env;
            if (particle.index % caps.particles_per_env < world.GetModel().particles.n_soft_particles) {
                cloth_impulses[env] += impulses[index];
                last_cloth_contact_step = i + 1u;
            } else {
                fluid_impulses[env] += impulses[index];
                last_fluid_contact_step = i + 1u;
            }
        }
        if (i >= options.warmup && (i % 25u == 0u || i + 1u == options.warmup + options.steps)) {
            workload_samples.PushBack(IslandWorkload(world, rows_host, i + 1u));
            xpbd_samples.PushBack(std::move(xpbd_quality));
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
    result.Set("schema_version", Json::Int(3));
    result.Set("scene", Json::Str(options.scene));
    Json configuration = Json::Object();
    configuration.Set("envs", Json::Int(options.envs));
    configuration.Set("dt", Json::Float(options.dt));
    configuration.Set("substeps", Json::Int(options.substeps));
    configuration.Set("state_sensors", Json::Int(options.state_sensors));
    configuration.Set("fixture_dt", Json::Float(fixture_config.dt));
    configuration.Set("fixture_substeps", Json::Int(fixture_config.substeps));
    configuration.Set("seed", Json::Int(options.seed));
    configuration.Set("seed_usage", Json::Str("deterministic held control; optional sensor error and delivery randomization"));
    configuration.Set("vel_iters", Json::Int(config.vel_iters));
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
    Json sensors = Json::Object();
    sensors.Set("count", Json::Int(world.StateSensors().Count()));
    sensors.Set("timed_replay_bit_equal", Json::Bool(timed_sensors == replay_sensors));
    sensors.Set("state_fnv1a64", Json::Str(Digest(replay_sensors)));
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
    workload.Set("scope", Json::Str("untimed quality replay; active rows and canonical island ownership checked"));
    workload.Set("samples", std::move(workload_samples));
    workload.Set("xpbd", XpbdWorkload(world.GetModel()));
    result.Set("workload", std::move(workload));
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
    const auto valid_impulses = [](const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(), [](double value) { return value > 0.0 && std::isfinite(value); });
    };
    const bool coupling_valid = valid_impulses(cloth_impulses) && valid_impulses(fluid_impulses);
    Json coupling = Json::Object(), cloth_by_env = Json::Array(), fluid_by_env = Json::Array();
    for (double value : cloth_impulses) cloth_by_env.PushBack(Json::Float(value));
    for (double value : fluid_impulses) fluid_by_env.PushBack(Json::Float(value));
    coupling.Set("scope", Json::Str("every replay step and environment, including warmup; positive normal impulse required for both media"));
    coupling.Set("cloth_normal_impulse_by_env_Ns", std::move(cloth_by_env));
    coupling.Set("fluid_normal_impulse_by_env_Ns", std::move(fluid_by_env));
    coupling.Set("last_cloth_contact_step", Json::Int(last_cloth_contact_step));
    coupling.Set("last_fluid_contact_step", Json::Int(last_fluid_contact_step));
    coupling.Set("timed_samples_include_both_media", Json::Bool(cloth_rows > 0u && fluid_rows > 0u));
    coupling.Set("valid", Json::Bool(coupling_valid));
    quality.Set("coupling_acceptance", std::move(coupling));
    quality.Set("xpbd_samples", std::move(xpbd_samples));
    quality.Set("xpbd_acceptance", xpbd_acceptance.Report());
    result.Set("quality", std::move(quality));
    bool render_valid = true;
    if (options.render_sensors != 0u) {
        auto render = RenderMeasurements(world, prepared, options, timed_state);
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
                       coupling_valid && mismatched_envs.empty() && xpbd_acceptance.Valid() && render_valid &&
                       timed_sensors == replay_sensors;
    validity.Set("valid", Json::Bool(valid));
    Json failures = Json::Array();
    if (!xpbd_acceptance.Valid()) failures.PushBack(Json::Str("cloth length error exceeds the physical quality budget"));
    if (!coupling_valid) failures.PushBack(Json::Str("missing positive cloth/fluid-articulation impulse during complete replay"));
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
