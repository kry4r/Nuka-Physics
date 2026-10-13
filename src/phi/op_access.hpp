#pragma once

#include "phi/backend.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace nuka::phi {

enum class OpResource : uint8_t {
    Coordinates,
    ArticulationVelocity,
    ArticulationFrames,
    ArticulationDynamics,
    MassResponse,
    BodyPose,
    BodyDynamics,
    Parameters,
    Drives,
    Particles,
    MaterialGrid,
    Collision,
    Contacts,
    ContactWorkspace,
    Rows,
    Surfaces,
    GeometryHistory,
    VelocityReference,
    Readout,
    Status,
    Sensors,
    Observation,
    StepAccumulator,
    CapturedState,
    Count
};

struct OpAccess {
    uint64_t read;
    uint64_t write;
};

namespace op_access_detail {
inline constexpr uint64_t Coordinates = uint64_t{1} << static_cast<uint8_t>(OpResource::Coordinates);
inline constexpr uint64_t ArticulationVelocity = uint64_t{1} << static_cast<uint8_t>(OpResource::ArticulationVelocity);
inline constexpr uint64_t ArticulationFrames = uint64_t{1} << static_cast<uint8_t>(OpResource::ArticulationFrames);
inline constexpr uint64_t ArticulationDynamics = uint64_t{1} << static_cast<uint8_t>(OpResource::ArticulationDynamics);
inline constexpr uint64_t MassResponse = uint64_t{1} << static_cast<uint8_t>(OpResource::MassResponse);
inline constexpr uint64_t BodyPose = uint64_t{1} << static_cast<uint8_t>(OpResource::BodyPose);
inline constexpr uint64_t BodyDynamics = uint64_t{1} << static_cast<uint8_t>(OpResource::BodyDynamics);
inline constexpr uint64_t Parameters = uint64_t{1} << static_cast<uint8_t>(OpResource::Parameters);
inline constexpr uint64_t Drives = uint64_t{1} << static_cast<uint8_t>(OpResource::Drives);
inline constexpr uint64_t Particles = uint64_t{1} << static_cast<uint8_t>(OpResource::Particles);
inline constexpr uint64_t MaterialGrid = uint64_t{1} << static_cast<uint8_t>(OpResource::MaterialGrid);
inline constexpr uint64_t Collision = uint64_t{1} << static_cast<uint8_t>(OpResource::Collision);
inline constexpr uint64_t Contacts = uint64_t{1} << static_cast<uint8_t>(OpResource::Contacts);
inline constexpr uint64_t ContactWorkspace = uint64_t{1} << static_cast<uint8_t>(OpResource::ContactWorkspace);
inline constexpr uint64_t Rows = uint64_t{1} << static_cast<uint8_t>(OpResource::Rows);
inline constexpr uint64_t Surfaces = uint64_t{1} << static_cast<uint8_t>(OpResource::Surfaces);
inline constexpr uint64_t GeometryHistory = uint64_t{1} << static_cast<uint8_t>(OpResource::GeometryHistory);
inline constexpr uint64_t VelocityReference = uint64_t{1} << static_cast<uint8_t>(OpResource::VelocityReference);
inline constexpr uint64_t Readout = uint64_t{1} << static_cast<uint8_t>(OpResource::Readout);
inline constexpr uint64_t Status = uint64_t{1} << static_cast<uint8_t>(OpResource::Status);
inline constexpr uint64_t Sensors = uint64_t{1} << static_cast<uint8_t>(OpResource::Sensors);
inline constexpr uint64_t Observation = uint64_t{1} << static_cast<uint8_t>(OpResource::Observation);
inline constexpr uint64_t StepAccumulator = uint64_t{1} << static_cast<uint8_t>(OpResource::StepAccumulator);
inline constexpr uint64_t CapturedState = uint64_t{1} << static_cast<uint8_t>(OpResource::CapturedState);
inline constexpr uint64_t World = (uint64_t{1} << static_cast<uint8_t>(OpResource::Count)) - 1u;
inline constexpr uint64_t Physical = World & ~(Sensors | Observation | CapturedState);
inline constexpr std::array<OpAccess, static_cast<size_t>(NkOp::Count)> kAccess{{
#define NUKA_OP(name, read, write) OpAccess{read, write},
#include "phi/op_catalog.inc"
#undef NUKA_OP
}};
} // namespace op_access_detail

struct OpScheduleEntry {
    std::vector<int> dependencies;
    uint32_t lane = 0;
};

inline std::vector<OpScheduleEntry> BuildOpSchedule(const OpCall* calls, int count,
                                                   uint32_t lanes) {
    constexpr size_t resources = static_cast<size_t>(OpResource::Count);
    std::array<int, resources> writer;
    writer.fill(-1);
    std::array<std::vector<int>, resources> readers;
    std::vector<OpScheduleEntry> schedule(count);
    std::vector<uint32_t> finish(count);
    std::vector<uint32_t> available(lanes, 0u);
    for (int i = 0; i < count; ++i) {
        auto& entry = schedule[i];
        const auto access = op_access_detail::kAccess[static_cast<size_t>(calls[i].op)];
        for (size_t resource = 0; resource < resources; ++resource) {
            const uint64_t bit = uint64_t{1} << resource;
            if ((access.read | access.write) & bit) {
                if (writer[resource] != -1) entry.dependencies.push_back(writer[resource]);
                if (access.write & bit)
                    entry.dependencies.insert(entry.dependencies.end(),
                                              readers[resource].begin(), readers[resource].end());
            }
        }
        auto& dependencies = entry.dependencies;
        std::sort(dependencies.begin(), dependencies.end());
        dependencies.erase(std::unique(dependencies.begin(), dependencies.end()), dependencies.end());
        uint32_t ready = 0u;
        for (const int dependency : dependencies) ready = std::max(ready, finish[dependency]);
        for (uint32_t lane = 1u; lane < lanes; ++lane)
            if (std::max(available[lane], ready) < std::max(available[entry.lane], ready))
                entry.lane = lane;
        finish[i] = std::max(available[entry.lane], ready) + 1u;
        available[entry.lane] = finish[i];
        for (size_t resource = 0; resource < resources; ++resource) {
            const uint64_t bit = uint64_t{1} << resource;
            if (access.write & bit) {
                writer[resource] = i;
                readers[resource].clear();
            } else if (access.read & bit) {
                readers[resource].push_back(i);
            }
        }
    }
    return schedule;
}

} // namespace nuka::phi
