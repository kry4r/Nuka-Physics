#pragma once
// Solver telemetry ring in 32-bit words: a header, then one record per physical substep.
// The record layout is nuka_telemetry_record_t of the C ABI.

#include <cstdint>

namespace nuka::nk {

inline constexpr uint32_t kTelemetrySolve = 1u << 0;
inline constexpr uint32_t kTelemetrySweeps = 1u << 1;
inline constexpr uint32_t kTelemetryEnergy = 1u << 2;
inline constexpr uint32_t kTelemetryVertices = 1u << 3;

inline constexpr uint32_t kTelemetryCap = 1u << 0;
inline constexpr uint32_t kTelemetryConverged = 1u << 1;
inline constexpr uint32_t kTelemetryEnergyValid = 1u << 2;

inline constexpr uint32_t kTelemetryElementNone = 0u;
inline constexpr uint32_t kTelemetryElementParticle = 1u;
inline constexpr uint32_t kTelemetryElementRow = 2u;

inline constexpr uint32_t kTelemetrySamples = 24u;
inline constexpr uint32_t kTelemetryRecords = 1024u;

struct TelemetryRecord {
    uint32_t sequence;
    uint32_t substep;
    uint32_t sweeps;
    uint32_t sweep_budget;
    uint32_t flags;
    uint32_t contacts;
    float vertex_residual;
    float vertex_force;
    float contact_residual;
    float max_penetration;
    uint32_t penetration_row;
    float acceleration;
    float energy;
    float energy_residual;
    uint32_t worst_kind;
    uint32_t worst_index;
    float worst_residual;
    uint32_t worst_side_kind[2];
    uint32_t worst_side_index[2];
    uint32_t sample_count;
    uint32_t sample_sweep[kTelemetrySamples];
    float sample_residual[kTelemetrySamples];
    uint32_t reserved[2];
};

// Committed record count, capacity, the open record's packed penetration (float bits high32,
// ~row low32) and the residual above which a spent budget counts as a cap.
struct TelemetryHeader {
    uint32_t head;
    uint32_t capacity;
    uint64_t penetration;
    float alert_velocity;
    uint32_t reserved[3];
};

inline constexpr uint32_t kTelemetryRecordWords = sizeof(TelemetryRecord) / sizeof(uint32_t);
inline constexpr uint32_t kTelemetryHeaderWords = sizeof(TelemetryHeader) / sizeof(uint32_t);
static_assert(kTelemetryRecordWords == 72u && kTelemetryHeaderWords == 8u, "telemetry words");

}  // namespace nuka::nk
