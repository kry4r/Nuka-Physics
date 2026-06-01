// ---------------------------------------------------------------------------
// nuka::c_abi -- C ABI for sim-to-real N1 sensor noise (v0.5 p04 Task 5.4.8)
// ---------------------------------------------------------------------------
//
// nuka_world_set_sensor_noise records a per-field noise descriptor on the
// WorldRecord; nuka_world_apply_sensor_noise resolves the field's live device
// buffer (via nuka_world_get_buffer_view -- the SAME resolution the zero-copy
// view uses) and launches the matching counter-based Philox kernel, then
// advances the field's sequence counter so successive applies are independent
// noise across steps. The Philox RNG is stateless / pure in (seed, idx, seq), so
// the noise is D1 two-run bit-exact and replay-stable (exit #6). Default NONE is
// a byte no-op -> V1 oracle scenes stay byte-identical.
//
// No exceptions cross the extern "C" boundary (try/catch -> MapExceptionToResult,
// the buffer.cpp pattern). No `throw`.
// ---------------------------------------------------------------------------

#include "nuka/nuka_noise.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "sensor/noise/n1_gaussian.hpp"
#include "sensor/noise/n1_poisson.hpp"
#include "sensor/noise/noise_config.hpp"

#include <exception>
#include <new>

namespace {

namespace noise = nuka::sensor::noise;

// True iff `field` is a valid index into the WorldRecord noise arrays.
bool FieldInRange(nuka_state_field_t field) {
    const uint32_t f = static_cast<uint32_t>(field);
    return f < nuka::c_abi::WorldRecord::kNoiseFieldCount;
}

}  // namespace

extern "C" {

nuka_result_t nuka_world_set_sensor_noise(nuka_world_handle world,
                                          nuka_state_field_t sensor_field,
                                          const nuka_sensor_noise_desc_t* desc) {
    if (!FieldInRange(sensor_field)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    const uint32_t f = static_cast<uint32_t>(sensor_field);

    // NULL desc clears the field's noise (back to None).
    if (desc == nullptr) {
        record->noise_config[f] = noise::SensorNoiseConfig{};
        record->noise_seq[f] = 0u;
        return NUKA_RESULT_OK;
    }

    noise::NoiseKind kind;
    switch (desc->kind) {
        case NUKA_NOISE_NONE:
            kind = noise::NoiseKind::None;
            break;
        case NUKA_NOISE_GAUSSIAN:
            kind = noise::NoiseKind::Gaussian;
            break;
        case NUKA_NOISE_POISSON:
            kind = noise::NoiseKind::Poisson;
            break;
        default:
            return NUKA_RESULT_INVALID_ARG;
    }

    noise::SensorNoiseConfig cfg;
    cfg.kind = kind;
    cfg.param1 = desc->param1;
    cfg.param2 = desc->param2;
    cfg.seed = desc->seed;
    record->noise_config[f] = cfg;
    record->noise_seq[f] = 0u;  // fresh registration -> fresh sequence
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_apply_sensor_noise(nuka_world_handle world,
                                            nuka_state_field_t sensor_field) {
    if (!FieldInRange(sensor_field)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    const uint32_t f = static_cast<uint32_t>(sensor_field);
    const noise::SensorNoiseConfig& cfg = record->noise_config[f];

    // NONE (or unregistered) -> byte no-op, no buffer touched, OK.
    if (cfg.kind == noise::NoiseKind::None) {
        return NUKA_RESULT_OK;
    }

    try {
        // Resolve the field's live device buffer the SAME way the zero-copy view
        // does (reuse the extern "C" entry in buffer.cpp -- same shared lib).
        nuka_buffer_view_t view;
        const nuka_result_t view_result =
            nuka_world_get_buffer_view(world, sensor_field, &view);
        if (view_result != NUKA_RESULT_OK) {
            return view_result;
        }
        if (view.device_ptr == nullptr || view.element_count == 0u) {
            return NUKA_RESULT_OK;  // nothing to perturb
        }
        // The noise primitive operates on float32 elements. Reject non-float-
        // stride fields (e.g. the 28-byte pose / 24-byte velocity struct views):
        // honest default for a float-noise op.
        if (view.element_stride_bytes != sizeof(float)) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        if (record->device == nullptr) {
            return NUKA_RESULT_NULL_HANDLE;
        }
        const nuka::phi::DeviceContext& ctx = record->device->context;
        float* data = static_cast<float*>(view.device_ptr);
        // Sensor buffers are small (<< 2^32 floats); guard the narrowing.
        if (view.element_count > 0xFFFFFFFFull) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        const uint32_t count = static_cast<uint32_t>(view.element_count);
        const uint64_t seq = record->noise_seq[f];

        switch (cfg.kind) {
            case noise::NoiseKind::Gaussian:
                noise::LaunchGaussianNoise(ctx, data, count, cfg.param1,
                                           cfg.param2, cfg.seed, seq);
                break;
            case noise::NoiseKind::Poisson:
                noise::LaunchPoissonNoise(ctx, data, count, cfg.param1, cfg.seed,
                                          seq);
                break;
            case noise::NoiseKind::None:
                return NUKA_RESULT_OK;  // unreachable (guarded above)
        }
        ctx.stream.Synchronize();
        record->noise_seq[f] = seq + 1u;  // advance for the next apply
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
