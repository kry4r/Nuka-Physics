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
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "sensor/noise/n1_gaussian.hpp"
#include "sensor/noise/n1_poisson.hpp"
#include "sensor/noise/n2_domain_randomization.hpp"
#include "sensor/noise/noise_config.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <new>
#include <vector>

namespace {

namespace noise = nuka::sensor::noise;

// True iff `field` is a valid index into the WorldRecord noise arrays.
bool FieldInRange(nuka_state_field_t field) {
    const uint32_t f = static_cast<uint32_t>(field);
    return f < nuka::c_abi::WorldRecord::kNoiseFieldCount;
}

namespace articulation = nuka::runtime::articulation;

// Reads a global link's NOMINAL scalar mass from the host inertia mirror. In the
// 6x6 spatial inertia row-major layout MakeSpatialInertia produces, the bottom-
// right 3x3 block is mass*Identity, so I[3*6+3] == I[21] is the scalar mass. This
// is the LIVE truth (set_link_mass keeps it in sync) and sidesteps the
// topo.masses vs bodies.masses ambiguity in the cooker fallback.
float MassFromInertia(const articulation::LinkSpatialInertia& inertia) {
    return inertia.I[21];
}

// Resolves a GLOBAL link index to its (diagonal_inertia, inertial_frame) by
// walking the articulations in concatenation order -- the SAME resolution
// nuka_world_set_link_mass uses to reproduce the global-link layout
// BuildArticulationHostState emits. Returns false if out of range.
bool ResolveLinkInertiaParams(const articulation::ArticulationHostState& host,
                              uint32_t link_index,
                              nuka::math::Vec3* diagonal_inertia,
                              nuka::math::Transform* inertial_frame) {
    uint32_t global = 0u;
    for (const auto& topo : host.articulations) {
        const uint32_t link_count =
            static_cast<uint32_t>(topo.link_bodies.size());
        if (link_index < global + link_count) {
            const uint32_t local = link_index - global;
            *diagonal_inertia = (local < topo.inertias.size())
                                    ? topo.inertias[local]
                                    : nuka::math::Vec3{0.0f, 0.0f, 0.0f};
            *inertial_frame = (local < topo.inertial_frames.size())
                                  ? topo.inertial_frames[local]
                                  : nuka::math::Transform::Identity();
            return true;
        }
        global += link_count;
    }
    return false;
}

// Snapshots the NOMINAL baseline ONCE (per-link mass, per-DOF armature,
// gravity.z) so a repeated apply re-randomizes AROUND nominal (idempotent)
// instead of compounding a random walk. Idempotent itself (guarded by
// dr_baseline_captured).
void CaptureNominalBaseline(nuka::c_abi::WorldRecord& record) {
    if (record.dr_baseline_captured) {
        return;
    }
    const auto& host = record.articulation_host;
    const uint32_t total_links = host.TotalLinkCount();
    record.dr_nominal_link_mass.resize(total_links);
    for (uint32_t l = 0u; l < total_links && l < host.link_inertia.size(); ++l) {
        record.dr_nominal_link_mass[l] = MassFromInertia(host.link_inertia[l]);
    }
    record.dr_nominal_joint_armature = host.joint_armature;  // per-DOF copy
    record.dr_nominal_gravity_z = record.step_options.gravity.z;
    record.dr_nominal_friction = record.batched_step_params.friction_coefficient;
    record.dr_baseline_captured = true;
}

// Applies the sampled per-episode randomization to the world's engine buffers.
// Single-env (env_count == 1) is the tested contact-free diff-sim path. mass +
// gravity are TAPE-VISIBLE (they change the contact-free ABA forward + gradient);
// armature is present but inert in that forward; friction/restitution have no
// contact buffer here (sampled into `sampled` for RL completeness, inert). See
// nuka_noise.h for the full per-param buffer mapping.
nuka_result_t ApplyPerEpisodeRandomization(
    nuka::c_abi::WorldRecord& record,
    const noise::DomainRandomizationConfig& cfg) {
    if (!cfg.enabled) {
        return NUKA_RESULT_OK;  // byte no-op -> oracle safe
    }
    if (record.articulation_device.Empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no link inertia to scale
    }
    if (record.device == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    CaptureNominalBaseline(record);

    auto& host = record.articulation_host;
    const uint32_t total_links = host.TotalLinkCount();
    const uint32_t env_count = (record.env_count == 0u) ? 1u : record.env_count;
    // Links/DOFs are tiled env-major (ReplicateArticulationHostState appends
    // replica e after e-1), so per-env link count = total_links / env_count.
    const uint32_t links_per_env =
        (env_count > 0u) ? (total_links / env_count) : total_links;

    const auto& ctx = record.device->context;
    nuka::phi::ScopedDeviceGuard guard(ctx.device_id);

    auto* device_inertia = static_cast<articulation::LinkSpatialInertia*>(
        record.articulation_device.link_inertia.Data());

    // --- mass: rebuild each link's spatial inertia from nominal*mult ----------
    // Per-env multiplier; applied to every link in that env replica.
    for (uint32_t env = 0u; env < env_count; ++env) {
        const noise::SampledRandomization sampled =
            noise::SampleEpisodeRandomization(cfg, env);
        for (uint32_t local = 0u; local < links_per_env; ++local) {
            const uint32_t link = env * links_per_env + local;
            if (link >= total_links || link >= host.link_inertia.size()) {
                continue;
            }
            const float nominal =
                (link < record.dr_nominal_link_mass.size())
                    ? record.dr_nominal_link_mass[link]
                    : MassFromInertia(host.link_inertia[link]);
            const float new_mass = nominal * sampled.mass_multiplier;
            if (!(new_mass > 0.0f)) {
                continue;  // MakeSpatialInertia zeroes a non-positive mass
            }
            nuka::math::Vec3 diag{0.0f, 0.0f, 0.0f};
            nuka::math::Transform frame = nuka::math::Transform::Identity();
            if (!ResolveLinkInertiaParams(host, link, &diag, &frame)) {
                continue;
            }
            const articulation::LinkSpatialInertia new_inertia =
                articulation::MakeSpatialInertia(new_mass, diag, frame);
            host.link_inertia[link] = new_inertia;  // keep host mirror in sync
            if (device_inertia != nullptr) {
                cudaError_t copy_status = cudaMemcpyAsync(
                    device_inertia + link, &new_inertia,
                    sizeof(articulation::LinkSpatialInertia),
                    cudaMemcpyHostToDevice, ctx.stream.Native());
                if (copy_status != cudaSuccess) {
                    return NUKA_RESULT_INTERNAL;
                }
            }
        }
    }

    // --- joint armature: nominal + offset, per-DOF (env-tiled) ---------------
    // Present on the articulation state but INERT in the contact-free tape
    // forward; set for RL completeness + a consistent host/device mirror.
    if (!host.joint_armature.empty()) {
        const uint32_t total_dof =
            static_cast<uint32_t>(host.joint_armature.size());
        const uint32_t dof_per_env =
            (env_count > 0u) ? (total_dof / env_count) : total_dof;
        bool armature_changed = false;
        for (uint32_t env = 0u; env < env_count; ++env) {
            const noise::SampledRandomization sampled =
                noise::SampleEpisodeRandomization(cfg, env);
            for (uint32_t local = 0u; local < dof_per_env; ++local) {
                const uint32_t dof = env * dof_per_env + local;
                if (dof >= total_dof) {
                    continue;
                }
                const float nominal =
                    (dof < record.dr_nominal_joint_armature.size())
                        ? record.dr_nominal_joint_armature[dof]
                        : 0.0f;
                host.joint_armature[dof] = nominal + sampled.joint_armature_offset;
                armature_changed = true;
            }
        }
        if (armature_changed &&
            record.articulation_device.joint_armature.Data() != nullptr) {
            cudaError_t copy_status = cudaMemcpyAsync(
                record.articulation_device.joint_armature.Data(),
                host.joint_armature.data(), total_dof * sizeof(float),
                cudaMemcpyHostToDevice, ctx.stream.Native());
            if (copy_status != cudaSuccess) {
                return NUKA_RESULT_INTERNAL;
            }
        }
    }

    // --- gravity.z: nominal + offset (one world scalar; env 0's sample) ------
    // Gravity is a single world property -> the SAME offset applies to all envs
    // (it cannot be per-env). Read by nuka_tape_create into RolloutParams.
    {
        const noise::SampledRandomization sampled =
            noise::SampleEpisodeRandomization(cfg, 0u);
        record.step_options.gravity.z =
            record.dr_nominal_gravity_z + sampled.gravity_z_offset;
        // friction_coefficient lives on the BATCHED contact path's step params;
        // mirror env 0's friction multiplier there for the batched RL case. The
        // single-env contact-free tape has no contact solve -> inert.
        if (record.batched != nullptr) {
            // friction is a MULTIPLIER against the captured nominal (idempotent
            // across resets). The single-env contact-free tape has no contact
            // solve -> inert; this serves the batched RL case.
            record.batched_step_params.friction_coefficient =
                record.dr_nominal_friction * sampled.friction_multiplier;
        }
        // restitution_offset: no engine buffer in the contact-free path -> inert.
    }

    ctx.stream.Synchronize();
    return NUKA_RESULT_OK;
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

nuka_result_t nuka_world_set_domain_randomization(
    nuka_world_handle world, const nuka_domain_randomization_desc_t* desc) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // NULL desc clears DR back to disabled (apply becomes a byte no-op).
    if (desc == nullptr) {
        record->dr_config = noise::DomainRandomizationConfig{};
        record->dr_config.enabled = false;
        return NUKA_RESULT_OK;
    }
    noise::DomainRandomizationConfig cfg;
    cfg.mass_multiplier = {desc->mass_mul_lo, desc->mass_mul_hi};
    cfg.friction_multiplier = {desc->friction_mul_lo, desc->friction_mul_hi};
    cfg.restitution_offset = {desc->restitution_off_lo, desc->restitution_off_hi};
    cfg.joint_armature_offset = {desc->armature_off_lo, desc->armature_off_hi};
    cfg.gravity_z_offset = {desc->gravity_off_lo, desc->gravity_off_hi};
    cfg.seed = desc->seed;
    cfg.enabled = (desc->enabled != 0);
    record->dr_config = cfg;
    // NOTE: recording does NOT reset dr_baseline_captured -- the nominal baseline
    // is the world's true cooked state, independent of which DR ranges are set.
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_apply_domain_randomization(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    try {
        return ApplyPerEpisodeRandomization(*record, record->dr_config);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
