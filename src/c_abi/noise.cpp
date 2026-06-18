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
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/scoped_device_guard.hpp"
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
// right 3x3 block is mass*Identity, so the (3,3) element is the scalar mass. The
// static_assert pins the 6x6 row-major layout so a block reorder fails to compile
// instead of silently extracting the wrong scalar. This is the LIVE truth
// (set_link_mass keeps it in sync) and sidesteps the topo.masses vs bodies.masses
// ambiguity in the cooker fallback.
float MassFromInertia(const articulation::LinkSpatialInertia& inertia) {
    static_assert(sizeof(inertia.I) / sizeof(inertia.I[0]) == 36u,
                  "LinkSpatialInertia is not a flat 6x6 (row-major) matrix");
    constexpr uint32_t kSpatialDim = 6u;          // 6x6 spatial inertia.
    constexpr uint32_t kMassRow = 3u;             // bottom-right 3x3 == mass*I.
    return inertia.I[kMassRow * kSpatialDim + kMassRow];  // (3,3) == scalar mass.
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
    // M9: the legacy batched contact step params are gone; contact friction lives
    // on the nk Model material buckets. The DR friction multiplier has no host
    // scalar to poke -> the nominal stays 0 (inert). M10 named gap (RL contact DR
    // rebuilt on the nk world at M10).
    record.dr_nominal_friction = 0.0f;
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
    if (!record.world || record.articulation_host.TotalLinkCount() == 0u) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no link inertia to scale
    }
    if (record.device == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    CaptureNominalBaseline(record);

    auto& host = record.articulation_host;
    // The host mirror is SINGLE-ENV (CaptureArticulationHostMirror cooks the first
    // articulation only); the live nk arena LinkInertia/JointArmature fields are
    // env-replicated (links_per_env * env_count). The DR multiplier is per-env, so
    // we iterate the arena's full env-major span: per-env link count == the host
    // mirror's link count, which is the env-major tile stride.
    const uint32_t links_per_env = host.TotalLinkCount();
    const uint32_t env_count = (record.env_count == 0u) ? 1u : record.env_count;
    const uint32_t total_links = links_per_env * env_count;

    const cudaStream_t stream = nullptr;  // BUF-14: stream 0
    const int device_id = record.device->device_id;
    nuka::phi::ScopedDeviceGuard guard(device_id);

    auto* device_inertia =
        record.world->FieldPtr<articulation::LinkSpatialInertia>(
            nuka::nk::FieldId::LinkInertia);

    // --- mass: rebuild each link's spatial inertia from nominal*mult ----------
    // Per-env multiplier; applied to every link in that env replica. The host
    // mirror + nominal baseline are SINGLE-ENV (indexed by `local`); the live nk
    // arena LinkInertia field is env-replicated (indexed by the global `link`).
    for (uint32_t env = 0u; env < env_count; ++env) {
        const noise::SampledRandomization sampled =
            noise::SampleEpisodeRandomization(cfg, env);
        for (uint32_t local = 0u; local < links_per_env; ++local) {
            const uint32_t link = env * links_per_env + local;
            if (link >= total_links || local >= host.link_inertia.size()) {
                continue;
            }
            const float nominal =
                (local < record.dr_nominal_link_mass.size())
                    ? record.dr_nominal_link_mass[local]
                    : MassFromInertia(host.link_inertia[local]);
            const float new_mass = nominal * sampled.mass_multiplier;
            if (!(new_mass > 0.0f)) {
                continue;  // MakeSpatialInertia zeroes a non-positive mass
            }
            nuka::math::Vec3 diag{0.0f, 0.0f, 0.0f};
            nuka::math::Transform frame = nuka::math::Transform::Identity();
            if (!ResolveLinkInertiaParams(host, local, &diag, &frame)) {
                continue;
            }
            const articulation::LinkSpatialInertia new_inertia =
                articulation::MakeSpatialInertia(new_mass, diag, frame);
            // Keep the single-env host mirror in sync (overwritten each env, last
            // env's value wins -- the legacy behavior, the mirror is only the
            // diffsim dI/dmass source which DR does not gate on).
            host.link_inertia[local] = new_inertia;
            if (device_inertia != nullptr) {
                cudaError_t copy_status = cudaMemcpyAsync(
                    device_inertia + link, &new_inertia,
                    sizeof(articulation::LinkSpatialInertia),
                    cudaMemcpyHostToDevice, stream);
                if (copy_status != cudaSuccess) {
                    return NUKA_RESULT_INTERNAL;
                }
            }
        }
    }

    // --- joint armature: nominal + offset, per-link (env-tiled) --------------
    // The arena JointArmature field is per:link (env-replicated). The host mirror
    // + nominal baseline are single-env (indexed by `local`). Present on the
    // articulation state but INERT in the contact-free tape forward; set for RL
    // completeness + a consistent host/device mirror.
    auto* device_armature =
        record.world->FieldPtr<float>(nuka::nk::FieldId::JointArmature);
    if (!host.joint_armature.empty() && device_armature != nullptr) {
        const uint32_t dof_per_env =
            static_cast<uint32_t>(host.joint_armature.size());
        bool armature_changed = false;
        for (uint32_t env = 0u; env < env_count; ++env) {
            const noise::SampledRandomization sampled =
                noise::SampleEpisodeRandomization(cfg, env);
            for (uint32_t local = 0u; local < dof_per_env; ++local) {
                const float nominal_n =
                    (local < record.dr_nominal_joint_armature.size())
                        ? record.dr_nominal_joint_armature[local]
                        : 0.0f;
                const float value = nominal_n + sampled.joint_armature_offset;
                host.joint_armature[local] = value;  // single-env mirror sync
                const uint32_t dof = env * dof_per_env + local;
                cudaError_t copy_status = cudaMemcpyAsync(
                    device_armature + dof, &value, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
                if (copy_status != cudaSuccess) {
                    return NUKA_RESULT_INTERNAL;
                }
                armature_changed = true;
            }
        }
        (void)armature_changed;
    }

    // --- gravity.z: nominal + offset (one world scalar; env 0's sample) ------
    // Gravity is a single world property -> the SAME offset applies to all envs
    // (it cannot be per-env). Read by nuka_tape_create into RolloutParams.
    {
        const noise::SampledRandomization sampled =
            noise::SampleEpisodeRandomization(cfg, 0u);
        record.step_options.gravity.z =
            record.dr_nominal_gravity_z + sampled.gravity_z_offset;
        // friction_multiplier: M9 named gap. The legacy batched contact step
        // params (the only friction host scalar) are gone; contact friction now
        // lives on the nk Model material buckets, which DR does not yet poke. The
        // single-env contact-free diffsim tape has no contact solve -> inert
        // either way. RL contact DR (incl. friction) is rebuilt on the nk world
        // at M10. restitution_offset: likewise no buffer in this path -> inert.
        (void)sampled;
    }

    cudaStreamSynchronize(stream);
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
        const cudaStream_t stream = nullptr;  // BUF-14: stream 0
        const int device_id = record->device->device_id;
        float* data = static_cast<float*>(view.device_ptr);
        // Sensor buffers are small (<< 2^32 floats); guard the narrowing.
        if (view.element_count > 0xFFFFFFFFull) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        const uint32_t count = static_cast<uint32_t>(view.element_count);
        const uint64_t seq = record->noise_seq[f];

        switch (cfg.kind) {
            case noise::NoiseKind::Gaussian:
                noise::LaunchGaussianNoise(stream, device_id, data, count, cfg.param1,
                                           cfg.param2, cfg.seed, seq);
                break;
            case noise::NoiseKind::Poisson:
                noise::LaunchPoissonNoise(stream, device_id, data, count, cfg.param1, cfg.seed,
                                          seq);
                break;
            case noise::NoiseKind::None:
                return NUKA_RESULT_OK;  // unreachable (guarded above)
        }
        cudaStreamSynchronize(stream);
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
