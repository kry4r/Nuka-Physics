// ---------------------------------------------------------------------------
// nuka::c_abi -- C ABI for the multi-step differentiable rollout (p02-B/C)
// ---------------------------------------------------------------------------
//
// Wraps the diffsim tape / checkpoint / recompute-orchestrator / backward-runner
// behind the nuka_diffsim.h handle surface. A TapeRecord owns the diffsim
// machinery bound to a world's articulation device state. The differentiable
// forward (nuka_world_step_with_tape) runs the contact-free PD step the p02-A
// adjoint reverses -- it is INDEPENDENT of the world's production contact stepper.
// ---------------------------------------------------------------------------

#include "nuka/nuka_diffsim.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "diffsim/backward_runner.hpp"
#include "diffsim/checkpoint.hpp"
#include "diffsim/recompute_orchestrator.hpp"
#include "diffsim/step_backward.hpp"
#include "diffsim/tape.hpp"
#include "phi/buffer.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nuka::c_abi {

namespace articulation = nuka::runtime::articulation;
namespace diffsim = nuka::diffsim;

// One differentiable-rollout context bound to a world. Owns the orchestrator
// (the contact-free forward + the uploaded gains / dI_dmass), the tape (per-step
// actions + records), the checkpoint manager (authoritative-state snapshots), and
// the backward runner. The world's articulation_device.View() is the live state
// all of them operate on.
struct TapeRecord {
    WorldRecord* world = nullptr;
    diffsim::TapeDesc desc;
    uint32_t total_link_count = 0u;
    std::unique_ptr<diffsim::RecomputeOrchestrator> orchestrator;
    std::unique_ptr<diffsim::Tape> tape;
    std::unique_ptr<diffsim::CheckpointManager> checkpoints;
    std::unique_ptr<diffsim::BackwardRunner> backward;
    uint32_t steps_since_checkpoint = 0u;  // for the K-cadence
};

}  // namespace nuka::c_abi

namespace {

using nuka::c_abi::TapeRecord;
using nuka::c_abi::WorldRecord;
namespace diffsim = nuka::diffsim;

nuka::c_abi::HandleTable<nuka_tape_t, TapeRecord>& TapeTable() {
    static nuka::c_abi::HandleTable<nuka_tape_t, TapeRecord> table;
    return table;
}

// Build the per-global-link mass params (mass / diagonal inertia / inertial
// frame) by iterating the host's articulations in the SAME global-link order
// BuildArticulationHostState concatenates them (the cooker always populates
// per-local-link masses / inertias / inertial_frames). These feed the
// representation-consistent dI/dmass slope the grad_mass spine contracts against.
std::vector<nuka::diffsim::LinkMassParams> BuildMassParams(
    const nuka::runtime::articulation::ArticulationHostState& host) {
    std::vector<nuka::diffsim::LinkMassParams> out;
    out.reserve(host.TotalLinkCount());
    for (const auto& topo : host.articulations) {
        const uint32_t link_count =
            static_cast<uint32_t>(topo.link_bodies.size());
        for (uint32_t local = 0u; local < link_count; ++local) {
            nuka::diffsim::LinkMassParams lp;
            lp.mass = (local < topo.masses.size()) ? topo.masses[local] : 0.0f;
            lp.diagonal_inertia =
                (local < topo.inertias.size())
                    ? topo.inertias[local]
                    : nuka::math::Vec3{0.0f, 0.0f, 0.0f};
            lp.inertial_frame =
                (local < topo.inertial_frames.size())
                    ? topo.inertial_frames[local]
                    : nuka::math::Transform::Identity();
            out.push_back(lp);
        }
    }
    return out;
}

}  // namespace

extern "C" {

nuka_result_t nuka_tape_create(nuka_world_handle world,
                               const nuka_tape_desc_t* desc,
                               nuka_tape_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    if (desc == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    WorldRecord* world_record = nuka::c_abi::WorldTable().Get(world);
    if (world_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (world_record->articulation_device.Empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // non-articulated world
    }
    try {
        auto record = std::make_unique<TapeRecord>();
        record->world = world_record;
        record->desc.checkpoint_interval = desc->checkpoint_interval;
        record->desc.max_tape_entries = desc->max_tape_entries;
        record->desc.max_checkpoints = desc->max_checkpoints;
        record->desc.recompute_on_backward = desc->recompute_on_backward;

        const auto& ctx = world_record->device->context;
        const auto state = world_record->articulation_device.View();
        record->total_link_count = state.total_link_count;

        diffsim::RolloutParams rp;
        rp.dt = world_record->step_options.dt;
        rp.gravity_z = world_record->step_options.gravity.z;

        const auto mass_params = BuildMassParams(world_record->articulation_host);

        record->orchestrator = std::make_unique<diffsim::RecomputeOrchestrator>(
            ctx, state, rp, world_record->drive_stiffness_host,
            world_record->drive_damping_host,
            world_record->drive_force_limits_host, mass_params);
        record->tape = std::make_unique<diffsim::Tape>(ctx, record->desc,
                                                       record->total_link_count);
        // Contact-free path: lambda_width 0 (the R2 slot is present-but-unused).
        record->checkpoints = std::make_unique<diffsim::CheckpointManager>(
            ctx, record->desc.max_checkpoints, state.articulation_count,
            record->total_link_count, /*lambda_width=*/0u);
        record->backward = std::make_unique<diffsim::BackwardRunner>(
            ctx, *record->orchestrator);

        *out = TapeTable().Insert(std::move(record));
        return (*out != nullptr) ? NUKA_RESULT_OK : NUKA_RESULT_INTERNAL;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_tape_destroy(nuka_tape_handle tape) { TapeTable().Remove(tape); }

nuka_result_t nuka_world_step_with_tape(nuka_world_handle world,
                                        nuka_tape_handle tape) {
    TapeRecord* record = TapeTable().Get(tape);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    WorldRecord* world_record = nuka::c_abi::WorldTable().Get(world);
    if (world_record == nullptr || world_record != record->world) {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        const auto& ctx = world_record->device->context;
        // The action this step is the world's current DRIVE_TARGET buffer (the
        // policy writes it through the buffer view before each step).
        const float* action = static_cast<const float*>(
            world_record->drive_targets_device.Data());

        // K-cadence: capture a checkpoint at step 0, then every K steps. In
        // full-tape mode (recompute_on_backward==0), capture EVERY step.
        const uint32_t step_index = record->tape->StepCount();
        const bool full_tape = (record->desc.recompute_on_backward == 0u);
        const uint32_t k = record->desc.checkpoint_interval;
        const bool capture_here =
            full_tape || (step_index == 0u) ||
            (record->steps_since_checkpoint >= k);

        uint32_t checkpoint_slot = 0u;
        uint32_t has_checkpoint = 0u;
        if (capture_here) {
            // Snapshot the PRE-step authoritative state (the start of this step).
            checkpoint_slot = record->checkpoints->Capture(
                record->orchestrator->State(), step_index, /*lambda=*/nullptr);
            has_checkpoint = 1u;
            record->steps_since_checkpoint = 0u;
        }

        // Record the action + entry, then advance the differentiable forward.
        record->tape->RecordStep(action, has_checkpoint, checkpoint_slot);
        record->orchestrator->StepOnce(action);
        ctx.stream.Synchronize();
        record->steps_since_checkpoint += 1u;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_tape_backward(nuka_tape_handle tape,
                                 const float* grad_observations_in,
                                 float* grad_actions_out,
                                 float* grad_parameters_out) {
    TapeRecord* record = TapeTable().Get(tape);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (grad_actions_out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        const auto& ctx = record->world->device->context;
        const uint32_t n = record->total_link_count;
        const uint32_t step_count = record->tape->StepCount();
        if (step_count == 0u) {
            return NUKA_RESULT_OK;
        }

        // Upload the final-state loss-adjoint seed (host -> device).
        nuka::phi::Buffer seed_q(n * sizeof(float), nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer seed_qdot(n * sizeof(float),
                                    nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer seed_lv(n * 6u * sizeof(float),
                                  nuka::phi::MemoryKind::Device);
        {
            nuka::phi::ScopedDeviceGuard guard(ctx.device_id);
            cudaMemsetAsync(seed_q.Data(), 0, n * sizeof(float),
                            ctx.stream.Native());
            cudaMemsetAsync(seed_qdot.Data(), 0, n * sizeof(float),
                            ctx.stream.Native());
            cudaMemsetAsync(seed_lv.Data(), 0, n * 6u * sizeof(float),
                            ctx.stream.Native());
        }
        if (grad_observations_in != nullptr) {
            seed_q.CopyFromHost(grad_observations_in, n * sizeof(float));
            seed_qdot.CopyFromHost(grad_observations_in + n, n * sizeof(float));
            seed_lv.CopyFromHost(grad_observations_in + 2u * n,
                                 n * 6u * sizeof(float));
        }

        // Device output buffers for the runner.
        nuka::phi::Buffer grad_actions_dev(
            static_cast<size_t>(step_count) * n * sizeof(float),
            nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer grad_mass_dev(n * sizeof(float),
                                        nuka::phi::MemoryKind::Device);

        diffsim::BackwardSeeds seeds;
        seeds.grad_q_final = static_cast<const float*>(seed_q.Data());
        seeds.grad_qdot_final = static_cast<const float*>(seed_qdot.Data());
        seeds.grad_link_velocity_final =
            static_cast<const float*>(seed_lv.Data());

        diffsim::BackwardOutputs outputs;
        outputs.grad_actions = static_cast<float*>(grad_actions_dev.Data());
        outputs.grad_mass = static_cast<float*>(grad_mass_dev.Data());

        record->backward->Run(*record->tape, *record->checkpoints, seeds,
                              outputs);

        // Readback device -> host.
        grad_actions_dev.CopyToHost(
            grad_actions_out,
            static_cast<size_t>(step_count) * n * sizeof(float));
        if (grad_parameters_out != nullptr) {
            grad_mass_dev.CopyToHost(grad_parameters_out, n * sizeof(float));
        }
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_tape_reset(nuka_tape_handle tape) {
    TapeRecord* record = TapeTable().Get(tape);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    try {
        record->tape->Reset();
        record->checkpoints->Reset();
        record->steps_since_checkpoint = 0u;
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

uint32_t nuka_tape_step_count(nuka_tape_handle tape) {
    TapeRecord* record = TapeTable().Get(tape);
    return (record != nullptr && record->tape) ? record->tape->StepCount() : 0u;
}

uint32_t nuka_tape_link_count(nuka_tape_handle tape) {
    TapeRecord* record = TapeTable().Get(tape);
    return (record != nullptr) ? record->total_link_count : 0u;
}

}  // extern "C"
