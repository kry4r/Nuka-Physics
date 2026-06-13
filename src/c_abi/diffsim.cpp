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
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/buffer_legacy.hpp"
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
    if (!world_record->world ||
        world_record->articulation_host.TotalLinkCount() == 0u) {
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
        // M9 SEAM (now LIVE): the diffsim machinery consumes ArticulationDeviceState
        // -- a pure pointer view. The pointers now come from the nk arena via
        // MakeArticulationDeviceStateFromViews(ModelView, DataView, links, artics):
        // algorithm/kernels are UNTOUCHED (proven bit-exact by
        // AbaReverse.NkArenaSeamForwardReverseBitExact). The legacy backward .cu
        // kernels still run on the phi v1 ctx (the legacy DeviceContext stays in
        // DeviceRecord until the backward op-ifies at T7). The diffsim path is
        // single-env (the tape tests create env_count == 1), so the host mirror's
        // single-env link/articulation counts == the arena's.
        const uint32_t total_link_count =
            world_record->articulation_host.TotalLinkCount();
        const uint32_t articulation_count =
            world_record->articulation_host.ArticulationCount();
        const auto state =
            nuka::runtime::articulation::MakeArticulationDeviceStateFromViews(
                world_record->world->ModelViewRef(),
                world_record->world->DataViewRef(), total_link_count,
                articulation_count);
        record->total_link_count = state.total_link_count;

        diffsim::RolloutParams rp;
        rp.dt = world_record->step_options.dt;
        rp.gravity_z = world_record->step_options.gravity.z;

        const auto mass_params = BuildMassParams(world_record->articulation_host);

        // The PD gains source: the cook's hold-drive template on the Model (the
        // legacy BuildHoldDriveTargets 1:1 -- byte-identical stiffness = gain,
        // damping = 2*sqrt(gain), force_limits). RolloutParams reads them host-side.
        const nuka::nk::ModelHoldDrives& drives =
            world_record->world->GetModel().hold_drives;
        record->orchestrator = std::make_unique<diffsim::RecomputeOrchestrator>(
            ctx, state, rp, drives.stiffness, drives.damping, drives.force_limits,
            mass_params);
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
        // The action this step is the world's current DRIVE_TARGET arena field
        // (the policy writes it through the buffer view before each step). The
        // cook seeds it to the hold-pose targets (SeedInitialState), so an
        // un-written single-env tape records a constant hold action -- byte-
        // identical to the legacy drive_targets_device the upload built.
        const float* action =
            world_record->world->FieldPtr<float>(nuka::nk::FieldId::DriveTarget);
        if (action == nullptr) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

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

nuka_result_t nuka_tape_state_view(nuka_tape_handle tape,
                                   nuka_state_field_t field,
                                   nuka_buffer_view_t* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    out->device_ptr = nullptr;
    out->element_count = 0u;
    out->element_stride_bytes = 0u;
    out->dtype = 0u;

    TapeRecord* record = TapeTable().Get(tape);
    if (record == nullptr || !record->orchestrator) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    try {
        // The orchestrator's live device state -- the SAME pointers
        // nuka_world_step_with_tape advances in place (StepOnce). Serving them
        // device-direct here is what makes the tape OBSERVABLE: a fresh view
        // after step_with_tape reflects the evolved q/qdot/link_velocity (the
        // host-mirror path in nuka_world_get_buffer_view's single-env arm does
        // NOT, because step_with_tape never updates the host mirror).
        const auto state = record->orchestrator->State();
        if (field == NUKA_FIELD_JOINT_POSITION) {
            out->device_ptr = static_cast<void*>(state.q);
            out->element_count = record->total_link_count;
            out->element_stride_bytes = sizeof(float);
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        if (field == NUKA_FIELD_JOINT_VELOCITY) {
            out->device_ptr = static_cast<void*>(state.qdot);
            out->element_count = record->total_link_count;
            out->element_stride_bytes = sizeof(float);
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        if (field == NUKA_FIELD_LINK_VELOCITY) {
            // Per-link spatial velocity, link-major; each element is a
            // LinkSpatialVel == 6 floats [wx,wy,wz, vx,vy,vz] (omega-first).
            // SAME layout/comment as buffer.cpp's batched LINK_VELOCITY branch.
            out->device_ptr = static_cast<void*>(state.link_velocity);
            out->element_count = record->total_link_count;
            out->element_stride_bytes = static_cast<uint32_t>(
                sizeof(nuka::runtime::articulation::LinkSpatialVel));
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        return NUKA_RESULT_NOT_SUPPORTED;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_set_link_mass(nuka_world_handle world,
                                       uint32_t link_index, float mass) {
    WorldRecord* world_record = nuka::c_abi::WorldTable().Get(world);
    if (world_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!world_record->world ||
        world_record->articulation_host.TotalLinkCount() == 0u) {
        return NUKA_RESULT_NOT_SUPPORTED;  // non-articulated world
    }
    auto& host = world_record->articulation_host;
    const uint32_t total_links = host.TotalLinkCount();
    if (link_index >= total_links || link_index >= host.link_inertia.size()) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // Reject a non-positive mass: MakeSpatialInertia's mass<=0 guard returns a
    // ZERO 6x6, which is degenerate for ABA (singular articulated inertia) and
    // makes the affine dI/dmass parameterization the adjoint assumes inapplicable.
    if (!(mass > 0.0f)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        // Resolve the GLOBAL link index to its (articulation, local) so we read
        // the SAME (diagonal_inertia, inertial_frame) the cooker stored -- the
        // identical source BuildMassParams uses to build dI/dmass. Iterating the
        // articulations in concatenation order reproduces the global-link layout
        // BuildArticulationHostState emits.
        nuka::math::Vec3 diagonal_inertia{0.0f, 0.0f, 0.0f};
        nuka::math::Transform inertial_frame =
            nuka::math::Transform::Identity();
        {
            uint32_t global = 0u;
            bool resolved = false;
            for (const auto& topo : host.articulations) {
                const uint32_t link_count =
                    static_cast<uint32_t>(topo.link_bodies.size());
                if (link_index < global + link_count) {
                    const uint32_t local = link_index - global;
                    diagonal_inertia = (local < topo.inertias.size())
                                           ? topo.inertias[local]
                                           : nuka::math::Vec3{0.0f, 0.0f, 0.0f};
                    inertial_frame =
                        (local < topo.inertial_frames.size())
                            ? topo.inertial_frames[local]
                            : nuka::math::Transform::Identity();
                    resolved = true;
                    break;
                }
                global += link_count;
            }
            if (!resolved) {
                return NUKA_RESULT_INVALID_ARG;
            }
        }

        // Reconstruct the link's spatial inertia from the new scalar mass using
        // the EXACT MakeSpatialInertia parameterization the engine cooked it with:
        // COM (inertial_frame.position) + the rotated diagonal inertia tensor are
        // HELD FIXED, only `mass` changes. This is affine in mass and matches the
        // backward's dI/dmass = MakeSpatialInertia(m+1) - MakeSpatialInertia(m)
        // (see diffsim::BuildSpatialInertiaMassJacobian). The COM-fixed convention
        // means dI/dmass is constant -- exactly what the p02 adjoint contracts
        // grad_I against.
        const nuka::runtime::articulation::LinkSpatialInertia new_inertia =
            nuka::runtime::articulation::MakeSpatialInertia(
                mass, diagonal_inertia, inertial_frame);

        // Update the host mirror (keeps BuildMassParams / the DR nominal baseline
        // consistent) and write the SINGLE link's 6x6 into the live nk arena
        // LinkInertia field in place (offset link_index, no realloc). The diffsim
        // seam reads link_inertia from THIS arena field (model_view.link_inertia),
        // and AbaPass1 reads it FRESH every step (CopySpatialInertia +
        // Mat66MulVec6(I, v)), so the write takes effect on the NEXT step; there
        // is no derived/cached inertia to invalidate. dI/dmass is mass-independent
        // (affine), so the tape's uploaded slope needs no refresh.
        host.link_inertia[link_index] = new_inertia;

        const auto& ctx = world_record->device->context;
        auto* device_inertia =
            world_record->world->FieldPtr<nuka::runtime::articulation::LinkSpatialInertia>(
                nuka::nk::FieldId::LinkInertia);
        if (device_inertia == nullptr) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        nuka::phi::ScopedDeviceGuard guard(ctx.device_id);
        cudaError_t copy_status = cudaMemcpyAsync(
            device_inertia + link_index, &new_inertia,
            sizeof(nuka::runtime::articulation::LinkSpatialInertia),
            cudaMemcpyHostToDevice, ctx.stream.Native());
        if (copy_status != cudaSuccess) {
            return NUKA_RESULT_INTERNAL;
        }
        ctx.stream.Synchronize();
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_set_gravity_z(nuka_world_handle world, float gravity_z) {
    WorldRecord* world_record = nuka::c_abi::WorldTable().Get(world);
    if (world_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // A plain host-side scalar write to the world's step options. nuka_tape_create
    // reads step_options.gravity.z into RolloutParams.gravity_z, so this must be
    // called BEFORE the tape is created to take effect on the differentiable
    // rollout. No device work; nothing cached/derived to invalidate.
    world_record->step_options.gravity.z = gravity_z;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_set_sparse_solver_backend(
    nuka_world_handle world, nuka_sparse_solver_backend_t backend) {
    WorldRecord* world_record = nuka::c_abi::WorldTable().Get(world);
    if (world_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    switch (backend) {
        case NUKA_SOLVER_BACKEND_SELF_CG:
        case NUKA_SOLVER_BACKEND_SELF_MINRES:
        case NUKA_SOLVER_BACKEND_SELF_GMRES:
            // SHIPPING backends: v0.7 p01 hardened self-written CG (SPD), v0.7 p02
            // self-written MINRES (symmetric indefinite + ILU(0)), and v0.7 p03-A
            // self-written GMRES (general non-symmetric Krylov, restarted GMRES(m)).
            // A plain host scalar write; read at solver-construction time. All round-
            // trip through MakeSparseSolverBackend ("self_cg"/"self_minres"/
            // "self_gmres").
            world_record->sparse_solver_backend =
                static_cast<uint32_t>(backend);
            return NUKA_RESULT_OK;
        default:
            return NUKA_RESULT_INVALID_ARG;
    }
}

nuka_result_t nuka_world_get_sparse_solver_backend(
    nuka_world_handle world, nuka_sparse_solver_backend_t* out) {
    WorldRecord* world_record = nuka::c_abi::WorldTable().Get(world);
    if (world_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = static_cast<nuka_sparse_solver_backend_t>(
        world_record->sparse_solver_backend);
    return NUKA_RESULT_OK;
}

}  // extern "C"
