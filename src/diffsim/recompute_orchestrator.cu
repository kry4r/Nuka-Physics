// ---------------------------------------------------------------------------
// nuka::diffsim -- RecomputeOrchestrator implementation (p02-B/C)
// ---------------------------------------------------------------------------

#include "diffsim/recompute_orchestrator.hpp"

#include "runtime/articulation/featherstone_aba.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {
void CheckCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("nuka::diffsim::RecomputeOrchestrator ") + what + ": " +
            cudaGetErrorString(status));
    }
}

phi::Buffer UploadFloat(const phi::DeviceContext& ctx,
                        const std::vector<float>& v) {
    phi::Buffer b(v.size() * sizeof(float), phi::MemoryKind::Device);
    if (!v.empty()) {
        phi::ScopedDeviceGuard guard(ctx.device_id);
        CheckCuda(cudaMemcpyAsync(b.Data(), v.data(), v.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, ctx.stream.Native()),
                  "UploadFloat");
    }
    return b;
}
}  // namespace

RecomputeOrchestrator::RecomputeOrchestrator(
    const phi::DeviceContext& context,
    articulation::ArticulationDeviceState state, const RolloutParams& params,
    const std::vector<float>& drive_stiffness,
    const std::vector<float>& drive_damping,
    const std::vector<float>& drive_force_limits,
    const std::vector<LinkMassParams>& mass_params)
    : context_(context), state_(state), params_(params) {
    total_link_count_ = state.total_link_count;
    if (total_link_count_ == 0u) {
        throw std::invalid_argument(
            "nuka::diffsim::RecomputeOrchestrator: empty state");
    }
    if (drive_stiffness.size() != total_link_count_ ||
        drive_damping.size() != total_link_count_ ||
        drive_force_limits.size() != total_link_count_ ||
        mass_params.size() != total_link_count_) {
        throw std::invalid_argument(
            "nuka::diffsim::RecomputeOrchestrator: gain/mass length != "
            "total_link_count");
    }
    stiffness_ = UploadFloat(context_, drive_stiffness);
    damping_ = UploadFloat(context_, drive_damping);
    force_limits_ = UploadFloat(context_, drive_force_limits);
    // Representation-consistent dI/dmass slope (parallel-axis + COM coupling),
    // the SAME mapping the engine forward uses (see step_backward_host.cpp).
    const std::vector<float> dIdm = BuildSpatialInertiaMassJacobian(mass_params);
    dI_dmass_ = UploadFloat(context_, dIdm);
    context_.stream.Synchronize();
}

void RecomputeOrchestrator::StepOnce(const float* device_actions) {
    // EXACTLY the contact-free PD step the p02-A adjoint reverses (explicit
    // damping, defer_velocity_damping=false; NO contact pipeline). The kernel
    // sequence mirrors the FD test's ForwardFullStep + the floating integrators.
    articulation::FeatherstoneAba::ApplyPositionDrives(
        context_, state_, device_actions, DriveStiffness(), DriveDamping(),
        DriveForceLimits(), /*defer_velocity_damping=*/false);
    articulation::FeatherstoneAba::ComputeAccelerations(context_, state_,
                                                        params_.gravity_z);
    articulation::FeatherstoneAba::IntegrateVelocity(context_, state_, params_.dt);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(
        context_, state_, params_.dt, params_.gravity_z);
    articulation::FeatherstoneAba::IntegratePosition(context_, state_, params_.dt);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, state_,
                                                            params_.dt);
}

StepBackwardInputs RecomputeOrchestrator::ReconstructInputs(const Tape& tape,
                                                            uint32_t step) const {
    StepBackwardInputs in;
    // q_pre / qdot_pre are filled by the backward runner (it owns the pre-step
    // snapshot buffers and snapshots them at the right moment).
    in.drive_targets = tape.ActionSlice(step);
    in.drive_stiffness = DriveStiffness();
    in.drive_damping = DriveDamping();
    in.drive_force_limits = DriveForceLimits();
    in.dI_dmass = DiDMass();
    in.dt = params_.dt;
    in.gravity_z = params_.gravity_z;
    in.has_drive = true;
    in.has_integrate = true;
    in.enable_q_channel = true;
    in.grad_qddot_seed = nullptr;
    return in;
}

}  // namespace nuka::diffsim
