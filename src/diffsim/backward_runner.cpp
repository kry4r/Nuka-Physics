// ---------------------------------------------------------------------------
// nuka::diffsim -- BackwardRunner implementation (p02-B/C)
// ---------------------------------------------------------------------------
//
// M9 T7: was backward_runner.cu. The runner is pure HOST orchestration of the
// reverse pass -- checkpoint restore, window roll-forward + cache, descending
// reverse, and per-step StepBackward dispatch. Its only device kernel was the
// fixed-order atomic-free grad_mass accumulate (acc += step); that kernel moved
// VERBATIM to the phi v2 op TU (phi/backend_cuda/ops/diffsim_backward.cu) and is
// driven here through diffsim::LaunchAddInPlace, so this file is now a .cpp. The
// reverse loop dispatches NkOp::StepBackward per step through StepBackward()
// (which drives the SINGLE op-resident kernel). Replay of the forward state uses
// the orchestrator's CONTACT-FREE op sequence (StepOnce: ApplyDrives mode0
// defer=false + explicit damping + ABA + integrate -- NOT World::Step), so the
// grads stay grad-vs-FD bit-exact AND replay-deterministic. Semantics + D2D
// copies + launch geometry are byte-for-byte UNCHANGED from backward_runner.cu.
// ---------------------------------------------------------------------------

#include "diffsim/backward_runner.hpp"

#include "math/transform.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::diffsim {

namespace {
void CheckCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::BackwardRunner ") +
                                 what + ": " + cudaGetErrorString(status));
    }
}

// Fixed-order, atomic-free accumulate: acc[i] += step[i]. The kernel itself lives
// in the phi v2 op TU (diffsim_backward.cu) so this file stays pure-host; the
// thin launcher (diffsim::LaunchAddInPlace) keeps the SAME 256-threads/block
// geometry and one-thread-per-link plain load/add/store -> D1 bit-exact.
void LaunchAdd(const phi::DeviceContext& ctx, float* acc, const float* step,
               uint32_t n) {
    phi::ScopedDeviceGuard guard(ctx.device_id);
    LaunchAddInPlace(acc, step, n, ctx.stream.Native());
    CheckCuda(cudaGetLastError(), "AddInPlace launch");
}

phi::Buffer ZeroedFloat(const phi::DeviceContext& ctx, size_t count) {
    phi::Buffer b(count * sizeof(float), phi::MemoryKind::Device);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    CheckCuda(cudaMemsetAsync(b.Data(), 0, count * sizeof(float),
                              ctx.stream.Native()),
              "ZeroedFloat memset");
    return b;
}
}  // namespace

BackwardRunner::BackwardRunner(const phi::DeviceContext& context,
                               RecomputeOrchestrator& orchestrator)
    : context_(context), orch_(orchestrator) {
    total_link_count_ = orch_.TotalLinkCount();
    articulation_count_ = orch_.State().articulation_count;
    const size_t n = total_link_count_;
    q_pre_ = ZeroedFloat(context_, n);
    qdot_pre_ = ZeroedFloat(context_, n);
    v_root_pre_ = ZeroedFloat(context_, n * 6u);
    grad_q_ = ZeroedFloat(context_, n);
    grad_qdot_ = ZeroedFloat(context_, n);
    grad_link_vel_ = ZeroedFloat(context_, n * 6u);
    grad_mass_step_ = ZeroedFloat(context_, n);
    grad_mass_acc_ = ZeroedFloat(context_, n);
    grad_tau_ = ZeroedFloat(context_, n);
}

void BackwardRunner::ApplyIftContactSeed(const IftContactInputs& inputs,
                                         const float* g, float* grad_qdot_free,
                                         float* grad_b_c) {
    // gradient_mode gate: the default ContactFree path never reaches here, so the
    // committed p02 backward/multistep tests stay byte-unchanged. Only a caller
    // that explicitly opted into ContactIft routes contact rows through IFT.
    if (gradient_mode_ != GradientMode::ContactIft) {
        throw std::runtime_error(
            "nuka::diffsim::BackwardRunner::ApplyIftContactSeed: gradient_mode is "
            "ContactFree (call SetGradientMode(GradientMode::ContactIft) first)");
    }
    if (!ift_) {
        ift_ = std::make_unique<IftRunner>(context_);  // lazy: only on first IFT use
    }
    IftContactGrads grads;
    grads.grad_qdot_free = grad_qdot_free;
    grads.grad_b_c = grad_b_c;
    ift_->Run(inputs, g, grads);
}

void BackwardRunner::ReverseStep(const Tape& tape, uint32_t step,
                                 const BackwardOutputs& out) {
    const articulation::ArticulationDeviceState state = orch_.State();
    const cudaStream_t stream = context_.stream.Native();
    const size_t n = total_link_count_;

    // 1. Snapshot pre-step q/qdot (StepBackward's drive + ABA reverses need the
    //    PRE-step values; StepOnce below overwrites the live q/qdot).
    {
        phi::ScopedDeviceGuard guard(context_.device_id);
        CheckCuda(cudaMemcpyAsync(q_pre_.Data(), state.q, n * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream),
                  "snapshot q_pre");
        CheckCuda(cudaMemcpyAsync(qdot_pre_.Data(), state.qdot, n * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream),
                  "snapshot qdot_pre");
        // Snapshot the PRE-INTEGRATION root spatial velocity. The forward's
        // IntegrateFloatingBaseVelocity (inside StepOnce below) overwrites
        // link_velocity[root] in place (v_pre -> v_post); the root gyroscopic
        // adjoint must linearize at v_pre. Whole-buffer copy (LinkSpatialVel is
        // float[6], no padding) -> v_root_pre_[link*6 .. +6).
        CheckCuda(cudaMemcpyAsync(v_root_pre_.Data(), state.link_velocity,
                                  n * sizeof(articulation::LinkSpatialVel),
                                  cudaMemcpyDeviceToDevice, stream),
                  "snapshot v_root_pre");
    }

    // 2. Re-run the forward of THIS step: regenerates step-j ABA intermediates
    //    (link_xup / articulated_I / u_spatial / accel / qddot ...) into the live
    //    buffers StepBackward reads, and advances the live state to step j+1.
    orch_.StepOnce(tape.ActionSlice(step));

    // 3. Zero grad_tau (StepBackward STAGES dL/dtau here; caller must pre-zero).
    {
        phi::ScopedDeviceGuard guard(context_.device_id);
        CheckCuda(cudaMemsetAsync(grad_tau_.Data(), 0, n * sizeof(float), stream),
                  "zero grad_tau");
    }

    // 4. StepBackward: seeds carried in grad_q_/grad_qdot_/grad_link_vel_ (the
    //    dL/d state' of step j == dL/d state of step j+1), OVERWRITTEN in place
    //    with the pre-step seed for step j-1.
    StepBackwardInputs in = orch_.ReconstructInputs(tape, step);
    in.q_pre = static_cast<const float*>(q_pre_.Data());
    in.qdot_pre = static_cast<const float*>(qdot_pre_.Data());
    in.v_root_pre = static_cast<const float*>(v_root_pre_.Data());

    StepBackwardGrads g;
    g.grad_q_out = static_cast<float*>(grad_q_.Data());
    g.grad_qdot_out = static_cast<float*>(grad_qdot_.Data());
    // grad_target_out: write this step's action gradient DIRECTLY into the output
    // slice for step `step` (StepBackward OVERWRITES it per call; distinct slices
    // per step -> no accumulation needed).
    g.grad_target_out = out.grad_actions +
                        static_cast<size_t>(step) * total_link_count_;
    // grad_mass_out: StepBackward OVERWRITES this per call, so point it at the
    // per-step scratch and accumulate into grad_mass_acc_ afterwards.
    g.grad_mass_out = static_cast<float*>(grad_mass_step_.Data());
    g.grad_tau_out = static_cast<float*>(grad_tau_.Data());
    g.grad_link_velocity_out = static_cast<float*>(grad_link_vel_.Data());

    StepBackward(context_, state, in, g);

    // 6. grad_mass_acc_ += grad_mass_step_ (fixed-order, atomic-free, D1).
    LaunchAdd(context_, static_cast<float*>(grad_mass_acc_.Data()),
              static_cast<const float*>(grad_mass_step_.Data()),
              total_link_count_);
}

void BackwardRunner::Run(const Tape& tape, const CheckpointManager& cm,
                         const BackwardSeeds& seeds, const BackwardOutputs& out) {
    const uint32_t step_count = tape.StepCount();
    const cudaStream_t stream = context_.stream.Native();
    const size_t n = total_link_count_;
    if (out.grad_actions == nullptr || out.grad_mass == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::BackwardRunner::Run: null output buffers");
    }
    if (step_count == 0u) {
        return;
    }

    // -- Seed the loss adjoint on the FINAL (post last-step) state. --
    {
        phi::ScopedDeviceGuard guard(context_.device_id);
        if (seeds.grad_q_final) {
            CheckCuda(cudaMemcpyAsync(grad_q_.Data(), seeds.grad_q_final,
                                      n * sizeof(float), cudaMemcpyDeviceToDevice,
                                      stream),
                      "seed grad_q_final");
        } else {
            CheckCuda(cudaMemsetAsync(grad_q_.Data(), 0, n * sizeof(float), stream),
                      "zero grad_q_final");
        }
        if (seeds.grad_qdot_final) {
            CheckCuda(cudaMemcpyAsync(grad_qdot_.Data(), seeds.grad_qdot_final,
                                      n * sizeof(float), cudaMemcpyDeviceToDevice,
                                      stream),
                      "seed grad_qdot_final");
        } else {
            CheckCuda(cudaMemsetAsync(grad_qdot_.Data(), 0, n * sizeof(float),
                                      stream),
                      "zero grad_qdot_final");
        }
        if (seeds.grad_link_velocity_final) {
            CheckCuda(cudaMemcpyAsync(grad_link_vel_.Data(),
                                      seeds.grad_link_velocity_final,
                                      n * 6u * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream),
                      "seed grad_link_velocity_final");
        } else {
            CheckCuda(cudaMemsetAsync(grad_link_vel_.Data(), 0,
                                      n * 6u * sizeof(float), stream),
                      "zero grad_link_velocity_final");
        }
        // grad_mass accumulator starts at zero (accumulates over all steps).
        CheckCuda(cudaMemsetAsync(grad_mass_acc_.Data(), 0, n * sizeof(float),
                                  stream),
                  "zero grad_mass_acc");
    }

    // -- Allocate the per-window state cache lazily, sized to the K-cadence (or 1
    //    in full-tape mode -- windows are size 1 there). --
    const uint32_t k = (tape.Desc().recompute_on_backward == 0u)
                           ? 1u
                           : tape.Desc().checkpoint_interval;
    if (window_capacity_ < k) {
        window_capacity_ = k;
        win_q_ = phi::Buffer(static_cast<size_t>(k) * n * sizeof(float),
                             phi::MemoryKind::Device);
        win_qdot_ = phi::Buffer(static_cast<size_t>(k) * n * sizeof(float),
                                phi::MemoryKind::Device);
        win_base_pose_ = phi::Buffer(
            static_cast<size_t>(k) * articulation_count_ *
                sizeof(nuka::math::Transform),
            phi::MemoryKind::Device);
        win_link_vel_ = phi::Buffer(
            static_cast<size_t>(k) * n * sizeof(articulation::LinkSpatialVel),
            phi::MemoryKind::Device);
    }

    const articulation::ArticulationDeviceState state = orch_.State();

    // -- Process steps in DESCENDING windows. A window is [base .. last], where
    //    `base` is the nearest checkpoint step <= the window's top step. To
    //    reverse it we restore the base checkpoint, roll forward refilling each
    //    step's PRE-state into the window cache, then reverse the window's steps
    //    in descending order, restoring each step's cached pre-state first. The
    //    SAME ReverseStep inner loop runs in both modes -> bit-identical grads. --
    uint32_t top = step_count;  // exclusive upper bound of the remaining range
    while (top > 0u) {
        const uint32_t window_top = top - 1u;  // highest step still to reverse
        // Find the checkpoint step that begins this window.
        uint32_t base_slot = 0u;
        if (!cm.NearestCheckpointBefore(window_top, &base_slot)) {
            throw std::runtime_error(
                "nuka::diffsim::BackwardRunner::Run: no checkpoint at or before "
                "step " +
                std::to_string(window_top) +
                " (forward must capture checkpoint 0)");
        }
        const uint32_t base_step = cm.StepOf(base_slot);
        const uint32_t window_len = window_top - base_step + 1u;
        if (window_len > window_capacity_) {
            throw std::runtime_error(
                "nuka::diffsim::BackwardRunner::Run: window length exceeds cache "
                "capacity (checkpoint cadence inconsistent)");
        }

        // Restore the base checkpoint -> live state == start of step base_step.
        cm.Restore(base_slot, state, /*lambda_device=*/nullptr);

        // Roll forward through the window, snapshotting the PRE-state of each
        // step (the state at the START of step base_step + w) into cache slot w.
        for (uint32_t w = 0u; w < window_len; ++w) {
            phi::ScopedDeviceGuard guard(context_.device_id);
            float* wq = static_cast<float*>(win_q_.Data()) +
                        static_cast<size_t>(w) * n;
            float* wqd = static_cast<float*>(win_qdot_.Data()) +
                         static_cast<size_t>(w) * n;
            auto* wbp = static_cast<nuka::math::Transform*>(win_base_pose_.Data()) +
                        static_cast<size_t>(w) * articulation_count_;
            auto* wlv =
                static_cast<articulation::LinkSpatialVel*>(win_link_vel_.Data()) +
                static_cast<size_t>(w) * n;
            CheckCuda(cudaMemcpyAsync(wq, state.q, n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cache pre-q");
            CheckCuda(cudaMemcpyAsync(wqd, state.qdot, n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cache pre-qdot");
            CheckCuda(cudaMemcpyAsync(wbp, state.base_pose,
                                      static_cast<size_t>(articulation_count_) *
                                          sizeof(nuka::math::Transform),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cache pre-base_pose");
            CheckCuda(cudaMemcpyAsync(wlv, state.link_velocity,
                                      n * sizeof(articulation::LinkSpatialVel),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cache pre-link_velocity");
            // Advance to the next step's pre-state (no need on the last iter, but
            // harmless; we always need the live state at base_step+w to be the
            // start of that step before caching it -- so step AFTER caching).
            orch_.StepOnce(tape.ActionSlice(base_step + w));
        }

        // Reverse the window's steps in DESCENDING order. For each, restore its
        // cached pre-state, then run the shared inner reverse (which snapshots
        // q_pre/qdot_pre, StepOnce-regens intermediates, StepBackward).
        for (uint32_t wi = window_len; wi-- > 0u;) {
            const uint32_t step = base_step + wi;
            {
                phi::ScopedDeviceGuard guard(context_.device_id);
                const float* wq = static_cast<const float*>(win_q_.Data()) +
                                  static_cast<size_t>(wi) * n;
                const float* wqd = static_cast<const float*>(win_qdot_.Data()) +
                                   static_cast<size_t>(wi) * n;
                const auto* wbp = static_cast<const nuka::math::Transform*>(
                                      win_base_pose_.Data()) +
                                  static_cast<size_t>(wi) * articulation_count_;
                const auto* wlv = static_cast<const articulation::LinkSpatialVel*>(
                                      win_link_vel_.Data()) +
                                  static_cast<size_t>(wi) * n;
                CheckCuda(cudaMemcpyAsync(state.q, wq, n * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream),
                          "restore pre-q");
                CheckCuda(cudaMemcpyAsync(state.qdot, wqd, n * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream),
                          "restore pre-qdot");
                CheckCuda(cudaMemcpyAsync(state.base_pose, wbp,
                                          static_cast<size_t>(articulation_count_) *
                                              sizeof(nuka::math::Transform),
                                          cudaMemcpyDeviceToDevice, stream),
                          "restore pre-base_pose");
                CheckCuda(cudaMemcpyAsync(state.link_velocity, wlv,
                                          n * sizeof(articulation::LinkSpatialVel),
                                          cudaMemcpyDeviceToDevice, stream),
                          "restore pre-link_velocity");
            }
            ReverseStep(tape, step, out);
        }

        top = base_step;  // next window ends just below this checkpoint
    }

    // -- Publish the accumulated mass gradient. --
    {
        phi::ScopedDeviceGuard guard(context_.device_id);
        CheckCuda(cudaMemcpyAsync(out.grad_mass, grad_mass_acc_.Data(),
                                  n * sizeof(float), cudaMemcpyDeviceToDevice,
                                  stream),
                  "publish grad_mass");
    }
    context_.stream.Synchronize();
}

}  // namespace nuka::diffsim
