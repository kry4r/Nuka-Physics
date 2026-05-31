// ---------------------------------------------------------------------------
// nuka::runtime::gpu -- batched articulated-with-contacts GPU step path (T6)
// ---------------------------------------------------------------------------
//
// Orchestrates the validated p01-F T1..T5 kernels into one deterministic
// per-step pipeline. See batched_articulated_world.hpp for the pipeline order
// and the validation rationale. This file owns no new physics -- every stage is
// a call into the already-validated FeatherstoneAba / articulation_contacts /
// articulation_jacobian / articulation_state kernels, wrapped with the canonical
// perf tags and reusing one set of scratch buffers.
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_articulated_world.hpp"

#include "core/perf/timing.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/featherstone_aba.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nuka::runtime::gpu {

namespace {

namespace articulation = nuka::runtime::articulation;
using nuka::math::Transform;
using nuka::math::Vec3;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

BatchedArticulatedWorld::BatchedArticulatedWorld(
    const phi::DeviceContext& context,
    const articulation::ArticulationHostState& host,
    const std::vector<articulation::FootShape>& feet,
    uint32_t max_dof,
    float ground_height)
    : context_(context) {
    articulation_count_ = host.ArticulationCount();
    if (articulation_count_ == 0u) {
        throw std::runtime_error(
            "BatchedArticulatedWorld: host has no articulations");
    }
    if (max_dof == 0u || max_dof > articulation::kMaxContactSolverDof) {
        throw std::runtime_error(
            "BatchedArticulatedWorld: max_dof out of range");
    }
    // Review fix #1 (float-neutral hardening): the M / M^-1 / Jacobian scratch is
    // sized as a flat [.. * max_dof * max_dof] / [.. * max_dof] tile, and every
    // T1..T5 kernel indexes it with the per-articulation DOF stride. That is only
    // correct if max_dof EQUALS each articulation's actual DOF count -- a mere
    // `<= kMaxContactSolverDof` bound would silently produce a wrong M^-1 on a
    // future heterogeneous batch (articulations with differing DOF). Assert exact
    // equality for every articulation here, at construction, where both numbers
    // are in scope.
    for (uint32_t articulation_index = 0u; articulation_index < host.ArticulationCount();
         ++articulation_index) {
        const uint32_t actual_dof =
            articulation::ArticulationDofCount(host, articulation_index);
        if (actual_dof != max_dof) {
            throw std::runtime_error(
                "BatchedArticulatedWorld: max_dof (" + std::to_string(max_dof) +
                ") must equal every articulation's DOF count; articulation " +
                std::to_string(articulation_index) + " has " +
                std::to_string(actual_dof) +
                " DOF. Heterogeneous-DOF batches are not supported by the shared "
                "max_dof-strided M/Jacobian scratch.");
        }
    }
    // env == articulation in the 1-articulation-per-env replication design.
    env_count_ = articulation_count_;
    total_link_count_ = host.TotalLinkCount();
    // Precondition shared by T1..T5: every articulation has the same link count
    // (replicated tiles), so base_link_count is total / env_count.
    if (total_link_count_ % env_count_ != 0u) {
        throw std::runtime_error(
            "BatchedArticulatedWorld: total_link_count not divisible by env_count");
    }
    base_link_count_ = total_link_count_ / env_count_;
    max_dof_ = max_dof;
    foot_count_ = static_cast<uint32_t>(feet.size());
    if (foot_count_ > articulation::kMaxFootContactsPerEnv) {
        throw std::runtime_error(
            "BatchedArticulatedWorld: foot_count exceeds kMaxFootContactsPerEnv");
    }
    slot_count_ = env_count_ * articulation::kMaxFootContactsPerEnv;
    ground_height_ = ground_height;

    // Upload the device state (q, qdot, link_pose [rest], topology, ...).
    device_ = articulation::UploadArticulationState(context_, host);

    const size_t tile = static_cast<size_t>(max_dof_) * max_dof_;

    // Allocate every scratch buffer ONCE; reused each Step().
    feet_ = phi::Buffer(static_cast<size_t>(foot_count_) * sizeof(articulation::FootShape),
                        phi::MemoryKind::Device);
    if (foot_count_ > 0u) {
        feet_.CopyFromHost(feet.data(),
                           static_cast<size_t>(foot_count_) * sizeof(articulation::FootShape));
    }
    composite_ = phi::Buffer(
        static_cast<size_t>(total_link_count_) * sizeof(articulation::LinkSpatialInertia),
        phi::MemoryKind::Device);
    m_ = phi::Buffer(static_cast<size_t>(articulation_count_) * tile * sizeof(float),
                     phi::MemoryKind::Device);
    m_inv_ = phi::Buffer(static_cast<size_t>(articulation_count_) * tile * sizeof(float),
                         phi::MemoryKind::Device);
    world_pose_ = phi::Buffer(static_cast<size_t>(total_link_count_) * sizeof(Transform),
                              phi::MemoryKind::Device);
    contact_link_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(uint32_t),
                                phi::MemoryKind::Device);
    contact_point_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(Vec3),
                                 phi::MemoryKind::Device);
    contact_normal_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(Vec3),
                                  phi::MemoryKind::Device);
    contact_depth_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(float),
                                 phi::MemoryKind::Device);
    contact_count_ = phi::Buffer(static_cast<size_t>(env_count_) * sizeof(uint32_t),
                                 phi::MemoryKind::Device);
    tangent1_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(Vec3),
                            phi::MemoryKind::Device);
    tangent2_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(Vec3),
                            phi::MemoryKind::Device);
    const size_t jac_len = static_cast<size_t>(slot_count_) * max_dof_;
    jac_normal_ = phi::Buffer(jac_len * sizeof(float), phi::MemoryKind::Device);
    jac_tangent1_ = phi::Buffer(jac_len * sizeof(float), phi::MemoryKind::Device);
    jac_tangent2_ = phi::Buffer(jac_len * sizeof(float), phi::MemoryKind::Device);
    meff_normal_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(float),
                               phi::MemoryKind::Device);
    meff_tangent1_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(float),
                                 phi::MemoryKind::Device);
    meff_tangent2_ = phi::Buffer(static_cast<size_t>(slot_count_) * sizeof(float),
                                 phi::MemoryKind::Device);
    rows_ = phi::Buffer(
        static_cast<size_t>(slot_count_) * sizeof(articulation::ArticulatedContactRow),
        phi::MemoryKind::Device);
    lambda_ = phi::Buffer(static_cast<size_t>(slot_count_) * 3u * sizeof(float),
                          phi::MemoryKind::Device);

    // Zero the persistent lambda buffer for a clean cold start (determinism +
    // a defined warm-start seed on the first step). Also zero the contact-point
    // readback buffer so a CONTACT_POINTS query BEFORE the first Step() returns
    // defined (all-zero == no contacts) data rather than uninitialized memory;
    // every Step() overwrites it.
    phi::ScopedDeviceGuard guard(context_.device_id);
    CheckCuda(cudaMemsetAsync(lambda_.Data(), 0,
                              static_cast<size_t>(slot_count_) * 3u * sizeof(float),
                              context_.stream.Native()),
              "BatchedArticulatedWorld lambda memset");
    CheckCuda(cudaMemsetAsync(contact_point_.Data(), 0,
                              static_cast<size_t>(slot_count_) * sizeof(Vec3),
                              context_.stream.Native()),
              "BatchedArticulatedWorld contact_point memset");
    context_.stream.Synchronize();
}

void BatchedArticulatedWorld::Step(const BatchedArticulatedStepParams& params) {
    const articulation::ArticulationDeviceState state = device_.View();
    const cudaStream_t stream = context_.stream.Native();
    const uint32_t slot_count = slot_count_;
    (void)slot_count;

    core::perf::ScopedCudaTimer step_timer(perf_, "step_total", stream);

    // -- 1. Position drives -> tau, 2. ABA accelerations -> qddot. -----------
    // The coarse `featherstone_aba` bucket is kept (canonical tag); the detail
    // scopes split it into the drive pass and the 3-pass ABA accel solve so the
    // hotspot wave can attribute the dominant cost. Detail scopes are no-ops
    // (no CUDA events / syncs) unless Perf().SetDetailEnabled(true).
    {
        NUKA_CUDA_TIME(perf_, "featherstone_aba", stream);
        if (params.drive_targets != nullptr && params.drive_stiffness != nullptr &&
            params.drive_damping != nullptr && params.drive_force_limits != nullptr) {
            NUKA_CUDA_TIME_DETAIL(perf_, "aba_apply_drives", stream);
            // defer_velocity_damping=true: the drive emits only the Kp stiffness
            // torque; the -Kd*qdot damping is applied IMPLICITLY in the contact
            // solve (general implicit joint damping via (M+dt*C)^-1), which is
            // unconditionally stable -- the explicit form buzzed at the native dt
            // for soft gains (instability ~ dt*Kd/m_eff, m_eff shrunk by contact).
            articulation::FeatherstoneAba::ApplyPositionDrives(
                context_, state, params.drive_targets, params.drive_stiffness,
                params.drive_damping, params.drive_force_limits,
                /*defer_velocity_damping=*/true);
        }
        {
            NUKA_CUDA_TIME_DETAIL(perf_, "aba_compute_accelerations", stream);
            articulation::FeatherstoneAba::ComputeAccelerations(context_, state,
                                                                params.gravity_z);
        }
    }

    // -- 3. Velocity-integrate qdot += qddot*dt. ----------------------------
    {
        NUKA_CUDA_TIME(perf_, "integrator", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "integrate_velocity", stream);
        articulation::FeatherstoneAba::IntegrateVelocity(context_, state, params.dt);
        // T8a: floating-base velocity integrate (link_velocity[root] += real
        // base accel * dt, where the real accel subtracts the gravity seed from
        // the apparent accel stored in link_acceleration[root]). No-op for
        // fixed/kinematic roots, so byte-safe for fixed-base scenes; placed
        // alongside the joint velocity-integrate (pre-contact half).
        articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, state,
                                                                     params.dt,
                                                                     params.gravity_z);
    }

    // -- 4. Refresh world poses, copy into state.link_pose. -----------------
    // ComputeContactChainJacobians reads state.link_pose; UploadArticulationState
    // leaves it at the static cooked rest pose, so it must be updated from the
    // current q each step for the contact geometry to be correct.
    {
        NUKA_CUDA_TIME(perf_, "contact_generation", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "update_world_poses", stream);
        articulation::UpdateWorldLinkPoses(
            context_, state, static_cast<Transform*>(world_pose_.Data()));
    }
    {
        NUKA_CUDA_TIME(perf_, "buffer_mgmt", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "link_pose_refresh_copy", stream);
        phi::ScopedDeviceGuard guard(context_.device_id);
        CheckCuda(cudaMemcpyAsync(
                      state.link_pose, world_pose_.Data(),
                      static_cast<size_t>(total_link_count_) * sizeof(Transform),
                      cudaMemcpyDeviceToDevice, stream),
                  "BatchedArticulatedWorld link_pose refresh");
    }

    // -- 5. Detect foot-vs-ground contacts. ---------------------------------
    {
        NUKA_CUDA_TIME(perf_, "contact_generation", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "detect_foot_contacts", stream);
        articulation::DetectFootGroundContacts(
            context_, static_cast<const Transform*>(world_pose_.Data()),
            static_cast<const articulation::FootShape*>(feet_.Data()), foot_count_,
            env_count_, base_link_count_, ground_height_,
            static_cast<uint32_t*>(contact_link_.Data()),
            static_cast<Vec3*>(contact_point_.Data()),
            static_cast<Vec3*>(contact_normal_.Data()),
            static_cast<float*>(contact_depth_.Data()),
            static_cast<uint32_t*>(contact_count_.Data()));
    }

    // -- 6. Tangent basis + chain Jacobians (normal, t1, t2). ---------------
    {
        NUKA_CUDA_TIME(perf_, "row_builder", stream);
        {
            NUKA_CUDA_TIME_DETAIL(perf_, "contact_tangent_basis", stream);
            articulation::ComputeContactTangentBasis(
                context_, static_cast<const uint32_t*>(contact_link_.Data()),
                static_cast<const Vec3*>(contact_normal_.Data()), env_count_,
                static_cast<Vec3*>(tangent1_.Data()),
                static_cast<Vec3*>(tangent2_.Data()));
        }
        {
            NUKA_CUDA_TIME_DETAIL(perf_, "chain_jacobians", stream);
            articulation::ComputeContactChainJacobians(
                context_, state,
                static_cast<const uint32_t*>(contact_link_.Data()),
                static_cast<const Vec3*>(contact_point_.Data()),
                static_cast<const Vec3*>(contact_normal_.Data()),
                slot_count_, max_dof_, static_cast<float*>(jac_normal_.Data()));
            articulation::ComputeContactChainJacobians(
                context_, state,
                static_cast<const uint32_t*>(contact_link_.Data()),
                static_cast<const Vec3*>(contact_point_.Data()),
                static_cast<const Vec3*>(tangent1_.Data()),
                slot_count_, max_dof_, static_cast<float*>(jac_tangent1_.Data()));
            articulation::ComputeContactChainJacobians(
                context_, state,
                static_cast<const uint32_t*>(contact_link_.Data()),
                static_cast<const Vec3*>(contact_point_.Data()),
                static_cast<const Vec3*>(tangent2_.Data()),
                slot_count_, max_dof_, static_cast<float*>(jac_tangent2_.Data()));
        }
    }

    // -- 7. Joint-space inertia M and its inverse. --------------------------
    {
        NUKA_CUDA_TIME(perf_, "row_builder", stream);
        {
            NUKA_CUDA_TIME_DETAIL(perf_, "crba_inertia_m", stream);
            // Fold dt*C into the joint diagonals so the factored inverse below is
            // (M + dt*C)^-1 -- the implicit-joint-damping admittance consumed by the
            // contact solve (drive_damping = the per-DOF c_j; deferred from the drive).
            articulation::ComputeArticulationInertiaM(
                context_, state, max_dof_,
                static_cast<articulation::LinkSpatialInertia*>(composite_.Data()),
                static_cast<float*>(m_.Data()),
                params.drive_damping, params.dt);
        }
        {
            NUKA_CUDA_TIME_DETAIL(perf_, "factor_inertia_m_inv", stream);
            articulation::FactorArticulationInertiaM(
                context_, state, max_dof_,
                static_cast<const float*>(m_.Data()),
                static_cast<float*>(m_inv_.Data()));
        }
    }

    // -- 8. Effective mass for normal + t1 + t2 rows. -----------------------
    {
        NUKA_CUDA_TIME(perf_, "row_builder", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "contact_effective_mass", stream);
        articulation::ComputeContactEffectiveMass(
            context_, state, static_cast<const uint32_t*>(contact_link_.Data()),
            static_cast<const float*>(jac_normal_.Data()),
            static_cast<const float*>(m_inv_.Data()), slot_count_, max_dof_,
            static_cast<float*>(meff_normal_.Data()));
        articulation::ComputeContactEffectiveMass(
            context_, state, static_cast<const uint32_t*>(contact_link_.Data()),
            static_cast<const float*>(jac_tangent1_.Data()),
            static_cast<const float*>(m_inv_.Data()), slot_count_, max_dof_,
            static_cast<float*>(meff_tangent1_.Data()));
        articulation::ComputeContactEffectiveMass(
            context_, state, static_cast<const uint32_t*>(contact_link_.Data()),
            static_cast<const float*>(jac_tangent2_.Data()),
            static_cast<const float*>(m_inv_.Data()), slot_count_, max_dof_,
            static_cast<float*>(meff_tangent2_.Data()));
    }

    // -- 9. Assemble the contact rows. --------------------------------------
    {
        NUKA_CUDA_TIME(perf_, "row_builder", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "assemble_rows", stream);
        articulation::AssembleArticulatedContactRows(
            context_, state,
            static_cast<const uint32_t*>(contact_link_.Data()),
            static_cast<const Vec3*>(contact_normal_.Data()),
            static_cast<const float*>(contact_depth_.Data()),
            static_cast<const float*>(meff_normal_.Data()),
            static_cast<const float*>(meff_tangent1_.Data()),
            static_cast<const float*>(meff_tangent2_.Data()),
            env_count_, max_dof_,
            static_cast<articulation::ArticulatedContactRow*>(rows_.Data()));
    }

    // -- 10. Solve the rows (fused apply Delta_qdot into qdot; lambda carry). -
    {
        NUKA_CUDA_TIME(perf_, "row_solver", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "solve_contact_rows", stream);
        // joint_damping = params.drive_damping: with m_inv_ = (M+dt*C)^-1 above, the
        // solve applies implicit joint damping (qdot -= dt*(M+dt*C)^-1*(C*qdot)) and
        // the contact rows consistently before the position integrate.
        articulation::SolveArticulatedContactRows(
            context_, state,
            static_cast<const articulation::ArticulatedContactRow*>(rows_.Data()),
            static_cast<const float*>(jac_normal_.Data()),
            static_cast<const float*>(jac_tangent1_.Data()),
            static_cast<const float*>(jac_tangent2_.Data()),
            static_cast<const float*>(m_inv_.Data()), env_count_, max_dof_,
            params.dt, static_cast<float*>(lambda_.Data()),
            params.friction_coefficient, params.baumgarte_max_velocity,
            params.drive_damping);
    }

    // -- 11. Position-integrate q += qdot*dt. -------------------------------
    {
        NUKA_CUDA_TIME(perf_, "integrator", stream);
        NUKA_CUDA_TIME_DETAIL(perf_, "integrate_position", stream);
        articulation::FeatherstoneAba::IntegratePosition(context_, state, params.dt);
        // T8a: floating-base pose integrate (advance base_pose[articulation]).
        // No-op for fixed/kinematic roots; placed alongside the joint
        // position-integrate (post-contact half) so the base pose reflects the
        // post-contact velocity, matching the joint DOFs.
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, state,
                                                                 params.dt);
    }
}

void BatchedArticulatedWorld::Download(articulation::ArticulationHostState* host) const {
    articulation::DownloadArticulationState(device_, host);
}

std::vector<float> BatchedArticulatedWorld::DownloadLambda() const {
    std::vector<float> out(static_cast<size_t>(slot_count_) * 3u);
    lambda_.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

std::vector<float> BatchedArticulatedWorld::DownloadContactDepth() const {
    std::vector<float> out(slot_count_);
    contact_depth_.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

std::vector<uint32_t> BatchedArticulatedWorld::DownloadContactLink() const {
    std::vector<uint32_t> out(slot_count_);
    contact_link_.CopyToHost(out.data(), out.size() * sizeof(uint32_t));
    return out;
}

std::vector<articulation::ArticulatedContactRow>
BatchedArticulatedWorld::DownloadRows() const {
    std::vector<articulation::ArticulatedContactRow> out(slot_count_);
    rows_.CopyToHost(out.data(), out.size() * sizeof(articulation::ArticulatedContactRow));
    return out;
}

} // namespace nuka::runtime::gpu
