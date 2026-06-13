// ---------------------------------------------------------------------------
// DEPRECATED(M9): src/runtime/coresident/ is deleted WHOLE at M9. KEPT ALIVE
// through M8 (controller R1/R2): BatchedUnifiedWorld is the union parity ORACLE
// (h1_union_parity) and consumes the BatchedSceneTemplate the M7 .nks cook now
// produces (the factory that used to author it is DELETED). No NEW consumers.
// ---------------------------------------------------------------------------
// nuka::runtime::coresident::BatchedUnifiedWorld -- implementation (v0.8 P2.2).
// ---------------------------------------------------------------------------
// P2.1 scope: per-env MOVABLE RIGID BODIES under gravity (free-fall, no contact,
// no articulation). The step loop is the SAME stage order as
// UnifiedCoResidentStepper / BatchedArticulatedWorld -- velocity stage (gravity
// kick) -> contact phase (EMPTY in P2.1) -> position stage.
//
// P2.2 (THIS): the batched box<->static-ground contact phase fills the previously
// EMPTY slot. Each env's local body 0 (a BOX) rests on ONE static +Z plane. For
// every env, the production narrowphase (BuildContactManifolds, box x plane) emits
// a manifold; EmitCompliantContactRows (condim=1) APPENDS condim=1 normal rows to a
// SHARED rows/sides buffer; the appended rows' body indices are OVERWRITTEN (rigid
// side -> BodyIndex(e,0), static side -> kInvalidBodyIndex). After all envs, ONE
// UnifiedSolve over the concatenated rows + the full env-major bodies advances every
// env at once. This is PURE PLUMBING: the PrimParams, the cfg{64,0,0,0}, the
// condim=1 inputs, and the body-index wiring are copied verbatim from
// UnifiedCoResidentStepper::Step()'s box<->ground branch (lines ~437-614), so an N=1
// BatchedUnifiedWorld is BYTE-IDENTICAL to that single-instance reference. NO new
// physics, NO solver change. The cross-env body-id disjointness (BodyIndex(e,*)) is
// exactly what makes the row-scheduler's greedy-in-index graph coloring partition
// the per-env rows into N independent solves (see row_scheduler.cu RowsConflict).
// ---------------------------------------------------------------------------

#include "runtime/coresident/batched_unified_world.hpp"

#include "core/perf/perf_recorder.hpp"         // PerfRecorder (host wall-clock attribution)
#include "collision/analytical_manifold.hpp"   // amf::PrimParams / BuildPrimFrame
#include "collision/candidate_pair.hpp"        // CandidatePair / CollidableRef
#include "collision/contact_stream_driver.hpp" // BuildContactManifolds / ResolvedShape (ground path)
#include "collision/convex_narrowphase.hpp"    // cvx::ConvexHullView (cup hull seam)
#include "collision/gpu/narrowphase_grasp.hpp" // P2.4c: batched GPU grasp narrowphase launcher
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "constraint/contact_row_sides.hpp"    // ContactRowSides
#include "constraint/reaction_provider.hpp"    // ReactionProviderKind
#include "constraint/row.hpp"                  // kInvalidBodyIndex / row_flags
#include "constraint/row_articulation_refs.hpp"  // RowArticulationRefs / RowArticulationSide
#include "constraint/row_buffers.hpp"          // RowBuffers / RowJacobian6
#include "constraint/row_builder.hpp"          // EmitCompliantContactRows / inputs
#include "phi/buffer_legacy.hpp"
#include "phi/buffer_transfer.hpp"             // UploadVector
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses / ArticulationDofCount / ArticulationJointDofCount
#include "runtime/articulation/articulation_drives.hpp"    // LaunchApplyTorqueDriveKernels
#include "runtime/articulation/articulation_jacobian.hpp"  // ComputeContactChainJacobians + M/M^-1
#include "runtime/articulation/featherstone_aba.hpp"
#include "scene/canonical_types.hpp"           // scene::ShapeType
#include "solver/solver_config.hpp"            // SolverConfig
#include "solver/unified_solve.hpp"            // UnifiedSolve / SolveContext

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::runtime::coresident {

namespace {

// ---------------------------------------------------------------------------
// P2.4a perf-gate instrumentation: a HOST WALL-CLOCK RAII timer that feeds a
// PerfRecorder under `tag` on scope exit. CHRONO (not ScopedCudaTimer) on purpose:
// the bracketed stages already Synchronize/CopyToHost internally, so the cost being
// attributed is the HOST round-trip wall time -- a CUDA-event timer measures only GPU
// stream time and would miss the per-env download/upload storm entirely. Pure host C++:
// no CUDA calls, no float math, no extra synchronizes -> ADDITIVE + physics-neutral, so
// the 16 byte-exact correctness gates are unaffected (a chrono read between existing
// calls changes nothing the solver sees). RAII accumulation lets a single tag span
// disjoint code regions across functions (e.g. scatter_integrate = the scatter loop +
// the Step() position integrate). Always-on (the overhead is a steady_clock read pair).
class ScopedWallTimer {
public:
    ScopedWallTimer(core::perf::PerfRecorder& recorder, const char* tag)
        : recorder_(recorder),
          tag_(tag),
          start_(std::chrono::steady_clock::now()) {}
    ~ScopedWallTimer() {
        const auto end = std::chrono::steady_clock::now();
        const double us =
            std::chrono::duration<double, std::micro>(end - start_).count();
        recorder_.AddSample(tag_, us);
    }
    ScopedWallTimer(const ScopedWallTimer&) = delete;
    ScopedWallTimer& operator=(const ScopedWallTimer&) = delete;

private:
    core::perf::PerfRecorder& recorder_;
    const char* tag_;
    std::chrono::steady_clock::time_point start_;
};

namespace amf = nuka::collision::amf;
namespace articulation = nuka::runtime::articulation;
using nuka::collision::CandidatePair;
using nuka::collision::ResolvedShape;
using nuka::collision::ShapeResolver;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ContactRowSides;
using nuka::constraint::ReactionProviderKind;
using nuka::constraint::RowArticulationRefs;
using nuka::constraint::RowArticulationSide;
using nuka::constraint::RowBuffers;
using nuka::constraint::RowJacobian6;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

// REPLICATED from UnifiedCoResidentStepper's anon namespace (file-static there), so
// this TU does NOT depend on the protected stepper. Byte-for-byte the same prims.

// A box PrimParams from the body pose + per-axis half-extents (the cup's flat-bottom
// table-contact proxy -- the stable C3b BoxPlane handler the W1a box<->ground gate
// uses). Mirrors UnifiedCoResidentStepper::BoxPrimXYZ.
amf::PrimParams BoxPrim(const Transform& pose, const Vec3& half_extents) {
    amf::PrimParams p;
    p.half_extents = half_extents;
    p.frame = amf::BuildPrimFrame(pose);  // bakes the box orientation into cx/cy/cz.
    return p;
}

// A static ground PrimParams whose plane normal is world +Z (C3b BoxPlane reads the
// normal from frame.cy). Mirrors UnifiedCoResidentStepper::GroundPrim.
amf::PrimParams GroundPrim(float height) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = Vec3{0.0f, 0.0f, 1.0f};   // plane normal = world +Z
    p.frame.cz = Vec3{0.0f, -1.0f, 0.0f};  // right-handed: cx x cy = cz
    p.frame.t = Vec3{0.0f, 0.0f, height};
    return p;
}

// The box collidable (RigidInvMass -- the movable rigid side). Mirrors MakeBoxRef.
CollidableRef MakeBoxRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::RigidBody;
    ref.react = ReactionProviderKind::RigidInvMass;
    ref.handle = handle;
    return ref;
}

// The static ground collidable (StaticWorld -> StaticNull reaction: invM=0, no-op
// apply). Mirrors MakeGroundRef.
CollidableRef MakeGroundRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = handle;
    return ref;
}

// Per-env distinct broadphase handles for the box + ground. These are ONLY used by
// the per-env resolver lambda (the row body indices, the thing the solve + coloring
// actually key on, are overwritten to BodyIndex(e,0) / kInvalidBodyIndex below).
constexpr uint32_t kBoxHandle = 9000u;     // distinct from the static ground handle.
constexpr uint32_t kGroundHandle = 8000u;  // distinct from the box handle.

// ===========================================================================
// P2.3a GRASP HELPERS -- REPLICATED BYTE-FOR-BYTE from the protected
// UnifiedCoResidentStepper anon namespace (so this TU does NOT depend on the
// stepper). Drift from those file-static bodies is the #1 risk; the A2 gate (N=1
// batched == real StepGrasp) catches any divergence. Copy verbatim; do not edit.
// ===========================================================================

// NOTE (P2.4c): the host FootSpherePrim helper was removed -- the grasp narrowphase now
// builds the fingertip sphere prim INSIDE the GPU kernel (narrowphase_grasp.cu), from the
// per-slot fingertip world center + radius. The ground path (box x plane) never used it.
// G1b RE-ADDS it for the FOOT-sphere x ground-plane HOST narrowphase (the union's feet
// rows; ~4 spheres/env on the trivial C3b SpherePlane -- brief §3.3 keeps this on the
// host; a GPU foot kernel is the NAMED G1f deferral). Byte-for-byte the oracle's helper
// (unified_coresident_stepper.cpp ~:171).
amf::PrimParams FootSpherePrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;  // identity rotation; sphere is rotation-invariant.
    return p;
}

}  // namespace

BatchedUnifiedWorld::BatchedUnifiedWorld(
    const phi::DeviceContext& context,
    const BatchedSceneTemplate& scene_template,
    uint32_t env_count,
    float gravity_z,
    float dt)
    : context_(context),
      env_count_(env_count),
      bodies_per_env_(
          static_cast<uint32_t>(scene_template.bodies_per_env.size())),
      gravity_z_(gravity_z),
      dt_(dt),
      has_ground_(scene_template.has_ground),
      box_half_extent_(scene_template.box_half_extent),
      ground_height_(scene_template.ground_height),
      has_grasp_(scene_template.has_grasp),
      cup_local_index_(scene_template.cup_local_index),
      friction_mu_(scene_template.friction_mu),
      condim_(scene_template.condim),
      has_feet_(scene_template.has_feet),
      feet_(scene_template.feet),
      foot_ground_(scene_template.ground),
      foot_mu_(scene_template.foot_mu),
      has_table_(scene_template.has_table),
      table_height_(scene_template.table_height),
      table_mu_(scene_template.table_mu),
      table_broadphase_id_(scene_template.table_broadphase_id),
      cup_table_proxy_half_(scene_template.cup_table_proxy_half),
      cup_table_proxy_offset_(scene_template.cup_table_proxy_offset),
      cup_table_proxy_id_(scene_template.cup_table_proxy_id),
      reset_jitter_x_(scene_template.reset_jitter_x),
      reset_jitter_y_(scene_template.reset_jitter_y),
      fingertips_(scene_template.fingertips),
      grasp_cup_(scene_template.cup) {
    // Replicate the per-env body template into the env-major SoA: env e's bodies
    // occupy [e*k, (e+1)*k). This is the per-env body-id offset the batched
    // UnifiedSolve relies on (cross-env body ids are disjoint -> the colored solve
    // is N independent solves). Per-env initial-condition perturbation is applied
    // by the caller through BodyMut() after construction.
    bodies_.reserve(static_cast<size_t>(env_count_) * bodies_per_env_);
    for (uint32_t e = 0u; e < env_count_; ++e) {
        for (uint32_t i = 0u; i < bodies_per_env_; ++i) {
            bodies_.push_back(scene_template.bodies_per_env[i]);
        }
    }

    // ----- G1b UNION honesty: feet ride the grasp branch (they share the env's
    // articulation). A has_feet template without has_grasp would be SILENTLY inert --
    // reject it LOUDLY at construction, never a no-op scene.
    if (has_feet_ && !has_grasp_) {
        throw std::runtime_error(
            "BatchedUnifiedWorld: has_feet requires has_grasp (the feet contact the "
            "ground through the env's articulation; a feet-only template would be "
            "silently inert)");
    }

    // ----- G1d UNION honesty: the table rides the grasp branch (the cup lives
    // there). A has_table template without has_grasp would be SILENTLY inert --
    // reject it LOUDLY at construction, never a no-op scene. The per-env live
    // toggle starts at has_table (the oracle's table_enabled_ = has_table).
    if (has_table_ && !has_grasp_) {
        throw std::runtime_error(
            "BatchedUnifiedWorld: has_table requires has_grasp (the cup x table pair "
            "supports the grasp cup; a table-only template would be silently inert)");
    }
    table_enabled_.assign(env_count_, has_table_ ? uint8_t{1} : uint8_t{0});

    // ----- P2.3a/P2.3b: stand up ONE gripper articulation device PER ENV (inert unless
    // grasp). Each env gets its OWN device copy of the env-invariant proto so per-env grasp
    // dynamics (drive / ABA / IntegrateVelocity / scatter) are independent. The host_proto
    // mirror is kept for the per-step InverseInertia / FootChainJ re-uploads (the oracle's
    // `live`). The constant grip torque + force limits are uploaded ONCE + re-applied to tau
    // each step (idempotent) -- they are SHARED across envs (the proto is env-invariant).
    // Mirrors the UnifiedCoResidentStepper grasp constructor exactly per env, so an N=1
    // grasp world matches the StepGrasp oracle and env e (N>1) matches its own N=1 run.
    if (has_grasp_) {
        gripper_proto_ = scene_template.gripper_proto;
        dof_stride_ = articulation::ArticulationDofCount(gripper_proto_, 0u);
        // G0 honesty: the contact-solve spine's factorization / working storage
        // is sized kMaxArticulationDof. Reject a bigger articulation HERE, at
        // construction, with a clear error (the pre-G0 path silently truncated
        // everything past DOF 18 -- welded joints, zero M^-1 coupling).
        if (dof_stride_ > articulation::kMaxArticulationDof) {
            throw std::runtime_error(
                "BatchedUnifiedWorld: articulation DOF count (" +
                std::to_string(dof_stride_) + ") exceeds kMaxArticulationDof (" +
                std::to_string(articulation::kMaxArticulationDof) +
                "); the contact-solve spine cannot factor/solve it");
        }
        root_link_ = gripper_proto_.articulation_link_offset[0];
        base_dof_ =
            articulation::ArticulationJointDofCount(gripper_proto_.joint_type[root_link_]);
        link_count_ = gripper_proto_.TotalLinkCount();   // per-env (single-articulation) links.
        base_link_count_ = link_count_;                  // replica stride in the consolidated state.

        // G1b: validate the authored foot spheres against the proto (a foot on a
        // non-existent link would read out-of-range FK poses -- fail LOUDLY here).
        if (has_feet_) {
            for (const CoResidentFootSphere& fs : feet_) {
                if (fs.link >= link_count_) {
                    throw std::runtime_error(
                        "BatchedUnifiedWorld: foot sphere link " +
                        std::to_string(fs.link) + " out of range (proto has " +
                        std::to_string(link_count_) + " links)");
                }
            }
        }

        // ----- P2.4b: ONE consolidated env-major device-resident multi-gripper state -----
        // REPLACES the P2.3b per-env std::vector<ArticulationDeviceBuffers> (the per-finger-
        // per-env upload+sync+copyback STORM). ReplicateArticulationHostState tiles the proto
        // into env_count_ independent replicas (replica e's links at [e*base_link_count_,
        // (e+1)*base_link_count_), articulation_count == env_count_, link_to_articulation /
        // articulation_link_offset per-replica offset), uploaded ONCE -> articulation_count==N.
        // This is BatchedArticulatedWorld::device_'s layout VERBATIM, which is what makes the
        // already-per-articulation-batched kernels advance all N grippers in ONE launch each.
        const articulation::ArticulationHostState env_major_host =
            articulation::ReplicateArticulationHostState(gripper_proto_, env_count_);
        env_device_ = articulation::UploadArticulationState(context_, env_major_host);
        // The replicated proto -> uniform dof_stride / base_link_count_ across envs, which is
        // WHAT makes the env-major M^-1 (@ e*dof_stride^2) / qdot (@ e*dof_stride) tiling valid
        // (the row solver / batched CRBA index those buffers by art_index = e). Assert the
        // consolidated device actually carries env_count_ articulations of base_link_count_
        // links each, so the tiling precondition BITES if a future change broke replication.
        assert(env_device_.articulation_count == env_count_ &&
               env_device_.total_link_count == env_count_ * base_link_count_ &&
               "env-major grasp tiling requires env_count_ uniform replicas of the proto");

        // Per-link grip torque / force limits, REPLICATED across all N envs (the drive kernel
        // reads them by the SAME global link index the consolidated state uses, length
        // total_link_count). One env's per-link torque == scene_template.grip_torque resized to
        // base_link_count_; tile it env_count_ times so global link e*base_link_count_+l reads
        // env e's (identical) value.
        std::vector<float> torque_one = scene_template.grip_torque;
        torque_one.resize(base_link_count_, 0.0f);
        std::vector<float> limits_one = scene_template.drive_force_limits;
        limits_one.resize(base_link_count_, 0.0f);
        std::vector<float> torque(static_cast<size_t>(env_count_) * base_link_count_, 0.0f);
        std::vector<float> limits(static_cast<size_t>(env_count_) * base_link_count_, 0.0f);
        for (uint32_t e = 0u; e < env_count_; ++e) {
            std::copy(torque_one.begin(), torque_one.end(),
                      torque.begin() + static_cast<std::ptrdiff_t>(
                                           static_cast<size_t>(e) * base_link_count_));
            std::copy(limits_one.begin(), limits_one.end(),
                      limits.begin() + static_cast<std::ptrdiff_t>(
                                           static_cast<size_t>(e) * base_link_count_));
        }
        grip_torque_dev_ = nuka::phi::UploadVector(torque);
        grip_limits_dev_ = nuka::phi::UploadVector(limits);

        // ----- A1 RL substrate: the LIVE per-step action drive buffer. -----------------
        // DEFAULT = a byte-identical copy of the replicated template grip torque, so a world
        // that never calls SetActions drives EXACTLY the constant grip the P2.3/P2.4 gates
        // proved -> GATE-1 byte-identity. SetActions overwrites env e's finger DOFs in this
        // env-major per-link buffer; the drive call points HERE (not grip_torque_dev_).
        action_torque_host_ = torque;  // env_count_*base_link_count_, env-major per-link.
        action_torque_dev_ = nuka::phi::UploadVector(action_torque_host_);
        // The gripper joint-DOF -> articulation-LOCAL link map (the inverse of DofIndexOf for
        // the single-DOF finger joints). dof_to_link_[DofIndexOf(link)] = link for every link
        // with a non-zero joint DOF count; a Fixed root contributes nothing. dof_stride_-wide.
        dof_to_link_.assign(dof_stride_, 0u);
        for (uint32_t l = root_link_; l < base_link_count_; ++l) {
            const uint32_t ndof =
                articulation::ArticulationJointDofCount(gripper_proto_.joint_type[l]);
            if (ndof == 0u) continue;  // Fixed joint -> no DOF column.
            const uint32_t d = DofIndexOf(l);
            for (uint32_t k = 0u; k < ndof && (d + k) < dof_stride_; ++k)
                dof_to_link_[d + k] = l;  // single-DOF finger -> one column == one link.
        }
        // The reset template = the per-env cup IC the scene was seeded with (template body
        // cup_local_index_), captured for ResetEnvs (jitter about this, no scene rebuild).
        if (cup_local_index_ < bodies_per_env_)
            reset_cup_template_ = scene_template.bodies_per_env[cup_local_index_];
        // Host mirror of the per-step fingertip world positions (env-major, filled by the
        // resolver from the FK download; ExportObsState surfaces it with no extra round-trip).
        fingertip_world_host_.assign(
            static_cast<size_t>(env_count_) * fingertips_.size() * 3u, 0.0f);
        // ExportObsState scratch (the bulk q/qdot/link_velocity download targets, reused).
        const size_t tlc2 = static_cast<size_t>(env_count_) * base_link_count_;
        obs_q_scratch_.assign(tlc2, 0.0f);
        obs_qdot_scratch_.assign(tlc2, 0.0f);
        obs_linkvel_scratch_.assign(tlc2, articulation::LinkSpatialVel{});

        // ----- P2.4b persistent batched articulation scratch (allocated once, reused) -----
        // world_pose_dev_ = the ONE batched UpdateWorldLinkPoses output (every env's link
        // poses); crba_* = the batched CRBA composite scratch + env-major M / M^-1 tiles
        // (one ComputeArticulationInertiaM/Factor over all N). Sized from the world; never
        // re-allocated per step (the per-step work is in-place kernel writes).
        const size_t tlc = static_cast<size_t>(env_count_) * base_link_count_;
        const size_t tile = static_cast<size_t>(dof_stride_) * dof_stride_;
        world_pose_dev_ = nuka::phi::Buffer(tlc * sizeof(Transform),
                                            nuka::phi::MemoryKind::Device);
        crba_composite_ = nuka::phi::Buffer(
            tlc * sizeof(articulation::LinkSpatialInertia), nuka::phi::MemoryKind::Device);
        crba_m_ = nuka::phi::Buffer(static_cast<size_t>(env_count_) * tile * sizeof(float),
                                    nuka::phi::MemoryKind::Device);
        crba_m_inv_ = nuka::phi::Buffer(static_cast<size_t>(env_count_) * tile * sizeof(float),
                                        nuka::phi::MemoryKind::Device);
    }
}

void BatchedUnifiedWorld::DownloadGripper(
    uint32_t env, articulation::ArticulationHostState* out) const {
    if (out == nullptr || !has_grasp_ || env >= env_count_) return;
    // P2.4b: env_device_ now holds ALL N grippers env-major. Download the consolidated
    // state, then COPY OUT env e's single-articulation slice (links [e*base_link_count_,
    // (e+1)*base_link_count_), the per-link / per-DOF arrays are tiled in replica order) so
    // the returned host state is the SAME shape the old per-env DownloadGripper produced.
    *out = gripper_proto_;
    articulation::ArticulationHostState all =
        articulation::ReplicateArticulationHostState(gripper_proto_, env_count_);
    articulation::DownloadArticulationState(env_device_, &all);
    const size_t off = static_cast<size_t>(env) * base_link_count_;
    auto slice = [&](auto& dst, const auto& src) {
        dst.assign(src.begin() + static_cast<std::ptrdiff_t>(off),
                   src.begin() + static_cast<std::ptrdiff_t>(off + base_link_count_));
    };
    slice(out->link_velocity, all.link_velocity);
    slice(out->link_acceleration, all.link_acceleration);
    slice(out->q, all.q);
    slice(out->qdot, all.qdot);
    slice(out->qddot, all.qddot);
    slice(out->tau, all.tau);
    // base_pose is per-articulation (one entry / env); copy env e's single base pose.
    if (env < all.base_pose.size() && !out->base_pose.empty())
        out->base_pose[0] = all.base_pose[env];
}

// ===========================================================================
// G1b: per-env FLOATING-BASE initial-condition seam (the articulation BodyMut).
// ===========================================================================
void BatchedUnifiedWorld::SetGripperBasePose(uint32_t env,
                                             const math::Transform& pose) {
    if (!has_grasp_ || env >= env_count_) return;
    // ONE bulk download-modify-upload of the env-major base_pose buffer (one Transform
    // per articulation == per env). OFF the per-step path: an IC-setup seam, called
    // before stepping (the G1b MIXED gate / future RL base-IC randomization).
    std::vector<Transform> base(env_count_);
    env_device_.base_pose.CopyToHost(base.data(), base.size() * sizeof(Transform));
    context_.stream.Synchronize();
    base[env] = pose;
    env_device_.base_pose.CopyFromHost(base.data(), base.size() * sizeof(Transform));
    context_.stream.Synchronize();
}

// ===========================================================================
// G1d: the mid-run TABLE toggle (the oracle's SetTableEnabled, per env).
// ===========================================================================
// OFF the per-step hot path: a host byte-vector write; the per-env table-emission
// gate in ResolveBatchedGraspContact reads it. Emission also requires has_table_
// (the oracle's emit_table = has_table && table_enabled_ && have_cup), so calls
// on a has_table=false world stay inert -> byte-compat.
void BatchedUnifiedWorld::SetTableEnabled(bool enabled) {
    std::fill(table_enabled_.begin(), table_enabled_.end(),
              enabled ? uint8_t{1} : uint8_t{0});
}

void BatchedUnifiedWorld::SetTableEnabled(uint32_t env, bool enabled) {
    if (env < table_enabled_.size())
        table_enabled_[env] = enabled ? uint8_t{1} : uint8_t{0};
}

// ===========================================================================
// A1 RL substrate: per-step per-env ACTION INJECTION.
// ===========================================================================
void BatchedUnifiedWorld::SetActions(const float* actions, size_t count) {
    if (!has_grasp_ || actions == nullptr || env_count_ == 0u) return;
    const size_t expect = static_cast<size_t>(env_count_) * dof_stride_;
    if (count != expect) return;  // wrong-shape actions ignored (default drive retained).
    // Scatter the env-major DOF-indexed host actions into the env-major per-link drive
    // buffer the grip kernel reads: action[e*dof_stride_+d] -> link e*base_link_count_+
    // dof_to_link_[d]. Non-finger links (Fixed root) keep their template value (0). One
    // host scatter + ONE device upload -- no per-env round-trip.
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const size_t aoff = static_cast<size_t>(e) * dof_stride_;
        const size_t loff = static_cast<size_t>(e) * base_link_count_;
        for (uint32_t d = 0u; d < dof_stride_; ++d) {
            const uint32_t link = dof_to_link_[d];
            action_torque_host_[loff + link] = actions[aoff + d];
        }
    }
    action_torque_dev_.CopyFromHost(action_torque_host_.data(),
                                    action_torque_host_.size() * sizeof(float));
    context_.stream.Synchronize();
}

// ===========================================================================
// A1 RL substrate: BATCHED OBS EXPORT (one bulk download per device buffer).
// ===========================================================================
void BatchedUnifiedWorld::ExportObsState(ObsStateBatch& out) const {
    const uint32_t nfinger = static_cast<uint32_t>(fingertips_.size());
    if (!has_grasp_ || env_count_ == 0u) {
        out.q.clear();
        out.qdot.clear();
        out.fingertip_world_pos.clear();
        out.finger_normal_impulse.clear();
        out.base_pose.clear();
        out.base_vel.clear();
        return;
    }
    // perf tag `obs_export`: the FULL batched RL readout cost (the ONE consolidated device
    // download + the host packing). The perf TU brackets this in the per-step measurement so
    // GATE-5 reports throughput INCLUDING the obs export and the breakdown shows its scaling.
    ScopedWallTimer obs_timer(perf_, "obs_export");
    // Size once, reuse (assign also fills any growth deterministically).
    out.q.assign(static_cast<size_t>(env_count_) * dof_stride_, 0.0f);
    out.qdot.assign(static_cast<size_t>(env_count_) * dof_stride_, 0.0f);
    out.fingertip_world_pos.assign(
        static_cast<size_t>(env_count_) * nfinger * 3u, 0.0f);
    out.finger_normal_impulse.assign(static_cast<size_t>(env_count_) * nfinger, 0.0f);
    out.base_pose.assign(static_cast<size_t>(env_count_) * 7u, 0.0f);
    out.base_vel.assign(static_cast<size_t>(env_count_) * 6u, 0.0f);

    // ----- q / qdot: ONE bulk cudaMemcpy-class download PER device buffer (NOT a per-env
    // DownloadGripper loop, which is exactly the O(N) sync storm P2.4b removed; and NOT a full
    // DownloadArticulationState of all ~20 SoA arrays -- we copy ONLY the q / qdot / link_velocity
    // device buffers this readout actually reads). Each CopyToHost is one device->host copy over
    // the whole env-major buffer, so the cost scales with the buffer size (linear in N), not with
    // a per-env round-trip count. We then pack env e's flat-qdot DOF slice from its
    // [e*base_link_count_, ...) slice using the SAME prefix-sum convention (root DOFs + per-finger
    // DofIndexOf) the resolver packs. link_velocity is only read for a FloatingBase root (base_dof_
    // > 0); for the Fixed-root gripper base_dof_==0 so that copy is unused but kept for generality.
    env_device_.q.CopyToHost(obs_q_scratch_.data(), obs_q_scratch_.size() * sizeof(float));
    env_device_.qdot.CopyToHost(obs_qdot_scratch_.data(),
                                obs_qdot_scratch_.size() * sizeof(float));
    if (base_dof_ > 0u)
        env_device_.link_velocity.CopyToHost(
            obs_linkvel_scratch_.data(),
            obs_linkvel_scratch_.size() * sizeof(articulation::LinkSpatialVel));
    // G2: the per-articulation base poses (env_count_ Transforms -- tiny; ONE bulk
    // copy alongside the q/qdot downloads, NOT a per-env round-trip).
    obs_basepose_scratch_.resize(env_count_);
    env_device_.base_pose.CopyToHost(obs_basepose_scratch_.data(),
                                     obs_basepose_scratch_.size() * sizeof(Transform));
    context_.stream.Synchronize();
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const size_t loff = static_cast<size_t>(e) * base_link_count_;
        const size_t qoff = static_cast<size_t>(e) * dof_stride_;
        // Root DOFs (base_dof_; 0 for the Fixed-root gripper) from link_velocity[root].
        for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
            out.q[qoff + i] = obs_q_scratch_[loff + root_link_];  // scalar q unused by FloatingBase root.
            out.qdot[qoff + i] = obs_linkvel_scratch_[loff + root_link_].v[i];
        }
        // Finger joint DOFs from q / qdot at the prefix-sum column.
        for (uint32_t leg = root_link_ + 1u; leg < base_link_count_; ++leg) {
            const uint32_t col = DofIndexOf(leg);
            if (col < dof_stride_) {
                out.q[qoff + col] = obs_q_scratch_[loff + leg];
                out.qdot[qoff + col] = obs_qdot_scratch_[loff + leg];
            }
        }
        // G2: base pose [px,py,pz,qw,qx,qy,qz] + base spatial velocity (the root
        // link_velocity slot; zeros for a Fixed root, whose link_velocity was not
        // downloaded -- base_dof_==0 -- and which never moves anyway).
        if (e < obs_basepose_scratch_.size()) {
            const Transform& bp = obs_basepose_scratch_[e];
            float* p = out.base_pose.data() + static_cast<size_t>(e) * 7u;
            p[0] = bp.position.x;    p[1] = bp.position.y;    p[2] = bp.position.z;
            p[3] = bp.rotation.w;    p[4] = bp.rotation.x;
            p[5] = bp.rotation.y;    p[6] = bp.rotation.z;
        }
        if (base_dof_ > 0u) {
            float* v = out.base_vel.data() + static_cast<size_t>(e) * 6u;
            for (uint32_t i = 0u; i < 6u; ++i)
                v[i] = obs_linkvel_scratch_[loff + root_link_].v[i];
        }
    }

    // ----- fingertip world positions: already host-resident (the resolver filled
    // fingertip_world_host_ from the SAME FK download); copy it straight out. ----------
    if (out.fingertip_world_pos.size() == fingertip_world_host_.size())
        std::copy(fingertip_world_host_.begin(), fingertip_world_host_.end(),
                  out.fingertip_world_pos.begin());

    // ----- per-finger normal impulse: from the LAST Step()'s host-side report aggregation
    // (the force-closure signal; no device read -- it was bucketed while the rows were live).
    for (uint32_t e = 0u; e < env_count_ && e < grasp_reports_.size(); ++e) {
        const std::vector<float>& fimp = grasp_reports_[e].finger_normal_impulse;
        const size_t foff = static_cast<size_t>(e) * nfinger;
        for (uint32_t f = 0u; f < nfinger && f < fimp.size(); ++f)
            out.finger_normal_impulse[foff + f] = fimp[f];
    }
}

// ===========================================================================
// A1 RL substrate: PER-ENV AUTORESET with deterministic randomization.
// ===========================================================================
void BatchedUnifiedWorld::ResetEnvs(const std::vector<uint32_t>& env_ids,
                                    uint64_t seed) {
    if (!has_grasp_ || env_ids.empty() || env_count_ == 0u) return;

    // ----- (a) per-env cup restore + deterministic XY jitter -------------------------
    // Each listed env's cup BodyState is restored to the reset template (Z + orientation +
    // mass/inertia), its X jittered within +/-reset_jitter_x_ and Y within +/-reset_jitter_y_
    // by a per-env mt19937 seeded from (seed XOR env-id) so two calls with the SAME seed
    // produce IDENTICAL states (D1). At the default (both == kResetCupJitterM == 0.025) the
    // two range-parameterized distributions are byte-identical to the legacy single
    // jit(-0.025,0.025) drawn X-then-Y (libstdc++ uniform_real_distribution is stateless
    // across operator() and consumes one engine draw per call, so the draw order/count is
    // unchanged). The host bodies_ are updated directly (Body()/BodyMut() read them).
    for (uint32_t e : env_ids) {
        if (e >= env_count_ || cup_local_index_ >= bodies_per_env_) continue;
        std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(e) + 1ull)));
        std::uniform_real_distribution<float> jit_x(-reset_jitter_x_, reset_jitter_x_);
        std::uniform_real_distribution<float> jit_y(-reset_jitter_y_, reset_jitter_y_);
        const float dx = jit_x(rng);
        const float dy = jit_y(rng);
        BodyState cup = reset_cup_template_;            // template Z / orientation / mass.
        cup.position.x += dx;
        cup.position.y += dy;
        cup.linear_velocity = Vec3::Zero();             // zero cup velocity.
        cup.angular_velocity = Vec3::Zero();
        bodies_[BodyIndex(e, cup_local_index_)] = cup;
    }

    // ----- (b) batched device gripper restore (masked over the listed envs) ----------
    // ONE consolidated download of the env-major device -> overwrite each listed env's
    // [e*base_link_count_, ...) slice with the proto's nominal open config (q = proto q,
    // qdot = 0, link_velocity = 0, link_acceleration = 0) -> ONE consolidated upload. NOT a
    // per-env synchronous round-trip and NOT a full-world rebuild: a single download/modify/
    // upload pair regardless of how many envs were listed. Untouched env slices upload their
    // own unchanged values (a no-op for them), so this is safe to call on a subset.
    articulation::ArticulationHostState all =
        articulation::ReplicateArticulationHostState(gripper_proto_, env_count_);
    articulation::DownloadArticulationState(env_device_, &all);
    const articulation::ArticulationHostState& proto = gripper_proto_;
    for (uint32_t e : env_ids) {
        if (e >= env_count_) continue;
        const size_t loff = static_cast<size_t>(e) * base_link_count_;
        for (uint32_t l = 0u; l < base_link_count_; ++l) {
            all.q[loff + l] = (l < proto.q.size()) ? proto.q[l] : 0.0f;
            all.qdot[loff + l] = 0.0f;
            all.link_velocity[loff + l] = articulation::LinkSpatialVel{};
            all.link_acceleration[loff + l] = articulation::LinkSpatialAccel{};
        }
        // Restore the env's base pose to the proto root (the Fixed gripper root never moves,
        // but reset it for generality so a future FloatingBase root resets cleanly too).
        if (e < all.base_pose.size() && !proto.base_pose.empty())
            all.base_pose[e] = proto.base_pose[0];
    }
    // ONE upload of the whole env-major q / qdot / velocities (untouched slices unchanged).
    env_device_.q.CopyFromHost(all.q.data(), all.q.size() * sizeof(float));
    env_device_.qdot.CopyFromHost(all.qdot.data(), all.qdot.size() * sizeof(float));
    env_device_.link_velocity.CopyFromHost(
        all.link_velocity.data(),
        all.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
    env_device_.link_acceleration.CopyFromHost(
        all.link_acceleration.data(),
        all.link_acceleration.size() * sizeof(articulation::LinkSpatialAccel));
    if (!all.base_pose.empty())
        env_device_.base_pose.CopyFromHost(
            all.base_pose.data(), all.base_pose.size() * sizeof(Transform));
    context_.stream.Synchronize();

    // Refresh the last-step reports for the reset envs so an ExportObsState before the next
    // Step() reads a clean (zero-impulse) force-closure signal for them.
    for (uint32_t e : env_ids) {
        if (e < grasp_reports_.size())
            grasp_reports_[e].finger_normal_impulse.assign(fingertips_.size(), 0.0f);
    }
}

uint32_t BatchedUnifiedWorld::DofIndexOf(uint32_t link) const {
    // REPLICATED byte-for-byte from UnifiedCoResidentStepper::DofIndexOf (~:346): the
    // flat-qdot prefix-sum DOF index of a device link (Σ JointDofCount over the gripper
    // links in [root_link_, link)). For a Fixed-root gripper base_dof_==0 -> every
    // column is a finger joint qdot.
    uint32_t idx = 0u;
    for (uint32_t k = root_link_; k < link; ++k) {
        idx += articulation::ArticulationJointDofCount(gripper_proto_.joint_type[k]);
    }
    return idx;
}

void BatchedUnifiedWorld::IntegrateBodyPosition(
    runtime::rigid::BodyState& body) const {
    // BYTE-IDENTICAL to UnifiedCoResidentStepper::IntegrateBoxPosition: pure
    // symplectic-Euler kinematics (NO floor clamp; contact support flows through
    // the unified spine in the contact phase, which P2.2 adds).
    if (body.inv_mass <= 0.0f) return;
    body.position += body.linear_velocity * dt_;
    const math::Vec3 w = body.angular_velocity;
    math::Quat dq;
    dq.w = 1.0f;
    dq.x = 0.5f * w.x * dt_;
    dq.y = 0.5f * w.y * dt_;
    dq.z = 0.5f * w.z * dt_;
    body.orientation = (body.orientation * dq).Normalized();
}

void BatchedUnifiedWorld::Step() {
    // ----- P2.3a GRASP branch (gated on has_grasp_) ------------------------------
    // The N=1 batched analog of UnifiedCoResidentStepper::StepGrasp. It owns its ENTIRE
    // stage order (the grip drive + gripper ABA + the cup gravity kick all live inside
    // ResolveBatchedGraspContact, mirroring the oracle's stages 0-10); the top-of-Step
    // rigid gravity kick + the P2.2 ground path are NOT run, so they stay byte-for-byte
    // unchanged when !has_grasp_. The position integrate (gripper device IntegratePosition
    // + IntegrateFloatingBasePose; cup IntegrateBodyPosition) advances the POST-contact
    // velocities, matching the oracle's stage 11.
    if (has_grasp_) {
        ResolveBatchedGraspContact();
        // ----- scatter_integrate (part 2/2): the POSITION integrate stage. ----------
        // The scatter-qdot-back half lives in ResolveBatchedGraspContact (same tag, RAII
        // accumulates); this is the gripper IntegratePosition + cup symplectic-Euler step.
        ScopedWallTimer integrate_timer(perf_, "scatter_integrate");
        // Advance EVERY env's gripper joints (q += qdot*dt) + the floating base (no-op for
        // the Fixed-root gripper, but called unconditionally to mirror the oracle stage 11)
        // with the post-contact velocity scattered by ResolveBatchedGraspContact. P2.4b: ONE
        // batched IntegratePosition / IntegrateFloatingBasePose over the consolidated env-major
        // device advances all N grippers in a single launch each (one block per articulation).
        {
            auto view = env_device_.View();
            articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
            articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        }
        // The cup (and any other movable body) symplectic-Euler position step.
        for (auto& body : bodies_) {
            IntegrateBodyPosition(body);
        }
        // Record the post-integrate cup height in each env's grasp report (cup_vz was
        // captured post-solve; velocity is unchanged by the position integrate).
        for (uint32_t e = 0u; e < env_count_ && e < grasp_reports_.size(); ++e) {
            grasp_reports_[e].cup_z = bodies_[BodyIndex(e, cup_local_index_)].position.z;
        }
        return;
    }

    // ----- velocity stage: gravity velocity-kick, env-major order (D1) ----------
    // Matches the co-resident box gravity kick (linear_velocity.z += g*dt) applied
    // BEFORE the contact phase. Immovable bodies (inv_mass<=0) are skipped, exactly
    // as IntegrateBoxPosition skips them.
    for (auto& body : bodies_) {
        if (body.inv_mass <= 0.0f) continue;
        body.linear_velocity.z += gravity_z_ * dt_;
    }

    // ----- contact phase ---------------------------------------------------------
    // P2.2: the batched box<->static-ground contact phase (a no-op when !has_ground_,
    // which preserves the P2.1 free-fall path byte-for-byte). Per-env narrowphase ->
    // concatenated compliant rows (per-env body-id offsets) -> ONE UnifiedSolve, which
    // mutates `bodies_` velocities in place BEFORE the position stage integrates them.
    ResolveBatchedGroundContact();

    // ----- position stage: symplectic-Euler integrate, env-major order (D1) ------
    for (auto& body : bodies_) {
        IntegrateBodyPosition(body);
    }
}

// DEPRECATED(M9): this per-env CPU narrowphase (BuildContactManifolds box x
// plane) + the per-env host pair download are SUPERSEDED by the M5 device-
// resident collision spine (the nk::World union slot-template path —
// contacts_union.cu's analytic detection + the pair-driven broadphase ops). The
// plan (M5 §3.5) calls for deleting them, but BatchedUnifiedWorld is STILL the
// reference oracle for the legacy G1c/G1d parity tests (and h1_union_parity /
// the grasp parity gate compare nk against it) until M9 retires the coresident
// directory. So these segments are RETAINED with this banner, NOT deleted — they
// have a live consumer. M9 removes BatchedUnifiedWorld and this CPU narrowphase
// together. (Controller ruling, mirrors the M4 row_solver/unified_solve banners.)
void BatchedUnifiedWorld::ResolveBatchedGroundContact() {
    // Inert unless a static ground is configured AND there is at least one body per
    // env to rest on it. Leaves the velocity untouched -> P2.1 free-fall preserved.
    if (!has_ground_ || bodies_per_env_ == 0u || env_count_ == 0u) return;

    // ONE shared rows/sides buffer across ALL envs (EmitCompliantContactRows APPENDS,
    // so repeated per-env calls concatenate). Per-env body-id offsets make the rows
    // from different envs share NO body id -> the greedy graph coloring partitions
    // them into N independent solves (row_scheduler.cu RowsConflict).
    RowBuffers rows;
    std::vector<ContactRowSides> sides;

    // condim=1 frictionless normal rows, refsafe -- COPIED from the oracle (lines
    // ~481-488). dt/vel/invweight identical so an N=1 env is byte-exact.
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = dt_;
    inputs.condim = 1u;
    inputs.refsafe = true;

    // The static ground PrimParams is env-invariant (one shared world +Z plane).
    const amf::PrimParams ground_prim = GroundPrim(ground_height_);

    for (uint32_t e = 0u; e < env_count_; ++e) {
        // env e's movable rigid body is local body 0 (the P2.2 one-box-per-env scope).
        const runtime::rigid::BodyState& body = bodies_[BodyIndex(e, 0u)];

        // The box<->ground candidate pair, emitted DIRECTLY (the box AABB always
        // overlaps the +Z half-space if its bottom is at/below the plane -- trivial
        // broadphase). Side A = box (RigidInvMass), side B = ground (StaticNull).
        // Mirrors the oracle stage 5d.
        CandidatePair ground_pair;
        ground_pair.a = MakeBoxRef(kBoxHandle);
        ground_pair.b = MakeGroundRef(kGroundHandle);

        // Per-env resolver bound to THIS env's live body pose. box -> C3b BoxPlane.
        const Transform box_pose{body.position, body.orientation};
        const amf::PrimParams box_prim = BoxPrim(box_pose, box_half_extent_);
        ShapeResolver resolve = [&](const CollidableRef& ref,
                                    ResolvedShape* out) -> bool {
            if (ref.type == CollidableType::RigidBody && ref.handle == kBoxHandle) {
                out->type = scene::ShapeType::Box;
                out->prim = box_prim;
                return true;
            }
            if (ref.type == CollidableType::StaticWorld &&
                ref.handle == kGroundHandle) {
                out->type = scene::ShapeType::Plane;
                out->prim = ground_prim;
                return true;
            }
            return false;
        };

        // Per-env narrowphase (box x plane). manifolds is rebuilt fresh per env.
        std::vector<ContactManifold> manifolds;
        const CandidatePair pairs[1] = {ground_pair};
        nuka::collision::BuildContactManifolds(pairs, resolve, &manifolds);
        if (manifolds.empty()) continue;  // box clear of the plane -> no rows this env.

        // APPEND env e's compliant normal rows to the shared buffer.
        const std::size_t row_start = rows.RowCount();
        nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);

        // OVERWRITE the appended rows' body indices: rigid (RigidInvMass) side ->
        // env e's flat env-major body index BodyIndex(e,0); static (StaticNull) side
        // -> kInvalidBodyIndex (no reaction, no coloring conflict). Mirrors the oracle
        // box<->ground branch (lines ~577-588), but with the batched env-major index
        // instead of a hard 0.
        const uint32_t box_body_index = BodyIndex(e, 0u);
        for (std::size_t r = row_start; r < rows.RowCount(); ++r) {
            const ContactRowSides& s = sides[r];
            const bool a_rigid = s.a.react == ReactionProviderKind::RigidInvMass;
            const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
            // The pair is always (box=A=RigidInvMass, ground=B=StaticNull); the
            // a_rigid/b_static check makes the local mapping robust + matches the
            // oracle's side-dispatch shape.
            const int box_local = a_rigid ? 0 : 1;
            const int static_local = a_rigid ? 1 : 0;
            (void)b_static;
            rows.body_indices[2u * r + static_cast<uint32_t>(box_local)] =
                box_body_index;
            rows.body_indices[2u * r + static_cast<uint32_t>(static_local)] =
                nuka::constraint::kInvalidBodyIndex;
        }
    }

    if (rows.RowCount() == 0u || sides.empty()) return;  // no env touched the plane.

    // ONE unified two-way solve over the concatenated rows + the FULL env-major
    // bodies. ctx.articulation left default (art_refs=nullptr -> the pure-rigid C5a
    // path; NO articulation in P2.2). cfg COPIED from the oracle (lines ~599-603) so
    // an N=1 env is byte-identical. UnifiedSolve mutates bodies_ velocities in place.
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies_;
    ctx.sides = &sides;
    ctx.dt = dt_;
    // ctx.articulation default-constructed (art_refs=nullptr): NO articulation arm.
    nuka::solver::UnifiedSolve(ctx, cfg);
}

// ===========================================================================
// P2.3a: the batched articulation<->rigid GRASP contact phase (N=1).
// ===========================================================================
// The N=1 batched analog of UnifiedCoResidentStepper::StepGrasp (~:681). Mirrors the
// oracle's EXACT stage order so an N=1 grasp world equals the validated co-resident
// oracle (the A2 gate). Structured in ENV TERMS (an env loop with e in [0, env_count_),
// art_index=e, total_body_count=bodies_.size(), cup side = BodyIndex(e, cup_local_index_))
// so P2.3b is JUST generalizing this loop to N>1 + concatenating env-major M^-1 / qdot
// tiles -- no restructuring. For N=1 every index reduces to the oracle's literal 0/1.
void BatchedUnifiedWorld::ResolveBatchedGraspContact() {
    if (!has_grasp_ || env_count_ == 0u || bodies_per_env_ == 0u) return;

    // ----- stages 0-3 BATCHED over ALL N envs in ONE launch each (P2.4b). --------------
    // The consolidated env-major device (articulation_count == env_count_) lets the already-
    // per-articulation-batched FeatherstoneAba:: methods advance every gripper at once. This
    // REPLACES the P2.3b per-env loop (the per-env upload/sync round-trips). Each kernel is
    // one block per articulation, so byte-for-byte the same op sequence env e ran in N=1, with
    // env e reading/writing its own [e*base_link_count_, ...) slice -- no cross-env coupling.
    // perf tag `aba_integrate`: the grip-torque drive + ABA accelerations + velocity integrate.
    {
    ScopedWallTimer aba_timer(perf_, "aba_integrate");
    auto view = env_device_.View();
    // ----- stage 0: the DRIVE path. Re-apply the constant (env-major-replicated) grip
    // torque to tau. LaunchApplyTorqueDriveKernels writes state.tau (per global link);
    // ComputeAccelerations then READS tau in its bias-force pass (never zeroes it). Re-
    // applying the same constant each step is idempotent -- the fingers actively squeeze.
    // A1: drive from the LIVE per-step action buffer (env-major per-link), NOT the constant
    // grip_torque_dev_. By default action_torque_dev_ == the replicated template grip torque
    // (byte-identical to today); SetActions overwrites env e's finger DOFs before the step.
    articulation::LaunchApplyTorqueDriveKernels(
        context_, view, static_cast<const float*>(action_torque_dev_.Data()),
        static_cast<const float*>(grip_limits_dev_.Data()));
    // ----- stage 1/2: ABA accelerations -> qddot (now WITH the grip tau). ---------
    articulation::FeatherstoneAba::ComputeAccelerations(context_, view, gravity_z_);
    // ----- stage 3: velocity integrate (gripper halves). -------------------------
    articulation::FeatherstoneAba::IntegrateVelocity(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, view, dt_,
                                                                 gravity_z_);
    }  // end aba_integrate timer scope.

    // ----- stage 4: ONE batched FK over all N envs -> world_pose_dev_ -> refresh device
    // link_pose (the chain-J reads state.link_pose) + ONE host download for the per-env
    // narrowphase. UpdateWorldLinkPoses walks every articulation's links (one block per
    // articulation) writing the live world pose; this REPLACES the P2.3b per-env
    // DownloadWorldPoses (a fresh UpdateWorldLinkPoses + Sync + CopyToHost PER ENV). The
    // device link_pose is refreshed from these FK poses (mirrors BatchedArticulatedWorld's
    // stage-4 world_pose->link_pose copy + the oracle FootChainJ's host.link_pose=fk seam)
    // via a host round-trip CopyFromHost -- byte-identical to the oracle (ABA/CRBA never read
    // link_pose; only ComputeContactChainJacobians does, and it now reads the FK pose).
    std::vector<Transform> all_poses(static_cast<size_t>(env_count_) * base_link_count_);
    {
        ScopedWallTimer pose_dl(perf_, "pose_download");
        auto view = env_device_.View();
        articulation::UpdateWorldLinkPoses(
            context_, view, static_cast<Transform*>(world_pose_dev_.Data()));
        context_.stream.Synchronize();
        world_pose_dev_.CopyToHost(all_poses.data(), all_poses.size() * sizeof(Transform));
        // Refresh the device link_pose from the FK poses so the batched chain-J reads the
        // current geometry (the cooked link_pose is the static rest pose).
        env_device_.link_pose.CopyFromHost(all_poses.data(),
                                           all_poses.size() * sizeof(Transform));
    }
    // ONE consolidated host download of the live gripper state (q/qdot/link_velocity/...) for
    // the per-env scatter + the env-major qdot pack. REPLACES the P2.3b per-env artic_download.
    articulation::ArticulationHostState all_live =
        articulation::ReplicateArticulationHostState(gripper_proto_, env_count_);
    {
        ScopedWallTimer artic_dl(perf_, "artic_download");
        articulation::DownloadArticulationState(env_device_, &all_live);
    }

    // The cup (env e's local body cup_local_index_) gravity velocity-kick, BEFORE the
    // contact solve. NOTE: this kick lives ONLY in the grasp branch -- the top-of-Step
    // rigid kick is NOT run when has_grasp_, so the cup is kicked exactly once. Capture
    // the post-kick / pre-contact cup vz per env (the baseline the contact impulse must
    // lift back to balance the weight -- the HOLD gate's force-balance reference).
    grasp_reports_.assign(env_count_, BatchedGraspEnvReport{});
    std::vector<float> cup_vz_pre_contact(env_count_, 0.0f);
    for (uint32_t e = 0u; e < env_count_; ++e) {
        BodyState& cup = bodies_[BodyIndex(e, cup_local_index_)];
        if (cup.inv_mass > 0.0f) cup.linear_velocity.z += gravity_z_ * dt_;
        cup_vz_pre_contact[e] = cup.linear_velocity.z;
    }

    // ===== CONTACT PHASE (the COMPLETE UNION: feet x ground [G1b, host SpherePlane]
    // + fingertips x cup [GPU sphere x hull] + cup-proxy x table [G1d, host
    // BoxPlane]) ================================================================
    // Slice the ONE consolidated host download (all_live / all_poses, taken ONCE above) into a
    // per-env single-articulation host snapshot (env e's links [e*base_link_count_, ...)). The
    // rest of the per-env loop (narrowphase, row assembly, qdot pack) reads env_live[e] /
    // env_poses[e] exactly as the P2.3b path did -- only the SOURCE changed (one batched
    // download + a host slice, NOT a fresh per-env device round-trip). The GUARDED scatter
    // below still reuses env e's `live`; a no-contact env's slice is built but never written
    // back, exactly mirroring the N=1 BITE early-return.
    std::vector<articulation::ArticulationHostState> env_live(env_count_);
    std::vector<std::vector<Transform>> env_poses(env_count_);
    for (uint32_t e = 0u; e < env_count_; ++e) {
        env_live[e] = gripper_proto_;
        const size_t loff = static_cast<size_t>(e) * base_link_count_;
        auto slice_link = [&](auto& dst, const auto& src) {
            dst.assign(src.begin() + static_cast<std::ptrdiff_t>(loff),
                       src.begin() + static_cast<std::ptrdiff_t>(loff + base_link_count_));
        };
        slice_link(env_live[e].link_velocity, all_live.link_velocity);
        slice_link(env_live[e].qdot, all_live.qdot);
        // env e's FK world poses (single-articulation slice of the batched FK output).
        env_poses[e].assign(all_poses.begin() + static_cast<std::ptrdiff_t>(loff),
                            all_poses.begin() + static_cast<std::ptrdiff_t>(loff + base_link_count_));
        // ----- A1 RL substrate: retain env e's fingertip WORLD positions (the SAME
        // finger_centers = link_pose * local_offset the narrowphase computes), so
        // ExportObsState surfaces them with no extra device round-trip. Filled here
        // (unconditionally in the grasp branch) from the ONE batched FK download. ----------
        const uint32_t nfinger = static_cast<uint32_t>(fingertips_.size());
        for (uint32_t f = 0u; f < nfinger; ++f) {
            const CoResidentFingertip& ft = fingertips_[f];
            const Transform& lp = env_poses[e][ft.link];
            const Vec3 center = lp.position + lp.rotation.Rotate(ft.local_offset);
            const size_t base = (static_cast<size_t>(e) * nfinger + f) * 3u;
            fingertip_world_host_[base + 0u] = center.x;
            fingertip_world_host_[base + 1u] = center.y;
            fingertip_world_host_[base + 2u] = center.z;
        }
    }

    // ONE shared rows/sides buffer across ALL envs (EmitCompliantContactRows APPENDS).
    // P2.3b concatenates every env's finger rows here; cross-env rows carry DISJOINT body
    // keys -> the greedy-in-index coloring partitions them into N independent solves.
    RowBuffers rows;
    std::vector<ContactRowSides> sides;

    // The per-env articulation refs + the flat per-row chain-J accumulator + the ENV-MAJOR
    // M^-1 / qdot buffers. The bijection invariant (see below) keys these by art_index = e.
    // ★ ENV-MAJOR SIZING (the P2.3b named-debt fix). The row solver indexes the M^-1 tile
    // at art_index*dof_stride^2 and the qdot slice at art_index*dof_stride (env-major, per
    // row_articulation_refs.hpp). So both buffers MUST be pre-sized to env_count_ tiles /
    // slices (zero-filled) and env e's data written AT its e-th offset -- NOT sequentially
    // appended. A sequential append silently corrupts at N>1 whenever ANY env lacks contact
    // (its tile would be skipped -> later envs' tiles land at the wrong offset, and the
    // solver reads env e's data from the wrong place). chain_jacobians STAYS slot-indexed
    // (a running append counter, keyed by the per-row slot, NOT art_index) -- append is
    // correct there.
    std::vector<RowArticulationRefs> art_refs;
    std::vector<float> chain_jacobians;  // one dof_stride-wide slot per finger row (slot-indexed; filled post-loop).
    std::vector<float> minv(static_cast<size_t>(env_count_) * dof_stride_ * dof_stride_,
                            0.0f);  // env e's CRBA tile @ e*dof_stride^2 (filled by the ONE batched CRBA).
    std::vector<float> qdot(static_cast<size_t>(env_count_) * dof_stride_, 0.0f);  // env e @ e*dof_stride.

    // ----- P2.4b: per-slot chain-J inputs, GATHERED in the per-env row-wiring loop in slot
    // order, then consumed by ONE batched ComputeContactChainJacobians AFTER the loop (this
    // REPLACES the P2.3b per-finger-row FootChainJ round-trip -- the measured 53.4%). The
    // slot index is the running count of finger rows; art_refs[r].slot is assigned the SAME
    // slot during the loop, so the batched output's [slot*dof_stride] slice lines up with the
    // row by construction. ★ contact_link MUST be the GLOBAL link e*base_link_count_+local
    // (the kernel selects env e's columns via link_to_articulation[global_link]); deriving it
    // from a running slot counter instead of from e is THE bug Gate_MixedContactNoContact
    // catches (interleaved no-contact envs make slot-index != env-index).
    std::vector<uint32_t> cj_link;       // GLOBAL contact link per slot.
    std::vector<Vec3>     cj_point;      // world contact point per slot.
    std::vector<Vec3>     cj_dir;        // world contact direction (finger jacobian linear) per slot.

    // The bijection invariant body keys (see the comment at the wiring site below).
    // total_body_count is the env-major BodyState count (NOT bodies_per_env_): every
    // movable body across all envs lives in `bodies_`, so the synthetic finger-side key
    // total_body_count + art_index is guaranteed disjoint from every rigid body index.
    const uint32_t total_body_count = static_cast<uint32_t>(bodies_.size());

    // Per-env row range [begin, end) into the shared rows buffer (for the post-solve
    // impulse attribution). For N=1 there is one range; P2.3b keeps one per env.
    std::vector<std::pair<std::size_t, std::size_t>> env_row_range(env_count_, {0u, 0u});

    // ----- P2.4c: BATCHED GPU narrowphase (fingertip Sphere x cup ConvexHull). -----------
    // REPLACES the P2.4b per-env host BuildContactManifolds loop (the measured 90.9% at
    // N=32). ONE kernel over ALL (env x fingertip) slots wraps the SAME HD-clean cvx::
    // SphereHull the host dispatch calls (sphere_is_a=true). We gather every slot's sphere
    // input (fingertip world center from the FK download + radius), the per-env cup world
    // frame (host-baked -- BuildPrimFrame/Quat::Rotate are host-only), and the per-slot
    // candidate side refs (a=fingertip ArticulationLink, b=cup RigidBody) the launcher uses
    // for StampSides, then ONE launch + ONE download yields the per-env manifolds EXACTLY as
    // host BuildContactManifolds would (StampSides + param merge applied in the launcher),
    // modulo the ULP-scale SphereHull host-vs-device float delta the A2 gate (1e-5) absorbs.
    // env_manifolds[e] holds env e's NON-EMPTY fingertip manifolds in fingertip order -- the
    // SAME input the per-env row assembly fed before.
    std::vector<std::vector<ContactManifold>> env_manifolds(env_count_);
    {
        // perf tag `narrowphase`: the batched-kernel prep (sphere inputs + cup frames) +
        // the ONE launch + the ONE download + the host StampSides/merge. The whole grasp
        // narrowphase cost lives under this tag so the re-measure honestly shows the
        // host->GPU collapse (vs the P2.4b per-env host BuildContactManifolds).
        ScopedWallTimer np_timer(perf_, "narrowphase");
        const uint32_t nfinger = static_cast<uint32_t>(fingertips_.size());
        const uint32_t cup_vcount = grasp_cup_.VertexCount();
        if (nfinger > 0u && cup_vcount > 0u) {
            // Per-env cup world frame (one entry / env, host-baked from the live cup pose).
            std::vector<amf::PrimFrame> cup_frames(env_count_);
            // Per-(env x fingertip) sphere slot inputs + side refs (fingertip order).
            std::vector<nuka::collision::gpu::GraspSphereInput> inputs;
            std::vector<CollidableRef> side_a, side_b;
            inputs.reserve(static_cast<size_t>(env_count_) * nfinger);
            side_a.reserve(inputs.capacity());
            side_b.reserve(inputs.capacity());
            CollidableRef cup_ref;
            cup_ref.type = CollidableType::RigidBody;
            cup_ref.react = ReactionProviderKind::RigidInvMass;
            cup_ref.handle = grasp_cup_.broadphase_body_id;
            for (uint32_t e = 0u; e < env_count_; ++e) {
                const BodyState& cup = bodies_[BodyIndex(e, cup_local_index_)];
                const Transform cup_pose{cup.position, cup.orientation};
                cup_frames[e] = amf::BuildPrimFrame(cup_pose);
                for (uint32_t f = 0u; f < nfinger; ++f) {
                    const CoResidentFingertip& ft = fingertips_[f];
                    const Transform& lp = env_poses[e][ft.link];
                    nuka::collision::gpu::GraspSphereInput in;
                    in.center = lp.position + lp.rotation.Rotate(ft.local_offset);
                    in.radius = ft.radius;
                    in.env_index = e;
                    inputs.push_back(in);
                    CollidableRef finger_ref;  // pair side A == fingertip (mirrors drive_pairs).
                    finger_ref.type = CollidableType::ArticulationLink;
                    finger_ref.react = ReactionProviderKind::ArticulationChainJ;
                    finger_ref.handle = ft.broadphase_handle;
                    side_a.push_back(finger_ref);
                    side_b.push_back(cup_ref);
                }
            }
            // ONE batched launch -> slot-indexed manifolds (one per (env x fingertip),
            // env-major then fingertip; a no-contact slot has point_count==0).
            std::vector<ContactManifold> flat;
            nuka::collision::gpu::LaunchGraspSphereHullNarrowphase(
                context_, inputs, cup_frames, grasp_cup_.hull_verts.data(), cup_vcount,
                side_a, side_b, &flat);
            // BUCKET the slot-indexed output back into per-env manifold lists. Slot
            // e*nfinger+f is env e's fingertip f; skip a no-contact slot (point_count==0),
            // mirroring host BuildContactManifolds' non-empty append + the fingertip order
            // the per-env row assembly consumed before.
            for (uint32_t e = 0u; e < env_count_; ++e) {
                env_manifolds[e].reserve(nfinger);
                for (uint32_t f = 0u; f < nfinger; ++f) {
                    const ContactManifold& m = flat[static_cast<size_t>(e) * nfinger + f];
                    if (m.point_count > 0u) env_manifolds[e].push_back(m);
                }
            }
        }
    }

    bool any_contact = false;
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const uint32_t art_index = e;  // env e's articulation index (the M^-1/qdot key).
        BodyState& cup = bodies_[BodyIndex(e, cup_local_index_)];

        // ----- G1b: per-env FOOT-sphere x static-ground HOST narrowphase. ------------
        // The union's feet rows (oracle StepStandGrasp drive_pairs :1420-1430: feet
        // FIRST, then fingers -- the manifold/row order below preserves that). DIRECT-
        // EMIT one (foot=ArticulationChainJ, ground=StaticNull) pair per authored ankle
        // sphere at its FK world center, resolved through the SAME host
        // BuildContactManifolds (C3b SpherePlane) the oracle runs -- byte-identical
        // inputs (the FK poses are the ONE batched download) -> byte-identical
        // manifolds. HOST on purpose (brief §3.3): ~4 spheres/env is trivial next to
        // the 30-fingertip GPU sphere x hull; a GPU foot kernel is the NAMED G1f
        // deferral, gated on a measured throughput finding.
        std::vector<ContactManifold> foot_manifolds;
        uint32_t foot_points = 0u;
        if (has_feet_ && !feet_.empty()) {
            ScopedWallTimer np_timer(perf_, "narrowphase");  // RAII accumulates with the GPU tag.
            std::vector<Vec3> foot_centers(feet_.size());
            std::vector<CandidatePair> foot_pairs;
            foot_pairs.reserve(feet_.size());
            for (size_t f = 0u; f < feet_.size(); ++f) {
                const CoResidentFootSphere& fs = feet_[f];
                const Transform& lp = env_poses[e][fs.link];
                foot_centers[f] = lp.position + lp.rotation.Rotate(fs.local_offset);
                CandidatePair p;
                p.a.type = CollidableType::ArticulationLink;
                p.a.react = ReactionProviderKind::ArticulationChainJ;
                p.a.handle = fs.broadphase_handle;
                p.b = MakeGroundRef(foot_ground_.broadphase_id);  // StaticWorld -> StaticNull.
                foot_pairs.push_back(p);
            }
            ShapeResolver foot_resolve = [&](const CollidableRef& ref,
                                             ResolvedShape* out) -> bool {
                if (ref.type == CollidableType::ArticulationLink) {
                    for (size_t f = 0u; f < feet_.size(); ++f) {
                        if (feet_[f].broadphase_handle == ref.handle) {
                            out->type = scene::ShapeType::Sphere;
                            out->prim = FootSpherePrim(foot_centers[f], feet_[f].radius);
                            out->geom = nullptr;
                            return true;
                        }
                    }
                    return false;
                }
                if (ref.type == CollidableType::StaticWorld &&
                    ref.handle == foot_ground_.broadphase_id) {
                    out->type = scene::ShapeType::Plane;
                    out->prim = GroundPrim(foot_ground_.height);  // +Z plane.
                    out->geom = nullptr;
                    return true;
                }
                return false;
            };
            nuka::collision::BuildContactManifolds(foot_pairs, foot_resolve,
                                                   &foot_manifolds);
            for (const auto& m : foot_manifolds) foot_points += m.point_count;
        }

        // The per-env fingertip<->cup manifolds, produced by the ONE batched GPU narrowphase
        // launch above (env e's NON-EMPTY slots in fingertip order). Replaces the P2.4b
        // per-env host BuildContactManifolds.
        uint32_t finger_points = 0u;
        if (grasp_cup_.VertexCount() > 0u) {
            for (const auto& m : env_manifolds[e]) finger_points += m.point_count;
        }
        grasp_reports_[e].finger_contacts = finger_points;

        // ----- G1d: per-env CUP(-proxy-box) x static-TABLE HOST narrowphase. ----------
        // The oracle StepStandGrasp's THIRD row class (drive_pairs :1455-1459 -- the
        // table pair is appended AFTER feet + fingers; the manifold/row order below
        // preserves that for row-layout parity). DIRECT-EMIT one (cup-proxy =
        // RigidInvMass, table = StaticNull) pair at the cup's LIVE pose, resolved
        // through the SAME host BuildContactManifolds (C3b BoxPlane via the flat-bottom
        // proxy when any cup_table_proxy_half_ component > 0 -- a real mug rests on its
        // flat base, sidestepping the hull-vs-plane coplanar-rim instability, named
        // engine debt; all-zero half-extents fall back to the detailed hull id, hull x
        // plane). This is ResolveBatchedGroundContact's exact emission shape with the
        // cup body index instead of body 0, appended into the SAME union rows -> the
        // ONE UnifiedSolve. Gated per env on the LIVE toggle (the oracle's emit_table =
        // has_table && table_enabled_ && have_cup, :1445): SetTableEnabled(false) is
        // the lift choreography's "remove the table". has_table_=false -> this whole
        // block is dead -> every pre-G1d path is byte-for-byte unchanged.
        std::vector<ContactManifold> table_manifolds;
        uint32_t table_points = 0u;
        if (has_table_ && table_enabled_[e] != 0u && grasp_cup_.VertexCount() > 0u) {
            ScopedWallTimer np_timer(perf_, "narrowphase");  // RAII accumulates.
            const Transform cup_pose{cup.position, cup.orientation};
            const bool use_proxy = cup_table_proxy_half_.x > 0.0f ||
                                   cup_table_proxy_half_.y > 0.0f ||
                                   cup_table_proxy_half_.z > 0.0f;
            const uint32_t cup_table_id =
                use_proxy ? cup_table_proxy_id_ : grasp_cup_.broadphase_body_id;
            CandidatePair tp;
            tp.a = MakeBoxRef(cup_table_id);             // cup proxy = RigidInvMass.
            tp.b = MakeGroundRef(table_broadphase_id_);  // table = StaticNull (+Z).
            // The hull view for the no-proxy fallback (mesh-local verts + live pose;
            // mirrors the oracle resolver's ConvexHull seam, :1499-1504).
            nuka::collision::cvx::ConvexHullView cup_hull;
            cup_hull.verts = grasp_cup_.hull_verts.data();
            cup_hull.vcount = grasp_cup_.VertexCount();
            cup_hull.frame = amf::BuildPrimFrame(cup_pose);
            ShapeResolver table_resolve = [&](const CollidableRef& ref,
                                              ResolvedShape* out) -> bool {
                if (ref.type == CollidableType::RigidBody &&
                    ref.handle == cup_table_id) {
                    if (use_proxy) {
                        // The flat-bottom proxy box at the cup's LIVE pose (the
                        // offset is in the cup body frame; oracle :1486-1497).
                        Transform proxy_pose = cup_pose;
                        proxy_pose.position =
                            cup_pose.position +
                            cup_pose.rotation.Rotate(cup_table_proxy_offset_);
                        out->type = scene::ShapeType::Box;
                        out->prim = BoxPrim(proxy_pose, cup_table_proxy_half_);
                        out->geom = nullptr;
                    } else {
                        out->type = scene::ShapeType::ConvexHull;
                        out->geom = &cup_hull;  // hull x plane fallback.
                    }
                    return true;
                }
                if (ref.type == CollidableType::StaticWorld &&
                    ref.handle == table_broadphase_id_) {
                    out->type = scene::ShapeType::Plane;
                    out->prim = GroundPrim(table_height_);  // +Z plane at table z.
                    out->geom = nullptr;
                    return true;
                }
                return false;
            };
            const CandidatePair table_pairs[1] = {tp};
            nuka::collision::BuildContactManifolds(table_pairs, table_resolve,
                                                   &table_manifolds);
            for (const auto& m : table_manifolds) table_points += m.point_count;
        }
        if (foot_points + finger_points + table_points == 0u)
            continue;  // env contact-free this step.

        // stage 7: compliant rows + sides (condim=3 -> normal row + 4 friction spokes
        // per contact point). APPEND this env's rows to the shared buffer, FEET rows
        // FIRST then FINGER rows (the oracle's single emission over the feet-then-
        // fingers manifold list; EmitCompliantContactRows is a pure per-manifold append,
        // so two calls over the split lists are byte-identical to its one call).
        nuka::constraint::ContactRowComplianceInputs inputs;
        inputs.vel = 0.0f;
        inputs.invweight = 1.0f;
        inputs.dt = dt_;
        inputs.condim = condim_;
        const std::size_t row_start = rows.RowCount();
        std::size_t foot_rows_end = row_start;
        std::size_t finger_rows_end = row_start;
        {
            // perf tag `row_assembly`: compliant contact-row emission (normal + friction
            // spokes). The host index/key wiring loop below is left UNTIMED (microseconds)
            // so this stays DISJOINT from chain_jacobian (FootChainJ lives in that loop).
            // Emission ORDER = the oracle's single emission over its feet-then-fingers-
            // then-table manifold list (EmitCompliantContactRows is a pure per-manifold
            // append, so the split calls are byte-identical to its one call).
            ScopedWallTimer ra_timer(perf_, "row_assembly");
            if (foot_points > 0u) {
                nuka::constraint::EmitCompliantContactRows(foot_manifolds, inputs, &rows,
                                                           &sides);
                foot_rows_end = rows.RowCount();
            }
            if (finger_points > 0u) {
                nuka::constraint::EmitCompliantContactRows(env_manifolds[e], inputs,
                                                           &rows, &sides);
            }
            finger_rows_end = rows.RowCount();
            if (table_points > 0u) {  // G1d: cup x table rows LAST (oracle pair order).
                nuka::constraint::EmitCompliantContactRows(table_manifolds, inputs,
                                                           &rows, &sides);
            }
        }
        // Stamp the per-contact friction by CATEGORY (oracle :1551-1560): cup AND
        // static -> cup x table (table_mu_); cup -> finger x cup (friction_mu_); else
        // -> foot x ground (foot_mu_). With has_table=false no (cup,static) row exists
        // -> this reduces EXACTLY to the pre-G1d two-way stamp (static -> foot_mu_,
        // else friction_mu_), byte-identical; with has_feet=false too, every row gets
        // friction_mu_ -- the pre-G1b stamp.
        for (std::size_t r = row_start; r < rows.RowCount(); ++r) {
            const ContactRowSides& s = sides[r];
            const bool is_cup = s.a.react == ReactionProviderKind::RigidInvMass ||
                                s.b.react == ReactionProviderKind::RigidInvMass;
            const bool is_static = s.a.react == ReactionProviderKind::StaticNull ||
                                   s.b.react == ReactionProviderKind::StaticNull;
            if (is_cup && is_static)
                rows.materials[r].friction = table_mu_;        // cup x table.
            else if (is_cup)
                rows.materials[r].friction = friction_mu_;     // finger x cup.
            else
                rows.materials[r].friction = foot_mu_;         // foot x ground.
        }
        if (rows.RowCount() == row_start) continue;  // no rows emitted this env.
        env_row_range[e] = {row_start, rows.RowCount()};
        grasp_reports_[e].finger_row_count =
            static_cast<uint32_t>(finger_rows_end - foot_rows_end);

        // Pack env e's flat prefix-sum qdot slice from env e's LIVE host slice, written AT env
        // e's env-major offset (qdot slice @ art_index*dof_stride). env e's M^-1 tile is NOT
        // built here in P2.4b -- it comes from the ONE batched CRBA after the loop, which
        // writes the SAME env-major layout (tile @ art_index*dof_stride^2). The env-major
        // PLACEMENT (write AT the e-th offset, never append) is the P2.3b named-debt fix and
        // is preserved: the batched CRBA writes EVERY env's tile (a no-contact env's tile is
        // computed but never read because it emits no rows), so there is no gap to misplace.
        const articulation::ArticulationHostState& live = env_live[e];
        for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
            qdot[art_index * dof_stride_ + i] = live.link_velocity[root_link_].v[i];
        }
        for (uint32_t leg = root_link_ + 1u; leg < base_link_count_; ++leg) {
            const uint32_t col = DofIndexOf(leg);
            if (col < dof_stride_) qdot[art_index * dof_stride_ + col] = live.qdot[leg];
        }

        // ----- wire each finger row's body indices + reaction refs -----------------
        // ★ THE BIJECTION INVARIANT (the batching correctness condition). The synthetic
        // finger-side coloring key `total_body_count + art_index` and the articulation
        // refs' `art_index` are set TOGETHER, per env, in THIS loop iteration. The row
        // scheduler's graph coloring keys on the raw row body indices: every finger row
        // of env e carries the SAME finger key (total_body_count + e) AND writes the
        // SAME M^-1 / qdot tile (art_index = e), so same-env finger rows SERIALIZE (no
        // race on env e's qdot tile or its cup) while cross-env rows carry DISJOINT keys
        // -> parallelize (P2.3b). DECOUPLING them -- a finger key that did not match the
        // art_index whose qdot tile the row mutates -- would be a data race. They MUST
        // be assigned as a pair; this is why both live here, in the same iteration.
        art_refs.resize(rows.RowCount());
        RowArticulationSide none_side{};
        const uint32_t cup_body_index = BodyIndex(e, cup_local_index_);
        const uint32_t finger_key = total_body_count + art_index;  // bijection invariant.
        for (std::size_t r = row_start; r < rows.RowCount(); ++r) {
            const ContactRowSides& s = sides[r];
            if (s.a.react == ReactionProviderKind::StaticNull ||
                s.b.react == ReactionProviderKind::StaticNull) {
                grasp_reports_[e].any_static_row = true;  // a static side (ground/table).
            }
            const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
            const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
            const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
            const bool b_cup = s.b.react == ReactionProviderKind::RigidInvMass;
            const bool a_static = s.a.react == ReactionProviderKind::StaticNull;
            const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
            art_refs[r].a = none_side;
            art_refs[r].b = none_side;
            if ((a_cup && b_static) || (b_cup && a_static)) {
                // ----- G1d cup<->table row (oracle StepStandGrasp :1610-1618): the
                // cup side -> env e's flat env-major cup body index, the static table
                // side -> kInvalidBodyIndex. NO chain-J, NO art_refs (both sides are
                // rigid/static -- the pure-rigid arm of the union; art_refs stay
                // none_side, set above). The cup body index is SHARED with env e's
                // finger rows, so same-env table+finger rows SERIALIZE on the cup
                // (correct -- both mutate it) while cross-env table rows carry
                // DISJOINT BodyIndex(e, cup_local) keys -> parallelize. The bijection
                // invariant is untouched (no articulation side on this class).
                const int cup_l2 = a_cup ? 0 : 1;
                const int static_l2 = a_cup ? 1 : 0;
                rows.body_indices[2u * r + static_cast<uint32_t>(cup_l2)] =
                    cup_body_index;
                rows.body_indices[2u * r + static_cast<uint32_t>(static_l2)] =
                    nuka::constraint::kInvalidBodyIndex;
                continue;
            }
            if ((a_art && b_static) || (b_art && a_static)) {
                // ----- G1b foot<->ground row (oracle StepStandGrasp :1620-1644): the
                // foot side gets a chain-J slot + the env's SAME synthetic coloring key
                // as the fingers (total_body_count + art_index -- feet and fingers are
                // the SAME articulation e, so same-env foot+finger rows SERIALIZE on env
                // e's qdot tile and cross-env rows parallelize; the bijection invariant
                // extends UNCHANGED). The static ground side -> kInvalidBodyIndex (no
                // reaction, no coloring conflict, no chain-J). The chain-J input joins
                // the SAME cj_link/cj_point/cj_dir slot list the fingers gather into --
                // ONE batched ComputeContactChainJacobians serves both classes.
                const int foot_local = a_art ? 0 : 1;
                const int static_local = a_art ? 1 : 0;
                const RowJacobian6 j_foot =
                    rows.JacobianForRowBody(static_cast<uint32_t>(r),
                                            static_cast<uint32_t>(foot_local));
                const Vec3 foot_dir = j_foot.linear;
                const Vec3 contact_point = sides[r].contact_point;
                // Map the foot's broadphase handle back to its REAL device (ankle) link.
                const uint32_t foot_handle = a_art ? s.a.handle : s.b.handle;
                uint32_t foot_link = foot_handle;
                for (size_t f = 0u; f < feet_.size(); ++f) {
                    if (feet_[f].broadphase_handle == foot_handle) {
                        foot_link = feet_[f].link;
                        break;
                    }
                }
                const uint32_t slot = static_cast<uint32_t>(cj_link.size());
                cj_link.push_back(art_index * base_link_count_ + foot_link);
                cj_point.push_back(contact_point);
                cj_dir.push_back(foot_dir);
                chain_jacobians.resize(static_cast<size_t>(slot + 1u) * dof_stride_,
                                       0.0f);
                rows.body_indices[2u * r + static_cast<uint32_t>(foot_local)] =
                    finger_key;  // ★ the SAME env key as fingers (bijection invariant).
                rows.body_indices[2u * r + static_cast<uint32_t>(static_local)] =
                    nuka::constraint::kInvalidBodyIndex;
                const RowArticulationSide foot_side{art_index, slot};
                art_refs[r].a = (foot_local == 0) ? foot_side : none_side;
                art_refs[r].b = (foot_local == 1) ? foot_side : none_side;
                continue;
            }
            if (!((a_art && b_cup) || (b_art && a_cup))) continue;  // finger<->cup only.
            const int finger_local = a_art ? 0 : 1;
            const int cup_local = a_art ? 1 : 0;
            const RowJacobian6 j_finger =
                rows.JacobianForRowBody(static_cast<uint32_t>(r),
                                        static_cast<uint32_t>(finger_local));
            const Vec3 finger_dir = j_finger.linear;
            const Vec3 contact_point = sides[r].contact_point;
            // Map the contact's broadphase handle back to the REAL device link the
            // chain-J needs (the handle == link in the default single-sphere layout).
            const uint32_t finger_handle =
                a_art ? s.a.handle : s.b.handle;
            uint32_t finger_link = finger_handle;
            for (size_t f = 0u; f < fingertips_.size(); ++f) {
                if (fingertips_[f].broadphase_handle == finger_handle) {
                    finger_link = fingertips_[f].link;
                    break;
                }
            }
            // ★ GATHER the chain-J input for this finger row into the next slot (the chain-J
            // VALUE is filled post-loop by ONE batched ComputeContactChainJacobians). The GLOBAL
            // contact link is e*base_link_count_+finger_link (the consolidated state's link
            // index for env e's finger) -- the kernel selects env e's columns via
            // link_to_articulation[global_link]. The slot is the running finger-row count;
            // art_refs[r].slot below is set to the SAME slot, so the batched output's
            // [slot*dof_stride] slice lines up with row r by construction.
            const uint32_t slot = static_cast<uint32_t>(cj_link.size());
            cj_link.push_back(art_index * base_link_count_ + finger_link);
            cj_point.push_back(contact_point);
            cj_dir.push_back(finger_dir);
            chain_jacobians.resize(static_cast<size_t>(slot + 1u) * dof_stride_, 0.0f);
            rows.body_indices[2u * r + static_cast<uint32_t>(cup_local)] =
                cup_body_index;
            rows.body_indices[2u * r + static_cast<uint32_t>(finger_local)] =
                finger_key;  // ★ paired with art_index below (the bijection invariant).
            const RowArticulationSide finger_side{art_index, slot};
            art_refs[r].a = (finger_local == 0) ? finger_side : none_side;
            art_refs[r].b = (finger_local == 1) ? finger_side : none_side;
        }
        any_contact = true;
    }

    if (!any_contact || rows.RowCount() == 0u || sides.empty()) {
        // No env established a finger<->cup contact -> the cup(s) free-fall. The post-
        // contact velocity == the gravity-kicked velocity (no solve). The scatter below
        // is skipped (the device velocities are already current from stage 3). Record the
        // (zero-impulse) report so the BITE gate sees cup_vz == the free-falling velocity.
        for (uint32_t e = 0u; e < env_count_; ++e) {
            grasp_reports_[e].cup_vz =
                bodies_[BodyIndex(e, cup_local_index_)].linear_velocity.z;
            // A1: per-fingertip normal impulse is all-zero when no contact (force-closure
            // signal == not grasped). Size it so ExportObsState always sees nfinger entries.
            grasp_reports_[e].finger_normal_impulse.assign(fingertips_.size(), 0.0f);
        }
        return;
    }

    // ----- P2.4b: ONE batched CRBA M^-1 over ALL N envs -> env-major minv. --------------
    // REPLACES the P2.3b per-env InverseInertia (a fresh Upload + ComputeM + Factor + Sync +
    // CopyToHost PER ENV). ComputeArticulationInertiaM reads link_xup, which the stage-1/2
    // ComputeAccelerations wrote on the consolidated device and which nothing since touched
    // (IntegrateVelocity / FK / link_pose-refresh do not change link_xup, and q is unchanged
    // until IntegratePosition in Step()) -- so the M tile is byte-identical to the oracle's
    // InverseInertia (which re-runs ComputeAccelerations only because it uploads a fresh temp
    // with stale link_xup; here link_xup is already current, mirroring BatchedArticulatedWorld).
    // The batched kernel writes EVERY env's tile env-major at art_index*dof_stride^2 -- the
    // SAME layout the P2.3b std::copy produced, so the row solver's per-env tile read is
    // unchanged. A no-contact env's tile is computed but never read (it emits no rows).
    {
        ScopedWallTimer minv_timer(perf_, "crba_minv");
        auto view = env_device_.View();
        articulation::ComputeArticulationInertiaM(
            context_, view, dof_stride_,
            static_cast<articulation::LinkSpatialInertia*>(crba_composite_.Data()),
            static_cast<float*>(crba_m_.Data()));
        articulation::FactorArticulationInertiaM(
            context_, view, dof_stride_, static_cast<const float*>(crba_m_.Data()),
            static_cast<float*>(crba_m_inv_.Data()));
        context_.stream.Synchronize();
        crba_m_inv_.CopyToHost(minv.data(), minv.size() * sizeof(float));
    }

    // ----- P2.4b: ONE batched chain-J over ALL envs' gathered finger contacts. ----------
    // REPLACES the P2.3b per-finger-row FootChainJ (the measured 53.4% storm). The gathered
    // (cj_link=GLOBAL link, cj_point, cj_dir) arrays are in slot order; ComputeContactChainJacobians
    // selects each contact's env columns via link_to_articulation[cj_link[slot]] and writes a
    // dof_stride-wide row per slot to chain_jacobians[slot*dof_stride] -- exactly the slot the
    // row's art_refs[r].slot points at. The device link_pose was refreshed to the FK poses
    // above (the kernel reads state.link_pose), so the chain-J is byte-identical to the
    // oracle's FootChainJ (which set host.link_pose=fk_world_poses before its upload).
    if (!cj_link.empty()) {
        ScopedWallTimer cj_timer(perf_, "chain_jacobian");
        const uint32_t contact_count = static_cast<uint32_t>(cj_link.size());
        nuka::phi::Buffer link_buf(static_cast<size_t>(contact_count) * sizeof(uint32_t),
                                   nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer point_buf(static_cast<size_t>(contact_count) * sizeof(Vec3),
                                    nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer dir_buf(static_cast<size_t>(contact_count) * sizeof(Vec3),
                                  nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer jbuf(
            static_cast<size_t>(contact_count) * dof_stride_ * sizeof(float),
            nuka::phi::MemoryKind::Device);
        link_buf.CopyFromHost(cj_link.data(), cj_link.size() * sizeof(uint32_t));
        point_buf.CopyFromHost(cj_point.data(), cj_point.size() * sizeof(Vec3));
        dir_buf.CopyFromHost(cj_dir.data(), cj_dir.size() * sizeof(Vec3));
        // The kernel only writes the ancestor-chain columns -> zero the output first.
        jbuf.CopyFromHost(chain_jacobians.data(), chain_jacobians.size() * sizeof(float));
        auto view = env_device_.View();
        articulation::ComputeContactChainJacobians(
            context_, view, static_cast<const uint32_t*>(link_buf.Data()),
            static_cast<const Vec3*>(point_buf.Data()),
            static_cast<const Vec3*>(dir_buf.Data()), contact_count, dof_stride_,
            static_cast<float*>(jbuf.Data()));
        context_.stream.Synchronize();
        jbuf.CopyToHost(chain_jacobians.data(), chain_jacobians.size() * sizeof(float));
    }

    // ----- stage 10: the unified two-way solve (ALL finger rows together) --------
    // cfg COPIED from the oracle so an N=1 env is byte-identical.
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies_;
    ctx.sides = &sides;
    ctx.dt = dt_;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians =
        chain_jacobians.empty() ? nullptr : &chain_jacobians;
    ctx.articulation.inertia_m_inv = &minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = dof_stride_;
    {
        // perf tag `row_solver`: the batched UnifiedSolve (host coloring + component
        // partition + uploads + the one-block-per-component sweep kernel) + the host
        // round-trip. G1d throughput increment: formerly a SINGLE-BLOCK kernel, which
        // made this tag 97.6% of union-scene wall at N=1024 (19011 ms/step, eps flat
        // across N); the per-component grid restores cross-env parallelism.
        ScopedWallTimer solve_timer(perf_, "row_solver");
        nuka::solver::UnifiedSolve(ctx, cfg);
    }

    // ----- per-env grasp metrics (the FORCE-BALANCE gate numbers) ----------------
    // Computed HERE, while the rows / lambdas are live -- the SAME quantities StepGrasp
    // reports (~:1013-1052). cup_vertical_impulse = Σ over EVERY finger row of (cup-side
    // λ * cup-side jacobian.linear.z): the vertical impulse the fingers deliver to the
    // cup. For a held cup at steady state this balances the cup weight kick m*g*dt. The
    // cross-check cup_dvz_impulse = m_cup * (vz_after_solve - vz_pre_contact) must agree.
    const uint32_t nfinger = static_cast<uint32_t>(fingertips_.size());
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const auto [rbegin, rend] = env_row_range[e];
        double cup_vert_impulse = 0.0;
        float max_lambda = 0.0f;
        double foot_normal_impulse = 0.0;  // G1b: Σ foot NORMAL λ (the stand support).
        uint32_t foot_normal_rows = 0u;    // G1b: # foot NORMAL rows.
        double table_vert_impulse = 0.0;   // G1d: Σ over ALL table rows of λ*j_cup.z.
        uint32_t table_rows_n = 0u;        // G1d: # table NORMAL rows.
        float max_table_lambda = 0.0f;     // G1d: max table NORMAL-row λ.
        // ----- A1 RL substrate: per-fingertip NORMAL-impulse bucket (the force-closure
        // signal). Σ over each fingertip's NORMAL rows (NOT friction spokes) of the row
        // lambda. The fingertip a row belongs to is read from sides[r]'s ArticulationChainJ
        // side handle (UNTOUCHED by the wiring loop, which only rewrote body_indices), mapped
        // to fingertip order via the SAME broadphase_handle lookup the wiring used. ----------
        std::vector<float>& fimp = grasp_reports_[e].finger_normal_impulse;
        fimp.assign(nfinger, 0.0f);
        for (std::size_t r = rbegin; r < rend; ++r) {
            const ContactRowSides& s = sides[r];
            // ----- G1b: classify by (react,react) exactly as the oracle's report walk
            // (StepStandGrasp :1706-1738). A (ChainJ, StaticNull) row is FOOT x GROUND:
            // Σ its NORMAL λ into the stand-support fields and SKIP the cup terms (a
            // foot row has NO cup side -- reading 'the other side' would project the
            // static jacobian into cup_vertical_impulse). With has_feet=false no static
            // row exists -> the finger accumulation below is bit-unchanged.
            const bool a_art2 = s.a.react == ReactionProviderKind::ArticulationChainJ;
            const bool b_art2 = s.b.react == ReactionProviderKind::ArticulationChainJ;
            const bool a_static2 = s.a.react == ReactionProviderKind::StaticNull;
            const bool b_static2 = s.b.react == ReactionProviderKind::StaticNull;
            const bool a_cup2 = s.a.react == ReactionProviderKind::RigidInvMass;
            const bool b_cup2 = s.b.react == ReactionProviderKind::RigidInvMass;
            if ((a_cup2 && b_static2) || (b_cup2 && a_static2)) {
                // ----- G1d: a (RigidInvMass, StaticNull) row is CUP x TABLE -- the
                // oracle report walk's first class (StepStandGrasp :1717-1728): Σ its
                // λ*j_cup.z (normal + friction spokes) into the table support, count/
                // max only the NORMAL rows, and SKIP the finger terms (crediting the
                // table's vertical support to cup_vertical_impulse would fake the
                // finger hold -- the lift gate's triangle would never close honestly).
                const int cup_l2 = a_cup2 ? 0 : 1;
                const RowJacobian6 j_cup2 =
                    rows.JacobianForRowBody(static_cast<uint32_t>(r),
                                            static_cast<uint32_t>(cup_l2));
                table_vert_impulse +=
                    static_cast<double>(rows.rows[r].lambda) * j_cup2.linear.z;
                if (!(rows.rows[r].flags & nuka::constraint::row_flags::Friction)) {
                    ++table_rows_n;
                    max_table_lambda = std::max(max_table_lambda, rows.rows[r].lambda);
                }
                continue;
            }
            if ((a_art2 && b_static2) || (b_art2 && a_static2)) {
                if (!(rows.rows[r].flags & nuka::constraint::row_flags::Friction)) {
                    foot_normal_impulse += static_cast<double>(rows.rows[r].lambda);
                    ++foot_normal_rows;
                }
                continue;
            }
            const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
            const int cup_local = a_cup ? 0 : 1;
            const RowJacobian6 j_cup =
                rows.JacobianForRowBody(static_cast<uint32_t>(r),
                                        static_cast<uint32_t>(cup_local));
            cup_vert_impulse +=
                static_cast<double>(rows.rows[r].lambda) * j_cup.linear.z;
            if (!(rows.rows[r].flags & nuka::constraint::row_flags::Friction)) {
                max_lambda = std::max(max_lambda, rows.rows[r].lambda);
                // Bucket this NORMAL row's lambda onto its fingertip.
                const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
                const uint32_t finger_handle = a_art ? s.a.handle : s.b.handle;
                for (uint32_t f = 0u; f < nfinger; ++f) {
                    if (fingertips_[f].broadphase_handle == finger_handle) {
                        fimp[f] += rows.rows[r].lambda;
                        break;
                    }
                }
            }
        }
        grasp_reports_[e].cup_vertical_impulse = cup_vert_impulse;
        grasp_reports_[e].max_lambda = max_lambda;
        grasp_reports_[e].foot_normal_impulse_sum = foot_normal_impulse;  // G1b.
        grasp_reports_[e].foot_normal_rows = foot_normal_rows;            // G1b.
        grasp_reports_[e].table_vertical_impulse = table_vert_impulse;    // G1d.
        grasp_reports_[e].table_row_count = table_rows_n;                 // G1d.
        grasp_reports_[e].table_lambda = max_table_lambda;                // G1d.
        const BodyState& cup = bodies_[BodyIndex(e, cup_local_index_)];
        const double cup_mass =
            cup.inv_mass > 0.0f ? 1.0 / static_cast<double>(cup.inv_mass) : 0.0;
        grasp_reports_[e].cup_dvz_impulse =
            cup_mass * (static_cast<double>(cup.linear_velocity.z) - cup_vz_pre_contact[e]);
        grasp_reports_[e].cup_vz = cup.linear_velocity.z;  // velocity unchanged by integrate.
    }

    // ----- SCATTER the post-contact flat-qdot back into the consolidated env-major device ---
    // The inverse of the prefix-sum pack (per env), written into the ONE consolidated host
    // download `all_live` at env e's [e*base_link_count_, ...) slice, then ONE CopyFromHost of
    // the whole env-major link_velocity + qdot back to the device (P2.4b: one upload, not a
    // per-env round-trip). ★ GUARDED to envs that EMITTED finger rows (env_row_range[e] non-
    // empty): UnifiedSolve only mutated those envs' qdot tiles. A no-contact env's `all_live`
    // slice is LEFT UNTOUCHED -- it already holds the correct post-IntegrateVelocity (stage-3)
    // velocity from the consolidated download, so writing the whole buffer back is a no-op for
    // it (byte-identical to the N=1 BITE early-return, which never scatters). UnifiedSolve
    // mutated `bodies_` (the cup velocities) in place for every env.
    // perf tag `scatter_integrate` (part 1/2): the qdot-scatter-to-device + sync. Part 2/2
    // (the position integrate) is in Step()'s grasp branch (same tag, RAII accumulates).
    {
    ScopedWallTimer scatter_timer(perf_, "scatter_integrate");
    bool any_scattered = false;
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const auto [rbegin, rend] = env_row_range[e];
        if (rend <= rbegin) continue;  // no finger rows this env -> slice already current.
        any_scattered = true;
        const uint32_t art_index = e;
        const size_t loff = static_cast<size_t>(e) * base_link_count_;
        // Write env e's post-solve flat qdot into the consolidated host slice at its offset.
        for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
            all_live.link_velocity[loff + root_link_].v[i] = qdot[art_index * dof_stride_ + i];
        }
        for (uint32_t leg = root_link_ + 1u; leg < base_link_count_; ++leg) {
            const uint32_t col = DofIndexOf(leg);
            if (col < dof_stride_)
                all_live.qdot[loff + leg] = qdot[art_index * dof_stride_ + col];
        }
    }
    if (any_scattered) {
        // ONE upload of the whole env-major link_velocity + qdot (untouched no-contact env
        // slices upload their own unchanged stage-3 values -> a no-op for them).
        env_device_.link_velocity.CopyFromHost(
            all_live.link_velocity.data(),
            all_live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
        env_device_.qdot.CopyFromHost(all_live.qdot.data(),
                                      all_live.qdot.size() * sizeof(float));
        context_.stream.Synchronize();
    }
    }  // end scatter_integrate (part 1/2) timer scope.
    // ===== END CONTACT PHASE -- the position integrate happens in Step(). =========
}

}  // namespace nuka::runtime::coresident
