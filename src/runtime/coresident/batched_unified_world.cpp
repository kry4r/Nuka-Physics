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

#include "collision/analytical_manifold.hpp"   // amf::PrimParams / BuildPrimFrame
#include "collision/candidate_pair.hpp"        // CandidatePair / CollidableRef
#include "collision/contact_stream_driver.hpp" // BuildContactManifolds / ResolvedShape
#include "collision/convex_narrowphase.hpp"    // cvx::ConvexHullView (cup hull seam)
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "constraint/contact_row_sides.hpp"    // ContactRowSides
#include "constraint/reaction_provider.hpp"    // ReactionProviderKind
#include "constraint/row.hpp"                  // kInvalidBodyIndex / row_flags
#include "constraint/row_articulation_refs.hpp"  // RowArticulationRefs / RowArticulationSide
#include "constraint/row_buffers.hpp"          // RowBuffers / RowJacobian6
#include "constraint/row_builder.hpp"          // EmitCompliantContactRows / inputs
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"             // UploadVector
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses / ArticulationDofCount / ArticulationJointDofCount
#include "runtime/articulation/articulation_drives.hpp"    // LaunchApplyTorqueDriveKernels
#include "runtime/articulation/articulation_jacobian.hpp"  // ComputeContactChainJacobians + M/M^-1
#include "runtime/articulation/featherstone_aba.hpp"
#include "scene/canonical_types.hpp"           // scene::ShapeType
#include "solver/rigid_solver.hpp"             // SolverConfig
#include "solver/unified_solve.hpp"            // UnifiedSolve / SolveContext

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace nuka::runtime::coresident {

namespace {

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

// Download the live FK world poses of every link from the current device state.
// (UnifiedCoResidentStepper::DownloadWorldPoses, ~:69.)
std::vector<Transform> DownloadWorldPoses(const nuka::phi::DeviceContext& context,
                                          articulation::ArticulationDeviceBuffers& device,
                                          uint32_t link_count) {
    auto view = device.View();
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                              nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, view,
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

// The dof_stride-wide chain-J on `contact_normal`, against the LIVE link poses.
// (UnifiedCoResidentStepper::FootChainJ, ~:88.)
std::vector<float> FootChainJ(const nuka::phi::DeviceContext& context,
                              articulation::ArticulationHostState host,  // by value
                              const std::vector<Transform>& fk_world_poses,
                              uint32_t contact_link, const Vec3& contact_point,
                              const Vec3& contact_normal, uint32_t dof_stride) {
    if (fk_world_poses.size() == host.link_pose.size()) {
        host.link_pose = fk_world_poses;
    }
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer link_buf(sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer point_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer normal_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer jbuf(static_cast<size_t>(dof_stride) * sizeof(float),
                           nuka::phi::MemoryKind::Device);
    link_buf.CopyFromHost(&contact_link, sizeof(uint32_t));
    point_buf.CopyFromHost(&contact_point, sizeof(Vec3));
    normal_buf.CopyFromHost(&contact_normal, sizeof(Vec3));
    std::vector<float> zero(dof_stride, 0.0f);
    jbuf.CopyFromHost(zero.data(), zero.size() * sizeof(float));
    articulation::ComputeContactChainJacobians(
        context, device.View(), static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()), 1u, dof_stride,
        static_cast<float*>(jbuf.Data()));
    context.stream.Synchronize();
    std::vector<float> j(dof_stride);
    jbuf.CopyToHost(j.data(), j.size() * sizeof(float));
    return j;
}

// The dof_stride x dof_stride M^-1 via the production CRBA (reuse, not hand-rolled).
// Requires ABA pass-1 (link_xup current), so ComputeAccelerations runs first.
// (UnifiedCoResidentStepper::InverseInertia, ~:120.)
std::vector<float> InverseInertia(const nuka::phi::DeviceContext& context,
                                  const articulation::ArticulationHostState& host,
                                  float gravity_z, uint32_t dof_stride) {
    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    const uint32_t link_count = host.TotalLinkCount();
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);
    nuka::phi::Buffer composite(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia),
        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m(static_cast<size_t>(dof_stride) * dof_stride * sizeof(float),
                        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m_inv(static_cast<size_t>(dof_stride) * dof_stride * sizeof(float),
                            nuka::phi::MemoryKind::Device);
    articulation::ComputeArticulationInertiaM(
        context, view, dof_stride,
        static_cast<articulation::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m.Data()));
    articulation::FactorArticulationInertiaM(
        context, view, dof_stride, static_cast<const float*>(m.Data()),
        static_cast<float*>(m_inv.Data()));
    context.stream.Synchronize();
    std::vector<float> out(static_cast<size_t>(dof_stride) * dof_stride);
    m_inv.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

// A sphere PrimParams at the fingertip world center (identity rotation; a sphere is
// rotation-invariant). (UnifiedCoResidentStepper::FootSpherePrim, ~:154.)
amf::PrimParams FootSpherePrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;
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

    // ----- P2.3a: stand up the single-env gripper articulation (inert unless grasp) ---
    // Upload the gripper proto ONCE (stepped in place); the host_proto mirror is kept
    // for the per-step InverseInertia / FootChainJ re-uploads (the oracle's `live`).
    // The constant grip torque + force limits are uploaded once + re-applied to tau each
    // step (idempotent). Mirrors the UnifiedCoResidentStepper grasp constructor exactly,
    // so an N=1 grasp world matches the StepGrasp oracle. P2.3b makes the gripper device
    // a per-env vector; for N=1 there is one persistent gripper here.
    if (has_grasp_) {
        gripper_proto_ = scene_template.gripper_proto;
        gripper_device_ = articulation::UploadArticulationState(context_, gripper_proto_);
        dof_stride_ = articulation::ArticulationDofCount(gripper_proto_, 0u);
        root_link_ = gripper_proto_.articulation_link_offset[0];
        base_dof_ =
            articulation::ArticulationJointDofCount(gripper_proto_.joint_type[root_link_]);
        link_count_ = gripper_proto_.TotalLinkCount();

        std::vector<float> torque = scene_template.grip_torque;
        torque.resize(link_count_, 0.0f);
        std::vector<float> limits = scene_template.drive_force_limits;
        limits.resize(link_count_, 0.0f);
        grip_torque_dev_ = nuka::phi::UploadVector(torque);
        grip_limits_dev_ = nuka::phi::UploadVector(limits);
    }
}

void BatchedUnifiedWorld::DownloadGripper(
    articulation::ArticulationHostState* out) const {
    if (out == nullptr || !has_grasp_) return;
    *out = gripper_proto_;
    articulation::DownloadArticulationState(gripper_device_, out);
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
        // Advance the gripper joints (q += qdot*dt) + the floating base (no-op for the
        // Fixed-root gripper, but called unconditionally to mirror the oracle stage 11)
        // with the post-contact velocity scattered by ResolveBatchedGraspContact.
        auto view = gripper_device_.View();
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
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

    auto view = gripper_device_.View();

    // ----- stage 0: the DRIVE path. Re-apply the constant grip torque to tau. -----
    // LaunchApplyTorqueDriveKernels writes state.tau; ComputeAccelerations then READS
    // tau in its bias-force pass (never zeroes it). Re-applying the same constant each
    // step is idempotent -- this is what makes the fingers actively squeeze.
    articulation::LaunchApplyTorqueDriveKernels(
        context_, view, static_cast<const float*>(grip_torque_dev_.Data()),
        static_cast<const float*>(grip_limits_dev_.Data()));

    // ----- stage 1/2: ABA accelerations -> qddot (now WITH the grip tau). ---------
    articulation::FeatherstoneAba::ComputeAccelerations(context_, view, gravity_z_);

    // ----- stage 3: velocity integrate (gripper halves + the cup gravity kick). ---
    articulation::FeatherstoneAba::IntegrateVelocity(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, view, dt_,
                                                                 gravity_z_);
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

    // ===== CONTACT PHASE (fingertips <-> cup, NO table) ======================
    // Snapshot the post-velocity-integrate gripper state for the host pipeline + the FK
    // world poses (the contact geometry needs current q / base). One persistent gripper
    // for N=1; P2.3b downloads per-env.
    articulation::ArticulationHostState live = gripper_proto_;
    articulation::DownloadArticulationState(gripper_device_, &live);
    const std::vector<Transform> poses =
        DownloadWorldPoses(context_, gripper_device_, link_count_);

    // ONE shared rows/sides buffer across ALL envs (EmitCompliantContactRows APPENDS).
    // For N=1 it holds one env's finger rows; P2.3b concatenates per-env rows here.
    RowBuffers rows;
    std::vector<ContactRowSides> sides;

    // The per-env articulation refs + the flat per-row chain-J accumulator + the flat
    // per-env qdot. The bijection invariant (see below) keys these by art_index = e.
    std::vector<RowArticulationRefs> art_refs;
    std::vector<float> chain_jacobians;  // one dof_stride-wide slot per finger row.
    std::vector<float> minv;             // dof_stride^2 (per-env tile; N=1 -> one tile).
    std::vector<float> qdot(dof_stride_, 0.0f);  // per-env qdot slice (N=1 -> one slice).
    std::vector<float> qdot_before;

    // The bijection invariant body keys (see the comment at the wiring site below).
    // total_body_count is the env-major BodyState count (NOT bodies_per_env_): every
    // movable body across all envs lives in `bodies_`, so the synthetic finger-side key
    // total_body_count + art_index is guaranteed disjoint from every rigid body index.
    const uint32_t total_body_count = static_cast<uint32_t>(bodies_.size());

    // Per-env row range [begin, end) into the shared rows buffer (for the post-solve
    // impulse attribution). For N=1 there is one range; P2.3b keeps one per env.
    std::vector<std::pair<std::size_t, std::size_t>> env_row_range(env_count_, {0u, 0u});

    bool any_contact = false;
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const uint32_t art_index = e;  // env e's articulation index (the M^-1/qdot key).
        BodyState& cup = bodies_[BodyIndex(e, cup_local_index_)];

        // The cup convex-hull view in WORLD space (mesh-local verts + live cup pose).
        const Transform cup_pose{cup.position, cup.orientation};
        nuka::collision::cvx::ConvexHullView cup_hull;
        cup_hull.verts = grasp_cup_.hull_verts.data();
        cup_hull.vcount = grasp_cup_.VertexCount();
        cup_hull.frame = amf::BuildPrimFrame(cup_pose);
        if (cup_hull.vcount == 0u) continue;  // no cup geometry -> no contact this env.

        // Each fingertip world center (from FK) + the direct fingertip<->cup pairs.
        std::vector<Vec3> finger_centers(fingertips_.size());
        std::vector<CandidatePair> drive_pairs;
        drive_pairs.reserve(fingertips_.size());
        for (size_t f = 0u; f < fingertips_.size(); ++f) {
            const CoResidentFingertip& ft = fingertips_[f];
            const Transform& lp = poses[ft.link];
            finger_centers[f] = lp.position + lp.rotation.Rotate(ft.local_offset);
            CandidatePair p;
            p.a.type = CollidableType::ArticulationLink;
            p.a.react = ReactionProviderKind::ArticulationChainJ;
            p.a.handle = ft.broadphase_handle;
            p.b.type = CollidableType::RigidBody;
            p.b.react = ReactionProviderKind::RigidInvMass;
            p.b.handle = grasp_cup_.broadphase_body_id;
            drive_pairs.push_back(p);
        }

        // stage 6: narrowphase. fingertip(Sphere) x cup(ConvexHull) -> the C3c Convex
        // tier (the cvx::SphereHull closest-point special-case bypasses EPA).
        ShapeResolver resolve = [&](const CollidableRef& ref,
                                    ResolvedShape* out) -> bool {
            if (ref.type == CollidableType::ArticulationLink) {
                for (size_t f = 0u; f < fingertips_.size(); ++f) {
                    if (fingertips_[f].broadphase_handle == ref.handle) {
                        out->type = scene::ShapeType::Sphere;
                        out->prim = FootSpherePrim(finger_centers[f],
                                                   fingertips_[f].radius);
                        out->geom = nullptr;
                        return true;
                    }
                }
                return false;
            }
            if (ref.type == CollidableType::RigidBody &&
                ref.handle == grasp_cup_.broadphase_body_id) {
                out->type = scene::ShapeType::ConvexHull;
                out->geom = &cup_hull;  // the convex-hull seam (geom passthrough).
                return true;
            }
            return false;
        };
        std::vector<ContactManifold> manifolds;
        nuka::collision::BuildContactManifolds(drive_pairs, resolve, &manifolds);

        uint32_t finger_points = 0u;
        for (const auto& m : manifolds) finger_points += m.point_count;
        grasp_reports_[e].finger_contacts = finger_points;
        if (manifolds.empty() || finger_points == 0u) continue;  // cup free this env.

        // stage 7: compliant rows + sides (condim=3 -> normal row + 4 friction spokes
        // per contact point). APPEND this env's rows to the shared buffer.
        nuka::constraint::ContactRowComplianceInputs inputs;
        inputs.vel = 0.0f;
        inputs.invweight = 1.0f;
        inputs.dt = dt_;
        inputs.condim = condim_;
        const std::size_t row_start = rows.RowCount();
        nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
        // Stamp the per-contact friction coefficient (the cone bound is mu * Σλ_n).
        for (std::size_t r = row_start; r < rows.RowCount(); ++r) {
            rows.materials[r].friction = friction_mu_;
        }
        if (rows.RowCount() == row_start) continue;  // no rows emitted this env.
        env_row_range[e] = {row_start, rows.RowCount()};
        grasp_reports_[e].finger_row_count =
            static_cast<uint32_t>(rows.RowCount() - row_start);

        // Build env e's M^-1 tile (CRBA) + its flat prefix-sum qdot slice from the LIVE
        // device state. For N=1 there is one tile / slice; P2.3b concatenates per env
        // (M^-1 tile @ art_index*dof_stride^2, qdot slice @ art_index*dof_stride).
        const std::vector<float> env_minv =
            InverseInertia(context_, live, gravity_z_, dof_stride_);
        minv.insert(minv.end(), env_minv.begin(), env_minv.end());
        for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
            qdot[art_index * dof_stride_ + i] = live.link_velocity[root_link_].v[i];
        }
        for (uint32_t leg = root_link_ + 1u; leg < link_count_; ++leg) {
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
                grasp_reports_[e].any_static_row = true;  // a table -> NO-TABLE gate fails.
            }
            const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
            const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
            const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
            const bool b_cup = s.b.react == ReactionProviderKind::RigidInvMass;
            art_refs[r].a = none_side;
            art_refs[r].b = none_side;
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
            const std::vector<float> chain_j =
                FootChainJ(context_, live, poses, finger_link, contact_point,
                           finger_dir, dof_stride_);
            const uint32_t slot =
                static_cast<uint32_t>(chain_jacobians.size() / dof_stride_);
            chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(),
                                   chain_j.end());
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
        }
        return;
    }
    qdot_before = qdot;

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
    nuka::solver::UnifiedSolve(ctx, cfg);

    // ----- per-env grasp metrics (the FORCE-BALANCE gate numbers) ----------------
    // Computed HERE, while the rows / lambdas are live -- the SAME quantities StepGrasp
    // reports (~:1013-1052). cup_vertical_impulse = Σ over EVERY finger row of (cup-side
    // λ * cup-side jacobian.linear.z): the vertical impulse the fingers deliver to the
    // cup. For a held cup at steady state this balances the cup weight kick m*g*dt. The
    // cross-check cup_dvz_impulse = m_cup * (vz_after_solve - vz_pre_contact) must agree.
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const auto [rbegin, rend] = env_row_range[e];
        double cup_vert_impulse = 0.0;
        float max_lambda = 0.0f;
        for (std::size_t r = rbegin; r < rend; ++r) {
            const ContactRowSides& s = sides[r];
            const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
            const int cup_local = a_cup ? 0 : 1;
            const RowJacobian6 j_cup =
                rows.JacobianForRowBody(static_cast<uint32_t>(r),
                                        static_cast<uint32_t>(cup_local));
            cup_vert_impulse +=
                static_cast<double>(rows.rows[r].lambda) * j_cup.linear.z;
            if (!(rows.rows[r].flags & nuka::constraint::row_flags::Friction))
                max_lambda = std::max(max_lambda, rows.rows[r].lambda);
        }
        grasp_reports_[e].cup_vertical_impulse = cup_vert_impulse;
        grasp_reports_[e].max_lambda = max_lambda;
        const BodyState& cup = bodies_[BodyIndex(e, cup_local_index_)];
        const double cup_mass =
            cup.inv_mass > 0.0f ? 1.0 / static_cast<double>(cup.inv_mass) : 0.0;
        grasp_reports_[e].cup_dvz_impulse =
            cup_mass * (static_cast<double>(cup.linear_velocity.z) - cup_vz_pre_contact[e]);
        grasp_reports_[e].cup_vz = cup.linear_velocity.z;  // velocity unchanged by integrate.
    }

    // ----- SCATTER the post-contact flat-qdot back to the device state -----------
    // The inverse of the prefix-sum pack (per env). For N=1 there is one gripper device;
    // P2.3b scatters per-env. UnifiedSolve mutated `bodies_` (the cup velocity) in place.
    for (uint32_t e = 0u; e < env_count_; ++e) {
        const uint32_t art_index = e;
        for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
            live.link_velocity[root_link_].v[i] = qdot[art_index * dof_stride_ + i];
        }
        for (uint32_t leg = root_link_ + 1u; leg < link_count_; ++leg) {
            const uint32_t col = DofIndexOf(leg);
            if (col < dof_stride_) live.qdot[leg] = qdot[art_index * dof_stride_ + col];
        }
    }
    gripper_device_.link_velocity.CopyFromHost(
        live.link_velocity.data(),
        live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
    gripper_device_.qdot.CopyFromHost(live.qdot.data(),
                                      live.qdot.size() * sizeof(float));
    context_.stream.Synchronize();
    // ===== END CONTACT PHASE -- the position integrate happens in Step(). =========
}

}  // namespace nuka::runtime::coresident
