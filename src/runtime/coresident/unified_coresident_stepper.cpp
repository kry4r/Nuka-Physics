// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- the MULTI-STEP unified co-resident contact
// stepper IMPLEMENTATION (v0.8 W1a). See the header for the design + contract.
// ---------------------------------------------------------------------------
// Every stage REUSES an already-validated piece. Nothing here re-derives ABA,
// integration, broadphase, narrowphase, row emission, or the solve. The ONLY new
// logic is the host orchestration glue that:
//   (a) downloads the live link poses / qdot / base spatial velocity each step,
//   (b) runs the unified contact pipeline (link AABB -> artic<->rigid broadphase
//       -> sphere x box narrowphase -> compliant rows -> UnifiedSolve), exactly
//       as the single-shot co-residence test does, and
//   (c) SCATTERS the post-contact flat-qdot back into the device base spatial
//       velocity + leg qdot, and mutates the box BodyState, BEFORE the stage-11
//       pose integrate.
// ---------------------------------------------------------------------------

#include "runtime/coresident/unified_coresident_stepper.hpp"

#include "collision/aabb.hpp"
#include "collision/analytical_manifold.hpp"     // amf::PrimParams / PrimFrame
#include "collision/candidate_pair.hpp"          // CandidatePair, CollidableRef
#include "collision/contact_stream_driver.hpp"   // BuildContactManifolds, ResolvedShape
#include "collision/rigid_candidate_pairs.hpp"   // BuildArticulationRigidCandidatePairs
#include "constraint/contact_manifold.hpp"
#include "constraint/reaction_provider.hpp"      // ReactionProviderKind
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "collision/convex_narrowphase.hpp"      // cvx::ConvexHullView (cup hull seam)
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses, ArticulationDofCount, ArticulationJointDofCount
#include "runtime/articulation/articulation_drives.hpp"    // LaunchApplyTorqueDriveKernels (the grip drive path)
#include "runtime/articulation/articulation_jacobian.hpp"  // ComputeContactChainJacobians + M/M^-1
#include "runtime/articulation/featherstone_aba.hpp"
#include "solver/unified_solve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::runtime::coresident {

namespace {

namespace articulation = nuka::runtime::articulation;
namespace amf = nuka::collision::amf;
using nuka::collision::AABB;
using nuka::collision::CandidatePair;
using nuka::collision::ExtractLinkShapeAabbs;
using nuka::collision::LinkShapeAabbs;
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

// G0 honesty: the contact-solve spine's factorization / PGS working storage is
// sized kMaxArticulationDof. Reject a bigger articulation at CONSTRUCTION with a
// clear error (the pre-G0 path silently truncated everything past DOF 18).
uint32_t ValidatedDofStride(const articulation::ArticulationHostState& host) {
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    if (dof > articulation::kMaxArticulationDof) {
        throw std::runtime_error(
            "UnifiedCoResidentStepper: articulation DOF count (" +
            std::to_string(dof) + ") exceeds kMaxArticulationDof (" +
            std::to_string(articulation::kMaxArticulationDof) +
            "); the contact-solve spine cannot factor/solve it");
    }
    return dof;
}

// Download the live FK world poses of every link from the current device state.
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

// The 18-wide foot chain-J on `normal`, computed against the LIVE link poses. The
// FK-refresh-correct version: write the current world poses into the host state's
// link_pose before upload (mirrors tests/solver/foot_chain_jacobian.hpp +
// batched_articulated_world.cu stage-4 link_pose refresh). Built directly here so
// the production stepper TU does NOT depend on a test header.
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

AABB BoxAabb(const Vec3& center, float half) {
    AABB b;
    b.min = center - Vec3{half, half, half};
    b.max = center + Vec3{half, half, half};
    return b;
}

amf::PrimParams FootSpherePrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;  // identity rotation; sphere is rotation-invariant.
    return p;
}

amf::PrimParams BoxPrim(const Transform& pose, float half) {
    amf::PrimParams p;
    p.half_extents = Vec3{half, half, half};
    p.frame = amf::BuildPrimFrame(pose);  // bakes the box orientation into cx/cy/cz.
    return p;
}

// A non-cubic box (per-axis half-extents) -- the cup's flat-bottom table-contact proxy.
// A real mug has a flat base, so a box is the FAITHFUL resting-contact representation;
// using it for the cup<->table pair (NOT finger<->cup, NOT render) sidesteps the
// hull-vs-plane coplanar-bottom-face narrowphase instability (named engine debt). The
// SAME stable C3b BoxPlane handler the W1a box<->ground gate uses.
amf::PrimParams BoxPrimXYZ(const Transform& pose, const Vec3& half_extents) {
    amf::PrimParams p;
    p.half_extents = half_extents;
    p.frame = amf::BuildPrimFrame(pose);
    return p;
}

// A static ground PrimParams whose plane normal is world +Z. The C3b BoxPlane
// reads the plane normal from frame.cy, so we put +Z there with an orthonormal
// basis (mirrors MakeGroundPrim in tests/solver/test_foot_ground_subsume.cpp).
amf::PrimParams GroundPrim(float height) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = Vec3{0.0f, 0.0f, 1.0f};   // plane normal = world +Z
    p.frame.cz = Vec3{0.0f, -1.0f, 0.0f};  // right-handed: cx x cy = cz
    p.frame.t = Vec3{0.0f, 0.0f, height};
    return p;
}

// A CollidableRef for the box (RigidInvMass) -- the movable rigid side.
CollidableRef MakeBoxRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::RigidBody;
    ref.react = ReactionProviderKind::RigidInvMass;
    ref.handle = handle;
    return ref;
}

// A CollidableRef for the static ground (StaticWorld -> StaticNull reaction). Its
// effective inv-mass is 0 and its apply is a no-op, so a box<->ground row has a
// well-defined zero-reaction side -- mirrors MakeGroundRef in the subsume test.
CollidableRef MakeGroundRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = handle;
    return ref;
}

}  // namespace

UnifiedCoResidentStepper::UnifiedCoResidentStepper(
    const phi::DeviceContext& context,
    const articulation::ArticulationHostState& host,
    const scene::CookedShapeTable& cooked_shapes, const CoResidentFoot& foot,
    const CoResidentBox& box, const CoResidentGround& ground,
    const runtime::rigid::BodyState& box_state, float gravity_z, float dt)
    : context_(context),
      host_proto_(host),
      cooked_shapes_(cooked_shapes),
      foot_(foot),
      box_(box),
      ground_(ground),
      box_state_(box_state),
      gravity_z_(gravity_z),
      dt_(dt) {
    device_ = articulation::UploadArticulationState(context_, host_proto_);
    dof_stride_ = ValidatedDofStride(host_proto_);
    root_link_ = host_proto_.articulation_link_offset[0];
    base_dof_ = articulation::ArticulationJointDofCount(host_proto_.joint_type[root_link_]);
}

// ---------------------------------------------------------------------------
// C7b-1 GRASP constructor: N fingertips + a convex-hull cup + a constant grip
// torque + friction, NO table. The foot/box/ground members stay default (inert);
// grasp_mode_ routes Step() to StepGrasp().
// ---------------------------------------------------------------------------
UnifiedCoResidentStepper::UnifiedCoResidentStepper(
    const phi::DeviceContext& context,
    const articulation::ArticulationHostState& host, const GraspConfig& grasp,
    float gravity_z, float dt)
    : context_(context),
      host_proto_(host),
      box_state_(grasp.cup_state),   // the cup IS the movable rigid body (box_state_).
      gravity_z_(gravity_z),
      dt_(dt),
      grasp_mode_(true),
      grasp_(grasp) {
    device_ = articulation::UploadArticulationState(context_, host_proto_);
    dof_stride_ = ValidatedDofStride(host_proto_);
    root_link_ = host_proto_.articulation_link_offset[0];
    base_dof_ = articulation::ArticulationJointDofCount(host_proto_.joint_type[root_link_]);

    // Upload the constant grip torque + force limits ONCE; re-applied to tau each
    // step (idempotent). Sized to the device link count.
    const uint32_t link_count = host_proto_.TotalLinkCount();
    std::vector<float> torque = grasp_.grip_torque;
    torque.resize(link_count, 0.0f);
    std::vector<float> limits = grasp_.drive_force_limits;
    limits.resize(link_count, 0.0f);
    grip_torque_dev_ = nuka::phi::UploadVector(torque);
    grip_limits_dev_ = nuka::phi::UploadVector(limits);

    // The cup<->table support starts at the config flag (inert by default). A caller
    // that constructs with has_table=false + never calls SetTableEnabled keeps the
    // exact prior StepGrasp behavior (no table pair, no StaticNull side, D1-exact).
    table_enabled_ = grasp_.has_table;
}

// ---------------------------------------------------------------------------
// S3 STANDING constructor: N foot spheres on a STATIC +Z ground + condim=3 friction
// + a DRIVE-torque balance path. The foot/box/cup members stay default (inert);
// stand_mode_ routes Step() to StepStand(). NO movable rigid body (box_state_ stays
// inv_mass=0 -> every box/cup integrate self-inerts).
// ---------------------------------------------------------------------------
UnifiedCoResidentStepper::UnifiedCoResidentStepper(
    const phi::DeviceContext& context,
    const articulation::ArticulationHostState& host, const StandConfig& stand,
    float gravity_z, float dt)
    : context_(context),
      host_proto_(host),
      gravity_z_(gravity_z),
      dt_(dt),
      stand_mode_(true),
      stand_(stand) {
    device_ = articulation::UploadArticulationState(context_, host_proto_);
    dof_stride_ = ValidatedDofStride(host_proto_);
    root_link_ = host_proto_.articulation_link_offset[0];
    base_dof_ = articulation::ArticulationJointDofCount(host_proto_.joint_type[root_link_]);

    // Upload the drive torque + force limits ONCE (re-applied each step via the SAME
    // device drive seam SetGripTorque overwrites). The balance controller overwrites
    // the torque every step from the Download'd state, so the initial values are just
    // a zero seed sized to the device link count.
    const uint32_t link_count = host_proto_.TotalLinkCount();
    std::vector<float> torque = stand_.drive_torque;
    torque.resize(link_count, 0.0f);
    std::vector<float> limits = stand_.drive_force_limits;
    limits.resize(link_count, 0.0f);
    grip_torque_dev_ = nuka::phi::UploadVector(torque);
    grip_limits_dev_ = nuka::phi::UploadVector(limits);
}

// ---------------------------------------------------------------------------
// C7b-2 SHOWCASE constructor: the CONJUNCTION (stand AND grasp). The cup IS the one
// movable rigid body (box_state_); the drive torque (legs RL + arm reach + finger grip)
// is uploaded once + overwritten each step via SetGripTorque; the table starts at the
// config flag. stand_grasp_mode_ routes Step() to StepStandGrasp().
// ---------------------------------------------------------------------------
UnifiedCoResidentStepper::UnifiedCoResidentStepper(
    const phi::DeviceContext& context,
    const articulation::ArticulationHostState& host,
    const StandGraspConfig& stand_grasp, float gravity_z, float dt)
    : context_(context),
      host_proto_(host),
      box_state_(stand_grasp.cup_state),   // the cup IS the movable rigid body.
      gravity_z_(gravity_z),
      dt_(dt),
      stand_grasp_mode_(true),
      stand_grasp_(stand_grasp) {
    device_ = articulation::UploadArticulationState(context_, host_proto_);
    dof_stride_ = ValidatedDofStride(host_proto_);
    root_link_ = host_proto_.articulation_link_offset[0];
    base_dof_ = articulation::ArticulationJointDofCount(host_proto_.joint_type[root_link_]);

    // Upload the drive torque + force limits ONCE (re-applied each step via the SAME
    // device drive seam SetGripTorque overwrites). The host overwrites it every step
    // (legs from the RL policy, arm from the reach trajectory, fingers from the grip),
    // so the initial values are a seed sized to the device link count.
    const uint32_t link_count = host_proto_.TotalLinkCount();
    std::vector<float> torque = stand_grasp_.drive_torque;
    torque.resize(link_count, 0.0f);
    std::vector<float> limits = stand_grasp_.drive_force_limits;
    limits.resize(link_count, 0.0f);
    grip_torque_dev_ = nuka::phi::UploadVector(torque);
    grip_limits_dev_ = nuka::phi::UploadVector(limits);

    // The cup<->table support starts at the config flag; SetTableEnabled toggles it.
    table_enabled_ = stand_grasp_.has_table;
}

// The flat-qdot prefix-sum DOF index of a device link: Σ JointDofCount over the
// articulation's links in [offset, link). Generic over fixed/floating roots.
uint32_t UnifiedCoResidentStepper::DofIndexOf(uint32_t link) const {
    uint32_t idx = 0u;
    for (uint32_t k = root_link_; k < link; ++k) {
        idx += articulation::ArticulationJointDofCount(host_proto_.joint_type[k]);
    }
    return idx;
}

CoResidentStepReport UnifiedCoResidentStepper::Step() {
    if (stand_grasp_mode_) return StepStandGrasp();
    if (stand_mode_) return StepStand();
    if (grasp_mode_) return StepGrasp();
    CoResidentStepReport report;
    auto view = device_.View();
    const uint32_t link_count = host_proto_.TotalLinkCount();

    // ----- stage 1/2: ABA accelerations -> qddot (tau held at 0, deterministic). --
    articulation::FeatherstoneAba::ComputeAccelerations(context_, view, gravity_z_);

    // ----- stage 3: velocity integrate (articulation). The split velocity half +
    //       the floating-base velocity half, EXACTLY as Step() stage-3 does. -------
    articulation::FeatherstoneAba::IntegrateVelocity(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, view, dt_,
                                                                 gravity_z_);
    // ----- stage 3 (box): velocity integrate the rigid box (vel += gravity*dt). ---
    if (box_state_.inv_mass > 0.0f) {
        box_state_.linear_velocity.z += gravity_z_ * dt_;
    }

    // ===== CONTACT PHASE (the swap) ==========================================
    // Snapshot the post-velocity-integrate state for the host-side pipeline.
    articulation::ArticulationHostState live = host_proto_;
    articulation::DownloadArticulationState(device_, &live);

    // stage 4: refresh FK world poses (the contact geometry needs current q/base).
    const std::vector<Transform> poses = DownloadWorldPoses(context_, device_, link_count);

    // The foot world center (the material contact point) from FK.
    const Transform& calf = poses[foot_.calf_link];
    const Vec3 foot_center = calf.position + calf.rotation.Rotate(foot_.local_offset);

    // stage 5a: per-link world AABBs from the FK (commit 1: link_aabb.hpp).
    const LinkShapeAabbs links =
        ExtractLinkShapeAabbs(cooked_shapes_, poses, live.link_body);
    if (links.aabbs.empty()) {
        // No collidable link shapes -> no contact this step. Still integrate pose.
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        return report;
    }
    std::vector<uint32_t> link_contypes(links.aabbs.size(), 1u);
    std::vector<uint32_t> link_conaff(links.aabbs.size(), 1u);

    // stage 5b: the box world AABB (rigid side) + its broadphase metadata.
    std::vector<AABB> rigid_aabbs = {BoxAabb(box_state_.position, box_.half_extent)};
    std::vector<uint32_t> rigid_body_ids = {box_.broadphase_body_id};
    std::vector<uint32_t> rigid_contypes = {1u};
    std::vector<uint32_t> rigid_conaff = {1u};
    nuka::phi::Buffer d_rigid = nuka::phi::UploadVector(rigid_aabbs);

    // stage 5c: the artic-link<->rigid broadphase (commit 2). Find the (foot, box).
    auto stream = nuka::collision::BuildArticulationRigidCandidatePairs(
        context_, static_cast<const AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid_aabbs.size()), rigid_body_ids.data(),
        rigid_contypes.data(), rigid_conaff.data(), links, link_contypes.data(),
        link_conaff.data(), /*excluded_body_pairs=*/{});
    const std::vector<CandidatePair> pairs = stream.DownloadPairs();

    // Keep only the (foot calf link, box) cross pairs to drive the narrowphase.
    std::vector<CandidatePair> drive_pairs;
    for (const auto& p : pairs) {
        const bool a_foot = p.a.type == CollidableType::ArticulationLink &&
                            p.a.handle == foot_.calf_link;
        const bool b_box = p.b.type == CollidableType::RigidBody &&
                           p.b.handle == box_.broadphase_body_id;
        const bool a_box = p.a.type == CollidableType::RigidBody &&
                           p.a.handle == box_.broadphase_body_id;
        const bool b_foot = p.b.type == CollidableType::ArticulationLink &&
                            p.b.handle == foot_.calf_link;
        if ((a_foot && b_box) || (a_box && b_foot)) {
            drive_pairs.push_back(p);
        }
    }
    report.pair_found = !drive_pairs.empty();

    // stage 5d: the box<->ground candidate pair, emitted DIRECTLY (the box AABB vs
    // a static ground plane is trivial broadphase -- the box always overlaps the
    // half-space if its bottom is at/below the plane). This routes the box's SUPPORT
    // through the SAME unified spine as foot<->box, replacing the deleted hand clamp.
    // Side A = box (RigidInvMass), side B = ground (StaticWorld -> StaticNull).
    CandidatePair ground_pair;
    ground_pair.a = MakeBoxRef(box_.broadphase_body_id);
    ground_pair.b = MakeGroundRef(ground_.broadphase_id);
    drive_pairs.push_back(ground_pair);
    report.ground_pair_found = true;

    // stage 6: narrowphase via the resolver-decoupled driver. The resolver supplies
    // the foot sphere (FK center), the box (its BodyState pose), and the static
    // ground plane (+Z normal). foot<->box -> sphere x box; box<->ground -> box x
    // plane (C3b BoxPlane, already in the dispatch table) -- both analytical.
    const Transform box_pose{box_state_.position, box_state_.orientation};
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::ArticulationLink && ref.handle == foot_.calf_link) {
            out->type = scene::ShapeType::Sphere;
            out->prim = FootSpherePrim(foot_center, foot_.radius);
            return true;
        }
        if (ref.type == CollidableType::RigidBody && ref.handle == box_.broadphase_body_id) {
            out->type = scene::ShapeType::Box;
            out->prim = BoxPrim(box_pose, box_.half_extent);
            return true;
        }
        if (ref.type == CollidableType::StaticWorld && ref.handle == ground_.broadphase_id) {
            out->type = scene::ShapeType::Plane;
            out->prim = GroundPrim(ground_.height);
            return true;
        }
        return false;
    };
    std::vector<ContactManifold> manifolds;
    nuka::collision::BuildContactManifolds(drive_pairs, resolve, &manifolds);
    report.manifold_count = static_cast<uint32_t>(manifolds.size());
    if (manifolds.empty()) {
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        return report;
    }

    // stage 7: compliant rows + sides (condim=1, a clean normal row per manifold
    // point). Both pairs feed the SAME rows buffer + sides stream in ONE call -- the
    // foot<->box normal row(s) AND the box<->ground corner row(s) coexist; the per-
    // row sides carry each side's reaction kind (ArticulationChainJ / RigidInvMass /
    // StaticNull), so the solve dispatches each side class-blind.
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = dt_;
    inputs.condim = 1u;
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    report.row_count = rows.RowCount();
    if (rows.RowCount() == 0u || sides.empty()) {
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        return report;
    }

    // The foot chain-J / M^-1 are needed only if a foot<->box row was emitted. Build
    // them lazily on the first foot<->box row encountered (using that row's normal).
    const std::vector<float> minv =
        InverseInertia(context_, live, gravity_z_, dof_stride_);
    std::vector<float> chain_jacobians;  // one dof_stride-wide slot per foot row.

    // ----- assemble the flat dof_stride-wide qdot from the LIVE device state ------
    // Generic prefix-sum packing (DofIndexOf): a FloatingBase root fills its 6 base
    // DOFs from link_velocity[root].v (omega-first); each scalar joint fills its own
    // column from live.qdot[link]. For go2 (floating root @ dof_index 0) this is
    // base@[0..5] + leg k@[6+k] -- BYTE-FOR-BYTE the previous hard-coded packing.
    std::vector<float> qdot(dof_stride_, 0.0f);
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
        qdot[i] = live.link_velocity[root_link_].v[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) qdot[col] = live.qdot[leg];
    }
    std::vector<float> qdot_before = qdot;

    // ----- wire each row's body indices + reaction refs by side.react -------------
    // Both pairs share ONE rows buffer. We walk every row and dispatch on its sides:
    //   * ArticulationChainJ + RigidInvMass  -> a foot<->box row. The articulation
    //       side gets the coloring key (body_count+art_index) + a chain-J slot; the
    //       box side gets body index 0.
    //   * RigidInvMass + StaticNull          -> a box<->ground row. The box side
    //       gets body index 0; the static side gets kInvalidBodyIndex (no reaction,
    //       no coloring conflict). No articulation ref.
    // The box ALWAYS gets body index 0, so EVERY row that mutates bodies[0] shares
    // that index -> the graph coloring SERIALIZES them (no race on the box velocity,
    // D1-safe). The articulation key (body_count+art_index) likewise serializes all
    // foot rows on the shared base DOFs.
    const uint32_t kArtIndex = 0u;
    const uint32_t body_count = 1u;  // ctx.state = [box].
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    RowArticulationSide none_side{};
    float max_ground_depth = 0.0f;
    float max_ground_lambda = 0.0f;
    uint32_t ground_rows = 0u;
    bool any_foot_row = false;
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
        const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
        const bool a_rigid = s.a.react == ReactionProviderKind::RigidInvMass;
        const bool b_rigid = s.b.react == ReactionProviderKind::RigidInvMass;
        const bool a_static = s.a.react == ReactionProviderKind::StaticNull;
        const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
        art_refs[r].a = none_side;
        art_refs[r].b = none_side;

        if ((a_art && b_rigid) || (b_art && a_rigid)) {
            // ----- foot<->box row -----
            const int foot_local = a_art ? 0 : 1;
            const int box_local = a_art ? 1 : 0;
            // The foot side's RowJacobian6.linear IS the contact normal; project the
            // chain-J on it (the reduced-coordinate reaction the kernel uses).
            const RowJacobian6 j_foot =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(foot_local));
            const Vec3 foot_normal = j_foot.linear;
            const Vec3 contact_point = s.contact_point;
            const std::vector<float> chain_j =
                FootChainJ(context_, live, poses, foot_.calf_link, contact_point,
                           foot_normal, dof_stride_);
            const uint32_t slot = static_cast<uint32_t>(chain_jacobians.size() / dof_stride_);
            chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(), chain_j.end());
            rows.body_indices[2u * r + static_cast<uint32_t>(box_local)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(foot_local)] =
                body_count + kArtIndex;
            const RowArticulationSide foot_side{kArtIndex, slot};
            art_refs[r].a = (foot_local == 0) ? foot_side : none_side;
            art_refs[r].b = (foot_local == 1) ? foot_side : none_side;
            if (!any_foot_row) {
                report.contact_depth = rows.materials[r].position_error;
                any_foot_row = true;
            } else {
                report.contact_depth =
                    std::max(report.contact_depth, rows.materials[r].position_error);
            }
        } else if ((a_rigid && b_static) || (b_rigid && a_static)) {
            // ----- box<->ground row -----
            const int box_local = a_rigid ? 0 : 1;
            const int static_local = a_rigid ? 1 : 0;
            rows.body_indices[2u * r + static_cast<uint32_t>(box_local)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(static_local)] =
                nuka::constraint::kInvalidBodyIndex;
            // art_refs[r] stays {none,none} -> the articulation arm never fires.
            ++ground_rows;
            max_ground_depth =
                std::max(max_ground_depth, rows.materials[r].position_error);
        }
        // (Any other side combination cannot occur in this scene; skip safely.)
    }
    report.ground_row_count = ground_rows;
    report.ground_depth = max_ground_depth;

    // ----- the box BodyState (the rigid body that reacts two-way) ----------------
    std::vector<BodyState> bodies = {box_state_};
    const Vec3 box_lin_before = box_state_.linear_velocity;

    // ----- stage 10: the unified two-way solve (ALL rows together) ---------------
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = dt_;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians = chain_jacobians.empty() ? nullptr : &chain_jacobians;
    ctx.articulation.inertia_m_inv = &minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = dof_stride_;
    nuka::solver::UnifiedSolve(ctx, cfg);

    // ----- read back the per-pair report metrics from the solved rows ------------
    float foot_lambda = 0.0f;
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool art_row = s.a.react == ReactionProviderKind::ArticulationChainJ ||
                             s.b.react == ReactionProviderKind::ArticulationChainJ;
        const bool ground_row = s.a.react == ReactionProviderKind::StaticNull ||
                                s.b.react == ReactionProviderKind::StaticNull;
        if (art_row) foot_lambda = std::max(foot_lambda, rows.rows[r].lambda);
        if (ground_row) max_ground_lambda = std::max(max_ground_lambda, rows.rows[r].lambda);
    }
    report.lambda = foot_lambda;
    report.ground_lambda = max_ground_lambda;

    // ----- read back: the box BodyState + the recoil metrics ---------------------
    box_state_ = bodies[0];
    const Vec3 box_dv = box_state_.linear_velocity - box_lin_before;
    report.box_dv_norm = static_cast<double>(box_dv.Length());
    double dn = 0.0;
    for (uint32_t i = 0u; i < dof_stride_; ++i)
        dn += std::fabs(static_cast<double>(qdot[i]) - qdot_before[i]);
    report.qdot_delta_l1 = dn;

    // ----- SCATTER the post-contact flat-qdot back into the device state ---------
    // The inverse of the prefix-sum pack: base DOFs -> link_velocity[root]; each
    // scalar joint column -> live.qdot[link]. Byte-identical to the old fixed layout.
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
        live.link_velocity[root_link_].v[i] = qdot[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) live.qdot[leg] = qdot[col];
    }
    // Push ONLY the mutated velocity buffers back to the device (q / base_pose /
    // accelerations are untouched by the solve and stay device-current for the
    // pose integrate below). A full re-upload would also stomp the (unchanged) q,
    // which is harmless but wasteful; we copy the two velocity arrays only.
    device_.link_velocity.CopyFromHost(
        live.link_velocity.data(),
        live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
    device_.qdot.CopyFromHost(live.qdot.data(), live.qdot.size() * sizeof(float));
    context_.stream.Synchronize();

    // ===== END CONTACT PHASE ================================================

    // ----- stage 11: position integrate (articulation) with the POST-contact
    //       velocity, matching Step()'s stage-11 ordering. -------------------------
    view = device_.View();
    articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
    // ----- stage 11 (box): position integrate (position += vel*dt; orientation
    //       from angular velocity, first-order normalized, matching the base). -----
    IntegrateBoxPosition();
    return report;
}

// ===========================================================================
// C7b-1 GRASP STEP: hold the cup against gravity by finger friction alone.
// ===========================================================================
// Mirrors Step()'s stage order but: (a) re-applies the constant grip torque to tau
// BEFORE ComputeAccelerations (the DRIVE path -> the fingers squeeze inward), (b)
// emits N fingertip<->cup contacts (NO table), (c) uses condim=3 (1 normal + 2
// tangential friction directions, discretized as a 4-spoke pyramid per contact) so
// FRICTION holds the cup, (d) reports the
// vertical contact impulse the fingers deliver to the cup (the force-balance gate).
CoResidentStepReport UnifiedCoResidentStepper::StepGrasp() {
    CoResidentStepReport report;
    auto view = device_.View();
    const uint32_t link_count = host_proto_.TotalLinkCount();

    // ----- stage 0: the DRIVE path. Re-apply the constant grip torque to tau. -----
    // LaunchApplyTorqueDriveKernels writes state.tau[link] = clamp(torque_input);
    // ComputeAccelerations then READS tau in its bias-force pass (never zeroes it).
    // Re-applying the same constant each step is idempotent. This is what makes the
    // fingers actively squeeze -- the W1a path ran passive (tau held at 0).
    articulation::LaunchApplyTorqueDriveKernels(
        context_, view, static_cast<const float*>(grip_torque_dev_.Data()),
        static_cast<const float*>(grip_limits_dev_.Data()));

    // ----- stage 1/2: ABA accelerations -> qddot (now WITH the grip tau). ---------
    articulation::FeatherstoneAba::ComputeAccelerations(context_, view, gravity_z_);

    // ----- stage 3: velocity integrate (articulation halves + the cup). ----------
    articulation::FeatherstoneAba::IntegrateVelocity(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, view, dt_,
                                                                 gravity_z_);
    if (box_state_.inv_mass > 0.0f) {
        box_state_.linear_velocity.z += gravity_z_ * dt_;  // cup gravity kick.
    }
    // The cup vertical velocity AFTER gravity but BEFORE the contact solve -- the
    // baseline the contact impulse must lift back to balance the weight.
    const float cup_vz_pre_contact = box_state_.linear_velocity.z;

    // ===== CONTACT PHASE (fingertips <-> cup, NO table) ======================
    articulation::ArticulationHostState live = host_proto_;
    articulation::DownloadArticulationState(device_, &live);
    const std::vector<Transform> poses = DownloadWorldPoses(context_, device_, link_count);

    // The cup convex hull view in WORLD space (mesh-local verts + live cup pose).
    const Transform cup_pose{box_state_.position, box_state_.orientation};
    nuka::collision::cvx::ConvexHullView cup_hull;
    cup_hull.verts = grasp_.cup.hull_verts.data();
    cup_hull.vcount = grasp_.cup.VertexCount();
    cup_hull.frame = amf::BuildPrimFrame(cup_pose);

    // Each fingertip world center (from FK) + the direct fingertip<->cup pairs. We
    // DIRECT-EMIT the known (fingertip, cup) pairs (the broadphase is proven by W1a/
    // the foot-box gate; the spike's risk is the SOLVE, not the broadphase). A cup
    // with NO hull geometry (the drive-path probe) emits NO contact -> the fingers
    // are free to accelerate under the grip torque alone.
    std::vector<Vec3> finger_centers(grasp_.fingertips.size());
    std::vector<CandidatePair> drive_pairs;
    if (cup_hull.vcount == 0u) {
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        report.cup_vz = box_state_.linear_velocity.z;
        report.cup_z = box_state_.position.z;
        return report;
    }
    drive_pairs.reserve(grasp_.fingertips.size());
    for (size_t f = 0u; f < grasp_.fingertips.size(); ++f) {
        const CoResidentFingertip& ft = grasp_.fingertips[f];
        const Transform& lp = poses[ft.link];
        finger_centers[f] = lp.position + lp.rotation.Rotate(ft.local_offset);
        CandidatePair p;
        p.a.type = CollidableType::ArticulationLink;
        p.a.react = ReactionProviderKind::ArticulationChainJ;
        p.a.handle = ft.broadphase_handle;
        p.b.type = CollidableType::RigidBody;
        p.b.react = ReactionProviderKind::RigidInvMass;
        p.b.handle = grasp_.cup.broadphase_body_id;
        drive_pairs.push_back(p);
    }

    // ----- C7b-2a-v2 OPTIONAL cup<->table support pair (the LIFT choreography) -----
    // Direct-emit the cup<->table pair (cup AABB vs a static +Z plane is trivial
    // broadphase). Side A = cup (RigidInvMass), side B = table (StaticWorld ->
    // StaticNull). Routed through the SAME unified spine as the fingertip rows. Only
    // when has_table AND the runtime toggle is on -- removing the table == toggling
    // this off, after which no StaticNull side is emitted (any_static_row -> false).
    const bool emit_table = grasp_.has_table && table_enabled_;
    if (emit_table) {
        CandidatePair tp;
        tp.a = MakeBoxRef(grasp_.cup.broadphase_body_id);   // cup = RigidInvMass.
        tp.b = MakeGroundRef(grasp_.table_broadphase_id);   // table = StaticNull (+Z plane).
        drive_pairs.push_back(tp);
    }

    // stage 6: narrowphase via the driver. fingertip(Sphere) x cup(ConvexHull) ->
    // the C3c Convex tier (GJK/EPA + face-clip), already in the dispatch table.
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::ArticulationLink) {
            for (size_t f = 0u; f < grasp_.fingertips.size(); ++f) {
                if (grasp_.fingertips[f].broadphase_handle == ref.handle) {
                    out->type = scene::ShapeType::Sphere;
                    out->prim = FootSpherePrim(finger_centers[f],
                                               grasp_.fingertips[f].radius);
                    out->geom = nullptr;
                    return true;
                }
            }
            return false;
        }
        if (ref.type == CollidableType::RigidBody &&
            ref.handle == grasp_.cup.broadphase_body_id) {
            out->type = scene::ShapeType::ConvexHull;
            out->geom = &cup_hull;  // the convex-hull seam (geom_a/geom_b passthrough).
            return true;
        }
        if (ref.type == CollidableType::StaticWorld &&
            ref.handle == grasp_.table_broadphase_id) {
            out->type = scene::ShapeType::Plane;
            out->prim = GroundPrim(grasp_.table_height);  // +Z plane at table_height.
            out->geom = nullptr;
            return true;
        }
        return false;
    };
    std::vector<ContactManifold> manifolds;
    nuka::collision::BuildContactManifolds(drive_pairs, resolve, &manifolds);
    report.manifold_count = static_cast<uint32_t>(manifolds.size());

    // Count finger contact points (the cup is pinched at >=1 point per finger).
    uint32_t finger_points = 0u;
    for (const auto& m : manifolds) finger_points += m.point_count;
    report.finger_contacts = finger_points;
    report.pair_found = finger_points > 0u;

#ifdef NUKA_GRASP_DIAG
    if (finger_points < grasp_.fingertips.size()) {
        std::fprintf(stderr, "[DIAG] DROP: cup pos=(%.5f,%.5f,%.5f) finger_pts=%u/%zu  ",
                     box_state_.position.x, box_state_.position.y, box_state_.position.z,
                     finger_points, grasp_.fingertips.size());
        for (size_t f = 0u; f < grasp_.fingertips.size(); ++f) {
            const Vec3 c = finger_centers[f];
            const float gap = (c - box_state_.position).Length();  // crude separation.
            const uint32_t pc = f < manifolds.size() ? manifolds[f].point_count : 0u;
            const float pen = (pc > 0u) ? manifolds[f].points[0].penetration : -999.0f;
            std::fprintf(stderr, "f%zu[cx=%.5f pc=%u pen=%.5f gap=%.5f] ", f, c.x, pc,
                         pen, gap);
        }
        std::fprintf(stderr, "\n");
    }
#endif

    if (manifolds.empty() || finger_points == 0u) {
        // No contact -> the cup free-falls. Still integrate (the BITE path).
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        report.cup_vz = box_state_.linear_velocity.z;
        report.cup_z = box_state_.position.z;
        return report;
    }

    // stage 7: compliant rows + sides (condim=3 -> normal row + 4 friction spokes
    // per contact point). mu in RowMaterial.friction (the friction cone bound is
    // applied by the solver as [0, mu*TotalNormalLambda(group)] per spoke). FRICTION
    // is what holds the cup -- a frictionless (condim=1) grasp could not.
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = dt_;
    inputs.condim = grasp_.condim;
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    report.row_count = rows.RowCount();
    // Stamp the per-contact friction coefficient (the merged manifold default is
    // 0.5; we set the spike's mu explicitly so the cone bound is the spike's). A
    // cup<->table row gets table_mu; every finger row gets friction_mu.
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool is_table = s.a.react == ReactionProviderKind::StaticNull ||
                              s.b.react == ReactionProviderKind::StaticNull;
        rows.materials[r].friction = is_table ? grasp_.table_mu : grasp_.friction_mu;
    }
    if (rows.RowCount() == 0u || sides.empty()) {
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        report.cup_vz = box_state_.linear_velocity.z;
        report.cup_z = box_state_.position.z;
        return report;
    }

    const std::vector<float> minv =
        InverseInertia(context_, live, gravity_z_, dof_stride_);
    std::vector<float> chain_jacobians;  // one dof_stride-wide slot per finger row.

    // ----- the flat dof_stride-wide qdot (prefix-sum packing). For a Fixed-root
    // gripper base_dof_==0 -> every column is a finger joint qdot. ------------------
    std::vector<float> qdot(dof_stride_, 0.0f);
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
        qdot[i] = live.link_velocity[root_link_].v[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) qdot[col] = live.qdot[leg];
    }
    std::vector<float> qdot_before = qdot;
#ifdef NUKA_H1_GRASP_DIAG
    {
        std::fprintf(stderr, "[QDOT] packed finger qdot (grip-driven inward vel): ");
        for (uint32_t c = 0u; c < dof_stride_; ++c)
            std::fprintf(stderr, "%.5e ", qdot[c]);
        std::fprintf(stderr, " | cup vz pre-contact=%.5e\n", cup_vz_pre_contact);
    }
#endif

    // ----- wire each row: the articulation (finger) side gets a chain-J slot +
    // coloring key body_count+art_index; the cup side gets BodyState index 0. -------
    const uint32_t kArtIndex = 0u;
    const uint32_t body_count = 1u;  // ctx.state = [cup].
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    RowArticulationSide none_side{};
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        if (s.a.react == ReactionProviderKind::StaticNull ||
            s.b.react == ReactionProviderKind::StaticNull) {
            report.any_static_row = true;  // there must be NO table -> assert in gate.
        }
        const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
        const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
        const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
        const bool b_cup = s.b.react == ReactionProviderKind::RigidInvMass;
        art_refs[r].a = none_side;
        art_refs[r].b = none_side;
        const bool a_static = s.a.react == ReactionProviderKind::StaticNull;
        const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
        if ((a_cup && b_static) || (b_cup && a_static)) {
            // ----- cup<->table row (the LIFT support) -----
            // Mirrors Step()'s box<->ground wiring: cup side -> BodyState index 0,
            // static side -> kInvalidBodyIndex (no reaction, no coloring conflict).
            // art_refs[r] stays {none,none} -> the articulation arm never fires.
            const int cup_l = a_cup ? 0 : 1;
            const int static_l = a_cup ? 1 : 0;
            rows.body_indices[2u * r + static_cast<uint32_t>(cup_l)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(static_l)] =
                nuka::constraint::kInvalidBodyIndex;
            continue;  // no chain-J for the static side.
        }
        if ((a_art && b_cup) || (b_art && a_cup)) {
            const int finger_local = a_art ? 0 : 1;
            const int cup_local = a_art ? 1 : 0;
            const RowJacobian6 j_finger =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(finger_local));
            // The finger side's RowJacobian6.linear is the contact direction (the
            // normal for the normal row, +/-tangent for a friction spoke). Project
            // the chain-J on it -- the reduced-coordinate reaction the kernel uses.
            const Vec3 finger_dir = j_finger.linear;
            const Vec3 contact_point = sides[r].contact_point;
            // Which device link owns this finger contact. The row carries the
            // fingertip's broadphase_handle on its ArticulationChainJ side; map it
            // back to the fingertip's REAL device link via the fingertips vector.
            // INERT BY DEFAULT: every existing fingertip sets broadphase_handle ==
            // link, so this returns the handle unchanged (byte-for-byte the prior
            // behavior). It only differs when a dense multi-sphere-per-link layout
            // gives each sphere a UNIQUE handle (so the resolver can pick the right
            // sphere geometry) while keeping `link` = the true articulation link the
            // chain-J needs. (A handle with no matching fingertip falls back to the
            // handle itself -- the prior assumption.)
            const uint32_t finger_handle = sides[r].a.react ==
                    ReactionProviderKind::ArticulationChainJ
                ? sides[r].a.handle : sides[r].b.handle;
            uint32_t finger_link = finger_handle;
            for (size_t f = 0u; f < grasp_.fingertips.size(); ++f) {
                if (grasp_.fingertips[f].broadphase_handle == finger_handle) {
                    finger_link = grasp_.fingertips[f].link;
                    break;
                }
            }
            const std::vector<float> chain_j =
                FootChainJ(context_, live, poses, finger_link, contact_point,
                           finger_dir, dof_stride_);
#ifdef NUKA_H1_GRASP_DIAG
            {
                double jnorm = 0.0;
                int argmax = -1;
                float amax = 0.0f;
                for (uint32_t c = 0u; c < dof_stride_; ++c) {
                    jnorm += static_cast<double>(chain_j[c]) * chain_j[c];
                    if (std::fabs(chain_j[c]) > amax) { amax = std::fabs(chain_j[c]); argmax = static_cast<int>(c); }
                }
                std::fprintf(stderr, "[CHAINJ] row=%u finger_link=%u dir=(%.3f,%.3f,%.3f) "
                             "|J|=%.5f argmaxcol=%d val=%.5f dof_stride=%u\n",
                             r, finger_link, finger_dir.x, finger_dir.y, finger_dir.z,
                             std::sqrt(jnorm), argmax, argmax >= 0 ? chain_j[argmax] : 0.0f,
                             dof_stride_);
            }
#endif
            const uint32_t slot =
                static_cast<uint32_t>(chain_jacobians.size() / dof_stride_);
            chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(),
                                   chain_j.end());
            rows.body_indices[2u * r + static_cast<uint32_t>(cup_local)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(finger_local)] =
                body_count + kArtIndex;
            const RowArticulationSide finger_side{kArtIndex, slot};
            art_refs[r].a = (finger_local == 0) ? finger_side : none_side;
            art_refs[r].b = (finger_local == 1) ? finger_side : none_side;
        }
    }

    // ----- the cup BodyState (the rigid body that reacts two-way) -----------------
    std::vector<BodyState> bodies = {box_state_};
    const Vec3 cup_lin_before = box_state_.linear_velocity;

    // ----- stage 10: the unified two-way solve (ALL finger rows together) --------
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = dt_;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians =
        chain_jacobians.empty() ? nullptr : &chain_jacobians;
    ctx.articulation.inertia_m_inv = &minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = dof_stride_;
    nuka::solver::UnifiedSolve(ctx, cfg);

    // ----- the FORCE-BALANCE gate numbers (the discriminating hold metric) -------
    // (1) Σ over EVERY finger row of (cup-side λ * cup-side jacobian.linear.z): the
    //     vertical impulse the fingers deliver to the cup (normal + friction spokes).
    //     For a held cup at steady state this balances the cup weight kick m*g*dt.
    // The FINGER vertical impulse (the hold/LIFT discriminator) sums ONLY the rows
    // whose OTHER side is the articulation (finger<->cup). A cup<->table row is
    // excluded so the LIFT gate reads the impulse the HAND delivers, separate from the
    // table support (report.table_lambda). max_lambda/max_depth likewise skip table
    // rows so the contact-depth metric stays the finger penetration.
    double cup_vert_impulse = 0.0;
    double finger_vimp_normal = 0.0, finger_vimp_friction = 0.0;
    double table_vert_impulse = 0.0;
    float max_depth = 0.0f, max_lambda = 0.0f;
    uint32_t table_rows = 0u;
    float max_table_lambda = 0.0f;
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
        const int cup_local = a_cup ? 0 : 1;
        const RowJacobian6 j_cup =
            rows.JacobianForRowBody(r, static_cast<uint32_t>(cup_local));
        const double vimp = static_cast<double>(rows.rows[r].lambda) * j_cup.linear.z;
        const bool is_table = s.a.react == ReactionProviderKind::StaticNull ||
                              s.b.react == ReactionProviderKind::StaticNull;
        if (is_table) {
            // The table support: SUM the cup-side vertical impulse over EVERY table
            // row (comparable to cup_vertical_impulse; both are Σ λ*j_z). max_lambda /
            // max_depth stay the FINGER metrics so the contact-depth gate is unchanged.
            ++table_rows;
            max_table_lambda = std::max(max_table_lambda, rows.rows[r].lambda);
            table_vert_impulse += vimp;
            continue;
        }
        cup_vert_impulse += vimp;
        if (rows.rows[r].flags & nuka::constraint::row_flags::Friction)
            finger_vimp_friction += vimp;
        else
            finger_vimp_normal += vimp;
        max_lambda = std::max(max_lambda, rows.rows[r].lambda);
        max_depth = std::max(max_depth, rows.materials[r].position_error);
    }
    report.cup_vertical_impulse = cup_vert_impulse;
    report.finger_vimpulse_normal = finger_vimp_normal;
    report.finger_vimpulse_friction = finger_vimp_friction;
    report.lambda = max_lambda;
    report.contact_depth = max_depth;
    report.table_row_count = table_rows;
    report.table_lambda = max_table_lambda;
    report.table_vertical_impulse = table_vert_impulse;

#ifdef NUKA_H1_GRASP_DIAG
    // Per-row diagnostic: finger link, contact direction (cup-side jacobian.linear),
    // its z-component (rim contacts carry a large |z|; pure-radial wall contacts ~0),
    // and lambda. Lets the spike characterize WHY a rim-heavy 3-point grasp fails.
    {
        for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
            const ContactRowSides& s = sides[r];
            const uint32_t fl = s.a.react == ReactionProviderKind::ArticulationChainJ
                                    ? s.a.handle : s.b.handle;
            const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
            const int cup_local = a_cup ? 0 : 1;
            const RowJacobian6 jc = rows.JacobianForRowBody(r, (uint32_t)cup_local);
            std::fprintf(stderr, "[CONTACT] row=%u link=%u dir=(%.3f,%.3f,%.3f) "
                         "lambda=%.5e depth=%.5f\n", r, fl, jc.linear.x, jc.linear.y,
                         jc.linear.z, rows.rows[r].lambda, rows.materials[r].position_error);
        }
    }
#endif

    // ----- read back the cup BodyState + recoil metrics --------------------------
    box_state_ = bodies[0];
    const Vec3 cup_dv = box_state_.linear_velocity - cup_lin_before;
    report.box_dv_norm = static_cast<double>(cup_dv.Length());
    // (2) the cross-check: cup mass * (vz_after_solve - vz_before_solve) = the net
    //     vertical contact impulse. Must AGREE with cup_vertical_impulse (same thing,
    //     measured from the cup velocity change vs from the row impulses).
    const double cup_mass =
        box_state_.inv_mass > 0.0f ? 1.0 / static_cast<double>(box_state_.inv_mass) : 0.0;
    report.cup_dvz_impulse =
        cup_mass * (static_cast<double>(box_state_.linear_velocity.z) - cup_vz_pre_contact);
    double dn = 0.0;
    for (uint32_t i = 0u; i < dof_stride_; ++i)
        dn += std::fabs(static_cast<double>(qdot[i]) - qdot_before[i]);
    report.qdot_delta_l1 = dn;

    // ----- SCATTER the post-contact flat-qdot back to the device state -----------
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
        live.link_velocity[root_link_].v[i] = qdot[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) live.qdot[leg] = qdot[col];
    }
    device_.link_velocity.CopyFromHost(
        live.link_velocity.data(),
        live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
    device_.qdot.CopyFromHost(live.qdot.data(), live.qdot.size() * sizeof(float));
    context_.stream.Synchronize();
    // ===== END CONTACT PHASE ================================================

    // ----- stage 11: position integrate (articulation + cup) with the POST-contact
    //       velocity. -------------------------------------------------------------
    view = device_.View();
    articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
    IntegrateBoxPosition();

    report.cup_vz = box_state_.linear_velocity.z;
    report.cup_z = box_state_.position.z;
    return report;
}

// ===========================================================================
// S3 STANDING STEP: N foot spheres on a STATIC +Z ground (the deploy path).
// ===========================================================================
// Mirrors StepGrasp()'s stage order but: (a) the contact is foot(Sphere) x ground
// (Plane) -- an ArticulationChainJ <-> StaticNull row, the C5c-1 subsume row class,
// (b) there is NO movable rigid body (bodies empty, body_count=0, coloring key 0+0),
// (c) condim=3 so the feet have friction (don't slip). The drive torque (the balance
// law, set each step via SetGripTorque) is re-applied to tau BEFORE ABA, so the legs
// are driven. The foot reaction recoils into the FLOATING BASE through the dof_stride-
// wide foot chain-J (the same 18-wide-for-go2 base-inclusive Jacobian).
CoResidentStepReport UnifiedCoResidentStepper::StepStand() {
    CoResidentStepReport report;
    auto view = device_.View();
    const uint32_t link_count = host_proto_.TotalLinkCount();

    // ----- stage 0: the DRIVE path. Re-apply the (balance-law) drive torque to tau. --
    articulation::LaunchApplyTorqueDriveKernels(
        context_, view, static_cast<const float*>(grip_torque_dev_.Data()),
        static_cast<const float*>(grip_limits_dev_.Data()));

    // ----- stage 1/2: ABA accelerations -> qddot (now WITH the drive tau). ---------
    articulation::FeatherstoneAba::ComputeAccelerations(context_, view, gravity_z_);

    // ----- stage 3: velocity integrate (articulation halves; NO movable body). -----
    articulation::FeatherstoneAba::IntegrateVelocity(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, view, dt_,
                                                                 gravity_z_);

    // ===== CONTACT PHASE (N feet <-> STATIC ground) ==========================
    articulation::ArticulationHostState live = host_proto_;
    articulation::DownloadArticulationState(device_, &live);
    const std::vector<Transform> poses = DownloadWorldPoses(context_, device_, link_count);

    // Each foot world center (from FK) + the direct (foot, ground) pairs. DIRECT-EMIT
    // (the broadphase is proven by W1a/subsume; the gate's risk is the controller, not
    // the broadphase). Side A = foot (ArticulationChainJ), side B = ground (StaticNull).
    std::vector<Vec3> foot_centers(stand_.feet.size());
    std::vector<CandidatePair> drive_pairs;
    drive_pairs.reserve(stand_.feet.size());
    for (size_t f = 0u; f < stand_.feet.size(); ++f) {
        const CoResidentFootSphere& fs = stand_.feet[f];
        const Transform& lp = poses[fs.link];
        foot_centers[f] = lp.position + lp.rotation.Rotate(fs.local_offset);
        CandidatePair p;
        p.a.type = CollidableType::ArticulationLink;
        p.a.react = ReactionProviderKind::ArticulationChainJ;
        p.a.handle = fs.broadphase_handle;
        p.b = MakeGroundRef(stand_.ground.broadphase_id);  // StaticWorld -> StaticNull.
        drive_pairs.push_back(p);
    }

    // stage 6: narrowphase via the driver. foot(Sphere) x ground(Plane) -> the C3b
    // analytical SpherePlane handler (the +Z plane reads its normal from frame.cy).
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::ArticulationLink) {
            for (size_t f = 0u; f < stand_.feet.size(); ++f) {
                if (stand_.feet[f].broadphase_handle == ref.handle) {
                    out->type = scene::ShapeType::Sphere;
                    out->prim = FootSpherePrim(foot_centers[f], stand_.feet[f].radius);
                    out->geom = nullptr;
                    return true;
                }
            }
            return false;
        }
        if (ref.type == CollidableType::StaticWorld &&
            ref.handle == stand_.ground.broadphase_id) {
            out->type = scene::ShapeType::Plane;
            out->prim = GroundPrim(stand_.ground.height);  // +Z plane at ground height.
            out->geom = nullptr;
            return true;
        }
        return false;
    };
    std::vector<ContactManifold> manifolds;
    nuka::collision::BuildContactManifolds(drive_pairs, resolve, &manifolds);
    report.manifold_count = static_cast<uint32_t>(manifolds.size());

    uint32_t foot_points = 0u;
    for (const auto& m : manifolds) foot_points += m.point_count;
    report.finger_contacts = foot_points;     // reuse: # foot contact points this step.
    report.pair_found = foot_points > 0u;

    if (manifolds.empty() || foot_points == 0u) {
        // No contact -> the robot free-falls (a clean RED if the feet leave the ground).
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        return report;
    }

    // stage 7: compliant rows + sides (condim=3 -> normal + 4 friction spokes / point).
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = dt_;
    inputs.condim = stand_.condim;
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    report.row_count = rows.RowCount();
    for (uint32_t r = 0u; r < rows.RowCount(); ++r)
        rows.materials[r].friction = stand_.friction_mu;
    if (rows.RowCount() == 0u || sides.empty()) {
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        return report;
    }

    const std::vector<float> minv =
        InverseInertia(context_, live, gravity_z_, dof_stride_);
    std::vector<float> chain_jacobians;  // one dof_stride-wide slot per foot row.

    // ----- the flat dof_stride-wide qdot (prefix-sum packing). FloatingBase root ->
    // base spatial v (omega-first) in [0..5]; each scalar joint in its own column. ---
    std::vector<float> qdot(dof_stride_, 0.0f);
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
        qdot[i] = live.link_velocity[root_link_].v[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) qdot[col] = live.qdot[leg];
    }
    std::vector<float> qdot_before = qdot;

    // ----- wire each row: the FOOT side (ArticulationChainJ) gets a chain-J slot +
    // the SHARED coloring key body_count+art_index (== 0+0 == 0, so all feet serialize
    // on the shared base DOFs -- the C5c-1 subsume recipe); the GROUND side (StaticNull)
    // gets kInvalidBodyIndex (no reaction, no coloring conflict, no chain-J). There is
    // NO movable body (body_count=0). -------------------------------------------------
    const uint32_t kArtIndex = 0u;
    const uint32_t body_count = 0u;  // ctx.state is empty (articulation-only contact).
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    RowArticulationSide none_side{};
    float max_depth = 0.0f, max_lambda = 0.0f;
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
        const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
        art_refs[r].a = none_side;
        art_refs[r].b = none_side;
        if (!(a_art || b_art)) continue;  // (cannot occur: every row has a foot side.)
        const int foot_local = a_art ? 0 : 1;
        const int static_local = a_art ? 1 : 0;
        const RowJacobian6 j_foot =
            rows.JacobianForRowBody(r, static_cast<uint32_t>(foot_local));
        const Vec3 foot_dir = j_foot.linear;  // normal for the normal row, +/-tangent.
        const Vec3 contact_point = s.contact_point;
        // Map the foot's broadphase_handle back to its REAL device (ankle) link.
        const uint32_t foot_handle = a_art ? s.a.handle : s.b.handle;
        uint32_t foot_link = foot_handle;
        for (size_t f = 0u; f < stand_.feet.size(); ++f) {
            if (stand_.feet[f].broadphase_handle == foot_handle) {
                foot_link = stand_.feet[f].link;
                break;
            }
        }
        const std::vector<float> chain_j =
            FootChainJ(context_, live, poses, foot_link, contact_point, foot_dir,
                       dof_stride_);
        const uint32_t slot =
            static_cast<uint32_t>(chain_jacobians.size() / dof_stride_);
        chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(), chain_j.end());
        rows.body_indices[2u * r + static_cast<uint32_t>(foot_local)] =
            body_count + kArtIndex;
        rows.body_indices[2u * r + static_cast<uint32_t>(static_local)] =
            nuka::constraint::kInvalidBodyIndex;
        const RowArticulationSide foot_side{kArtIndex, slot};
        art_refs[r].a = (foot_local == 0) ? foot_side : none_side;
        art_refs[r].b = (foot_local == 1) ? foot_side : none_side;
        max_depth = std::max(max_depth, rows.materials[r].position_error);
    }

    // ----- stage 10: the unified solve (ALL foot rows together; NO rigid body). ----
    std::vector<BodyState> empty_bodies;  // articulation-only contact (the subsume case).
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &empty_bodies;
    ctx.sides = &sides;
    ctx.dt = dt_;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians =
        chain_jacobians.empty() ? nullptr : &chain_jacobians;
    ctx.articulation.inertia_m_inv = &minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = dof_stride_;
    nuka::solver::UnifiedSolve(ctx, cfg);

    // ----- read back the report metrics (max + Σ foot normal impulse + penetration). -
    double normal_impulse_sum = 0.0;
    uint32_t normal_rows = 0u;
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        if (!(rows.rows[r].flags & nuka::constraint::row_flags::Friction)) {
            max_lambda = std::max(max_lambda, rows.rows[r].lambda);
            normal_impulse_sum += static_cast<double>(rows.rows[r].lambda);
            ++normal_rows;
        }
    }
    report.lambda = max_lambda;
    report.contact_depth = max_depth;
    report.foot_normal_impulse_sum = normal_impulse_sum;
    report.foot_normal_rows = normal_rows;
    double dn = 0.0;
    for (uint32_t i = 0u; i < dof_stride_; ++i)
        dn += std::fabs(static_cast<double>(qdot[i]) - qdot_before[i]);
    report.qdot_delta_l1 = dn;

    // ----- SCATTER the post-contact flat-qdot back to the device state -----------
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i) {
        live.link_velocity[root_link_].v[i] = qdot[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) live.qdot[leg] = qdot[col];
    }
    device_.link_velocity.CopyFromHost(
        live.link_velocity.data(),
        live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
    device_.qdot.CopyFromHost(live.qdot.data(), live.qdot.size() * sizeof(float));
    context_.stream.Synchronize();
    // ===== END CONTACT PHASE ================================================

    // ----- stage 11: position integrate (articulation) with the POST-contact vel. --
    view = device_.View();
    articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
    return report;
}

// ===========================================================================
// C7b-2 SHOWCASE STEP: the CONJUNCTION -- stand AND grasp in ONE step.
// ===========================================================================
// Emits the UNION of all three proven row classes every step:
//   foot<->ground  (ArticulationChainJ <-> StaticNull),
//   finger<->cup   (ArticulationChainJ <-> RigidInvMass),
//   cup<->table    (RigidInvMass <-> StaticNull, optional).
// ONE movable cup (body_count=1) co-resident with the floating-base H1. Row class is
// read off the (reaction-kind, reaction-kind) pair -- the cup is the only RigidInvMass
// side, the ground/table are the only StaticNull sides, feet+fingers are the only
// ChainJ sides (disjoint broadphase handles tell a foot sphere from a fingertip). The
// articulation (kArtIndex=0) is shared by feet + fingers; both recoil into the floating
// base through the dof_stride-wide chain-J. Reuses StepStand()'s foot wiring +
// StepGrasp()'s finger/table wiring verbatim, merged onto body_count=1.
CoResidentStepReport UnifiedCoResidentStepper::StepStandGrasp() {
    CoResidentStepReport report;
    auto view = device_.View();
    const uint32_t link_count = host_proto_.TotalLinkCount();

    // ----- stage 0: DRIVE path (legs RL + arm reach + finger grip) -> tau. ---------
    articulation::LaunchApplyTorqueDriveKernels(
        context_, view, static_cast<const float*>(grip_torque_dev_.Data()),
        static_cast<const float*>(grip_limits_dev_.Data()));

    // ----- stage 1/2: ABA accelerations -> qddot (now WITH the drive tau). ---------
    articulation::FeatherstoneAba::ComputeAccelerations(context_, view, gravity_z_);

    // ----- stage 3: velocity integrate (articulation halves + the cup gravity). ----
    articulation::FeatherstoneAba::IntegrateVelocity(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context_, view, dt_,
                                                                 gravity_z_);
    if (box_state_.inv_mass > 0.0f) {
        box_state_.linear_velocity.z += gravity_z_ * dt_;  // cup gravity kick.
    }
    const float cup_vz_pre_contact = box_state_.linear_velocity.z;

    // ===== CONTACT PHASE (feet<->ground + fingers<->cup + optional cup<->table) ====
    articulation::ArticulationHostState live = host_proto_;
    articulation::DownloadArticulationState(device_, &live);
    const std::vector<Transform> poses = DownloadWorldPoses(context_, device_, link_count);

    // The cup convex hull view in WORLD space (mesh-local verts + live cup pose).
    const Transform cup_pose{box_state_.position, box_state_.orientation};
    nuka::collision::cvx::ConvexHullView cup_hull;
    cup_hull.verts = stand_grasp_.cup.hull_verts.data();
    cup_hull.vcount = stand_grasp_.cup.VertexCount();
    cup_hull.frame = amf::BuildPrimFrame(cup_pose);
    const bool have_cup = cup_hull.vcount > 0u;

    // FK foot + finger world centers; DIRECT-EMIT the known pairs (broadphase proven).
    std::vector<Vec3> foot_centers(stand_grasp_.feet.size());
    std::vector<Vec3> finger_centers(stand_grasp_.fingertips.size());
    std::vector<CandidatePair> drive_pairs;
    drive_pairs.reserve(stand_grasp_.feet.size() + stand_grasp_.fingertips.size() + 1u);
    for (size_t f = 0u; f < stand_grasp_.feet.size(); ++f) {
        const CoResidentFootSphere& fs = stand_grasp_.feet[f];
        const Transform& lp = poses[fs.link];
        foot_centers[f] = lp.position + lp.rotation.Rotate(fs.local_offset);
        CandidatePair p;
        p.a.type = CollidableType::ArticulationLink;
        p.a.react = ReactionProviderKind::ArticulationChainJ;
        p.a.handle = fs.broadphase_handle;
        p.b = MakeGroundRef(stand_grasp_.ground.broadphase_id);  // StaticWorld -> StaticNull.
        drive_pairs.push_back(p);
    }
    for (size_t f = 0u; f < stand_grasp_.fingertips.size(); ++f) {
        const CoResidentFingertip& ft = stand_grasp_.fingertips[f];
        const Transform& lp = poses[ft.link];
        finger_centers[f] = lp.position + lp.rotation.Rotate(ft.local_offset);
        if (!have_cup) continue;  // no cup geometry -> no finger<->cup pair.
        CandidatePair p;
        p.a.type = CollidableType::ArticulationLink;
        p.a.react = ReactionProviderKind::ArticulationChainJ;
        p.a.handle = ft.broadphase_handle;
        p.b.type = CollidableType::RigidBody;
        p.b.react = ReactionProviderKind::RigidInvMass;
        p.b.handle = stand_grasp_.cup.broadphase_body_id;
        drive_pairs.push_back(p);
    }
    const bool emit_table = stand_grasp_.has_table && table_enabled_ && have_cup;
    // The cup<->table pair uses the FLAT-BOTTOM PROXY (a real mug rests on its flat base;
    // sidesteps the hull-vs-plane coplanar-face instability). Falls back to the hull id if
    // no proxy is configured (half-extents all zero). Both ids are RigidInvMass / cup body.
    const bool use_proxy =
        stand_grasp_.cup_table_proxy_half.x > 0.0f ||
        stand_grasp_.cup_table_proxy_half.y > 0.0f ||
        stand_grasp_.cup_table_proxy_half.z > 0.0f;
    const uint32_t cup_table_id =
        use_proxy ? stand_grasp_.cup_table_proxy_id : stand_grasp_.cup.broadphase_body_id;
    if (emit_table) {
        CandidatePair tp;
        tp.a = MakeBoxRef(cup_table_id);                         // cup proxy = RigidInvMass.
        tp.b = MakeGroundRef(stand_grasp_.table_broadphase_id);  // table = StaticNull (+Z).
        drive_pairs.push_back(tp);
    }

    // stage 6: narrowphase via the driver. foot/finger(Sphere) x ground/table(Plane) or
    // cup(ConvexHull). Disjoint broadphase handles distinguish a foot sphere from a
    // fingertip; the cup/ground/table ids are distinct.
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::ArticulationLink) {
            for (size_t f = 0u; f < stand_grasp_.feet.size(); ++f) {
                if (stand_grasp_.feet[f].broadphase_handle == ref.handle) {
                    out->type = scene::ShapeType::Sphere;
                    out->prim = FootSpherePrim(foot_centers[f], stand_grasp_.feet[f].radius);
                    out->geom = nullptr;
                    return true;
                }
            }
            for (size_t f = 0u; f < stand_grasp_.fingertips.size(); ++f) {
                if (stand_grasp_.fingertips[f].broadphase_handle == ref.handle) {
                    out->type = scene::ShapeType::Sphere;
                    out->prim = FootSpherePrim(finger_centers[f],
                                               stand_grasp_.fingertips[f].radius);
                    out->geom = nullptr;
                    return true;
                }
            }
            return false;
        }
        if (ref.type == CollidableType::RigidBody &&
            ref.handle == stand_grasp_.cup_table_proxy_id) {
            // The cup's flat-bottom table-contact proxy box at the cup's LIVE pose. The
            // offset (cup body frame) puts the box bottom at the hull bottom (flush). Used
            // ONLY in the cup<->table pair (fingers pair with the hull id 7000 below).
            out->type = scene::ShapeType::Box;
            Transform proxy_pose = cup_pose;
            proxy_pose.position =
                cup_pose.position + cup_pose.rotation.Rotate(stand_grasp_.cup_table_proxy_offset);
            out->prim = BoxPrimXYZ(proxy_pose, stand_grasp_.cup_table_proxy_half);
            out->geom = nullptr;
            return true;
        }
        if (ref.type == CollidableType::RigidBody &&
            ref.handle == stand_grasp_.cup.broadphase_body_id) {
            out->type = scene::ShapeType::ConvexHull;
            out->geom = &cup_hull;  // the convex-hull seam (geom passthrough).
            return true;
        }
        if (ref.type == CollidableType::StaticWorld &&
            ref.handle == stand_grasp_.ground.broadphase_id) {
            out->type = scene::ShapeType::Plane;
            out->prim = GroundPrim(stand_grasp_.ground.height);  // +Z plane at ground z.
            out->geom = nullptr;
            return true;
        }
        if (ref.type == CollidableType::StaticWorld &&
            ref.handle == stand_grasp_.table_broadphase_id) {
            out->type = scene::ShapeType::Plane;
            out->prim = GroundPrim(stand_grasp_.table_height);   // +Z plane at table z.
            out->geom = nullptr;
            return true;
        }
        return false;
    };
    std::vector<ContactManifold> manifolds;
    nuka::collision::BuildContactManifolds(drive_pairs, resolve, &manifolds);
    report.manifold_count = static_cast<uint32_t>(manifolds.size());

    uint32_t total_points = 0u;
    for (const auto& m : manifolds) total_points += m.point_count;
    report.pair_found = total_points > 0u;

    auto integrate_and_return = [&]() -> CoResidentStepReport {
        articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
        IntegrateBoxPosition();
        report.cup_vz = box_state_.linear_velocity.z;
        report.cup_z = box_state_.position.z;
        return report;
    };
    if (manifolds.empty() || total_points == 0u) {
        return integrate_and_return();  // free-fall (clean RED if the feet leave ground).
    }

    // stage 7: compliant rows + sides (condim=3). Stamp per-CATEGORY friction.
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = dt_;
    inputs.condim = stand_grasp_.condim;
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    report.row_count = rows.RowCount();
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool is_cup = s.a.react == ReactionProviderKind::RigidInvMass ||
                            s.b.react == ReactionProviderKind::RigidInvMass;
        const bool is_static = s.a.react == ReactionProviderKind::StaticNull ||
                               s.b.react == ReactionProviderKind::StaticNull;
        if (is_cup && is_static)   rows.materials[r].friction = stand_grasp_.table_mu;   // cup<->table
        else if (is_cup)           rows.materials[r].friction = stand_grasp_.finger_mu;  // finger<->cup
        else                       rows.materials[r].friction = stand_grasp_.foot_mu;    // foot<->ground
    }
    if (rows.RowCount() == 0u || sides.empty()) {
        return integrate_and_return();
    }

    const std::vector<float> minv =
        InverseInertia(context_, live, gravity_z_, dof_stride_);
    std::vector<float> chain_jacobians;  // one dof_stride-wide slot per chain-J row.

    // ----- the flat dof_stride-wide qdot (prefix-sum packing). FloatingBase root ->
    // base spatial v (omega-first) in [0..5]; each scalar joint in its own column. ---
    std::vector<float> qdot(dof_stride_, 0.0f);
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i)
        qdot[i] = live.link_velocity[root_link_].v[i];
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) qdot[col] = live.qdot[leg];
    }
    std::vector<float> qdot_before = qdot;

    // Map a chain-J side's broadphase_handle back to its REAL device link. Searches feet
    // first, then fingertips (handles are disjoint); falls back to handle==link.
    auto handle_to_link = [&](uint32_t handle) -> uint32_t {
        for (size_t f = 0u; f < stand_grasp_.feet.size(); ++f)
            if (stand_grasp_.feet[f].broadphase_handle == handle)
                return stand_grasp_.feet[f].link;
        for (size_t f = 0u; f < stand_grasp_.fingertips.size(); ++f)
            if (stand_grasp_.fingertips[f].broadphase_handle == handle)
                return stand_grasp_.fingertips[f].link;
        return handle;
    };

    // ----- wire each row by reaction-kind class. body_count=1 (the cup); the shared
    // articulation kArtIndex=0 holds BOTH feet and fingers -> chain-J side body index
    // body_count+kArtIndex; cup side -> 0; static side -> kInvalidBodyIndex. -----------
    const uint32_t kArtIndex = 0u;
    const uint32_t body_count = 1u;  // ctx.state = [cup].
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    RowArticulationSide none_side{};
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
        const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
        const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
        const bool b_cup = s.b.react == ReactionProviderKind::RigidInvMass;
        const bool a_static = s.a.react == ReactionProviderKind::StaticNull;
        const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
        art_refs[r].a = none_side;
        art_refs[r].b = none_side;

        if ((a_cup && b_static) || (b_cup && a_static)) {
            // ----- cup<->table row: cup side -> BodyState 0, table -> kInvalid, no J. --
            report.any_static_row = true;
            const int cup_l = a_cup ? 0 : 1;
            const int static_l = a_cup ? 1 : 0;
            rows.body_indices[2u * r + static_cast<uint32_t>(cup_l)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(static_l)] =
                nuka::constraint::kInvalidBodyIndex;
            continue;
        }
        if ((a_art && b_static) || (b_art && a_static)) {
            // ----- foot<->ground row: foot -> chain-J, ground -> kInvalid. -----------
            report.any_static_row = true;
            const int foot_l = a_art ? 0 : 1;
            const int static_l = a_art ? 1 : 0;
            const RowJacobian6 j_foot =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(foot_l));
            const Vec3 foot_dir = j_foot.linear;
            const Vec3 contact_point = s.contact_point;
            const uint32_t foot_handle = a_art ? s.a.handle : s.b.handle;
            const uint32_t foot_link = handle_to_link(foot_handle);
            const std::vector<float> chain_j =
                FootChainJ(context_, live, poses, foot_link, contact_point, foot_dir,
                           dof_stride_);
            const uint32_t slot =
                static_cast<uint32_t>(chain_jacobians.size() / dof_stride_);
            chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(), chain_j.end());
            rows.body_indices[2u * r + static_cast<uint32_t>(foot_l)] =
                body_count + kArtIndex;
            rows.body_indices[2u * r + static_cast<uint32_t>(static_l)] =
                nuka::constraint::kInvalidBodyIndex;
            const RowArticulationSide foot_side{kArtIndex, slot};
            art_refs[r].a = (foot_l == 0) ? foot_side : none_side;
            art_refs[r].b = (foot_l == 1) ? foot_side : none_side;
            continue;
        }
        if ((a_art && b_cup) || (b_art && a_cup)) {
            // ----- finger<->cup row: finger -> chain-J, cup -> BodyState 0. ----------
            const int finger_l = a_art ? 0 : 1;
            const int cup_l = a_art ? 1 : 0;
            const RowJacobian6 j_finger =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(finger_l));
            const Vec3 finger_dir = j_finger.linear;
            const Vec3 contact_point = s.contact_point;
            const uint32_t finger_handle = a_art ? s.a.handle : s.b.handle;
            const uint32_t finger_link = handle_to_link(finger_handle);
            const std::vector<float> chain_j =
                FootChainJ(context_, live, poses, finger_link, contact_point, finger_dir,
                           dof_stride_);
            const uint32_t slot =
                static_cast<uint32_t>(chain_jacobians.size() / dof_stride_);
            chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(), chain_j.end());
            rows.body_indices[2u * r + static_cast<uint32_t>(cup_l)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(finger_l)] =
                body_count + kArtIndex;
            const RowArticulationSide finger_side{kArtIndex, slot};
            art_refs[r].a = (finger_l == 0) ? finger_side : none_side;
            art_refs[r].b = (finger_l == 1) ? finger_side : none_side;
            continue;
        }
        // (cannot occur: every row is one of the three classes above.)
    }

    // ----- the cup BodyState (the rigid body that reacts two-way) -----------------
    std::vector<BodyState> bodies = {box_state_};
    const Vec3 cup_lin_before = box_state_.linear_velocity;

    // ----- stage 10: the unified two-way solve (ALL rows together; bodies=[cup]). ---
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = dt_;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians =
        chain_jacobians.empty() ? nullptr : &chain_jacobians;
    ctx.articulation.inertia_m_inv = &minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = dof_stride_;
    nuka::solver::UnifiedSolve(ctx, cfg);

    // ----- read back the report metrics, CLASSIFIED per category. -----------------
    double foot_normal_impulse = 0.0;
    uint32_t foot_normal_rows = 0u;
    float max_foot_depth = 0.0f, max_foot_lambda = 0.0f;
    uint32_t finger_points = 0u;
    double cup_vert_impulse = 0.0, finger_vimp_normal = 0.0, finger_vimp_friction = 0.0;
    double table_vert_impulse = 0.0;
    uint32_t table_rows = 0u;
    float max_table_lambda = 0.0f;
    float max_finger_depth = 0.0f, max_finger_lambda = 0.0f;
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        const ContactRowSides& s = sides[r];
        const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
        const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
        const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
        const bool b_cup = s.b.react == ReactionProviderKind::RigidInvMass;
        const bool a_static = s.a.react == ReactionProviderKind::StaticNull;
        const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
        const bool is_friction =
            (rows.rows[r].flags & nuka::constraint::row_flags::Friction) != 0u;

        if ((a_cup && b_static) || (b_cup && a_static)) {
            // cup<->table support.
            const int cup_l = a_cup ? 0 : 1;
            const RowJacobian6 j_cup =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(cup_l));
            table_vert_impulse +=
                static_cast<double>(rows.rows[r].lambda) * j_cup.linear.z;
            if (!is_friction) {
                ++table_rows;
                max_table_lambda = std::max(max_table_lambda, rows.rows[r].lambda);
            }
            continue;
        }
        if ((a_art && b_static) || (b_art && a_static)) {
            // foot<->ground (the standing support).
            if (!is_friction) {
                foot_normal_impulse += static_cast<double>(rows.rows[r].lambda);
                ++foot_normal_rows;
                max_foot_depth = std::max(max_foot_depth, rows.materials[r].position_error);
                max_foot_lambda = std::max(max_foot_lambda, rows.rows[r].lambda);
            }
            continue;
        }
        // finger<->cup (the grasp hold).
        const int cup_l = a_cup ? 0 : 1;
        const RowJacobian6 j_cup =
            rows.JacobianForRowBody(r, static_cast<uint32_t>(cup_l));
        const double vimp = static_cast<double>(rows.rows[r].lambda) * j_cup.linear.z;
        cup_vert_impulse += vimp;
        if (is_friction) {
            finger_vimp_friction += vimp;
        } else {
            finger_vimp_normal += vimp;
            ++finger_points;
            max_finger_depth = std::max(max_finger_depth, rows.materials[r].position_error);
        }
        max_finger_lambda = std::max(max_finger_lambda, rows.rows[r].lambda);
    }
    // standing metrics (S3 fields + the ground diagnostics).
    report.foot_normal_impulse_sum = foot_normal_impulse;
    report.foot_normal_rows = foot_normal_rows;
    report.ground_pair_found = foot_normal_rows > 0u;
    report.ground_row_count = foot_normal_rows;
    report.ground_depth = max_foot_depth;
    report.ground_lambda = max_foot_lambda;
    // grasp metrics.
    report.finger_contacts = finger_points;
    report.cup_vertical_impulse = cup_vert_impulse;
    report.finger_vimpulse_normal = finger_vimp_normal;
    report.finger_vimpulse_friction = finger_vimp_friction;
    report.lambda = max_finger_lambda;
    report.contact_depth = max_finger_depth;
    // table metrics.
    report.table_row_count = table_rows;
    report.table_lambda = max_table_lambda;
    report.table_vertical_impulse = table_vert_impulse;

    // ----- read back the cup BodyState + recoil metrics --------------------------
    box_state_ = bodies[0];
    const Vec3 cup_dv = box_state_.linear_velocity - cup_lin_before;
    report.box_dv_norm = static_cast<double>(cup_dv.Length());
    const double cup_mass =
        box_state_.inv_mass > 0.0f ? 1.0 / static_cast<double>(box_state_.inv_mass) : 0.0;
    report.cup_dvz_impulse =
        cup_mass * (static_cast<double>(box_state_.linear_velocity.z) - cup_vz_pre_contact);
    double dn = 0.0;
    for (uint32_t i = 0u; i < dof_stride_; ++i)
        dn += std::fabs(static_cast<double>(qdot[i]) - qdot_before[i]);
    report.qdot_delta_l1 = dn;

    // ----- SCATTER the post-contact flat-qdot back to the device state -----------
    for (uint32_t i = 0u; i < base_dof_ && i < dof_stride_; ++i)
        live.link_velocity[root_link_].v[i] = qdot[i];
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = DofIndexOf(leg);
        if (col < dof_stride_) live.qdot[leg] = qdot[col];
    }
    device_.link_velocity.CopyFromHost(
        live.link_velocity.data(),
        live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
    device_.qdot.CopyFromHost(live.qdot.data(), live.qdot.size() * sizeof(float));
    context_.stream.Synchronize();
    // ===== END CONTACT PHASE ================================================

    // ----- stage 11: position integrate (articulation + cup) with POST-contact vel. -
    view = device_.View();
    articulation::FeatherstoneAba::IntegratePosition(context_, view, dt_);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context_, view, dt_);
    IntegrateBoxPosition();
    report.cup_vz = box_state_.linear_velocity.z;
    report.cup_z = box_state_.position.z;
    return report;
}

void UnifiedCoResidentStepper::IntegrateBoxPosition() {
    if (box_state_.inv_mass <= 0.0f) return;
    box_state_.position += box_state_.linear_velocity * dt_;
    const Vec3 w = box_state_.angular_velocity;
    Quat dq;
    dq.w = 1.0f;
    dq.x = 0.5f * w.x * dt_;
    dq.y = 0.5f * w.y * dt_;
    dq.z = 0.5f * w.z * dt_;
    box_state_.orientation = (box_state_.orientation * dq).Normalized();
    // NO floor clamp here. The box<->ground support is resolved THROUGH the unified
    // spine in the contact phase (box x plane narrowphase -> compliant row ->
    // UnifiedSolve), exactly like foot<->box. The integrate path is pure kinematics.
}

void UnifiedCoResidentStepper::SetGripTorque(const std::vector<float>& torque) {
    // Overwrite the DEVICE grip-torque buffer the next StepGrasp() drive path reads.
    // grip_torque_dev_ was sized to link_count at construction; size the host copy to
    // match (pad/truncate) so the copy is exactly the buffer's byte length -- a host-
    // only update would be a silent no-op (the kernel reads the device pointer).
    const uint32_t link_count = host_proto_.TotalLinkCount();
    std::vector<float> padded = torque;
    padded.resize(link_count, 0.0f);
    grip_torque_dev_.CopyFromHost(padded.data(),
                                  static_cast<size_t>(link_count) * sizeof(float));
}

void UnifiedCoResidentStepper::Download(articulation::ArticulationHostState* out) const {
    if (out == nullptr) return;
    *out = host_proto_;
    articulation::DownloadArticulationState(device_, out);
}

CoResidentEnergy UnifiedCoResidentStepper::Energy() const {
    CoResidentEnergy e;
    articulation::ArticulationHostState live = host_proto_;
    articulation::DownloadArticulationState(device_, &live);
    const uint32_t link_count = live.TotalLinkCount();
    // FK world poses (origins) + the per-link COM offset give the world COM height.
    auto& mutable_device =
        const_cast<articulation::ArticulationDeviceBuffers&>(device_);
    const std::vector<Transform> poses =
        DownloadWorldPoses(context_, mutable_device, link_count);
    const float g = -gravity_z_;  // gravity magnitude (gravity_z is signed, e.g. -9.81).

    for (uint32_t link = 0u; link < link_count; ++link) {
        // KE = 0.5 v^T I v  (link-frame spatial velocity x link-frame spatial
        // inertia about the link origin -> frame-correct total kinetic energy).
        const float* I = live.link_inertia[link].I;
        const float* v = live.link_velocity[link].v;
        double iv[6];
        for (uint32_t r = 0u; r < 6u; ++r) {
            double sum = 0.0;
            for (uint32_t c = 0u; c < 6u; ++c)
                sum += static_cast<double>(I[r * 6u + c]) * static_cast<double>(v[c]);
            iv[r] = sum;
        }
        double vtiv = 0.0;
        for (uint32_t r = 0u; r < 6u; ++r) vtiv += static_cast<double>(v[r]) * iv[r];
        e.articulation_ke += 0.5 * vtiv;

        // PE = m g z_com. m = I(3,3); z_com = link origin + R * inertial-frame COM.
        const double mass = static_cast<double>(I[3u * 6u + 3u]);
        const Vec3 com_local = live.link_inertial_frame[link].position;
        const Vec3 com_world = poses[link].position + poses[link].rotation.Rotate(com_local);
        e.articulation_pe += mass * static_cast<double>(g) * static_cast<double>(com_world.z);
    }

    // Box KE + PE.
    if (box_state_.inv_mass > 0.0f) {
        const double m = 1.0 / static_cast<double>(box_state_.inv_mass);
        const Vec3 lv = box_state_.linear_velocity;
        const Vec3 av = box_state_.angular_velocity;
        e.box_ke += 0.5 * m * (static_cast<double>(lv.x) * lv.x +
                               static_cast<double>(lv.y) * lv.y +
                               static_cast<double>(lv.z) * lv.z);
        // Diagonal body inertia I_axis = 1 / inv_inertia_axis.
        const Vec3 invI = box_state_.inv_inertia;
        if (invI.x > 0.0f)
            e.box_ke += 0.5 * (1.0 / invI.x) * static_cast<double>(av.x) * av.x;
        if (invI.y > 0.0f)
            e.box_ke += 0.5 * (1.0 / invI.y) * static_cast<double>(av.y) * av.y;
        if (invI.z > 0.0f)
            e.box_ke += 0.5 * (1.0 / invI.z) * static_cast<double>(av.z) * av.z;
        e.box_pe += m * static_cast<double>(g) * static_cast<double>(box_state_.position.z);
    }
    return e;
}

}  // namespace nuka::runtime::coresident
