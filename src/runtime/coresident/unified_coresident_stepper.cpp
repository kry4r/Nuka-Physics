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
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses, ArticulationDofCount
#include "runtime/articulation/articulation_jacobian.hpp"  // ComputeContactChainJacobians + M/M^-1
#include "runtime/articulation/featherstone_aba.hpp"
#include "solver/unified_solve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    dof_stride_ = articulation::ArticulationDofCount(host_proto_, 0u);
    root_link_ = host_proto_.articulation_link_offset[0];
}

CoResidentStepReport UnifiedCoResidentStepper::Step() {
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
    // [0..5] = base spatial velocity (omega-first) from link_velocity[root].v;
    // [6..]  = leg qdot, in link order (column 6+(link-root-1) == leg link's qdot).
    std::vector<float> qdot(dof_stride_, 0.0f);
    for (uint32_t i = 0u; i < 6u && i < dof_stride_; ++i) {
        qdot[i] = live.link_velocity[root_link_].v[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = 6u + (leg - root_link_ - 1u);
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
    // base spatial velocity slots [0..5] -> link_velocity[root]; legs [6..] -> qdot.
    for (uint32_t i = 0u; i < 6u && i < dof_stride_; ++i) {
        live.link_velocity[root_link_].v[i] = qdot[i];
    }
    for (uint32_t leg = root_link_ + 1u; leg < link_count; ++leg) {
        const uint32_t col = 6u + (leg - root_link_ - 1u);
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
