// ---------------------------------------------------------------------------
// v0.8 co-residence commit 3/3 -- THE GATE (the grasp crux)
// ---------------------------------------------------------------------------
// A go2 articulation FOOT and a MOVABLE RIGID BODY box, co-resident in ONE unified
// contact solve, with BOTH reaction arms firing (foot = ArticulationChainJ, box =
// RigidInvMass) and the box's GPU BodyState velocity actually changing. This
// de-orphans commits 1 (src/collision/link_aabb.hpp) + 2 (BuildArticulationRigid-
// CandidatePairs in src/collision/rigid_candidate_pairs.{hpp,cu}). VALIDATED-NOT-
// WIRED: no production stepper / Row struct is touched.
//
// The NEW surface is INTEGRATION, not MAGNITUDE. Force magnitude is already proven
// by composition (R-sweep guard, C4a solref/solimp vs live MuJoCo, C4c eff-mass vs
// production kernel, C5a RigidInvMass, C5c-2 MJX-parity). This gate catches
// PLUMBING/assembly failures. It is the foot-ground subsume scene (test_foot_
// ground_subsume.cpp) with the StaticWorld ground swapped for a movable RigidBody
// box -- that single swap isolates: (1) per-link AABB -> broadphase finds (foot,
// box); (2) the box is a GPU rigid body whose DOWNLOADED velocity must change;
// (3) the box-side RigidInvMass arm fires alongside the foot-side ArticulationChainJ
// arm.
//
// THE PIPELINE (the full stack -- this is what de-orphans 1+2):
//   ExtractLinkShapeAabbs (foot link AABB) + box world AABB
//     -> BuildArticulationRigidCandidatePairs  (asserts (foot,box) pair found)
//       -> contact_stream_driver::BuildContactManifolds (sphere x box, C3b)
//         -> EmitCompliantContactRows (compliant normal row + ContactRowSides)
//           -> UnifiedSolve: ctx.state=[box BodyState], ctx.articulation=go2
//
// THE LINEAR-ONLY EMITTER GAP (the real grasp blocker -- surfaced here, FIXED in
// the follow-on commit). EmitCompliantContactRows (and the production
// AppendContactGroup) build a contact normal row whose Jacobian is LINEAR ONLY --
// the box-side RowJacobian6.angular is ZERO. RigidApplyImpulse / RigidEffectiveInvMass
// touch angular only through j.angular, so the box gets NO torque even though this
// contact is deliberately OFF the box COM (a real lever, r x n != 0) -> a rigid box
// cannot rotate under contact -> a grasp cannot reorient a held object. The legacy
// MAXIMAL solver dodges this by computing r x n IN-KERNEL from body state
// (batched_device_world.cu: jacobian_angular = Cross(r, dir)); the unified row path
// has no such step, so the rigid side's angular reaction must be supplied. The
// REQUIRED angular reaction is asserted by DISABLED_BoxRotatesUnderOffCenterContact
// (a VISIBLY-PENDING test, not a silent print, not a bug-locking angular~0 assert);
// the follow-on commit computes the rigid-side r x n and ENABLES it. This gate
// commit asserts only the genuinely-proven parts (the broadphase finds the pair,
// linear two-way reaction, exact-solve of the assembled linear system, qdot recoil).
//
// THE GATE (MJX-free; force magnitude proven elsewhere):
//   PRIMARY (the sharp instrument): assemble the co-resident (A+R) lambda = b for
//     the single normal row in DOUBLE precision, where
//       A = J_foot . M^-1 . J_foot^T (articulation) + box_effective_inv_mass (rigid,
//           mirroring the kernel's diagonal-inertia r x n formula EXACTLY -- here
//           the box angular Jacobian is 0 so its term is just inv_mass*|n|^2),
//       b = aref*dt - (chainJ . qdot_seed + box_J . v_box_seed),
//     solve in double, compare to the kernel's converged lambda. rel err < 1e-4.
//   STRUCTURAL (the genuinely-new surface): the DOWNLOADED box BodyState changed:
//     (a) box linear velocity changed, nonzero, pushed AWAY from the foot along the
//         contact normal (sign-checked); (b) box angular velocity from the off-COM
//         contact -- asserted by the PENDING DISABLED_BoxRotatesUnderOffCenterContact
//         (the grasp blocker; enabled by the follow-on r x n commit); (c) the
//         articulation qdot ALSO changed (BOTH sides react). One-sided reaction = bug.
//   D1: two full runs -> byte-identical lambda + box BodyState + qdot (memcmp).
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"    // amf::PrimParams / PrimFrame
#include "collision/candidate_pair.hpp"         // CandidatePair, CollidableRef
#include "collision/contact_stream_driver.hpp"  // BuildContactManifolds, ResolvedShape
#include "collision/link_aabb.hpp"              // commit 1: ExtractLinkShapeAabbs
#include "collision/rigid_candidate_pairs.hpp"  // commit 2: BuildArticulationRigidCandidatePairs
#include "constraint/contact_manifold.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "constraint/row_builder.hpp"           // EmitCompliantContactRows
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/rigid/body_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "solver/unified_solve.hpp"

#include "foot_chain_jacobian.hpp"  // shared FK-refresh-correct ComputeFootChainJ18

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace amf = nuka::collision::amf;
using nuka::collision::AABB;
using nuka::collision::BuildContactManifolds;
using nuka::collision::CandidatePair;
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
using nuka::scene::ShapeType;

// A "return-early-with-the-captured-struct on failure" guard for the helper
// (which returns a value, so it cannot use the void-returning ASSERT_* directly).
// The TEST bodies re-assert the same conditions on the returned struct, so a
// helper-stage failure surfaces as a clear test failure WITH the captured state.
#define ASSERT_OR(cond, result_obj) \
    do { if (!(cond)) { ADD_FAILURE() << "co-residence helper precondition failed: " #cond; return (result_obj); } } while (0)

constexpr uint32_t kInvalidLink = ~0u;
constexpr uint32_t kDof = 18u;  // 6-DOF floating base + 12 revolute legs.

// The box: a unit-density (1000 kg/m^3 water) solid cube, half-extent 0.05 m
// (side 0.1 m). volume = 0.001 m^3 -> mass = 1.0 kg -> inv_mass = 1.0. A solid
// cube's diagonal inertia is I = m*s^2/6 = 1.0 * 0.01 / 6 = 1.6667e-3 kg*m^2 about
// each axis -> inv_inertia = 600.0 per axis. PHYSICAL values, documented (NOT hand-
// faked). (The inv_inertia does not actually enter the solve here: the linear-only
// contact normal row has a ZERO angular Jacobian -- see the file header finding --
// but we set it physically so the rigid 6-DOF state is well-formed and the report
// is honest about the off-COM lever.)
constexpr float kBoxHalfExtent = 0.05f;
constexpr float kBoxMass = 1.0f;
constexpr float kBoxInvMass = 1.0f;
constexpr float kBoxInvInertia = 600.0f;  // 1 / (m*s^2/6) = 1/1.6667e-3.
// A rigid body id for the box in broadphase BODY-id space, chosen clearly distinct
// from every go2 link body id (the cooked go2 has < 32 bodies), so the same-body
// drop + exclude filter never confuses it with a foot link's body. The kernel's
// rigid arm indexes the BodyState array by the ROW body index (we set 0), NOT by
// this broadphase body id, so the two are independent on purpose.
constexpr uint32_t kBoxRigidBodyId = 9000u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

struct CookedFloat {
    articulation::ArticulationHostState host;
    nuka::scene::CookedShapeTable shapes;   // for ExtractLinkShapeAabbs (commit 1).
    std::vector<articulation::FootShape> feet;
    std::vector<float> link_mass;
};

CookedFloat CookGo2Float() {
    const auto scene = nuka::import::LoadUsd(SourcePath("examples/scenes/go2_float.usda").string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    CookedFloat result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    result.shapes = world.template_view.shape_table;
    const uint32_t link_count = result.host.TotalLinkCount();
    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link) {
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    }
    const auto& shapes = world.template_view.shape_table;
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) continue;
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) { calf_link = link; break; }
        }
        if (calf_link == kInvalidLink) continue;
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }
    return result;
}

std::vector<Transform> ForwardKinematics(const nuka::phi::DeviceContext& context,
                                         const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

Vec3 FootWorldCenter(const std::vector<Transform>& poses, const articulation::FootShape& foot) {
    const Transform calf = poses[foot.calf_local_link];
    return calf.position + calf.rotation.Rotate(foot.local_offset);
}

// 18x18 M^-1 via the production CRBA (reuse, not hand-rolled). Requires ABA pass-1
// (link_xup current), so we run ComputeAccelerations first.
std::vector<float> ComputeInverseInertia18(const nuka::phi::DeviceContext& context,
                                           const articulation::ArticulationHostState& host) {
    const float kGravityZ = -9.81f;
    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    const uint32_t link_count = host.TotalLinkCount();
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, kGravityZ);
    nuka::phi::Buffer composite(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia),
        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m(static_cast<size_t>(kDof) * kDof * sizeof(float),
                        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m_inv(static_cast<size_t>(kDof) * kDof * sizeof(float),
                            nuka::phi::MemoryKind::Device);
    articulation::ComputeArticulationInertiaM(
        context, view, kDof,
        static_cast<articulation::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m.Data()));
    articulation::FactorArticulationInertiaM(
        context, view, kDof, static_cast<const float*>(m.Data()),
        static_cast<float*>(m_inv.Data()));
    context.stream.Synchronize();
    std::vector<float> out(static_cast<size_t>(kDof) * kDof);
    m_inv.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

// A sphere prim at the foot FK center; an axis-aligned box prim at its world pose.
amf::PrimParams MakeFootPrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;  // identity rotation; sphere is rotation-invariant.
    return p;
}
amf::PrimParams MakeBoxPrim(const Vec3& center, float half_extent) {
    amf::PrimParams p;
    p.half_extents = Vec3{half_extent, half_extent, half_extent};
    p.frame.t = center;  // identity frame -> axis-aligned box.
    return p;
}

// The box's world AABB (axis-aligned cube centered at `center`).
AABB BoxWorldAabb(const Vec3& center, float half_extent) {
    AABB box;
    box.min = center - Vec3{half_extent, half_extent, half_extent};
    box.max = center + Vec3{half_extent, half_extent, half_extent};
    return box;
}

// ---------------------------------------------------------------------------
// The whole foot<->box UNIFIED-spine assembly result.
// ---------------------------------------------------------------------------
struct CoResidenceResult {
    bool pair_found = false;        // broadphase emitted the (foot, box) cross pair.
    uint32_t row_count = 0u;
    float contact_depth = 0.0f;
    float lambda = 0.0f;            // kernel's converged normal impulse.
    double exact_lambda = 0.0;      // DOUBLE-precision (A+R) exact solve.
    double exact_rel_err = 0.0;     // |kernel - exact| / max(1, |exact|).
    Vec3 contact_normal{};          // manifold normal (separation dir for side A).
    Vec3 box_normal{};              // the box-side row normal (n into the box).
    Vec3 contact_point{};
    // Box BodyState before/after the solve.
    Vec3 box_lin_before{}, box_lin_after{};
    Vec3 box_ang_before{}, box_ang_after{};
    // Articulation qdot before/after (the foot recoil).
    std::vector<float> qdot_before, qdot_after;
    double qdot_delta_norm = 0.0;
    // Which local slot (0=A, 1=B) carried the box / foot (from the emitted sides).
    int box_local = -1, foot_local = -1;
    // Raw row jacobians (per local slot) for the report + the exact-solve oracle.
    RowJacobian6 j_foot{}, j_box{};
    // Final BodyState (for D1 memcmp).
    BodyState box_state{};
};

// Build the box just below `foot_center` so the foot sphere penetrates the box top
// (+Z) face, OFFSET in x by `x_offset` from the box center -> an OFF-COM contact
// (r x n != 0). Returns the box world center.
Vec3 PlaceBoxUnderFoot(const Vec3& foot_center, float foot_radius, float x_offset,
                       float penetration) {
    // Foot sphere bottom = foot_center.z - foot_radius. We want the box top face
    // (box_center.z + half) ABOVE the foot bottom by `penetration`, i.e.
    //   box_center.z + half = foot_bottom + penetration
    //   box_center.z = foot_center.z - foot_radius + penetration - half.
    Vec3 c;
    c.x = foot_center.x - x_offset;  // shift the box so the foot is off-center in x.
    c.y = foot_center.y;
    c.z = foot_center.z - foot_radius + penetration - kBoxHalfExtent;
    return c;
}

CoResidenceResult RunCoResidence(const nuka::phi::DeviceContext& context,
                                 const CookedFloat& cooked,
                                 const nuka::solver::SolverConfig& config) {
    CoResidenceResult result;
    const auto& host = cooked.host;

    // --- FK -> foot world centers (the material contact points) --------------
    const auto poses = ForwardKinematics(context, host);

    // Pick foot 0 (FL) as the single contacting foot.
    const articulation::FootShape& foot = cooked.feet[0];
    const Vec3 foot_center = FootWorldCenter(poses, foot);
    constexpr float kXOffset = 0.02f;      // 2 cm lever in x -> off-COM contact.
    constexpr float kPenetration = 0.004f; // 4 mm penetration into the box top.
    const Vec3 box_center =
        PlaceBoxUnderFoot(foot_center, foot.radius, kXOffset, kPenetration);
    result.contact_point = box_center;  // overwritten by the real manifold point.

    // --- (1) commit 1: per-link world AABBs from the articulation FK ---------
    const LinkShapeAabbs links =
        nuka::collision::ExtractLinkShapeAabbs(cooked.shapes, poses, host.link_body);
    ASSERT_OR(!links.aabbs.empty(), result);
    // All-permissive contact masks for the links (MuJoCo default everything collides).
    std::vector<uint32_t> link_contypes(links.aabbs.size(), 1u);
    std::vector<uint32_t> link_conaff(links.aabbs.size(), 1u);

    // --- (1) the box world AABB (rigid side) ---------------------------------
    std::vector<AABB> rigid_aabbs = {BoxWorldAabb(box_center, kBoxHalfExtent)};
    std::vector<uint32_t> rigid_body_ids = {kBoxRigidBodyId};
    std::vector<uint32_t> rigid_contypes = {1u};
    std::vector<uint32_t> rigid_conaff = {1u};
    nuka::phi::Buffer d_rigid = nuka::phi::UploadVector(rigid_aabbs);
    const AABB* d_rigid_ptr = static_cast<const AABB*>(d_rigid.Data());

    // --- (2) commit 2: the cross-type CandidatePair stream -------------------
    auto stream = nuka::collision::BuildArticulationRigidCandidatePairs(
        context, d_rigid_ptr, static_cast<uint32_t>(rigid_aabbs.size()),
        rigid_body_ids.data(), rigid_contypes.data(), rigid_conaff.data(),
        links, link_contypes.data(), link_conaff.data(),
        /*excluded_body_pairs=*/{});
    const auto pairs = stream.DownloadPairs();

    // --- assert the (foot, box) cross-type pair is present (broadphase found it,
    // we did NOT hand-build it). The foot side is ArticulationLink handle=link idx;
    // the box side is RigidBody handle=box body id. ------------------------------
    CandidatePair foot_box_pair;
    bool found = false;
    for (const auto& p : pairs) {
        const bool a_link = p.a.type == CollidableType::ArticulationLink &&
                            p.a.handle == foot.calf_local_link;
        const bool b_box = p.b.type == CollidableType::RigidBody &&
                           p.b.handle == kBoxRigidBodyId;
        const bool a_box = p.a.type == CollidableType::RigidBody &&
                           p.a.handle == kBoxRigidBodyId;
        const bool b_link = p.b.type == CollidableType::ArticulationLink &&
                            p.b.handle == foot.calf_local_link;
        if ((a_link && b_box) || (a_box && b_link)) {
            foot_box_pair = p;
            found = true;
            break;
        }
    }
    result.pair_found = found;
    ASSERT_OR(found, result);

    // --- (3) contact_stream_driver: the ONLY pair we drive is (foot, box) so the
    // gate produces exactly one manifold. The resolver maps each side by TYPE to
    // its ResolvedShape (the decoupling seam -- the driver never reads cooked
    // tables). ------------------------------------------------------------------
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::ArticulationLink &&
            ref.handle == foot.calf_local_link) {
            out->type = ShapeType::Sphere;
            out->prim = MakeFootPrim(foot_center, foot.radius);
            return true;
        }
        if (ref.type == CollidableType::RigidBody && ref.handle == kBoxRigidBodyId) {
            out->type = ShapeType::Box;
            out->prim = MakeBoxPrim(box_center, kBoxHalfExtent);
            return true;
        }
        return false;
    };
    std::vector<CandidatePair> drive_pairs = {foot_box_pair};
    std::vector<ContactManifold> manifolds;
    BuildContactManifolds(drive_pairs, resolve, &manifolds);
    ASSERT_OR(manifolds.size() == 1u, result);
    ASSERT_OR(manifolds[0].point_count > 0u, result);
    result.contact_depth = manifolds[0].points[0].penetration;
    result.contact_normal = manifolds[0].points[0].normal;
    result.contact_point = manifolds[0].points[0].position;

    // --- (4) compliant rows + sides (condim=1, clean normal row) -------------
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = 1.0f / 240.0f;
    inputs.condim = 1u;  // frictionless: one clean normal row.
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    result.row_count = rows.RowCount();
    ASSERT_OR(rows.RowCount() == 1u, result);
    ASSERT_OR(sides.size() == 1u, result);

    // --- identify which local slot (0=A, 1=B) is the foot (ArticulationChainJ)
    // and which is the box (RigidInvMass), from the EMITTED sides (we do NOT
    // hand-stamp them -- they flow from the broadphase through the manifold). ----
    const ContactRowSides& s = sides[0];
    if (s.a.react == ReactionProviderKind::ArticulationChainJ &&
        s.b.react == ReactionProviderKind::RigidInvMass) {
        result.foot_local = 0; result.box_local = 1;
    } else if (s.b.react == ReactionProviderKind::ArticulationChainJ &&
               s.a.react == ReactionProviderKind::RigidInvMass) {
        result.foot_local = 1; result.box_local = 0;
    }
    ASSERT_OR(result.foot_local >= 0 && result.box_local >= 0, result);

    // --- the row's per-side jacobians (for the report + the exact-solve oracle) -
    result.j_foot = rows.JacobianForRowBody(0u, static_cast<uint32_t>(result.foot_local));
    result.j_box = rows.JacobianForRowBody(0u, static_cast<uint32_t>(result.box_local));
    // The box-side row normal n_box = the box jacobian's linear part (the direction
    // the box is pushed for lambda>0).
    result.box_normal = result.j_box.linear;

    // --- the 18-wide foot chain-J on the manifold's normal-into-the-foot --------
    // The foot side's RowJacobian6.linear IS the foot's contact normal (separation
    // dir for the foot side); we project the chain-J on it (the reduced-coordinate
    // reaction the kernel uses). --------------------------------------------------
    const Vec3 foot_normal = result.j_foot.linear;
    const std::vector<float> minv = ComputeInverseInertia18(context, host);
    const std::vector<float> chain_j = nuka::test::ComputeFootChainJ18(
        context, host, poses, foot.calf_local_link, result.contact_point, foot_normal, kDof);
    std::vector<float> chain_jacobians = chain_j;  // one slot.

    // --- wire art_refs (foot side) + body_indices (box side -> BodyState 0;
    // foot side -> coloring key body_count+art_index = 1 so the rigid arm stays
    // inert on the foot, ValidBody(1,1)==false, while art_refs drives the chain
    // arm). body_count = 1 (the box). ------------------------------------------
    const uint32_t kArtIndex = 0u;
    const uint32_t body_count = 1u;  // ctx.state = [box].
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    // Box side -> BodyState index 0; foot side -> the coloring key (>= body_count
    // so ValidBody is false for the rigid arm, but it is the per-articulation key).
    rows.body_indices[2u * 0u + static_cast<uint32_t>(result.box_local)] = 0u;
    rows.body_indices[2u * 0u + static_cast<uint32_t>(result.foot_local)] =
        body_count + kArtIndex;
    RowArticulationSide foot_side{kArtIndex, /*chain_jac_slot=*/0u};
    RowArticulationSide none_side{};  // the box is not an articulation.
    art_refs[0].a = (result.foot_local == 0) ? foot_side : none_side;
    art_refs[0].b = (result.foot_local == 1) ? foot_side : none_side;

    // --- the box BodyState (real rigid body) + the articulation qdot seed -------
    // Seed: a downward base velocity (the base "falls") AND the box at rest. The
    // foot is pressed into the box from above; the converged solve drives the
    // contact-normal velocity to its compliant target, pushing the box DOWN (-Z,
    // away from the foot) and recoiling the foot/base UP.
    BodyState box;
    box.inv_mass = kBoxInvMass;
    box.inv_inertia = Vec3{kBoxInvInertia, kBoxInvInertia, kBoxInvInertia};
    box.position = box_center;
    box.orientation = Quat::Identity();
    box.linear_velocity = Vec3::Zero();
    box.angular_velocity = Vec3::Zero();
    std::vector<BodyState> bodies = {box};
    result.box_lin_before = box.linear_velocity;
    result.box_ang_before = box.angular_velocity;

    std::vector<float> qdot(kDof, 0.0f);
    qdot[5] = -1.0f;  // downward base vertical velocity (body z == world z @ identity).
    result.qdot_before = qdot;

    // --- PRIMARY exact-solve oracle: assemble (A+R) lambda = b in DOUBLE before
    // the kernel runs (it uses the SAME seed). A = J_foot M^-1 J_foot^T + box
    // eff-inv-mass (mirroring the kernel's RigidEffectiveInvMass: inv_mass*|Jlin|^2
    // + sum Jang_k^2 invI_k -- the box angular Jacobian is 0, so this is just
    // inv_mass*|n_box|^2). b = aref*dt - Jv_seed where Jv_seed is the SAME per-side
    // dispatch the kernel uses (chainJ.qdot for the foot, n_box.v_box for the box). -
    {
        // A_art = J_foot M^-1 J_foot^T (double).
        double a_art = 0.0;
        for (uint32_t r = 0u; r < kDof; ++r) {
            double row = 0.0;
            for (uint32_t c = 0u; c < kDof; ++c)
                row += static_cast<double>(minv[r * kDof + c]) *
                       static_cast<double>(chain_j[c]);
            a_art += static_cast<double>(chain_j[r]) * row;
        }
        // A_box = box RigidEffectiveInvMass on j_box (diagonal-inertia mirror).
        const RowJacobian6& jb = result.j_box;
        double a_box = static_cast<double>(box.inv_mass) *
                       (static_cast<double>(jb.linear.x) * jb.linear.x +
                        static_cast<double>(jb.linear.y) * jb.linear.y +
                        static_cast<double>(jb.linear.z) * jb.linear.z);
        a_box += static_cast<double>(jb.angular.x) * jb.angular.x * box.inv_inertia.x;
        a_box += static_cast<double>(jb.angular.y) * jb.angular.y * box.inv_inertia.y;
        a_box += static_cast<double>(jb.angular.z) * jb.angular.z * box.inv_inertia.z;
        const double A = a_art + a_box;
        const double R = static_cast<double>(rows.rows[0].compliance_alpha);
        // Jv_seed: foot chainJ . qdot_seed + box (Jlin.v + Jang.w) on the seed.
        double jv = 0.0;
        for (uint32_t r = 0u; r < kDof; ++r)
            jv += static_cast<double>(chain_j[r]) * static_cast<double>(qdot[r]);
        jv += static_cast<double>(jb.linear.x) * box.linear_velocity.x +
              static_cast<double>(jb.linear.y) * box.linear_velocity.y +
              static_cast<double>(jb.linear.z) * box.linear_velocity.z;
        jv += static_cast<double>(jb.angular.x) * box.angular_velocity.x +
              static_cast<double>(jb.angular.y) * box.angular_velocity.y +
              static_cast<double>(jb.angular.z) * box.angular_velocity.z;
        const double b = static_cast<double>(rows.rows[0].rhs) *
                             static_cast<double>(inputs.dt) - jv;
        // 1x1 (A+R) lambda = b, lambda >= 0.
        const double lam = (A + R) > 1.0e-12 ? std::max(0.0, b / (A + R)) : 0.0;
        result.exact_lambda = lam;
    }

    // --- (5) UnifiedSolve: ctx.state=[box], ctx.articulation=go2 --------------
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = inputs.dt;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians = &chain_jacobians;
    ctx.articulation.inertia_m_inv = &minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = kDof;
    nuka::solver::UnifiedSolve(ctx, config);

    // --- read back ------------------------------------------------------------
    result.lambda = rows.rows[0].lambda;
    result.box_state = bodies[0];
    result.box_lin_after = bodies[0].linear_velocity;
    result.box_ang_after = bodies[0].angular_velocity;
    result.qdot_after = qdot;
    double dn = 0.0;
    for (uint32_t i = 0u; i < kDof; ++i)
        dn += std::fabs(static_cast<double>(qdot[i]) - result.qdot_before[i]);
    result.qdot_delta_norm = dn;
    const double den = std::max(1.0, std::fabs(result.exact_lambda));
    result.exact_rel_err = std::fabs(static_cast<double>(result.lambda) -
                                     result.exact_lambda) / den;
    return result;
}

// Converged compliant PGS config (a single coupled row -> converges fast; we use a
// generous iteration count so the kernel lands on its assembled (A+R) fixed point).
nuka::solver::SolverConfig ConvergedConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}

}  // namespace

// ===========================================================================
// THE GATE (PRIMARY + STRUCTURAL): foot<->box co-residence through the unified
// spine, both reaction arms firing, the box BodyState velocity changing.
// ===========================================================================
TEST(FootBoxCoResidence, BothArmsFireAndBoxVelocityChanges) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : cooked.host.qdot) qd = 0.0f;
    ASSERT_FALSE(cooked.feet.empty());

    const auto out = RunCoResidence(context, cooked, ConvergedConfig());

    // --- broadphase + driver plumbing (de-orphans commits 1 + 2) -------------
    EXPECT_TRUE(out.pair_found)
        << "BuildArticulationRigidCandidatePairs did not emit the (foot, box) pair";
    EXPECT_EQ(out.row_count, 1u) << "expected exactly one compliant normal row";
    EXPECT_GT(out.contact_depth, 0.0f) << "the foot did not penetrate the box";
    EXPECT_GE(out.box_local, 0); EXPECT_GE(out.foot_local, 0);

    std::printf("[diag] contact normal=(%+.4f,%+.4f,%+.4f) box_normal=(%+.4f,%+.4f,%+.4f) "
                "depth=%.5f\n",
                out.contact_normal.x, out.contact_normal.y, out.contact_normal.z,
                out.box_normal.x, out.box_normal.y, out.box_normal.z, out.contact_depth);
    std::printf("[diag] foot side local=%d (ArticulationChainJ)  box side local=%d "
                "(RigidInvMass)\n", out.foot_local, out.box_local);

    // --- PRIMARY: the kernel lambda matches the DOUBLE exact (A+R) solve --------
    std::printf("[diag] PRIMARY: kernel lambda=%.8f  exact (A+R) solve=%.8f  rel err=%.3e\n",
                out.lambda, out.exact_lambda, out.exact_rel_err);
    EXPECT_GT(out.exact_lambda, 0.0)
        << "the assembled (A+R) system has no positive impulse -- contact not closing?";
    EXPECT_LT(out.exact_rel_err, 1.0e-4)
        << "kernel lambda diverged from the exact (A+R) solve of the co-resident "
           "system (a PLUMBING/assembly bug, NOT a magnitude issue)";

    // --- STRUCTURAL (a): box LINEAR velocity changed, nonzero, pushed AWAY from
    // the foot along the contact normal. The box-side row normal (box_normal) is
    // the direction the box is pushed for lambda>0; the foot is above the box so
    // box_normal points DOWN (-Z, away from the foot). The applied dv is
    // box_normal * inv_mass * lambda, so dv . box_normal > 0. ---------------------
    const Vec3 dlin = out.box_lin_after - out.box_lin_before;
    const float dlin_along_normal = dlin.Dot(out.box_normal);
    std::printf("[diag] STRUCTURAL(a) box lin vel: before=(%+.5f,%+.5f,%+.5f) "
                "after=(%+.5f,%+.5f,%+.5f)  delta.n=%.6f\n",
                out.box_lin_before.x, out.box_lin_before.y, out.box_lin_before.z,
                out.box_lin_after.x, out.box_lin_after.y, out.box_lin_after.z,
                dlin_along_normal);
    EXPECT_GT(dlin.Length(), 1.0e-5f) << "box linear velocity did not change";
    EXPECT_GT(dlin_along_normal, 1.0e-5f)
        << "box was not pushed AWAY from the foot along the contact normal";
    // The foot is ABOVE the box -> the box is pushed DOWN (-Z): the box-side normal
    // z-component is negative AND the box vertical velocity decreased.
    EXPECT_LT(out.box_normal.z, 0.0f) << "box-side contact normal is not pointing down";
    EXPECT_LT(out.box_lin_after.z, out.box_lin_before.z)
        << "box was not pushed downward (away from the foot above it)";

    // --- STRUCTURAL (b): the box ANGULAR reaction from the off-COM contact is the
    // grasp blocker. Its assertion lives in DISABLED_BoxRotatesUnderOffCenterContact
    // (the rigid side's r x n angular Jacobian is still dropped; the follow-on commit
    // computes it and enables that test). Here we only (1) report the measured
    // angular delta and (2) assert the geometry is genuinely off-COM, so the pending
    // test is not vacuous. We do NOT assert angular ~ 0 -- that would lock in the
    // buggy zero-torque physics (a weakened acceptance criterion). ----------------
    const Vec3 dang = out.box_ang_after - out.box_ang_before;
    std::printf("[diag] STRUCTURAL(b) box ANG vel delta=%.3e  j_box.angular=(%.3e,%.3e,%.3e) "
                "[angular reaction PENDING -- see DISABLED_BoxRotatesUnderOffCenterContact]\n",
                dang.Length(), out.j_box.angular.x, out.j_box.angular.y, out.j_box.angular.z);
    const float lever_x = out.contact_point.x - out.box_state.position.x;
    std::printf("[diag] STRUCTURAL(b) off-COM lever (contact_x - box_com_x) = %+.5f m\n",
                lever_x);
    EXPECT_GT(std::fabs(lever_x), 1.0e-3f)
        << "the contact is NOT off-COM -- the pending angular test would be vacuous";

    // --- STRUCTURAL (c): the articulation qdot ALSO changed (BOTH sides react) --
    std::printf("[diag] STRUCTURAL(c) qdot delta L1 norm=%.6f  base vz: %+.5f -> %+.5f\n",
                out.qdot_delta_norm, out.qdot_before[5], out.qdot_after[5]);
    EXPECT_GT(out.qdot_delta_norm, 1.0e-4)
        << "the articulation qdot did not change -- the foot/base did not recoil "
           "(one-sided reaction = a bug)";
    // The base vertical velocity recoils UPWARD (less negative): the contact pushes
    // the foot/base up while it pushes the box down -> a genuine TWO-WAY reaction.
    EXPECT_GT(out.qdot_after[5], out.qdot_before[5])
        << "the base vertical velocity was not corrected upward by the contact";
}

// ===========================================================================
// PENDING (the grasp blocker): an off-COM contact MUST give the rigid box angular
// velocity. The current EmitCompliantContactRows emits a LINEAR-ONLY normal row
// (box-side RowJacobian6.angular == 0) -> no torque -> the box cannot rotate, so a
// grasp cannot reorient a held object. The follow-on commit computes the rigid
// side's r x n angular Jacobian (mirroring the legacy in-kernel
// batched_device_world.cu jacobian_angular = Cross(r, dir)) and ENABLES this test
// by dropping the DISABLED_ prefix. DISABLED (not deleted, not a silent print, not
// a bug-locking angular~0 assert) so the gap is VISIBLE in the test runner.
// ===========================================================================
TEST(FootBoxCoResidence, DISABLED_BoxRotatesUnderOffCenterContact) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : cooked.host.qdot) qd = 0.0f;
    ASSERT_FALSE(cooked.feet.empty());

    const auto out = RunCoResidence(context, cooked, ConvergedConfig());

    // Precondition: the contact is genuinely off the box COM (a real lever), so a
    // correct rigid contact MUST give the box angular velocity.
    const float lever_x = out.contact_point.x - out.box_state.position.x;
    ASSERT_GT(std::fabs(lever_x), 1.0e-3f) << "contact must be off-COM for this test";

    const Vec3 dang = out.box_ang_after - out.box_ang_before;
    EXPECT_GT(dang.Length(), 1.0e-4f)
        << "off-COM contact gave the box NO angular velocity -- the rigid-side "
           "angular Jacobian (r x n) is still dropped (the grasp blocker)";
}

// ===========================================================================
// D1: two full runs -> byte-identical lambda + box BodyState + qdot (memcmp).
// ===========================================================================
TEST(FootBoxCoResidence, Deterministic_TwoRun) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : cooked.host.qdot) qd = 0.0f;
    ASSERT_FALSE(cooked.feet.empty());

    const auto a = RunCoResidence(context, cooked, ConvergedConfig());
    const auto b = RunCoResidence(context, cooked, ConvergedConfig());

    ASSERT_TRUE(a.pair_found && b.pair_found);
    ASSERT_EQ(a.qdot_after.size(), b.qdot_after.size());
    EXPECT_EQ(std::memcmp(&a.lambda, &b.lambda, sizeof(float)), 0)
        << "two runs produced different lambda (nondeterministic)";
    EXPECT_EQ(std::memcmp(&a.box_state, &b.box_state, sizeof(BodyState)), 0)
        << "two runs produced a different box BodyState (nondeterministic)";
    EXPECT_EQ(std::memcmp(a.qdot_after.data(), b.qdot_after.data(),
                          a.qdot_after.size() * sizeof(float)), 0)
        << "two runs produced different qdot (nondeterministic)";
    std::printf("[diag] (D1) lambda + box BodyState + qdot byte-identical across 2 runs\n");
}
