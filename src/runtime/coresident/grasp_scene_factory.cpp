// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- grasp scene factory (v0.8 A2). The bodies are
// PROMOTED VERBATIM from the test helpers (tests/coresident/test_batched_unified_world.cpp
// LoadGraspCupHull / BuildGraspGripper / BuildGraspSceneBundle / MakeGraspTemplate)
// so the test can delegate to THIS code and keep its 21 gates byte-identical, and
// so the C-ABI grasp world builds the SAME validated BatchedSceneTemplate. The ONLY
// change is parametrizing the cup asset path (an argument, not the kCupModelUsda
// literal). NO physics / numeric change.
// ---------------------------------------------------------------------------

#include "runtime/coresident/grasp_scene_factory.hpp"

#include "import/usd_importer.hpp"  // LoadUsd
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "scene/canonical_types.hpp"  // DecomposeMode
#include "scene/cooker.hpp"           // CookScene

#include <algorithm>

namespace nuka::runtime::coresident {

namespace {
namespace articulation = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;
}  // namespace

GraspCupHull LoadGraspCupHull(const std::string& asset_path) {
    auto scene = nuka::import::LoadUsd(asset_path);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;
    }
    const auto blob = nuka::scene::CookScene(scene);
    GraspCupHull out;
    const auto& cg = blob.convex_geometry;
    if (cg.vertex_counts.empty()) return out;
    const uint32_t voff = cg.vertex_offsets[0];
    const uint32_t vcnt = cg.vertex_counts[0];
    out.vcount = vcnt;
    out.verts.assign(cg.vertices.begin() + voff * 3u,
                     cg.vertices.begin() + (voff + vcnt) * 3u);
    out.lo = Vec3{out.verts[0], out.verts[1], out.verts[2]};
    out.hi = out.lo;
    for (uint32_t i = 0u; i < vcnt; ++i) {
        const Vec3 v{out.verts[i * 3u], out.verts[i * 3u + 1u], out.verts[i * 3u + 2u]};
        out.lo = Vec3{std::min(out.lo.x, v.x), std::min(out.lo.y, v.y),
                      std::min(out.lo.z, v.z)};
        out.hi = Vec3{std::max(out.hi.x, v.x), std::max(out.hi.y, v.y),
                      std::max(out.hi.z, v.z)};
    }
    return out;
}

GraspGripper BuildGraspGripper(const Vec3& base_pos, float cup_half_x,
                               float fingertip_radius) {
    constexpr uint32_t kInvalidLink = ~0u;
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u};
    topo.parent_links = {kInvalidLink, 0u, 0u};
    topo.joint_types = {articulation::ArticulationJointType::Fixed,
                        articulation::ArticulationJointType::Prismatic,
                        articulation::ArticulationJointType::Prismatic};
    topo.joint_axes = {Vec3::UnitX(), Vec3{-1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}};
    topo.parent_frames = {Transform::Identity(), Transform::Identity(),
                          Transform::Identity()};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity()};
    Transform base_lp = Transform::Identity();
    base_lp.position = base_pos;
    topo.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    topo.inertial_frames = {Transform::Identity(), Transform::Identity(),
                            Transform::Identity()};
    topo.initial_positions = {0.0f, 0.0f, 0.0f};
    topo.joint_dampings = {0.0f, 0.0f, 0.0f};
    topo.joint_armatures = {0.0f, 0.0f, 0.0f};
    topo.masses = {1.0f, 0.05f, 0.05f};
    topo.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                     Vec3{1e-4f, 1e-4f, 1e-4f}};

    nuka::scene::CookedBodyTable bodies;
    Transform base_pose = Transform::Identity();
    base_pose.position = base_pos;
    bodies.poses = {base_pose, Transform::Identity(), Transform::Identity()};
    bodies.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    bodies.inertial_frames = {Transform::Identity(), Transform::Identity(),
                              Transform::Identity()};
    bodies.masses = {1.0f, 0.05f, 0.05f};
    bodies.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                       Vec3{1e-4f, 1e-4f, 1e-4f}};
    bodies.is_static = {0u, 0u, 0u};

    GraspGripper g;
    g.host = articulation::BuildArticulationHostState({topo}, bodies);
    g.finger_link[0] = 1u;
    g.finger_link[1] = 2u;
    g.fingertip_radius = fingertip_radius;
    const float kPenetration = 0.0015f;  // 1.5 mm shallow pre-pose -> contact live @ step 0.
    const float reach = cup_half_x + fingertip_radius - kPenetration;
    g.fingertip_local[0] = Vec3{reach, 0.0f, 0.0f};
    g.fingertip_local[1] = Vec3{-reach, 0.0f, 0.0f};
    return g;
}

GraspSceneBundle BuildGraspSceneBundle(const GraspCupHull& hull, float grip_force,
                                       float mu, float cup_start_z_offset) {
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    const Vec3 cup_local_center = (hull.hi + hull.lo) * 0.5f;
    // The FIXED fingertip catch plane (z=0.20) -- the fingertip pre-pose math below
    // keeps using THIS Z so the fingertips never move with the cup-start knob.
    const Vec3 cup_center{0.0f, 0.0f, 0.20f};
    const Vec3 base_pos{0.0f, 0.0f, 0.30f};

    GraspSceneBundle gs;
    gs.gripper = BuildGraspGripper(base_pos, half.x, 0.008f);
    const float drop = cup_center.z - base_pos.z;
    gs.gripper.fingertip_local[0].z += drop;
    gs.gripper.fingertip_local[1].z += drop;

    CoResidentFingertip ft0;
    ft0.link = gs.gripper.finger_link[0];
    ft0.broadphase_handle = gs.gripper.finger_link[0];
    ft0.local_offset = gs.gripper.fingertip_local[0];
    ft0.radius = gs.gripper.fingertip_radius;
    CoResidentFingertip ft1;
    ft1.link = gs.gripper.finger_link[1];
    ft1.broadphase_handle = gs.gripper.finger_link[1];
    ft1.local_offset = gs.gripper.fingertip_local[1];
    ft1.radius = gs.gripper.fingertip_radius;
    gs.config.fingertips = {ft0, ft1};

    gs.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        gs.config.cup.hull_verts[i * 3u + 0u] -= cup_local_center.x;
        gs.config.cup.hull_verts[i * 3u + 1u] -= cup_local_center.y;
        gs.config.cup.hull_verts[i * 3u + 2u] -= cup_local_center.z;
    }
    gs.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = kGraspCupInvMass;
    const float ix = kGraspCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kGraspCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kGraspCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = cup_center;
    // A3 discriminative-IC knob: raise ONLY the cup body's start Z above the (fixed)
    // fingertip catch plane. At offset 0 cup.position == cup_center (bit-identical).
    cup.position.z += cup_start_z_offset;
    cup.orientation = Quat::Identity();
    cup.linear_velocity = Vec3::Zero();
    cup.angular_velocity = Vec3::Zero();
    gs.config.cup_state = cup;

    const uint32_t link_count = gs.gripper.host.TotalLinkCount();
    gs.config.grip_torque.assign(link_count, 0.0f);
    gs.config.grip_torque[gs.gripper.finger_link[0]] = grip_force;
    gs.config.grip_torque[gs.gripper.finger_link[1]] = grip_force;
    gs.config.drive_force_limits.assign(link_count, 0.0f);
    gs.config.friction_mu = mu;
    gs.config.condim = 3u;
    return gs;
}

BatchedSceneTemplate MakeGraspTemplate(const GraspSceneBundle& gs) {
    BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {gs.config.cup_state};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = gs.gripper.host;
    tmpl.fingertips = gs.config.fingertips;
    tmpl.cup = gs.config.cup;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque = gs.config.grip_torque;
    tmpl.drive_force_limits = gs.config.drive_force_limits;
    tmpl.friction_mu = gs.config.friction_mu;
    tmpl.condim = gs.config.condim;
    return tmpl;
}

}  // namespace nuka::runtime::coresident
