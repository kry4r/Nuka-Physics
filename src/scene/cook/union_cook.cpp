// ---------------------------------------------------------------------------
// nuka::scene::cook — CookSceneToUnionTemplate (M7 T4) implementation.
//
// Reproduces h1_union_scene_factory.cpp's MakeUnionTemplate OUTPUT (+ the
// H1UnionScene metadata) by READING the authored / baked .nks-loaded SceneIR
// instead of generating via the factory's FK / placement / 200-step settle.
//
// THE FACTORY-EQUIVALENCE MAP (each factory step -> what this cook reads):
//   * LoadFloatingGraspCook (factory :447): LoadMjcf -> DROP MESH (inertia +
//     topology only, dodge V-HACD) -> CookScene -> CookArticulations, with the
//     12 wrap-driven finger BODIES pinned joint_damping=1.0 / armature=0.1.
//     REPRODUCED HERE on a working copy of the imported SceneIR (the .nks
//     `imports h1_with_hand.xml` already composed the H1 records in; the cook
//     re-applies the mesh-drop + finger-pin so the cooked articulation matches
//     the factory's bit-for-bit on topology/inertia/damping).
//   * Quiescent IC + stance + seat + curl + 200-step settle (factory steps
//     2..6): the SETTLED per-link q + base pose are BAKED in initial_state["h1"]
//     (R4). Applied straight to host.q / base_pose; qdot / link_velocity are
//     ZEROED (the .nks initial_state stores only q + root — see the report's
//     schema flag: the cooked IC is the settled POSE, not the residual settle
//     velocities the factory's gripper_proto carried).
//   * 30 wrap + 4 foot spheres (factory WrapSpheres / foot toe-heel loop):
//     READ from GraspConfig.wrap_spheres / .foot_spheres, link_name re-resolved
//     to the cooked device link, emitted feet -> fingers (MakeUnionTemplate's
//     order; the table slot is appended last by BuildNkUnionModel).
//   * Cup hull + scaled inertia + settled placement (factory LoadCupHull ->
//     ScaleCupHull -> COM-centre -> placement -> hand-carry): the COM-centred
//     scaled hull verts are in GraspConfig.cup_hull_verts; the settled cup pose
//     + the scaled-bbox inertia are on the cup tree BODY record.
//   * Drive tables / dof_torque_limit / poly_cx / total_mass / ankle CoP tau
//     (factory ExportDriveTable / the per-DOF limit loop): all BAKED in the
//     GraspConfig; the cook re-resolves each drive entry's link_name -> device
//     link + flat DOF column.
//
// DOF-COUNT INLINING (== cook_to_model.cpp / h1_union_nk_model.cpp): the
// ArticulationJointDofCount / DofIndexOf symbols live in the GPU lib
// (articulation_contacts.cu); this pure-cook TU inlines the joint-DOF switch
// (Fixed->0, FloatingBase->6, else->1) so it links only nuka_articulation.
// ---------------------------------------------------------------------------

#include "scene/cook/union_cook.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"  // CoResident* descriptors
#include "runtime/rigid/body_state.hpp"
#include "scene/cooker.hpp"

namespace nuka::scene::cook {

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

constexpr uint32_t kInvalidLink = ~0u;

// The joint-DOF count (inlined ArticulationJointDofCount semantics; that symbol
// lives in the GPU articulation lib the pure cook must not link).
uint32_t JointDof(articulation::ArticulationJointType t) {
    switch (t) {
        case articulation::ArticulationJointType::Fixed: return 0u;
        case articulation::ArticulationJointType::FloatingBase: return 6u;
        default: return 1u;
    }
}

// The flat DOF column of `link` (prefix-sum of the joint DOF counts from the
// root link up to `link`) — inlined DofIndexOf (factory :775, byte-identical).
uint32_t DofIndexOf(const articulation::ArticulationHostState& host,
                    uint32_t root_link, uint32_t link) {
    uint32_t idx = 0u;
    for (uint32_t l = root_link; l < link; ++l) {
        idx += JointDof(host.joint_type[l]);
    }
    return idx;
}

// The cooked H1 + the SceneIR it was cooked from (so link <-> body-name
// resolution works exactly like the factory's LinkName / LinkByName).
struct CookedH1 {
    articulation::ArticulationHostState host;
    SceneIR scene;  // the mesh-dropped working copy that was cooked.
};

// The cooked link's owning body NAME (== factory LinkName :117).
std::string LinkName(const CookedH1& c, uint32_t link) {
    if (c.host.link_body.size() <= link) return "?";
    const uint32_t body = c.host.link_body[link];
    if (body < c.scene.Bodies().size()) return c.scene.GetBody(body).name;
    return "?";
}

// Resolve a body NAME to its cooked device link (== factory LinkByName :123).
uint32_t LinkByName(const CookedH1& c, const std::string& name) {
    for (uint32_t l = 0u; l < c.host.TotalLinkCount(); ++l) {
        if (LinkName(c, l) == name) return l;
    }
    return kInvalidLink;
}

// Drop H1 mesh geometry + pin the 12 wrap-driven finger joints (damping=1.0 /
// armature=0.1), then cook the articulation — the factory's LoadFloatingGraspCook
// (:447) reproduced on the .nks-imported SceneIR. The 12 driven body names are the
// grasp block's grip_links (the factory's kWrapDriven set, authored into the .nks).
CookedH1 CookH1(const SceneIR& imported, const GraspConfig& grasp) {
    CookedH1 out;
    out.scene = imported;  // deep copy (SceneIR rebuilds its facade from records).

    // Drop mesh geometry on every shape (== LoadFloating / LoadFloatingGraspCook:
    // inertia + topology only — dodge V-HACD). Operates on the working copy.
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }

    const CookedBlob blob = CookScene(out.scene);
    std::vector<articulation::ArticulationCookedTopology> topos =
        articulation::CookArticulations(blob);

    // Resolve the wrap-driven finger BODIES by name (the grip_links set) and pin
    // their joint damping/armature to the H1.1 driven-finger values (factory
    // kRealDamping=1.0 / kRealArmature=0.1, :71-72). Same per-topology loop the
    // factory uses (:472-478).
    constexpr float kRealDamping = 1.0f;
    constexpr float kRealArmature = 0.1f;
    std::vector<uint32_t> driven_bodies;
    driven_bodies.reserve(grasp.grip_links.size());
    for (const std::string& nm : grasp.grip_links) {
        for (uint32_t b = 0u; b < out.scene.RigidBodyCount(); ++b) {
            if (out.scene.GetBody(b).name == nm) {
                driven_bodies.push_back(b);
                break;
            }
        }
    }
    auto is_driven = [&](uint32_t body) {
        return std::find(driven_bodies.begin(), driven_bodies.end(), body) !=
               driven_bodies.end();
    };
    for (auto& topo : topos) {
        for (size_t i = 0u; i < topo.link_bodies.size(); ++i) {
            if (!is_driven(topo.link_bodies[i])) continue;
            topo.joint_dampings[i] = kRealDamping;
            topo.joint_armatures[i] = kRealArmature;
        }
    }

    out.host = articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
}

// Build one H1UnionDriveEntry from a name-keyed GraspDriveEntry, resolving the
// link name -> device link + flat DOF column against the cooked H1. Unresolved
// names (a link the cook did not produce) are SKIPPED (returns false) — the
// factory's drive tables only carry resolved links, so a clean .nks resolves
// every entry; a skip would surface as a drive-table length mismatch in the gate.
bool ResolveDrive(const CookedH1& c, uint32_t root_link,
                  const GraspDriveEntry& src, coresident::H1UnionDriveEntry* out) {
    const uint32_t link = LinkByName(c, src.link_name);
    if (link == kInvalidLink) return false;
    out->link = link;
    out->dof = DofIndexOf(c.host, root_link, link);
    out->target = src.target;
    out->kp = src.kp;
    out->kd = src.kd;
    out->tlim = src.tlim;
    out->grip = src.grip;
    return true;
}

std::vector<coresident::H1UnionDriveEntry> ResolveDriveTable(
    const CookedH1& c, uint32_t root_link,
    const std::vector<GraspDriveEntry>& src) {
    std::vector<coresident::H1UnionDriveEntry> out;
    out.reserve(src.size());
    for (const GraspDriveEntry& d : src) {
        coresident::H1UnionDriveEntry e;
        if (ResolveDrive(c, root_link, d, &e)) out.push_back(e);
    }
    return out;
}

// Find the movable CUP body in the loaded SceneIR: the non-static body that owns
// a ConvexHull collision shape. (The cup tree node carries the SETTLED pose +
// the scaled-bbox inertia + mass; robust against record naming.)
BodyId FindCupBody(const SceneIR& scene) {
    for (const CollisionShapeRecord& s : scene.Shapes()) {
        if (s.type != ShapeType::ConvexHull) continue;
        if (s.body_id == kInvalidBody || s.body_id >= scene.Bodies().size()) continue;
        if (scene.GetBody(s.body_id).is_static) continue;
        return s.body_id;
    }
    return kInvalidBody;
}

// World transform of a body (walk the parent chain to the root, compose locals).
// The cup/table tree bodies are ROOTS (parent == kInvalidBody) so this is just
// their local_transform; the walk keeps it correct for any future re-parenting.
Transform BodyWorldTransform(const SceneIR& scene, BodyId body) {
    Transform out = Transform::Identity();
    BodyId cur = body;
    while (cur != kInvalidBody && cur < scene.Bodies().size()) {
        const RigidBodyRecord& rec = scene.GetBody(cur);
        out = rec.local_transform * out;
        cur = rec.parent_id;
    }
    return out;
}

}  // namespace

CookedUnionScene CookSceneToUnionTemplate(const SceneIR& scene, int env_count) {
    const uint32_t envs = env_count > 0 ? static_cast<uint32_t>(env_count) : 1u;
    (void)envs;  // The union template is the PER-ENV scene; BuildNkUnionModel /
                 // BatchedUnifiedWorld replicate it across envs at construction.

    const GraspConfig& grasp = scene.Grasp();
    if (!grasp.present) {
        throw std::runtime_error(
            "CookSceneToUnionTemplate: the SceneIR carries no GraspConfig block "
            "(scene.Grasp().present == false) — this cook needs the authored "
            "union grasp metadata (load an h1_cup_table-class .nks)");
    }

    // ----- cook the H1 import (mesh-drop + finger-pin == LoadFloatingGraspCook) -
    const CookedH1 h1 = CookH1(scene, grasp);
    const uint32_t link_count = h1.host.TotalLinkCount();
    if (link_count == 0u) {
        throw std::runtime_error(
            "CookSceneToUnionTemplate: the imported H1 cooked to ZERO links — the "
            "scene's `imports h1_with_hand.xml` did not produce an articulation");
    }
    const uint32_t root_link = h1.host.articulation_link_offset.empty()
                                   ? 0u
                                   : h1.host.articulation_link_offset[0];

    CookedUnionScene out;

    // ----- apply the BAKED settled IC (initial_state["h1"]) -------------------
    // qpos -> host.q (per-link, length == TotalLinkCount); root -> base pose.
    // qdot / link_velocity are zeroed (the .nks stores only the settled POSE; see
    // the report schema flag). The articulation cook already set the rest q; we
    // overwrite with the settled q.
    articulation::ArticulationHostState proto = h1.host;
    for (auto& v : proto.link_velocity)
        for (float& cv : v.v) cv = 0.0f;
    for (float& qd : proto.qdot) qd = 0.0f;

    const SceneInitialState& ic = scene.InitialState();
    const auto it = ic.find("h1");
    if (it == ic.end()) {
        throw std::runtime_error(
            "CookSceneToUnionTemplate: initial_state has no \"h1\" entry — the "
            "baked settled stance IC (R4) is required (run from the authored "
            "h1_cup_table.nks)");
    }
    const ArticulationInitialState& art_ic = it->second;
    if (art_ic.qpos.size() != proto.q.size()) {
        throw std::runtime_error(
            "CookSceneToUnionTemplate: initial_state qpos length (" +
            std::to_string(art_ic.qpos.size()) + ") != cooked link count (" +
            std::to_string(proto.q.size()) +
            ") — the baked IC does not match this cook's articulation");
    }
    proto.q = art_ic.qpos;
    if (!proto.base_pose.empty()) proto.base_pose[0] = art_ic.root;

    // ----- the cup BodyState (settled pose + scaled-bbox inertia) -------------
    const BodyId cup_body = FindCupBody(scene);
    if (cup_body == kInvalidBody) {
        throw std::runtime_error(
            "CookSceneToUnionTemplate: no movable ConvexHull cup body in the scene "
            "(the cup tree node carries the settled cup pose + inertia)");
    }
    const RigidBodyRecord& cup_rec = scene.GetBody(cup_body);
    const Transform cup_pose = BodyWorldTransform(scene, cup_body);
    BodyState cup;
    cup.inv_mass = cup_rec.mass > 0.0f ? 1.0f / cup_rec.mass : 0.0f;
    cup.inv_inertia = Vec3{
        cup_rec.inertia.x > 0.0f ? 1.0f / cup_rec.inertia.x : 0.0f,
        cup_rec.inertia.y > 0.0f ? 1.0f / cup_rec.inertia.y : 0.0f,
        cup_rec.inertia.z > 0.0f ? 1.0f / cup_rec.inertia.z : 0.0f};
    cup.position = cup_pose.position;
    cup.orientation = cup_pose.rotation;

    // ----- the wrap (finger) + foot contact spheres ---------------------------
    // wrap -> FingerSphereHull (vs cup); foot -> FootSpherePlane (vs ground).
    // Resolved + emitted in the GraspConfig authoring order (== MakeUnionTemplate:
    // fingertips from config, feet from sg; the .nks preserves that order).
    std::vector<coresident::CoResidentFingertip> fingertips;
    fingertips.reserve(grasp.wrap_spheres.size());
    for (const GraspContactSphere& s : grasp.wrap_spheres) {
        const uint32_t l = LinkByName(h1, s.link_name);
        if (l == kInvalidLink) continue;  // factory `continue`s an unresolved name.
        coresident::CoResidentFingertip ft;
        ft.link = l;
        ft.broadphase_handle = s.broadphase_id;
        ft.local_offset = s.local_offset;
        ft.radius = s.radius;
        fingertips.push_back(ft);
    }
    std::vector<coresident::CoResidentFootSphere> feet;
    feet.reserve(grasp.foot_spheres.size());
    for (const GraspContactSphere& s : grasp.foot_spheres) {
        const uint32_t l = LinkByName(h1, s.link_name);
        if (l == kInvalidLink) continue;
        coresident::CoResidentFootSphere fs;
        fs.link = l;
        fs.broadphase_handle = s.broadphase_id;
        fs.local_offset = s.local_offset;
        fs.radius = s.radius;
        feet.push_back(fs);
    }
    const bool has_feet = !feet.empty();
    const bool has_table = grasp.cup_table_proxy_half.x > 0.0f ||
                           grasp.cup_table_proxy_half.y > 0.0f ||
                           grasp.cup_table_proxy_half.z > 0.0f;

    // ----- assemble the BatchedSceneTemplate (== MakeUnionTemplate :566) -------
    coresident::BatchedSceneTemplate& tmpl = out.tmpl;
    tmpl.bodies_per_env = {cup};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = proto;
    tmpl.fingertips = fingertips;
    tmpl.cup.hull_verts = grasp.cup_hull_verts;
    tmpl.cup.broadphase_body_id = grasp.cup_broadphase_id;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque.assign(link_count, 0.0f);          // drive is via SetActions.
    tmpl.drive_force_limits.assign(link_count, 0.0f);
    tmpl.friction_mu = grasp.finger_mu;
    tmpl.condim = 3u;
    tmpl.has_feet = has_feet;
    if (has_feet) {
        tmpl.feet = feet;
        tmpl.ground.height = grasp.ground_height;
        tmpl.ground.broadphase_id = grasp.ground_broadphase_id;
        tmpl.foot_mu = grasp.foot_mu;
    }
    tmpl.has_table = has_table;
    if (has_table) {
        tmpl.table_height = grasp.table_height;
        tmpl.table_mu = grasp.table_mu;
        tmpl.table_broadphase_id = grasp.table_broadphase_id;
        tmpl.cup_table_proxy_half = grasp.cup_table_proxy_half;
        tmpl.cup_table_proxy_offset = grasp.cup_table_proxy_offset;
        tmpl.cup_table_proxy_id = grasp.cup_table_proxy_id;
    }
    // reset_jitter_x/y keep their template defaults (kDefaultResetCupJitterM) —
    // the factory likewise never sets them (A5a is a runtime knob, not authored).

    // ----- the consumer-facing metadata (== H1UnionScene fields) --------------
    out.dof_stride = grasp.dof_stride;
    out.base_dof = grasp.base_dof;
    out.root_link = root_link;
    out.link_count = link_count;
    out.total_mass = grasp.total_mass;
    out.cup_mass = grasp.cup_mass;
    out.poly_cx = grasp.poly_cx;
    out.place_found = true;  // the .nks carries a placed cup (R4-baked).
    out.table_height = grasp.table_height;
    out.ankle_settle_tau[0] = grasp.ankle_settle_tau[0];
    out.ankle_settle_tau[1] = grasp.ankle_settle_tau[1];

    // grip links + their flat DOF columns (resolve grasp.grip_links by name).
    out.grip_links.reserve(grasp.grip_links.size());
    out.grip_dofs.reserve(grasp.grip_links.size());
    for (const std::string& nm : grasp.grip_links) {
        const uint32_t l = LinkByName(h1, nm);
        if (l == kInvalidLink) continue;
        out.grip_links.push_back(l);
        out.grip_dofs.push_back(DofIndexOf(h1.host, root_link, l));
    }

    // the three reference drive tables (resolve each name-keyed entry).
    out.drive_hold = ResolveDriveTable(h1, root_link, grasp.drive_hold);
    out.drive_rest = ResolveDriveTable(h1, root_link, grasp.drive_rest);
    out.drive_close = ResolveDriveTable(h1, root_link, grasp.drive_close);

    // per-DOF torque limits (length dof_stride; baked, copied verbatim).
    out.dof_torque_limit = grasp.dof_torque_limit;

    return out;
}

}  // namespace nuka::scene::cook
