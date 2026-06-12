// ---------------------------------------------------------------------------
// BatchedSceneTemplate -> nk::Model (M4 transitional cook bridge).
// See h1_union_nk_model.hpp for the mapping contract.
// ---------------------------------------------------------------------------

#include "runtime/coresident/h1_union_nk_model.hpp"

#include <stdexcept>

namespace nuka::runtime::coresident {

namespace {

uint32_t JointDof(articulation::ArticulationJointType t) {
    switch (t) {
        case articulation::ArticulationJointType::Fixed: return 0u;
        case articulation::ArticulationJointType::FloatingBase: return 6u;
        default: return 1u;
    }
}

}  // namespace

nk::Model BuildNkUnionModel(const BatchedSceneTemplate& tmpl, uint32_t env_count) {
    if (!tmpl.has_grasp) {
        throw std::runtime_error(
            "BuildNkUnionModel: the union family needs the grasp articulation "
            "(has_grasp) — a template without it has no nk mapping yet");
    }
    const articulation::ArticulationHostState& proto = tmpl.gripper_proto;
    if (proto.ArticulationCount() != 1u) {
        throw std::runtime_error(
            "BuildNkUnionModel: expected exactly ONE proto articulation");
    }

    nk::Model model;
    model.contact_family = nk::ContactFamily::UnionCsr;
    model.drive_mode = 1u;  // torque drive (LaunchApplyTorqueDriveKernels path).

    // ---- articulation template (the SETTLED proto, 1:1) ---------------------
    nk::ModelArticulation& a = model.articulation;
    const uint32_t L = proto.TotalLinkCount();
    a.link_count = L;
    a.root_link = 0u;
    a.parent_link = proto.parent_link;
    a.joint_axis = proto.joint_axis;
    a.parent_offset = proto.parent_offset;
    a.link_local_pose = proto.link_local_pose;
    a.link_inertial_frame = proto.link_inertial_frame;
    a.joint_damping = proto.joint_damping;
    a.joint_armature = proto.joint_armature;
    a.initial_q = proto.q;
    a.initial_qdot = proto.qdot;
    a.initial_link_pose = proto.link_pose;
    a.link_body = proto.link_body;
    a.base_pose = proto.base_pose.empty() ? math::Transform::Identity()
                                          : proto.base_pose.front();
    a.joint_type.clear();
    a.joint_type.reserve(L);
    for (auto jt : proto.joint_type) {
        a.joint_type.push_back(static_cast<uint8_t>(jt));
    }
    a.link_inertia_spatial.assign(static_cast<size_t>(L) * 36u, 0.0f);
    for (uint32_t l = 0; l < L && l < proto.link_inertia.size(); ++l) {
        for (uint32_t k = 0; k < 36u; ++k) {
            a.link_inertia_spatial[static_cast<size_t>(l) * 36u + k] =
                proto.link_inertia[l].I[k];
        }
    }
    a.initial_link_velocity.assign(static_cast<size_t>(L) * 6u, 0.0f);
    for (uint32_t l = 0; l < L && l < proto.link_velocity.size(); ++l) {
        for (uint32_t k = 0; k < 6u; ++k) {
            a.initial_link_velocity[static_cast<size_t>(l) * 6u + k] =
                proto.link_velocity[l].v[k];
        }
    }
    uint32_t dofs = 0;
    for (auto jt : proto.joint_type) dofs += JointDof(jt);
    a.dof_count = dofs;

    // ---- the flat-DOF -> (local link, base component) maps (DofIndexOf^-1) --
    model.dof_to_link.clear();
    model.dof_to_component.clear();
    for (uint32_t l = 0; l < L; ++l) {
        const auto jt = proto.joint_type[l];
        if (l == 0u && proto.parent_link[l] == ~0u &&
            jt == articulation::ArticulationJointType::FloatingBase) {
            for (uint32_t b = 0; b < 6u; ++b) {
                model.dof_to_link.push_back(l);
                model.dof_to_component.push_back(b);
            }
            continue;
        }
        if (JointDof(jt) == 1u) {
            model.dof_to_link.push_back(l);
            model.dof_to_component.push_back(~0u);
        } else if (JointDof(jt) == 6u) {
            throw std::runtime_error(
                "BuildNkUnionModel: non-root FloatingBase unsupported");
        }
    }
    if (model.dof_to_link.size() != dofs) {
        throw std::runtime_error("BuildNkUnionModel: dof map / dof count mismatch");
    }

    // ---- torque drive template (grip torque + force limits, per link) -------
    nk::ModelHoldDrives& d = model.hold_drives;
    d.targets.assign(L, 0.0f);
    d.stiffness.assign(L, 0.0f);
    d.damping.assign(L, 0.0f);
    d.force_limits.assign(L, 0.0f);
    for (uint32_t l = 0; l < L && l < tmpl.grip_torque.size(); ++l) {
        d.targets[l] = tmpl.grip_torque[l];
    }
    for (uint32_t l = 0; l < L && l < tmpl.drive_force_limits.size(); ++l) {
        d.force_limits[l] = tmpl.drive_force_limits[l];
    }

    // ---- rigid bodies (the cup template) ------------------------------------
    for (const auto& b : tmpl.bodies_per_env) {
        nk::Model::BodyInit init;
        init.pose.position = b.position;
        init.pose.rotation = b.orientation;
        init.linear_velocity = b.linear_velocity;
        init.angular_velocity = b.angular_velocity;
        init.inv_mass = b.inv_mass;
        init.inv_inertia = b.inv_inertia;
        model.body_init.push_back(init);
    }

    // ---- union contact slots: FEET then FINGERS then TABLE (legacy order) ---
    const uint32_t condim = tmpl.condim;
    if (tmpl.has_feet) {
        if (!tmpl.has_grasp) {
            throw std::runtime_error("BuildNkUnionModel: has_feet without has_grasp");
        }
        for (const CoResidentFootSphere& f : tmpl.feet) {
            nk::UnionSlot s;
            s.cls = nk::UnionSlot::kFootSpherePlane;
            s.link = f.link;
            s.offset = f.local_offset;
            s.radius = f.radius;
            s.plane_height = tmpl.ground.height;
            s.mu = tmpl.foot_mu;
            s.condim = condim;
            model.union_slots.push_back(s);
        }
    }
    if (tmpl.cup.VertexCount() > 0u) {
        for (const CoResidentFingertip& f : tmpl.fingertips) {
            nk::UnionSlot s;
            s.cls = nk::UnionSlot::kFingerSphereHull;
            s.link = f.link;
            s.body = tmpl.cup_local_index;
            s.offset = f.local_offset;
            s.radius = f.radius;
            s.mu = tmpl.friction_mu;
            s.condim = condim;
            model.union_slots.push_back(s);
        }
        model.hull_verts = tmpl.cup.hull_verts;
    }
    if (tmpl.has_table) {
        const bool use_proxy = tmpl.cup_table_proxy_half.x > 0.0f ||
                               tmpl.cup_table_proxy_half.y > 0.0f ||
                               tmpl.cup_table_proxy_half.z > 0.0f;
        if (!use_proxy) {
            throw std::runtime_error(
                "BuildNkUnionModel: hull-x-plane table fallback not mapped "
                "(the production union scene uses the flat-bottom proxy box)");
        }
        nk::UnionSlot s;
        s.cls = nk::UnionSlot::kBodyBoxPlane;
        s.body = tmpl.cup_local_index;
        s.offset = tmpl.cup_table_proxy_offset;
        s.box_half = tmpl.cup_table_proxy_half;
        s.plane_height = tmpl.table_height;
        s.mu = tmpl.table_mu;
        s.condim = condim;
        s.flags = nk::kUnionSlotGatedOnTable;
        model.union_slots.push_back(s);
    }
    model.table_enabled_default = tmpl.has_table;

    // ---- capacities ----------------------------------------------------------
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = env_count > 0 ? env_count : 1u;
    cap.dofs_per_env = dofs;
    cap.links_per_env = L;
    cap.bodies_per_env = static_cast<uint32_t>(tmpl.bodies_per_env.size());
    cap.max_contacts_per_env = static_cast<uint32_t>(model.union_slots.size());
    uint32_t rows = 0;
    for (const nk::UnionSlot& s : model.union_slots) rows += s.MaxRows();
    cap.max_rows_per_env = rows;
    cap.max_hull_verts = tmpl.cup.VertexCount();
    cap.num_material_buckets = 0;

    return model;
}

}  // namespace nuka::runtime::coresident
