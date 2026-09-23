// ---------------------------------------------------------------------------
// nuka::scene::Compose implementation
// ---------------------------------------------------------------------------

#include "scene/scene_compose.hpp"

#include "scene/canonical_types.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nuka::scene {

namespace {

// Offset a cross-reference id by `off`, preserving the kInvalid* sentinel.
// All scene ids are uint32_t and the three sentinels (kInvalidBody /
// kInvalidJoint / kInvalidMaterial) share the same ~uint32_t(0) value, so a
// single helper is correct for every id kind.
inline uint32_t RemapId(uint32_t id, uint32_t off) {
    if (id == kInvalidBody) {  // == kInvalidJoint == kInvalidMaterial
        return id;
    }
    return id + off;
}

// Prepend `prefix` to `name` only when the prefix is non-empty.
inline std::string PrefixName(const std::string& prefix, const std::string& name) {
    if (prefix.empty()) {
        return name;
    }
    return prefix + name;
}

}  // namespace

SceneIR Compose(const SceneIR& base, const SceneIR& addon,
                const math::Transform& placement,
                const std::string& addon_name_prefix) {
    // Start from a copy of base; Add* on the copy assigns dense ids starting at
    // base's current count, which is exactly the offset every addon id needs.
    SceneIR out = base;

    const auto body_off = static_cast<uint32_t>(base.RigidBodyCount());
    const auto mat_off  = static_cast<uint32_t>(base.MaterialCount());
    const auto joint_off = static_cast<uint32_t>(base.JointCount());
    const auto shape_off = static_cast<uint32_t>(base.ShapeCount());

    const auto& rotation = placement.rotation;
    const bool axis_aligned_media = std::any_of(addon.Media().begin(), addon.Media().end(),
        [](const MediaRecord& m) { return m.kind != MediaRecord::Kind::Cloth || m.baked.Empty(); });
    if ((axis_aligned_media || !addon.Terrain().empty()) &&
        (rotation.x != 0.0f || rotation.y != 0.0f || rotation.z != 0.0f))
        throw std::invalid_argument("Procedural media and terrain require axis-aligned placement");

    // Only a joint to another body makes a body parent-relative.
    // A joint to the world still leaves its child at the scene root.
    std::vector<uint8_t> addon_is_joint_child(addon.Bodies().size(), 0u);
    for (const JointRecord& j : addon.Joints()) {
        if (j.parent_body != kInvalidBody && j.child_body != kInvalidBody &&
            j.child_body < addon_is_joint_child.size()) {
            addon_is_joint_child[j.child_body] = 1u;
        }
    }

    // -- Materials ----------------------------------------------------------
    // (No cross-refs; id reassigned by AddMaterial. Append in order.)
    for (const MaterialRecord& src : addon.Materials()) {
        MaterialRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        out.AddMaterial(std::move(rec));
    }

    // -- Rigid bodies -------------------------------------------------------
    uint32_t addon_body_idx = 0u;
    for (const RigidBodyRecord& src : addon.Bodies()) {
        RigidBodyRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.parent_id = RemapId(rec.parent_id, body_off);
        // Re-root only a true root: no body parent AND not a joint child (a joint
        // child's local_transform is parent-relative, reached via the re-rooted root).
        const bool is_joint_child =
            addon_body_idx < addon_is_joint_child.size() &&
            addon_is_joint_child[addon_body_idx] != 0u;
        if (src.parent_id == kInvalidBody && !is_joint_child) {
            rec.local_transform = placement * rec.local_transform;
        }
        out.AddRigidBody(std::move(rec));
        ++addon_body_idx;
    }

    // -- Collision shapes ---------------------------------------------------
    for (const CollisionShapeRecord& src : addon.Shapes()) {
        CollisionShapeRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.body_id = RemapId(rec.body_id, body_off);
        rec.material_id = RemapId(rec.material_id, mat_off);
        if (src.body_id == kInvalidBody) rec.local_transform = placement * rec.local_transform;
        out.AddCollisionShape(std::move(rec));
    }

    // -- Joints -------------------------------------------------------------
    for (const JointRecord& src : addon.Joints()) {
        JointRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.parent_body = RemapId(rec.parent_body, body_off);
        rec.child_body = RemapId(rec.child_body, body_off);
        out.AddJoint(std::move(rec));
    }

    // -- Actuators ----------------------------------------------------------
    for (const ActuatorRecord& src : addon.Actuators()) {
        ActuatorRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.joint_id = RemapId(rec.joint_id, joint_off);
        out.AddActuator(std::move(rec));
    }

    // -- Sensors ------------------------------------------------------------
    // mount_index is a body row only for a Body mount; Link/Base mounts index
    // articulation-internal rows that compose does not renumber, so remap then.
    for (const SensorDesc& src : addon.Sensors()) {
        SensorDesc rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.joint_id = RemapId(rec.joint_id, joint_off);
        if (rec.mount == MountFrame::Body) {
            rec.mount_index = RemapId(rec.mount_index, body_off);
        }
        out.AddSensor(std::move(rec));
    }

    // -- Cameras ------------------------------------------------------------
    for (const CameraRecord& src : addon.Cameras()) {
        CameraRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.attached_body = RemapId(rec.attached_body, body_off);
        if (src.attached_body == kInvalidBody) rec.local_transform = placement * rec.local_transform;
        out.AddCamera(std::move(rec));
    }

    // -- Lights -------------------------------------------------------------
    // NOTE: LightRecord carries attached_body too (the task's enumerated remap
    // list omits it). It must be remapped or an addon light would silently
    // re-point at a base body after compose.
    for (const LightRecord& src : addon.Lights()) {
        LightRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.attached_body = RemapId(rec.attached_body, body_off);
        if (src.attached_body == kInvalidBody) rec.local_transform = placement * rec.local_transform;
        out.AddLight(std::move(rec));
    }

    // -- Collision-filter carry-over (v0.8 C1c) -----------------------------
    // `out = base` already copied base's exclude_pairs_ / contact_pairs_; here we
    // APPEND only the addon's, with the same append-with-offset remap the rest of
    // the compose uses. Body-pair ids offset by body_off; <pair> geom ShapeIds
    // offset by shape_off (shapes are reassigned dense ids by AddCollisionShape,
    // so an addon shape at index s lands at shape_off + s -- identical to body
    // remapping). AddExcludePair re-canonicalizes (min,max); offset preserves
    // ordering so canonical form is stable. kInvalid* sentinels stay sentinel
    // via RemapId.
    for (const std::pair<BodyId, BodyId>& e : addon.ExcludePairs()) {
        out.AddExcludePair(RemapId(e.first, body_off), RemapId(e.second, body_off));
    }
    for (const ContactPairOverride& src : addon.ContactPairs()) {
        ContactPairOverride pair = src;
        pair.geom1 = RemapId(pair.geom1, shape_off);
        pair.geom2 = RemapId(pair.geom2, shape_off);
        out.AddContactPair(pair);
    }

    for (const MediaRecord& src : addon.Media()) {
        MediaRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.render_material_id = RemapId(rec.render_material_id, mat_off);
        rec.cable_line.slab.render_material_id = RemapId(rec.cable_line.slab.render_material_id, mat_off);
        if (rec.kind == MediaRecord::Kind::Cloth && !rec.baked.Empty()) {
            rec.cloth_mesh.local_transform = placement * rec.cloth_mesh.local_transform;
            out.AddMedia(std::move(rec));
            continue;
        }
        const auto offset = placement.position;
        if ((!rec.baked.Empty() || !rec.render_skin.skin_mesh.Empty()) && offset.LengthSq() != 0.0f)
            throw std::invalid_argument("Baked media require placement in their mesh asset");
        rec.cloth_grid.origin += offset;
        rec.tet_sphere.center += offset;
        rec.fluid_box.min += offset;
        rec.fluid_box.max += offset;
        rec.cable_line.start += offset;
        rec.cable_line.end += offset;
        rec.pbf.walls_min += offset;
        rec.pbf.walls_max += offset;
        rec.pbf.floor_z += offset.z;
        rec.mpm.floor_d += rec.mpm.floor_normal.Dot(offset);
        for (auto& fill : rec.mpm_fills) {
            fill.box.min += offset;
            fill.box.max += offset;
            fill.render_material_id = RemapId(fill.render_material_id, mat_off);
        }
        out.AddMedia(std::move(rec));
    }
    for (const TerrainRecord& src : addon.Terrain()) {
        TerrainRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.origin += placement.position;
        rec.base_z += placement.position.z;
        out.AddTerrain(std::move(rec));
    }
    for (const auto& initial : addon.InitialState()) {
        auto state = initial.second;
        state.root = placement * state.root;
        out.InitialStateMut()[PrefixName(addon_name_prefix, initial.first)] = std::move(state);
    }
    if (addon.Settle().steps != 0 || !addon.Settle().holds.empty()) {
        auto& settle = out.SettleMut();
        if (settle.steps != 0 && (settle.steps != addon.Settle().steps || settle.dt != addon.Settle().dt))
            throw std::invalid_argument("Composed scenes require matching settle steps and timestep");
        settle.steps = addon.Settle().steps;
        settle.dt = addon.Settle().dt;
        for (auto hold : addon.Settle().holds) {
            hold.dof_pattern = PrefixName(addon_name_prefix, hold.dof_pattern);
            settle.holds.push_back(std::move(hold));
        }
    }
    for (const ScriptRecord& src : addon.Scripts()) {
        ScriptRecord rec = src;
        rec.parent_path = PrefixName(addon_name_prefix, rec.parent_path);
        if (!rec.parent_path.empty() && rec.parent_path.back() == '/') rec.parent_path.pop_back();
        out.AddScript(std::move(rec));
    }
    if (addon.Environment().Authored()) {
        out.EnvironmentMut() = addon.Environment();
        if (out.Environment().shadow.enabled)
            out.EnvironmentMut().shadow.center = placement.TransformPoint(addon.Environment().shadow.center);
    }
    return out;
}

}  // namespace nuka::scene
