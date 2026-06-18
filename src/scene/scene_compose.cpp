// ---------------------------------------------------------------------------
// nuka::scene::Compose implementation
// ---------------------------------------------------------------------------

#include "scene/scene_compose.hpp"

#include "scene/canonical_types.hpp"

#include <cstdint>
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

    // A body takes the placement only if nothing joints onto it as a child: flat
    // body lists (all parent_id kInvalidBody) carry their tree in joints, not parents.
    std::vector<uint8_t> addon_is_joint_child(addon.Bodies().size(), 0u);
    for (const JointRecord& j : addon.Joints()) {
        if (j.child_body != kInvalidBody &&
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
    for (const SensorRecord& src : addon.Sensors()) {
        SensorRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.attached_body = RemapId(rec.attached_body, body_off);
        out.AddSensor(std::move(rec));
    }

    // -- Cameras ------------------------------------------------------------
    for (const CameraRecord& src : addon.Cameras()) {
        CameraRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.attached_body = RemapId(rec.attached_body, body_off);
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

    return out;
}

}  // namespace nuka::scene
