// ---------------------------------------------------------------------------
// nuka::scene::Compose implementation
// ---------------------------------------------------------------------------

#include "scene/scene_compose.hpp"

#include "scene/canonical_types.hpp"

#include <cstdint>
#include <utility>

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

    // -- Materials ----------------------------------------------------------
    // (No cross-refs; id reassigned by AddMaterial. Append in order.)
    for (const MaterialRecord& src : addon.Materials()) {
        MaterialRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        out.AddMaterial(std::move(rec));
    }

    // -- Rigid bodies -------------------------------------------------------
    for (const RigidBodyRecord& src : addon.Bodies()) {
        RigidBodyRecord rec = src;
        rec.name = PrefixName(addon_name_prefix, rec.name);
        rec.parent_id = RemapId(rec.parent_id, body_off);
        // Re-root the addon: each root (parent_id == kInvalidBody) is placed at
        // `placement` in base's frame. Non-root bodies keep their local
        // transform (their world pose follows the re-rooted parent).
        if (src.parent_id == kInvalidBody) {
            rec.local_transform = placement * rec.local_transform;
        }
        out.AddRigidBody(std::move(rec));
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

    return out;
}

}  // namespace nuka::scene
