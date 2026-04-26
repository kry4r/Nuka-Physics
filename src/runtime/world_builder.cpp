// ---------------------------------------------------------------------------
// nuka::runtime::BuildWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/world_builder.hpp"

namespace nuka::runtime {

BuiltWorld BuildWorld(const scene::CookedBlob& blob) {
    BuiltWorld result;

    // --- Template: copy cooked tables ---
    auto& tmpl         = result.template_view;
    tmpl.body_count    = blob.body_count;
    tmpl.joint_count   = blob.joint_count;
    tmpl.shape_count   = blob.shape_count;
    tmpl.body_table    = blob.bodies;
    tmpl.joint_table   = blob.joints;
    tmpl.shape_table   = blob.shapes;

    // --- Instance: initialise mutable state ---
    auto& inst        = result.instance;
    inst.body_count   = blob.body_count;
    inst.poses        = blob.bodies.poses;                     // from template
    inst.linear_velocities.resize(blob.body_count, math::Vec3::Zero());
    inst.angular_velocities.resize(blob.body_count, math::Vec3::Zero());
    inst.forces.resize(blob.body_count, math::Vec3::Zero());
    inst.torques.resize(blob.body_count, math::Vec3::Zero());

    // --- Batch: single template, single instance ---
    result.batch.template_count = 1;
    result.batch.instance_count = 1;

    return result;
}

} // namespace nuka::runtime
