// ---------------------------------------------------------------------------
// nuka::runtime::BuildWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/world_builder.hpp"

#include "runtime/articulation/articulation_cooker.hpp"

namespace nuka::runtime {

BuiltWorld BuildWorld(const scene::CookedBlob& blob) {
    BuiltWorld result;

    // --- Template: copy cooked tables ---
    auto& tmpl         = result.template_view;
    tmpl.body_count    = blob.body_count;
    tmpl.joint_count   = blob.joint_count;
    tmpl.shape_count   = blob.shape_count;
    tmpl.actuator_count = blob.actuator_count;
    tmpl.body_table    = blob.bodies;
    tmpl.joint_table   = blob.joints;
    tmpl.shape_table   = blob.shapes;
    tmpl.actuator_table = blob.actuators;
    tmpl.convex_geometry = blob.convex_geometry;  // v0.7 p06
    tmpl.sdf_table     = blob.sdfs;                // v0.7 p07
    tmpl.articulations = articulation::CookArticulations(blob);

    // --- Instance: initialise mutable state ---
    auto& inst        = result.instance;
    inst.body_count   = blob.body_count;
    inst.poses        = blob.bodies.poses;                     // from template
    inst.linear_velocities.resize(blob.body_count, math::Vec3::Zero());
    inst.angular_velocities.resize(blob.body_count, math::Vec3::Zero());
    inst.forces.resize(blob.body_count, math::Vec3::Zero());
    inst.torques.resize(blob.body_count, math::Vec3::Zero());
    const auto articulation_state =
        articulation::BuildArticulationHostState(tmpl.articulations, tmpl.body_table);
    inst.articulation_q = articulation_state.q;
    inst.articulation_qdot = articulation_state.qdot;
    inst.articulation_qddot = articulation_state.qddot;
    inst.articulation_tau = articulation_state.tau;

    // --- Batch: single template, single instance ---
    result.batch.template_count = 1;
    result.batch.instance_count = 1;

    return result;
}

} // namespace nuka::runtime
