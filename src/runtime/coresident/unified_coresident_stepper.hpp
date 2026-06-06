#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- the MULTI-STEP unified co-resident contact
// stepper (v0.8 W1a). VALIDATED-NOT-WIRED.
// ---------------------------------------------------------------------------
// WHY THIS EXISTS. The unified collision/contact spine (C1->C6) is complete but
// has only ever been driven by SINGLE-SOLVE test gates. The flagship C7 grasp
// demo needs a stepper that CO-RESIDES an articulation (a go2 foot/leg) and a
// MOVABLE rigid body (a box) and steps them TOGETHER -- predict -> detect ->
// solve -> integrate -> repeat -- so a contact HOLDS over many steps. No such
// stepper existed; this is it.
//
// ARCHITECTURE (a deliberate decision). It MIRRORS the stage order of the
// production BatchedArticulatedWorld::Step() (batched_articulated_world.cu ~531)
// but SWAPS the foot-vs-ground contact phase for the unified pipeline (per-link
// AABB -> artic<->rigid broadphase -> sphere x box narrowphase -> compliant rows
// -> UnifiedSolve) and ADDS a movable rigid box that reacts two-way.
//
// IT REUSES THE PRODUCTION INTEGRATOR VERBATIM. There is NO second integrator
// here. Every articulation advance routes through the already-validated
// FeatherstoneAba:: methods (the same ones Step() calls):
//   ComputeAccelerations / IntegrateVelocity / IntegrateFloatingBaseVelocity /
//   IntegratePosition / IntegrateFloatingBasePose.
// The box is a host BodyState advanced with the same symplectic-Euler scheme as
// the articulation (velocity stage before position stage).
//
// HOST-ORCHESTRATED. The articulation lives as a GPU ArticulationDeviceBuffers
// (UploadArticulationState once; stepped in place). The contact PHASE downloads
// the current link poses / qdot / base velocity, runs the host-side unified
// pipeline (broadphase/narrowphase/row emission/solve, exactly as the single-
// shot co-residence test does), and SCATTERS the post-contact velocities back:
//   flat-qdot[0..5]  -> link_velocity[root] (the base spatial velocity)
//   flat-qdot[6..N]  -> device qdot[leg links] (the joint velocities)
//   the box BodyState velocity is mutated in place.
// The post-contact velocity is what the stage-11 pose integrate then advances --
// matching Step()'s stage ordering exactly.
//
// ADDITIVE. This is a NEW translation unit. It does NOT modify
// BatchedArticulatedWorld, world_stepper, any FeatherstoneAba integration method
// (REUSE only), or any golden.
// ---------------------------------------------------------------------------

#include "collision/link_aabb.hpp"               // ExtractLinkShapeAabbs / LinkShapeAabbs
#include "constraint/row_builder.hpp"            // ContactRowComplianceInputs
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/cooked_blob.hpp"                  // scene::CookedShapeTable
#include "solver/rigid_solver.hpp"               // SolverConfig

#include <cstdint>
#include <vector>

namespace nuka::runtime::coresident {

// One go2 foot, identified the same way the co-residence test does: the calf link
// that owns the foot sphere + the sphere's local offset + radius.
struct CoResidentFoot {
    uint32_t calf_link = ~0u;      // the ArticulationLink handle (the broadphase id).
    math::Vec3 local_offset{};     // sphere center in the calf link frame.
    float      radius = 0.0f;
};

// A movable rigid box co-resident with the articulation. Its world AABB / Box prim
// ride the unified broadphase/narrowphase alongside the foot sphere.
struct CoResidentBox {
    float      half_extent = 0.05f;
    uint32_t   broadphase_body_id = 9000u;  // distinct from every go2 link body id.
};

// A STATIC ground plane the box rests on -- routed THROUGH the unified spine, NOT a
// hand-coded clamp. The box<->ground contact is detected (box AABB vs the plane),
// narrowphased (C3b analytical box x plane), emitted (EmitCompliantContactRows, box
// side RigidInvMass / ground side StaticNull), and solved (UnifiedSolve) in the SAME
// pipeline as the foot<->box pair. This is the support the grasp-hold scene needs: a
// movable box on a static ground, with the falling foot pressing DOWN onto it -> the
// contact force builds + HOLDS instead of both bodies free-falling together (two
// unsupported free-falling bodies separate -- physically correct, but not a "hold").
// The box reacts two-way to BOTH the foot above and the ground below; the ground is
// immovable (StaticNull: invM=0, no-op apply). The plane normal is world +Z.
struct CoResidentGround {
    float    height = 0.0f;          // plane z (the box bottom rests on this).
    uint32_t broadphase_id = 8000u;  // distinct from box + every go2 link body id.
};

// Per-step report (the gates read this).
struct CoResidentStepReport {
    bool     pair_found    = false;   // broadphase emitted the (foot, box) pair.
    uint32_t manifold_count = 0u;
    uint32_t row_count     = 0u;
    float    contact_depth = 0.0f;    // foot<->box penetration (peak point).
    float    lambda        = 0.0f;    // foot<->box converged normal impulse.
    double   qdot_delta_l1 = 0.0;     // |qdot_after - qdot_before|_1 (recoil proof).
    double   box_dv_norm   = 0.0;     // |box linear-velocity change| in the solve.
    // box<->ground (the NEW spine-routed support; replaces the hand clamp). These
    // let the gate prove the ground contact is NON-vacuous (carries load + bounded
    // penetration) instead of silently proving only the old foot<->box pair.
    bool     ground_pair_found = false;
    uint32_t ground_row_count  = 0u;
    float    ground_depth      = 0.0f;  // box<->ground penetration (peak corner).
    float    ground_lambda     = 0.0f;  // box<->ground converged normal impulse (max).
};

// The full co-resident state, mechanical-energy probe inputs included. KE/PE use a
// consistent reference: articulation KE = 0.5 sum_i v_i^T I_i v_i over the Pass-1
// link spatial velocities; PE = sum_i m_i * (-gravity_z) * z_i (link world COM).
struct CoResidentEnergy {
    double articulation_ke = 0.0;
    double articulation_pe = 0.0;
    double box_ke          = 0.0;
    double box_pe          = 0.0;
    double Total() const {
        return articulation_ke + articulation_pe + box_ke + box_pe;
    }
};

// The stepper. Owns the GPU articulation buffers + the host box; advances both in
// lockstep. The articulation is uploaded once at construction and stepped in
// place; Download() snapshots q/qdot/base_pose/link_velocity back.
class UnifiedCoResidentStepper {
public:
    UnifiedCoResidentStepper(const phi::DeviceContext& context,
                             const articulation::ArticulationHostState& host,
                             const scene::CookedShapeTable& cooked_shapes,
                             const CoResidentFoot& foot,
                             const CoResidentBox& box,
                             const CoResidentGround& ground,
                             const runtime::rigid::BodyState& box_state,
                             float gravity_z, float dt);

    // Advance ONE step: ComputeAccelerations -> velocity integrate (artic + box) ->
    // CONTACT PHASE (unified pipeline, two-way) -> position integrate (artic + box).
    CoResidentStepReport Step();

    // Current box state (host-resident, always live).
    const runtime::rigid::BodyState& Box() const { return box_state_; }

    // Snapshot the articulation host state (downloads q/qdot/base_pose/link_velocity).
    void Download(articulation::ArticulationHostState* out) const;

    // Total mechanical energy at the CURRENT state (consistent KE+PE reference).
    CoResidentEnergy Energy() const;

private:
    const phi::DeviceContext& context_;
    articulation::ArticulationHostState host_proto_;  // a refresh-able CPU mirror.
    scene::CookedShapeTable cooked_shapes_;
    articulation::ArticulationDeviceBuffers device_;
    CoResidentFoot foot_;
    CoResidentBox  box_;
    CoResidentGround ground_;
    runtime::rigid::BodyState box_state_;
    float gravity_z_;
    float dt_;
    uint32_t dof_stride_;
    uint32_t root_link_;

    // Advance the box pose by one symplectic position step (position += vel*dt +
    // orientation from angular vel). NO contact physics here -- the box<->ground
    // support flows through the unified spine in the contact phase, not a clamp.
    // Shared by the main step and every contact-free early-return path so the box
    // integrate is identical everywhere (D1).
    void IntegrateBoxPosition();
};

}  // namespace nuka::runtime::coresident
