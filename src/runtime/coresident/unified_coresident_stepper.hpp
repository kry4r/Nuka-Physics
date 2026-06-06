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

    // ----- C7b-1 grasp-spike fields (only populated in grasp mode) ---------------
    // The grasp gate proves the cup is HELD by FINGER FRICTION ALONE (no table). The
    // discriminating number is the VERTICAL contact impulse the fingers deliver to
    // the cup, which at a steady hold must balance the cup's weight kick m*g*dt.
    uint32_t finger_contacts = 0u;   // # cup<->fingertip manifold points this step.
    // The cup-side vertical contact impulse summed over EVERY finger contact row
    // (normal rows + friction spokes), projected on +Z. This is the upward impulse
    // the fingers deliver to the cup; balances the cup weight kick when held.
    double   cup_vertical_impulse = 0.0;
    // Cross-check (must agree with cup_vertical_impulse): the cup's vertical-velocity
    // change ACROSS the contact solve * cup mass = the net vertical contact impulse.
    double   cup_dvz_impulse = 0.0;  // m_cup * (vz_after - vz_before_contact).
    double   cup_vz          = 0.0;  // cup vertical velocity AFTER this step.
    double   cup_z           = 0.0;  // cup height AFTER this step.
    bool     any_static_row  = false;  // a row carried a StaticNull side (a table!).
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

// ===========================================================================
// C7b-1 GRASP SPIKE: the generalized contact set (the W1a additive generalization).
// ===========================================================================
// W1a's stepper held ONE foot-sphere contact + ONE box on a static ground, passive
// (tau=0, condim=1). The grasp spike GENERALIZES the same substrate to:
//   * N fingertip contacts (a std::vector of {articulation link, local offset,
//     radius}) instead of one foot,
//   * a movable rigid CUP whose collision shape is a CONVEX HULL (the C7a cup hull)
//     instead of a box,
//   * a CONSTANT GRIP TORQUE driven through the articulation drive path
//     (LaunchApplyTorqueDriveKernels -> tau -> ABA) instead of a passive tau=0,
//   * FRICTION (condim=3: a normal row + a 4-edge linearized pyramid per contact)
//     instead of frictionless condim=1 -- friction is what HOLDS the cup,
//   * NO TABLE / NO STATIC GROUND -- the cup is held against gravity by finger
//     friction ALONE (the anti-green-wash discriminator).
// The foot+box+ground case is just one instantiation of this generic substrate;
// the W1a constructor below is preserved BYTE-FOR-BYTE.

// One fingertip contact: a sphere collidable on an articulation link.
struct CoResidentFingertip {
    uint32_t   link = ~0u;          // the ArticulationLink handle (broadphase id).
    math::Vec3 local_offset{};      // sphere center in the link frame.
    float      radius = 0.0f;
    uint32_t   broadphase_handle = ~0u;  // same as `link` (the cross-pair handle).
};

// A movable rigid CUP whose collision shape is a CONVEX HULL (the C7a cup). The
// hull vertices are MESH-LOCAL (cook-time constant); each step the stepper rebuilds
// a world-space ConvexHullView from these + the cup's live BodyState pose.
struct CoResidentCup {
    std::vector<float> hull_verts;          // flat x,y,z triples, MESH-LOCAL, constant.
    uint32_t           broadphase_body_id = 7000u;  // distinct from every link id.
    uint32_t           VertexCount() const {
        return static_cast<uint32_t>(hull_verts.size() / 3u);
    }
};

// The grasp-spike configuration. NO ground (the cup is held by friction alone).
struct GraspConfig {
    std::vector<CoResidentFingertip> fingertips;  // the N pinch contacts.
    CoResidentCup                    cup;
    runtime::rigid::BodyState        cup_state;    // mass/inertia/initial pose.
    // The constant grip torque, PER DEVICE LINK (length == total link count). Each
    // step it is re-applied to tau via LaunchApplyTorqueDriveKernels BEFORE ABA, so
    // the fingers squeeze inward every step (idempotent: ComputeAccelerations reads
    // tau, never zeroes it). drive_force_limits is the optional per-link clamp.
    std::vector<float> grip_torque;          // per device link (0 on non-finger links).
    std::vector<float> drive_force_limits;   // per device link (0 -> no clamp).
    float              friction_mu = 0.6f;   // per-contact isotropic friction coeff.
    uint32_t           condim      = 3u;     // 3 -> normal + 4 friction spokes.
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

    // C7b-1 GRASP constructor: N fingertips + a convex-hull cup + a constant grip
    // torque + friction, NO ground/table. Advances the gripper articulation + the
    // movable cup in lockstep through the unified spine, holding the cup against
    // gravity by finger friction alone. Reuses the SAME Step() machinery; the foot/
    // box/ground members stay inert (the W1a path never runs in grasp mode).
    UnifiedCoResidentStepper(const phi::DeviceContext& context,
                             const articulation::ArticulationHostState& host,
                             const GraspConfig& grasp,
                             float gravity_z, float dt);

    // Advance ONE step: ComputeAccelerations -> velocity integrate (artic + box) ->
    // CONTACT PHASE (unified pipeline, two-way) -> position integrate (artic + box).
    CoResidentStepReport Step();

    // Current box / cup state (host-resident, always live). Box() is the W1a name;
    // Cup() is the grasp alias -- both return the same movable rigid body.
    const runtime::rigid::BodyState& Box() const { return box_state_; }
    const runtime::rigid::BodyState& Cup() const { return box_state_; }

    // C7b-2a DISTURBANCE hook (additive). Inject a velocity impulse on the movable
    // body's linear (dv) + angular (dw) velocity, applied BEFORE the next Step()'s
    // contact phase. Used by the H1 grasp spike's disturbance gate to perturb the
    // held cup (lateral nudge, angular tilt, brief upward kick) and prove it stays
    // caged. Purely additive: the W1a/foot-box + C7b-1 grasp gates never call it.
    void ApplyCupImpulse(const math::Vec3& dv, const math::Vec3& dw) {
        box_state_.linear_velocity += dv;
        box_state_.angular_velocity += dw;
    }

    // C7b-2b PD-CONTROL hook (additive). Overwrite the per-device-link grip torque
    // that the NEXT StepGrasp() re-applies to tau (the DRIVE path). The W1a foot+box
    // path + the C7b-1/C7b-2a constant-torque grasp gates NEVER call this, so they
    // are byte-for-byte unchanged (the constructor still uploads grasp.grip_torque
    // once; a caller that never invokes the setter sees the exact prior behavior).
    // This lets a test compute a PD law tau_i = Kp*(q_target_i - q_i) - Kd*qdot_i
    // host-side each step from the Download'd q/qdot and feed it through the SAME
    // device drive seam. `torque` is per device link (length == total link count);
    // it is copied straight into grip_torque_dev_ (device memory the kernel reads).
    void SetGripTorque(const std::vector<float>& torque);

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
    uint32_t base_dof_ = 0u;       // root DOF count (FloatingBase=6, Fixed=0).

    // ----- C7b-1 grasp mode -----------------------------------------------------
    bool         grasp_mode_ = false;  // false -> W1a foot+box+ground path.
    GraspConfig  grasp_;               // the grasp config (fingertips/cup/grip/...).
    phi::Buffer  grip_torque_dev_;     // per-link constant grip torque (device).
    phi::Buffer  grip_limits_dev_;     // per-link force limits (device).

    // Run ONE grasp step: re-apply the constant grip torque -> ABA -> velocity
    // integrate (artic + cup) -> N fingertip<->cup contacts through the spine (NO
    // table) -> position integrate. The cup is held by finger friction alone.
    CoResidentStepReport StepGrasp();

    // Advance the box pose by one symplectic position step (position += vel*dt +
    // orientation from angular vel). NO contact physics here -- the box<->ground
    // support flows through the unified spine in the contact phase, not a clamp.
    // Shared by the main step and every contact-free early-return path so the box
    // integrate is identical everywhere (D1).
    void IntegrateBoxPosition();

    // The flat-qdot prefix-sum DOF index of a device link (Σ JointDofCount over the
    // articulation's links before it). Generic over fixed/floating roots; for go2
    // (FloatingBase root) it reproduces base@[0..5], leg k@[6+k] BYTE-FOR-BYTE.
    uint32_t DofIndexOf(uint32_t link) const;
};

}  // namespace nuka::runtime::coresident
