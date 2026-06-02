#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor -- per-link contact wrench + per-slot contact force readout
// ---------------------------------------------------------------------------
//
// p14a (v0.7) any-link contact-force readout with MuJoCo mj_contactForce parity.
//
// This is a pure READBACK of data the batched articulated step already produced.
// It does NOT add physics: the per-contact impulses (PGS lambda, N*s) are solved
// at step stage 10, and the per-contact world point/normal/owning-link are set at
// step stages 4/5. This free function turns those into the two query surfaces:
//
//   (A) per-LINK net contact wrench (force + torque) in the WORLD frame, and
//   (B) per-SLOT contact force (the {Fn,Ft1,Ft2} = {lambda_n,lambda_t1,lambda_t2}
//       / dt contact-basis components),
//
// in one launch, at the END of each Step()/StepKernels().
//
// Impulse -> force: PGS lambda is an IMPULSE (N*s), so force = lambda / dt. The
// per-slot world force vector is F_slot = (n*lambda_n + t1*lambda_t1 +
// t2*lambda_t2)/dt, where n / t1 / t2 are the world contact normal and the two
// world friction tangents the solver built its chain Jacobians from (so lambda_n
// / lambda_t1 / lambda_t2 are signed magnitudes along those world directions --
// see SolveArticulatedContactRows: it applies qdot += M^-1 J^T lambda with J built
// from n / t1 / t2).
//
// (A) Net per-link wrench. For each link g, summed over the contact slots tagged
// to g (a slot is tagged via contact_link[slot] == g):
//     F[g]   = sum_slots F_slot
//     tau[g] = sum_slots (contact_point[slot] - link_origin_world[g]) x F_slot
// TORQUE REFERENCE POINT: the LINK FRAME ORIGIN (world_pose[g].position), NOT the
// link COM. Documented here and in nuka.h (NUKA_FIELD_LINK_CONTACT_WRENCH).
//
// DETERMINISM (D1): the wrench kernel runs ONE THREAD PER OUTPUT LINK g. Each
// thread loops that env's kMaxFootContactsPerEnv slots in FIXED slot order and
// accumulates in fp32 (no atomics, fixed order) -> byte-exact across runs and
// replicas. Inactive slots carry lambda == 0 (the solver leaves them zero) so
// they contribute nothing; the gate is contact_link[slot] == g, which an inactive
// slot (link == ~0u) can never satisfy.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"

#include <cstdint>

namespace nuka::sensor {

// Computes BOTH outputs in one launch (eager, every step):
//   out_contact_force[slot_count * 3]  -- {lambda_n,lambda_t1,lambda_t2}/dt per
//       slot (contact-basis components: normal force then the two friction-force
//       magnitudes). Cheap elementwise scale of `lambda`.
//   out_link_wrench[total_link_count * 6] -- per GLOBAL link g: [g*6+0..2]=world
//       force F, [g*6+3..5]=world torque tau about the link frame origin.
//
// Inputs (all device pointers; raw so this TU depends only on nuka_core/nuka_phi
// + math, never on the runtime/articulation headers):
//   lambda          [slot_count * 3] float -- solved per-slot impulses
//                   {lambda_n,lambda_t1,lambda_t2}, slot-major (the
//                   BatchedArticulatedWorld lambda_ buffer / DownloadLambda layout).
//   contact_point   [slot_count] Vec3 -- world contact point per slot.
//   contact_normal  [slot_count] Vec3 -- world unit normal per slot.
//   tangent1        [slot_count] Vec3 -- world friction tangent 1 per slot.
//   tangent2        [slot_count] Vec3 -- world friction tangent 2 per slot.
//   contact_link    [slot_count] uint32 -- GLOBAL link index per slot (~0u inactive).
//   link_world_pose [total_link_count] Transform -- per-link world pose; the link
//                   frame ORIGIN is .position (the torque reference point).
// Counts: env_count, base_link_count (slot_count == env_count *
//   kMaxFootContactsPerEnv == env_count * max_foot_contacts_per_env;
//   total_link_count == env_count * base_link_count). dt is the step dt
//   (force = impulse / dt); dt > 0 required (a non-positive dt yields zero forces).
void ComputeContactWrench(const phi::DeviceContext& context,
                          const float* lambda,
                          const math::Vec3* contact_point,
                          const math::Vec3* contact_normal,
                          const math::Vec3* tangent1,
                          const math::Vec3* tangent2,
                          const uint32_t* contact_link,
                          const math::Transform* link_world_pose,
                          uint32_t env_count,
                          uint32_t base_link_count,
                          uint32_t max_foot_contacts_per_env,
                          float dt,
                          float* out_contact_force,
                          float* out_link_wrench);

}  // namespace nuka::sensor
