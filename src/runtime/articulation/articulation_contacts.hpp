#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- world-pose FK + foot-vs-ground contacts
// ---------------------------------------------------------------------------
//
// p01-F T2. Two GPU passes that together produce the per-step, per-env contact
// data (which Go2 feet touch a static ground plane, where, how deep) that a
// later row-assembly task turns into constraint rows.
//
//  (A) UpdateWorldLinkPoses -- forward kinematics. The cooked link_pose in
//      ArticulationDeviceState is cook-time STATIC (rest pose); the Featherstone
//      ABA pass writes link_xup (spatial parent->child transforms) but never
//      world poses. This pass recomputes an up-to-date world math::Transform per
//      link from the current generalized coordinates q, walking links in local
//      order (the cooker guarantees parent-local-index < child, so one forward
//      pass suffices). The relative (parent->child) transform mirrors
//      featherstone_aba.cu's JointTransform in SE(3) form:
//        relative.position = link_local_pose.position + parent_offset
//                            (+ joint_axis * q for prismatic)
//        relative.rotation = link_local_pose.rotation * jointRotation(axis, q)
//      with world[child] = world[parent] o relative, root parent = identity.
//      Output is written to a SEPARATE buffer (not state.link_pose, which is
//      treated as the static rest pose elsewhere).
//
//  (B) DetectFootGroundContacts -- deterministic sphere-vs-plane. The Go2 feet
//      are spheres on the 4 calf links; the scene has no ground collider, so a
//      static half-space (z = ground_height, normal +Z) is supplied here. One
//      thread per env walks its (base-relative) feet in fixed order and compacts
//      penetrating feet into the env's own fixed-stride output slots. No atomics,
//      each env owns its slots -> D1-deterministic.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

namespace nuka::runtime::articulation {

// Max foot contacts emitted per environment (Go2 has 4 feet).
constexpr uint32_t kMaxFootContactsPerEnv = 4u;

// Per-foot static description, expressed in BASE (single-replica) link indices.
// The detection kernel turns calf_local_link into a per-replica global link via
//   global_link = env * base_link_count + calf_local_link
// so the same table drives every replicated environment.
struct FootShape {
    uint32_t calf_local_link = 0u;   // base-relative global link index of the calf
    math::Vec3 local_offset{};       // sphere center in the calf link frame
    float radius = 0.0f;             // sphere radius
};

// (A) Recompute world poses for every link from the current q. `out_world_pose`
// must be a device buffer of length state.total_link_count. One block per
// articulation, single forward pass over the links.
void UpdateWorldLinkPoses(const phi::DeviceContext& context,
                          ArticulationDeviceState state,
                          math::Transform* out_world_pose);

// ---------------------------------------------------------------------------
// p01-F T4 -- joint-space inertia M (CRBA) + factorization + contact m_eff
// ---------------------------------------------------------------------------
//
// Three GPU passes (one block / single lane per articulation, mirroring the
// Featherstone ABA kernels -- fixed loop order, no atomics => D1-deterministic,
// bit-identical across runs and replicas by construction).
//
//  (1) ComputeArticulationInertiaM -- the dense symmetric reduced-coordinate
//      joint-space inertia M (dof x dof) via the Composite Rigid Body Algorithm:
//      composite-rigid-body inertias accumulated leaf->root through link_xup
//      (reusing the ABA Pass-2 spatial algebra without the articulated-inertia
//      reduction), then for each non-fixed joint i, F = Ic_i S_i and
//        M[i][i] = S_i^T F,  M[i][j] = M[j][i] = S_j^T (X^T-propagated F)
//      for every non-fixed ancestor j. Output one row-major tile per
//      articulation, stride `max_dof * max_dof`.
//
//  (2) FactorArticulationInertiaM -- per-articulation unpivoted LDL^T of the
//      leading dof x dof block of M (M is SPD for positive masses), then form
//      the explicit inverse M^-1 (solve against identity columns). The inverse
//      tile lets the apply-step solve M y = J^T and form dqdot = M^-1 J^T lambda
//      directly. joint_armature[link] is folded into the diagonal (reflected
//      rotor inertia, matching ABA Pass-2) plus a tiny floor epsilon so a
//      near-singular leg config never produces NaN/Inf.
//
//  (3) ComputeContactEffectiveMass -- for each contact's dof_stride-wide chain
//      Jacobian J (from ComputeContactChainJacobians), m_eff = 1 / (J M^-1 J^T)
//      with the denominator clamped to a tiny epsilon (the straightened-leg
//      J ~ 0 config yields large-but-finite positive m_eff, never 0 or Inf).
//      Generic in J, so a later task reuses it for friction-tangent rows.
// ---------------------------------------------------------------------------

// DOF columns contributed by a joint type. Single source of truth shared with
// the chain Jacobian (Revolute=Prismatic=1, Fixed=0, future FloatingBase=6) so
// M's local dof indexing matches J's columns exactly.
uint32_t ArticulationJointDofCount(ArticulationJointType type);

// Per-articulation base-inclusive DOF count (prefix sum over the articulation's
// links). This is the M tile's leading block size and must equal the chain
// Jacobian's `dof_stride` for the same articulation.
uint32_t ArticulationDofCount(const ArticulationHostState& host, uint32_t articulation);

// Diagonal floor / regularization epsilon folded into M's diagonal alongside
// joint_armature. Tied to the ABA Pass-2 kMinDiagonal; negligible against
// mass-matrix entries (order kg.m^2) but keeps LDL^T finite at degenerate poses.
constexpr float kInertiaDiagonalEpsilon = 1.0e-6f;

// Denominator floor for the effective-mass reciprocal. At a straightened leg the
// vertical-force chain Jacobian collapses (J M^-1 J^T -> 0); clamping here yields
// a large-but-finite positive m_eff instead of an Inf.
constexpr float kEffectiveMassDenomEpsilon = 1.0e-9f;

// (1) Dense symmetric joint-space inertia M per articulation via CRBA. One block
// per articulation. Requires ABA Pass-1 to have run (link_xup,
// joint_motion_subspace current). `composite_inertia_scratch` is a device buffer
// of length state.total_link_count (LinkSpatialInertia, i.e. 36 floats each):
// the kernel seeds it from state.link_inertia and mutates it in place (so the
// ABA articulated-inertia buffer is left untouched). `out_inertia_M` is a device
// buffer of length state.articulation_count * max_dof * max_dof floats; each
// articulation's tile is row-major with stride max_dof*max_dof, zero-padded
// beyond its leading dof_count x dof_count block. `max_dof` must be >= every
// articulation's DOF count (and equal to the chain Jacobian's dof_stride).
void ComputeArticulationInertiaM(const phi::DeviceContext& context,
                                 ArticulationDeviceState state,
                                 uint32_t max_dof,
                                 LinkSpatialInertia* composite_inertia_scratch,
                                 float* out_inertia_M);

// (2) Per-articulation LDL^T factorization of M's leading block and explicit
// inverse. `inertia_M` is the buffer ComputeArticulationInertiaM filled (read
// only). `out_inertia_M_inv` is a device buffer of the same layout/length; each
// tile holds the symmetric M^-1 in its leading dof_count x dof_count block,
// zero-padded beyond. One block per articulation.
void FactorArticulationInertiaM(const phi::DeviceContext& context,
                                ArticulationDeviceState state,
                                uint32_t max_dof,
                                const float* inertia_M,
                                float* out_inertia_M_inv);

// (3) Per-contact effective mass m_eff = 1 / (J M^-1 J^T). `chain_jacobian` is
// the [contact_count * dof_stride] buffer from ComputeContactChainJacobians;
// `inertia_M_inv` is the per-articulation inverse from FactorArticulationInertiaM
// with the SAME stride (max_dof == dof_stride). One thread per contact; each
// contact selects its articulation's M^-1 tile via
// link_to_articulation[contact_link]. `out_effective_mass` is a device buffer of
// length contact_count. Invalid/empty contacts (link out of range) write 0.
void ComputeContactEffectiveMass(const phi::DeviceContext& context,
                                 ArticulationDeviceState state,
                                 const uint32_t* contact_link_indices,
                                 const float* chain_jacobian,
                                 const float* inertia_M_inv,
                                 uint32_t contact_count,
                                 uint32_t dof_stride,
                                 float* out_effective_mass);

// (B) Detect foot-vs-ground contacts. One thread per environment; each env
// inspects its `foot_count` feet (device buffer of FootShape, length foot_count,
// shared across all envs) against the half-space z = ground_height (normal +Z).
//
// Output layout (all device buffers, fixed stride kMaxFootContactsPerEnv per
// env, env-major):
//   out_contact_link   [env_count * kMaxFootContactsPerEnv] uint32_t -- global
//                       link index (the calf) of each emitted contact.
//   out_contact_point  [env_count * kMaxFootContactsPerEnv] Vec3 -- world contact
//                       point (foot center projected onto the ground surface).
//   out_contact_normal [env_count * kMaxFootContactsPerEnv] Vec3 -- world normal,
//                       always (0,0,1).
//   out_contact_depth  [env_count * kMaxFootContactsPerEnv] float -- penetration
//                       depth d = (ground_height + radius) - foot_world_z, > 0.
//   out_contact_count  [env_count] uint32_t -- number of emitted contacts in
//                       this env's slot block (0..kMaxFootContactsPerEnv).
// Unused trailing slots are cleared: link = ~0u (kInvalidLink, which downstream
// chain-Jacobian consumers skip), point/normal = 0, depth = 0. `foot_count` must
// be <= kMaxFootContactsPerEnv.
void DetectFootGroundContacts(const phi::DeviceContext& context,
                              const math::Transform* world_pose,
                              const FootShape* feet,
                              uint32_t foot_count,
                              uint32_t env_count,
                              uint32_t base_link_count,
                              float ground_height,
                              uint32_t* out_contact_link,
                              math::Vec3* out_contact_point,
                              math::Vec3* out_contact_normal,
                              float* out_contact_depth,
                              uint32_t* out_contact_count);

} // namespace nuka::runtime::articulation
