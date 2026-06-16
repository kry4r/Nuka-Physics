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
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>
#include <limits>

namespace nuka::runtime::articulation {

// Max foot contacts emitted per environment (Go2 has 4 feet).
constexpr uint32_t kMaxFootContactsPerEnv = 4u;

// Task 1/2: the per-ARTICULATION contact-slot stride for a MULTI-DOG (K>1) world.
// Each co-resident dog needs room in its OWN fused slot block for its 4 feet PLUS
// body-vs-terrain contacts PLUS dog-dog rows. 12 covers the worst realistic case
// (4 feet + up to ~6 body links + a couple dog-dog) with margin and stays within
// the FUSED solver's compile-time register ceiling (kMaxContactsPerArtic). At K<=1
// the stride collapses to kMaxFootContactsPerEnv (4) so K==1 stays byte-identical.
constexpr uint32_t kMultiDogContactsPerArtic = 12u;

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
void UpdateWorldLinkPoses(cudaStream_t stream, int device_id,
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
// `joint_damping` (optional, per-DOF c_j indexed by global link) + `dt` fold an
// implicit joint-damping term into M: when non-null the joint diagonals get
// += dt*c_j, so a subsequent FactorArticulationInertiaM yields (M + dt*C)^-1, the
// backward-Euler admittance used by the constrained-velocity solve to apply joint
// damping unconditionally stably. null/0 -> pure CRBA M (unit tests / single-env
// path unchanged). The free floating-base DOFs are never damped (kernel skips the
// floating root), matching c_j=0 on the base in the solve.
void ComputeArticulationInertiaM(cudaStream_t stream, int device_id,
                                 ArticulationDeviceState state,
                                 uint32_t max_dof,
                                 LinkSpatialInertia* composite_inertia_scratch,
                                 float* out_inertia_M,
                                 const float* joint_damping = nullptr,
                                 float dt = 0.0f);

// (2) Per-articulation LDL^T factorization of M's leading block and explicit
// inverse. `inertia_M` is the buffer ComputeArticulationInertiaM filled (read
// only). `out_inertia_M_inv` is a device buffer of the same layout/length; each
// tile holds the symmetric M^-1 in its leading dof_count x dof_count block,
// zero-padded beyond. One block per articulation. `max_dof` must be <=
// kMaxArticulationDof (the factor scratch ceiling) AND equal every
// articulation's exact DOF count; exceeding the ceiling throws LOUDLY here
// (G0: never a silent clamp -- the old 18-cap silently welded DOFs past it).
void FactorArticulationInertiaM(cudaStream_t stream, int device_id,
                                ArticulationDeviceState state,
                                uint32_t max_dof,
                                const float* inertia_M,
                                float* out_inertia_M_inv);

// General standalone implicit joint damping (no contacts). The damping-only
// sibling of SolveArticulatedContactRows' implicit-damping seed: applies the
// SAME backward-Euler joint damping qdot -= dt*(M+dt*C)^-1*(C*qdot) in place,
// where `inertia_M_inv` is the (M+dt*C)^-1 tile from FactorArticulationInertiaM
// (fed ComputeArticulationInertiaM the SAME joint_damping + dt). `joint_damping`
// is the per-DOF c_j indexed by global link (the deferred drive damping). Base
// (free floating-base) DOFs are never damped (c=0). Used by the single-env
// oracle path to run the same implicit damping scheme as the batched contacts
// path WITHOUT any contact rows. `dof_stride` == max_dof. One block per
// articulation, single lane, fixed loop order, no atomics => D1-deterministic;
// bit-for-bit identical to the batched contacts-OFF solve. joint_damping==nullptr
// || dt<=0 -> qdot unchanged.
void ApplyImplicitJointDamping(cudaStream_t stream, int device_id,
                               ArticulationDeviceState state,
                               const float* inertia_M_inv,
                               const float* joint_damping,
                               uint32_t dof_stride,
                               float dt);

// (3) Per-contact effective mass m_eff = 1 / (J M^-1 J^T). `chain_jacobian` is
// the [contact_count * dof_stride] buffer from ComputeContactChainJacobians;
// `inertia_M_inv` is the per-articulation inverse from FactorArticulationInertiaM
// with the SAME stride (max_dof == dof_stride). One thread per contact; each
// contact selects its articulation's M^-1 tile via
// link_to_articulation[contact_link]. `out_effective_mass` is a device buffer of
// length contact_count. Invalid/empty contacts (link out of range) write 0.
void ComputeContactEffectiveMass(cudaStream_t stream, int device_id,
                                 ArticulationDeviceState state,
                                 const uint32_t* contact_link_indices,
                                 const float* chain_jacobian,
                                 const float* inertia_M_inv,
                                 uint32_t contact_count,
                                 uint32_t dof_stride,
                                 float* out_effective_mass);

// ---------------------------------------------------------------------------
// p01-F T5 -- articulated contact rows + block-PGS solve (contact<->ABA coupling)
// ---------------------------------------------------------------------------
//
// The genuinely new coupling: turn the detected foot contacts (T2) into
// constraint rows that carry their chain Jacobian (T3) and effective mass (T4),
// then solve them per-articulation with a projected Gauss-Seidel (PGS) and apply
// the resulting joint-space impulse Delta_qdot = M^-1 J^T lambda straight back
// into the joint velocities. The solve+apply are FUSED (one kernel), one block
// per articulation, single lane, fixed iteration count, fixed row order, no
// atomics => D1-strong-deterministic (bit-identical across runs and replicas).
//
//  AssembleArticulatedContactRows -- gathers, per ACTIVE detected contact
//    (link != kInvalidLink), its dof-wide chain-Jacobian row, m_eff, world
//    normal, penetration depth, and owning articulation. Builds a per-contact
//    orthonormal tangent basis (t1,t2) from the normal (branch-stable, so the
//    same normal always yields the same basis => friction stays deterministic),
//    and emits ONE normal row plus TWO friction tangent rows per active contact.
//    Output is a fixed-stride env-major ArticulatedContactRow buffer (length
//    env_count * kMaxFootContactsPerEnv), trailing/inactive slots marked
//    row_type == kRowInactive so the solver skips them.
//
//    Scope deferral: a kRowJointLimit row TYPE is scaffolded in the enum, but NO
//    limit rows are emitted -- ArticulationDeviceState carries no joint lower/
//    upper limit fields, so there is no data to drive them. T6/a later task adds
//    limit data + emission.
//
//  SolveArticulatedContactRowsKernel -- block-per-articulation fused PGS.
//    Seeds a working joint-velocity vector qdot_work from state.qdot (which at
//    call time holds qdot_free = the post-velocity-integration joint velocity),
//    runs kContactSolverIterations sweeps over the rows in fixed order (all
//    normal rows of the articulation's contacts, then the friction rows), and
//    for each row i:
//        jv    = dot(J_i, qdot_work)          (over the articulation's dof cols)
//        delta = m_eff_i * (b_i - jv)         (row_solver.cu:289 convention:
//                                              lambda = m_eff*(rhs - jv))
//        new   = clamp(lambda_i + delta, lower, upper)
//        qdot_work += (M^-1 tile) . (J_i^T * (new - lambda_i)); lambda_i = new
//    NORMAL row:  b_i = kBaumgarteBeta/dt * max(depth_i - kPenetrationSlop, 0)
//                 (positional bias; restitution = 0), lower=0, upper=+inf.
//    FRICTION row: b_i = 0, lower=-mu*sum_lambda_n, upper=+mu*sum_lambda_n with
//                 mu = kContactFriction and sum_lambda_n the contact's
//                 accumulated normal impulse (Coulomb cone, mirroring
//                 row_solver.cu:293-301). Finally writes qdot_work back into
//                 state.qdot for every DOF of the articulation. The dof column ->
//                 global qdot index map is the SAME base-inclusive prefix-sum
//                 convention as T3's dof_index / T4's LocalDofIndex, so J_i,
//                 M^-1 and the joint velocities all share one column ordering.
//
//  Warm-start: SolveArticulatedContactRowsKernel can be seeded from a persisted
//    per-slot lambda buffer (keyed by the fixed contact slot
//    [env*kMaxFootContactsPerEnv + slot] and row type) to accelerate convergence
//    across steps. Warm-start affects CONVERGENCE ONLY -- never determinism or
//    correctness. A cold start (all-zero lambda, the default when warm_start is
//    null) is fully acceptable and is NOT a defect; the determinism gate runs
//    cold so two runs are trivially comparable.
// ---------------------------------------------------------------------------

// PGS sweep count. The normal and friction rows of a bent Go2 leg are strongly
// coupled (Jn.M^-1.Jt^T is O(1)), which slows Gauss-Seidel; a sticking contact
// needs ~40 sweeps to drive the tangential foot velocity to ~0. 48 clears that
// with margin (resting-contact stick residual ~3e-3 m/s) while staying cheap at
// one block per articulation; fixed count + fixed row order keeps the solve D1.
constexpr uint32_t kContactSolverIterations = 48u;

// Baumgarte positional-stabilization coefficient (fraction of penetration error
// fed back as a target separating velocity per step). 0.2 is the standard mild
// value -- large enough to arrest penetration growth, small enough not to inject
// excessive energy.
constexpr float kBaumgarteBeta = 0.2f;

// Penetration allowance (m): depth below this is not corrected, so a resting
// contact does not jitter about exactly zero penetration.
constexpr float kPenetrationSlop = 1.0e-4f;

// Coulomb friction coefficient applied at every foot contact (mu). 0.8 is a
// typical rubber-on-hard-ground value and matches the Go2 foot material intent.
constexpr float kContactFriction = 0.8f;

// Constraint-row type tag. kRowJointLimit is scaffolded but never emitted (no
// joint-limit data exists in ArticulationDeviceState yet -- see the doc above).
enum class ContactRowType : uint32_t {
    kRowInactive = 0u,    // empty / trailing slot; solver skips it.
    kRowNormal = 1u,      // contact normal (non-penetration) row.
    kRowFriction1 = 2u,   // first friction tangent row.
    kRowFriction2 = 3u,   // second friction tangent row.
    kRowJointLimit = 4u,  // SCAFFOLD ONLY -- never emitted.
};

// One assembled contact's full row set, stored env-major at fixed stride
// kMaxFootContactsPerEnv (slot == the T2 contact slot within the env). The
// normal + two friction rows share the contact's geometry; the solver reads the
// dof-wide jacobian/normal/tangent slices through the contact's owning
// articulation tile. Padding columns of the jacobians are zero.
struct ArticulatedContactRow {
    ContactRowType row_type = ContactRowType::kRowInactive;
    uint32_t articulation = ~0u;   // owning global articulation index.
    uint32_t contact_link = ~0u;   // global link index of the contact.
    float effective_mass = 0.0f;   // normal-row m_eff (= 1/(Jn M^-1 Jn^T)).
    float effective_mass_t1 = 0.0f;  // friction tangent-1 m_eff.
    float effective_mass_t2 = 0.0f;  // friction tangent-2 m_eff.
    float depth = 0.0f;            // penetration depth (>0 active).
    math::Vec3 normal{};           // world contact normal (unit).
    math::Vec3 tangent1{};         // world friction tangent 1 (unit, n-perp).
    math::Vec3 tangent2{};         // world friction tangent 2 (= n x t1).
};

// Engine-wide articulation DOF ceiling for the contact-solve spine's working
// storage: the M^-1 factorization scratch, the on-thread PGS working vectors
// (dof_to_link / qdot_work), and the implicit-joint-damping working vectors.
// 64 = clean headroom over the ~51-DOF whole-body H1 (6-DOF floating base +
// ~45 scalar joints) whose FINGER-contact chain DOF indices exceed the old 18
// cap. G0 honesty contract: exceeding this ceiling is a LOUD host-side failure
// (the launchers throw; the coresident worlds reject at construction) -- NEVER
// a silent clamp. Raising the array sizes does not change any float op for
// articulations within the old cap, so all <=18-DOF results stay byte-identical.
constexpr uint32_t kMaxArticulationDof = 64u;

// LEGACY 18-DOF cap (6-DOF floating base + 12 revolute = the Go2). The contact
// solve itself now sizes by kMaxArticulationDof above; this constant remains
// ONLY for its named consumers whose scratch is genuinely pinned at 18:
//   * the ComputedTorque/Osc drive kernels' dense scratch (articulation_drives.cu)
//     -- which must stay in lockstep with their diffsim adjoint,
//   * the diffsim KKT/IFT/Osc-adjoint __shared__ tiles (kkt_builder.cu /
//     ift_runner.cu / osc_adjoint.cu), where 64-wide double tiles would not fit
//     the 48KB static shared-memory budget.
// All of those consumers fail LOUDLY (throw / construction error) beyond it.
constexpr uint32_t kMaxContactSolverDof = 18u;

// Assemble the per-active-contact row set from the T2/T3/T4 outputs. Inputs are
// the detection buffers (env-major, fixed stride kMaxFootContactsPerEnv) plus the
// chain Jacobians (one dof_stride-wide normal row per contact slot from
// ComputeContactChainJacobians fed the contact NORMAL) and the per-contact normal
// m_eff (from ComputeContactEffectiveMass). `out_rows` is a device buffer of
// length env_count * kMaxFootContactsPerEnv. `out_tangent_jacobian` is a device
// buffer of length env_count * kMaxFootContactsPerEnv * 2 * dof_stride: the two
// friction-tangent chain-Jacobian rows per slot, which the caller fills by
// re-running ComputeContactChainJacobians with the tangent directions as the
// "normal" (so the assembled tangent rows reuse the proven J builder). `dof_stride`
// must equal max_dof. One thread per contact slot. No atomics.
void AssembleArticulatedContactRows(cudaStream_t stream, int device_id,
                                    ArticulationDeviceState state,
                                    const uint32_t* contact_link,
                                    const math::Vec3* contact_normal,
                                    const float* contact_depth,
                                    const float* normal_effective_mass,
                                    const float* tangent1_effective_mass,
                                    const float* tangent2_effective_mass,
                                    uint32_t env_count,
                                    uint32_t dof_stride,
                                    ArticulatedContactRow* out_rows);

// Builds the per-contact friction tangent directions (t1,t2) from each active
// contact's world normal, written env-major at fixed stride
// kMaxFootContactsPerEnv. Inactive/empty slots get zero vectors. Deterministic
// branch-stable basis (fixed reference axis). The caller feeds out_tangent1 /
// out_tangent2 to ComputeContactChainJacobians as the contact "normal" to obtain
// the tangent chain-Jacobian rows. One thread per contact slot.
void ComputeContactTangentBasis(cudaStream_t stream, int device_id,
                                const uint32_t* contact_link,
                                const math::Vec3* contact_normal,
                                uint32_t env_count,
                                math::Vec3* out_tangent1,
                                math::Vec3* out_tangent2);

// Fused block-per-articulation PGS solve + joint-space impulse apply. Reads the
// assembled rows and the per-contact normal / tangent chain Jacobians (each a
// [env_count*kMaxFootContactsPerEnv*dof_stride] buffer, row-major per slot) plus
// the per-articulation M^-1 tiles (stride max_dof*max_dof, dof_stride == max_dof).
// Modifies state.qdot IN PLACE: on entry qdot holds qdot_free, on exit qdot holds
// the post-contact joint velocity. `dt` is the integration step (drives the
// Baumgarte bias). `inout_lambda` (optional, may be null) is the warm-start /
// persisted impulse buffer, length env_count*kMaxFootContactsPerEnv*3 (normal,
// tangent1, tangent2 per slot); when null the solve cold-starts at zero. On
// return it holds this step's converged impulses for the next step's warm start.
// `friction_coefficient` (default kContactFriction) is the Coulomb mu used by the
// friction-cone clamp; passing 0 makes the friction rows inert (a frictionless
// baseline) and a material-specific value will drive per-contact friction in T6.
// It is a uniform scalar, so threading it preserves D1 determinism.
// `baumgarte_max_velocity` (default +inf == non-binding, so committed T5 calls
// are unaffected bit-for-bit) caps the normal-row positional bias at
// min(kBaumgarteBeta/dt*max(depth-slop,0), baumgarte_max_velocity); a finite
// tuned value prevents large-penetration one-step velocity transients. It is a
// uniform scalar applied identically at every normal-row bias site, so it
// preserves D1 determinism.
// One block per articulation, single lane, fixed order, no atomics => D1.
// `joint_damping` (optional, per-DOF c_j indexed by global link): when non-null
// AND `inertia_M_inv` was factored from (M + dt*C), the solve first applies implicit
// joint damping (qdot -= dt*(M+dt*C)^-1 * (C*qdot)) before the contact sweeps, so
// joint damping and contacts share one consistent (M+dt*C)^-1 admittance. null ->
// no implicit damping (prior behaviour, bit-for-bit). It is a per-DOF coefficient
// applied in fixed order with no atomics, so it preserves D1 determinism.
void SolveArticulatedContactRows(cudaStream_t stream, int device_id,
                                 ArticulationDeviceState state,
                                 const ArticulatedContactRow* rows,
                                 const float* normal_jacobian,
                                 const float* tangent1_jacobian,
                                 const float* tangent2_jacobian,
                                 const float* inertia_M_inv,
                                 uint32_t env_count,
                                 uint32_t dof_stride,
                                 float dt,
                                 float* inout_lambda,
                                 float friction_coefficient = kContactFriction,
                                 float baumgarte_max_velocity =
                                     std::numeric_limits<float>::infinity(),
                                 const float* joint_damping = nullptr);

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
void DetectFootGroundContacts(cudaStream_t stream, int device_id,
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
