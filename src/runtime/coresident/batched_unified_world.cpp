// ---------------------------------------------------------------------------
// nuka::runtime::coresident::BatchedUnifiedWorld -- implementation (v0.8 P2.2).
// ---------------------------------------------------------------------------
// P2.1 scope: per-env MOVABLE RIGID BODIES under gravity (free-fall, no contact,
// no articulation). The step loop is the SAME stage order as
// UnifiedCoResidentStepper / BatchedArticulatedWorld -- velocity stage (gravity
// kick) -> contact phase (EMPTY in P2.1) -> position stage.
//
// P2.2 (THIS): the batched box<->static-ground contact phase fills the previously
// EMPTY slot. Each env's local body 0 (a BOX) rests on ONE static +Z plane. For
// every env, the production narrowphase (BuildContactManifolds, box x plane) emits
// a manifold; EmitCompliantContactRows (condim=1) APPENDS condim=1 normal rows to a
// SHARED rows/sides buffer; the appended rows' body indices are OVERWRITTEN (rigid
// side -> BodyIndex(e,0), static side -> kInvalidBodyIndex). After all envs, ONE
// UnifiedSolve over the concatenated rows + the full env-major bodies advances every
// env at once. This is PURE PLUMBING: the PrimParams, the cfg{64,0,0,0}, the
// condim=1 inputs, and the body-index wiring are copied verbatim from
// UnifiedCoResidentStepper::Step()'s box<->ground branch (lines ~437-614), so an N=1
// BatchedUnifiedWorld is BYTE-IDENTICAL to that single-instance reference. NO new
// physics, NO solver change. The cross-env body-id disjointness (BodyIndex(e,*)) is
// exactly what makes the row-scheduler's greedy-in-index graph coloring partition
// the per-env rows into N independent solves (see row_scheduler.cu RowsConflict).
// ---------------------------------------------------------------------------

#include "runtime/coresident/batched_unified_world.hpp"

#include "collision/analytical_manifold.hpp"   // amf::PrimParams / BuildPrimFrame
#include "collision/candidate_pair.hpp"        // CandidatePair / CollidableRef
#include "collision/contact_stream_driver.hpp" // BuildContactManifolds / ResolvedShape
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "constraint/contact_row_sides.hpp"    // ContactRowSides
#include "constraint/reaction_provider.hpp"    // ReactionProviderKind
#include "constraint/row.hpp"                  // kInvalidBodyIndex
#include "constraint/row_buffers.hpp"          // RowBuffers
#include "constraint/row_builder.hpp"          // EmitCompliantContactRows / inputs
#include "scene/canonical_types.hpp"           // scene::ShapeType
#include "solver/rigid_solver.hpp"             // SolverConfig
#include "solver/unified_solve.hpp"            // UnifiedSolve / SolveContext

#include <cstddef>
#include <vector>

namespace nuka::runtime::coresident {

namespace {

namespace amf = nuka::collision::amf;
using nuka::collision::CandidatePair;
using nuka::collision::ResolvedShape;
using nuka::collision::ShapeResolver;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ContactRowSides;
using nuka::constraint::ReactionProviderKind;
using nuka::constraint::RowBuffers;
using nuka::math::Transform;
using nuka::math::Vec3;

// REPLICATED from UnifiedCoResidentStepper's anon namespace (file-static there), so
// this TU does NOT depend on the protected stepper. Byte-for-byte the same prims.

// A box PrimParams from the body pose + per-axis half-extents (the cup's flat-bottom
// table-contact proxy -- the stable C3b BoxPlane handler the W1a box<->ground gate
// uses). Mirrors UnifiedCoResidentStepper::BoxPrimXYZ.
amf::PrimParams BoxPrim(const Transform& pose, const Vec3& half_extents) {
    amf::PrimParams p;
    p.half_extents = half_extents;
    p.frame = amf::BuildPrimFrame(pose);  // bakes the box orientation into cx/cy/cz.
    return p;
}

// A static ground PrimParams whose plane normal is world +Z (C3b BoxPlane reads the
// normal from frame.cy). Mirrors UnifiedCoResidentStepper::GroundPrim.
amf::PrimParams GroundPrim(float height) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = Vec3{0.0f, 0.0f, 1.0f};   // plane normal = world +Z
    p.frame.cz = Vec3{0.0f, -1.0f, 0.0f};  // right-handed: cx x cy = cz
    p.frame.t = Vec3{0.0f, 0.0f, height};
    return p;
}

// The box collidable (RigidInvMass -- the movable rigid side). Mirrors MakeBoxRef.
CollidableRef MakeBoxRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::RigidBody;
    ref.react = ReactionProviderKind::RigidInvMass;
    ref.handle = handle;
    return ref;
}

// The static ground collidable (StaticWorld -> StaticNull reaction: invM=0, no-op
// apply). Mirrors MakeGroundRef.
CollidableRef MakeGroundRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = handle;
    return ref;
}

// Per-env distinct broadphase handles for the box + ground. These are ONLY used by
// the per-env resolver lambda (the row body indices, the thing the solve + coloring
// actually key on, are overwritten to BodyIndex(e,0) / kInvalidBodyIndex below).
constexpr uint32_t kBoxHandle = 9000u;     // distinct from the static ground handle.
constexpr uint32_t kGroundHandle = 8000u;  // distinct from the box handle.

}  // namespace

BatchedUnifiedWorld::BatchedUnifiedWorld(
    const phi::DeviceContext& context,
    const BatchedSceneTemplate& scene_template,
    uint32_t env_count,
    float gravity_z,
    float dt)
    : context_(context),
      env_count_(env_count),
      bodies_per_env_(
          static_cast<uint32_t>(scene_template.bodies_per_env.size())),
      gravity_z_(gravity_z),
      dt_(dt),
      has_ground_(scene_template.has_ground),
      box_half_extent_(scene_template.box_half_extent),
      ground_height_(scene_template.ground_height) {
    // Replicate the per-env body template into the env-major SoA: env e's bodies
    // occupy [e*k, (e+1)*k). This is the per-env body-id offset the batched
    // UnifiedSolve relies on (cross-env body ids are disjoint -> the colored solve
    // is N independent solves). Per-env initial-condition perturbation is applied
    // by the caller through BodyMut() after construction.
    bodies_.reserve(static_cast<size_t>(env_count_) * bodies_per_env_);
    for (uint32_t e = 0u; e < env_count_; ++e) {
        for (uint32_t i = 0u; i < bodies_per_env_; ++i) {
            bodies_.push_back(scene_template.bodies_per_env[i]);
        }
    }
}

void BatchedUnifiedWorld::IntegrateBodyPosition(
    runtime::rigid::BodyState& body) const {
    // BYTE-IDENTICAL to UnifiedCoResidentStepper::IntegrateBoxPosition: pure
    // symplectic-Euler kinematics (NO floor clamp; contact support flows through
    // the unified spine in the contact phase, which P2.2 adds).
    if (body.inv_mass <= 0.0f) return;
    body.position += body.linear_velocity * dt_;
    const math::Vec3 w = body.angular_velocity;
    math::Quat dq;
    dq.w = 1.0f;
    dq.x = 0.5f * w.x * dt_;
    dq.y = 0.5f * w.y * dt_;
    dq.z = 0.5f * w.z * dt_;
    body.orientation = (body.orientation * dq).Normalized();
}

void BatchedUnifiedWorld::Step() {
    // ----- velocity stage: gravity velocity-kick, env-major order (D1) ----------
    // Matches the co-resident box gravity kick (linear_velocity.z += g*dt) applied
    // BEFORE the contact phase. Immovable bodies (inv_mass<=0) are skipped, exactly
    // as IntegrateBoxPosition skips them.
    for (auto& body : bodies_) {
        if (body.inv_mass <= 0.0f) continue;
        body.linear_velocity.z += gravity_z_ * dt_;
    }

    // ----- contact phase ---------------------------------------------------------
    // P2.2: the batched box<->static-ground contact phase (a no-op when !has_ground_,
    // which preserves the P2.1 free-fall path byte-for-byte). Per-env narrowphase ->
    // concatenated compliant rows (per-env body-id offsets) -> ONE UnifiedSolve, which
    // mutates `bodies_` velocities in place BEFORE the position stage integrates them.
    ResolveBatchedGroundContact();

    // ----- position stage: symplectic-Euler integrate, env-major order (D1) ------
    for (auto& body : bodies_) {
        IntegrateBodyPosition(body);
    }
}

void BatchedUnifiedWorld::ResolveBatchedGroundContact() {
    // Inert unless a static ground is configured AND there is at least one body per
    // env to rest on it. Leaves the velocity untouched -> P2.1 free-fall preserved.
    if (!has_ground_ || bodies_per_env_ == 0u || env_count_ == 0u) return;

    // ONE shared rows/sides buffer across ALL envs (EmitCompliantContactRows APPENDS,
    // so repeated per-env calls concatenate). Per-env body-id offsets make the rows
    // from different envs share NO body id -> the greedy graph coloring partitions
    // them into N independent solves (row_scheduler.cu RowsConflict).
    RowBuffers rows;
    std::vector<ContactRowSides> sides;

    // condim=1 frictionless normal rows, refsafe -- COPIED from the oracle (lines
    // ~481-488). dt/vel/invweight identical so an N=1 env is byte-exact.
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = 1.0f;
    inputs.dt = dt_;
    inputs.condim = 1u;
    inputs.refsafe = true;

    // The static ground PrimParams is env-invariant (one shared world +Z plane).
    const amf::PrimParams ground_prim = GroundPrim(ground_height_);

    for (uint32_t e = 0u; e < env_count_; ++e) {
        // env e's movable rigid body is local body 0 (the P2.2 one-box-per-env scope).
        const runtime::rigid::BodyState& body = bodies_[BodyIndex(e, 0u)];

        // The box<->ground candidate pair, emitted DIRECTLY (the box AABB always
        // overlaps the +Z half-space if its bottom is at/below the plane -- trivial
        // broadphase). Side A = box (RigidInvMass), side B = ground (StaticNull).
        // Mirrors the oracle stage 5d.
        CandidatePair ground_pair;
        ground_pair.a = MakeBoxRef(kBoxHandle);
        ground_pair.b = MakeGroundRef(kGroundHandle);

        // Per-env resolver bound to THIS env's live body pose. box -> C3b BoxPlane.
        const Transform box_pose{body.position, body.orientation};
        const amf::PrimParams box_prim = BoxPrim(box_pose, box_half_extent_);
        ShapeResolver resolve = [&](const CollidableRef& ref,
                                    ResolvedShape* out) -> bool {
            if (ref.type == CollidableType::RigidBody && ref.handle == kBoxHandle) {
                out->type = scene::ShapeType::Box;
                out->prim = box_prim;
                return true;
            }
            if (ref.type == CollidableType::StaticWorld &&
                ref.handle == kGroundHandle) {
                out->type = scene::ShapeType::Plane;
                out->prim = ground_prim;
                return true;
            }
            return false;
        };

        // Per-env narrowphase (box x plane). manifolds is rebuilt fresh per env.
        std::vector<ContactManifold> manifolds;
        const CandidatePair pairs[1] = {ground_pair};
        nuka::collision::BuildContactManifolds(pairs, resolve, &manifolds);
        if (manifolds.empty()) continue;  // box clear of the plane -> no rows this env.

        // APPEND env e's compliant normal rows to the shared buffer.
        const std::size_t row_start = rows.RowCount();
        nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);

        // OVERWRITE the appended rows' body indices: rigid (RigidInvMass) side ->
        // env e's flat env-major body index BodyIndex(e,0); static (StaticNull) side
        // -> kInvalidBodyIndex (no reaction, no coloring conflict). Mirrors the oracle
        // box<->ground branch (lines ~577-588), but with the batched env-major index
        // instead of a hard 0.
        const uint32_t box_body_index = BodyIndex(e, 0u);
        for (std::size_t r = row_start; r < rows.RowCount(); ++r) {
            const ContactRowSides& s = sides[r];
            const bool a_rigid = s.a.react == ReactionProviderKind::RigidInvMass;
            const bool b_static = s.b.react == ReactionProviderKind::StaticNull;
            // The pair is always (box=A=RigidInvMass, ground=B=StaticNull); the
            // a_rigid/b_static check makes the local mapping robust + matches the
            // oracle's side-dispatch shape.
            const int box_local = a_rigid ? 0 : 1;
            const int static_local = a_rigid ? 1 : 0;
            (void)b_static;
            rows.body_indices[2u * r + static_cast<uint32_t>(box_local)] =
                box_body_index;
            rows.body_indices[2u * r + static_cast<uint32_t>(static_local)] =
                nuka::constraint::kInvalidBodyIndex;
        }
    }

    if (rows.RowCount() == 0u || sides.empty()) return;  // no env touched the plane.

    // ONE unified two-way solve over the concatenated rows + the FULL env-major
    // bodies. ctx.articulation left default (art_refs=nullptr -> the pure-rigid C5a
    // path; NO articulation in P2.2). cfg COPIED from the oracle (lines ~599-603) so
    // an N=1 env is byte-identical. UnifiedSolve mutates bodies_ velocities in place.
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies_;
    ctx.sides = &sides;
    ctx.dt = dt_;
    // ctx.articulation default-constructed (art_refs=nullptr): NO articulation arm.
    nuka::solver::UnifiedSolve(ctx, cfg);
}

}  // namespace nuka::runtime::coresident
