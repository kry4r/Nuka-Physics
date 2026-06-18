#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::ContactManifold -- the unified detection->solver handoff
// (v0.8 C3a, extends the v0.7 rigid-maximal struct per docs/plans/
// 2026-06-03-v08-detailed-tasks.md §0.1).
// ---------------------------------------------------------------------------
// EXTEND, not replace: the v0.7 struct carried only `uint32 body_a/body_b`, a
// scalar friction/restitution, and ContactPoint[4]. v0.8 makes it class-blind:
//   - the two sides are now type-tagged CollidableRef (a/b) -- the rigid case is
//     preserved EXACTLY because CollidableRef defaults to
//     {type=RigidBody, react=RigidInvMass}, so a former `body_a = X` write is now
//     `a.handle = X` with identical rigid-maximal semantics (OPEN-A resolution).
//   - per-point compliant params (solref) + a D1 stable_key per point, and a
//     per-manifold solimp + manifold_key for the C4 compliant contact row and
//     the C2/C3 deterministic sort.
//
// The production stepper path (row_builder.cpp -> AppendContactGroup) reads ONLY
// `a.handle`/`b.handle`/point_count/friction/restitution/points -- the NEW fields
// (type/react/solimp/manifold_key/solref_*/stable_key) default to the
// rigid-maximal values and are UNREAD by the v0.7 solver, so goldens do not move.
// They are consumed by C4 (compliant rows) once that path lands.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"  // CollidableRef (C2a) + generated registry
#include "math/vec3.hpp"

#include <cstdint>

// HD guard: matches candidate_pair.hpp. The on-device contact kernels
// (contact_generation.cu) value-construct a ContactManifold and write fields
// directly; they do NOT call AddPoint/Clear today, but marking the helpers HD is
// harmless future-proofing for the C3b/c/d stub handlers (which may run on
// device) and costs nothing on the host path.
#if defined(__CUDACC__)
#define NUKA_MANIFOLD_HD __host__ __device__
#else
#define NUKA_MANIFOLD_HD
#endif

namespace nuka::constraint {

struct ContactPoint {
    math::Vec3 position;
    math::Vec3 normal;                   // separation dir for side A (existing convention)
    float penetration        = 0.0f;     // >0 == overlap depth
    float normal_impulse     = 0.0f;     // warm start (existing)
    float friction_impulse_1 = 0.0f;
    float friction_impulse_2 = 0.0f;
    uint64_t stable_key      = 0u;       // NEW (C3a): D1 contact identity (see §0.6)
    // compliant params, resolved per-point from the merged contact params (§0.2);
    // defaults = MuJoCo defaults so the rigid case is unchanged:
    float solref_timeconst   = 0.02f;    // NEW (C3a): solref[0]
    float solref_dampratio   = 1.0f;     // NEW (C3a): solref[1]
};

// Named MuJoCo solimp params (dmin,dmax,width,mid,power), replacing the bare
// solimp[5] positional convention. Standard-layout: the 5 contiguous floats
// decay to a const float* / float* and index via operator[], so existing
// solimp[i] readers and the ComputeCompliantRow(const float solimp[5]) call site
// are unchanged.
struct SolImp {
    float dmin  = 0.9f;
    float dmax  = 0.95f;
    float width = 0.001f;
    float mid   = 0.5f;
    float power = 2.0f;

    NUKA_MANIFOLD_HD float& operator[](unsigned i) { return (&dmin)[i]; }
    NUKA_MANIFOLD_HD float operator[](unsigned i) const { return (&dmin)[i]; }
    NUKA_MANIFOLD_HD operator float*() { return &dmin; }
    NUKA_MANIFOLD_HD operator const float*() const { return &dmin; }
};

struct ContactManifold {
    CollidableRef a;                     // NEW (C3a): was uint32 body_a (rigid case: a.handle)
    CollidableRef b;                     // NEW (C3a): was uint32 body_b (rigid case: b.handle)
    uint32_t point_count = 0;
    float friction       = 0.5f;         // merged (elementwise-max), isotropic μ
    float restitution    = 0.0f;
    SolImp solimp;                       // NEW (C3a): named dmin,dmax,width,mid,power
    float margin = 0.0f;                 // NEW (C5c): merged detection margin (max of sides)
    float gap    = 0.0f;                 // NEW (C5c): merged unforced gap (max of sides)
    uint64_t manifold_key = 0u;          // NEW (C3a): (min,max handle)+type, D1 sort key

    static constexpr uint32_t kMaxPoints = 4;   // KEEP 4: D1-friendly + matches MuJoCo face-clip ≤4
    ContactPoint points[kMaxPoints];

    /// Add a contact point. If the manifold is full the point is dropped.
    NUKA_MANIFOLD_HD void AddPoint(const ContactPoint& pt) {
        if (point_count < kMaxPoints) {
            points[point_count] = pt;
            ++point_count;
        }
    }

    /// Reset the manifold to its DEFAULT-CONSTRUCTED state (C3c-named debt, folded
    /// at W1a). A bare `point_count = 0` left the unused points[] tail + a/b/
    /// friction/restitution/solimp/manifold_key at STALE values from a prior fill --
    /// so a manifold buffer REUSED across steps could carry nondeterministic padding
    /// into a multi-step rollout and break the D1 byte-identity gate. Reset EVERY
    /// member to its default initializer (the whole points[] array, the
    /// CollidableRefs, the merged params, the key) so a cleared-then-refilled
    /// manifold is byte-identical to a value-initialized `ContactManifold{}`.
    /// AddPoint/StampSides re-populate the live fields; this only guarantees the
    /// UNUSED tail is deterministic. Behavior-preserving for the single-shot callers
    /// (the driver already value-inits `ContactManifold m{}` before each fill, so
    /// their tail already matched these defaults -- nothing they read changes).
    NUKA_MANIFOLD_HD void Clear() {
        a = CollidableRef{};
        b = CollidableRef{};
        point_count = 0;
        friction = 0.5f;
        restitution = 0.0f;
        solimp = SolImp{};
        margin = 0.0f;
        gap = 0.0f;
        manifold_key = 0u;
        for (uint32_t i = 0u; i < kMaxPoints; ++i) {
            points[i] = ContactPoint{};
        }
    }
};

} // namespace nuka::constraint
