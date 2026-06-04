#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::contact_filter -- MuJoCo-parity contact filtering primitives
// (v0.8 C1c). HOST + DEVICE pure-arithmetic helpers, no std:: containers.
// ---------------------------------------------------------------------------
// Two standalone, dependency-free helpers, reused by the cook-time filter
// policy (C1c) AND by the runtime dynamic-pair row emission (C4):
//
//   1. PassesContactBitmask -- MuJoCo's two-way contype/conaffinity AND/OR
//      broadphase admission test.
//   2. MergeContactParams   -- MuJoCo's `mj_contactParam` per-contact parameter
//      combination rule for a pair that has NO explicit <contact><pair> (the
//      authored-pair path stores params as-authored and does NOT route through
//      this merge).
//
// HD CONTRACT. Both helpers are NUKA_CF_HD (mirrors the sdf_contact.hpp HD
// pattern): host tests call them directly, the device row builder (C4) wraps
// them -- ONE code path. They take plain scalars / fixed arrays (NOT the SoA
// CookedContactParamTable) so a caller can pass either a cooked row's columns
// or an authored ContactPairOverride's fields. No dynamic allocation, no
// atomics, fixed-order arithmetic => deterministic (D1) and order-stable.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_CF_HD __host__ __device__
#else
#define NUKA_CF_HD
#endif

namespace nuka::scene {

// MuJoCo's `mjMINVAL` (engine/engine_util_errmem) -- the floor below which a
// solver-mix weight is treated as "absent". Used by the solmix branch.
constexpr float kContactSolmixMinVal = 1e-15f;

// The MERGE INPUT: one shape's contact parameters, as plain values. `priority`
// and `solmix` participate in the merge (they pick the winner / mix weight) but
// are NOT part of the merged OUTPUT (a merged contact has a single resolved
// param set). Callers fill this from a cooked row or from defaults.
struct ContactParamsIn {
    int32_t priority    = 0;
    uint8_t condim      = 3;
    float   friction_mu = 1.0f;
    float   solref[2]   = {0.02f, 1.0f};
    float   solimp[5]   = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float   solmix      = 1.0f;
    float   margin      = 0.0f;
    float   gap         = 0.0f;
};

// The MERGE OUTPUT: a single resolved contact-parameter set (no priority/solmix;
// those are consumed during the merge).
struct MergedContactParams {
    uint8_t condim      = 3;
    float   friction_mu = 1.0f;
    float   solref[2]   = {0.02f, 1.0f};
    float   solimp[5]   = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float   margin      = 0.0f;
    float   gap         = 0.0f;
};

// --- tiny HD arithmetic (avoid <algorithm> / std:: in the device path) -----
NUKA_CF_HD inline float CfMaxF(float a, float b) { return (a > b) ? a : b; }
NUKA_CF_HD inline float CfMinF(float a, float b) { return (a < b) ? a : b; }

// MuJoCo broadphase admission (verified, research doc §8 line 77):
//   geoms i,j collide iff (contype_i & conaffinity_j) || (contype_j & conaffinity_i)
// i.e. a bitwise AND in EACH direction, OR'd. Defaults contype=conaffinity=1
// (everything collides). Returns true when the pair is admitted.
NUKA_CF_HD inline bool PassesContactBitmask(uint32_t contypeA, uint32_t conaffA,
                                            uint32_t contypeB, uint32_t conaffB) {
    return ((contypeA & conaffB) != 0u) || ((contypeB & conaffA) != 0u);
}

// MuJoCo `mj_contactParam` per-contact merge (research doc §8 lines 79-89,
// VERBATIM from mujoco engine_collision_driver.c). Used ONLY for a pair that
// has no explicit <contact><pair> (authored pairs are stored as-authored).
//
//   If priorities differ: the HIGHER-priority geom supplies ALL params wholesale
//   (condim, friction, solref, solimp, margin, gap).
//   If priorities are equal:
//     condim   = max(condimA, condimB)
//     friction = max(frictionA, frictionB)           (scalar isotropic mu)
//     mix      = (solmixA>=MINVAL && solmixB>=MINVAL) ? solmixA/(solmixA+solmixB)
//                                                     : 0.5  (degenerate, see note)
//     solref   = both[0]>0 (standard) ? mix*A + (1-mix)*B  (elementwise)
//                                     : min(A,B)           (either direct/<=0)
//     solimp   = mix*A + (1-mix)*B                   (elementwise)
//     margin   = max(marginA, marginB)
//     gap      = max(gapA, gapB)
//
// NOTE on the degenerate solmix branch: the research doc (line 83) shows the
// guard `(solmix1>=MINVAL && solmix2>=MINVAL)` but truncates the else. The task
// summary instead used `(solmixA+solmixB > MINVAL) ? .. : 0.5`. Following the
// task instruction to PREFER THE DOC, we implement the doc's two-sided guard
// and choose an explicit else of 0.5 (equal-weight average) since neither source
// pins the degenerate else value. For any normal solmix (both >= MINVAL) the two
// formulations agree; they diverge only when a geom authored solmix < 1e-15.
NUKA_CF_HD inline MergedContactParams MergeContactParams(const ContactParamsIn& A,
                                                         const ContactParamsIn& B) {
    MergedContactParams out;

    if (A.priority != B.priority) {
        // Higher-priority geom wins wholesale.
        const ContactParamsIn& w = (A.priority > B.priority) ? A : B;
        out.condim      = w.condim;
        out.friction_mu = w.friction_mu;
        out.solref[0]   = w.solref[0];
        out.solref[1]   = w.solref[1];
        for (int i = 0; i < 5; ++i) out.solimp[i] = w.solimp[i];
        out.margin      = w.margin;
        out.gap         = w.gap;
        return out;
    }

    // Equal priority => combine.
    out.condim      = (A.condim > B.condim) ? A.condim : B.condim;
    out.friction_mu = CfMaxF(A.friction_mu, B.friction_mu);

    float mix = 0.5f;
    if (A.solmix >= kContactSolmixMinVal && B.solmix >= kContactSolmixMinVal) {
        mix = A.solmix / (A.solmix + B.solmix);
    }
    const float imix = 1.0f - mix;

    // solref: standard form (both [0] > 0) => mix-weighted; else (either direct
    // / non-positive) => elementwise min.
    if (A.solref[0] > 0.0f && B.solref[0] > 0.0f) {
        out.solref[0] = mix * A.solref[0] + imix * B.solref[0];
        out.solref[1] = mix * A.solref[1] + imix * B.solref[1];
    } else {
        out.solref[0] = CfMinF(A.solref[0], B.solref[0]);
        out.solref[1] = CfMinF(A.solref[1], B.solref[1]);
    }

    for (int i = 0; i < 5; ++i) {
        out.solimp[i] = mix * A.solimp[i] + imix * B.solimp[i];
    }

    out.margin = CfMaxF(A.margin, B.margin);
    out.gap    = CfMaxF(A.gap, B.gap);
    return out;
}

}  // namespace nuka::scene
