#pragma once
// Contact filtering and MuJoCo Warp parameter merging shared by cooking and row emission.

#include <cmath>
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

// One shape's contact parameters; priority and solmix determine the pair's mixing weight.
struct ContactParamsIn {
    int32_t priority    = 0;
    uint8_t condim      = 3;
    union { float friction_mu = 1.0f; float mu1; };
    float   mu2         = 1.0f;
    float   friction_spin = 0.005f;
    float   friction_roll = 0.0001f;
    float   solref[2]   = {0.02f, 1.0f};
    float   solimp[5]   = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float   solmix      = 1.0f;
    float   margin      = 0.0f;
    float   gap         = 0.0f;
    float   friction_stiffness = -1.0f;
};

// The MERGE OUTPUT: a single resolved contact-parameter set (no priority/solmix;
// those are consumed during the merge).
struct MergedContactParams {
    uint8_t condim      = 3;
    union { float friction_mu = 1.0f; float mu1; };
    float   mu2         = 1.0f;
    float   friction_spin = 0.005f;
    float   friction_roll = 0.0001f;
    float   solref[2]   = {0.02f, 1.0f};
    float   solimp[5]   = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float   margin      = 0.0f;
    float   gap         = 0.0f;
    float   friction_stiffness = -1.0f;
};

// --- tiny HD arithmetic (avoid <algorithm> / std:: in the device path) -----
NUKA_CF_HD inline float CfMaxF(float a, float b) { return (a > b) ? a : b; }
NUKA_CF_HD inline float CfMinF(float a, float b) { return (a < b) ? a : b; }

// Either shape's contact type can match the other's affinity.
NUKA_CF_HD inline bool PassesContactBitmask(uint32_t contypeA, uint32_t conaffA,
                                            uint32_t contypeB, uint32_t conaffB) {
    return ((contypeA & conaffB) != 0u) || ((contypeB & conaffA) != 0u);
}

// Unauthored pairs mix references by priority and solmix; margins and gaps add across shapes.
// A nonpositive solref selects the componentwise minimum independently of priority.
NUKA_CF_HD inline MergedContactParams MergeContactParams(const ContactParamsIn& A,
                                                         const ContactParamsIn& B) {
    MergedContactParams out;

    float mix;
    if (A.priority != B.priority) {
        const ContactParamsIn& w = (A.priority > B.priority) ? A : B;
        mix = A.priority > B.priority ? 1.0f : 0.0f;
        out.condim      = w.condim;
        out.friction_mu = w.friction_mu;
        out.mu2         = w.mu2;
        out.friction_spin = w.friction_spin;
        out.friction_roll = w.friction_roll;
    } else {
        out.condim = (A.condim > B.condim) ? A.condim : B.condim;
        out.friction_mu = CfMaxF(A.friction_mu, B.friction_mu);
        out.mu2 = CfMaxF(A.mu2, B.mu2);
        out.friction_spin = CfMaxF(A.friction_spin, B.friction_spin);
        out.friction_roll = CfMaxF(A.friction_roll, B.friction_roll);
        const bool a_ok = A.solmix >= kContactSolmixMinVal;
        const bool b_ok = B.solmix >= kContactSolmixMinVal;
        if (a_ok && b_ok) mix = A.solmix / (A.solmix + B.solmix);
        else if (!a_ok && !b_ok) mix = 0.5f;
        else if (!a_ok) mix = 0.0f;
        else mix = 1.0f;
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

    out.margin = A.margin + B.margin;
    out.gap = A.gap + B.gap;
    if (A.friction_stiffness >= 0.0f && B.friction_stiffness >= 0.0f)
        out.friction_stiffness = mix * A.friction_stiffness + imix * B.friction_stiffness;
    return out;
}

// A deformable particle against a body grips with the geometric mean of both coefficients.
NUKA_CF_HD inline float ParticleBodyFriction(float particle_mu, float body_mu) {
    return sqrtf(particle_mu * body_mu);
}

}  // namespace nuka::scene
