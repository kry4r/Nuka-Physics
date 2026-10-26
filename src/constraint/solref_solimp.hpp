#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define NUKA_SRSI_HD __host__ __device__
#else
#define NUKA_SRSI_HD
#endif

namespace nuka::constraint {

constexpr float kSolMinVal = 1.0e-15f;
constexpr float kSolMinImp = 1.0e-4f;
constexpr float kSolMaxImp = 0.9999f;

NUKA_SRSI_HD inline float SrsiMaxF(float a, float b) { return (a > b) ? a : b; }
NUKA_SRSI_HD inline float SrsiMinF(float a, float b) { return (a < b) ? a : b; }
NUKA_SRSI_HD inline float SrsiAbsF(float a) { return (a < 0.0f) ? -a : a; }
NUKA_SRSI_HD inline float SrsiClampF(float v, float lo, float hi) {
    return SrsiMinF(SrsiMaxF(v, lo), hi);
}

struct CompliantContactRow {
    float R = 0.0f;
    float aref_bias = 0.0f;
    float damping = 0.0f;
};

struct CompliantRowTerms {
    float k = 0.0f;
    float b = 0.0f;
    float imp = 0.0f;
    float D = 0.0f;
    float R = 0.0f;
    float aref = 0.0f;
};

// MuJoCo impedance and explicit spring-damper reference in acceleration units.
NUKA_SRSI_HD inline CompliantRowTerms ComputeCompliantRowTerms(
    const float solref[2], const float solimp[5],
    float pos_imp, float pos_aref, float vel, float invweight, float dt,
    bool refsafe = true) {
    float timeconst = solref[0];
    const float dampratio = solref[1];
    if (refsafe) timeconst = SrsiMaxF(timeconst, 2.0f * dt);
    const float dmin = SrsiClampF(solimp[0], kSolMinImp, kSolMaxImp);
    const float dmax = SrsiClampF(solimp[1], kSolMinImp, kSolMaxImp);
    const float width = SrsiMaxF(kSolMinVal, solimp[2]);
    const float mid = SrsiClampF(solimp[3], kSolMinImp, kSolMaxImp);
    const float power = SrsiMaxF(1.0f, solimp[4]);
    const float dmax_sq = dmax * dmax;
    float k = 1.0f / (dmax_sq * timeconst * timeconst * dampratio * dampratio);
    float b = 2.0f / (dmax * timeconst);
    if (solref[0] <= 0.0f) k = -solref[0] / dmax_sq;
    if (solref[1] <= 0.0f) b = -solref[1] / dmax;

    const float x = SrsiAbsF(pos_imp) / width;
    const float a = (1.0f / powf(mid, power - 1.0f)) * powf(x, power);
    const float b_imp = 1.0f - (1.0f / powf(1.0f - mid, power - 1.0f)) * powf(1.0f - x, power);
    const float y = x < mid ? a : b_imp;
    float imp = SrsiClampF(dmin + y * (dmax - dmin), dmin, dmax);
    if (x > 1.0f) imp = dmax;
    const float D = 1.0f / SrsiMaxF(invweight * (1.0f - imp) / imp, kSolMinVal);
    return {k, b, imp, D, 1.0f / D, -k * imp * pos_aref - b * vel};
}

NUKA_SRSI_HD inline CompliantContactRow ComputeCompliantRow(
    const float solref[2], const float solimp[5],
    float pos_imp, float pos_aref, float vel, float invweight, float dt,
    bool refsafe = true) {
    const auto terms = ComputeCompliantRowTerms(
        solref, solimp, pos_imp, pos_aref, vel, invweight, dt, refsafe);
    return {terms.R, terms.aref, terms.b};
}

}  // namespace nuka::constraint
