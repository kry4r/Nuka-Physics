#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::solref_solimp -- the MuJoCo `_efc_row` compliance math
// (v0.8 C4a), ported VERBATIM from mujoco_warp.
// ---------------------------------------------------------------------------
// Computes the spring-damper-impedance constraint row terms (k, b, imp, D,
// R=1/D, aref) from a contact's merged solref[2]/solimp[5] parameters, exactly
// per the MuJoCo `_efc_row` function:
//   mujoco_warp/_src/constraint.py:53-117 (the @wp.func _efc_row), and the
//   contact CALL SITE _efc_contact kernel (constraint.py:2188-2253) for the
//   sign convention.
// Cross-checked against mujoco docs modeling.html#solver-parameters.
//
// WHAT IS PRODUCED (and which is which -- DO NOT SWAP, see below):
//   k    : reference spring stiffness   (1/(dmax^2*timeconst^2*dampratio^2))
//   b    : reference damping            (2/(dmax*timeconst))
//   imp  : impedance d in [dmin,dmax]   (the solimp sigmoid blended over pos)
//   D    : the PRIMAL constraint diagonal stiffness == MuJoCo `efc_D`
//          (D = 1/max(invweight*(1-imp)/imp, MINVAL)). d->1 (hard) => D->inf.
//   R    : the DUAL diagonal REGULARIZER == 1/D, the term ADDED to the
//          A = J M^-1 J^T denominator (so larger imp => smaller R => harder).
//          d->1 (hard) => R->0. R ~ invweight*(1-imp)/imp.
//   aref : the reference acceleration   (-k*imp*pos_aref - b*vel), the bias the
//          solver drives the constraint toward (rhs).
//
// *** DO NOT SWAP D AND R *** (research-newton-mujocowarp.md:73). The MuJoCo
// formula OUTPUTS D (the cost weight; warp's per-row cost is 0.5*Jaref^2*efc_D).
// The DUAL regularizer that a PGS / co-solve adds to the effective-mass
// denominator A is R = 1/D. C4b writes R into Row.compliance_alpha (the term
// added to the denominator, exactly as XPBD adds alpha/dt^2) and aref into
// Row.rhs. This header only COMPUTES; it writes nothing to any Row.
//
// SIGN CONVENTION (sourced from the contact CALL SITE, not the spec prose) --
// mujoco_warp constraint.py:2187-2253:
//   pos      = dist - includemargin       // signed gap; pos < 0 => penetrating
//   pos_aref = pos                        // for the NORMAL row (friction => 0)
//   pos_imp  = pos                        // SAME signed value
//   vel      = Jqvel                      // constraint-space (normal) velocity
//   invweight= body_invweight0[b1] + body_invweight0[b2]   // reaction inv-mass
// C5-WIRING NOTE (C4-review NIT-2): MuJoCo's invweight is `body_invweight0` (the
// DIRECTION-INDEPENDENT approximation evaluated at qpos0). C5 will instead feed
// the SUM of both sides' C4c `ReactionProvider::effective_inv_mass` (the EXACT
// direction-dependent J M^-1 J^T). Same units/sense (additive J M^-1 J^T), so the
// contract holds, but the value differs from MuJoCo's approx (measured ratio ~0.6
// on an off-axis 2-link contact) — a DELIBERATE accuracy improvement, not a bug.
// So for a Nuka contact: pos (== pos_imp == pos_aref for the normal row) is the
// SIGNED separation distance: NEGATIVE when overlapping (penetration depth is
// -pos), positive when there is a gap. The impedance uses |pos_imp| so it is
// sign-blind (penetration MAGNITUDE drives the sigmoid). `pos_aref` and `vel`
// enter aref SIGNED, so the caller (C4b/C5) MUST pass the signed pos and the
// signed normal velocity. Friction rows pass pos_aref = 0 (no positional bias).
//
// HOST + DEVICE (NUKA_SRSI_HD), mirrors sdf_contact.hpp: header-only inline so
// g++ (host tests) and nvcc (device kernels) share ONE definition. Uses NO
// non-HD helper -- only built-in float arithmetic and the integer-power unroll.
//
// DETERMINISM (D1) + OPEN-F (the pow trade-off):
//   libm `powf` is NOT guaranteed bit-identical between g++ and nvcc, so a
//   naive `powf(x, power)` would be NON-D1 across host/device. OPEN-F is RULED
//   (v08-detailed-tasks.md:586): restrict the solimp exponent to INTEGER powers
//   evaluated by an UNROLLED repeated-multiply helper (IntPow below). With that
//   + IEEE single-precision division (prec-div) + NO FMA contraction on either
//   side (the parity test TU compiles --fmad=false / -ffp-contract=off), every
//   op is a correctly-rounded IEEE single => host and device are BIT-IDENTICAL.
//   The MuJoCo default power=2 (and power=1 linear) are exact integers.
//   *** DOCUMENTED LIMITATION: a non-integer authored `power` is ROUNDED to the
//   NEAREST integer (clamped >= 1). This is the deliberate, owner-ruled OPEN-F
//   determinism trade-off, NOT a bug. *** If a future material truly needs a
//   fractional power it must accept a documented host-vs-device tolerance
//   (out of scope for v0.8).
//
//   PRODUCTION-FMAD CAVEAT (flag for C4b/C5): the host/device BYTE-identity
//   proven by the C4a parity test holds under the --fmad=false / -ffp-contract=
//   off pairing on that test TU. The `aref` and `D` formulas are multiply-add
//   chains; a DEFAULT-flag (--fmad=true) production device TU may contract them
//   to FMA and differ from a host .cpp by ~1 ULP. This does NOT break the
//   codebase's real D1 invariant (device-vs-device two-run identity is
//   fmad-independent). But any wiring that needs host==device byte-identity
//   must reuse the --fmad=false / -ffp-contract=off pairing.
// ---------------------------------------------------------------------------

#if defined(__CUDACC__)
#define NUKA_SRSI_HD __host__ __device__
#else
#define NUKA_SRSI_HD
#endif

namespace nuka::constraint {

// MuJoCo constants (sourced from mujoco.mjMINVAL / mjMINIMP / mjMAXIMP, the
// values used by mujoco_warp _src/types.py MJ_MINVAL/MJ_MINIMP/MJ_MAXIMP):
//   mjMINVAL = 1e-15   (the floor below which a quantity is "absent")
//   mjMINIMP = 0.0001  (minimum constraint impedance)
//   mjMAXIMP = 0.9999  (maximum constraint impedance)
// These match docs/plans/2026-06-03-v08-detailed-tasks.md:300/305 and were
// confirmed live via `python -c "import mujoco; print(mujoco.mjMINVAL, ...)"`.
constexpr float kSolMinVal = 1.0e-15f;  // mjMINVAL
constexpr float kSolMinImp = 1.0e-4f;   // mjMINIMP (0.0001)
constexpr float kSolMaxImp = 0.9999f;   // mjMAXIMP

// --- tiny HD arithmetic (avoid <algorithm>/<cmath> in the device path) ------
NUKA_SRSI_HD inline float SrsiMaxF(float a, float b) { return (a > b) ? a : b; }
NUKA_SRSI_HD inline float SrsiMinF(float a, float b) { return (a < b) ? a : b; }
NUKA_SRSI_HD inline float SrsiAbsF(float a) { return (a < 0.0f) ? -a : a; }
NUKA_SRSI_HD inline float SrsiClampF(float v, float lo, float hi) {
    // MuJoCo wp.clamp(v, lo, hi): order matches (lo first, then hi). Assumes
    // lo <= hi (true for MINIMP < MAXIMP); deterministic two-compare form.
    return SrsiMinF(SrsiMaxF(v, lo), hi);
}

// Round a (possibly fractional) authored power to the NEAREST non-negative
// integer used for the integer-power unroll. Deterministic round-half-away-from
// -zero via floor(x + 0.5) implemented without <cmath> (truncation toward zero
// on a non-negative argument == floor). `power` is already clamped >= 1.0 by the
// caller before this is reached for the exponents that can underflow, but we
// guard non-negativity here for the standalone helper.
NUKA_SRSI_HD inline int SrsiRoundToInt(float x) {
    // Powers are >= 0 here; add 0.5 and truncate toward zero (== floor for >=0).
    if (x < 0.0f) {
        const float t = -x + 0.5f;
        return -static_cast<int>(t);
    }
    const float t = x + 0.5f;
    return static_cast<int>(t);
}

// Integer power by repeated multiply (the OPEN-F deterministic unroll). Fixed
// left-to-right multiply order => identical FP rounding on host and device.
// Returns 1.0f for n <= 0 (so mid^(power-1) with power==1 gives mid^0 == 1).
NUKA_SRSI_HD inline float SrsiIntPow(float base, int n) {
    float acc = 1.0f;
    for (int i = 0; i < n; ++i) {
        acc = acc * base;  // fixed order; no FMA opportunity (single multiply)
    }
    return acc;
}

// The C4a DELIVERABLE: the per-contact-point compliant row terms the solver
// consumes. R is the dual regularizer (-> Row.compliance_alpha), aref_bias is
// the reference accel (-> Row.rhs). See file header for the no-swap warning.
struct CompliantContactRow {
    float R = 0.0f;          // dual regularizer = 1/D (ADD to A denominator)
    float aref_bias = 0.0f;  // reference acceleration = -k*imp*pos_aref - b*vel
};

// The RICHER output: every intermediate term so a test can cross-check each
// step against the hand-evaluated MuJoCo formula (not just R/aref).
struct CompliantRowTerms {
    float k = 0.0f;     // reference spring stiffness
    float b = 0.0f;     // reference damping
    float imp = 0.0f;   // impedance d in [dmin, dmax]
    float D = 0.0f;     // PRIMAL diagonal stiffness (efc_D)
    float R = 0.0f;     // DUAL regularizer = 1/D
    float aref = 0.0f;  // reference acceleration
};

// Full computation. `refsafe` mirrors MuJoCo's REFSAFE: when true (DEFAULT --
// REFSAFE is enabled / NOT disabled in MuJoCo's disableflags), timeconst is
// clamped to >= 2*dt. Pass false only to model an explicitly REFSAFE-disabled
// model.
//
// Grouping is preserved VERBATIM from mujoco_warp for both numeric parity AND
// host/device order-identity:
//   dmax_sq        = dmax * dmax
//   k              = 1 / (dmax_sq * timeconst * timeconst * dampratio*dampratio)
//   b              = 2 / (dmax * timeconst)
//   D              = 1 / max((invweight * (1 - imp)) / imp, MINVAL)
//   aref           = -k * imp * pos_aref - b * vel
NUKA_SRSI_HD inline CompliantRowTerms ComputeCompliantRowTerms(
    const float solref[2], const float solimp[5],
    float pos_imp, float pos_aref, float vel, float invweight, float dt,
    bool refsafe = true) {
    // calculate kbi
    float timeconst = solref[0];
    const float dampratio = solref[1];
    float dmin = solimp[0];
    float dmax = solimp[1];
    float width = solimp[2];
    float mid = solimp[3];
    float power_f = solimp[4];

    // REFSAFE ref-safety clamp (MuJoCo: applied when NOT disabled; default on).
    if (refsafe) {
        timeconst = SrsiMaxF(timeconst, 2.0f * dt);
    }

    dmin = SrsiClampF(dmin, kSolMinImp, kSolMaxImp);
    dmax = SrsiClampF(dmax, kSolMinImp, kSolMaxImp);
    width = SrsiMaxF(kSolMinVal, width);
    mid = SrsiClampF(mid, kSolMinImp, kSolMaxImp);
    power_f = SrsiMaxF(1.0f, power_f);

    // OPEN-F: round the (clamped >=1) power to the nearest integer for the
    // deterministic integer-power unroll. Default power=2 and power=1 are exact.
    int power_i = SrsiRoundToInt(power_f);
    if (power_i < 1) power_i = 1;  // defensive (power_f >= 1 already)

    // stiffness k, damping b (standard / positive-solref convention).
    const float dmax_sq = dmax * dmax;
    float k = 1.0f / (dmax_sq * timeconst * timeconst * dampratio * dampratio);
    float b = 2.0f / (dmax * timeconst);
    // DIRECT / negative convention: non-positive solref entries override.
    if (solref[0] <= 0.0f) k = -solref[0] / dmax_sq;
    if (solref[1] <= 0.0f) b = -solref[1] / dmax;

    // impedance sigmoid in x = |pos_imp| / width. The two branches use integer
    // powers `power` and `power-1` via the deterministic unroll (OPEN-F).
    const float imp_x = SrsiAbsF(pos_imp) / width;
    const int pm1 = power_i - 1;  // >= 0; SrsiIntPow returns 1.0f for 0
    const float imp_a = (1.0f / SrsiIntPow(mid, pm1)) * SrsiIntPow(imp_x, power_i);
    const float imp_b =
        1.0f - (1.0f / SrsiIntPow(1.0f - mid, pm1)) * SrsiIntPow(1.0f - imp_x, power_i);
    float imp_y = (imp_x < mid) ? imp_a : imp_b;
    float imp = dmin + imp_y * (dmax - dmin);
    imp = SrsiClampF(imp, dmin, dmax);
    if (imp_x > 1.0f) imp = dmax;

    // outputs (D, aref). Grouping verbatim from MuJoCo.
    const float D = 1.0f / SrsiMaxF((invweight * (1.0f - imp)) / imp, kSolMinVal);
    const float R = 1.0f / D;  // dual regularizer
    const float aref = -k * imp * pos_aref - b * vel;

    CompliantRowTerms out;
    out.k = k;
    out.b = b;
    out.imp = imp;
    out.D = D;
    out.R = R;
    out.aref = aref;
    return out;
}

// The compact deliverable wrapper: returns just (R, aref_bias) for C4b.
NUKA_SRSI_HD inline CompliantContactRow ComputeCompliantRow(
    const float solref[2], const float solimp[5],
    float pos_imp, float pos_aref, float vel, float invweight, float dt,
    bool refsafe = true) {
    const CompliantRowTerms t = ComputeCompliantRowTerms(
        solref, solimp, pos_imp, pos_aref, vel, invweight, dt, refsafe);
    CompliantContactRow out;
    out.R = t.R;
    out.aref_bias = t.aref;
    return out;
}

}  // namespace nuka::constraint
