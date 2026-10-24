#pragma once
// Material-field elements inside the vertex blocks: each element's gradient, Hessian block and energy change
// at one vertex, from the plane-stress law and the vertex-block membrane and hinge-angle kinematics.

#include <cfloat>
#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "math/vec3.hpp"
#include "nk/material/field_response.hpp"
#include "nk/solve/vertex_block.hpp"
#include "nk/solve/vertex_block_forms.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MF_HD __host__ __device__
#else
#define NUKA_MF_HD
#endif

namespace nuka::nk {

inline constexpr uint32_t kMfTriangle = 0u;
inline constexpr uint32_t kMfBend = 1u;
inline constexpr uint32_t kMfMaxVertices = 12u;
inline constexpr uint32_t kMfIncidenceLocalBits = 4u;
// Node pairs across an edge, then two bends; `hinges` packs their local edges and the pair count from bit 4.
// Rest entries are pair area L t / 2, stiffness, nucleation-strength factor and local cohesive-strength factor.
inline constexpr uint32_t kMfSeam = 2u;

// Triangle: corners, opposite seams, inverse material edges and area. Bend: corners, adjacent hinge faces,
// active-hinge bits, row-major turn-to-curvature map, area and rest angles.
struct MfElement {
    uint32_t kind = kMfTriangle;
    uint32_t material = 0u;
    uint32_t vertex[kMfMaxVertices]{~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    uint32_t hinges = 0u;
    float rest[15]{};
    float curvature_dual[9]{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t state_offset = 0u;
    uint32_t trial_offset = 0u;
    uint32_t state_count = kMfPlaneState;
    uint32_t state_version = kMfStateVersion;
};
static_assert(sizeof(MfElement) == 43u * sizeof(uint32_t), "MfElement packs to 172 bytes");

NUKA_MF_HD constexpr uint32_t MfElementStateCount(const MfElement& e, const MfMaterial& m) {
    return e.kind == kMfTriangle ? kMfPlaneState : e.kind == kMfBend ? kMfPlateState
         : e.kind == kMfSeam ? (m.fracture_law == kMfMixedModeCohesive ? kMfMixedCohesiveState : kMfCohesiveState)
                             : 0u;
}

NUKA_MF_HD constexpr uint32_t MfElementVertexCount(const MfElement& e) {
    return e.kind == kMfTriangle ? 3u
         : e.kind == kMfSeam     ? 2u * (e.hinges >> 4u)
                                 : 3u + 3u * ((e.hinges & 1u) + ((e.hinges >> 1u) & 1u) + ((e.hinges >> 2u) & 1u));
}

// An element with the corners its env's vertex splits gave it, when it has any.
NUKA_MF_HD inline MfElement WithCorners(MfElement e, const uint32_t* corners) {
    if (corners != nullptr)
        for (uint32_t j = 0u; j < kMfMaxVertices; ++j)
            if (j < MfElementVertexCount(e)) e.vertex[j] = corners[j];
    return e;
}

// A field incidence packs its element with the vertex's local slot.
NUKA_MF_HD constexpr uint32_t PackMfIncidence(uint32_t element, uint32_t local) {
    return element << kMfIncidenceLocalBits | local;
}
NUKA_MF_HD constexpr uint32_t MfIncidenceElement(uint32_t packed) { return packed >> kMfIncidenceLocalBits; }
NUKA_MF_HD constexpr uint32_t MfIncidenceLocal(uint32_t packed) { return packed & ((1u << kMfIncidenceLocalBits) - 1u); }

// Only the first live alias contributes an element's block to a physical vertex.
NUKA_MF_HD inline bool MfVertexSlotHolds(const uint32_t* corners, uint32_t local, uint32_t vertex) {
    if (corners[local] != vertex) return false;
    for (uint32_t j = 0u; j < local; ++j)
        if (corners[j] == vertex) return false;
    return true;
}

namespace mf {

using math::SymmetricMat3;
using math::Vec3;

NUKA_MF_HD inline float PrincipalTension(Vec3 stress) {
    const float mean = 0.5f * (stress.x + stress.y), half = 0.5f * (stress.x - stress.y);
    return fmaxf(mean + sqrtf(half * half + stress.z * stress.z), 0.0f);
}

// Traction across a candidate crack in the material plane, with compression excluded from mode I.
NUKA_MF_HD inline float MixedTraction(Vec3 stress, Vec3 normal, float tensile, float shear) {
    const float nn = normal.x * normal.x * stress.x + normal.y * normal.y * stress.y +
                     2.0f * normal.x * normal.y * stress.z;
    const float nt = normal.x * normal.y * (stress.y - stress.x) +
                     (normal.x * normal.x - normal.y * normal.y) * stress.z;
    const float opening = fmaxf(nn, 0.0f) / tensile, sliding = nt / shear;
    return sqrtf(opening * opening + sliding * sliding);
}

// Element vertices as interval-start positions plus in-step displacements, as in the vertex blocks.
struct FieldGeometry {
    Vec3 start[kMfMaxVertices];
    Vec3 delta[kMfMaxVertices];
    NUKA_MF_HD Vec3 Relative(uint32_t j) const { return (start[j] - start[0]) + (delta[j] - delta[0]); }
    NUKA_MF_HD vbd::Point Precise(uint32_t j) const {
        return {double(start[j].x) - start[0].x + (double(delta[j].x) - delta[0].x),
                double(start[j].y) - start[0].y + (double(delta[j].y) - delta[0].y),
                double(start[j].z) - start[0].z + (double(delta[j].z) - delta[0].z)};
    }
    NUKA_MF_HD vbd::ElementGeometry Corners() const {
        vbd::ElementGeometry g{};
        for (uint32_t j = 0u; j < 3u; ++j) {
            g.start[j] = start[j];
            g.delta[j] = delta[j];
        }
        return g;
    }
};

// The vertex-block view of a triangle's rest edges, which the membrane kinematics read.
NUKA_MF_HD inline VbdElement RestEdges(const MfElement& e) {
    VbdElement v;
    v.kind = kVbdTriangle;
    for (uint32_t j = 0u; j < 4u; ++j) v.rest[j] = e.rest[j];
    return v;
}

NUKA_MF_HD inline vbd::TriangleStrain Strain(const MfElement& e, const vbd::ElementGeometry& g,
                                             const vbd::MembraneStartState* start) {
    const VbdElement rest = RestEdges(e);
    return start != nullptr ? vbd::MembraneStrain(rest, g, *start) : vbd::MembraneStrain(rest, g);
}

// Rest area, which scales the element's per-area response.
NUKA_MF_HD inline float ElementArea(const MfElement& e) {
    return e.kind == kMfTriangle ? e.rest[4] : e.kind == kMfSeam ? e.rest[0] : e.rest[9];
}

// A hinge over a seam bends with what the seam's damage leaves of it.
NUKA_MF_HD inline float HingeKeep(const float* history, uint32_t i) {
    return history != nullptr ? sqrtf(fmaxf(1.0f - history[kMfHingeDamage + i], 0.0f)) : 1.0f;
}

NUKA_MF_HD inline CohesiveLaw SeamLaw(const MfElement& e, const MfMaterial& m) {
    return {double(e.rest[1]), double(m.tear_strength) * e.rest[3], double(m.tear_energy)};
}

NUKA_MF_HD inline double SeamReached(const float* history, uint32_t pair) {
    return history != nullptr ? double(history[pair]) : 0.0;
}

NUKA_MF_HD inline MixedCohesiveLaw MixedSeamLaw(const MfElement& e, const MfMaterial& m) {
    const double tensile = double(m.tear_strength) * e.rest[3];
    const double shear = (m.shear_strength > 0.0f ? double(m.shear_strength) : double(m.tear_strength)) * e.rest[3];
    return {double(e.rest[1]), tensile, shear, double(m.tear_energy),
            m.shear_tear_energy > 0.0f ? double(m.shear_tear_energy) : double(m.tear_energy),
            m.mixity_exponent > 0.0f ? double(m.mixity_exponent) : 1.0};
}

// The crack normal at the interval start, in the plane of the faces and pointing from the first side to the second.
NUKA_MF_HD inline Vec3 SeamNormal(const float* history) {
    return {history[kMfSeamNormal], history[kMfSeamNormal + 1u], history[kMfSeamNormal + 2u]};
}

NUKA_MF_HD inline MixedSeamResponse MixedPair(const MfElement& e, const MfMaterial& m, const float* history,
                                              uint32_t pair, double dx, double dy, double dz) {
    const double opening = dx * history[kMfSeamNormal] + dy * history[kMfSeamNormal + 1u] +
                           dz * history[kMfSeamNormal + 2u];
    const double sliding = fmax(dx * dx + dy * dy + dz * dz - opening * opening, 0.0);
    return EvaluateMixedSeam(MixedSeamLaw(e, m), history[pair], history[kMfSeamMixity + pair], opening, sliding);
}

NUKA_MF_HD inline MixedSeamResponse MixedPair(const MfElement& e, const MfMaterial& m, const float* history,
                                              uint32_t pair, Vec3 delta) {
    return MixedPair(e, m, history, pair, delta.x, delta.y, delta.z);
}

// Per-area stiffness along the crack normal and across it at an opening.
NUKA_MF_HD inline void MixedStiffness(const MixedCohesiveLaw& law, const MixedSeamResponse& s, float opening,
                                      float* along, float* across) {
    *across = float(s.keep * law.ShearStiffness());
    *along = float(opening > 0.0f ? s.keep * law.normal_stiffness : s.squeeze);
}

// A pair whose sides are still one vertex holds nothing until a split separates them.
NUKA_MF_HD inline bool SeamPairJoined(const MfElement& e, uint32_t pair) {
    return e.vertex[2u * pair] == e.vertex[2u * pair + 1u];
}

// Separation of node pair `pair` from this side to the far side.
NUKA_MF_HD inline Vec3 SeamSeparation(const FieldGeometry& g, uint32_t pair) {
    return g.Relative(2u * pair + 1u) - g.Relative(2u * pair);
}

// A pair pulls its sides together along their separation; the secant T / r keeps the block positive.
NUKA_MF_HD inline void SeamBlock(const MfElement& e, const MfMaterial& m, const float* history, uint32_t local,
                                 const FieldGeometry& g, Vec3* gradient, SymmetricMat3* hessian) {
    const uint32_t pair = local >> 1u;
    if (SeamPairJoined(e, pair)) {
        *gradient = {};
        *hessian = {};
        return;
    }
    const Vec3 delta = SeamSeparation(g, pair);
    if (m.fracture_law == kMfMixedModeCohesive) {
        const Vec3 normal = SeamNormal(history);
        const float opening = delta.Dot(normal);
        float along, across;
        MixedStiffness(MixedSeamLaw(e, m), MixedPair(e, m, history, pair, delta), opening, &along, &across);
        along *= e.rest[0];
        across *= e.rest[0];
        const Vec3 traction = delta * across + normal * ((along - across) * opening);
        *gradient = (local & 1u) != 0u ? traction : -traction;
        SymmetricMat3 block{};
        vbd::AddIdentity(block, across);
        vbd::AddTo(block, vbd::Outer(normal, along - across));
        *hessian = block;
        return;
    }
    const SeamResponse s = EvaluateSeam(SeamLaw(e, m), SeamReached(history, pair), vbd::Norm(delta));
    const float weight = e.rest[0] * float(s.secant);
    *gradient = delta * ((local & 1u) != 0u ? weight : -weight);
    SymmetricMat3 block{};
    vbd::AddIdentity(block, weight);
    *hessian = block;
}

// Energy change of one pair as its separation moves by `step`, exact in product form on the elastic branch.
NUKA_MF_HD inline float SeamChange(const MfElement& e, const MfMaterial& m, const float* history, uint32_t pair,
                                   Vec3 delta, Vec3 step) {
    if (SeamPairJoined(e, pair)) return 0.0f;
    if (m.fracture_law == kMfMixedModeCohesive) {
        const MixedCohesiveLaw law = MixedSeamLaw(e, m);
        const Vec3 to = delta + step, normal = SeamNormal(history);
        const MixedSeamResponse a = MixedPair(e, m, history, pair, delta), b = MixedPair(e, m, history, pair, to);
        const float from_n = delta.Dot(normal), to_n = to.Dot(normal);
        const double held = MixedHeld(law, history[pair], history[kMfSeamMixity + pair]);
        // On one elastic branch the change is the quadratic form's exact product.
        if (a.keep != held || b.keep != held || (from_n >= 0.0f) != (to_n >= 0.0f))
            return e.rest[0] * float(b.psi - a.psi);
        float along, across;
        MixedStiffness(law, a, from_n, &along, &across);
        return 0.5f * e.rest[0] *
               (across * step.Dot(delta + to) + (along - across) * step.Dot(normal) * (from_n + to_n));
    }
    const CohesiveLaw law = SeamLaw(e, m);
    const double reached = SeamReached(history, pair), knee = fmax(reached, law.Onset());
    const Vec3 to = delta + step;
    const double from_size = vbd::Norm(delta), to_size = vbd::Norm(to);
    const SeamResponse a = EvaluateSeam(law, reached, from_size);
    if (from_size <= knee && to_size <= knee) return 0.5f * e.rest[0] * float(a.secant) * step.Dot(delta + to);
    return e.rest[0] * float(EvaluateSeam(law, reached, to_size).psi - a.psi);
}

// Stored energy of a seam once its separations are committed, from double coordinates.
NUKA_MF_HD inline double SeamEnergy(const MfElement& e, const MfMaterial& m, const float* history,
                                    const FieldGeometry& g) {
    double energy = 0.0;
    for (uint32_t pair = 0u; pair < (e.hinges >> 4u); ++pair) {
        const vbd::Point a = g.Precise(2u * pair), b = g.Precise(2u * pair + 1u);
        const double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
        // Stored at the damage the separation reaches, as the trial keeps it.
        if (m.fracture_law == kMfMixedModeCohesive) {
            const MixedSeamResponse s = MixedPair(e, m, history, pair, dx, dy, dz);
            energy += MixedStored(MixedSeamLaw(e, m), s.knee, s.mixity, s);
        } else {
            energy += EvaluateSeam(SeamLaw(e, m), SeamReached(history, pair), sqrt(dx * dx + dy * dy + dz * dz)).stored;
        }
    }
    return double(e.rest[0]) * energy;
}

// A seam's trial keeps each pair's largest separation, or its knee and mode mixity; its cohesive loss takes the
// plastic slot.
NUKA_MF_HD inline MfStatus SeamTrial(const MfElement& e, const MfMaterial& m, const float* history,
                                     const FieldGeometry& g, FieldResponse<float>& out) {
    FieldResponse<float> r{};
    double psi = 0.0, stored = 0.0, loss = 0.0;
    if (m.fracture_law == kMfMixedModeCohesive) {
        for (uint32_t i = 0u; i < kMfMixedCohesiveState; ++i) r.history[i] = history[i];
        for (uint32_t pair = 0u; pair < (e.hinges >> 4u); ++pair) {
            const MixedSeamResponse s = MixedPair(e, m, history, pair, SeamSeparation(g, pair));
            const double kept = MixedStored(MixedSeamLaw(e, m), s.knee, s.mixity, s);
            r.history[pair] = float(s.knee);
            r.history[kMfSeamMixity + pair] = float(s.mixity);
            psi += s.psi;
            stored += kept;
            loss += s.psi - kept;
        }
        r.psi = float(psi);
        r.stored = float(stored);
        r.plastic = float(loss);
        out = r;
        return MfStatus::Ok;
    }
    for (uint32_t pair = 0u; pair < (e.hinges >> 4u); ++pair) {
        const SeamResponse s =
            EvaluateSeam(SeamLaw(e, m), SeamReached(history, pair), vbd::Norm(SeamSeparation(g, pair)));
        r.history[pair] = float(s.reached);
        psi += s.psi;
        stored += s.stored;
        loss += s.loss;
    }
    r.history[kMfSeamCracked] = history != nullptr ? history[kMfSeamCracked] : 0.0f;
    r.psi = float(psi);
    r.stored = float(stored);
    r.plastic = float(loss);
    out = r;
    return MfStatus::Ok;
}

// Each adjacent face stores its own edge endpoints, then its opposite vertex.
NUKA_MF_HD inline uint32_t HingeFace(const MfElement& e, uint32_t i) {
    return 3u + 3u * ((i > 0u ? (e.hinges & 1u) : 0u) + (i > 1u ? ((e.hinges >> 1u) & 1u) : 0u));
}

NUKA_MF_HD inline uint32_t HingeLocal(const MfElement& e, uint32_t i, uint32_t slot) {
    return slot < 3u ? (i + slot + 1u) % 3u : HingeFace(e, i) + slot - 3u;
}

NUKA_MF_HD inline void HingePoints(const MfElement& e, const FieldGeometry& g, uint32_t i, vbd::Point (&x)[6]) {
    for (uint32_t j = 0u; j < 6u; ++j) x[j] = g.Precise(HingeLocal(e, i, j));
}

// The face normals are projected perpendicular to the mean edge direction before measuring their turn.
struct HingeFrame {
    vbd::Point edge[2], arm[2], normal[2], axis;
    double length, sine, cosine;
};

NUKA_MF_HD inline bool HingeFrameAt(const vbd::Point (&x)[6], HingeFrame& f) {
    f.edge[0] = x[1] - x[0];
    f.edge[1] = x[4] - x[3];
    f.arm[0] = x[2] - x[0];
    f.arm[1] = x[5] - x[3];
    f.normal[0] = f.edge[0].Cross(f.arm[0]);
    f.normal[1] = f.arm[1].Cross(f.edge[1]);
    const vbd::Point sum = f.edge[0] + f.edge[1];
    f.length = sqrt(sum.Dot(sum));
    if (!(f.length > 0.0)) return false;
    f.axis = sum / f.length;
    f.cosine = f.normal[0].Dot(f.normal[1]) - f.axis.Dot(f.normal[0]) * f.axis.Dot(f.normal[1]);
    f.sine = f.axis.Dot(f.normal[0].Cross(f.normal[1]));
    return f.cosine * f.cosine + f.sine * f.sine > 0.0;
}

NUKA_MF_HD inline bool HingeAngle(const vbd::Point (&x)[6], double* theta) {
    HingeFrame f{};
    if (!HingeFrameAt(x, f)) return false;
    *theta = atan2(f.sine, f.cosine);
    return true;
}

// Relative vectors are formed before rounding and scaled by one edge to keep the angular products well sized.
struct HingeVectors {
    Vec3 edge[2], arm[2];
    float inverse_length;
};

NUKA_MF_HD inline bool HingeVectorsAt(const vbd::Point (&x)[6], HingeVectors& f) {
    const vbd::Point edge = x[1] - x[0];
    const double length = sqrt(edge.Dot(edge));
    if (!(length > 0.0)) return false;
    f.inverse_length = float(1.0 / length);
    const vbd::Point vector[4] = {edge, x[4] - x[3], x[2] - x[0], x[5] - x[3]};
    Vec3 scaled[4];
    for (uint32_t j = 0u; j < 4u; ++j)
        scaled[j] = Vec3{float(vector[j].x), float(vector[j].y), float(vector[j].z)} * f.inverse_length;
    f.edge[0] = scaled[0];
    f.edge[1] = scaled[1];
    f.arm[0] = scaled[2];
    f.arm[1] = scaled[3];
    return true;
}

// Differentiate both face normals and the mean axis; aliases are combined by the element's vertex block.
NUKA_MF_HD inline bool HingeAngleGradient(const vbd::Point (&x)[6], Vec3 (&q)[6], float* theta) {
    HingeVectors f{};
    if (!HingeVectorsAt(x, f)) return false;
    const Vec3 n0 = f.edge[0].Cross(f.arm[0]), n1 = f.arm[1].Cross(f.edge[1]);
    const Vec3 sum = f.edge[0] + f.edge[1];
    const float length = sqrtf(sum.LengthSq());
    if (!(length > 0.0f)) return false;
    const Vec3 t = sum / length;
    const float X = n0.Dot(n1) - t.Dot(n0) * t.Dot(n1), Y = t.Dot(n0.Cross(n1));
    const float D = X * X + Y * Y;
    if (!(D > 0.0f)) return false;
    const Vec3 g0 = (n1.Cross(t) * X - (n1 - t * t.Dot(n1)) * Y) / D;
    const Vec3 g1 = (t.Cross(n0) * X - (n0 - t * t.Dot(n0)) * Y) / D;
    const Vec3 gt = (n0.Cross(n1) * X + (n0 * t.Dot(n1) + n1 * t.Dot(n0)) * Y) / D;
    const Vec3 axis = (gt - t * t.Dot(gt)) / length;
    const Vec3 a = (f.arm[0].Cross(g0) + axis) * f.inverse_length;
    const Vec3 b = g0.Cross(f.edge[0]) * f.inverse_length;
    const Vec3 c = (g1.Cross(f.arm[1]) + axis) * f.inverse_length;
    const Vec3 d = f.edge[1].Cross(g1) * f.inverse_length;
    q[0] = -a - b; q[1] = a; q[2] = b;
    q[3] = -c - d; q[4] = c; q[5] = d;
    *theta = atan2f(Y, X);
    return true;
}

// Product differences retain arbitrarily small trial turns without subtracting two rounded angles.
NUKA_MF_HD inline void HingeMovesChange(const vbd::Point (&x)[6], const Vec3 (&move)[6], bool* had, bool* has,
                                        float* before, float* after, float* turn) {
    HingeVectors f{};
    *had = *has = false;
    if (!HingeVectorsAt(x, f)) return;
    const Vec3 e = f.edge[0], b = f.edge[1], r = f.arm[0], s = f.arm[1];
    const Vec3 de = (move[1] - move[0]) * f.inverse_length;
    const Vec3 db = (move[4] - move[3]) * f.inverse_length;
    const Vec3 dr = (move[2] - move[0]) * f.inverse_length;
    const Vec3 ds = (move[5] - move[3]) * f.inverse_length;
    const Vec3 n = e.Cross(r), m = s.Cross(b), a = e + b, da = de + db;
    const Vec3 dn = de.Cross(r) + e.Cross(dr) + de.Cross(dr);
    const Vec3 dm = ds.Cross(b) + s.Cross(db) + ds.Cross(db);
    const float aa = a.Dot(a), daa = da.Dot(a * 2.0f + da);
    const float nm = n.Dot(m), dnm = dn.Dot(m) + n.Dot(dm) + dn.Dot(dm);
    const float an = a.Dot(n), dan = da.Dot(n) + a.Dot(dn) + da.Dot(dn);
    const float am = a.Dot(m), dam = da.Dot(m) + a.Dot(dm) + da.Dot(dm);
    const Vec3 cross = n.Cross(m), dcross = dn.Cross(m) + n.Cross(dm) + dn.Cross(dm);
    const float triple = a.Dot(cross), dtriple = da.Dot(cross) + a.Dot(dcross) + da.Dot(dcross);
    const float length = sqrtf(aa), next_length = sqrtf(fmaxf(aa + daa, 0.0f));
    const float C = aa * nm - an * am, S = length * triple;
    const float dC = daa * nm + aa * dnm + daa * dnm - dan * am - an * dam - dan * dam;
    const float dl = length + next_length > 0.0f ? daa / (length + next_length) : 0.0f;
    const float dS = dl * triple + length * dtriple + dl * dtriple;
    *had = aa > 0.0f && C * C + S * S > 0.0f;
    *has = next_length > 0.0f && (C + dC) * (C + dC) + (S + dS) * (S + dS) > 0.0f;
    if (*had) *before = atan2f(S, C);
    if (!*has) return;
    if (!*had) {
        *after = atan2f(S + dS, C + dC);
        return;
    }
    *turn = atan2f(dS * C - dC * S, C * C + S * S + dC * C + dS * S);
    const float pi = 3.14159265358979323846f, sum = *before + *turn;
    *after = sum > pi ? sum - 2.0f * pi : sum <= -pi ? sum + 2.0f * pi : sum;
}

// Voigt curvature (11, 22, 2 12) of a bend from its hinge turns.
template <class Real> NUKA_MF_HD inline void Curvature(const MfElement& e, const Real (&turn)[3], Real (&kappa)[3]) {
    for (uint32_t r = 0u; r < 3u; ++r)
        kappa[r] = Real(e.rest[3u * r]) * turn[0] + Real(e.rest[3u * r + 1u]) * turn[1] +
                   Real(e.rest[3u * r + 2u]) * turn[2];
}

// The elastic-metric dual condenses unrepresented free-edge curvature, including committed history.
inline bool InitializeBendDual(MfElement& e, const MfMaterial& material) {
    uint32_t index[3], count = 0u;
    for (uint32_t i = 0u; i < 3u; ++i)
        if ((e.hinges >> i) & 1u) index[count++] = i;
    const auto metric = BendBase<double>(material);
    double weighted[3][3]{}, lower[3][3]{};
    for (uint32_t i = 0u; i < count; ++i)
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            for (uint32_t r = 0u; r < 3u; ++r)
                weighted[i][axis] += double(e.rest[3u * r + index[i]]) * metric.value[r][axis];
    for (uint32_t i = 0u; i < count; ++i)
        for (uint32_t j = 0u; j <= i; ++j) {
            double sum = 0.0;
            for (uint32_t r = 0u; r < 3u; ++r) sum += weighted[i][r] * e.rest[3u * r + index[j]];
            for (uint32_t k = 0u; k < j; ++k) sum -= lower[i][k] * lower[j][k];
            if (i == j) {
                if (!(sum > 0.0) || !math::IsFinite(sum)) return false;
                lower[i][j] = std::sqrt(sum);
            } else lower[i][j] = sum / lower[j][j];
        }
    for (float& value : e.curvature_dual) value = 0.0f;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        double value[3]{};
        for (uint32_t i = 0u; i < count; ++i) {
            value[i] = weighted[i][axis];
            for (uint32_t j = 0u; j < i; ++j) value[i] -= lower[i][j] * value[j];
            value[i] /= lower[i][i];
        }
        for (uint32_t k = count; k > 0u; --k) {
            const uint32_t i = k - 1u;
            for (uint32_t j = i + 1u; j < count; ++j) value[i] -= lower[j][i] * value[j];
            value[i] /= lower[i][i];
            e.curvature_dual[3u * index[i] + axis] = static_cast<float>(value[i]);
        }
    }
    return true;
}

// Damage acts on elastic curvature through a congruence; committed plastic and Maxwell tensors keep their frame.
template <class Real>
NUKA_MF_HD inline void BendMetric(const MfElement& e, const MfMaterial& m, const float* history,
                                  Real (&entries)[6], Real (&projection)[3][3]) {
    const auto base = BendBase<Real>(m);
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j) {
            projection[i][j] = Real(0);
            for (uint32_t k = 0u; k < 3u; ++k)
                projection[i][j] += Real(e.rest[3u * i + k]) * Real(HingeKeep(history, k)) *
                                     Real(e.curvature_dual[3u * k + j]);
        }
    Real c[3][3]{};
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j)
            for (uint32_t a = 0u; a < 3u; ++a)
                for (uint32_t b = 0u; b < 3u; ++b)
                    c[i][j] += projection[a][i] * base.value[a][b] * projection[b][j];
    entries[0] = c[0][0]; entries[1] = c[1][1]; entries[2] = c[2][2];
    entries[3] = Real(0.5) * (c[0][1] + c[1][0]);
    entries[4] = Real(0.5) * (c[0][2] + c[2][0]);
    entries[5] = Real(0.5) * (c[1][2] + c[2][1]);
}

template <class Real>
NUKA_MF_HD inline MfStatus EvaluateBend(const MfElement& e, const MfMaterial& m, const float* history,
                                       const Real (&curvature)[3], Real h, FieldResponse<Real>& out) {
    Real entries[6], projection[3][3];
    BendMetric(e, m, history, entries, projection);
    FieldResponse<Real> response;
    const auto status = EvaluatePlaneStress<Real>(BendLaw(m), history, curvature, h, response, nullptr, entries);
    if (status != MfStatus::Ok) return status;
    Real maximum = Real(0);
    for (uint32_t i = 0u; i < 3u; ++i) {
        const Real live = projection[i][0] * curvature[0] + projection[i][1] * curvature[1] +
                          projection[i][2] * curvature[2];
        maximum = std::fmax(maximum, std::fabs(live) * (i == 2u ? Real(0.5) : Real(1)));
    }
    response.beyond_range = maximum > Real(BendLaw(m).range);
    response.local.beyond_range = response.beyond_range;
    for (uint32_t i = kMfHingeDamage; i < kMfPlateState; ++i)
        response.history[i] = history != nullptr ? Real(history[i]) : Real(0);
    out = response;
    return MfStatus::Ok;
}

template <class Real>
NUKA_MF_HD inline Real BendStoredEnergy(const MfElement& e, const MfMaterial& m, const float* history,
                                       const Real (&curvature)[3]) {
    Real entries[6], projection[3][3];
    BendMetric(e, m, history, entries, projection);
    return StoredEnergy(BendLaw(m), history, curvature, entries);
}

// Entry (i, j) of A B^T C B, with C the tangent and B the curvature map.
NUKA_MF_HD inline float HingeStiffness(const MfElement& e, const float (&c)[3][3], uint32_t i, uint32_t j) {
    float sum = 0.0f;
    for (uint32_t a = 0u; a < 3u; ++a)
        for (uint32_t b = 0u; b < 3u; ++b) sum += e.rest[3u * a + i] * c[a][b] * e.rest[3u * b + j];
    return e.rest[9] * sum;
}

// Hinge-space moments A B^T S and stiffness A B^T C B of a bend response, the stiffness stored like the tangent.
NUKA_MF_HD inline void HingeSpace(const MfElement& e, const FieldResponse<float>& r, float (&moment)[3],
                                  float (&k)[6]) {
    const float* t = r.tangent;
    const float c[3][3] = {{t[0], t[3], t[4]}, {t[3], t[1], t[5]}, {t[4], t[5], t[2]}};
    for (uint32_t i = 0u; i < 3u; ++i)
        moment[i] = e.rest[9] * (e.rest[i] * r.stress[0] + e.rest[3u + i] * r.stress[1] + e.rest[6u + i] * r.stress[2]);
    k[0] = HingeStiffness(e, c, 0u, 0u);
    k[1] = HingeStiffness(e, c, 1u, 1u);
    k[2] = HingeStiffness(e, c, 2u, 2u);
    k[3] = HingeStiffness(e, c, 0u, 1u);
    k[4] = HingeStiffness(e, c, 0u, 2u);
    k[5] = HingeStiffness(e, c, 1u, 2u);
}

// s (a b^T + b a^T).
NUKA_MF_HD inline SymmetricMat3 SymmetricProduct(Vec3 a, Vec3 b, float s) {
    return {2.0f * s * a.x * b.x, 2.0f * s * a.y * b.y, 2.0f * s * a.z * b.z,
            s * (a.x * b.y + a.y * b.x), s * (a.x * b.z + a.z * b.x), s * (a.y * b.z + a.z * b.y)};
}

NUKA_MF_HD inline void TriangleBlock(const MfElement& e, const MfMaterial& m, const float* history, uint32_t local,
                                     const vbd::TriangleStrain& t, float h, Vec3* gradient, SymmetricMat3* hessian) {
    FieldResponse<float> r{};
    const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
    MembraneResponse<float>(m, history, strain, h, r, nullptr, false);
    const VbdElement rest = RestEdges(e);
    float bx, by;
    vbd::MembraneWeights(rest, local, &bx, &by);
    const float area = e.rest[4], s00 = r.stress[0], s11 = r.stress[1], s01 = r.stress[2];
    const Vec3 f0 = t.f0, f1 = t.f1;
    *gradient = (f0 * (s00 * bx + s01 * by) + f1 * (s01 * bx + s11 * by)) * area;
    // Strain rows dE/dx of this vertex: J^T C J is the material block, the stress the geometric one.
    const Vec3 r0 = f0 * bx, r1 = f1 * by, r2 = f1 * bx + f0 * by;
    const float* c = r.tangent;
    SymmetricMat3 block = vbd::Outer(r0, c[0]);
    vbd::AddTo(block, vbd::Outer(r1, c[1]));
    vbd::AddTo(block, vbd::Outer(r2, c[2]));
    vbd::AddTo(block, SymmetricProduct(r0, r1, c[3]));
    vbd::AddTo(block, SymmetricProduct(r0, r2, c[4]));
    vbd::AddTo(block, SymmetricProduct(r1, r2, c[5]));
    const float bsb = bx * (s00 * bx + s01 * by) + by * (s01 * bx + s11 * by);
    vbd::AddIdentity(block, bsb > 0.0f ? bsb : 0.0f);
    // The vertex-block area barrier keeps a compressed membrane from collapsing in plane.
    const float mu = ShearModulus<float>(m);
    const Vec3 n = f0.Cross(f1);
    const float normal_length = vbd::Norm(n), J = 1.0f + t.area_change;
    if (t.area_change < 0.0f && J > 0.0f && normal_length > 0.0f) {
        const Vec3 q = (f1 * bx - f0 * by).Cross(n / normal_length);
        *gradient = *gradient + q * (area * mu * (t.area_change / J));
        vbd::AddTo(block, vbd::Outer(q, mu / (J * J)));
    }
    *hessian = vbd::Scaled(block, area);
}

// The turn gradients of vertex `local` meet the hinge-space moments; the Hessian block is G K G^T over them.
NUKA_MF_HD inline void BendBlock(const MfElement& e, const MfMaterial& m, const float* history, uint32_t local,
                                 const FieldGeometry& g, float h, Vec3* gradient, SymmetricMat3* hessian) {
    float turn[3] = {0.0f, 0.0f, 0.0f};
    Vec3 q[3] = {};
    for (uint32_t i = 0u; i < 3u; ++i) {
        vbd::Point x[6];
        Vec3 dq[6];
        float theta = 0.0f;
        if (!((e.hinges >> i) & 1u)) continue;
        HingePoints(e, g, i, x);
        if (!HingeAngleGradient(x, dq, &theta)) continue;
        turn[i] = theta - e.rest[10u + i];
        for (uint32_t slot = 0u; slot < 6u; ++slot)
            if (e.vertex[HingeLocal(e, i, slot)] == e.vertex[local]) q[i] += dq[slot];
    }
    float kappa[3], moment[3], k[6];
    Curvature(e, turn, kappa);
    FieldResponse<float> r{};
    EvaluateBend(e, m, history, kappa, h, r);
    HingeSpace(e, r, moment, k);
    *gradient = q[0] * moment[0] + q[1] * moment[1] + q[2] * moment[2];
    SymmetricMat3 block = vbd::Outer(q[0], k[0]);
    vbd::AddTo(block, vbd::Outer(q[1], k[1]));
    vbd::AddTo(block, vbd::Outer(q[2], k[2]));
    vbd::AddTo(block, SymmetricProduct(q[0], q[1], k[3]));
    vbd::AddTo(block, SymmetricProduct(q[0], q[2], k[4]));
    vbd::AddTo(block, SymmetricProduct(q[1], q[2], k[5]));
    *hessian = block;
}

NUKA_MF_HD inline void ElementBlock(const MfElement& e, const MfMaterial& m, const float* history, uint32_t local,
                                    const FieldGeometry& g, const vbd::MembraneStartState* start, float h,
                                    Vec3* gradient, SymmetricMat3* hessian) {
    if (e.kind == kMfTriangle) {
        TriangleBlock(e, m, history, local, Strain(e, g.Corners(), start), h, gradient, hessian);
        return;
    }
    if (e.kind == kMfSeam) {
        SeamBlock(e, m, history, local, g, gradient, hessian);
        return;
    }
    BendBlock(e, m, history, local, g, h, gradient, hessian);
}

// The quadratic model of the incremental potential is exact while neither end yields; otherwise the
// difference of the two potentials is. `step` is the exact change from `from` to `to` when `exact`.
NUKA_MF_HD inline float ResponseChange(const PlaneStressLaw& law, const float* history, const float (&from)[3],
                                       const float (&to)[3], const float (&step)[3], bool exact, float h,
                                       const float* elastic_entries = nullptr) {
    FieldResponse<float> a{}, b{};
    EvaluatePlaneStress<float>(law, history, from, h, a, nullptr, elastic_entries);
    if (law.yield > 0.0f || !exact) EvaluatePlaneStress<float>(law, history, to, h, b, nullptr, elastic_entries);
    if (!exact || a.yielded || b.yielded) {
        double from_wide[3], to_wide[3], entries[6];
        for (uint32_t i = 0u; i < 3u; ++i) {
            from_wide[i] = from[i];
            to_wide[i] = exact ? double(from[i]) + double(step[i]) : double(to[i]);
        }
        for (uint32_t i = 0u; i < 6u; ++i) entries[i] = elastic_entries ? elastic_entries[i] : 0.0;
        FieldResponse<double> start{}, finish{};
        EvaluatePlaneStress<double>(law, history, from_wide, double(h), start, nullptr,
                                     elastic_entries ? entries : nullptr);
        EvaluatePlaneStress<double>(law, history, to_wide, double(h), finish, nullptr,
                                     elastic_entries ? entries : nullptr);
        return static_cast<float>(finish.psi - start.psi);
    }
    float response[3];
    StressJacobianProduct(a, step, response);
    return step[0] * (a.stress[0] + 0.5f * response[0]) + step[1] * (a.stress[1] + 0.5f * response[1]) +
           step[2] * (a.stress[2] + 0.5f * response[2]);
}

// Membrane energy change from `from` by `step`. The log-strain law compares double potentials, as no product form
// of its strain is exact; an end it cannot map has no finite energy.
NUKA_MF_HD inline float MembraneChange(const MfMaterial& m, const float* history, const float (&from)[3],
                                       const float (&to)[3], const float (&step)[3], float h) {
    if (m.law != kMfLogStrainOrthotropicLaw) return ResponseChange(MembraneLaw(m), history, from, to, step, true, h);
    double start_strain[3], finish_strain[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        start_strain[i] = from[i];
        finish_strain[i] = double(from[i]) + double(step[i]);
    }
    FieldResponse<double> start{}, finish{};
    if (MembraneResponse<double>(m, history, start_strain, double(h), start) != MfStatus::Ok ||
        MembraneResponse<double>(m, history, finish_strain, double(h), finish) != MfStatus::Ok) return FLT_MAX;
    return static_cast<float>(finish.psi - start.psi);
}

NUKA_MF_HD inline float BendResponseChange(const MfElement& e, const MfMaterial& m, const float* history,
                                          const float (&from)[3], const float (&to)[3], const float (&step)[3],
                                          bool exact, float h) {
    float entries[6], projection[3][3];
    BendMetric(e, m, history, entries, projection);
    return ResponseChange(BendLaw(m), history, from, to, step, exact, h, entries);
}

// Turns of the hinges before and after vertex `local` moves; `step` keeps a small turn exact.
NUKA_MF_HD inline float BendEnergyChange(const MfElement& e, const MfMaterial& m, const float* history,
                                         uint32_t local, const FieldGeometry& g, float h, Vec3 move) {
    float before[3] = {0.0f, 0.0f, 0.0f}, after[3] = {0.0f, 0.0f, 0.0f}, step[3] = {0.0f, 0.0f, 0.0f};
    bool exact = true;
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        vbd::Point x[6];
        HingePoints(e, g, i, x);
        const float rest = e.rest[10u + i];
        Vec3 shift[6]{};
        for (uint32_t slot = 0u; slot < 6u; ++slot)
            if (e.vertex[HingeLocal(e, i, slot)] == e.vertex[local]) shift[slot] = move;
        bool had = false, has = false;
        float from = 0.0f, to = 0.0f, turn = 0.0f;
        HingeMovesChange(x, shift, &had, &has, &from, &to, &turn);
        if (had) before[i] = from - rest;
        if (has) after[i] = to - rest;
        if (had && has && to == from + turn) step[i] = turn;
        else exact = false;
    }
    float from[3], to[3], moved[3];
    Curvature(e, before, from);
    Curvature(e, step, moved);
    if (exact) {
        for (uint32_t r = 0u; r < 3u; ++r) to[r] = from[r] + moved[r];
    } else {
        Curvature(e, after, to);
    }
    return e.rest[9] * BendResponseChange(e, m, history, from, to, moved, exact, h);
}

// Energy change as vertex `local` moves by `move`, from the strain change formed of products of the move.
NUKA_MF_HD inline float ElementEnergyChange(const MfElement& e, const MfMaterial& m, const float* history,
                                            uint32_t local, const FieldGeometry& g,
                                            const vbd::MembraneStartState* start, float h, Vec3 move) {
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), start);
        const VbdElement rest = RestEdges(e);
        float bx, by;
        vbd::MembraneWeights(rest, local, &bx, &by);
        const Vec3 d0 = move * bx, d1 = move * by;
        const Vec3 f0 = t.f0 + d0, f1 = t.f1 + d1;
        const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
        const float change[3] = {0.5f * d0.Dot(t.f0 + f0), 0.5f * d1.Dot(t.f1 + f1), d0.Dot(f1) + t.f0.Dot(d1)};
        const float moved[3] = {strain[0] + change[0], strain[1] + change[1], strain[2] + change[2]};
        const Vec3 n = t.f0.Cross(t.f1), turn = move.Cross(t.f1 * bx - t.f0 * by);
        const float J = 1.0f + t.area_change, growth = 2.0f * n.Dot(turn) + turn.LengthSq();
        const float moved_sq = J * J + growth;
        const float area_step = n.Dot(n + turn) > 0.0f && moved_sq > 0.0f
            ? growth / (sqrtf(moved_sq) + J) : -1.0f - t.area_change;
        return e.rest[4] * (MembraneChange(m, history, strain, moved, change, h) +
                            ShearModulus<float>(m) * vbd::AreaBarrierChange(t.area_change, area_step));
    }
    if (e.kind == kMfSeam)
        return SeamChange(e, m, history, local >> 1u, SeamSeparation(g, local >> 1u), (local & 1u) != 0u ? move : -move);
    return BendEnergyChange(e, m, history, local, g, h, move);
}

// Stored energy at the committed history; hinge angles come from double coordinates.
NUKA_MF_HD inline double ElementEnergy(const MfElement& e, const MfMaterial& m, const float* history,
                                       const FieldGeometry& g) {
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), nullptr);
        const double strain[3] = {t.g00, t.g11, 2.0 * double(t.g01)};
        return double(e.rest[4]) * (MembraneStoredEnergy(m, history, strain) +
                                    ShearModulus<double>(m) * vbd::AreaBarrier(t.area_change));
    }
    if (e.kind == kMfSeam) return SeamEnergy(e, m, history, g);
    double turn[3] = {0.0, 0.0, 0.0}, kappa[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        vbd::Point x[6];
        HingePoints(e, g, i, x);
        double theta = 0.0;
        if (HingeAngle(x, &theta)) turn[i] = theta - double(e.rest[10u + i]);
    }
    Curvature(e, turn, kappa);
    return double(e.rest[9]) * BendStoredEnergy(e, m, history, kappa);
}

// The per-area response at the state a step reached, whose history the step commits.
NUKA_MF_HD inline MfStatus ElementTrial(const MfElement& e, const MfMaterial& m, const float* history,
                                        const FieldGeometry& g, float h, FieldResponse<float>& out) {
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), nullptr);
        const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
        return MembraneResponse(m, history, strain, h, out);
    }
    if (e.kind == kMfSeam) return SeamTrial(e, m, history, g, out);
    float turn[3] = {0.0f, 0.0f, 0.0f}, kappa[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        vbd::Point x[6];
        HingePoints(e, g, i, x);
        double theta = 0.0;
        if (HingeAngle(x, &theta)) turn[i] = float(theta - e.rest[10u + i]);
    }
    Curvature(e, turn, kappa);
    return EvaluateBend(e, m, history, kappa, h, out);
}

// A triangle collapsed through zero area has no defined energy.
NUKA_MF_HD inline bool ElementStateValid(const MfElement& e, const FieldGeometry& g,
                                         const vbd::MembraneStartState* start) {
    return e.kind != kMfTriangle || Strain(e, g.Corners(), start).area_change > -1.0f;
}

// Gauss-Newton factor: membrane strain and compressed area, or hinge turns, with area-scaled stiffness.
// Seam pairs inherit the same material-vertex interpolation, so their coarse moves leave them unstrained.
struct FieldCurvature {
    Vec3 row[4][kMfMaxVertices];
    float stiffness[4][4];
};

NUKA_MF_HD inline FieldCurvature GaussNewtonCurvature(const MfElement& e, const MfMaterial& m, const float* history,
                                                     const FieldGeometry& g, const vbd::MembraneStartState* start,
                                                     float h) {
    FieldCurvature c{};
    if (e.kind == kMfSeam) return c;
    FieldResponse<float> r{};
    float k[6];
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), start);
        const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
        MembraneResponse<float>(m, history, strain, h, r, nullptr, false);
        const VbdElement rest = RestEdges(e);
        const Vec3 n = t.f0.Cross(t.f1);
        const float normal_length = vbd::Norm(n), J = 1.0f + t.area_change;
        const bool compressed = t.area_change < 0.0f && J > 0.0f && normal_length > 0.0f;
        for (uint32_t j = 0u; j < 3u; ++j) {
            float bx, by;
            vbd::MembraneWeights(rest, j, &bx, &by);
            c.row[0][j] = t.f0 * bx;
            c.row[1][j] = t.f1 * by;
            c.row[2][j] = t.f1 * bx + t.f0 * by;
            if (compressed) c.row[3][j] = (t.f1 * bx - t.f0 * by).Cross(n / normal_length);
        }
        for (uint32_t i = 0u; i < 6u; ++i) k[i] = e.rest[4] * r.tangent[i];
        if (compressed) c.stiffness[3][3] = e.rest[4] * ShearModulus<float>(m) / (J * J);
    } else {
        float turn[3] = {0.0f, 0.0f, 0.0f}, kappa[3], moment[3];
        for (uint32_t i = 0u; i < 3u; ++i) {
            vbd::Point x[6];
            Vec3 dq[6];
            float theta = 0.0f;
            if (!((e.hinges >> i) & 1u)) continue;
            HingePoints(e, g, i, x);
            if (!HingeAngleGradient(x, dq, &theta)) continue;
            turn[i] = theta - e.rest[10u + i];
            for (uint32_t slot = 0u; slot < 6u; ++slot)
                c.row[i][HingeLocal(e, i, slot)] = dq[slot];
        }
        Curvature(e, turn, kappa);
        EvaluateBend(e, m, history, kappa, h, r);
        HingeSpace(e, r, moment, k);
    }
    const float s[3][3] = {{k[0], k[3], k[4]}, {k[3], k[1], k[5]}, {k[4], k[5], k[2]}};
    for (uint32_t a = 0u; a < 3u; ++a)
        for (uint32_t b = 0u; b < 3u; ++b) c.stiffness[a][b] = s[a][b];
    return c;
}

// Generalized strain variations under per-vertex moves, and their stiffness product.
NUKA_MF_HD inline void Vary(const FieldCurvature& c, const Vec3 (&move)[kMfMaxVertices], float (&out)[4]) {
    for (uint32_t a = 0u; a < 4u; ++a) {
        out[a] = 0.0f;
        for (uint32_t j = 0u; j < kMfMaxVertices; ++j) out[a] += c.row[a][j].Dot(move[j]);
    }
}

NUKA_MF_HD inline float Form(const FieldCurvature& c, const float (&x)[4], const float (&y)[4]) {
    float sum = 0.0f;
    for (uint32_t a = 0u; a < 4u; ++a)
        for (uint32_t b = 0u; b < 4u; ++b) sum += x[a] * c.stiffness[a][b] * y[b];
    return sum;
}

// Energy change as all vertices move by `shift`: the membrane strain change is formed of products of the moves,
// and each hinge turn is one relative rotation, as in the vertex descent.
NUKA_MF_HD inline float ElementMovesEnergyChange(const MfElement& e, const MfMaterial& m, const float* history,
                                                const FieldGeometry& g, const vbd::MembraneStartState* start,
                                                float h, const Vec3 (&shift)[kMfMaxVertices]) {
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), start);
        const VbdElement rest = RestEdges(e);
        Vec3 d0{}, d1{};
        for (uint32_t j = 0u; j < 3u; ++j) {
            float bx, by;
            vbd::MembraneWeights(rest, j, &bx, &by);
            d0 = d0 + shift[j] * bx;
            d1 = d1 + shift[j] * by;
        }
        const Vec3 f0 = t.f0 + d0, f1 = t.f1 + d1;
        const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
        const float change[3] = {0.5f * d0.Dot(t.f0 + f0), 0.5f * d1.Dot(t.f1 + f1), d0.Dot(f1) + t.f0.Dot(d1)};
        const float moved[3] = {strain[0] + change[0], strain[1] + change[1], strain[2] + change[2]};
        const Vec3 n = t.f0.Cross(t.f1), turn = d0.Cross(t.f1) + t.f0.Cross(d1) + d0.Cross(d1);
        const float J = 1.0f + t.area_change, growth = 2.0f * n.Dot(turn) + turn.LengthSq();
        const float moved_sq = J * J + growth;
        const float area_step = n.Dot(n + turn) > 0.0f && moved_sq > 0.0f
            ? growth / (sqrtf(moved_sq) + J) : -1.0f - t.area_change;
        return e.rest[4] * (MembraneChange(m, history, strain, moved, change, h) +
                            ShearModulus<float>(m) * vbd::AreaBarrierChange(t.area_change, area_step));
    }
    if (e.kind == kMfSeam) {
        float change = 0.0f;
        for (uint32_t pair = 0u; pair < (e.hinges >> 4u); ++pair)
            change += SeamChange(e, m, history, pair, SeamSeparation(g, pair), shift[2u * pair + 1u] - shift[2u * pair]);
        return change;
    }
    float before[3] = {0.0f, 0.0f, 0.0f}, after[3] = {0.0f, 0.0f, 0.0f}, step[3] = {0.0f, 0.0f, 0.0f};
    bool exact = true;
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        vbd::Point x[6];
        HingePoints(e, g, i, x);
        Vec3 move[6];
        for (uint32_t slot = 0u; slot < 6u; ++slot) move[slot] = shift[HingeLocal(e, i, slot)];
        const float rest = e.rest[10u + i];
        bool had = false, has = false;
        float from = 0.0f, to = 0.0f, turn = 0.0f;
        HingeMovesChange(x, move, &had, &has, &from, &to, &turn);
        if (had) before[i] = from - rest;
        if (has) after[i] = to - rest;
        if (had && has && to == from + turn) step[i] = turn;
        else exact = false;
    }
    float from[3], to[3], moved[3];
    Curvature(e, before, from);
    Curvature(e, step, moved);
    if (exact) {
        for (uint32_t r = 0u; r < 3u; ++r) to[r] = from[r] + moved[r];
    } else {
        Curvature(e, after, to);
    }
    return e.rest[9] * BendResponseChange(e, m, history, from, to, moved, exact, h);
}

}  // namespace mf

}  // namespace nuka::nk

#undef NUKA_MF_HD
