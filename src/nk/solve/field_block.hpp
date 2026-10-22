#pragma once
// Material-field elements inside the vertex blocks: each element's gradient, Hessian block and energy change
// at one vertex, from the plane-stress law and the vertex-block membrane and hinge-angle kinematics.

#include <cfloat>
#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "math/vec3.hpp"
#include "nk/material/field_response.hpp"
#include "nk/solve/vertex_block.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MF_HD __host__ __device__
#else
#define NUKA_MF_HD
#endif

namespace nuka::nk {

inline constexpr uint32_t kMfTriangle = 0u;
inline constexpr uint32_t kMfBend = 1u;
// Node pairs (this side, far side) across a declared edge, then its two bends in vertex[4..5]; `hinges` packs each
// bend's hinge over the edge in two bits and the pair count from bit 4; rest holds the pair weight L t / 2 and K.
inline constexpr uint32_t kMfSeam = 2u;

// Triangle: corners, inverse material edges (row-major 2x2) and area. Bend: corners, then the far vertex of
// each hinge in `hinges` (bit i: the hinge opposite corner i); the hinge-turn to Voigt-curvature map
// (row-major 3x3), area and rest angles.
struct MfElement {
    uint32_t kind = kMfTriangle;
    uint32_t material = 0u;
    uint32_t vertex[6]{~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    uint32_t hinges = 0u;
    float rest[15]{};
};
static_assert(sizeof(MfElement) == 24u * sizeof(uint32_t), "MfElement packs to 96 bytes");

NUKA_MF_HD constexpr uint32_t MfElementVertexCount(const MfElement& e) {
    return e.kind == kMfTriangle ? 3u
         : e.kind == kMfSeam     ? 2u * (e.hinges >> 4u)
                                 : 3u + (e.hinges & 1u) + ((e.hinges >> 1u) & 1u) + ((e.hinges >> 2u) & 1u);
}

// A field incidence packs its element with the vertex's slot 0..5 in it.
NUKA_MF_HD constexpr uint32_t PackMfIncidence(uint32_t element, uint32_t local) { return element << 3u | local; }
NUKA_MF_HD constexpr uint32_t MfIncidenceElement(uint32_t packed) { return packed >> 3u; }
NUKA_MF_HD constexpr uint32_t MfIncidenceLocal(uint32_t packed) { return packed & 7u; }

namespace mf {

using math::SymmetricMat3;
using math::Vec3;

// Element vertices as interval-start positions plus in-step displacements, as in the vertex blocks.
struct FieldGeometry {
    Vec3 start[6];
    Vec3 delta[6];
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
    return {double(e.rest[1]), double(m.tear_strength), double(m.tear_energy)};
}

NUKA_MF_HD inline double SeamReached(const float* history, uint32_t pair) {
    return history != nullptr ? double(history[pair]) : 0.0;
}

// Separation of node pair `pair` from this side to the far side.
NUKA_MF_HD inline Vec3 SeamSeparation(const FieldGeometry& g, uint32_t pair) {
    return g.Relative(2u * pair + 1u) - g.Relative(2u * pair);
}

// A pair pulls its sides together along their separation; the secant T / r keeps the block positive.
NUKA_MF_HD inline void SeamBlock(const MfElement& e, const MfMaterial& m, const float* history, uint32_t local,
                                 const FieldGeometry& g, Vec3* gradient, SymmetricMat3* hessian) {
    const uint32_t pair = local >> 1u;
    const Vec3 delta = SeamSeparation(g, pair);
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
        energy += EvaluateSeam(SeamLaw(e, m), SeamReached(history, pair), sqrt(dx * dx + dy * dy + dz * dz)).stored;
    }
    return double(e.rest[0]) * energy;
}

// A seam's trial keeps each pair's largest separation; its cohesive loss takes the plastic slot.
NUKA_MF_HD inline MfStatus SeamTrial(const MfElement& e, const MfMaterial& m, const float* history,
                                     const FieldGeometry& g, FieldResponse<float>& out) {
    FieldResponse<float> r{};
    double psi = 0.0, stored = 0.0, loss = 0.0;
    for (uint32_t pair = 0u; pair < (e.hinges >> 4u); ++pair) {
        const SeamResponse s =
            EvaluateSeam(SeamLaw(e, m), SeamReached(history, pair), vbd::Norm(SeamSeparation(g, pair)));
        r.history[pair] = float(s.reached);
        psi += s.psi;
        stored += s.stored;
        loss += s.loss;
    }
    r.psi = float(psi);
    r.stored = float(stored);
    r.plastic = float(loss);
    out = r;
    return MfStatus::Ok;
}

// Hinge i runs from corner i + 1 to corner i + 2, with corner i on this side and its far vertex beyond.
NUKA_MF_HD inline uint32_t FarVertex(const MfElement& e, uint32_t i) {
    return 3u + (i > 0u ? (e.hinges & 1u) : 0u) + (i > 1u ? ((e.hinges >> 1u) & 1u) : 0u);
}

// Element vertex `local` in hinge i's (edge start, edge end, near, far) order, or 4 when outside it.
NUKA_MF_HD inline uint32_t HingeSlot(const MfElement& e, uint32_t i, uint32_t local) {
    if (local == (i + 1u) % 3u) return 0u;
    if (local == (i + 2u) % 3u) return 1u;
    if (local == i) return 2u;
    return local == FarVertex(e, i) ? 3u : 4u;
}

NUKA_MF_HD inline void HingePoints(const MfElement& e, const FieldGeometry& g, uint32_t i, Vec3 (&x)[4]) {
    x[0] = g.Relative((i + 1u) % 3u);
    x[1] = g.Relative((i + 2u) % 3u);
    x[2] = g.Relative(i);
    x[3] = g.Relative(FarVertex(e, i));
}

// Voigt curvature (11, 22, 2 12) of a bend from its hinge turns.
template <class Real> NUKA_MF_HD inline void Curvature(const MfElement& e, const Real (&turn)[3], Real (&kappa)[3]) {
    for (uint32_t r = 0u; r < 3u; ++r)
        kappa[r] = Real(e.rest[3u * r]) * turn[0] + Real(e.rest[3u * r + 1u]) * turn[1] +
                   Real(e.rest[3u * r + 2u]) * turn[2];
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
    EvaluatePlaneStress(MembraneLaw(m), history, strain, h, r);
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
        Vec3 x[4], dq[4];
        float theta = 0.0f;
        if (!((e.hinges >> i) & 1u)) continue;
        HingePoints(e, g, i, x);
        if (!vbd::HingeAngleGradient(x, dq, &theta)) continue;
        const float keep = HingeKeep(history, i);
        turn[i] = keep * (theta - e.rest[10u + i]);
        const uint32_t slot = HingeSlot(e, i, local);
        if (slot < 4u) q[i] = dq[slot] * keep;
    }
    float kappa[3], moment[3], k[6];
    Curvature(e, turn, kappa);
    FieldResponse<float> r{};
    EvaluatePlaneStress(BendLaw(m), history, kappa, h, r);
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
                                       const float (&to)[3], const float (&step)[3], bool exact, float h) {
    FieldResponse<float> a{}, b{};
    EvaluatePlaneStress(law, history, from, h, a);
    if (law.yield > 0.0f || !exact) EvaluatePlaneStress(law, history, to, h, b);
    if (!exact || a.yielded || b.yielded) return b.psi - a.psi;
    float response[3];
    TangentProduct(a.tangent, step, response);
    return step[0] * (a.stress[0] + 0.5f * response[0]) + step[1] * (a.stress[1] + 0.5f * response[1]) +
           step[2] * (a.stress[2] + 0.5f * response[2]);
}

// Turns of the hinges before and after vertex `local` moves; `step` keeps a small turn exact.
NUKA_MF_HD inline float BendEnergyChange(const MfElement& e, const MfMaterial& m, const float* history,
                                         uint32_t local, const FieldGeometry& g, float h, Vec3 move) {
    float before[3] = {0.0f, 0.0f, 0.0f}, after[3] = {0.0f, 0.0f, 0.0f}, step[3] = {0.0f, 0.0f, 0.0f};
    bool exact = true;
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        Vec3 x[4];
        HingePoints(e, g, i, x);
        const float rest = e.rest[10u + i], keep = HingeKeep(history, i);
        const uint32_t slot = HingeSlot(e, i, local);
        if (slot == 4u) {
            float theta = 0.0f;
            if (vbd::HingeAngle(x, &theta)) before[i] = after[i] = keep * (theta - rest);
            continue;
        }
        bool had = false, has = false;
        float from = 0.0f, to = 0.0f, turn = 0.0f;
        vbd::HingeAngleChange(x, slot, move, &had, &has, &from, &to, &turn);
        if (had) before[i] = keep * (from - rest);
        if (has) after[i] = keep * (to - rest);
        if (had && has && to == from + turn) step[i] = keep * turn;
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
    return e.rest[9] * ResponseChange(BendLaw(m), history, from, to, moved, exact, h);
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
        return e.rest[4] * (ResponseChange(MembraneLaw(m), history, strain, moved, change, true, h) +
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
        return double(e.rest[4]) * (StoredEnergy(MembraneLaw(m), history, strain) +
                                    ShearModulus<double>(m) * vbd::AreaBarrier(t.area_change));
    }
    if (e.kind == kMfSeam) return SeamEnergy(e, m, history, g);
    double turn[3] = {0.0, 0.0, 0.0}, kappa[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        const vbd::Point x[4] = {g.Precise((i + 1u) % 3u), g.Precise((i + 2u) % 3u), g.Precise(i),
                                 g.Precise(FarVertex(e, i))};
        double theta = 0.0;
        if (vbd::HingeAngle(x, &theta)) turn[i] = double(HingeKeep(history, i)) * (theta - double(e.rest[10u + i]));
    }
    Curvature(e, turn, kappa);
    return double(e.rest[9]) * StoredEnergy(BendLaw(m), history, kappa);
}

// The per-area response at the state a step reached, whose history the step commits.
NUKA_MF_HD inline MfStatus ElementTrial(const MfElement& e, const MfMaterial& m, const float* history,
                                        const FieldGeometry& g, float h, FieldResponse<float>& out) {
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), nullptr);
        const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
        return EvaluatePlaneStress(MembraneLaw(m), history, strain, h, out);
    }
    if (e.kind == kMfSeam) return SeamTrial(e, m, history, g, out);
    float turn[3] = {0.0f, 0.0f, 0.0f}, kappa[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        Vec3 x[4];
        HingePoints(e, g, i, x);
        float theta = 0.0f;
        if (vbd::HingeAngle(x, &theta)) turn[i] = HingeKeep(history, i) * (theta - e.rest[10u + i]);
    }
    Curvature(e, turn, kappa);
    return EvaluatePlaneStress(BendLaw(m), history, kappa, h, out);
}

// A triangle collapsed through zero area has no defined energy.
NUKA_MF_HD inline bool ElementStateValid(const MfElement& e, const FieldGeometry& g,
                                         const vbd::MembraneStartState* start) {
    return e.kind != kMfTriangle || Strain(e, g.Corners(), start).area_change > -1.0f;
}

// Gauss-Newton factor of an element: generalized strain rows over its vertices and their area-scaled stiffness,
// the membrane strain and a compressed area for a triangle, the three hinge turns for a bend. A seam's pairs share
// their coarse moves, which leave it unstrained.
struct FieldCurvature {
    Vec3 row[4][6];
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
        EvaluatePlaneStress(MembraneLaw(m), history, strain, h, r);
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
            Vec3 x[4], dq[4];
            float theta = 0.0f;
            if (!((e.hinges >> i) & 1u)) continue;
            HingePoints(e, g, i, x);
            if (!vbd::HingeAngleGradient(x, dq, &theta)) continue;
            const float keep = HingeKeep(history, i);
            turn[i] = keep * (theta - e.rest[10u + i]);
            for (uint32_t local = 0u; local < 6u; ++local) {
                const uint32_t slot = HingeSlot(e, i, local);
                if (slot < 4u) c.row[i][local] = dq[slot] * keep;
            }
        }
        Curvature(e, turn, kappa);
        EvaluatePlaneStress(BendLaw(m), history, kappa, h, r);
        HingeSpace(e, r, moment, k);
    }
    const float s[3][3] = {{k[0], k[3], k[4]}, {k[3], k[1], k[5]}, {k[4], k[5], k[2]}};
    for (uint32_t a = 0u; a < 3u; ++a)
        for (uint32_t b = 0u; b < 3u; ++b) c.stiffness[a][b] = s[a][b];
    return c;
}

// Generalized strain variations under per-vertex moves, and their stiffness product.
NUKA_MF_HD inline void Vary(const FieldCurvature& c, const Vec3 (&move)[6], float (&out)[4]) {
    for (uint32_t a = 0u; a < 4u; ++a) {
        out[a] = 0.0f;
        for (uint32_t j = 0u; j < 6u; ++j) out[a] += c.row[a][j].Dot(move[j]);
    }
}

NUKA_MF_HD inline float Form(const FieldCurvature& c, const float (&x)[4], const float (&y)[4]) {
    float sum = 0.0f;
    for (uint32_t a = 0u; a < 4u; ++a)
        for (uint32_t b = 0u; b < 4u; ++b) sum += x[a] * c.stiffness[a][b] * y[b];
    return sum;
}

// Energy change as all vertices move by `shift`: the membrane strain change is formed of products of the moves,
// and the hinge turns are measured in double before and after.
NUKA_MF_HD inline float ElementMovesEnergyChange(const MfElement& e, const MfMaterial& m, const float* history,
                                                const FieldGeometry& g, const vbd::MembraneStartState* start,
                                                float h, const Vec3 (&shift)[6]) {
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
        return e.rest[4] * (ResponseChange(MembraneLaw(m), history, strain, moved, change, true, h) +
                            ShearModulus<float>(m) * vbd::AreaBarrierChange(t.area_change, area_step));
    }
    if (e.kind == kMfSeam) {
        float change = 0.0f;
        for (uint32_t pair = 0u; pair < (e.hinges >> 4u); ++pair)
            change += SeamChange(e, m, history, pair, SeamSeparation(g, pair), shift[2u * pair + 1u] - shift[2u * pair]);
        return change;
    }
    double before[3] = {0.0, 0.0, 0.0}, after[3] = {0.0, 0.0, 0.0};
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        const uint32_t v[4] = {(i + 1u) % 3u, (i + 2u) % 3u, i, FarVertex(e, i)};
        vbd::Point x[4], y[4];
        for (uint32_t k = 0u; k < 4u; ++k) {
            x[k] = g.Precise(v[k]);
            y[k] = {x[k].x + (double(shift[v[k]].x) - shift[0].x), x[k].y + (double(shift[v[k]].y) - shift[0].y),
                    x[k].z + (double(shift[v[k]].z) - shift[0].z)};
        }
        double theta = 0.0;
        const double keep = HingeKeep(history, i);
        if (vbd::HingeAngle(x, &theta)) before[i] = keep * (theta - double(e.rest[10u + i]));
        if (vbd::HingeAngle(y, &theta)) after[i] = keep * (theta - double(e.rest[10u + i]));
    }
    double from_turn[3], to_turn[3];
    Curvature(e, before, from_turn);
    Curvature(e, after, to_turn);
    float from[3], step[3], to[3];
    for (uint32_t k = 0u; k < 3u; ++k) {
        from[k] = float(from_turn[k]);
        step[k] = float(to_turn[k] - from_turn[k]);
        to[k] = from[k] + step[k];
    }
    return e.rest[9] * ResponseChange(BendLaw(m), history, from, to, step, true, h);
}

}  // namespace mf

}  // namespace nuka::nk

#undef NUKA_MF_HD
