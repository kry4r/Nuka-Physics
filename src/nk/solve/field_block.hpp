#pragma once
// Material-field elements inside the vertex blocks: each element's gradient, Hessian block and energy change
// at one vertex, from the response contract and the vertex-block membrane and hinge-angle kinematics.

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

// Triangle: corners, inverse material edges (row-major 2x2) and area. Bend: corners, then the far vertex of
// each hinge in `hinges` (bit i: the hinge opposite corner i); hinge-space plate stiffness and rest angles.
struct MfElement {
    uint32_t kind = kMfTriangle;
    uint32_t material = 0u;
    uint32_t vertex[6]{~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    uint32_t hinges = 0u;
    uint32_t history = ~0u;
    float rest[14]{};
};
static_assert(sizeof(MfElement) == 24u * sizeof(uint32_t), "MfElement packs to 96 bytes");

NUKA_MF_HD constexpr uint32_t MfElementVertexCount(const MfElement& e) {
    return e.kind == kMfTriangle ? 3u : 3u + (e.hinges & 1u) + ((e.hinges >> 1u) & 1u) + ((e.hinges >> 2u) & 1u);
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

// s (a b^T + b a^T).
NUKA_MF_HD inline SymmetricMat3 SymmetricProduct(Vec3 a, Vec3 b, float s) {
    return {2.0f * s * a.x * b.x, 2.0f * s * a.y * b.y, 2.0f * s * a.z * b.z,
            s * (a.x * b.y + a.y * b.x), s * (a.x * b.z + a.z * b.x), s * (a.y * b.z + a.z * b.y)};
}

NUKA_MF_HD inline void TriangleBlock(const MfElement& e, const MfMaterial& m, const float* history, uint32_t local,
                                     const vbd::TriangleStrain& t, float h, Vec3* gradient, SymmetricMat3* hessian) {
    MembraneResult<float> r{};
    const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
    EvaluateMembrane(m, history, strain, h, r);
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

// The turn gradients of vertex `local` meet the moments; the Hessian block is G K G^T over those gradients.
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
        turn[i] = theta - e.rest[6u + i];
        const uint32_t slot = HingeSlot(e, i, local);
        if (slot < 4u) q[i] = dq[slot];
    }
    BendResult<float> r{};
    EvaluateBend(m, history, turn, e.rest, h, r);
    const float* k = r.tangent;
    *gradient = q[0] * r.moment[0] + q[1] * r.moment[1] + q[2] * r.moment[2];
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
    BendBlock(e, m, history, local, g, h, gradient, hessian);
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
        const float rest = e.rest[6u + i];
        const uint32_t slot = HingeSlot(e, i, local);
        if (slot == 4u) {
            float theta = 0.0f;
            if (vbd::HingeAngle(x, &theta)) before[i] = after[i] = theta - rest;
            continue;
        }
        bool had = false, has = false;
        float from = 0.0f, to = 0.0f, turn = 0.0f;
        vbd::HingeAngleChange(x, slot, move, &had, &has, &from, &to, &turn);
        if (had) before[i] = from - rest;
        if (has) after[i] = to - rest;
        if (had && has && to == from + turn) step[i] = turn;
        else exact = false;
    }
    BendResult<float> from{}, to{};
    EvaluateBend(m, history, before, e.rest, h, from);
    if (exact) {
        float response[3];
        TangentProduct(from.tangent, step, response);
        return step[0] * (from.moment[0] + 0.5f * response[0]) + step[1] * (from.moment[1] + 0.5f * response[1]) +
               step[2] * (from.moment[2] + 0.5f * response[2]);
    }
    EvaluateBend(m, history, after, e.rest, h, to);
    return to.psi - from.psi;
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
        const float change[3] = {0.5f * d0.Dot(t.f0 + f0), 0.5f * d1.Dot(t.f1 + f1),
                                 d0.Dot(f1) + t.f0.Dot(d1)};
        const Vec3 n = t.f0.Cross(t.f1), turn = move.Cross(t.f1 * bx - t.f0 * by);
        const float J = 1.0f + t.area_change, growth = 2.0f * n.Dot(turn) + turn.LengthSq();
        const float moved_sq = J * J + growth;
        const float area_step = n.Dot(n + turn) > 0.0f && moved_sq > 0.0f
            ? growth / (sqrtf(moved_sq) + J) : -1.0f - t.area_change;
        MembraneResult<float> r{};
        const float strain[3] = {t.g00, t.g11, 2.0f * t.g01};
        EvaluateMembrane(m, history, strain, h, r);
        float response[3];
        TangentProduct(r.tangent, change, response);
        const float membrane = change[0] * (r.stress[0] + 0.5f * response[0]) +
                               change[1] * (r.stress[1] + 0.5f * response[1]) +
                               change[2] * (r.stress[2] + 0.5f * response[2]);
        return e.rest[4] * (membrane + ShearModulus<float>(m) * vbd::AreaBarrierChange(t.area_change, area_step));
    }
    return BendEnergyChange(e, m, history, local, g, h, move);
}

// Stored energy with the same response as the vertex update; hinge angles come from double coordinates.
NUKA_MF_HD inline double ElementEnergy(const MfElement& e, const MfMaterial& m, const float* history,
                                       const FieldGeometry& g, float h) {
    if (e.kind == kMfTriangle) {
        const vbd::TriangleStrain t = Strain(e, g.Corners(), nullptr);
        MembraneResult<double> r{};
        const double strain[3] = {t.g00, t.g11, 2.0 * double(t.g01)};
        EvaluateMembrane(m, history, strain, double(h), r);
        return double(e.rest[4]) * (r.psi + ShearModulus<double>(m) * vbd::AreaBarrier(t.area_change));
    }
    double turn[3] = {0.0, 0.0, 0.0};
    for (uint32_t i = 0u; i < 3u; ++i) {
        if (!((e.hinges >> i) & 1u)) continue;
        const vbd::Point x[4] = {g.Precise((i + 1u) % 3u), g.Precise((i + 2u) % 3u), g.Precise(i),
                                 g.Precise(FarVertex(e, i))};
        double theta = 0.0;
        if (vbd::HingeAngle(x, &theta)) turn[i] = theta - double(e.rest[6u + i]);
    }
    BendResult<double> r{};
    EvaluateBend(m, history, turn, e.rest, double(h), r);
    return r.psi;
}

// A triangle collapsed through zero area has no defined energy.
NUKA_MF_HD inline bool ElementStateValid(const MfElement& e, const FieldGeometry& g,
                                         const vbd::MembraneStartState* start) {
    return e.kind != kMfTriangle || Strain(e, g.Corners(), start).area_change > -1.0f;
}

}  // namespace mf

}  // namespace nuka::nk

#undef NUKA_MF_HD
