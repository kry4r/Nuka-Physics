#pragma once
// Gauss-Newton bilinear forms of vertex-block elements and their energy change when every vertex moves;
// vertex_block.hpp holds the single-vertex functions.

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "nk/solve/vertex_block.hpp"

#if defined(__CUDACC__)
#define NUKA_VBD_HD __host__ __device__
#else
#define NUKA_VBD_HD
#endif

namespace nuka::nk::vbd {

// Positive semidefinite part of the symmetric 2x2 stress (s00, s01, s11).
NUKA_VBD_HD inline void PositiveStress(float s00, float s01, float s11, float* out) {
    const float mean = 0.5f * (s00 + s11), half = 0.5f * (s00 - s11);
    const float radius = sqrtf(half * half + s01 * s01);
    const float high = mean + radius, low = mean - radius;
    if (!(low < 0.0f)) {
        out[0] = s00;
        out[1] = s01;
        out[2] = s11;
    } else if (!(high > 0.0f)) {
        out[0] = out[1] = out[2] = 0.0f;
    } else {
        const float scale = high / (high - low);
        out[0] = scale * (s00 - low);
        out[1] = scale * s01;
        out[2] = scale * (s11 - low);
    }
}

// One element's curvature: Gauss-Newton at the current geometry or the Rayleigh metric of the interval
// start. Triangles keep F, the clamped stress, the unit normal and the area-barrier factor.
struct ElementCurvature {
    Vec3 f0, f1, normal;
    float stress[3] = {};
    float barrier = 0.0f;
    // Hinges and bent rods act through angle gradients, springs along `normal` with a lateral share and
    // straight rods through their segment projectors with the segment-order sign.
    Vec3 q[4];
    Vec3 t1, t2;
    float l1 = 0.0f, l2 = 0.0f, sign = 1.0f;
    float lateral = 0.0f;
    float stiffness = 0.0f;
    bool angle = false;
};

// First-order change of an element's strain measures under per-vertex displacements.
struct ElementVariation {
    Vec3 a, b;
    float g00 = 0.0f, g11 = 0.0f, g01 = 0.0f, s = 0.0f;
};

// The Gauss-Newton curvature of ElementBlock, with the full stress clamped to its positive part.
NUKA_VBD_HD inline ElementCurvature GaussNewtonCurvature(const VbdElement& e, const ElementGeometry& g,
                                                         const MembraneStartState* start = nullptr) {
    ElementCurvature c;
    if (e.kind == kVbdTriangle) {
        const TriangleStrain t = start != nullptr ? MembraneStrain(e, g, *start) : MembraneStrain(e, g);
        const float mu = e.rest[5], lambda = e.rest[6], trace = t.g00 + t.g11;
        c.f0 = t.f0;
        c.f1 = t.f1;
        PositiveStress(2.0f * mu * t.g00 + lambda * trace, 2.0f * mu * t.g01,
                       2.0f * mu * t.g11 + lambda * trace, c.stress);
        const Vec3 n = t.f0.Cross(t.f1);
        const float length = Norm(n), J = 1.0f + t.area_change;
        if (length > 0.0f) c.normal = n / length;
        if (t.area_change < 0.0f && J > 0.0f && length > 0.0f) c.barrier = mu / (J * J);
        return c;
    }
    Vec3 x[4];
    for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) x[j] = g.Relative(j);
    if (e.kind == kVbdHinge) {
        c.angle = HingeAngleGradient(x, c.q);
        c.stiffness = c.angle ? 2.0f * e.rest[1] : 0.0f;
        return c;
    }
    if (e.kind == kVbdSpring) {
        const Vec3 d = x[1] - x[0];
        const float length = Norm(d);
        if (!(length > 0.0f)) return c;
        const float lateral = 1.0f - e.rest[0] / length;
        c.normal = d / length;
        c.lateral = lateral > 0.0f ? lateral : 0.0f;
        c.stiffness = e.rest[1];
        return c;
    }
    const Vec3 e1 = x[1] - x[0], e2 = x[2] - x[1];
    const float l1 = Norm(e1), l2 = Norm(e2);
    if (!(l1 > 0.0f) || !(l2 > 0.0f)) return c;
    c.t1 = e1 / l1;
    c.t2 = e2 / l2;
    c.l1 = l1;
    c.l2 = l2;
    c.stiffness = e.rest[0];
    return c;
}

// The metric of ElementRayleighBlock at the interval start; a collapsed start membrane is invalid.
NUKA_VBD_HD inline ElementCurvature RayleighCurvature(const VbdElement& e, const ElementGeometry& start,
                                                      const MembraneStartState* cached = nullptr) {
    ElementCurvature c;
    if (e.kind == kVbdTriangle) {
        const TriangleStrain t = cached != nullptr ? MembraneStrain(e, start, *cached)
                                                   : MembraneStrain(e, start);
        c.f0 = t.f0;
        c.f1 = t.f1;
        const Vec3 n = t.f0.Cross(t.f1);
        const float length = Norm(n), J = 1.0f + t.area_change;
        if (!(J > 0.0f && length > 0.0f)) {
            c.barrier = NAN;
            return c;
        }
        c.normal = n / length;
        if (t.area_change < 0.0f) c.barrier = e.rest[5] / (J * J);
        return c;
    }
    Vec3 x[4];
    for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) x[j] = start.Relative(j);
    if (e.kind == kVbdHinge) {
        c.angle = HingeAngleGradient(x, c.q);
        c.stiffness = c.angle ? 2.0f * e.rest[1] : 0.0f;
        return c;
    }
    if (e.kind == kVbdSpring) {
        const Vec3 d = x[1] - x[0];
        const float length = Norm(d);
        if (!(length > 0.0f)) return c;
        c.normal = d / length;
        c.stiffness = e.rest[1];
        return c;
    }
    const Vec3 e1 = x[1] - x[0], e2 = x[2] - x[1];
    const float l1 = Norm(e1), l2 = Norm(e2);
    if (!(l1 > 0.0f) || !(l2 > 0.0f)) return c;
    c.t1 = e1 / l1;
    c.t2 = e2 / l2;
    c.l1 = l1;
    c.l2 = l2;
    c.stiffness = e.rest[0];
    const Vec3 axis = c.t1.Cross(c.t2);
    const float sine = Norm(axis);
    if (sine > 0.0f) {
        const Vec3 n = axis / sine;
        c.q[0] = n.Cross(c.t1) / l1;
        c.q[2] = n.Cross(c.t2) / l2;
        c.q[1] = -c.q[0] - c.q[2];
        c.angle = true;
    } else {
        c.sign = c.t1.Dot(c.t2) >= 0.0f ? 1.0f : -1.0f;
    }
    return c;
}

// Variations of the displacement field x, taken relative to vertex 0 against a large common motion.
NUKA_VBD_HD inline ElementVariation Vary(const VbdElement& e, const ElementCurvature& c, const Vec3* x) {
    ElementVariation v;
    if (e.kind == kVbdTriangle) {
        const Vec3 x1 = x[1] - x[0], x2 = x[2] - x[0];
        v.a = x1 * e.rest[0] + x2 * e.rest[2];
        v.b = x1 * e.rest[1] + x2 * e.rest[3];
        v.g00 = c.f0.Dot(v.a);
        v.g11 = c.f1.Dot(v.b);
        v.g01 = 0.5f * (c.f0.Dot(v.b) + c.f1.Dot(v.a));
        v.s = c.normal.Dot(v.a.Cross(c.f1) + c.f0.Cross(v.b));
        return v;
    }
    if (c.angle) {
        for (uint32_t j = 1u; j < VbdElementVertexCount(e.kind); ++j) v.s += c.q[j].Dot(x[j] - x[0]);
        return v;
    }
    if (e.kind == kVbdSpring) {
        v.a = x[1] - x[0];
        return v;
    }
    if (e.kind == kVbdRodBend && c.l1 > 0.0f && c.l2 > 0.0f) {
        const Vec3 u1 = x[1] - x[0], u2 = x[2] - x[1];
        v.a = (u1 - c.t1 * c.t1.Dot(u1)) / c.l1 - (u2 - c.t2 * c.t2.Dot(u2)) * (c.sign / c.l2);
    }
    return v;
}

// The element's symmetric bilinear form between two displacement fields given by their variations.
NUKA_VBD_HD inline float ElementForm(const VbdElement& e, const ElementCurvature& c,
                                     const ElementVariation& x, const ElementVariation& y) {
    if (e.kind == kVbdTriangle) {
        const float mu = e.rest[5], lambda = e.rest[6];
        return e.rest[4] * (2.0f * mu * (x.g00 * y.g00 + x.g11 * y.g11 + 2.0f * x.g01 * y.g01) +
                            lambda * (x.g00 + x.g11) * (y.g00 + y.g11) + c.stress[0] * x.a.Dot(y.a) +
                            c.stress[2] * x.b.Dot(y.b) + c.stress[1] * (x.a.Dot(y.b) + x.b.Dot(y.a)) +
                            c.barrier * x.s * y.s);
    }
    if (c.angle) return c.stiffness * x.s * y.s;
    if (e.kind == kVbdSpring)
        return c.stiffness * ((1.0f - c.lateral) * c.normal.Dot(x.a) * c.normal.Dot(y.a) +
                              c.lateral * x.a.Dot(y.a));
    return c.stiffness * x.a.Dot(y.a);
}

// HingeAngleChange with every vertex moving: the edge and both normal products change by products of
// the moves relative to vertex 0.
NUKA_VBD_HD inline void HingeMovesChange(const Vec3* x, const Vec3* move, bool* had, bool* has,
                                         float* before, float* after, float* turn) {
    const Vec3 edge = x[1] - x[0];
    const float edge_sq = edge.LengthSq();
    *had = *has = false;
    if (!(edge_sq > 0.0f) || !(edge_sq <= FLT_MAX)) return;
    int exponent = 0;
    frexpf(sqrtf(edge_sq), &exponent);
    const float scale = ldexpf(1.0f, -exponent);
    const Vec3 e = edge * scale, a = (x[2] - x[0]) * scale, b = (x[3] - x[0]) * scale;
    const Vec3 de = (move[1] - move[0]) * scale, da = (move[2] - move[0]) * scale;
    const Vec3 db = (move[3] - move[0]) * scale;
    HingeTurn(e, e.Cross(a), b.Cross(e), de, de.Cross(a) + e.Cross(da) + de.Cross(da),
              db.Cross(e) + b.Cross(de) + db.Cross(de), had, has, before, after, turn);
}

// Drop of a bend's cosine as its vertices move, from products of the moves.
NUKA_VBD_HD inline float BendCosineDrop(const Vec3* x, const Vec3* move) {
    const Vec3 e1 = x[1] - x[0], e2 = x[2] - x[1], u1 = move[1] - move[0], u2 = move[2] - move[1];
    const Vec3 f1 = e1 + u1, f2 = e2 + u2;
    const float l1 = Norm(e1), l2 = Norm(e2), k1 = Norm(f1), k2 = Norm(f2);
    if (!(l1 > 0.0f && l2 > 0.0f && k1 > 0.0f && k2 > 0.0f)) {
        const float before = l1 > 0.0f && l2 > 0.0f ? e1.Dot(e2) / (l1 * l2) : 1.0f;
        return before - (k1 > 0.0f && k2 > 0.0f ? f1.Dot(f2) / (k1 * k2) : 1.0f);
    }
    const float d1 = (2.0f * e1.Dot(u1) + u1.Dot(u1)) / (k1 + l1);
    const float d2 = (2.0f * e2.Dot(u2) + u2.Dot(u2)) / (k2 + l2);
    const float turn = e1.Dot(u2) + u1.Dot(e2) + u1.Dot(u2);
    return (e1.Dot(e2) * (d1 * l2 + l1 * d2 + d1 * d2) - turn * (l1 * l2)) / ((l1 * l2) * (k1 * k2));
}

// Energy change of an element whose vertices move by `move`, as products of the move so that a small
// move is not lost to cancellation.
NUKA_VBD_HD inline float ElementMovesEnergyChange(const VbdElement& e, const ElementGeometry& g,
                                                  const Vec3* move,
                                                  const MembraneStartState* start = nullptr) {
    if (e.kind == kVbdTriangle) {
        const TriangleStrain t = start != nullptr ? MembraneStrain(e, g, *start) : MembraneStrain(e, g);
        const Vec3 m1 = move[1] - move[0], m2 = move[2] - move[0];
        const Vec3 d0 = m1 * e.rest[0] + m2 * e.rest[2], d1 = m1 * e.rest[1] + m2 * e.rest[3];
        const Vec3 f0 = t.f0 + d0, f1 = t.f1 + d1;
        const float c00 = 0.5f * d0.Dot(t.f0 + f0), c11 = 0.5f * d1.Dot(t.f1 + f1);
        const float c01 = 0.5f * (d0.Dot(f1) + t.f0.Dot(d1));
        const float trace = t.g00 + t.g11, change = c00 + c11;
        // The area normal moves by f0 x d1 + d0 x f1'; a move turning it past 90 degrees collapses it.
        const Vec3 n = t.f0.Cross(t.f1), turn = t.f0.Cross(d1) + d0.Cross(f1);
        const float J = 1.0f + t.area_change, growth = 2.0f * n.Dot(turn) + turn.LengthSq();
        const float moved_sq = J * J + growth;
        const float area_step = n.Dot(n + turn) > 0.0f && moved_sq > 0.0f
            ? growth / (sqrtf(moved_sq) + J) : -1.0f - t.area_change;
        const float barrier = AreaBarrierChange(t.area_change, area_step);
        return e.rest[4] * (e.rest[5] * (c00 * (2.0f * t.g00 + c00) + c11 * (2.0f * t.g11 + c11) +
                                         2.0f * c01 * (2.0f * t.g01 + c01) + barrier) +
                            0.5f * e.rest[6] * change * (2.0f * trace + change));
    }
    Vec3 x[4];
    for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) x[j] = g.Relative(j);
    if (e.kind == kVbdHinge) {
        bool had = false, has = false;
        float before = 0.0f, after = 0.0f, turn = 0.0f;
        HingeMovesChange(x, move, &had, &has, &before, &after, &turn);
        const float k = e.rest[1], rest = e.rest[0];
        if (had && has && after == before + turn) return k * turn * (turn + 2.0f * (before - rest));
        return (has ? k * (after - rest) * (after - rest) : 0.0f) -
               (had ? k * (before - rest) * (before - rest) : 0.0f);
    }
    if (e.kind == kVbdSpring) {
        const Vec3 a = x[1] - x[0], d = move[1] - move[0];
        const float before = Norm(a), after = Norm(a + d), sum = before + after;
        const float growth = sum > 0.0f ? (2.0f * a.Dot(d) + d.Dot(d)) / sum : 0.0f;
        return 0.5f * e.rest[1] * growth * (sum - 2.0f * e.rest[0]);
    }
    return e.rest[0] * BendCosineDrop(x, move);
}

}  // namespace nuka::nk::vbd

#undef NUKA_VBD_HD
