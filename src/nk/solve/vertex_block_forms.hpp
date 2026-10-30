#pragma once
// Gauss-Newton bilinear forms of vertex-block elements and their energy change when every vertex moves;
// vertex_block.hpp holds the single-vertex functions.

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "nk/solve/vertex_block.hpp"
#include "math/spectral3.hpp"

#if defined(__CUDACC__)
#define NUKA_VBD_HD __host__ __device__
#else
#define NUKA_VBD_HD
#endif

namespace nuka::nk::vbd {

struct TetCurvature {
    Vec3 u[3], v[3], f[3];
    SymmetricMat3 scaling;
    float spin[3]{}, shear[3]{};
    bool viscous = false;
};

// The scaling block and the three twist/flip pairs form the complete isotropic 9x9 Hessian.
NUKA_VBD_HD inline TetCurvature TetModes(const VbdElement& e, const TetStrain& t) {
    TetCurvature c;
    const TetRest r(e); const auto svd = math::ProperSvd3(t.f);
    const float l = r.lambda + r.mu, stress = l * t.volume_change - r.mu;
    float q[3];
    for (uint32_t i = 0; i < 3; ++i) {
        c.u[i] = svd.u[i]; c.v[i] = svd.v[i];
        q[i] = svd.sigma[(i + 1) % 3] * svd.sigma[(i + 2) % 3];
        const float other = svd.sigma[2 - i];
        c.spin[i] = fmaxf(r.mu * (1.0f - other) + l * t.volume_change * other, 0.0f);
        c.shear[i] = fmaxf(r.mu - stress * other, 0.0f);
    }
    c.scaling = math::PositivePart3({r.mu + l * q[0] * q[0], r.mu + l * q[1] * q[1],
        r.mu + l * q[2] * q[2], l * q[0] * q[1] + stress * svd.sigma[2],
        l * q[0] * q[2] + stress * svd.sigma[1], l * q[1] * q[2] + stress * svd.sigma[0]});
    return c;
}

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
// start. Triangles keep F, the clamped stress, the unit normal, the area-barrier factor and the tangent.
struct ElementCurvature {
    TetCurvature tet;
    Vec3 f0, f1, normal;
    float stress[3] = {};
    float barrier = 0.0f;
    MembraneTangent tangent;
    // A Neo-Hookean triangle keeps its Hessian modes in F = U S V^T: `stress` holds the scaling block, then
    // twist, flip and the two out-of-plane weights; a collapsed one (not `modal`) keeps mu |dF|^2.
    bool neo_hookean = false, modal = false;
    Vec3 u0, u1;
    float cosine = 1.0f, sine = 0.0f, twist = 0.0f, flip = 0.0f, out[2] = {};
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
    Vec3 a, b, c;
    float g00 = 0.0f, g11 = 0.0f, g01 = 0.0f, s = 0.0f;
};

// The Neo-Hookean membrane Hessian by its analytic eigensystem (Smith et al. 2019), each mode clamped to its
// nonnegative part: with s = dpsi/dJ, twist mu + s, flip mu - s and out-of-plane mu + s S_other / S_own.
NUKA_VBD_HD inline void NeoHookeanModes(float mu, float lambda, const TriangleStrain& t, ElementCurvature* c) {
    const Vec3 n = t.f0.Cross(t.f1);
    const float length = Norm(n), J = 1.0f + t.area_change;
    if (!(length > 0.0f && J > 0.0f)) return;
    const float c00 = t.f0.Dot(t.f0), c01 = t.f0.Dot(t.f1), c11 = t.f1.Dot(t.f1);
    const float half = 0.5f * (c00 - c11);
    const float high = 0.5f * (c00 + c11) + sqrtf(half * half + c01 * c01);
    const float s1 = sqrtf(high), s2 = J / s1, s = (lambda + mu) * t.area_change - mu;
    const float angle = 0.5f * atan2f(c01, half);
    c->cosine = cosf(angle);
    c->sine = sinf(angle);
    c->u0 = (t.f0 * c->cosine + t.f1 * c->sine) / s1;
    c->u1 = (t.f1 * c->cosine - t.f0 * c->sine) / s2;
    c->normal = n / length;
    PositiveStress(mu + (lambda + mu) * s2 * s2, (lambda + mu) * J + s, mu + (lambda + mu) * high, c->stress);
    // mu + s = (lambda + mu) dA, so the weights near rest are formed without cancelling mu.
    const float growth = (lambda + mu) * t.area_change, gap = mu * s1 * (s1 - s2);
    c->twist = fmaxf(growth, 0.0f);
    c->flip = fmaxf(2.0f * mu - growth, 0.0f);
    c->out[0] = fmaxf((gap + growth * J) / high, 0.0f);
    c->out[1] = fmaxf((growth * high - gap) / J, 0.0f);
    c->modal = true;
}

// The Gauss-Newton curvature of ElementBlock, with the full stress clamped to its positive part; Neo-Hookean
// triangles take their projected Hessian.
NUKA_VBD_HD inline ElementCurvature GaussNewtonCurvature(const VbdElement& e, const ElementGeometry& g,
                                                         const MembraneStartState* start = nullptr) {
    ElementCurvature c;
    if (e.kind == kVbdTet) { c.tet = TetModes(e, TetrahedronStrain(e, g)); return c; }
    if (e.kind == kVbdTriangle) {
        const TriangleStrain t = start != nullptr ? MembraneStrain(e, g, *start) : MembraneStrain(e, g);
        const float mu = e.rest[5], lambda = e.rest[6], trace = t.g00 + t.g11;
        c.f0 = t.f0;
        c.f1 = t.f1;
        if (NeoHookean(e)) {
            c.neo_hookean = true;
            NeoHookeanModes(mu, lambda, t, &c);
            return c;
        }
        if (e.rest[7] > 0.0f) {
            c.tangent = SoftenedTangent(e.rest[7], t.g00, t.g11, t.g01);
            float s[3];
            SoftenedStress(c.tangent, mu, lambda, t.g00, t.g11, t.g01, s);
            PositiveStress(s[0], s[2], s[1], c.stress);
        } else {
            PositiveStress(2.0f * mu * t.g00 + lambda * trace, 2.0f * mu * t.g01,
                           2.0f * mu * t.g11 + lambda * trace, c.stress);
        }
        const Vec3 n = t.f0.Cross(t.f1);
        const float length = Norm(n), J = 1.0f + t.area_change;
        if (length > 0.0f) c.normal = n / length;
        if (t.area_change < 0.0f && J > 0.0f && length > 0.0f) c.barrier = mu * (1.0f - e.rest[7]) / (J * J);
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
    if (e.kind == kVbdTet) {
        const auto t = TetrahedronStrain(e, start);
        for (uint32_t i = 0; i < 3; ++i) c.tet.f[i] = t.f[i];
        c.tet.viscous = true;
        return c;
    }
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
        if (e.rest[7] > 0.0f) c.tangent = SoftenedTangent(e.rest[7], t.g00, t.g11, t.g01);
        if (t.area_change < 0.0f && !NeoHookean(e)) c.barrier = e.rest[5] * (1.0f - e.rest[7]) / (J * J);
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
    if (e.kind == kVbdTet) {
        Vec3 f[3]; TetRest(e).Columns(x, f);
        v.a = f[0]; v.b = f[1]; v.c = f[2];
        return v;
    }
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

// A variation's coordinates along the Neo-Hookean modes: the two scalings, twist, flip and out-of-plane.
NUKA_VBD_HD inline void NeoHookeanModeCoordinates(const ElementCurvature& c, const ElementVariation& x,
                                                  float (&m)[6]) {
    const Vec3 g0 = x.a * c.cosine + x.b * c.sine, g1 = x.b * c.cosine - x.a * c.sine;
    const float w10 = c.u1.Dot(g0), w01 = c.u0.Dot(g1), r = 0.70710678118654752f;
    m[0] = c.u0.Dot(g0);
    m[1] = c.u1.Dot(g1);
    m[2] = (w10 - w01) * r;
    m[3] = (w10 + w01) * r;
    m[4] = c.normal.Dot(g0);
    m[5] = c.normal.Dot(g1);
}

// The element's symmetric bilinear form between two displacement fields given by their variations.
NUKA_VBD_HD inline float ElementForm(const VbdElement& e, const ElementCurvature& c,
                                     const ElementVariation& x, const ElementVariation& y) {
    if (e.kind == kVbdTet) {
        const TetRest r(e); const auto& t = c.tet;
        const Vec3 dx[3] = {x.a, x.b, x.c}, dy[3] = {y.a, y.b, y.c};
        if (t.viscous) {
            float product = 0.0f;
            for (uint32_t i = 0; i < 3; ++i) for (uint32_t j = 0; j < 3; ++j)
                product += (t.f[i].Dot(dx[j]) + dx[i].Dot(t.f[j])) *
                           (t.f[i].Dot(dy[j]) + dy[i].Dot(t.f[j]));
            return 0.5f * r.Volume() * r.mu * product;
        }
        float mx[3][3], my[3][3];
        for (uint32_t j = 0; j < 3; ++j) {
            const Vec3 a = dx[0] * t.v[j].x + dx[1] * t.v[j].y + dx[2] * t.v[j].z;
            const Vec3 b = dy[0] * t.v[j].x + dy[1] * t.v[j].y + dy[2] * t.v[j].z;
            for (uint32_t i = 0; i < 3; ++i) { mx[i][j] = t.u[i].Dot(a); my[i][j] = t.u[i].Dot(b); }
        }
        float product = Vec3{mx[0][0], mx[1][1], mx[2][2]}.Dot(
            t.scaling.Multiply({my[0][0], my[1][1], my[2][2]}));
        uint32_t pair = 0;
        for (uint32_t i = 0; i < 2; ++i) for (uint32_t j = i + 1; j < 3; ++j, ++pair)
            product += 0.5f * (t.spin[pair] * (mx[i][j] - mx[j][i]) * (my[i][j] - my[j][i]) +
                               t.shear[pair] * (mx[i][j] + mx[j][i]) * (my[i][j] + my[j][i]));
        return r.Volume() * product;
    }
    if (e.kind == kVbdTriangle) {
        const float mu = e.rest[5], lambda = e.rest[6];
        if (c.neo_hookean) {
            if (!c.modal) return e.rest[4] * mu * (x.a.Dot(y.a) + x.b.Dot(y.b));
            float mx[6], my[6];
            NeoHookeanModeCoordinates(c, x, mx);
            NeoHookeanModeCoordinates(c, y, my);
            return e.rest[4] * (c.stress[0] * mx[0] * my[0] + c.stress[1] * (mx[0] * my[1] + mx[1] * my[0]) +
                                c.stress[2] * mx[1] * my[1] + c.twist * mx[2] * my[2] + c.flip * mx[3] * my[3] +
                                c.out[0] * mx[4] * my[4] + c.out[1] * mx[5] * my[5]);
        }
        if (e.rest[7] > 0.0f) {
            float s[3];
            SoftenedStress(c.tangent, mu, lambda, y.g00, y.g11, y.g01, s);
            return e.rest[4] * (x.g00 * s[0] + x.g11 * s[1] + 2.0f * x.g01 * s[2] + c.stress[0] * x.a.Dot(y.a) +
                                c.stress[2] * x.b.Dot(y.b) + c.stress[1] * (x.a.Dot(y.b) + x.b.Dot(y.a)) +
                                c.barrier * x.s * y.s);
        }
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
    if (e.kind == kVbdTet) {
        const auto t = TetrahedronStrain(e, g); Vec3 df[3]; TetRest(e).Columns(move, df);
        const float dj = t.cof[0].Dot(df[0]) + t.cof[1].Dot(df[1]) + t.cof[2].Dot(df[2]) +
            t.f[0].Dot(df[1].Cross(df[2])) + df[0].Dot(t.f[1].Cross(df[2]) + df[1].Cross(t.f[2])) +
            df[0].Dot(df[1].Cross(df[2]));
        return TetEnergyChange(e, t, df, dj);
    }
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
        if (NeoHookean(e))
            return e.rest[4] * NeoHookeanEnergyChange(e, t.area_change, change,
                                                      NeoHookeanAreaStep(n, turn, t.area_change));
        const float J = 1.0f + t.area_change, growth = 2.0f * n.Dot(turn) + turn.LengthSq();
        const float moved_sq = J * J + growth;
        const float area_step = n.Dot(n + turn) > 0.0f && moved_sq > 0.0f
            ? growth / (sqrtf(moved_sq) + J) : -1.0f - t.area_change;
        const float barrier = AreaBarrierChange(t.area_change, area_step);
        if (e.rest[7] > 0.0f) return e.rest[4] * SoftenedEnergyChange(e, t, c00, c11, c01, barrier);
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
