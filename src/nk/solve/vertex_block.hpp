#pragma once
// Vertex-block elastic elements: each vertex takes a 3x3 Newton step against the elements it touches.
// Element energies return the vertex force and a positive semidefinite Hessian block.

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_VBD_HD __host__ __device__
#else
#define NUKA_VBD_HD
#endif

namespace nuka::nk {

// Membrane triangle: rest = inverse material edges (row-major 2x2), area, mu, lambda.
inline constexpr uint32_t kVbdTriangle = 0u;
// Dihedral hinge over edge (v0, v1) with opposite vertices v2, v3: rest = angle, stiffness.
inline constexpr uint32_t kVbdHinge = 1u;
// Axial spring: rest = length, stiffness per length.
inline constexpr uint32_t kVbdSpring = 2u;
// Segment bend of three consecutive vertices: rest = stiffness over mean length.
inline constexpr uint32_t kVbdRodBend = 3u;
inline constexpr uint32_t kVbdNoColor = ~0u;

struct VbdElement {
    uint32_t kind = kVbdSpring;
    uint32_t vertex[4]{~0u, ~0u, ~0u, ~0u};
    float rest[8]{};
    float damping = 0.0f;  // Rayleigh coefficient in seconds on the element stiffness
    uint32_t reserved[2]{};
};
static_assert(sizeof(VbdElement) == 16u * sizeof(uint32_t), "VbdElement packs to 64 bytes");

NUKA_VBD_HD constexpr uint32_t VbdElementVertexCount(uint32_t kind) {
    return kind == kVbdTriangle ? 3u : kind == kVbdHinge ? 4u : kind == kVbdSpring ? 2u : 3u;
}
NUKA_VBD_HD constexpr uint32_t PackVbdIncidence(uint32_t element, uint32_t local) {
    return (element << 2u) | local;
}
NUKA_VBD_HD constexpr uint32_t VbdIncidenceElement(uint32_t packed) { return packed >> 2u; }
NUKA_VBD_HD constexpr uint32_t VbdIncidenceLocal(uint32_t packed) { return packed & 3u; }

namespace vbd {

// A displacement rate includes the integration offset; physical velocity removes it.
NUKA_VBD_HD inline math::Vec3 PhysicalVelocity(math::Vec3 rate, math::Vec3 offset,
                                               float dt, float effective_step) {
    return (rate * dt - offset) / effective_step;
}

// The force equation's momentum defect expressed as a physical velocity error.
NUKA_VBD_HD inline math::Vec3 MomentumResidualVelocity(math::Vec3 force, float inverse_mass,
                                                       float effective_step) {
    return force * (effective_step * inverse_mass);
}

using math::SymmetricMat3;
using math::Vec3;

NUKA_VBD_HD inline float Norm(Vec3 v) { return sqrtf(v.LengthSq()); }

NUKA_VBD_HD inline SymmetricMat3 Outer(Vec3 a, float s) {
    return {s * a.x * a.x, s * a.y * a.y, s * a.z * a.z,
            s * a.x * a.y, s * a.x * a.z, s * a.y * a.z};
}
NUKA_VBD_HD inline void AddTo(SymmetricMat3& m, const SymmetricMat3& b) {
    m.xx += b.xx; m.yy += b.yy; m.zz += b.zz; m.xy += b.xy; m.xz += b.xz; m.yz += b.yz;
}
NUKA_VBD_HD inline void AddIdentity(SymmetricMat3& m, float s) {
    m.xx += s; m.yy += s; m.zz += s;
}
NUKA_VBD_HD inline SymmetricMat3 Scaled(const SymmetricMat3& m, float s) {
    return {m.xx * s, m.yy * s, m.zz * s, m.xy * s, m.xz * s, m.yz * s};
}
NUKA_VBD_HD inline float Determinant(const SymmetricMat3& m) {
    return m.xx * (m.yy * m.zz - m.yz * m.yz) - m.xy * (m.xy * m.zz - m.yz * m.xz) +
           m.xz * (m.xy * m.yz - m.yy * m.xz);
}
// Wider cofactors retain the inertial eigenvalue in anisotropic elastic blocks.
// Rejects nonfinite or below-floor inverses without changing the output.
NUKA_VBD_HD inline bool Invert(const SymmetricMat3& m, float floor, SymmetricMat3* out) {
    const double xx = m.xx, yy = m.yy, zz = m.zz, xy = m.xy, xz = m.xz, yz = m.yz;
    const double cxx = yy * zz - yz * yz, cyy = xx * zz - xz * xz;
    const double czz = xx * yy - xy * xy, cxy = xz * yz - xy * zz;
    const double cxz = xy * yz - xz * yy, cyz = xy * xz - xx * yz;
    const double det = xx * cxx + xy * cxy + xz * cxz;
    if (!(det > floor && det <= DBL_MAX)) return false;
    const double inv = 1.0 / det;
    const double values[6] = {cxx * inv, cyy * inv, czz * inv, cxy * inv, cxz * inv, cyz * inv};
    for (uint32_t i = 0u; i < 6u; ++i)
        if (!(values[i] >= -FLT_MAX && values[i] <= FLT_MAX)) return false;
    *out = {float(values[0]), float(values[1]), float(values[2]),
            float(values[3]), float(values[4]), float(values[5])};
    return true;
}

// Smallest mu with det(a - mu b) = 0 for positive definite a and b, given b's inverse. Below that root the
// monic characteristic cubic of b^-1 a is increasing and concave, so Newton from zero stays a lower bound.
NUKA_VBD_HD inline double SmallestRelativeEigenvalue(const SymmetricMat3& a,
                                                      const SymmetricMat3& b_inverse) {
    const double p[3][3] = {{b_inverse.xx, b_inverse.xy, b_inverse.xz},
                            {b_inverse.xy, b_inverse.yy, b_inverse.yz},
                            {b_inverse.xz, b_inverse.yz, b_inverse.zz}};
    const double q[3][3] = {{a.xx, a.xy, a.xz}, {a.xy, a.yy, a.yz}, {a.xz, a.yz, a.zz}};
    double c[3][3];
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j)
            c[i][j] = p[i][0] * q[0][j] + p[i][1] * q[1][j] + p[i][2] * q[2][j];
    const double s1 = c[0][0] + c[1][1] + c[2][2];
    const double s2 = c[0][0] * c[1][1] - c[0][1] * c[1][0] + c[0][0] * c[2][2] - c[0][2] * c[2][0] +
                      c[1][1] * c[2][2] - c[1][2] * c[2][1];
    const double s3 = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) -
                      c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
                      c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
    double mu = 0.0;
    for (uint32_t k = 0u; k < 16u; ++k) {
        const double slope = (3.0 * mu - 2.0 * s1) * mu + s2;
        if (!(slope > 0.0)) break;
        const double next = mu - (((mu - s1) * mu + s2) * mu - s3) / slope;
        if (!(next >= mu)) break;
        const bool settled = next - mu <= 1.0e-9 * next;
        mu = next;
        if (settled) break;
    }
    return mu;
}

// Vertex coordinates in double for element energy changes that subtract nearby values.
struct Point {
    double x, y, z;
    NUKA_VBD_HD Point operator-(const Point& o) const { return {x - o.x, y - o.y, z - o.z}; }
    NUKA_VBD_HD Point Cross(const Point& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    NUKA_VBD_HD double Dot(const Point& o) const { return x * o.x + y * o.y + z * o.z; }
};

// Element vertices as interval-start positions plus in-step displacements. Start differences
// are exact in double, so a small strain need not be recovered from cancelling unit lengths.
struct ElementGeometry {
    Vec3 start[4];
    Vec3 delta[4];
    NUKA_VBD_HD Vec3 Relative(uint32_t j) const {
        return (start[j] - start[0]) + (delta[j] - delta[0]);
    }
    NUKA_VBD_HD Point Precise(uint32_t j) const {
        return {double(start[j].x) - start[0].x + (double(delta[j].x) - delta[0].x),
                double(start[j].y) - start[0].y + (double(delta[j].y) - delta[0].y),
                double(start[j].z) - start[0].z + (double(delta[j].z) - delta[0].z)};
    }
};

// StVK membrane: F = sum_j x_j b_j^T with b_j the rows of the inverse material edges.
// The strain keeps the two columns of F, the Green strain G = (F^T F - I) / 2 and J - 1.
struct TriangleStrain {
    Vec3 f0, f1;
    float g00, g11, g01;
    float area_change;
};

// The interval-start strain is formed in double from exact start edges; the in-step part
// F_s^T F_d + F_d^T F_s + F_d^T F_d involves only the displacements.
NUKA_VBD_HD inline TriangleStrain MembraneStrain(const VbdElement& e, const ElementGeometry& g) {
    const double a = e.rest[0], b = e.rest[1], c = e.rest[2], d = e.rest[3];
    const double e1[3] = {double(g.start[1].x) - g.start[0].x, double(g.start[1].y) - g.start[0].y,
                          double(g.start[1].z) - g.start[0].z};
    const double e2[3] = {double(g.start[2].x) - g.start[0].x, double(g.start[2].y) - g.start[0].y,
                          double(g.start[2].z) - g.start[0].z};
    double p0[3], p1[3];
    for (uint32_t k = 0u; k < 3u; ++k) {
        p0[k] = e1[k] * a + e2[k] * c;
        p1[k] = e1[k] * b + e2[k] * d;
    }
    const double s00 = 0.5 * (p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2] - 1.0);
    const double s11 = 0.5 * (p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2] - 1.0);
    const double s01 = 0.5 * (p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2]);
    const Vec3 fs0{float(p0[0]), float(p0[1]), float(p0[2])};
    const Vec3 fs1{float(p1[0]), float(p1[1]), float(p1[2])};
    const Vec3 m1 = g.delta[1] - g.delta[0], m2 = g.delta[2] - g.delta[0];
    const Vec3 fd0 = m1 * e.rest[0] + m2 * e.rest[2], fd1 = m1 * e.rest[1] + m2 * e.rest[3];
    TriangleStrain t;
    t.f0 = fs0 + fd0;
    t.f1 = fs1 + fd1;
    t.g00 = float(s00) + (fs0.Dot(fd0) + 0.5f * fd0.Dot(fd0));
    t.g11 = float(s11) + (fs1.Dot(fd1) + 0.5f * fd1.Dot(fd1));
    t.g01 = float(s01) + 0.5f * (fs0.Dot(fd1) + fd0.Dot(fs1) + fd0.Dot(fd1));
    // J^2 = det(I + 2G) = 1 + s, so J - 1 = s / (J + 1) keeps its small value.
    const float s = 2.0f * (t.g00 + t.g11) + 4.0f * (t.g00 * t.g11 - t.g01 * t.g01);
    t.area_change = s > -1.0f ? s / (sqrtf(1.0f + s) + 1.0f) : -1.0f;
    return t;
}

// Weights of vertex `local` in the two columns of F.
NUKA_VBD_HD inline void MembraneWeights(const VbdElement& e, uint32_t local, float* bx, float* by) {
    const float a = e.rest[0], b = e.rest[1], c = e.rest[2], d = e.rest[3];
    *bx = local == 0u ? -(a + c) : local == 1u ? a : c;
    *by = local == 0u ? -(b + d) : local == 1u ? b : d;
}

// Area loss costs mu (J - 1 - ln J) per rest area for the area change d = J - 1, so a compressed
// membrane buckles out of plane instead of collapsing in it; the cost is zero from J = 1 upward.
NUKA_VBD_HD inline float AreaBarrier(float d) {
    if (!(d < 0.0f)) return 0.0f;
    if (!(d > -1.0f)) return 3.0e38f;
    return d > -1.0e-2f ? d * d * (0.5f - d * (1.0f / 3.0f - 0.25f * d)) : d - log1pf(d);
}

// Barrier change as the area change moves from d by s, without subtracting nearby barrier values.
NUKA_VBD_HD inline float AreaBarrierChange(float d, float s) {
    const float moved = d + s;
    if (!(d < 0.0f && d > -1.0f && moved < 0.0f && moved > -1.0f))
        return AreaBarrier(moved) - AreaBarrier(d);
    const float r = s / (1.0f + d);
    const float tail = fabsf(r) < 1.0e-2f ? r * r * (0.5f - r * (1.0f / 3.0f - 0.25f * r))
                                          : r - log1pf(r);
    return s * d / (1.0f + d) + tail;
}

NUKA_VBD_HD inline void TriangleBlock(const VbdElement& e, uint32_t local, const TriangleStrain& t,
                                      Vec3* gradient, SymmetricMat3* hessian) {
    const float area = e.rest[4], mu = e.rest[5], lambda = e.rest[6];
    const Vec3 f0 = t.f0, f1 = t.f1;
    const float g00 = t.g00, g11 = t.g11, g01 = t.g01;
    const float trace = g00 + g11;
    const float s00 = 2.0f * mu * g00 + lambda * trace, s11 = 2.0f * mu * g11 + lambda * trace;
    const float s01 = 2.0f * mu * g01;
    float bx, by;
    MembraneWeights(e, local, &bx, &by);
    *gradient = (f0 * (s00 * bx + s01 * by) + f1 * (s01 * bx + s11 * by)) * area;
    // dJ/dx = w x n^ with w = bx f1 - by f0.
    const Vec3 n = f0.Cross(f1);
    const float normal_length = Norm(n), J = 1.0f + t.area_change;
    const bool compressed = t.area_change < 0.0f && J > 0.0f && normal_length > 0.0f;
    Vec3 q{};
    if (compressed) {
        q = (f1 * bx - f0 * by).Cross(n / normal_length);
        *gradient = *gradient + q * (area * mu * (t.area_change / J));
    }
    const Vec3 fb = f0 * bx + f1 * by;
    const float bsb = bx * (s00 * bx + s01 * by) + by * (s01 * bx + s11 * by);
    SymmetricMat3 h = Outer(fb, mu + lambda);
    AddTo(h, Outer(f0, mu * (bx * bx + by * by)));
    AddTo(h, Outer(f1, mu * (bx * bx + by * by)));
    AddIdentity(h, bsb > 0.0f ? bsb : 0.0f);
    if (compressed) AddTo(h, Outer(q, mu / (J * J)));
    *hessian = Scaled(h, area);
}

// Signed dihedral angle; false for a degenerate hinge, which exerts no force.
NUKA_VBD_HD inline bool HingeAngle(const Vec3* x, float* theta) {
    const Vec3 edge = x[1] - x[0];
    const float edge_sq = edge.LengthSq();
    const Vec3 m1 = edge.Cross(x[2] - x[0]), m2 = (x[3] - x[0]).Cross(edge);
    const float m1_sq = m1.LengthSq(), m2_sq = m2.LengthSq();
    if (!(edge_sq > 0.0f) || !(m1_sq > 0.0f) || !(m2_sq > 0.0f)) return false;
    const Vec3 n1 = m1 / sqrtf(m1_sq), n2 = m2 / sqrtf(m2_sq);
    *theta = atan2f(n1.Cross(n2).Dot(edge) / sqrtf(edge_sq), n1.Dot(n2));
    return true;
}

// The same angle from double coordinates; both atan2 arguments carry the factor |m1| |m2|.
NUKA_VBD_HD inline bool HingeAngle(const Point* x, double* theta) {
    const Point edge = x[1] - x[0];
    const Point m1 = edge.Cross(x[2] - x[0]), m2 = (x[3] - x[0]).Cross(edge);
    const double edge_sq = edge.Dot(edge);
    if (!(edge_sq > 0.0) || !(m1.Dot(m1) > 0.0) || !(m2.Dot(m2) > 0.0)) return false;
    *theta = atan2(m1.Cross(m2).Dot(edge) / sqrt(edge_sq), m1.Dot(m2));
    return true;
}

// The angle gradient uses the opposite-vertex heights and edge projections.
NUKA_VBD_HD inline void HingeBlock(const VbdElement& e, uint32_t local, const Vec3* x,
                                   Vec3* gradient, SymmetricMat3* hessian) {
    *gradient = {};
    *hessian = {};
    float theta = 0.0f;
    if (!HingeAngle(x, &theta)) return;
    const Vec3 edge = x[1] - x[0];
    const float edge_sq = edge.LengthSq();
    const Vec3 m1 = edge.Cross(x[2] - x[0]), m2 = (x[3] - x[0]).Cross(edge);
    const float m1_sq = m1.LengthSq(), m2_sq = m2.LengthSq();
    const float edge_len = sqrtf(edge_sq);
    // d(theta)/dx2 = -n1/h1 with h1 = |m1|/|e|, likewise for x3.
    const Vec3 g2 = m1 * (-edge_len / m1_sq), g3 = m2 * (-edge_len / m2_sq);
    const float s1 = (x[2] - x[0]).Dot(edge) / edge_sq, s2 = (x[3] - x[0]).Dot(edge) / edge_sq;
    const Vec3 g = local == 0u ? g2 * -(1.0f - s1) - g3 * (1.0f - s2)
                 : local == 1u ? g2 * -s1 - g3 * s2
                 : local == 2u ? g2 : g3;
    const float stiffness = e.rest[1];
    *gradient = g * (2.0f * stiffness * (theta - e.rest[0]));
    *hessian = Outer(g, 2.0f * stiffness);
}

NUKA_VBD_HD inline void SpringBlock(const VbdElement& e, uint32_t local, const Vec3* x,
                                    Vec3* gradient, SymmetricMat3* hessian) {
    *gradient = {};
    *hessian = {};
    const Vec3 d = x[1] - x[0];
    const float length = Norm(d);
    if (!(length > 0.0f)) return;
    const Vec3 n = d / length;
    const float k = e.rest[1];
    const Vec3 g = n * (k * (length - e.rest[0]));
    *gradient = local == 0u ? -g : g;
    const float lateral = 1.0f - e.rest[0] / length;
    SymmetricMat3 h = Outer(n, k * (1.0f - (lateral > 0.0f ? lateral : 0.0f)));
    AddIdentity(h, k * (lateral > 0.0f ? lateral : 0.0f));
    *hessian = h;
}

// E = k (1 - cos theta) between consecutive segments; the Hessian is its value at the straight rest.
NUKA_VBD_HD inline void RodBendBlock(const VbdElement& e, uint32_t local, const Vec3* x,
                                     Vec3* gradient, SymmetricMat3* hessian) {
    *gradient = {};
    *hessian = {};
    const Vec3 e1 = x[1] - x[0], e2 = x[2] - x[1];
    const float l1 = Norm(e1), l2 = Norm(e2);
    if (!(l1 > 0.0f) || !(l2 > 0.0f)) return;
    const Vec3 t1 = e1 / l1, t2 = e2 / l2;
    const float cosine = t1.Dot(t2);
    const Vec3 dc1 = (t2 - t1 * cosine) / l1, dc2 = (t1 - t2 * cosine) / l2;
    const float k = e.rest[0];
    if (local == 0u) {
        *gradient = dc1 * k;
        SymmetricMat3 h = Outer(t1, -k / (l1 * l1));
        AddIdentity(h, k / (l1 * l1));
        *hessian = h;
    } else if (local == 1u) {
        *gradient = (dc2 - dc1) * k;
        Vec3 axis = t1 + t2;
        const float axis_len = Norm(axis);
        axis = axis_len > 0.0f ? axis / axis_len : t1;
        const float w = k * (1.0f / l1 + 1.0f / l2) * (1.0f / l1 + 1.0f / l2);
        SymmetricMat3 h = Outer(axis, -w);
        AddIdentity(h, w);
        *hessian = h;
    } else {
        *gradient = dc2 * -k;
        SymmetricMat3 h = Outer(t2, -k / (l2 * l2));
        AddIdentity(h, k / (l2 * l2));
        *hessian = h;
    }
}

NUKA_VBD_HD inline float BendCosine(const Vec3* x) {
    const Vec3 e1 = x[1] - x[0], e2 = x[2] - x[1];
    const float l1 = Norm(e1), l2 = Norm(e2);
    return l1 > 0.0f && l2 > 0.0f ? e1.Dot(e2) / (l1 * l2) : 1.0f;
}

// Energy change of an element when vertex `local` moves by `move`, as products of the change
// so that a small move is not lost to cancellation; a degenerate hinge or bend stores none.
NUKA_VBD_HD inline float ElementEnergyChange(const VbdElement& e, uint32_t local,
                                             const ElementGeometry& g, Vec3 move) {
    if (e.kind == kVbdTriangle) {
        const TriangleStrain t = MembraneStrain(e, g);
        float bx, by;
        MembraneWeights(e, local, &bx, &by);
        const Vec3 d0 = move * bx, d1 = move * by;
        const Vec3 f0 = t.f0 + d0, f1 = t.f1 + d1;
        const float c00 = 0.5f * d0.Dot(t.f0 + f0), c11 = 0.5f * d1.Dot(t.f1 + f1);
        const float c01 = 0.5f * (d0.Dot(f1) + t.f0.Dot(d1));
        const float trace = t.g00 + t.g11, change = c00 + c11;
        // One vertex moves the area normal exactly by move x w; a move that turns it past 90 degrees
        // counts as collapsing the triangle.
        const Vec3 n = t.f0.Cross(t.f1), turn = move.Cross(t.f1 * bx - t.f0 * by);
        const float J = 1.0f + t.area_change, growth = 2.0f * n.Dot(turn) + turn.LengthSq();
        const float moved_sq = J * J + growth;
        const float area_step = n.Dot(n + turn) > 0.0f && moved_sq > 0.0f
            ? growth / (sqrtf(moved_sq) + J) : -1.0f - t.area_change;
        const float barrier = AreaBarrierChange(t.area_change, area_step);
        return e.rest[4] * (e.rest[5] * (c00 * (2.0f * t.g00 + c00) + c11 * (2.0f * t.g11 + c11) +
                                         2.0f * c01 * (2.0f * t.g01 + c01) + barrier) +
                            0.5f * e.rest[6] * change * (2.0f * trace + change));
    }
    if (e.kind == kVbdHinge) {
        // A tolerance-sized move turns the hinge by less than float angles resolve.
        Point x[4], moved[4];
        for (uint32_t j = 0u; j < 4u; ++j) x[j] = moved[j] = g.Precise(j);
        moved[local] = {x[local].x + move.x, x[local].y + move.y, x[local].z + move.z};
        double before = 0.0, after = 0.0;
        const bool had = HingeAngle(x, &before), has = HingeAngle(moved, &after);
        const double k = e.rest[1], rest = e.rest[0];
        if (had && has) return float(k * (after - before) * (after + before - 2.0 * rest));
        return float((has ? k * (after - rest) * (after - rest) : 0.0) -
                     (had ? k * (before - rest) * (before - rest) : 0.0));
    }
    Vec3 x[4], moved[4];
    for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) x[j] = moved[j] = g.Relative(j);
    moved[local] = x[local] + move;
    if (e.kind == kVbdSpring) {
        const float before = Norm(x[1] - x[0]), after = Norm(moved[1] - moved[0]);
        return 0.5f * e.rest[1] * (after - before) * (after + before - 2.0f * e.rest[0]);
    }
    return e.rest[0] * (BendCosine(x) - BendCosine(moved));
}

// Stored elastic energy uses the same constitutive functions as the vertex update.
NUKA_VBD_HD inline float ElementEnergy(const VbdElement& e, const ElementGeometry& g) {
    if (e.kind == kVbdTriangle) {
        const TriangleStrain t = MembraneStrain(e, g);
        const float trace = t.g00 + t.g11;
        return e.rest[4] * (e.rest[5] * (t.g00 * t.g00 + t.g11 * t.g11 +
            2.0f * t.g01 * t.g01 + AreaBarrier(t.area_change)) +
            0.5f * e.rest[6] * trace * trace);
    }
    if (e.kind == kVbdHinge) {
        Point x[4];
        for (uint32_t j = 0u; j < 4u; ++j) x[j] = g.Precise(j);
        double theta = 0.0;
        if (!HingeAngle(x, &theta)) return 0.0f;
        const double delta = theta - e.rest[0];
        return float(e.rest[1] * delta * delta);
    }
    Vec3 x[4];
    for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) x[j] = g.Relative(j);
    if (e.kind == kVbdSpring) {
        const float delta = Norm(x[1] - x[0]) - e.rest[0];
        return 0.5f * e.rest[1] * delta * delta;
    }
    return e.rest[0] * (1.0f - BendCosine(x));
}

// Energy gradient and Hessian block of one element at vertex `local`.
NUKA_VBD_HD inline void ElementBlock(const VbdElement& e, uint32_t local, const ElementGeometry& g,
                                     Vec3* gradient, SymmetricMat3* hessian) {
    if (e.kind == kVbdTriangle) {
        TriangleBlock(e, local, MembraneStrain(e, g), gradient, hessian);
        return;
    }
    Vec3 x[4];
    for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) x[j] = g.Relative(j);
    switch (e.kind) {
        case kVbdHinge: HingeBlock(e, local, x, gradient, hessian); break;
        case kVbdSpring: SpringBlock(e, local, x, gradient, hessian); break;
        default: RodBendBlock(e, local, x, gradient, hessian); break;
    }
}

}  // namespace vbd

}  // namespace nuka::nk

#undef NUKA_VBD_HD
