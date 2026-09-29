#pragma once
// Vertex-block elastic elements: each vertex takes a 3x3 Newton step against the elements it touches.
// Element energies return the vertex force and a positive semidefinite Hessian block.

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
// Returns false when the determinant is below `floor`; the inverse is then left unchanged.
NUKA_VBD_HD inline bool Invert(const SymmetricMat3& m, float floor, SymmetricMat3* out) {
    const float cxx = m.yy * m.zz - m.yz * m.yz, cyy = m.xx * m.zz - m.xz * m.xz;
    const float czz = m.xx * m.yy - m.xy * m.xy, cxy = m.xz * m.yz - m.xy * m.zz;
    const float cxz = m.xy * m.yz - m.xz * m.yy, cyz = m.xy * m.xz - m.xx * m.yz;
    const float det = m.xx * cxx + m.xy * cxy + m.xz * cxz;
    if (!(det > floor)) return false;
    const float inv = 1.0f / det;
    *out = {cxx * inv, cyy * inv, czz * inv, cxy * inv, cxz * inv, cyz * inv};
    return true;
}

// StVK membrane: F = sum_j x_j b_j^T with b_j the rows of the inverse material edges.
NUKA_VBD_HD inline void TriangleBlock(const VbdElement& e, uint32_t local, const Vec3* x,
                                      Vec3* gradient, SymmetricMat3* hessian) {
    const float a = e.rest[0], b = e.rest[1], c = e.rest[2], d = e.rest[3];
    const float area = e.rest[4], mu = e.rest[5], lambda = e.rest[6];
    const Vec3 d1 = x[1] - x[0], d2 = x[2] - x[0];
    const Vec3 f0 = d1 * a + d2 * c, f1 = d1 * b + d2 * d;
    const float g00 = 0.5f * (f0.Dot(f0) - 1.0f), g11 = 0.5f * (f1.Dot(f1) - 1.0f);
    const float g01 = 0.5f * f0.Dot(f1);
    const float trace = g00 + g11;
    const float s00 = 2.0f * mu * g00 + lambda * trace, s11 = 2.0f * mu * g11 + lambda * trace;
    const float s01 = 2.0f * mu * g01;
    const float bx = local == 0u ? -(a + c) : local == 1u ? a : c;
    const float by = local == 0u ? -(b + d) : local == 1u ? b : d;
    *gradient = (f0 * (s00 * bx + s01 * by) + f1 * (s01 * bx + s11 * by)) * area;
    const Vec3 fb = f0 * bx + f1 * by;
    const float bsb = bx * (s00 * bx + s01 * by) + by * (s01 * bx + s11 * by);
    SymmetricMat3 h = Outer(fb, mu + lambda);
    AddTo(h, Outer(f0, mu * (bx * bx + by * by)));
    AddTo(h, Outer(f1, mu * (bx * bx + by * by)));
    AddIdentity(h, bsb > 0.0f ? bsb : 0.0f);
    *hessian = Scaled(h, area);
}

// Signed dihedral angle; the gradient uses the opposite-vertex heights and edge projections.
NUKA_VBD_HD inline void HingeBlock(const VbdElement& e, uint32_t local, const Vec3* x,
                                   Vec3* gradient, SymmetricMat3* hessian) {
    *gradient = {};
    *hessian = {};
    const Vec3 edge = x[1] - x[0];
    const float edge_sq = edge.LengthSq();
    const Vec3 m1 = edge.Cross(x[2] - x[0]), m2 = (x[3] - x[0]).Cross(edge);
    const float m1_sq = m1.LengthSq(), m2_sq = m2.LengthSq();
    if (!(edge_sq > 0.0f) || !(m1_sq > 0.0f) || !(m2_sq > 0.0f)) return;
    const float edge_len = sqrtf(edge_sq);
    const Vec3 n1 = m1 / sqrtf(m1_sq), n2 = m2 / sqrtf(m2_sq);
    const float theta = atan2f(n1.Cross(n2).Dot(edge) / edge_len, n1.Dot(n2));
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

// Energy gradient and Hessian block of one element at vertex `local`.
NUKA_VBD_HD inline void ElementBlock(const VbdElement& e, uint32_t local, const Vec3* x,
                                     Vec3* gradient, SymmetricMat3* hessian) {
    switch (e.kind) {
        case kVbdTriangle: TriangleBlock(e, local, x, gradient, hessian); break;
        case kVbdHinge: HingeBlock(e, local, x, gradient, hessian); break;
        case kVbdSpring: SpringBlock(e, local, x, gradient, hessian); break;
        default: RodBendBlock(e, local, x, gradient, hessian); break;
    }
}

}  // namespace vbd

}  // namespace nuka::nk

#undef NUKA_VBD_HD
