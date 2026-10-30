#pragma once

#include <cfloat>
#include <cmath>

#include "math/symmetric_mat3.hpp"

#if defined(__CUDACC__)
#define NUKA_SPECTRAL_HD __host__ __device__
#else
#define NUKA_SPECTRAL_HD
#endif

namespace nuka::math {

struct SymmetricEigen3 {
    Vec3 axis[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float value[3]{};
};

// Jacobi rotations of a scaled symmetric tensor; columns are its orthonormal eigenvectors.
NUKA_SPECTRAL_HD inline SymmetricEigen3 EigenSymmetric3(const SymmetricMat3& m) {
    SymmetricEigen3 e;
    const float scale = fmaxf(fmaxf(fabsf(m.xx), fabsf(m.yy)), fmaxf(fabsf(m.zz),
        fmaxf(fabsf(m.xy), fmaxf(fabsf(m.xz), fabsf(m.yz)))));
    if (!(scale > 0.0f)) return e;
    float a[3][3] = {{m.xx / scale, m.xy / scale, m.xz / scale},
                     {m.xy / scale, m.yy / scale, m.yz / scale},
                     {m.xz / scale, m.yz / scale, m.zz / scale}};
    for (unsigned step = 0; step < 24; ++step) {
        unsigned p = 0, q = 1;
        if (fabsf(a[0][2]) > fabsf(a[p][q])) { p = 0; q = 2; }
        if (fabsf(a[1][2]) > fabsf(a[p][q])) { p = 1; q = 2; }
        if (!(fabsf(a[p][q]) > FLT_EPSILON)) break;
        const float angle = 0.5f * atan2f(2.0f * a[p][q], a[q][q] - a[p][p]);
        const float c = cosf(angle), s = sinf(angle), ap = a[p][p], aq = a[q][q], b = a[p][q];
        a[p][p] = c * c * ap - 2.0f * c * s * b + s * s * aq;
        a[q][q] = s * s * ap + 2.0f * c * s * b + c * c * aq;
        a[p][q] = a[q][p] = 0.0f;
        for (unsigned k = 0; k < 3; ++k) if (k != p && k != q) {
            const float x = a[k][p], y = a[k][q];
            a[k][p] = a[p][k] = c * x - s * y;
            a[k][q] = a[q][k] = s * x + c * y;
        }
        const Vec3 x = e.axis[p], y = e.axis[q];
        e.axis[p] = x * c - y * s;
        e.axis[q] = x * s + y * c;
    }
    for (unsigned i = 0; i < 3; ++i) e.value[i] = a[i][i] * scale;
    for (unsigned i = 0; i < 2; ++i) for (unsigned j = i + 1; j < 3; ++j)
        if (e.value[j] > e.value[i]) {
            const float d = e.value[i]; e.value[i] = e.value[j]; e.value[j] = d;
            const Vec3 v = e.axis[i]; e.axis[i] = e.axis[j]; e.axis[j] = v;
        }
    e.axis[2] = e.axis[0].Cross(e.axis[1]);
    return e;
}

NUKA_SPECTRAL_HD inline SymmetricMat3 PositivePart3(const SymmetricMat3& m) {
    const auto e = EigenSymmetric3(m);
    SymmetricMat3 r;
    for (unsigned i = 0; i < 3; ++i) {
        const Vec3 v = e.axis[i];
        const float w = fmaxf(e.value[i], 0.0f);
        r.xx += w * v.x * v.x; r.yy += w * v.y * v.y; r.zz += w * v.z * v.z;
        r.xy += w * v.x * v.y; r.xz += w * v.x * v.z; r.yz += w * v.y * v.z;
    }
    return r;
}

struct SignedSvd3 {
    Vec3 u[3], v[3];
    float sigma[3];
};

NUKA_SPECTRAL_HD inline Vec3 Perpendicular3(Vec3 n) {
    const Vec3 a = fabsf(n.x) <= fabsf(n.y) && fabsf(n.x) <= fabsf(n.z) ? Vec3{1, 0, 0}
        : fabsf(n.y) <= fabsf(n.z) ? Vec3{0, 1, 0} : Vec3{0, 0, 1};
    const Vec3 p = n.Cross(a);
    return p / sqrtf(p.LengthSq());
}

// Both frames are proper rotations; the final singular value carries an inversion's sign.
NUKA_SPECTRAL_HD inline SignedSvd3 ProperSvd3(const Vec3 (&f)[3]) {
    const auto e = EigenSymmetric3({f[0].Dot(f[0]), f[1].Dot(f[1]), f[2].Dot(f[2]),
                                    f[0].Dot(f[1]), f[0].Dot(f[2]), f[1].Dot(f[2])});
    SignedSvd3 r;
    Vec3 fv[3];
    for (unsigned i = 0; i < 3; ++i) {
        r.v[i] = e.axis[i];
        fv[i] = f[0] * r.v[i].x + f[1] * r.v[i].y + f[2] * r.v[i].z;
    }
    r.sigma[0] = sqrtf(fv[0].LengthSq());
    r.u[0] = r.sigma[0] > 0.0f ? fv[0] / r.sigma[0] : Vec3{1, 0, 0};
    const Vec3 b = fv[1] - r.u[0] * r.u[0].Dot(fv[1]);
    r.sigma[1] = sqrtf(b.LengthSq());
    r.u[1] = r.sigma[1] > 8.0f * FLT_EPSILON * r.sigma[0] ? b / r.sigma[1] : Perpendicular3(r.u[0]);
    r.u[2] = r.u[0].Cross(r.u[1]);
    r.sigma[2] = r.u[2].Dot(fv[2]);
    return r;
}

}  // namespace nuka::math

#undef NUKA_SPECTRAL_HD
