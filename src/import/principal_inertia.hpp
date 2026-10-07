#pragma once
// ---------------------------------------------------------------------------
// nuka::import - Principal axes of an imported inertia tensor
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/vec3.hpp"

#include <cmath>

namespace nuka::import {

// Build a unit quaternion (w-first) from a row-major rotation matrix whose
// columns are orthonormal axes. Shepperd's method; matches
// articulation_state RotationMatrixFromQuat so R = V reconstructs R*diag*R^T.
inline math::Quat QuatFromMatrix(const double m[3][3]) {
    const double tr = m[0][0] + m[1][1] + m[2][2];
    math::Quat q;
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        q.w = static_cast<float>(0.25 * s);
        q.x = static_cast<float>((m[2][1] - m[1][2]) / s);
        q.y = static_cast<float>((m[0][2] - m[2][0]) / s);
        q.z = static_cast<float>((m[1][0] - m[0][1]) / s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q.w = static_cast<float>((m[2][1] - m[1][2]) / s);
        q.x = static_cast<float>(0.25 * s);
        q.y = static_cast<float>((m[0][1] + m[1][0]) / s);
        q.z = static_cast<float>((m[0][2] + m[2][0]) / s);
    } else if (m[1][1] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q.w = static_cast<float>((m[0][2] - m[2][0]) / s);
        q.x = static_cast<float>((m[0][1] + m[1][0]) / s);
        q.y = static_cast<float>(0.25 * s);
        q.z = static_cast<float>((m[1][2] + m[2][1]) / s);
    } else {
        const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q.w = static_cast<float>((m[1][0] - m[0][1]) / s);
        q.x = static_cast<float>((m[0][2] + m[2][0]) / s);
        q.y = static_cast<float>((m[1][2] + m[2][1]) / s);
        q.z = static_cast<float>(0.25 * s);
    }
    return q.Normalized();
}

// Diagonalize a symmetric inertia tensor [Ixx Iyy Izz Ixy Ixz Iyz] (MuJoCo
// fullinertia, body frame) into principal moments + a body->principal rotation
// whose matrix columns are the principal axes. Jacobi sweeps (proper rotations,
// det +1), so the returned rotation feeds inertial_transform directly.
inline void DiagonalizeInertia(const float full[6], math::Vec3& diag, math::Quat& rot) {
    double a[3][3] = {
        {full[0], full[3], full[4]},
        {full[3], full[1], full[5]},
        {full[4], full[5], full[2]},
    };
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // eigenvector columns.
    for (int sweep = 0; sweep < 32; ++sweep) {
        const double off =
            std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
        if (off < 1e-20) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    diag = math::Vec3{static_cast<float>(a[0][0]), static_cast<float>(a[1][1]),
                      static_cast<float>(a[2][2])};
    rot = QuatFromMatrix(v);
}

} // namespace nuka::import
