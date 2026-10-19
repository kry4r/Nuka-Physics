// ---------------------------------------------------------------------------
// nuka::import - Mass properties of collision shapes, summed into a body
// ---------------------------------------------------------------------------

#include "import/geom_inertia.hpp"

#include "import/principal_inertia.hpp"

#include <cstddef>
#include <vector>

namespace nuka::import {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Signed tetrahedra against the origin integrate the closed surface; inside-out winding flips every sign.
MassProperties MeshMassProperties(const scene::CollisionShapeRecord& shape) {
    const std::vector<float>& v = shape.mesh_vertices;
    const std::vector<uint32_t>& f = shape.mesh_indices;
    double volume = 0.0, first[3] = {}, second[3][3] = {};
    for (std::size_t t = 0; t + 2 < f.size(); t += 3) {
        double p[3][3];
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) p[k][c] = v[3u * f[t + k] + c];
        const double det = p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) -
                           p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0]) +
                           p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
        volume += det / 6.0;
        double s[3];
        for (int c = 0; c < 3; ++c) {
            s[c] = p[0][c] + p[1][c] + p[2][c];
            first[c] += det / 24.0 * s[c];
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                second[i][j] += det / 120.0 *
                    (p[0][i] * p[0][j] + p[1][i] * p[1][j] + p[2][i] * p[2][j] + s[i] * s[j]);
    }
    MassProperties out;
    const double sign = volume < 0.0 ? -1.0 : 1.0;
    out.volume = sign * volume;
    if (out.volume <= 0.0) return out;
    const double c[3] = {first[0] / volume, first[1] / volume, first[2] / volume};
    out.centroid = math::Vec3{static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2])};
    double cov[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cov[i][j] = sign * second[i][j] - out.volume * c[i] * c[j];
    const double trace = cov[0][0] + cov[1][1] + cov[2][2];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.inertia[i][j] = (i == j ? trace : 0.0) - cov[i][j];
    return out;
}

} // namespace

MassProperties ShapeMassProperties(const scene::CollisionShapeRecord& shape) {
    if (shape.type == scene::ShapeType::TriMesh) return MeshMassProperties(shape);
    MassProperties out;
    const double r = shape.radius, h = shape.half_height;
    const math::Vec3 e = shape.half_extents;
    double d[3] = {0.0, 0.0, 0.0};
    switch (shape.type) {
    case scene::ShapeType::Sphere:
        out.volume = 4.0 / 3.0 * kPi * r * r * r;
        d[0] = d[1] = d[2] = 0.4 * out.volume * r * r;
        break;
    case scene::ShapeType::Capsule: {
        const double tube = 2.0 * kPi * r * r * h, caps = 4.0 / 3.0 * kPi * r * r * r;
        out.volume = tube + caps;
        d[0] = d[1] = tube * (r * r / 4.0 + h * h / 3.0) + caps * (0.4 * r * r + h * h + 0.75 * h * r);
        d[2] = (tube / 2.0 + caps * 0.4) * r * r;
        break;
    }
    case scene::ShapeType::Cylinder:
        out.volume = 2.0 * kPi * r * r * h;
        d[0] = d[1] = out.volume * (r * r / 4.0 + h * h / 3.0);
        d[2] = out.volume * r * r / 2.0;
        break;
    case scene::ShapeType::Box:
        out.volume = 8.0 * e.x * e.y * e.z;
        d[0] = out.volume * (e.y * e.y + e.z * e.z) / 3.0;
        d[1] = out.volume * (e.x * e.x + e.z * e.z) / 3.0;
        d[2] = out.volume * (e.x * e.x + e.y * e.y) / 3.0;
        break;
    default:
        break;  // planes carry no volume
    }
    for (int i = 0; i < 3; ++i) out.inertia[i][i] = d[i];
    return out;
}

void AddGeomMass(const scene::CollisionShapeRecord& shape, float density, float mass, BodyMass& body) {
    const MassProperties p = ShapeMassProperties(shape);
    if (p.volume <= 0.0) return;
    const double rho = mass < 0.0f ? density : mass / p.volume;
    const double m = rho * p.volume;
    const math::Quat& q = shape.local_transform.rotation;
    const math::Vec3 x = q.Rotate(math::Vec3{1.0f, 0.0f, 0.0f});
    const math::Vec3 y = q.Rotate(math::Vec3{0.0f, 1.0f, 0.0f});
    const math::Vec3 z = q.Rotate(math::Vec3{0.0f, 0.0f, 1.0f});
    const double r[3][3] = {{x.x, y.x, z.x}, {x.y, y.y, z.y}, {x.z, y.z, z.z}};
    const math::Vec3 at = shape.local_transform.position + q.Rotate(p.centroid);
    const double c[3] = {at.x, at.y, at.z};
    const double cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    for (int i = 0; i < 3; ++i) {
        body.first[i] += m * c[i];
        for (int j = 0; j < 3; ++j) {
            double turned = 0.0;
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) turned += r[i][k] * p.inertia[k][l] * r[j][l];
            body.inertia[i][j] += rho * turned + m * ((i == j ? cc : 0.0) - c[i] * c[j]);
        }
    }
    body.mass += m;
}

void SetBodyMass(const BodyMass& sum, scene::RigidBodyRecord& body) {
    body.mass = static_cast<float>(sum.mass);
    body.inertia = math::Vec3::Zero();
    body.inertial_transform = math::Transform::Identity();
    if (sum.mass <= 0.0) return;
    const double c[3] = {sum.first[0] / sum.mass, sum.first[1] / sum.mass, sum.first[2] / sum.mass};
    const double cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    double i[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) i[a][b] = sum.inertia[a][b] - sum.mass * ((a == b ? cc : 0.0) - c[a] * c[b]);
    auto f = [](double value) { return static_cast<float>(value); };
    const float full[6] = {f(i[0][0]), f(i[1][1]), f(i[2][2]), f(i[0][1]), f(i[0][2]), f(i[1][2])};
    DiagonalizeInertia(full, body.inertia, body.inertial_transform.rotation);
    body.inertial_transform.position = math::Vec3{f(c[0]), f(c[1]), f(c[2])};
}

} // namespace nuka::import
