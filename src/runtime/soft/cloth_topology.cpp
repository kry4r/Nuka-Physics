#include "runtime/soft/cloth_topology.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace nuka::runtime::soft {

namespace {

bool FiniteNonnegative(float value) { return std::isfinite(value) && value >= 0.0f; }

// Signed dihedral angle over edge (x0, x1), matching the vertex-block hinge energy.
float DihedralAngle(math::Vec3 x0, math::Vec3 x1, math::Vec3 x2, math::Vec3 x3) {
    const math::Vec3 edge = x1 - x0;
    const math::Vec3 m1 = edge.Cross(x2 - x0), m2 = (x3 - x0).Cross(edge);
    const float edge_len = edge.Length(), m1_len = m1.Length(), m2_len = m2.Length();
    if (!(edge_len > 0.0f) || !(m1_len > 0.0f) || !(m2_len > 0.0f)) return 0.0f;
    const math::Vec3 n1 = m1 / m1_len, n2 = m2 / m2_len;
    return std::atan2(n1.Cross(n2).Dot(edge) / edge_len, n1.Dot(n2));
}

using Point2 = std::array<double, 2>;

double Cross2(Point2 a, Point2 b) { return a[0] * b[1] - a[1] * b[0]; }
double Dot2(Point2 a, Point2 b) { return a[0] * b[0] + a[1] * b[1]; }
Point2 Sub2(Point2 a, Point2 b) { return {a[0] - b[0], a[1] - b[1]}; }

// A uniform draw in (0, 1) fixed by an edge's two welded ends.
double EdgeUniform(uint32_t a, uint32_t b) {
    uint64_t h = (uint64_t{std::min(a, b)} << 32u | std::max(a, b)) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 31u;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 29u;
    return (double(h >> 11u) + 0.5) / 9007199254740992.0;
}

// Rest corners of a face in its own plane: the pattern face when given, else the 3D rest triangle.
std::array<Point2, 3> FaceLayout(const std::vector<math::Vec3>& rest_positions, const ClothTriangle& tri,
                                 const std::vector<std::array<math::Vec3, 3>>& material_faces, size_t f) {
    if (!material_faces.empty()) {
        const auto& face = material_faces[f];
        return {Point2{0.0, 0.0}, Point2{double(face[1].x) - face[0].x, double(face[1].y) - face[0].y},
                Point2{double(face[2].x) - face[0].x, double(face[2].y) - face[0].y}};
    }
    const math::Vec3 e1 = rest_positions[tri.v[1]] - rest_positions[tri.v[0]];
    const math::Vec3 e2 = rest_positions[tri.v[2]] - rest_positions[tri.v[0]];
    const math::Vec3 normal = e1.Cross(e2), t = e1 / e1.Length(), b = normal.Cross(t) / normal.Length();
    return {Point2{0.0, 0.0}, Point2{e1.Length(), 0.0}, Point2{e2.Dot(t), e2.Dot(b)}};
}

// Turn of a flat hinge (edge x0 -> x1, near x2, far x3) under the height (kxx x^2 + 2 kxy x y + kyy y^2) / 2,
// as a row over (kxx, kxy, kyy), through the out-of-plane part of the vertex-block angle gradient.
std::array<double, 3> TurnResponse(const std::array<Point2, 4>& x) {
    const Point2 e = Sub2(x[1], x[0]), a = Sub2(x[2], x[0]), b = Sub2(x[3], x[0]);
    const double length = std::sqrt(Dot2(e, e)), g2 = -length / Cross2(e, a), g3 = -length / Cross2(b, e);
    const double s1 = Dot2(a, e) / Dot2(e, e), s2 = Dot2(b, e) / Dot2(e, e);
    const double q[4] = {-(1.0 - s1) * g2 - (1.0 - s2) * g3, -s1 * g2 - s2 * g3, g2, g3};
    std::array<double, 3> row{};
    for (uint32_t k = 0u; k < 4u; ++k) {
        row[0] += 0.5 * q[k] * x[k][0] * x[k][0];
        row[1] += q[k] * x[k][0] * x[k][1];
        row[2] += 0.5 * q[k] * x[k][1] * x[k][1];
    }
    return row;
}

}  // namespace

nk::VbdElement VertexBlockSpring(uint32_t a, uint32_t b, float rest_length, float stiffness,
                                 float damping) {
    if (!(rest_length > 0.0f) || !std::isfinite(rest_length) || !FiniteNonnegative(stiffness))
        throw std::invalid_argument("Vertex-block spring needs a positive rest length");
    nk::VbdElement e;
    e.kind = nk::kVbdSpring;
    e.vertex[0] = a;
    e.vertex[1] = b;
    e.rest[0] = rest_length;
    e.rest[1] = stiffness / rest_length;
    e.damping = damping;
    return e;
}

void BuildClothVertexBlocks(const std::vector<math::Vec3>& rest_positions,
                            const std::vector<ClothTriangle>& triangles,
                            const std::vector<std::array<math::Vec3, 3>>& material_faces,
                            const std::vector<ShellMaterial>& materials,
                            std::vector<nk::VbdElement>& out) {
    if (materials.size() != triangles.size())
        throw std::invalid_argument("Cloth shell materials must match its triangles");
    for (const ShellMaterial& material : materials)
        if (!FiniteNonnegative(material.stretch_stiffness) || !FiniteNonnegative(material.bend_stiffness) ||
            !FiniteNonnegative(material.damping) || !std::isfinite(material.poisson) ||
            material.poisson < 0.0f || material.poisson >= 0.5f ||
            !(material.compression_ratio > 0.0f && material.compression_ratio <= 1.0f))
            throw std::invalid_argument(
                "Cloth shell material needs nonnegative stiffness, 0 <= poisson < 0.5 and 0 < compression ratio <= 1");
    if (!material_faces.empty() && material_faces.size() != triangles.size())
        throw std::invalid_argument("Cloth material faces must match its triangles");
    // Rest edge vectors of each face in its own plane, from the pattern or the 3D rest shape.
    std::vector<float> areas(triangles.size());
    using Edge = std::pair<uint32_t, uint32_t>;
    std::map<Edge, std::vector<std::pair<uint32_t, uint32_t>>> edge_faces;  // (face, corner)
    for (size_t f = 0u; f < triangles.size(); ++f) {
        const ClothTriangle& tri = triangles[f];
        for (uint32_t v : tri.v)
            if (v >= rest_positions.size())
                throw std::invalid_argument("Cloth triangle vertex is out of range");
        float u1, v1, u2, v2;
        if (!material_faces.empty()) {
            const auto& face = material_faces[f];
            u1 = face[1].x - face[0].x; v1 = face[1].y - face[0].y;
            u2 = face[2].x - face[0].x; v2 = face[2].y - face[0].y;
        } else {
            const math::Vec3 e1 = rest_positions[tri.v[1]] - rest_positions[tri.v[0]];
            const math::Vec3 e2 = rest_positions[tri.v[2]] - rest_positions[tri.v[0]];
            const float e1_len = e1.Length();
            const math::Vec3 normal = e1.Cross(e2);
            if (!(e1_len > 0.0f) || !(normal.Length() > 0.0f))
                throw std::invalid_argument("Cloth rest triangle is degenerate");
            const math::Vec3 t = e1 / e1_len;
            const math::Vec3 b = normal.Cross(t) / normal.Length();
            u1 = e1_len; v1 = 0.0f;
            u2 = e2.Dot(t); v2 = e2.Dot(b);
        }
        const float det = u1 * v2 - u2 * v1;
        if (!(std::fabs(det) > 0.0f) || !std::isfinite(det))
            throw std::invalid_argument("Cloth rest triangle is degenerate");
        const float youngs = materials[f].stretch_stiffness, nu = materials[f].poisson;
        nk::VbdElement e;
        e.kind = nk::kVbdTriangle;
        for (uint32_t j = 0u; j < 3u; ++j) e.vertex[j] = tri.v[j];
        e.rest[0] = v2 / det;
        e.rest[1] = -u2 / det;
        e.rest[2] = -v1 / det;
        e.rest[3] = u1 / det;
        e.rest[4] = areas[f] = 0.5f * std::fabs(det);
        e.rest[5] = youngs / (2.0f * (1.0f + nu));
        e.rest[6] = youngs * nu / (1.0f - nu * nu);
        e.rest[7] = 1.0f - materials[f].compression_ratio;
        e.damping = materials[f].damping;
        out.push_back(e);
        for (uint32_t j = 0u; j < 3u; ++j) {
            const uint32_t a = tri.v[j], b = tri.v[(j + 1u) % 3u];
            edge_faces[{std::min(a, b), std::max(a, b)}].push_back({static_cast<uint32_t>(f), j});
        }
    }
    // Hinge weight 3 |e|^2 / (A1 + A2) over rest quantities: E = kb (theta - theta0)^2 |e| / h_e.
    for (const auto& entry : edge_faces) {
        if (entry.second.size() != 2u) continue;
        const auto [fa, ca] = entry.second[0];
        const auto [fb, cb] = entry.second[1];
        const float bend = 0.5f * (materials[fa].bend_stiffness + materials[fb].bend_stiffness);
        if (!(bend > 0.0f)) continue;
        const ClothTriangle& ta = triangles[fa];
        const ClothTriangle& tb = triangles[fb];
        // Face a runs x0 -> x1 along the edge and face b runs back, so both normals share one sense.
        const uint32_t x0 = ta.v[ca], x1 = ta.v[(ca + 1u) % 3u], x2 = ta.v[(ca + 2u) % 3u];
        const uint32_t x3 = tb.v[(cb + 2u) % 3u];
        if (tb.v[cb] != x1 || tb.v[(cb + 1u) % 3u] != x0)
            throw std::invalid_argument("Cloth faces sharing an edge must be consistently wound");
        float edge_len = 0.0f;
        if (!material_faces.empty()) {
            const auto& face = material_faces[fa];
            edge_len = (face[(ca + 1u) % 3u] - face[ca]).Length();
        } else {
            edge_len = (rest_positions[x1] - rest_positions[x0]).Length();
        }
        nk::VbdElement e;
        e.kind = nk::kVbdHinge;
        e.vertex[0] = x0; e.vertex[1] = x1; e.vertex[2] = x2; e.vertex[3] = x3;
        e.rest[0] = material_faces.empty()
            ? DihedralAngle(rest_positions[x0], rest_positions[x1], rest_positions[x2], rest_positions[x3])
            : 0.0f;
        e.rest[1] = bend * 3.0f * edge_len * edge_len / (areas[fa] + areas[fb]);
        e.damping = 0.5f * (materials[fa].damping + materials[fb].damping);
        out.push_back(e);
    }
}

void BuildFieldShellElements(const std::vector<math::Vec3>& rest_positions,
                             const std::vector<ClothTriangle>& triangles,
                             const std::vector<std::array<math::Vec3, 3>>& material_faces,
                             const std::vector<uint32_t>& weld,
                             const nk::MfMaterial& material, uint32_t material_index,
                             float weibull_modulus, const math::Vec3& grain, float grain_ratio,
                             std::vector<nk::MfElement>& out, float weibull_reference_area) {
    if (!nk::mf::ValidMaterial(material))
        throw std::invalid_argument("Material-field shell needs positive density, thickness and modulus");
    const bool orthotropic = material.law == nk::kMfLogStrainOrthotropicLaw;
    ShellMaterial shell;
    shell.stretch_stiffness = material.youngs * material.thickness;
    shell.poisson = orthotropic ? 0.0f : material.poisson;
    std::vector<nk::VbdElement> elements;
    BuildClothVertexBlocks(rest_positions, triangles, material_faces,
                           std::vector<ShellMaterial>(triangles.size(), shell), elements);
    // The orthotropic law reads each face in its grain frame: axis 1 along the grain projected into the face.
    std::vector<Point2> axis(orthotropic ? triangles.size() : 0u);
    for (size_t f = 0u; f < axis.size(); ++f) {
        Point2 g{grain.x, grain.y};
        if (material_faces.empty()) {
            const math::Vec3 e1 = rest_positions[triangles[f].v[1]] - rest_positions[triangles[f].v[0]];
            const math::Vec3 normal = e1.Cross(rest_positions[triangles[f].v[2]] - rest_positions[triangles[f].v[0]]);
            const math::Vec3 t = e1 / e1.Length(), b = normal.Cross(t) / normal.Length();
            g = {grain.Dot(t), grain.Dot(b)};
        }
        const double length = std::sqrt(Dot2(g, g));
        if (!(length >= 0.5 * grain.Length()))
            throw std::invalid_argument("Material-field grain must lie within 60 degrees of every rest face");
        axis[f] = {g[0] / length, g[1] / length};
    }
    const size_t first_triangle = out.size();
    for (size_t k = 0u; k < elements.size(); ++k) {
        const nk::VbdElement& e = elements[k];
        nk::MfElement f;
        f.material = material_index;
        f.kind = nk::kMfTriangle;
        for (uint32_t j = 0u; j < 3u; ++j) f.vertex[j] = e.vertex[j];
        for (uint32_t j = 0u; j < 5u; ++j) f.rest[j] = e.rest[j];
        if (orthotropic) {
            // Inverse rest edges times Q^T, with Q taking face coordinates to the grain frame.
            const double c = axis[k][0], s = axis[k][1];
            for (uint32_t row = 0u; row < 2u; ++row) {
                const double r0 = e.rest[2u * row], r1 = e.rest[2u * row + 1u];
                f.rest[2u * row] = static_cast<float>(r0 * c + r1 * s);
                f.rest[2u * row + 1u] = static_cast<float>(r1 * c - r0 * s);
            }
        }
        out.push_back(f);
    }
    std::vector<std::array<Point2, 3>> layout(triangles.size());
    const auto welded = [&](uint32_t v) { return weld.empty() ? v : weld[v]; };
    // A face corner's rest point, in the pattern's plane when a pattern is given.
    const auto corner = [&](size_t f, uint32_t k) {
        return material_faces.empty() ? rest_positions[triangles[f].v[k]]
                                      : math::Vec3{material_faces[f][k].x, material_faces[f][k].y, 0.0f};
    };
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> edge_faces;
    for (size_t f = 0u; f < triangles.size(); ++f) {
        layout[f] = FaceLayout(rest_positions, triangles[f], material_faces, f);
        if (orthotropic)
            for (Point2& p : layout[f]) {
                const Point2 q = p;
                p = {axis[f][0] * q[0] + axis[f][1] * q[1], axis[f][0] * q[1] - axis[f][1] * q[0]};
            }
        for (uint32_t j = 0u; j < 3u; ++j) {
            const uint32_t a = welded(triangles[f].v[j]), b = welded(triangles[f].v[(j + 1u) % 3u]);
            edge_faces[{std::min(a, b), std::max(a, b)}].push_back({static_cast<uint32_t>(f), j});
        }
    }
    const double D = nk::mf::BendingModulus<double>(material), nu = material.poisson;
    // Plate rigidity over (kxx, kxy, kyy): D [nu tr^2 + (1 - nu)(kxx^2 + 2 kxy^2 + kyy^2)].
    double P[3][3] = {{D, 0.0, nu * D}, {0.0, 2.0 * (1.0 - nu) * D, 0.0}, {nu * D, 0.0, D}};
    if (orthotropic) {
        // The orthotropic section stiffness over Voigt (kxx, kyy, 2 kxy), reordered to (kxx, kxy, kyy).
        const auto base = nk::mf::BendBase<double>(material);
        const auto& v = base.value;
        const double reordered[3][3] = {{v[0][0], 2.0 * v[0][2], v[0][1]},
                                        {2.0 * v[0][2], 4.0 * v[2][2], 2.0 * v[1][2]},
                                        {v[0][1], 2.0 * v[1][2], v[1][1]}};
        for (uint32_t a = 0u; a < 3u; ++a)
            for (uint32_t b = 0u; b < 3u; ++b) P[a][b] = reordered[a][b];
    }
    // Each cut edge as (face, hinge) on both sides, and each face's bend.
    std::vector<std::array<uint32_t, 4>> cuts, uncut;
    std::vector<uint32_t> bend_of(triangles.size(), ~0u);
    for (size_t f = 0u; f < triangles.size(); ++f) {
        const ClothTriangle& tri = triangles[f];
        const std::array<Point2, 3>& c = layout[f];
        const Point2 centre{(c[0][0] + c[1][0] + c[2][0]) / 3.0, (c[0][1] + c[1][1] + c[2][1]) / 3.0};
        nk::MfElement bend;
        bend.kind = nk::kMfBend;
        bend.material = material_index;
        for (uint32_t j = 0u; j < 3u; ++j) bend.vertex[j] = tri.v[j];
        double R[3][3];
        uint32_t far_count = 0u;
        for (uint32_t i = 0u; i < 3u; ++i) {
            const uint32_t i0 = (i + 1u) % 3u, i1 = (i + 2u) % 3u;
            const uint32_t w0 = welded(tri.v[i0]), w1 = welded(tri.v[i1]);
            const auto& faces = edge_faces[{std::min(w0, w1), std::max(w0, w1)}];
            const Point2 x0 = c[i0], x1 = c[i1], near = c[i];
            const Point2 edge = Sub2(x1, x0);
            const double length = std::sqrt(Dot2(edge, edge));
            const Point2 u{edge[0] / length, edge[1] / length}, r = Sub2(near, x0);
            // A free edge takes the near corner mirrored across it as its far vertex, then is condensed out.
            Point2 far{x0[0] + 2.0 * Dot2(r, u) * u[0] - r[0], x0[1] + 2.0 * Dot2(r, u) * u[1] - r[1]};
            if (faces.size() == 2u) {
                const auto [g, cg] = faces[0].first == f ? faces[1] : faces[0];
                const ClothTriangle& other = triangles[g];
                if (welded(other.v[cg]) != w1 || welded(other.v[(cg + 1u) % 3u]) != w0)
                    throw std::invalid_argument("Cloth faces sharing an edge must be consistently wound");
                if (f < g && (other.v[cg] != tri.v[i1] || other.v[(cg + 1u) % 3u] != tri.v[i0]))
                    cuts.push_back({static_cast<uint32_t>(f), i, g, (cg + 2u) % 3u});
                else if (f < g && weibull_modulus > 0.0f)
                    uncut.push_back({static_cast<uint32_t>(f), i, g, (cg + 2u) % 3u});
                const uint32_t w = (cg + 2u) % 3u;
                const Point2 to0 = Sub2(layout[g][w], layout[g][(cg + 1u) % 3u]);
                const Point2 to1 = Sub2(layout[g][w], layout[g][cg]);
                const double d0 = Dot2(to0, to0), d1 = Dot2(to1, to1);
                const double along = (d0 - d1 + length * length) / (2.0 * length);
                const double height = std::sqrt(std::max(d0 - along * along, 0.0));
                const double side = Cross2(u, r) > 0.0 ? -1.0 : 1.0;
                far = {x0[0] + along * u[0] - side * height * u[1], x0[1] + along * u[1] + side * height * u[0]};
                bend.hinges |= 1u << i;
                const uint32_t offset = 3u + 3u * far_count++;
                bend.vertex[offset] = other.v[(cg + 1u) % 3u];
                bend.vertex[offset + 1u] = other.v[cg];
                bend.vertex[offset + 2u] = other.v[w];
                bend.rest[10u + i] = material_faces.empty()
                    ? DihedralAngle(rest_positions[tri.v[i0]], rest_positions[tri.v[i1]],
                                    rest_positions[tri.v[i]], rest_positions[other.v[w]])
                    : 0.0f;
            }
            const std::array<double, 3> row =
                TurnResponse({Sub2(x0, centre), Sub2(x1, centre), Sub2(near, centre), Sub2(far, centre)});
            for (uint32_t k = 0u; k < 3u; ++k) R[i][k] = row[k];
        }
        if (bend.hinges == 0u) continue;
        // B = R^-1 recovers (kxx, kxy, kyy) from the three turns; M = A B^T P B is the hinge-space stiffness.
        const double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
                           R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
                           R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
        if (!(std::fabs(det) > 0.0) || !std::isfinite(det))
            throw std::invalid_argument("Material-field bend patch is degenerate");
        double B[3][3];
        for (uint32_t a = 0u; a < 3u; ++a)
            for (uint32_t b = 0u; b < 3u; ++b) {
                const uint32_t r0 = (b + 1u) % 3u, r1 = (b + 2u) % 3u, c0 = (a + 1u) % 3u, c1 = (a + 2u) % 3u;
                B[a][b] = (R[r0][c0] * R[r1][c1] - R[r0][c1] * R[r1][c0]) / det;
            }
        const double area = 0.5 * std::fabs(Cross2(Sub2(c[1], c[0]), Sub2(c[2], c[0])));
        double M[3][3] = {};
        for (uint32_t i = 0u; i < 3u; ++i)
            for (uint32_t j = 0u; j < 3u; ++j)
                for (uint32_t a = 0u; a < 3u; ++a)
                    for (uint32_t b = 0u; b < 3u; ++b) M[i][j] += area * B[a][i] * P[a][b] * B[b][j];
        // A free edge's turn follows the other turns elastically, so B maps only the hinged turns.
        for (uint32_t g = 0u; g < 3u; ++g) {
            if ((bend.hinges >> g) & 1u) continue;
            for (uint32_t j = 0u; j < 3u; ++j)
                if (j != g)
                    for (uint32_t a = 0u; a < 3u; ++a) B[a][j] -= B[a][g] * M[g][j] / M[g][g];
            for (uint32_t a = 0u; a < 3u; ++a) B[a][g] = 0.0;
            for (uint32_t i = 0u; i < 3u; ++i)
                for (uint32_t j = 0u; j < 3u; ++j)
                    if (i != g && j != g) M[i][j] -= M[i][g] * M[g][j] / M[g][g];
            for (uint32_t k = 0u; k < 3u; ++k) M[g][k] = M[k][g] = 0.0;
        }
        // Rows (kxx, kyy, 2 kxy) of the Voigt curvature map, then the area.
        for (uint32_t k = 0u; k < 3u; ++k) {
            bend.rest[k] = static_cast<float>(B[0][k]);
            bend.rest[3u + k] = static_cast<float>(B[2][k]);
            bend.rest[6u + k] = static_cast<float>(2.0 * B[1][k]);
        }
        bend.rest[9] = static_cast<float>(area);
        if (!nk::mf::InitializeBendDual(bend, material))
            throw std::invalid_argument("Material-field bend curvature map is rank deficient");
        bend_of[f] = static_cast<uint32_t>(out.size());
        out.push_back(bend);
    }
    // Corresponding edge endpoints form cohesive pairs of weight L t / 2 and 1% of the face compliance.
    // An uncut edge pairs each endpoint with itself until fracture separates its copies.
    for (size_t k = 0u; k < cuts.size() + uncut.size(); ++k) {
        const bool cut = k < cuts.size();
        const auto [fa, i, fb, j] = cut ? cuts[k] : uncut[k - cuts.size()];
        const ClothTriangle &a = triangles[fa], &b = triangles[fb];
        const uint32_t ends[2][2] = {{a.v[(i + 1u) % 3u], b.v[(j + 2u) % 3u]}, {a.v[(i + 2u) % 3u], b.v[(j + 1u) % 3u]}};
        nk::MfElement seam;
        seam.kind = nk::kMfSeam;
        seam.material = material_index;
        uint32_t pairs = 0u;
        for (const auto& end : ends)
            if (end[0] != end[1] || !cut) {
                seam.vertex[2u * pairs] = end[0];
                seam.vertex[2u * pairs + 1u] = end[1];
                ++pairs;
            }
        seam.vertex[4] = bend_of[fa];
        seam.vertex[5] = bend_of[fb];
        seam.hinges = i | j << 2u | pairs << 4u;
        const Point2 edge = Sub2(layout[fa][(i + 2u) % 3u], layout[fa][(i + 1u) % 3u]);
        const double length = std::sqrt(Dot2(edge, edge));
        seam.rest[0] = static_cast<float>(0.5 * length * material.thickness);
        seam.rest[1] = static_cast<float>(100.0 * material.youngs / length);
        const double tributary_area = (out[first_triangle + fa].rest[4] + out[first_triangle + fb].rest[4]) / 3.0;
        const double scale = weibull_reference_area > 0.0f ? weibull_reference_area / tributary_area : 1.0;
        const double scatter = cut ? 1.0 : std::pow(
            -std::log(EdgeUniform(welded(ends[0][0]), welded(ends[1][0]))), 1.0 / weibull_modulus);
        // Paper parts along its grain at the tear strength and across it at grain_ratio times that.
        const math::Vec3 line = corner(fa, (i + 2u) % 3u) - corner(fa, (i + 1u) % 3u);
        const double along = grain_ratio > 0.0f ? line.Dot(grain) / (line.Length() * grain.Length()) : 1.0;
        seam.rest[3] = static_cast<float>((1.0 + (grain_ratio - 1.0) * (1.0 - along * along)) * scatter);
        // Weakest-link area scaling controls nucleation; the cohesive traction law is a local material property.
        seam.rest[2] = seam.rest[3] * static_cast<float>(cut ? 1.0 : std::pow(scale, 1.0 / weibull_modulus));
        const nk::mf::CohesiveLaw law = nk::mf::SeamLaw(seam, material);
        const nk::mf::MixedCohesiveLaw mixed = nk::mf::MixedSeamLaw(seam, material);
        if (!(material.tear_strength > 0.0f) || !(law.Parting() > law.Onset()) ||
            (material.fracture_law == nk::kMfMixedModeCohesive && !(mixed.shear_toughness > mixed.Onset())))
            throw std::invalid_argument("Material-field seams need a tear strength and a tear energy above "
                                        "strength^2 / (2 x seam stiffness), in opening and in sliding");
        if (weibull_modulus > 0.0f)
            out[first_triangle + fa].vertex[3u + i] = out[first_triangle + fb].vertex[3u + j] =
                static_cast<uint32_t>(out.size());
        out.push_back(seam);
    }
}

void BuildRodVertexBlocks(const std::vector<math::Vec3>& rest_positions,
                          const std::vector<uint32_t>& chain, const RodMaterial& material,
                          std::vector<nk::VbdElement>& out) {
    if (!FiniteNonnegative(material.stretch_stiffness) || !FiniteNonnegative(material.bend_stiffness) ||
        !FiniteNonnegative(material.damping))
        throw std::invalid_argument("Rod material needs nonnegative stiffness and damping");
    for (uint32_t v : chain)
        if (v >= rest_positions.size()) throw std::invalid_argument("Rod vertex is out of range");
    std::vector<float> lengths;
    for (size_t i = 0u; i + 1u < chain.size(); ++i) {
        lengths.push_back((rest_positions[chain[i + 1u]] - rest_positions[chain[i]]).Length());
        out.push_back(VertexBlockSpring(chain[i], chain[i + 1u], lengths.back(),
                                        material.stretch_stiffness, material.damping));
    }
    if (!(material.bend_stiffness > 0.0f)) return;
    // E = kb (1 - cos theta) / mean segment length at each interior vertex.
    for (size_t i = 1u; i + 1u < chain.size(); ++i) {
        nk::VbdElement e;
        e.kind = nk::kVbdRodBend;
        e.vertex[0] = chain[i - 1u]; e.vertex[1] = chain[i]; e.vertex[2] = chain[i + 1u];
        e.rest[0] = material.bend_stiffness / (0.5f * (lengths[i - 1u] + lengths[i]));
        e.damping = material.damping;
        out.push_back(e);
    }
}

}  // namespace nuka::runtime::soft
