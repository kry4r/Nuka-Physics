#include "runtime/soft/cloth_topology.hpp"

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
                            const ShellMaterial& material,
                            std::vector<nk::VbdElement>& out) {
    if (!FiniteNonnegative(material.stretch_stiffness) || !FiniteNonnegative(material.bend_stiffness) ||
        !FiniteNonnegative(material.damping) || !std::isfinite(material.poisson) ||
        material.poisson < 0.0f || material.poisson >= 0.5f)
        throw std::invalid_argument("Cloth shell material needs nonnegative stiffness and 0 <= poisson < 0.5");
    if (!material_faces.empty() && material_faces.size() != triangles.size())
        throw std::invalid_argument("Cloth material faces must match its triangles");
    const float youngs = material.stretch_stiffness, nu = material.poisson;
    const float mu = youngs / (2.0f * (1.0f + nu));
    const float lambda = youngs * nu / (1.0f - nu * nu);
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
        nk::VbdElement e;
        e.kind = nk::kVbdTriangle;
        for (uint32_t j = 0u; j < 3u; ++j) e.vertex[j] = tri.v[j];
        e.rest[0] = v2 / det;
        e.rest[1] = -u2 / det;
        e.rest[2] = -v1 / det;
        e.rest[3] = u1 / det;
        e.rest[4] = areas[f] = 0.5f * std::fabs(det);
        e.rest[5] = mu;
        e.rest[6] = lambda;
        e.damping = material.damping;
        out.push_back(e);
        for (uint32_t j = 0u; j < 3u; ++j) {
            const uint32_t a = tri.v[j], b = tri.v[(j + 1u) % 3u];
            edge_faces[{std::min(a, b), std::max(a, b)}].push_back({static_cast<uint32_t>(f), j});
        }
    }
    if (!(material.bend_stiffness > 0.0f)) return;
    // Hinge weight 3 |e|^2 / (A1 + A2) over rest quantities: E = kb (theta - theta0)^2 |e| / h_e.
    for (const auto& entry : edge_faces) {
        if (entry.second.size() != 2u) continue;
        const auto [fa, ca] = entry.second[0];
        const auto [fb, cb] = entry.second[1];
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
        e.rest[1] = material.bend_stiffness * 3.0f * edge_len * edge_len / (areas[fa] + areas[fb]);
        e.damping = material.damping;
        out.push_back(e);
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
