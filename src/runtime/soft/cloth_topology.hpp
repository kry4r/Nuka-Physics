#pragma once

#include "math/vec3.hpp"
#include "nk/solve/vertex_block.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace nuka::runtime::soft {

struct ClothTriangle {
    uint32_t v[3] = {0u, 0u, 0u};
};

// Membrane and bending response of a vertex-block shell.
struct ShellMaterial {
    float stretch_stiffness = 0.0f;  // Young's modulus times thickness, N/m
    float poisson = 0.0f;
    float bend_stiffness = 0.0f;     // N m
    float damping = 0.0f;            // Rayleigh coefficient, s
    float compression_ratio = 1.0f;  // compressive over tensile membrane stiffness, (0, 1]
    uint32_t membrane = nk::kVbdMembraneStvk;  // triangle law; softening applies to StVK only
};

// Axial and bending response of a vertex-block rod.
struct RodMaterial {
    float stretch_stiffness = 0.0f;  // EA, N
    float bend_stiffness = 0.0f;     // EI, N m^2
    float damping = 0.0f;
};

// Membrane triangles of per-face materials and one dihedral hinge per interior edge at the mean of its
// faces; material faces set the rest metric and a flat rest angle, which `rest_angles` overrides per edge.
void BuildClothVertexBlocks(const std::vector<math::Vec3>& rest_positions,
                            const std::vector<ClothTriangle>& triangles,
                            const std::vector<std::array<math::Vec3, 3>>& material_faces,
                            const std::map<std::pair<uint32_t, uint32_t>, float>& rest_angles,
                            const std::vector<ShellMaterial>& materials,
                            std::vector<nk::VbdElement>& out);

// Axial springs between consecutive chain vertices and a bend over each interior vertex.
void BuildRodVertexBlocks(const std::vector<math::Vec3>& rest_positions,
                          const std::vector<uint32_t>& chain, const RodMaterial& material,
                          std::vector<nk::VbdElement>& out);

// One axial spring of stiffness `stiffness` / rest length between two vertices.
nk::VbdElement VertexBlockSpring(uint32_t a, uint32_t b, float rest_length, float stiffness,
                                 float damping);

}  // namespace nuka::runtime::soft
