#pragma once

#include "math/vec3.hpp"
#include "nk/solve/field_block.hpp"
#include "nk/solve/vertex_block.hpp"

#include <array>
#include <cstdint>
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
};

// Axial and bending response of a vertex-block rod.
struct RodMaterial {
    float stretch_stiffness = 0.0f;  // EA, N
    float bend_stiffness = 0.0f;     // EI, N m^2
    float damping = 0.0f;
};

// StVK membrane triangles of per-face materials and one dihedral hinge per interior edge at the
// mean of its faces; planar material faces, when given, set the rest metric and a flat rest angle.
void BuildClothVertexBlocks(const std::vector<math::Vec3>& rest_positions,
                            const std::vector<ClothTriangle>& triangles,
                            const std::vector<std::array<math::Vec3, 3>>& material_faces,
                            const std::vector<ShellMaterial>& materials,
                            std::vector<nk::VbdElement>& out);

// Material-field shell elements over the same topology: plane-stress membranes, and per face a bend whose
// curvature, exact for quadratic deflection, comes from its three hinge turns; free-edge hinges are condensed.
// `weld` maps seam copies to the vertex they were cut from (empty: none); bends reach across a cut edge, which
// then takes a seam over the pairs of copies at its ends.
void BuildFieldShellElements(const std::vector<math::Vec3>& rest_positions,
                             const std::vector<ClothTriangle>& triangles,
                             const std::vector<std::array<math::Vec3, 3>>& material_faces,
                             const std::vector<uint32_t>& weld,
                             const nk::MfMaterial& material, uint32_t material_index,
                             std::vector<nk::MfElement>& out);

// Axial springs between consecutive chain vertices and a bend over each interior vertex.
void BuildRodVertexBlocks(const std::vector<math::Vec3>& rest_positions,
                          const std::vector<uint32_t>& chain, const RodMaterial& material,
                          std::vector<nk::VbdElement>& out);

// One axial spring of stiffness `stiffness` / rest length between two vertices.
nk::VbdElement VertexBlockSpring(uint32_t a, uint32_t b, float rest_length, float stiffness,
                                 float damping);

}  // namespace nuka::runtime::soft
