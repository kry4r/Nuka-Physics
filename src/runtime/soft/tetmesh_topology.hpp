#pragma once
// nuka::runtime::soft -- cook-time tet lattices, boundary surfaces and embedded skins.

#include "math/vec3.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace nuka::runtime::soft {

// One tetrahedron: four particle indices with arbitrary winding.
struct TetMeshTet {
    uint32_t v[4] = {0u, 0u, 0u, 0u};
};

// A rest-lattice tetrahedralisation: lattice vertex rest positions + a consistent
// 5-tets-per-cell mesh whose face diagonals match between neighbours (no cracks).
struct TetLattice {
    std::vector<math::Vec3> rest;     // lattice vertex rest positions
    std::vector<TetMeshTet> tets;     // 5 tets per kept cell
};

// Tessellate the grid of `cells` cubic cells (edge `cell_len`) from `origin`,
// keeping cells where `keep_cell(center)` holds, each split into 5 parity tets.
TetLattice BuildTetLattice(const math::Vec3& origin, uint32_t cells[3],
                           float cell_len,
                           const std::function<bool(const math::Vec3&)>& keep_cell);

// Sphere rest-lattice: `cells` cubic cells per axis (edge `cell_len`) centered on
// `center`, keeping cells whose 8 corners all fall inside `radius` (watertight).
TetLattice BuildSphereTetLattice(const math::Vec3& center, float radius,
                                 uint32_t cells, float cell_len);

// 6 * signed volume of a tet: det([p1-p0, p2-p0, p3-p0]).
float TetSignedVolumeTimes6(const math::Vec3& p0, const math::Vec3& p1,
                            const math::Vec3& p2, const math::Vec3& p3);

// Extract the boundary surface of `tets` as a flat triangle index list (3
// particle indices per triangle): a face on exactly ONE incident tet is a
// boundary face, a face shared by two tets is interior and dropped. Each emitted
// triangle is wound to face OUTWARD (the fourth tet vertex sits behind it).
// Deterministic emission order. The output feeds runtime::soft::BuildSurfaceMesh
// so a deforming tet body renders as a smooth surface on the same path as cloth.
std::vector<uint32_t> ExtractBoundaryTriangles(
    const std::vector<math::Vec3>& rest_positions,
    const std::vector<TetMeshTet>& tets);

// A triangle mesh (positions + flat 3-index triangle list): the sphere skin
// generator output and the input surface to EmbedSurfaceInTetCage.
struct TriMesh {
    std::vector<math::Vec3> positions;
    std::vector<uint32_t>   triangles;
};

// Subdivided icosphere of `radius` at `center`: a unit icosahedron with each face
// split `subdivisions` times and projected to the sphere (3 -> 642 verts smooth).
TriMesh BuildIcosphere(const math::Vec3& center, float radius, uint32_t subdivisions);

// One skin vertex's binding to a cage tet: the tet's 4 GLOBAL vertex indices and
// the 4 barycentric weights (Σ w == 1). Deform reads only these.
struct SkinVertexBind {
    uint32_t vi[4] = {0u, 0u, 0u, 0u};
    float    w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

// A render skin embedded in a tet cage: the skin triangles plus a per-vertex
// barycentric binding into the cage. Deforms with the cage, no `tets` needed.
struct EmbeddedSkin {
    std::vector<uint32_t>       triangles;
    std::vector<SkinVertexBind> binds;
};

// Embed any surface in any tet cage: bind each vertex to the CONTAINING tet else
// the smallest-outside-ness one, un-clamped (outside verts extrapolate affinely).
EmbeddedSkin EmbedSurfaceInTetCage(const std::vector<math::Vec3>& skin_rest,
                                   const std::vector<uint32_t>& skin_tris,
                                   const std::vector<math::Vec3>& tet_rest,
                                   const std::vector<TetMeshTet>& tets,
                                   uint32_t* out_extrapolated = nullptr);

// Embed any surface in a dense PARTICLE set: bind each skin vertex to its K=4
// NEAREST particles (rest configuration) with normalized inverse-distance weights
// (Σ w == 1). Reuses the SkinVertexBind/EmbeddedSkin (vi index particles now) so
// DeformEmbeddedSkin blends from the live particle positions UNCHANGED. A dense
// MPM particle fill has no out-of-set vertex, so the blend cannot spike on crush.
EmbeddedSkin EmbedSurfaceInParticleSet(const std::vector<math::Vec3>& skin_rest,
                                       const std::vector<uint32_t>& skin_tris,
                                       const std::vector<math::Vec3>& particle_rest);

// Deform the embedded skin by the live cage/particles: out[v] = Σ_i w_i*driver[vi_i],
// an affine blend (smooth, exact under rigid/affine motion). out sizes to binds.
void DeformEmbeddedSkin(const EmbeddedSkin& e,
                        const std::vector<math::Vec3>& tet_live,
                        std::vector<math::Vec3>& out);

// Laplacian-smooth a closed triangle surface in place: each vertex moves a
// `lambda` fraction toward its edge-neighbour centroid, for `iterations` passes.
// Removes high-frequency facet/cage corrugation from a deformed embedded skin
// while preserving the gross shape (the smoothing is a pure mesh operation).
void SmoothSurface(const std::vector<uint32_t>& triangles, uint32_t iterations,
                   float lambda, std::vector<math::Vec3>& positions);

} // namespace nuka::runtime::soft
