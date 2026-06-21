#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::soft -- COOK-TIME tetrahedral-mesh XPBD topology builder
// ---------------------------------------------------------------------------
//
// v0.7 p09-B. Given a tet list (4 vertex indices per tetrahedron) and the rest
// positions, builds the XPBD constraint list a tet soft body needs:
//   - one DISTANCE constraint per UNIQUE tet edge (the 6 edges of each tet,
//     de-duplicated across shared faces), rest length = the rest edge length;
//   - one VOLUME constraint per tet, rest_volume_times6 = the rest triple product
//     det([p1-p0, p2-p0, p3-p0]) computed with the IDENTICAL expression the GPU
//     volume solver uses, so C == 0 at rest regardless of vertex winding.
//
// This is a ONE-TIME cook-time builder (master plan: cookers may be CPU; the
// per-step SIM remains GPU-only). It returns host XpbdConstraintSet entries the
// caller feeds them into the nk soft cook; the nk solve iterates them on the GPU.
// Determinism: edges are emitted in a fixed, sorted, de-duplicated order so the
// constraint list (and therefore the serial GS sweep order) is reproducible.
// ---------------------------------------------------------------------------

#include "import/cooker/xpbd_cooker_types.hpp"  // XpbdConstraintSet (CUDA-free PODs)
#include "math/vec3.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace nuka::runtime::soft {

// One tetrahedron: four particle indices. Winding is arbitrary -- the volume
// rest value is computed with the same det as the solver, so any winding yields
// C == 0 at rest (a left-handed tet just gets a negative rest_volume_times6).
struct TetMeshTet {
    uint32_t v[4] = {0u, 0u, 0u, 0u};
};

struct TetMeshTopologyOptions {
    float distance_compliance_alpha = 0.0f;  // 0 == rigid edges.
    float volume_compliance_alpha = 0.0f;    // 0 == rigid volume.
    // When true, emit the per-edge distance constraints. A volume-isolation test
    // sets this false to exercise the volume constraint alone (the 6 stiff edges
    // of a tet otherwise fully determine its shape, masking the volume gradient).
    bool emit_distance_constraints = true;
};

// Append distance + volume constraints for `tets` (rest geometry = `rest_positions`)
// onto `out`. rest_positions indices must match the tet vertex indices.
void BuildTetMeshConstraints(const std::vector<math::Vec3>& rest_positions,
                             const std::vector<TetMeshTet>& tets,
                             const TetMeshTopologyOptions& options,
                             XpbdConstraintSet& out);

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

// 6 * signed volume of a tet, == the det the GPU volume solver computes:
//   det([p1-p0, p2-p0, p3-p0]) = (p1-p0) . ((p2-p0) x (p3-p0)).
// Exposed so tests can assert volume preservation against the same expression.
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

// Deform the embedded skin by the live cage: out[v] = Σ_i w_i * tet_live[vi_i], an
// affine blend (smooth, exact under rigid/affine cage motion). out sizes to binds.
void DeformEmbeddedSkin(const EmbeddedSkin& e,
                        const std::vector<math::Vec3>& tet_live,
                        std::vector<math::Vec3>& out);

} // namespace nuka::runtime::soft
