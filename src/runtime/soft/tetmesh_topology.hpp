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

// 6 * signed volume of a tet, == the det the GPU volume solver computes:
//   det([p1-p0, p2-p0, p3-p0]) = (p1-p0) . ((p2-p0) x (p3-p0)).
// Exposed so tests can assert volume preservation against the same expression.
float TetSignedVolumeTimes6(const math::Vec3& p0, const math::Vec3& p1,
                            const math::Vec3& p2, const math::Vec3& p3);

} // namespace nuka::runtime::soft
