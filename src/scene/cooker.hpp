#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::CookScene – flatten SceneIR into a runtime CookedBlob
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include "scene/cooked_blob.hpp"

namespace nuka::scene {

/// Cook-time options that gate the two HEAVY per-shape stages. The defaults
/// reproduce the legacy cook BYTE-FOR-BYTE (every existing golden cooks with the
/// 1-arg overload below, which passes these defaults). The GENERAL (PairDriven)
/// contact path opts OUT of both stages because the cvx-only narrowphase consumes
/// NEITHER product (L-RECON-B): it never reads a sparse SDF (OpNarrowphaseSdf is a
/// no-op for every family) and it wants ONE convex hull per mesh, not V-HACD's N
/// concave pieces. Skipping them makes the whole-body H1 union scene cook
/// tractable. These flags are STAGE gates only — NO entity/scene-name branch.
struct CookSceneOptions {
    // Bake a narrow-band sparse SDF per unique convex piece (cooker.cpp). The
    // general cvx narrowphase never reads it, so PairDriven sets this false.
    bool bake_sdf = true;
    // Treat EVERY mesh collision shape as a single convex hull (its own verts)
    // instead of running V-HACD convex decomposition — UNLESS the shape author
    // explicitly forced decomposition (DecomposeMode::Force). The general cvx
    // support function wants one hull per mesh; a genuinely-concave shape may
    // still opt INTO V-HACD via Force. Auto/Skip both collapse to one hull when
    // this is set. Leaves the per-shape decompose_mode UNCHANGED when false.
    bool general_single_hull = false;
};

/// Flatten the scene intermediate representation into struct-of-arrays
/// tables suitable for direct runtime / GPU consumption.
/// The 1-arg overload uses the default (legacy) CookSceneOptions.
CookedBlob CookScene(const SceneIR& scene, const CookSceneOptions& options);
CookedBlob CookScene(const SceneIR& scene);

} // namespace nuka::scene
