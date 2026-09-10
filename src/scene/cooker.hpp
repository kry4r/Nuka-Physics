#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::CookScene – flatten SceneIR into a runtime CookedBlob
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include "scene/cooked_blob.hpp"
#include "import/cooker/mesh_surface_cooker.hpp"

namespace nuka::scene {

// Optional acceleration assets never change the authored collision geometry.
struct CookSceneOptions {
    // Bake a narrow-band sparse SDF for validated convex pieces.
    bool bake_sdf = true;
    // Retained for source compatibility; it does not change collision geometry.
    bool general_single_hull = false;
    // Produce optional visual distance fields; they do not supply collision geometry.
    bool bake_link_sdf = false;
    import::cooker::MeshSurfaceCookOptions mesh_surface;
};

/// Flatten the scene intermediate representation into struct-of-arrays
/// tables suitable for direct runtime / GPU consumption.
/// The 1-arg overload uses the default (legacy) CookSceneOptions.
CookedBlob CookScene(const SceneIR& scene, const CookSceneOptions& options);
CookedBlob CookScene(const SceneIR& scene);

} // namespace nuka::scene
