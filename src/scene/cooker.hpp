#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::CookScene – flatten SceneIR into a runtime CookedBlob
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include "scene/cooked_blob.hpp"

namespace nuka::scene {

/// Flatten the scene intermediate representation into struct-of-arrays
/// tables suitable for direct runtime / GPU consumption.
CookedBlob CookScene(const SceneIR& scene);

} // namespace nuka::scene
