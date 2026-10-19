#pragma once
// ---------------------------------------------------------------------------
// terrain_field.hpp -- the HeightField a scene's TerrainRecord describes, shared by the cook and the
// USD exporter so both see the same grid.
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"

namespace nuka::terrain {

// The grid `record` describes: its grayscale image when it names one, else its parametric features.
// Throws std::runtime_error when the image cannot be read or the parameters give no grid.
HeightField TerrainField(const scene::TerrainRecord& record);

}  // namespace nuka::terrain
