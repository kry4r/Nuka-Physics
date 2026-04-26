#pragma once
// ---------------------------------------------------------------------------
// nuka::import – MJCF (MuJoCo XML) importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include <string>

namespace nuka::import {

/// Load a MuJoCo XML (.xml) file and return a populated SceneIR.
nuka::scene::SceneIR LoadMjcf(const std::string& path);

} // namespace nuka::import
