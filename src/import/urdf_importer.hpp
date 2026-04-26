#pragma once
// ---------------------------------------------------------------------------
// nuka::import – URDF importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include <string>

namespace nuka::import {

/// Load a URDF (.urdf) file and return a populated SceneIR.
nuka::scene::SceneIR LoadUrdf(const std::string& path);

} // namespace nuka::import
