#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- articulation cooker
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_state.hpp"
#include "scene/cooked_blob.hpp"

#include <vector>

namespace nuka::runtime::articulation {

std::vector<ArticulationCookedTopology> CookArticulations(const scene::CookedBlob& blob);

} // namespace nuka::runtime::articulation
