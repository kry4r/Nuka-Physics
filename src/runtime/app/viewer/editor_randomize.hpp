#pragma once
// Domain randomization settings as JSON, for the dr.* actions and the document's profile file.

#include "nuka/nuka.h"
#include "runtime/app/viewer/randomize_view.hpp"
#include "scene/format/json.hpp"

#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// {"seed", "render": {"on", jitter...}, "physics": {"on", "mass": [lo, hi], "armature", "gravity"},
//  "params": {key: {"on", "range": [lo, hi]}}, "reset_draws"}
scene::json::Value RandomizeToJson(const RandomizeConfig& c);
// Writes the keys `v` has over `c`. False with `error` for a negative jitter, a range whose lo
// exceeds its hi, a factor that is not positive, or a parameter outside `keys`.
bool RandomizeFromJson(const scene::json::Value& v, const std::vector<std::string>& keys, RandomizeConfig* c,
                       std::string* error);
// The registry parameters a range can scale: live, float and one value per link.
bool RandomizableParam(const nuka_param_info_t& info);

}  // namespace nuka::runtime::app::viewer
