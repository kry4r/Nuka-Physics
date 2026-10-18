#pragma once
// Sensor records as the editor edits them: JSON whose mount is a body's node path, and the
// shape rule the batched sensor tensors put on a scene's cameras and lidars.

#include "scene/format/json.hpp"
#include "scene/scene_ir.hpp"

#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

const char* SensorTypeName(scene::SensorType type);
bool ParseSensorType(const std::string& name, scene::SensorType* out);
// Cameras, depth cameras, lidars and range scans render; the rest read the physics state.
bool RenderSensor(scene::SensorType type);
bool LidarSensor(scene::SensorType type);

// The body a sensor rides, through its joint when it names one; kInvalidBody when it has none.
scene::BodyId SensorBody(const scene::SceneIR& scene, const scene::SensorDesc& s);
// The node path of a body; "" for kInvalidBody.
std::string BodyPath(const scene::SceneIR& scene, scene::BodyId body);
// The body at `path` or the nearest one above it; kInvalidBody when none is.
scene::BodyId BodyAtOrAbove(const scene::SceneIR& scene, const std::string& path);

scene::json::Value SensorToJson(const scene::SceneIR& scene, const scene::SensorDesc& s);
// Writes the fields `v` names over `s`: `mount` "" is the world; `position` and `rotation_deg` (XYZ) set parts
// of `offset`. False with `error` for an unknown type, a mount that is no body, or a state sensor left without one.
bool SensorFromJson(const scene::json::Value& v, const scene::SceneIR& scene, scene::SensorDesc* s,
                    std::string* error);
// Why these sensors cannot share the batched tensors (one camera size, one lidar fan); empty when they can.
std::string SensorShapeError(const std::vector<scene::SensorDesc>& sensors);
// Gives every camera of `sensors` the image size of camera `s`, or every lidar the fan of lidar `s`.
void ShareSensorShape(const scene::SensorDesc& s, std::vector<scene::SensorDesc>* sensors);
// A new sensor of `s->type`: a camera looks along its body's +X with +Z up, and its image size
// and lidar fan match the ones `sensors` already have.
void NewSensorDefaults(const std::vector<scene::SensorDesc>& sensors, scene::SensorDesc* s);
// `base` with a number after it, unused by any of `sensors`.
std::string UniqueSensorName(const std::vector<scene::SensorDesc>& sensors, const std::string& base);

}  // namespace nuka::runtime::app::viewer
