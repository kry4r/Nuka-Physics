#pragma once
// The robot tools' joint table, built from the scene and the world layout, and the joint
// fields they read through the C ABI.

#include "nuka/nuka.h"
#include "runtime/app/viewer/editor_world.hpp"
#include "runtime/app/viewer/robot_view.hpp"
#include "scene/format/json.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

const char* ActuatorTypeName(scene::ActuatorType type);
// False for a name other than position, motor, velocity or force.
bool ParseActuatorType(const std::string& name, scene::ActuatorType* out);

// The revolute and prismatic joints of one environment's links, in link order.
std::vector<RobotJoint> RobotJoints(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of);
// The nodes of one environment's articulation roots.
std::vector<std::string> RobotRoots(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of);
// The joint named `name`, or -1.
int FindJoint(const std::vector<RobotJoint>& joints, const std::string& name);

void DownloadJointFrame(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointFrame* out);
void DownloadJointSample(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointSample* out);

// A joint's table entry with its live values.
scene::json::Value JointJson(const RobotJoint& joint, const JointFrame& live);

}  // namespace nuka::runtime::app::viewer
