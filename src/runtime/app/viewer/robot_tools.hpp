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
const char* JointTypeName(scene::JointType type);
// False for a name other than fixed, revolute, prismatic, spherical or free.
bool ParseJointType(const std::string& name, scene::JointType* out);
// A joint value as the panels read it: prismatic joints in m, revolute ones in degrees or radians.
float JointShown(const RobotJoint& j, float v, bool radians);
const char* JointUnit(const RobotJoint& j, bool radians);
std::string JointText(const RobotJoint& j, float v, bool radians);

// The revolute and prismatic joints of one environment's links, in link order.
std::vector<RobotJoint> RobotJoints(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of);
// The nodes of one environment's articulation roots.
std::vector<std::string> RobotRoots(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of);
// One environment's articulation links, in link order.
std::vector<RobotLink> RobotLinks(const scene::SceneIR& scene, const WorldLayout& layout,
                                  const std::function<std::string(scene::EntityId)>& path_of);
// A link's centre of mass in the world at `pose`.
math::Vec3 LinkCom(const RobotLink& link, const math::Transform& pose);
// The shown environment's centre of mass and mass; false without mass.
bool RobotCom(const RobotView& v, math::Vec3* com, float* mass);
// The edges of the uniform box with a link's mass and principal moments, in its inertia frame;
// false when the moments break the triangle inequality.
bool InertiaBox(const RobotLink& link, math::Vec3* edges);
// Where a link or joint end of the Measure tool is now; false for another end or a stale link.
bool MeasurePoint(const RobotView& v, const MeasureEnd& e, math::Vec3* out);
// Where the Measure tool's ends are now; false until both resolve.
bool MeasureEnds(const RobotView& v, math::Vec3* from, math::Vec3* to);
// A link's entry; a `pose` adds its world origin and centre of mass.
scene::json::Value LinkJson(const RobotLink& link, const std::vector<RobotLink>& links, const math::Transform* pose);
// The joint named `name`, or -1.
int FindJoint(const std::vector<RobotJoint>& joints, const std::string& name);

void DownloadJointFrame(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointFrame* out);
void DownloadJointSample(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointSample* out);

// A joint's table entry with its live values.
scene::json::Value JointJson(const RobotJoint& joint, const JointFrame& live);

// A step test's rise, overshoot, peak and settling times, settled error and second-order fit,
// or a sine test's gain, lag and tracking error; both with the share of saturated steps.
GainMetrics FitGain(const GainTrace& trace);
// A run's settings, gains and figures; `trace` adds its samples.
scene::json::Value GainRunJson(const GainRun& run, bool trace);

}  // namespace nuka::runtime::app::viewer
