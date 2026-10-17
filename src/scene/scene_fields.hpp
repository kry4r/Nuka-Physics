#pragma once
// Float fields "<record>.<field>" (a vector part is "<record>.<vector>.<x|y|z>") from a body or shape: body.*,
// joint.* (the joint whose child is the body), actuator.* (its actuators), shape.*, material.* (the shape's material).

#include "scene/scene_ir.hpp"

#include <string_view>

namespace nuka::scene {

struct SceneTarget {
    BodyId body = kInvalidBody;
    ShapeId shape = kInvalidShape;
};

bool IsSceneField(std::string_view field);

// Reads the first record the field reaches; false when it reaches none.
bool GetSceneField(const SceneIR& scene, const SceneTarget& target, std::string_view field, float* value);

// Writes every record the field reaches; false when it reaches none.
bool SetSceneField(SceneIR& scene, const SceneTarget& target, std::string_view field, float value);

}  // namespace nuka::scene
