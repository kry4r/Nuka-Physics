#pragma once
// The Properties panel's schema: the scene fields a node shows, in groups, with units, ranges and
// whether an edit reaches the running world (live) or rebuilds it.

#include "scene/scene_fields.hpp"
#include "scene/scene_ir.hpp"

#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

class EditorDocument;

struct PropertyField {
    std::string field;  // the edit op field, e.g. "body.mass"
    std::string label;
    std::string unit;
    float speed = 0.01f;
    float min = 0.0f, max = 0.0f;  // equal: unbounded
    bool integer = false;
    bool live = false;
    bool world = false;  // a world option, which no record holds
    float value = 0.0f;  // the primary target's
    bool inherited = false;  // the record holds none, so the value is its material's
    bool mixed = false;  // the targets hold different values
    bool has_saved = false;  // the saved scene has the field, as `saved`
    float saved = 0.0f;
    float saved_raw = 0.0f;  // what the saved record holds, which a revert writes back
};

// One group edits the same field on every target; the first target is the primary selection.
struct PropertyGroup {
    std::string title;
    std::string detail;  // the shape a body's collision group edits
    std::string kind;  // "body", "joint", "drive", "collision" or "world"
    std::vector<std::string> targets;
    std::vector<scene::SceneTarget> records;
    std::vector<PropertyField> fields;
};

// The records a node's fields reach: its body or shape, or for a joint node its child body.
scene::SceneTarget TargetAt(const scene::SceneIR& scene, const std::string& path);

// The groups of `paths`; with several, only the groups and fields they all have. An empty
// selection shows the world options.
std::vector<PropertyGroup> PropertySchema(const scene::SceneIR& scene, const EditorDocument& doc,
                                          const std::vector<std::string>& paths);

}  // namespace nuka::runtime::app::viewer
