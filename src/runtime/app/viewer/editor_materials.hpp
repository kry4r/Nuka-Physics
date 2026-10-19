#pragma once
// Materials in the editor: the shapes a node renders with, materials by their names (unique, as
// the render registry keys on them) and the presets a new material starts from.

#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/material_view.hpp"
#include "scene/format/json.hpp"
#include "scene/scene_ir.hpp"

#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// The RenderMaterial field each of kMaterialMaps names.
inline constexpr std::string scene::RenderMaterial::*kMaterialMapFields[5] = {
    &scene::RenderMaterial::albedo_map, &scene::RenderMaterial::roughness_map, &scene::RenderMaterial::metallic_map,
    &scene::RenderMaterial::normal_map, &scene::RenderMaterial::emissive_map};

// A shape node and the shape record behind it.
struct MaterialPart {
    std::string path;
    scene::ShapeId shape = 0;
};

// The shapes at or under the node at `path` in tree order; `rendering` keeps those the world draws.
std::vector<MaterialPart> ShapesUnder(const EditorScene& es, const EntityRecordIndex& idx, const std::string& path,
                                      bool rendering);

// The material named `name`, or kInvalidMaterial.
scene::MaterialId FindMaterial(const scene::SceneIR& scene, const std::string& name);

// `base` when no material has it, else `base` less any ".001" suffix with the next free one.
std::string UniqueMaterialName(const scene::SceneIR& scene, const std::string& base);

// Renames the materials from `first` on that an earlier one already names.
void UniqueMaterialNames(scene::SceneIR& scene, scene::MaterialId first);

// How many shapes and media use material `id`.
uint32_t MaterialUsers(const scene::SceneIR& scene, scene::MaterialId id);

// Material `id`, with the render slot the world draws it with when it has one.
MaterialBinding BindMaterial(const EditorScene& es, scene::MaterialId id);

// The look `bind` has now: its render slot's, else its record's.
scene::RenderMaterial BoundMaterial(const EditorScene& es, const MaterialBinding& bind);

// A preset's fields as MaterialToJson names them, with "friction": plastic, metal, rubber, glass,
// emissive, or textured (the grid maps in the folder `textures`). Null for another name.
scene::json::Value MaterialPreset(const std::string& preset, const std::string& textures);

}  // namespace nuka::runtime::app::viewer
