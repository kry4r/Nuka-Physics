#pragma once
// The document's materials and the selection's parts as the Properties panel shows them.

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// A material's map fields, in the order the editor lists them; albedo and emissive hold color.
inline constexpr const char* kMaterialMaps[5] = {"albedo_map", "roughness_map", "metallic_map", "normal_map",
                                                 "emissive_map"};

struct MaterialItem {
    std::string name;
    float base_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    bool textured = false;  // it has a map
    uint32_t users = 0u;    // the shapes and media that use it
    std::string ball;       // a PPM of it on a ball, or empty while it is drawn
};

// A part of the selection: a shape node the world draws, and its material's name (empty: the default).
struct MaterialPartItem {
    std::string path;
    std::string name;
    std::string material;
    float base_color[4] = {0.8f, 0.8f, 0.8f, 1.0f};
};

// One map slot of the edited material.
struct MaterialMapItem {
    std::string path;       // empty: no map
    std::string shown;      // the file's name; one inside a file shows as "ToyCar.glb › embedded image"
    std::string thumbnail;  // a PPM of the map, or empty
};

// The selection's parts and the material of the active one, which the Material section edits.
// The materials list every material of the document, selected or not.
struct MaterialView {
    std::vector<MaterialItem> materials;
    std::vector<MaterialPartItem> parts;
    int active = 0;  // UI-set: the part whose material shows
    std::string name;  // the active part's material; empty for the default one
    uint32_t users = 0u;  // the shapes and media that use it
    float friction = 1.0f;
    MaterialMapItem maps[5];  // albedo, roughness, metallic, normal, emissive
    uint32_t roughness_channel = 0u;
    uint32_t metallic_channel = 0u;
    float uv_scale = 1.0f;
    bool triplanar = true;
    uint64_t version = 0u;  // bumped whenever the host rebuilds it
};

}  // namespace nuka::runtime::app::viewer
