#pragma once
// Asset import for the editor: what a file is, what importing it gives (located diagnostics,
// counts, bounds) and composing the imported scene into the document.

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/format/json.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

enum class AssetKind : uint8_t { Unknown, Scene, Urdf, Mjcf, Usd, Mesh };

// By extension; an .xml file is MJCF when its root element is <mujoco>.
AssetKind AssetKindOf(const std::string& path);
const char* AssetKindName(AssetKind kind);

struct ImportDiagnostic {
    std::string severity;  // "error" keeps the asset out, "warning" reports a loss
    std::string code;
    std::string element;
    std::string owner;  // the top-level element holding the line, e.g. "link pelvis"
    std::string message;
    int line = 0;
};

struct ImportStats {
    uint32_t links = 0u;
    uint32_t joints = 0u;
    uint32_t colliders = 0u;
    uint32_t mesh_colliders = 0u;
    uint32_t visual_meshes = 0u;
    float mass = 0.0f;
    math::Vec3 lo{};
    math::Vec3 hi{};
};

struct ImportAnalysis {
    std::string path;
    AssetKind kind = AssetKind::Unknown;
    std::string profile;  // "strict" or "legacy"; URDF only
    std::vector<ImportDiagnostic> diagnostics;
    std::optional<scene::SceneIR> scene;  // absent when the file does not load
    ImportStats stats;
};

ImportAnalysis AnalyzeImport(const std::string& path, const std::string& profile);
// A loaded asset imports unless a diagnostic is an error; the message names the first one.
bool Importable(const ImportAnalysis& analysis);
std::string ImportError(const ImportAnalysis& analysis);
scene::json::Value AnalysisToJson(const ImportAnalysis& analysis);

// A mesh file as one dynamic body: the mesh is its visual and its convex hull its collider,
// with water density over the hull and the inertia of the hull's box.
scene::SceneIR MeshAssetScene(const std::string& path);

// The world-space box around every finite shape; false when there is none.
bool SceneBounds(const scene::SceneIR& scene, math::Vec3* lo, math::Vec3* hi);
// A body's pose at its joints' initial positions: a link hangs off its parent link through the joint,
// as the forward kinematics composes it; any other body follows its parent chain.
math::Transform BodyWorld(const scene::SceneIR& scene, scene::BodyId body);

// Composes `addon` at `placement` under a new root group `name` (made unique), or as that one body for a
// mesh asset; `ground` lifts it so its lowest point rests at the placement height. Returns the node's path.
std::string ComposeAsset(scene::SceneIR& doc, const scene::SceneIR& addon, AssetKind kind, const std::string& name,
                         math::Transform placement, bool ground);

}  // namespace nuka::runtime::app::viewer
