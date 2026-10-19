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

enum class AssetKind : uint8_t { Unknown, Scene, Urdf, Mjcf, Usd, Mesh, Image };

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
    bool fix_base = false;  // URDF only
    std::vector<std::string> package_paths;  // URDF only
    std::string light_units;  // "isaac", "nits" or "raw"; USD only
    std::vector<ImportDiagnostic> diagnostics;
    std::optional<scene::SceneIR> scene;  // absent when the file does not load
    ImportStats stats;
};

// How a URDF reads: its profile, and whether the base is fixed or packages are searched elsewhere; how a
// USD stage's authored light intensities read (import::UsdLightUnits).
struct ImportOptions {
    std::string profile = "strict";
    bool fix_base = false;
    std::vector<std::string> package_paths;
    std::string light_units = "isaac";
};
// From an action's or an op's `profile`, `fix_base`, `package_paths` and `light_units` fields; `fallback` is
// the profile when the fields name none.
ImportOptions ImportOptionsOf(const scene::json::Value& fields, const std::string& fallback = "strict");
void SetImportOptions(scene::json::Value& fields, const ImportOptions& options);

ImportAnalysis AnalyzeImport(const std::string& path, const ImportOptions& options);
// A loaded asset imports unless a diagnostic is an error; the message names the first one.
bool Importable(const ImportAnalysis& analysis);
std::string ImportError(const ImportAnalysis& analysis);
scene::json::Value AnalysisToJson(const ImportAnalysis& analysis);
// Where importing `analysis` came from, with its options and `ground`; the node is set where it lands.
scene::SourceRecord SourceOf(const ImportAnalysis& analysis, bool ground);
ImportOptions OptionsOf(const scene::SourceRecord& source);
scene::json::Value SourceToJson(const scene::SourceRecord& source);

// A mesh file as one dynamic body: the mesh is its visual and its convex hull its collider,
// with water density over the hull and the inertia of the hull's box.
scene::SceneIR MeshAssetScene(const std::string& path);

// The world-space box around every finite shape; false when there is none.
bool SceneBounds(const scene::SceneIR& scene, math::Vec3* lo, math::Vec3* hi);
// Lifts `scene` so its lowest finite shape rests at z = 0, unless it brings its own floor (a plane or a
// terrain), which keeps the heights it was authored at.
void RestOnGround(scene::SceneIR& scene);
// A body's pose at its joints' initial positions: a link hangs off its parent link through the joint,
// as the forward kinematics composes it; any other body follows its parent chain.
math::Transform BodyWorld(const scene::SceneIR& scene, scene::BodyId body);

// Composes `addon` at `placement` under a new root group `name` (made unique), or as that one body for a
// mesh asset; `ground` lifts it so its lowest point rests at the placement height, and `source` is recorded
// on the node. Returns the node's path.
std::string ComposeAsset(scene::SceneIR& doc, const scene::SceneIR& addon, AssetKind kind, const std::string& name,
                         math::Transform placement, bool ground,
                         std::optional<scene::SourceRecord> source = std::nullopt);
// The placement that puts `addon`'s root where the root of the node `source` names stands in `doc` now.
math::Transform ReimportPlacement(const scene::SceneIR& doc, const scene::SceneIR& addon,
                                  const scene::SourceRecord& source);

}  // namespace nuka::runtime::app::viewer
