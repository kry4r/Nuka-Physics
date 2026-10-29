#pragma once
// ---------------------------------------------------------------------------
// nuka::import – URDF importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include <string>
#include <optional>
#include <vector>

namespace nuka::import {

struct MeshPart;

/// The render material a mesh file gives one of its parts, under `name`; its maps follow the part's UVs.
scene::MaterialRecord PartMaterial(const MeshPart& part, const std::string& name);

enum class UrdfImportProfile { LegacyCompatible, Strict };
enum class UrdfDiagnosticKind { Io, Syntax, Invalid, Unsupported };

struct UrdfDiagnostic {
    UrdfDiagnosticKind kind;
    std::string code;
    std::string source;
    int line = 0;
    std::string element;
    std::string message;
};

struct UrdfImportResult {
    UrdfImportProfile profile = UrdfImportProfile::Strict;
    std::optional<scene::SceneIR> scene;
    std::vector<UrdfDiagnostic> diagnostics;
    bool StrictSuccess() const {
        return profile == UrdfImportProfile::Strict && scene.has_value() && diagnostics.empty();
    }
};

// Choices the URDF text does not carry.
struct UrdfImportOptions {
    bool fix_base = false;                   // every link no joint carries becomes static
    bool collapse_fixed_joints = false;      // fold each fixed joint's child link into its parent
    std::vector<std::string> package_paths;  // "<pkg>=<dir>" names a package's folder; other entries are
                                             // searched for package://<pkg>/... before ROS_PACKAGE_PATH
};

// Strict rejects unrepresented semantics; legacy reports losses alongside its projection.
UrdfImportResult ImportUrdf(const std::string& path,
                            UrdfImportProfile profile = UrdfImportProfile::Strict,
                            const UrdfImportOptions& options = {});

/// Load a URDF (.urdf) file and return a populated SceneIR.
nuka::scene::SceneIR LoadUrdf(const std::string& path, const UrdfImportOptions& options = {});

} // namespace nuka::import
