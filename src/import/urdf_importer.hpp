#pragma once
// ---------------------------------------------------------------------------
// nuka::import – URDF importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"
#include <string>
#include <optional>
#include <vector>

namespace nuka::import {

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

// Strict rejects unrepresented semantics; legacy reports losses alongside its projection.
UrdfImportResult ImportUrdf(const std::string& path,
                            UrdfImportProfile profile = UrdfImportProfile::Strict);

/// Load a URDF (.urdf) file and return a populated SceneIR.
nuka::scene::SceneIR LoadUrdf(const std::string& path);

} // namespace nuka::import
