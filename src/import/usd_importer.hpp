#pragma once
// ---------------------------------------------------------------------------
// nuka::import - USD importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <optional>
#include <string>
#include <vector>

namespace nuka::import {

enum class UsdDiagnosticSeverity { Warning, Error };

// One thing the importer could not carry over exactly; an error leaves that part out.
struct UsdDiagnostic {
    UsdDiagnosticSeverity severity = UsdDiagnosticSeverity::Error;
    std::string code;
    std::string prim;     // path of the prim it concerns
    std::string message;
};

struct UsdImportResult {
    std::optional<scene::SceneIR> scene;
    std::vector<UsdDiagnostic> diagnostics;
    bool HasErrors() const;
};

// Reads a USD stage with the UsdPhysics semantics: world-placed bodies, joint frames, drives, mass,
// metersPerUnit, kilogramsPerUnit and upAxis. Throws only when OpenUSD cannot open the file.
UsdImportResult ImportUsd(const std::string& path);

// Loads a USD scene into the canonical SceneIR. OpenUSD opens and composes the file (references,
// payloads, variants, instancing), so every format it reads imports the same way. Throws on the
// first error diagnostic.
nuka::scene::SceneIR LoadUsd(const std::string& path);

// True when OpenUSD reads the path's extension: usd, usda, usdc, usdz, plus plugin formats (abc, drc).
bool IsUsdPath(const std::string& path);

} // namespace nuka::import
