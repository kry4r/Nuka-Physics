#pragma once
// ---------------------------------------------------------------------------
// nuka::import - USD exporter: a SceneIR as the UsdPhysics stage ImportUsd reads back
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "scene/scene_ir.hpp"

#include <string>
#include <vector>

namespace nuka::import {

struct UsdExportOptions {
    // The units light intensities are written in; ImportUsd with the same units reads them back.
    UsdLightUnits light_units = UsdLightUnits::Isaac;
    // A scene tree path (as the Stage shows it): only the bodies under it, the bodies hanging from them and the
    // lights and cameras they carry are written.
    std::string node;
};

// What the stage could not carry, as warnings; everything else reads back through ImportUsd.
struct UsdExportResult {
    std::vector<UsdDiagnostic> diagnostics;
};

// Writes `scene` as UsdPhysics to `path`: usda, usdc (.usdc, .usd) or usdz by its extension. The images its
// materials and environment name are copied beside it into "<stem>_textures", or into the package.
// Throws std::runtime_error when `node` holds no body or the file cannot be written.
UsdExportResult ExportUsd(const scene::SceneIR& scene, const std::string& path,
                          const UsdExportOptions& options = {});

}  // namespace nuka::import
