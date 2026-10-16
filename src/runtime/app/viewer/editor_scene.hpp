#pragma once
// The editor's scene: the editable SceneIR document mirrored from a world's retained
// scene, plus the render binding and live frame loop over that world.

#include "nuka/nuka.h"
#include "runtime/app/viewer/editor_world.hpp"
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// Scene files written for exports and rebuilds. Files stay for the session because
// documents and render worlds reference their mesh archives.
class SceneCache {
public:
    SceneCache();
    std::string NextFile();
    void Remove();

private:
    std::filesystem::path dir_;
    uint32_t next_ = 0u;
};

struct EditorScene {
    nuka::scene::SceneIR scene;          // the editable document
    nuka_world_handle world = nullptr;   // not owned
    std::unique_ptr<EditorSim> sim;
    nuka::terrain::HeightField terrain;  // invalid without a cooked terrain
    std::vector<float> rest_targets;     // drive targets at bind time, all environments
    std::string path;

    const WorldLayout& Layout() const { return sim->Layout(); }
};

// Mirrors the world's current scene into a new editor scene. Throws on failure.
std::unique_ptr<EditorScene> AttachEditorScene(nuka_world_handle world, const std::string& path,
                                               SceneCache& cache);

// Rebuilds the world from the edited document, carrying state over by body name. False
// (logged) when the world refused it, leaving world and binding unchanged.
bool RecookEditorScene(EditorScene& es, SceneCache& cache);

// Restores the initial physical state and the drive targets captured at bind time.
bool ResetWorldPhysics(nuka_world_handle world, const std::vector<float>& rest_targets);

}  // namespace nuka::runtime::app::viewer
