#include "runtime/app/viewer/editor_scene.hpp"

#include "render/render_world.hpp"
#include "scene/format/nks.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <system_error>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

// Binds the world's current layout: render world, terrain, rest drive targets.
void BindWorld(EditorScene& es) {
    WorldLayout layout = QueryLayout(es.world);
    render::RenderWorld render_world =
        render::BuildRenderWorld(es.scene.Ecs(), BindScene(es.scene, layout));
    es.terrain = QueryTerrain(es.world);
    es.rest_targets.assign(static_cast<size_t>(layout.links_per_env) * layout.env_count, 0.0f);
    if (!es.rest_targets.empty())
        CheckResult(nuka_world_download_field(es.world, NUKA_FIELD_DRIVE_TARGET,
                                              es.rest_targets.data(),
                                              es.rest_targets.size() * sizeof(float), 0u),
                    "download drive targets");
    es.sim = std::make_unique<EditorSim>(es.world, std::move(layout), std::move(render_world));
}

}  // namespace

SceneCache::SceneCache()
    : dir_(std::filesystem::temp_directory_path() / "nuka_editor" /
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) {}

std::string SceneCache::NextFile() {
    std::filesystem::create_directories(dir_);
    return (dir_ / ("scene_" + std::to_string(next_++) + ".nks")).string();
}

void SceneCache::Remove() {
    std::error_code error;
    std::filesystem::remove_all(dir_, error);
}

std::unique_ptr<EditorScene> AttachEditorScene(nuka_world_handle world, const std::string& path,
                                               SceneCache& cache) {
    auto es = std::make_unique<EditorScene>();
    es->world = world;
    es->path = path;
    const std::string file = cache.NextFile();
    CheckResult(nuka_world_export_scene(world, file.c_str()), "nuka_world_export_scene");
    es->scene = scene::nks::Load(file);
    BindWorld(*es);
    return es;
}

bool RecookEditorScene(EditorScene& es, SceneCache& cache) {
    try {
        const std::string file = cache.NextFile();
        scene::nks::Save(es.scene, file);
        const nuka_result_t result = nuka_world_rebuild_from_scene(es.world, file.c_str(), 0u);
        if (result != NUKA_RESULT_OK) {
            std::fprintf(stderr, "[nuka_editor] rebuild refused: %s\n", nuka_result_message(result));
            return false;
        }
        BindWorld(es);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] rebuild failed: %s\n", e.what());
        return false;
    }
}

bool ResetEditorScenePhysics(EditorScene& es) {
    if (nuka_world_reset(es.world) != NUKA_RESULT_OK) return false;
    return es.rest_targets.empty() ||
           nuka_world_upload_field(es.world, NUKA_FIELD_DRIVE_TARGET, es.rest_targets.data(),
                                   es.rest_targets.size() * sizeof(float), 0u) == NUKA_RESULT_OK;
}

}  // namespace nuka::runtime::app::viewer
