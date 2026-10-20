#include "runtime/app/viewer/editor_scene.hpp"

#include "render/render_world.hpp"
#include "runtime/app/viewer/editor_profile.hpp"
#include "scene/asset/asset_bytes.hpp"
#include "scene/format/nks.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

// Applies `rewrite` to every material's image map paths, touching only the records it changes.
template <typename Rewrite>
void RewriteMaps(scene::SceneIR& scene, Rewrite rewrite) {
    for (size_t i = 0; i < scene.Materials().size(); ++i) {
        const scene::MaterialRecord& m = scene.Materials()[i];
        std::string albedo = rewrite(m.albedo_map), roughness = rewrite(m.roughness_map);
        std::string normal = rewrite(m.normal_map), metallic = rewrite(m.metallic_map);
        std::string emissive = rewrite(m.emissive_map);
        if (albedo == m.albedo_map && roughness == m.roughness_map && normal == m.normal_map &&
            metallic == m.metallic_map && emissive == m.emissive_map)
            continue;
        scene::MaterialRecord& rec = scene.GetMaterialMut(m.id);
        rec.albedo_map = std::move(albedo);
        rec.roughness_map = std::move(roughness);
        rec.normal_map = std::move(normal);
        rec.metallic_map = std::move(metallic);
        rec.emissive_map = std::move(emissive);
    }
}

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
    es.sim = std::make_unique<EditorSim>(std::move(layout), std::move(render_world), es.scene);
    es.sim->SetTerrain(es.terrain);
}

// Saves the document to the cache, rebuilds the world from that file and binds the result.
template <typename Rebuild>
std::string RebuildFromDocument(EditorScene& es, SceneCache& cache, Rebuild rebuild) {
    try {
        const std::string file = cache.NextFile();
        scene::nks::Save(es.scene, file);
        const nuka_result_t result = rebuild(file.c_str());
        if (result != NUKA_RESULT_OK) {
            const std::string why = ResultText(result);
            std::fprintf(stderr, "[nuka_editor] rebuild refused: %s\n", why.c_str());
            return why;
        }
        BindWorld(es);
        return {};
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] rebuild failed: %s\n", e.what());
        return e.what();
    }
}

}  // namespace

void AnchorMaps(scene::SceneIR& scene, const std::string& scene_path) {
    const std::filesystem::path dir = Utf8Path(NormalPath(scene_path)).parent_path();
    RewriteMaps(scene, [&](const std::string& map) {
        return map.empty() || Utf8Path(map).is_absolute() ? map : PathText((dir / Utf8Path(map)).lexically_normal());
    });
}

void RelativeMaps(scene::SceneIR& scene, const std::string& scene_path) {
    const std::filesystem::path dir = Utf8Path(NormalPath(scene_path)).parent_path();
    // A packed image stays absolute: the save copies it into the scene's own .nka.
    RewriteMaps(scene, [&](const std::string& map) {
        if (map.empty() || Utf8Path(map).is_relative() || scene::NamesTextureChunk(map)) return map;
        const std::u8string relative = Utf8Path(map).lexically_relative(dir).generic_u8string();
        return relative.empty() ? map : std::string(relative.begin(), relative.end());
    });
}

SceneCache::SceneCache()
    : dir_(std::filesystem::temp_directory_path() / "nuka_editor" /
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) {}

std::string SceneCache::NextFile() {
    std::filesystem::create_directories(dir_);
    return (dir_ / ("scene_" + std::to_string(next_++) + ".nks")).string();
}

std::string SceneCache::File(const std::string& name) {
    std::filesystem::create_directories(dir_);
    return (dir_ / name).string();
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
    AnchorMaps(es->scene, path);
    BindWorld(*es);
    return es;
}

std::string RecookEditorScene(EditorScene& es, SceneCache& cache) {
    return RebuildFromDocument(es, cache,
                               [&](const char* file) { return nuka_world_rebuild_from_scene(es.world, file, 0u); });
}

std::string ResizeEditorEnvs(EditorScene& es, SceneCache& cache, uint32_t envs) {
    return RebuildFromDocument(es, cache,
                               [&](const char* file) { return nuka_world_rebuild_with_envs(es.world, file, envs); });
}

std::string RetimeEditorScene(EditorScene& es, SceneCache& cache, float dt) {
    return RebuildFromDocument(es, cache,
                               [&](const char* file) { return nuka_world_rebuild_with_dt(es.world, file, dt); });
}

bool ResetWorldPhysics(nuka_world_handle world, const std::vector<float>& rest_targets) {
    if (nuka_world_reset(world) != NUKA_RESULT_OK) return false;
    return rest_targets.empty() ||
           nuka_world_upload_field(world, NUKA_FIELD_DRIVE_TARGET, rest_targets.data(),
                                   rest_targets.size() * sizeof(float), 0u) == NUKA_RESULT_OK;
}

}  // namespace nuka::runtime::app::viewer
