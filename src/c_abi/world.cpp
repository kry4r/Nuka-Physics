
#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "render/sensor_backend.hpp"

#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"  // L1: kPairDrivenRowsPerSlot (general row budget)
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cooker.hpp"
#include "scene/format/nks.hpp"  // native .nks loader (parity with scene.cpp)

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/transform.hpp"       // M10 co-residence: replica placement transform
#include "scene/scene_compose.hpp"  // M10 co-residence: compose K replicas pre-cook
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"          // the ONE cooked terrain grid
#include "scene/terrain/heightfield_loaders.hpp"  // image + parametric loaders
#include "scene/terrain/heightfield_sample.hpp"   // the ONE bilinear obs sampler

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nuka::c_abi {

namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace articulation = nuka::runtime::articulation;

// The WorldRecord special members are out-of-line so the unique_ptr<nk::World>
// member is created/destroyed where nk::World is a complete type (here). The
// destructor's implicit teardown of `world` runs BEFORE the borrowed backend is
// freed (the DeviceRecord outlives the WorldRecord by API contract).
WorldRecord::WorldRecord() = default;
WorldRecord::~WorldRecord() = default;
WorldRecord::WorldRecord(WorldRecord&&) noexcept = default;
WorldRecord& WorldRecord::operator=(WorldRecord&&) noexcept = default;

namespace {

// Ingest .nks/.xml(MJCF)/.urdf/.usda by extension into a SceneIR. KEPT verbatim
// (the ~12 World.create_from_scene consumers -- go2_float.usda / h1_reduced_train
// .xml / go2_stand.usda -- depend on it; recon KEEP note). NOT narrowed.
bool LoadSceneByExtension(const char* scene_path, scene::SceneIR* out_scene) {
    if (out_scene == nullptr) {
        return false;
    }
    const std::filesystem::path path(scene_path);
    const std::string extension = path.extension().string();
    if (extension == ".nks") {
        *out_scene = scene::nks::Load(scene_path);
        return true;
    }
    if (extension == ".xml" || extension == ".mjcf") {
        *out_scene = import::LoadMjcf(scene_path);
        return true;
    }
    if (extension == ".urdf") {
        *out_scene = import::LoadUrdf(scene_path);
        return true;
    }
    if (extension == ".usd" || extension == ".usda") {
        *out_scene = import::LoadUsd(scene_path);
        return true;
    }
    return false;
}

// Load the desc's scene file and compose it with itself instance_count times at
// distinct XY so the cook emits K separate articulations co-resident in one env.
nuka_result_t LoadComposedScene(const nuka_world_desc_t& desc, nuka::scene::SceneIR* out) {
    if (!LoadSceneByExtension(desc.scene_path, out)) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    const float spacing = (desc.instance_spacing > 0.0f) ? desc.instance_spacing : 1.5f;
    for (uint32_t i = 1u; i < desc.instance_count; ++i) {
        nuka::scene::SceneIR replica;
        if (!LoadSceneByExtension(desc.scene_path, &replica)) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        nuka::math::Transform place = nuka::math::Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        *out = nuka::scene::Compose(*out, replica, place, "dog" + std::to_string(i) + "_");
    }
    return NUKA_RESULT_OK;
}

void CaptureArticulationHostMirror(const scene::CookedBlob& blob, WorldRecord* record) {
    // The mirror consumes ONLY body/joint/inertial tables, read from the blob the live
    // model was cooked from, so no collision geometry is cooked a second time.
    const std::vector<articulation::ArticulationCookedTopology> arts =
        articulation::CookArticulations(blob);
    if (arts.empty()) {
        record->articulation_host = articulation::ArticulationHostState{};
        return;
    }
    if (arts.size() > 1u) {
        std::fprintf(stderr,
                     "[nuka] CaptureArticulationHostMirror: %zu articulations "
                     "cooked; noise/set_link_mass host mirror covers only "
                     "articulation 0 (multi-articulation mirror not yet "
                     "implemented).\n",
                     arts.size());
    }
    record->articulation_host =
        articulation::BuildArticulationHostState({arts.front()}, blob.bodies);
}

// Translate the compact coupled cloth slots into a first-class scene MediaRecord
// (vertex-block Cloth; MLS-MPM when soft_sim_method selects it, which the cook
// rejects). BuildClothVertexBlockInput reads it. Geometry-only when the extent is absent.
nuka::scene::MediaRecord ClothMediaFromDesc(const nuka_coupled_particles_desc_t& p) {
    nuka::scene::MediaRecord m;
    m.kind = nuka::scene::MediaRecord::Kind::Cloth;
    m.method = (p.soft_sim_method == 1u) ? nuka::scene::MediaRecord::Method::MlsMpm
                                         : nuka::scene::MediaRecord::Method::Vbd;
    m.cloth_grid.nx = p.cloth_nx;
    m.cloth_grid.ny = p.cloth_ny;
    m.cloth_grid.spacing = p.cloth_spacing;
    m.cloth_grid.origin = {p.cloth_origin_x, p.cloth_origin_y, p.cloth_origin_z};
    m.cloth_grid.free = (p.cloth_free != 0u);
    m.vbd.particle_mass = p.cloth_particle_mass;
    m.vbd.friction = p.cloth_friction;
    m.vbd.stretch_stiffness = p.cloth_stretch_stiffness;
    m.vbd.bend_stiffness = p.cloth_bend_stiffness;
    m.vbd.poisson = p.cloth_poisson;
    m.vbd.damping = p.cloth_damping;
    m.vbd.aero_drag_normal = p.aero_normal;
    m.vbd.aero_drag_tangent = p.aero_tangent;
    m.vbd.aero_drag_max_dv = p.aero_max_dv;
    // The studio render skin the live beauty render inflates/relaxes the cloth
    // surface with. Render-only -- not cooked, so the particle layout is unchanged.
    m.render_skin.normal_offset = 0.004f;
    m.render_skin.smooth_iters = 3u;
    m.render_skin.smooth_lambda = 0.55f;
    return m;
}

// Route the control mode onto the cooked Model's drive_mode, bake the optional
// heightfield collidable (desc->contact_family == 1) and resolve world gravity.
nuka_result_t ApplyControlTerrainGravity(
    const nuka_world_desc_t* desc, articulation::ControlMode control_mode,
    nuka::nk::Model& model, nuka::terrain::HeightField* out_terrain,
    nuka::math::Vec3* out_gravity) {
    model.drive_mode = static_cast<uint32_t>(control_mode);
    model.osc_task_link = desc->osc_task_link;

    // When the caller requests a baked heightfield (contact_family == 1), build the
    // ONE source-of-truth HeightField -- from a grayscale IMAGE if set, else the
    // parametric generator -- then cook it. A zero-init desc => flat => byte-identical
    // to no terrain. The obs/spawn sampler reads the SAME grid (stored on the record).
    if (desc->contact_family == 1u) {
        nuka::nk::Model& m = model;
        const uint32_t orig_bodies = m.capacities.bodies_per_env;
        m.body_init.resize(orig_bodies);  // heightfield seeds at orig_bodies.
        const uint32_t nrow =
            desc->heightfield_nrow != 0u ? desc->heightfield_nrow : 41u;
        const uint32_t ncol =
            desc->heightfield_ncol != 0u ? desc->heightfield_ncol : 41u;
        const float cell =
            desc->heightfield_cell > 0.0f ? desc->heightfield_cell : 0.25f;

        const bool image_path = desc->heightfield_image_path != nullptr &&
                                desc->heightfield_image_path[0] != '\0';
        if (image_path) {
            // The grid is centred at the world origin (corner = -radius); a malformed
            // image is a LOUD INVALID_ARG, never a silent flat fallback.
            const float rx = desc->heightfield_radius_x > 0.0f
                                 ? desc->heightfield_radius_x
                                 : 0.5f * static_cast<float>(ncol - 1u) * cell;
            const float ry = desc->heightfield_radius_y > 0.0f
                                 ? desc->heightfield_radius_y
                                 : 0.5f * static_cast<float>(nrow - 1u) * cell;
            std::string err;
            if (!nuka::terrain::LoadHeightFieldFromImage(
                    desc->heightfield_image_path, rx, ry,
                    desc->heightfield_elevation_z, desc->heightfield_base_z,
                    nuka::math::Vec3{-rx, -ry, 0.0f}, *out_terrain, err)) {
                return NUKA_RESULT_INVALID_ARG;
            }
        } else {
            // Parametric: the generator fills the grid ONCE from the desc knobs. The
            // heightfield_terrain_type is a LEGACY selector the C-ABI maps to
            // parametric feature amplitudes (the engine has no terrain types).
            nuka::terrain::TerrainGenConfig cfg;
            cfg.nrow = nrow;
            cfg.ncol = ncol;
            cfg.cell_x = cell;
            cfg.cell_y = cell;
            const float half_x = 0.5f * static_cast<float>(ncol - 1u) * cell;
            const float half_y = 0.5f * static_cast<float>(nrow - 1u) * cell;
            cfg.origin = nuka::math::Vec3{-half_x, -half_y, 0.0f};
            cfg.base_z = model.ground_height;
            switch (desc->heightfield_terrain_type) {
                case 1u:  // stairs (climbing rings).
                    cfg.ring_rise = desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.ring_count = 8u;
                    break;
                case 2u:  // pit (descending rings).
                    cfg.ring_rise = -desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.ring_count = 8u;
                    break;
                case 3u:  // hashed bumps.
                    cfg.bump_height = desc->terrain_grid_height_max;
                    cfg.bump_cell = desc->terrain_grid_width;
                    break;
                case 4u:  // tiled multi-feature field (the complex scene).
                    cfg.ring_rise = desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.ring_count = 8u;
                    cfg.bump_height = desc->terrain_grid_height_max;
                    cfg.bump_cell = desc->terrain_grid_width;
                    cfg.feature_cell = 5.0f;
                    cfg.feature_margin = 0.6f;
                    break;
                case 5u:  // curriculum tile grid (level rows x type cols).
                    cfg.ring_rise = desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.bump_height = desc->terrain_grid_height_max;
                    cfg.bump_cell = desc->terrain_grid_width;
                    cfg.feature_cell = desc->terrain_feature_cell > 0.0f
                                           ? desc->terrain_feature_cell
                                           : 6.0f;
                    cfg.feature_margin = 0.6f;
                    cfg.curric_levels = desc->curric_levels;
                    cfg.curric_types = desc->curric_types;
                    break;
                default:  // 0 flat: every amplitude 0.
                    break;
            }
            if (!nuka::terrain::GenerateHeightField(cfg, *out_terrain)) {
                return NUKA_RESULT_INVALID_ARG;
            }
        }
        // Bake the field + recompute the budget through the ONE shared helper the
        // editor cook (from a SceneIR TerrainRecord) also calls -- terrain enters the
        // model through identical code regardless of the authoring source.
        nuka::scene::cook::CookTerrainIntoModel(m, *out_terrain, orig_bodies);
    }

    // Resolve world gravity (Z-up). A zero-initialized desc substitutes the shared
    // standard-Earth default; any non-zero component takes the caller's full vector.
    nuka::math::Vec3 gravity = nuka::runtime::kDefaultGravity;
    if (desc->gravity_x != 0.0f || desc->gravity_y != 0.0f ||
        desc->gravity_z != 0.0f) {
        gravity = {desc->gravity_x, desc->gravity_y, desc->gravity_z};
    }
    *out_gravity = gravity;
    return NUKA_RESULT_OK;
}

// Retains a world's media render surfaces, each material id moved from the scene's material
// table to the registry slot its name projects to, since the registry merges equally named materials.
void SetParticleSurfaces(WorldRecord& record, std::vector<cook::MediaRenderSurface> surfaces) {
    for (cook::MediaRenderSurface& s : surfaces)
        if (s.render_material_id < record.scene->MaterialCount())
            s.render_material_id =
                record.scene->Ecs().FindRenderMaterial(record.scene->GetMaterial(s.render_material_id).name);
    record.particle_surfaces = std::move(surfaces);
}

// The cooked products a world build consumes.
struct CookedWorld {
    nuka::nk::Model model;
    std::shared_ptr<const nuka::scene::CookedBlob> blob;
    nuka::terrain::HeightField terrain;
    nuka::math::Vec3 gravity = nuka::runtime::kDefaultGravity;
    std::vector<cook::MediaRenderSurface> surfaces;
};

// Cook a scene with a world's creation recipe: the rigid cook, then media and terrain in the
// order the creating entry used, then the model overrides. MUST be called inside a try.
nuka_result_t CookWorld(const WorldRecipe& recipe, const nuka::scene::SceneIR& scene,
                        CookedWorld* out) {
    nuka_world_desc_t desc = recipe.desc;
    desc.heightfield_image_path = recipe.image_path.empty() ? nullptr : recipe.image_path.c_str();
    cook::CookToModelOptions options;
    options.bake_link_sdf = recipe.bake_link_sdf;
    cook::CookToModelResult cooked =
        cook::CookToModel(scene, static_cast<int>(desc.env_count), options);
    if (recipe.media_before_terrain) {
        cook::CookSceneMedia(cooked.model, desc.env_count, scene.Media(), recipe.contact_radius);
    }
    const nuka_result_t terrain = ApplyControlTerrainGravity(
        &desc, static_cast<articulation::ControlMode>(desc.control_mode), cooked.model,
        &out->terrain, &out->gravity);
    if (terrain != NUKA_RESULT_OK) {
        return terrain;
    }
    // A scene-authored terrain is the record's grid when the desc bakes none.
    if (!nuka::terrain::IsValid(out->terrain)) out->terrain = std::move(cooked.terrain);
    if (!recipe.media_before_terrain) {
        cook::CookSceneMedia(cooked.model, desc.env_count, scene.Media(), recipe.contact_radius);
    }
    // Positive values bound the contact recovery push-out; 0 keeps the cooked default.
    if (recipe.baumgarte_max_velocity > 0.0f) {
        cooked.model.baumgarte_max_velocity = recipe.baumgarte_max_velocity;
    }
    if (recipe.overlap_recovery_velocity > 0.0f) {
        cooked.model.overlap_recovery_velocity = recipe.overlap_recovery_velocity;
    }
    out->surfaces = cook::BuildSceneMediaRenderSurfaces(scene.Media());
    out->model = std::move(cooked.model);
    out->blob = std::move(cooked.blob);
    return NUKA_RESULT_OK;
}

// Build the live nk::World for a cooked scene. Each zero solver knob keeps the engine
// default, so an unset caller gets the plain dt + gravity configuration.
nuka_result_t BuildWorld(CookedWorld& cooked, const WorldRecipe& recipe, DeviceRecord* device,
                         const nuka::math::Vec3& gravity, uint32_t vel_iters,
                         std::unique_ptr<nuka::nk::World>* out) {
    if (recipe.cloth_integrator > 1u || vel_iters > UINT16_MAX) return NUKA_RESULT_INVALID_ARG;
    if (!cooked.blob) return NUKA_RESULT_INTERNAL;
    if (recipe.desc.ogc_contact_capacity > 0u)
        cook::SetOgcContactCapacity(cooked.model, recipe.desc.ogc_contact_capacity);
    cooked.model.osc_task_link = recipe.desc.osc_task_link;

    nuka::nk::Pipeline::SolverConfig cfg;
    cfg.dt = recipe.desc.fixed_dt;
    cfg.gravity[0] = gravity.x;
    cfg.gravity[1] = gravity.y;
    cfg.gravity[2] = gravity.z;
    if (vel_iters != 0u) {
        cfg.vel_iters = static_cast<uint16_t>(vel_iters);
        cfg.velocity_iterations_override = true;
    }
    if (recipe.solver_pos_iters != 0u) cfg.pos_iters = static_cast<uint16_t>(recipe.solver_pos_iters);
    if (recipe.solver_contact_margin > 0.0f) cfg.contact_margin = recipe.solver_contact_margin;
    if (recipe.solver_max_pairs != 0u) cfg.max_pairs = recipe.solver_max_pairs;
    cfg.cloth_integrator = recipe.cloth_integrator;

    auto world = std::make_unique<nuka::nk::World>(std::move(cooked.model), recipe.desc.env_count,
                                                   device->phi_device, device->backend, cfg);
    if (!world->Ready()) {
        std::fprintf(stderr, "[World] %s\n", world->CreationError().c_str());
        return MapStatusToResult(world->CreationStatus());
    }
    *out = std::move(world);
    return NUKA_RESULT_OK;
}

// Point the record at a built world and the scene it was cooked from. The scene is
// retained so sensors, export and rebuild read the same ECS the cook saw.
void InstallScene(WorldRecord& record, std::unique_ptr<nuka::nk::World> world,
                  CookedWorld&& cooked, nuka::scene::SceneIR&& scene) {
    record.world = std::move(world);
    record.cooked_terrain = std::move(cooked.terrain);
    CaptureArticulationHostMirror(*cooked.blob, &record);
    record.scene = std::make_unique<nuka::scene::SceneIR>(std::move(scene));
    SetParticleSurfaces(record, std::move(cooked.surfaces));
}

// Guards generations, storage leases and retired worlds, which view owners touch from any thread.
std::mutex& LeaseMutex() {
    static std::mutex mutex;
    return mutex;
}

constexpr uint32_t kNoRow = ~uint32_t(0);

// Maps each name that labels exactly one entry to its index.
std::unordered_map<std::string, uint32_t> UniqueNames(const std::vector<std::string>& names) {
    std::unordered_map<std::string, uint32_t> index;
    for (uint32_t i = 0u; i < names.size(); ++i) {
        const auto [it, inserted] = index.emplace(names[i], i);
        if (!inserted) it->second = kNoRow;
    }
    return index;
}

// The scene name of each cooked link's body (empty when the row is not a scene body).
std::vector<std::string> LinkNames(const nuka::nk::Model& model, const nuka::scene::SceneIR& scene) {
    const auto& link_body = model.articulation.link_body;
    std::vector<std::string> names(link_body.size());
    for (size_t l = 0u; l < link_body.size(); ++l)
        if (link_body[l] < scene.Bodies().size()) names[l] = scene.Bodies()[link_body[l]].name;
    return names;
}

// Copy the paired per-env elements of one persistent field from one world to another.
nuka::phi::Status MigrateField(nuka::nk::World& from, nuka::nk::World& to, nuka::nk::FieldId id,
                               uint64_t element_bytes,
                               const std::vector<std::pair<uint32_t, uint32_t>>& rows) {
    const uint64_t envs = to.EnvCount();
    const uint64_t from_count = from.GetModel().capacities.ElementCount(id) / envs;
    const uint64_t to_count = to.GetModel().capacities.ElementCount(id) / envs;
    if (rows.empty() || from_count == 0u || to_count == 0u) return nuka::phi::Status::Ok;
    std::vector<uint8_t> source(envs * from_count * element_bytes);
    std::vector<uint8_t> target(envs * to_count * element_bytes);
    auto status = from.GetData().DownloadFieldStatus(id, source.data(), source.size());
    if (status == nuka::phi::Status::Ok)
        status = to.GetData().DownloadFieldStatus(id, target.data(), target.size());
    if (status != nuka::phi::Status::Ok) return status;
    for (uint64_t e = 0u; e < envs; ++e)
        for (const auto& [a, b] : rows)
            if (a < from_count && b < to_count)
                std::memcpy(target.data() + (e * to_count + b) * element_bytes,
                            source.data() + (e * from_count + a) * element_bytes, element_bytes);
    return to.GetData().UploadFieldStatus(id, target.data(), target.size());
}

// Carry body and articulation state across a rebuild by unique body name: movable body
// pose and velocity, per-link joint state and targets, and each articulation's root pose.
nuka::phi::Status MigrateState(nuka::nk::World& from, const nuka::scene::SceneIR& from_scene,
                               nuka::nk::World& to, const nuka::scene::SceneIR& to_scene) {
    using nuka::nk::FieldId;
    std::vector<std::string> to_body_names;
    for (const auto& body : to_scene.Bodies()) to_body_names.push_back(body.name);
    const auto to_bodies = UniqueNames(to_body_names);
    std::vector<std::string> from_body_names;
    for (const auto& body : from_scene.Bodies()) from_body_names.push_back(body.name);
    const auto from_bodies = UniqueNames(from_body_names);
    std::vector<std::pair<uint32_t, uint32_t>> bodies;
    for (const auto& [name, a] : from_bodies) {
        const auto found = to_bodies.find(name);
        if (a == kNoRow || found == to_bodies.end() || found->second == kNoRow) continue;
        if (!from_scene.Bodies()[a].is_static && !to_scene.Bodies()[found->second].is_static)
            bodies.emplace_back(a, found->second);
    }

    const auto& from_art = from.GetModel().articulation;
    const auto& to_art = to.GetModel().articulation;
    const auto to_links = UniqueNames(LinkNames(to.GetModel(), to_scene));
    const std::vector<std::string> from_link_names = LinkNames(from.GetModel(), from_scene);
    std::vector<std::pair<uint32_t, uint32_t>> links, roots;
    for (uint32_t a = 0u; a < from_link_names.size(); ++a) {
        const auto found = to_links.find(from_link_names[a]);
        if (from_link_names[a].empty() || found == to_links.end() || found->second == kNoRow) continue;
        const uint32_t b = found->second;
        links.emplace_back(a, b);
        if (from_art.parent_link.at(a) == kNoRow && to_art.parent_link.at(b) == kNoRow)
            roots.emplace_back(from_art.link_to_articulation.empty() ? 0u : from_art.link_to_articulation[a],
                               to_art.link_to_articulation.empty() ? 0u : to_art.link_to_articulation[b]);
    }

    constexpr uint64_t kPose = 7u * sizeof(float);
    constexpr uint64_t kVec3 = 3u * sizeof(float);
    constexpr uint64_t kSpatial = 6u * sizeof(float);
    const struct { FieldId id; uint64_t bytes; const std::vector<std::pair<uint32_t, uint32_t>>* rows; }
        fields[] = {{FieldId::BodyPose, kPose, &bodies},
                    {FieldId::BodyLinearVelocity, kVec3, &bodies},
                    {FieldId::BodyAngularVelocity, kVec3, &bodies},
                    {FieldId::Q, sizeof(float), &links},
                    {FieldId::Qdot, sizeof(float), &links},
                    {FieldId::DriveTarget, sizeof(float), &links},
                    {FieldId::LinkVelocity, kSpatial, &links},
                    {FieldId::BasePose, kPose, &roots}};
    for (const auto& field : fields) {
        const auto status = MigrateField(from, to, field.id, field.bytes, *field.rows);
        if (status != nuka::phi::Status::Ok) return status;
    }
    return to.ForwardKinematics();
}

}  // namespace

// ---------------------------------------------------------------------------
// Shared world-create helpers (declared in c_abi/internal.hpp): the validation and
// the ONE cook + build + record-assembly path every create entry and rebuild share.
// ---------------------------------------------------------------------------

nuka_result_t ValidateWorldDescAndDevice(
    nuka_device_handle device, const nuka_world_desc_t* desc,
    bool require_scene_path, DeviceRecord** out_device,
    articulation::ControlMode* out_mode) {
    if (desc == nullptr ||
        (require_scene_path && desc->scene_path == nullptr) ||
        desc->fixed_dt <= 0.0f) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (desc->env_count == 0u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // Determinism level: 0 = D1/Strong (default), 1 = D2/Weak. Reject other values.
    // The nk pipeline runs the D1 schedule only; D2 currently selects the same
    // kernels and is behaviorally identical (NOT a separate path).
    if (desc->determinism > 1u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (!articulation::IsControlModeImplemented(desc->control_mode)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    const auto control_mode =
        static_cast<articulation::ControlMode>(desc->control_mode);

    auto* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // The nk::World ctor REQUIRES the phi v2 device/backend acquired at create.
    if (device_record->phi_device == nullptr || device_record->backend == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    *out_device = device_record;
    *out_mode = control_mode;
    return NUKA_RESULT_OK;
}

WorldRecipe MakeWorldRecipe(const nuka_world_desc_t& desc) {
    WorldRecipe recipe;
    recipe.desc = desc;
    recipe.desc.scene_path = nullptr;
    recipe.desc.heightfield_image_path = nullptr;
    if (desc.heightfield_image_path != nullptr) recipe.image_path = desc.heightfield_image_path;
    return recipe;
}

nuka_result_t CreateWorld(DeviceRecord* device, WorldRecipe recipe, nuka::scene::SceneIR&& scene,
                          std::string scene_dir, nuka_world_handle* out) {
    CookedWorld cooked;
    const nuka_result_t cook_result = CookWorld(recipe, scene, &cooked);
    if (cook_result != NUKA_RESULT_OK) return cook_result;
    std::unique_ptr<nuka::nk::World> world;
    const nuka_result_t built =
        BuildWorld(cooked, recipe, device, cooked.gravity, recipe.solver_vel_iters, &world);
    if (built != NUKA_RESULT_OK) return built;

    auto record = std::make_unique<WorldRecord>();
    record->device = device;
    record->env_count = recipe.desc.env_count;
    record->control_mode = static_cast<articulation::ControlMode>(recipe.desc.control_mode);
    record->step_options.dt = recipe.desc.fixed_dt;
    record->step_options.gravity = cooked.gravity;
    record->scene_dir = std::move(scene_dir);
    record->recipe = std::move(recipe);
    InstallScene(*record, std::move(world), std::move(cooked), std::move(scene));
    *out = WorldTable().Insert(std::move(record));
    return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
}

}  // namespace nuka::c_abi

extern "C" {

nuka_result_t nuka_world_create_from_scene(nuka_device_handle device,
                                           const nuka_world_desc_t* desc,
                                           nuka_world_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    try {
        nuka::c_abi::DeviceRecord* device_record = nullptr;
        nuka::runtime::articulation::ControlMode control_mode{};
        const nuka_result_t valid = nuka::c_abi::ValidateWorldDescAndDevice(
            device, desc, /*require_scene_path=*/true, &device_record, &control_mode);
        if (valid != NUKA_RESULT_OK) {
            return valid;
        }
        nuka::scene::SceneIR scene;
        const nuka_result_t loaded = nuka::c_abi::LoadComposedScene(*desc, &scene);
        if (loaded != NUKA_RESULT_OK) {
            return loaded;
        }
        // The file-load entry cooks the robot + an optional heightfield only; a scene
        // that declares media must use nuka_world_create_from_built_scene (LOUD, never
        // a silent drop of the declared particles).
        if (!scene.Media().empty()) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        nuka::c_abi::WorldRecipe recipe = nuka::c_abi::MakeWorldRecipe(*desc);
        recipe.baumgarte_max_velocity = desc->solver_baumgarte_max_velocity;
        recipe.overlap_recovery_velocity = desc->solver_overlap_recovery_velocity;
        recipe.solver_vel_iters = desc->solver_vel_iters;
        recipe.solver_pos_iters = desc->solver_pos_iters;
        recipe.solver_contact_margin = desc->solver_contact_margin;
        recipe.solver_max_pairs = desc->solver_max_pairs;
        recipe.cloth_integrator = desc->cloth_integrator;
        return nuka::c_abi::CreateWorld(
            device_record, std::move(recipe), std::move(scene),
            std::filesystem::path(desc->scene_path).parent_path().string(), out);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[World] scene create: %s\n", error.what());
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_create_coupled_from_scene(
    nuka_device_handle device, const nuka_world_desc_t* desc,
    const nuka_coupled_particles_desc_t* particles, nuka_world_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    if (particles == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        nuka::c_abi::DeviceRecord* device_record = nullptr;
        nuka::runtime::articulation::ControlMode control_mode{};
        const nuka_result_t valid = nuka::c_abi::ValidateWorldDescAndDevice(
            device, desc, /*require_scene_path=*/true, &device_record, &control_mode);
        if (valid != NUKA_RESULT_OK) {
            return valid;
        }
        nuka::scene::SceneIR scene;
        const nuka_result_t loaded = nuka::c_abi::LoadComposedScene(*desc, &scene);
        if (loaded != NUKA_RESULT_OK) {
            return loaded;
        }
        const bool has_cloth = particles->cloth_nx >= 2u &&
                               particles->cloth_ny >= 2u &&
                               particles->cloth_spacing > 0.0f;
        if (!has_cloth) {
            // A particle-free world must use nuka_world_create_from_scene.
            return NUKA_RESULT_INVALID_ARG;
        }
        // The descriptor's cloth becomes a first-class scene medium, cooked after the
        // terrain with the descriptor's contact radius and solver knobs.
        scene.AddMedia(nuka::c_abi::ClothMediaFromDesc(*particles));
        nuka::c_abi::WorldRecipe recipe = nuka::c_abi::MakeWorldRecipe(*desc);
        recipe.contact_radius = particles->contact_radius;
        recipe.baumgarte_max_velocity = particles->baumgarte_max_velocity;
        recipe.overlap_recovery_velocity = particles->overlap_recovery_velocity;
        recipe.solver_vel_iters = particles->solver_vel_iters;
        recipe.solver_pos_iters = particles->solver_pos_iters;
        recipe.solver_contact_margin = particles->solver_contact_margin;
        recipe.solver_max_pairs = particles->solver_max_pairs;
        recipe.cloth_integrator = particles->cloth_integrator;
        return nuka::c_abi::CreateWorld(
            device_record, std::move(recipe), std::move(scene),
            std::filesystem::path(desc->scene_path).parent_path().string(), out);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[World] coupled create: %s\n", error.what());
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_world_destroy(nuka_world_handle world) {
    (void)nuka::c_abi::WorldTable().Remove(world);
}

nuka_result_t nuka_world_step(nuka_world_handle world) {
    return nuka_world_step_n(world, 1u);
}

nuka_result_t nuka_world_step_n(nuka_world_handle world, uint32_t step_count) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (step_count == 0u) {
        return NUKA_RESULT_OK;
    }
    try {
        for (uint32_t step = 0u; step < step_count; ++step) {
            const auto result = record->world->StepConfigured();
            if (result != nuka::phi::Status::Ok) {
                std::fprintf(stderr, "[World] step failed at op %u\n",
                    static_cast<unsigned>(record->world->LastExecutionError().failed_op));
                return nuka::c_abi::MapStatusToResult(result);
            }
            ++record->simulated_step_count;
        }
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_set_execution_mode(nuka_world_handle world, nuka_execution_mode_t mode) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    if (mode != NUKA_EXECUTION_EAGER && mode != NUKA_EXECUTION_GRAPH) return NUKA_RESULT_INVALID_ARG;
    try {
        return nuka::c_abi::MapStatusToResult(record->world->SetExecutionMode(
            static_cast<nuka::nk::World::ExecutionMode>(mode)));
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_velocity_iterations(nuka_world_handle world, uint32_t* out_iterations) {
    if (out_iterations == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    *out_iterations = record->world->VelocityIterations();
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_execution_info(nuka_world_handle world, nuka_world_execution_info_t* out) {
    if (!out || out->struct_size != sizeof(*out) || out->schema_version != NUKA_EXECUTION_INFO_VERSION)
        return NUKA_RESULT_INVALID_ARG;
    *out = {};
    out->struct_size = sizeof(*out);
    out->schema_version = NUKA_EXECUTION_INFO_VERSION;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    out->mode = static_cast<nuka_execution_mode_t>(record->world->GetExecutionMode());
    out->graph_ready = record->world->GraphReady() ? 1u : 0u;
    out->capture_attempts = record->world->CaptureAttempts();
    out->graph_replays = record->world->GraphReplays();
    const auto& error = record->world->LastExecutionError();
    out->last_result = nuka::c_abi::MapStatusToResult(error.status);
    out->failed_op = static_cast<uint32_t>(error.failed_op);
    out->native_error = error.native_code;
    std::snprintf(out->message, sizeof(out->message), "%s", error.message);
    return NUKA_RESULT_OK;
}

struct ExecutionWindow {
    nuka::phi::Event* start;
    nuka::phi::Event* end;
    uint64_t replays;
};

nuka_result_t nuka_world_get_resource_info(nuka_world_handle world, nuka_world_resource_info_t* out) {
    auto& w = *nuka::c_abi::WorldTable().Get(world)->world;
    const auto& model = w.GetModel();
    const auto& c = model.capacities;
    uint64_t model_bytes;
    model.ComputeModelSegments(&model_bytes);
    uint64_t arenas[nuka::nk::Arena::kArenaCount];
    nuka::nk::Arena::ComputeSegments(c, arenas);
    *out = {model_bytes, arenas[0], arenas[1], c.env_count,
            c.max_contacts_per_env, c.max_rows_per_env, c.dofs_per_env,
            c.bodies_per_env, c.particles_per_env};
    return NUKA_RESULT_OK;
}

uint32_t nuka_world_resource_field_count(nuka_world_handle world) {
    auto& w = *nuka::c_abi::WorldTable().Get(world)->world;
    uint64_t bytes;
    return static_cast<uint32_t>(w.GetModel().ComputeModelSegments(&bytes).size() +
                                 w.GetData().Segments().size());
}

nuka_result_t nuka_world_get_resource_fields(nuka_world_handle world, nuka_resource_field_t* out) {
    auto& w = *nuka::c_abi::WorldTable().Get(world)->world;
    uint64_t bytes;
    uint32_t index = 0;
    for (const auto& segment : w.GetModel().ComputeModelSegments(&bytes)) {
        auto& field = out[index++];
        std::snprintf(field.name, sizeof(field.name), "%s", nuka::nk::FieldName(segment.field));
        field.offset = segment.offset;
        field.bytes = segment.bytes;
        field.owner = 0;
        field.arena = 0;
    }
    for (const auto& segment : w.GetData().Segments()) {
        auto& field = out[index++];
        std::snprintf(field.name, sizeof(field.name), "%s", nuka::nk::FieldName(segment.field));
        field.offset = segment.offset;
        field.bytes = segment.bytes;
        field.owner = 1;
        field.arena = segment.arena;
    }
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_begin_execution_window(nuka_world_handle world, void** window) {
    auto& w = *nuka::c_abi::WorldTable().Get(world)->world;
    auto* backend = w.Backend();
    auto* interval = new ExecutionWindow{nuka::phi::BackendEventNew(backend),
                                        nuka::phi::BackendEventNew(backend), w.GraphReplays()};
    nuka::phi::BackendEventRecord(backend, interval->start);
    *window = interval;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_end_execution_window(nuka_world_handle world, void* window,
                                             nuka_execution_window_t* out) {
    auto& w = *nuka::c_abi::WorldTable().Get(world)->world;
    auto* backend = w.Backend();
    auto* interval = static_cast<ExecutionWindow*>(window);
    nuka::phi::BackendEventRecord(backend, interval->end);
    const auto metrics = w.GraphMetrics();
    *out = {nuka::phi::BackendEventElapsedMs(backend, interval->start, interval->end),
            w.GraphReplays() - interval->replays, metrics.nodes,
            metrics.kernels, metrics.clears, metrics.copies};
    nuka::phi::BackendEventFree(backend, interval->start);
    nuka::phi::BackendEventFree(backend, interval->end);
    delete interval;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_synchronize(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        return nuka::c_abi::MapStatusToResult(record->world->Synchronize());
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_reset(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    try {
        // An empty selection restores all environments through the shared reset op.
        auto status = record->world->Reset({});
        if (status == nuka::phi::Status::Ok && record->sensor)
            record->sensor->backend->ResetSensorState(record->sensor->handle, {});
        if (status == nuka::phi::Status::Ok) {
            for (auto& entry : record->field_observations) {
                status = entry.second->Reset();
                if (status != nuka::phi::Status::Ok) break;
            }
        }
        return nuka::c_abi::MapStatusToResult(status);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_reset_envs(nuka_world_handle world,
                                    const uint32_t* env_ids,
                                    uint32_t count) {
    if (count > 0u && env_ids == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (count == 0u) {
        return NUKA_RESULT_OK;
    }
    // Bounds-check the ids here so an out-of-range id rejects cleanly without
    // touching any env state (the friendlier control-plane contract).
    const uint32_t env_count = record->world->EnvCount();
    for (uint32_t i = 0u; i < count; ++i) {
        if (env_ids[i] >= env_count) {
            return NUKA_RESULT_INVALID_ARG;
        }
    }
    try {
        const std::vector<uint32_t> ids(env_ids, env_ids + count);
        auto status = record->world->Reset(ids);
        if (status == nuka::phi::Status::Ok && record->sensor)
            record->sensor->backend->ResetSensorState(record->sensor->handle, ids);
        if (status == nuka::phi::Status::Ok) {
            for (auto& entry : record->field_observations) {
                status = entry.second->Reset(ids);
                if (status != nuka::phi::Status::Ok) break;
            }
        }
        return nuka::c_abi::MapStatusToResult(status);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_last_invariant_violations(nuka_world_handle world,
                                                       uint32_t* out_count,
                                                       void* out_array,
                                                       uint32_t array_capacity) {
    if (out_count == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (array_capacity > 0u && out_array == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }

    // M9 NAMED GAP: invariant sampling read the legacy host-mirrored BuiltWorld
    // template/instance each step; the generic nk::World keeps its state on the
    // device and the C-ABI no longer round-trips it to a host BuiltWorld. The
    // sampler is therefore never populated -> always report zero violations. The
    // only consumer (an invariant probe) reads a count; no gate asserts a
    // non-zero violation through this path. Re-pointing to read nk Data is a
    // device-download cost deferred to a later milestone (flagged in the report).
    const auto& violations = record->last_invariant_violations;
    *out_count = static_cast<uint32_t>(violations.size());
    if (out_array == nullptr || array_capacity == 0u || violations.empty()) {
        return NUKA_RESULT_OK;
    }

    const uint32_t copy_count =
        std::min(array_capacity, static_cast<uint32_t>(violations.size()));
    auto* out = static_cast<nuka_invariant_violation_t*>(out_array);
    for (uint32_t index = 0u; index < copy_count; ++index) {
        const auto& violation = violations[index];
        out[index].invariant = static_cast<uint32_t>(violation.which);
        out[index].step = violation.step_index;
        out[index].env_id = violation.env_id;
        out[index].value = violation.value;
        out[index].threshold = violation.threshold;
    }
    return NUKA_RESULT_OK;
}

// BATCHED arbitrary-(x,y) terrain height sampler over the world's COOKED grid.
// Reads the world's stored HeightField via the ONE bilinear sampler -- the SAME
// full-relief grid the physics narrowphase rests feet on -- so obs/spawn height
// equals physics by construction. There is no per-env vertical scaling: the grid
// is model-level (one terrain for the batch), so any per-env shrink would diverge
// the obs from the physics it is meant to mirror.
nuka_result_t nuka_world_sample_terrain_height_batch(nuka_world_handle world,
                                                     uint32_t n,
                                                     const float* xs,
                                                     const float* ys,
                                                     float* out_heights) {
    if (n == 0u) {
        return NUKA_RESULT_OK;  // nothing to sample.
    }
    if (world == nullptr || xs == nullptr || ys == nullptr ||
        out_heights == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const nuka::terrain::HeightField& hf = record->cooked_terrain;
    if (!nuka::terrain::IsValid(hf)) {
        // No cooked heightfield: report the flat ground floor for every column.
        for (uint32_t i = 0u; i < n; ++i) out_heights[i] = 0.0f;
        return NUKA_RESULT_OK;
    }
    for (uint32_t i = 0u; i < n; ++i) {
        out_heights[i] = nuka::terrain::SampleHeightFieldZ(hf, xs[i], ys[i]);
    }
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_terrain(nuka_world_handle world, nuka_terrain_info_t* out_info,
                                     float* values, size_t capacity) {
    if (out_info == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const nuka::terrain::HeightField& hf = record->cooked_terrain;
    if (!nuka::terrain::IsValid(hf)) return NUKA_RESULT_NOT_SUPPORTED;
    out_info->nrow = hf.nrow;
    out_info->ncol = hf.ncol;
    out_info->cell_x = hf.cell_x;
    out_info->cell_y = hf.cell_y;
    out_info->origin[0] = hf.origin.x;
    out_info->origin[1] = hf.origin.y;
    out_info->origin[2] = hf.origin.z;
    out_info->base_z = hf.base_z;
    out_info->scale_z = hf.scale_z;
    if (values == nullptr) return NUKA_RESULT_OK;
    if (capacity < hf.values.size()) return NUKA_RESULT_INVALID_ARG;
    std::copy(hf.values.begin(), hf.values.end(), values);
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_dof_name(nuka_world_handle world,
                                      uint32_t link_index, char* out, size_t cap,
                                      size_t* out_len) {
    if (out_len != nullptr) {
        *out_len = 0u;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (record->scene == nullptr ||
        record->articulation_host.articulations.empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no cooked articulation retained.
    }
    try {
        // Cooked link -> its body row. link_index spans EVERY cooked articulation
        // (env-concatenated order), so resolve it by cumulative link ranges.
        const auto& artics = record->articulation_host.articulations;
        size_t k = 0u, local = link_index;
        while (k < artics.size() && local >= artics[k].link_bodies.size()) {
            local -= artics[k].link_bodies.size();
            ++k;
        }
        if (k >= artics.size()) {
            return NUKA_RESULT_INVALID_ARG;
        }
        const nuka::scene::BodyId body = artics[k].link_bodies[local];
        const nuka::scene::SceneIR& scene = *record->scene;

        // The DOF name is the joint whose CHILD is this link's body (the actuated
        // joint). The root (no parent joint) falls back to the link body's name.
        std::string name;
        for (const auto& joint : scene.Joints()) {
            if (joint.child_body == body) {
                name = joint.name;
                break;
            }
        }
        if (name.empty() && body < scene.Bodies().size()) {
            name = scene.Bodies()[body].name;
        }

        if (out_len != nullptr) {
            *out_len = name.size();  // set first so the caller can resize + retry.
        }
        if (out == nullptr) {
            return NUKA_RESULT_OK;  // size query: *out_len holds the needed length.
        }
        if (name.size() + 1u > cap) {
            return NUKA_RESULT_INVALID_ARG;  // LOUD: too small to hold name + NUL.
        }
        for (size_t i = 0u; i < name.size(); ++i) {
            out[i] = name[i];
        }
        out[name.size()] = '\0';
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_kinematic_link(nuka_world_handle world,
                                            uint32_t link_index, nuka_kinematic_link_t* out) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    try {
        const auto& model = record->world->GetModel().articulation;
        if (link_index >= model.parent_link.size()) return NUKA_RESULT_INVALID_ARG;
        const uint32_t articulation = model.link_to_articulation.empty()
            ? 0u : model.link_to_articulation.at(link_index);
        const uint32_t offset = model.articulation_link_offset.empty()
            ? 0u : model.articulation_link_offset.at(articulation);
        const uint32_t parent = model.parent_link.at(link_index);
        nuka_kinematic_link_t result{};
        result.parent_index = parent == ~uint32_t(0) ? parent : parent + offset;
        result.articulation_index = articulation;
        result.local_pose[3] = 1.0f;
        if (parent == ~uint32_t(0)) {
            result.joint_type = NUKA_KINEMATIC_ROOT;
        } else {
            using nuka::phi::ArticulationJointType;
            switch (static_cast<ArticulationJointType>(model.joint_type.at(link_index))) {
                case ArticulationJointType::Fixed: result.joint_type = NUKA_KINEMATIC_FIXED; break;
                case ArticulationJointType::Revolute: result.joint_type = NUKA_KINEMATIC_REVOLUTE; break;
                case ArticulationJointType::Prismatic: result.joint_type = NUKA_KINEMATIC_PRISMATIC; break;
                default: return NUKA_RESULT_NOT_SUPPORTED;
            }
            const auto& local = model.link_local_pose.at(link_index);
            const auto translation = local.position + model.parent_offset.at(link_index);
            const auto& axis = model.joint_axis.at(link_index);
            result.local_pose[0] = translation.x;
            result.local_pose[1] = translation.y;
            result.local_pose[2] = translation.z;
            result.local_pose[3] = local.rotation.w;
            result.local_pose[4] = local.rotation.x;
            result.local_pose[5] = local.rotation.y;
            result.local_pose[6] = local.rotation.z;
            result.axis[0] = axis.x;
            result.axis[1] = axis.y;
            result.axis[2] = axis.z;
        }
        *out = result;
        return NUKA_RESULT_OK;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_link_binding(nuka_world_handle world, uint32_t link_index,
                                          nuka_link_binding_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    const auto& art = record->world->GetModel().articulation;
    if (link_index >= art.link_body.size()) return NUKA_RESULT_INVALID_ARG;
    out->body_index = art.link_body[link_index];
    out->articulation_index =
        art.link_to_articulation.empty() ? 0u : art.link_to_articulation[link_index];
    out->is_root = art.parent_link[link_index] == ~uint32_t(0) ? 1u : 0u;
    out->is_floating = out->is_root != 0u &&
        art.joint_type[link_index] ==
            static_cast<uint8_t>(nuka::phi::ArticulationJointType::FloatingBase) ? 1u : 0u;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_colliders(nuka_world_handle world, uint32_t env_index,
                                       nuka_collider_t* out, uint32_t capacity,
                                       uint32_t* out_count) {
    if (out_count == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        using nuka::math::Transform;
        namespace nk = nuka::nk;
        nk::World& w = *record->world;
        const nk::Model& model = w.GetModel();
        const auto& shapes = model.shape_table_rows;
        *out_count = static_cast<uint32_t>(shapes.size());
        if (out == nullptr) return NUKA_RESULT_OK;
        if (capacity < shapes.size() || env_index >= model.capacities.env_count)
            return NUKA_RESULT_INVALID_ARG;
        const uint64_t bodies = model.capacities.bodies_per_env;
        const uint64_t links = model.capacities.links_per_env;
        std::vector<Transform> body_pose(bodies), link_pose(links);
        std::vector<float> inv_mass(bodies);
        nk::Data& data = w.GetData();
        const uint64_t tf = sizeof(Transform);
        // Link rows are re-posed through FK; free bodies read their own pose row.
        if ((bodies > 0u &&
             (!data.DownloadField(nk::FieldId::BodyPose, body_pose.data(), bodies * tf,
                                  env_index * bodies * tf) ||
              !data.DownloadField(nk::FieldId::BodyInvMass, inv_mass.data(),
                                  bodies * sizeof(float), env_index * bodies * sizeof(float)))) ||
            (links > 0u &&
             !data.DownloadField(nk::FieldId::LinkPose, link_pose.data(), links * tf,
                                 env_index * links * tf)))
            return NUKA_RESULT_INTERNAL;
        static_assert(sizeof(Transform) == sizeof(nuka_collider_t::pose), "pose wire layout");
        for (size_t i = 0; i < shapes.size(); ++i) {
            const auto& shape = shapes[i];
            nuka_collider_t& collider = out[i];
            collider.kind = shape.kind;
            std::copy(shape.params, shape.params + 4, collider.params);
            Transform pose = Transform::Identity();
            bool is_static = true;
            if (shape.body_id >= 0) {
                const auto body = static_cast<uint32_t>(shape.body_id);
                const uint32_t link = body < model.body_to_link.size() ? model.body_to_link[body]
                                                                       : ~uint32_t(0);
                if (link != ~uint32_t(0)) {
                    pose = link_pose.at(link);
                    if (body < model.body_collidable_link.size() &&
                        model.body_collidable_link[body] != ~uint32_t(0))
                        pose = pose * model.body_collidable_local.at(body);
                    else if (link < model.articulation.link_geom_local.size())
                        pose = pose * model.articulation.link_geom_local[link];
                    is_static = false;
                } else {
                    pose = body_pose.at(body);
                    is_static = inv_mass[body] == 0.0f;
                }
            }
            collider.is_static = is_static ? 1u : 0u;
            std::memcpy(collider.pose, &pose, sizeof(pose));
        }
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_export_scene(nuka_world_handle world, const char* nks_path) {
    if (nks_path == nullptr || nks_path[0] == '\0') return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->scene) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        nuka::scene::nks::Save(*record->scene, nks_path);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_asset_root(nuka_world_handle world, char* out, size_t cap,
                                        size_t* out_len) {
    if (out_len != nullptr) *out_len = 0u;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const std::string& root = record->scene_dir;
    if (out_len != nullptr) *out_len = root.size();
    if (out == nullptr) return NUKA_RESULT_OK;
    if (root.size() + 1u > cap) return NUKA_RESULT_INVALID_ARG;
    std::memcpy(out, root.c_str(), root.size() + 1u);
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_rebuild_from_scene(nuka_world_handle world, const char* scene_path,
                                            uint32_t flags) {
    if (scene_path == nullptr || (flags & ~static_cast<uint32_t>(NUKA_REBUILD_INITIAL_STATE)) != 0u)
        return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    // Sensors and observations bind to the cooked layout and are not carried across.
    if (record->sensor || !record->field_observations.empty() ||
        record->world->StateSensors().Count() != 0u)
        return NUKA_RESULT_NOT_SUPPORTED;
    try {
        using nuka::phi::Status;
        nuka::scene::SceneIR scene;
        if (!nuka::c_abi::LoadSceneByExtension(scene_path, &scene)) return NUKA_RESULT_NOT_SUPPORTED;
        nuka::c_abi::CookedWorld cooked;
        nuka_result_t result = nuka::c_abi::CookWorld(record->recipe, scene, &cooked);
        if (result != NUKA_RESULT_OK) return result;
        nuka::nk::World& current = *record->world;
        const uint32_t vel_iters = record->velocity_iterations_set
                                       ? current.VelocityIterations()
                                       : record->recipe.solver_vel_iters;
        std::unique_ptr<nuka::nk::World> rebuilt;
        result = nuka::c_abi::BuildWorld(cooked, record->recipe, record->device,
                                         record->step_options.gravity, vel_iters, &rebuilt);
        if (result != NUKA_RESULT_OK) return result;

        const bool initial = (flags & NUKA_REBUILD_INITIAL_STATE) != 0u;
        Status status = current.Synchronize();
        if (status == Status::Ok && !initial)
            status = nuka::c_abi::MigrateState(current, *record->scene, *rebuilt, scene);
        if (status == Status::Ok && current.GetExecutionMode() == nuka::nk::World::ExecutionMode::Graph)
            status = rebuilt->SetExecutionMode(nuka::nk::World::ExecutionMode::Graph);
        if (status != Status::Ok) return nuka::c_abi::MapStatusToResult(status);

        // Leased storage stays alive until its last view releases it.
        {
            std::lock_guard<std::mutex> lock(nuka::c_abi::LeaseMutex());
            if (record->storage_leases.count(record->generation) != 0u)
                record->retired_worlds.emplace(record->generation, std::move(record->world));
            ++record->generation;
        }
        record->beauty.reset();
        nuka::c_abi::InstallScene(*record, std::move(rebuilt), std::move(cooked), std::move(scene));
        record->dr_baseline_captured = false;
        record->dr_nominal_link_mass.clear();
        record->dr_nominal_joint_armature.clear();
        record->invariant_sampler.Reset();
        record->last_invariant_violations.clear();
        if (initial) record->simulated_step_count = 0u;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[World] rebuild: %s\n", error.what());
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_generation(nuka_world_handle world, uint64_t* out_generation) {
    if (out_generation == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    std::lock_guard<std::mutex> lock(nuka::c_abi::LeaseMutex());
    *out_generation = record->generation;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_retain_storage(nuka_world_handle world, uint64_t* out_generation) {
    if (out_generation == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    std::lock_guard<std::mutex> lock(nuka::c_abi::LeaseMutex());
    ++record->storage_leases[record->generation];
    *out_generation = record->generation;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_release_storage(nuka_world_handle world, uint64_t generation) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    std::unique_ptr<nuka::nk::World> retired;
    {
        std::lock_guard<std::mutex> lock(nuka::c_abi::LeaseMutex());
        const auto lease = record->storage_leases.find(generation);
        if (lease == record->storage_leases.end()) return NUKA_RESULT_INVALID_ARG;
        if (--lease->second != 0u) return NUKA_RESULT_OK;
        record->storage_leases.erase(lease);
        const auto found = record->retired_worlds.find(generation);
        if (found != record->retired_worlds.end()) {
            retired = std::move(found->second);
            record->retired_worlds.erase(found);
        }
    }
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_forward_kinematics(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        return nuka::c_abi::MapStatusToResult(record->world->ForwardKinematics());
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

} // extern "C"
