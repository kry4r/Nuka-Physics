#include "runtime/app/viewer/editor_world.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

static_assert(sizeof(math::Transform) == 7u * sizeof(float), "pose wire layout");
static_assert(sizeof(math::Vec3) == 3u * sizeof(float), "velocity wire layout");

uint32_t RowsPerEnv(nuka_world_handle world, nuka_state_field_t field, uint32_t env_count) {
    nuka_buffer_view_t view{};
    if (nuka_world_get_buffer_view(world, field, &view) != NUKA_RESULT_OK) return 0u;
    return static_cast<uint32_t>(view.element_count / env_count);
}

// One environment's rows of a pose field; false when the field is empty.
bool DownloadPoses(nuka_world_handle world, nuka_state_field_t field, uint32_t rows,
                   uint32_t env, std::vector<math::Transform>& out) {
    out.resize(rows);
    const size_t bytes = static_cast<size_t>(rows) * sizeof(math::Transform);
    const bool ok = rows > 0u &&
                    nuka_world_download_field(world, field, out.data(), bytes, env * bytes) == NUKA_RESULT_OK;
    if (!ok) out.clear();
    return ok;
}

}  // namespace

void CheckResult(nuka_result_t result, const char* what) {
    if (result != NUKA_RESULT_OK)
        throw std::runtime_error(std::string(what) + ": " + nuka_result_message(result));
}

WorldLayout QueryLayout(nuka_world_handle world) {
    nuka_world_resource_info_t info{};
    CheckResult(nuka_world_get_resource_info(world, &info), "nuka_world_get_resource_info");
    WorldLayout layout;
    layout.env_count = std::max(info.env_count, 1u);
    layout.bodies_per_env = info.bodies_per_env;
    layout.dofs_per_env = info.dofs_per_articulation;
    layout.contacts_per_env = info.contacts_per_env;
    layout.links_per_env = RowsPerEnv(world, NUKA_FIELD_ARTICULATION_LINK_POSE, layout.env_count);
    layout.contact_slots_per_env = RowsPerEnv(world, NUKA_FIELD_UCONTACT_COUNT, layout.env_count);
    layout.links.resize(layout.links_per_env);
    for (uint32_t l = 0; l < layout.links_per_env; ++l) {
        CheckResult(nuka_world_get_link_binding(world, l, &layout.links[l]),
                    "nuka_world_get_link_binding");
        layout.articulations_per_env =
            std::max(layout.articulations_per_env, layout.links[l].articulation_index + 1u);
    }
    return layout;
}

terrain::HeightField QueryTerrain(nuka_world_handle world) {
    terrain::HeightField field;
    nuka_terrain_info_t info{};
    if (nuka_world_get_terrain(world, &info, nullptr, 0u) != NUKA_RESULT_OK) return field;
    field.values.resize(static_cast<size_t>(info.nrow) * info.ncol);
    CheckResult(nuka_world_get_terrain(world, &info, field.values.data(), field.values.size()),
                "nuka_world_get_terrain");
    field.nrow = info.nrow;
    field.ncol = info.ncol;
    field.cell_x = info.cell_x;
    field.cell_y = info.cell_y;
    field.origin = math::Vec3{info.origin[0], info.origin[1], info.origin[2]};
    field.base_z = info.base_z;
    field.scale_z = info.scale_z;
    return field;
}

scene::SceneMap BindScene(const scene::SceneIR& scene, const WorldLayout& layout) {
    const auto bodies = static_cast<uint32_t>(scene.RigidBodyCount());
    std::vector<uint32_t> link_of_body(bodies, scene::SceneMap::kNoRow);
    for (uint32_t l = 0; l < layout.links_per_env; ++l)
        if (layout.links[l].body_index < bodies) link_of_body[layout.links[l].body_index] = l;
    scene::SceneMap map;
    for (uint32_t b = 0; b < bodies; ++b) {
        const scene::EntityId entity = scene.EntityOfBody(b);
        if (entity == scene::kInvalidEntity) continue;
        scene::CookedRef ref;
        ref.body_row = b;
        ref.link_index = link_of_body[b];
        map.Bind(entity, ref);
    }
    return map;
}

bool IsMovableSource(const render::PoseSource& source, const WorldLayout& layout) {
    if (source.kind == render::PoseSource::Kind::Body) return true;
    return source.kind == render::PoseSource::Kind::Link && source.row < layout.links.size() &&
           layout.links[source.row].is_floating != 0u;
}

bool ResolvePoseWrite(const render::RenderWorld& render_world, const WorldLayout& layout,
                      scene::EntityId entity, const math::Transform& world_xform, PoseWrite* out) {
    const auto it = std::find_if(render_world.instances.begin(), render_world.instances.end(),
                                 [&](const render::RenderInstance& inst) {
                                     return inst.entity == entity;
                                 });
    if (it == render_world.instances.end() || !IsMovableSource(it->pose_source, layout))
        return false;
    // Pose fields hold the physics frame; publishing re-applies the visual offset.
    out->source = it->pose_source;
    out->pose = world_xform * it->cached_visual_local.Inverse();
    return true;
}

bool ApplyPoseWrite(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                    const PoseWrite& write) {
    const math::Transform& pose = write.pose;
    const size_t env = env_index % layout.env_count;
    const uint32_t row = write.source.row;
    if (write.source.kind == render::PoseSource::Kind::Body) {
        if (row >= layout.bodies_per_env) return false;
        const size_t at = env * layout.bodies_per_env + row;
        const math::Vec3 zero = math::Vec3::Zero();
        return nuka_world_upload_field(world, NUKA_FIELD_RIGID_BODY_TRANSFORM, &pose, sizeof(pose),
                                       at * sizeof(pose)) == NUKA_RESULT_OK &&
               nuka_world_upload_field(world, NUKA_FIELD_BODY_LINEAR_VELOCITY, &zero, sizeof(zero),
                                       at * sizeof(zero)) == NUKA_RESULT_OK &&
               nuka_world_upload_field(world, NUKA_FIELD_BODY_ANGULAR_VELOCITY, &zero, sizeof(zero),
                                       at * sizeof(zero)) == NUKA_RESULT_OK;
    }
    // A floating root moves through its articulation's base pose; FK re-poses the links.
    const size_t base = env * layout.articulations_per_env + layout.links[row].articulation_index;
    const size_t link = env * layout.links_per_env + row;
    const float zero_spatial[6] = {};
    return nuka_world_upload_field(world, NUKA_FIELD_BASE_POSE, &pose, sizeof(pose),
                                   base * sizeof(pose)) == NUKA_RESULT_OK &&
           nuka_world_upload_field(world, NUKA_FIELD_LINK_VELOCITY, zero_spatial,
                                   sizeof(zero_spatial), link * sizeof(zero_spatial)) == NUKA_RESULT_OK &&
           nuka_world_forward_kinematics(world) == NUKA_RESULT_OK;
}

void ApplyHoldGains(nuka_world_handle world, const WorldLayout& layout, float kp, float kd) {
    const size_t links = layout.links_per_env;
    if (links == 0u) return;
    std::vector<float> stiffness(links * layout.env_count), damping(stiffness.size());
    for (size_t i = 0; i < stiffness.size(); ++i) {
        const bool root = layout.links[i % links].is_root != 0u;
        stiffness[i] = root ? 0.0f : kp;
        damping[i] = root ? 0.0f : kd;
    }
    CheckResult(nuka_world_upload_field(world, NUKA_FIELD_DRIVE_STIFFNESS, stiffness.data(),
                                        stiffness.size() * sizeof(float), 0u),
                "upload drive stiffness");
    CheckResult(nuka_world_upload_field(world, NUKA_FIELD_DRIVE_DAMPING, damping.data(),
                                        damping.size() * sizeof(float), 0u),
                "upload drive damping");
}

void DownloadEnvPoses(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                      std::vector<math::Transform>* links, std::vector<math::Transform>* bodies) {
    const uint32_t env = env_index % layout.env_count;
    DownloadPoses(world, NUKA_FIELD_ARTICULATION_LINK_POSE, layout.links_per_env, env, *links);
    DownloadPoses(world, NUKA_FIELD_RIGID_BODY_TRANSFORM, layout.bodies_per_env, env, *bodies);
}

std::vector<ScatterRow> BuildScatterRows(const render::RenderWorld& render_world) {
    std::vector<ScatterRow> rows(render_world.instances.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const render::RenderInstance& inst = render_world.instances[i];
        ScatterRow& r = rows[i];
        const bool link = inst.pose_source.kind == render::PoseSource::Kind::Link;
        const bool body = inst.pose_source.kind == render::PoseSource::Kind::Body;
        r.kind = link ? 1u : (body ? 2u : 0u);
        r.row = r.kind != 0u ? inst.pose_source.row : 0u;
        const math::Transform& local = r.kind != 0u ? inst.cached_visual_local : inst.world_xform;
        r.local[0] = local.position.x;
        r.local[1] = local.position.y;
        r.local[2] = local.position.z;
        r.local[3] = local.rotation.w;
        r.local[4] = local.rotation.x;
        r.local[5] = local.rotation.y;
        r.local[6] = local.rotation.z;
    }
    return rows;
}

EditorSim::EditorSim(WorldLayout layout, render::RenderWorld render_world)
    : layout_(std::move(layout)), render_world_(std::move(render_world)) {}

std::vector<PoseWrite> EditorSim::TakePoseWrites() {
    std::vector<PoseWrite> out;
    out.swap(pose_writes_);
    return out;
}

void EditorSim::ApplyPoses(const std::vector<math::Transform>& links,
                           const std::vector<math::Transform>& bodies) {
    const auto resolve = [&](const render::PoseSource& source, math::Transform* out) {
        if (source.kind == render::PoseSource::Kind::Link && source.row < links.size()) {
            *out = links[source.row];
            return true;
        }
        if (source.kind == render::PoseSource::Kind::Body && source.row < bodies.size()) {
            *out = bodies[source.row];
            return true;
        }
        return false;
    };
    math::Transform pose;
    for (render::RenderInstance& inst : render_world_.instances)
        if (resolve(inst.pose_source, &pose)) inst.world_xform = pose * inst.cached_visual_local;
    for (render::RenderCamera& cam : render_world_.cameras)
        if (resolve(cam.pose_source, &pose)) cam.world_xform = pose * cam.cached_visual_local;
    for (render::RenderLight& light : render_world_.lights)
        if (resolve(light.pose_source, &pose)) light.world_xform = pose * light.cached_visual_local;
}

}  // namespace nuka::runtime::app::viewer
