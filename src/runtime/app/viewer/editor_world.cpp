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
    return rows > 0u &&
           nuka_world_download_field(world, field, out.data(), bytes, env * bytes) == NUKA_RESULT_OK;
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

bool MoveEntityLive(nuka_world_handle world, const WorldLayout& layout,
                    const render::RenderWorld& render_world, uint32_t env_index,
                    const MoveEntity& cmd) {
    const auto it = std::find_if(render_world.instances.begin(), render_world.instances.end(),
                                 [&](const render::RenderInstance& inst) {
                                     return inst.entity == cmd.entity;
                                 });
    if (it == render_world.instances.end() || !IsMovableSource(it->pose_source, layout))
        return false;
    // Pose fields hold the physics frame; publishing re-applies the visual offset.
    const math::Transform pose = cmd.world_xform * it->cached_visual_local.Inverse();
    const size_t env = env_index % layout.env_count;
    const uint32_t row = it->pose_source.row;
    if (it->pose_source.kind == render::PoseSource::Kind::Body) {
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

EditorSim::EditorSim(nuka_world_handle world, WorldLayout layout, render::RenderWorld render_world)
    : world_(world), layout_(std::move(layout)), render_world_(std::move(render_world)) {}

bool EditorSim::FramePublish(InputIntents* intents, uint32_t steps) {
    input_.Run(commands_, intents);
    for (const MoveEntity& move : intents->move_entities)
        MoveEntityLive(world_, layout_, render_world_, env_index_, move);
    bool healthy = true;
    for (uint32_t s = 0; s < steps; ++s) {
        if (controller_ != nullptr) controller_->OnStep(world_, env_index_);
        healthy = nuka_world_step(world_) == NUKA_RESULT_OK && healthy;
    }
    Publish();
    return healthy;
}

void EditorSim::Publish() {
    const bool have_links = DownloadPoses(world_, NUKA_FIELD_ARTICULATION_LINK_POSE,
                                          layout_.links_per_env, env_index_, link_poses_);
    const bool have_bodies = DownloadPoses(world_, NUKA_FIELD_RIGID_BODY_TRANSFORM,
                                           layout_.bodies_per_env, env_index_, body_poses_);
    const auto resolve = [&](const render::PoseSource& source, math::Transform* out) {
        if (source.kind == render::PoseSource::Kind::Link && have_links &&
            source.row < link_poses_.size()) {
            *out = link_poses_[source.row];
            return true;
        }
        if (source.kind == render::PoseSource::Kind::Body && have_bodies &&
            source.row < body_poses_.size()) {
            *out = body_poses_[source.row];
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
