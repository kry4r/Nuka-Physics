#include "runtime/app/viewer/editor_world.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"

#include <algorithm>
#include <cmath>
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

// The surface drawing a per-environment particle, or null.
const ParticleSurfaceBinding* SurfaceOf(const WorldLayout& layout, uint32_t particle) {
    for (const ParticleSurfaceBinding& s : layout.surfaces)
        if (std::find(s.triangles.begin(), s.triangles.end(), particle) != s.triangles.end()) return &s;
    return nullptr;
}

template <class T>
T DownloadRow(nuka_world_handle world, nuka_state_field_t field, size_t row) {
    T value{};
    CheckResult(nuka_world_download_field(world, field, &value, sizeof(T), row * sizeof(T)), "nuka_world_download_field");
    return value;
}

// The link or body pose a source names; false for static sources and rows the poses lack.
bool ResolvePose(const render::PoseSource& source, const std::vector<math::Transform>& links,
                 const std::vector<math::Transform>& bodies, math::Transform* out) {
    if (source.kind == render::PoseSource::Kind::Link && source.row < links.size()) {
        *out = links[source.row];
        return true;
    }
    if (source.kind == render::PoseSource::Kind::Body && source.row < bodies.size()) {
        *out = bodies[source.row];
        return true;
    }
    return false;
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
    layout.particles_per_env = info.particles_per_env;
    layout.links.resize(layout.links_per_env);
    for (uint32_t l = 0; l < layout.links_per_env; ++l) {
        CheckResult(nuka_world_get_link_binding(world, l, &layout.links[l]),
                    "nuka_world_get_link_binding");
        layout.articulations_per_env =
            std::max(layout.articulations_per_env, layout.links[l].articulation_index + 1u);
    }
    uint32_t surfaces = 0u;
    CheckResult(nuka_world_get_particle_surface_count(world, &surfaces), "nuka_world_get_particle_surface_count");
    for (uint32_t s = 0; s < surfaces; ++s) {
        ParticleSurfaceBinding surface;
        CheckResult(nuka_world_get_particle_surface(world, s, &surface.info, nullptr, 0u), "nuka_world_get_particle_surface");
        if (surface.info.triangle_count == 0u) continue;
        surface.triangles.resize(static_cast<size_t>(surface.info.triangle_count) * 3u);
        CheckResult(nuka_world_get_particle_surface(world, s, &surface.info, surface.triangles.data(),
                                                    surface.triangles.size()),
                    "nuka_world_get_particle_surface");
        layout.surfaces.push_back(std::move(surface));
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

void DownloadEnvParticles(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                          std::vector<math::Vec3>* out) {
    out->resize(layout.particles_per_env);
    const size_t bytes = out->size() * sizeof(math::Vec3);
    if (bytes == 0u) return;
    CheckResult(nuka_world_download_field(world, NUKA_FIELD_PARTICLE_POSITION, out->data(), bytes,
                                          (env_index % layout.env_count) * bytes),
                "download particle positions");
}

bool LocateTelemetry(nuka_world_handle world, const WorldLayout& layout, const scene::SceneIR& scene,
                     const nuka_telemetry_record_t& record, LocatedElement* out) {
    const auto particle = [&](uint32_t index) {
        out->env = index / layout.particles_per_env;
        const uint32_t local = index % layout.particles_per_env;
        out->position = DownloadRow<math::Vec3>(world, NUKA_FIELD_PARTICLE_POSITION, index);
        const ParticleSurfaceBinding* s = SurfaceOf(layout, local);
        out->entity = s != nullptr ? scene.EntityOfMedia(s->info.medium) : scene::kInvalidEntity;
        out->kind = "particle %u";
        out->index = local;
        return true;
    };
    const auto body = [&](uint32_t index) {
        out->env = index / layout.bodies_per_env;
        const uint32_t row = index % layout.bodies_per_env;
        out->position = DownloadRow<math::Transform>(world, NUKA_FIELD_RIGID_BODY_TRANSFORM, index).position;
        out->entity = scene.EntityOfBody(row);
        out->kind = "body %u";
        out->index = row;
        return true;
    };
    // A link side names its articulation; the root link stands for it.
    const auto articulation = [&](uint32_t index) {
        out->env = index / layout.articulations_per_env;
        const uint32_t a = index % layout.articulations_per_env;
        const auto root = std::find_if(layout.links.begin(), layout.links.end(), [&](const nuka_link_binding_t& l) {
            return l.articulation_index == a && l.is_root != 0u;
        });
        const size_t link = static_cast<size_t>(out->env) * layout.links_per_env +
                            static_cast<size_t>(root - layout.links.begin());
        out->position = DownloadRow<math::Transform>(world, NUKA_FIELD_ARTICULATION_LINK_POSE, link).position;
        out->entity = scene.EntityOfBody(root->body_index);
        out->kind = "articulation %u";
        out->index = a;
        return true;
    };
    // A cloth endpoint stands for its heaviest particle in the latest interval's stencil maps.
    const auto endpoint = [&](uint32_t index) {
        const auto range = DownloadRow<nuka_point_endpoint_range_t>(world, NUKA_FIELD_POINT_ENDPOINT_RANGES, index);
        std::vector<nuka_point_endpoint_term_t> terms(range.count);
        CheckResult(nuka_world_download_field(world, NUKA_FIELD_POINT_ENDPOINT_TERMS, terms.data(),
                                              terms.size() * sizeof(nuka_point_endpoint_term_t),
                                              range.first * sizeof(nuka_point_endpoint_term_t)),
                    "download point endpoint terms");
        const nuka_point_endpoint_term_t* heaviest = nullptr;
        float weight = 0.0f;
        for (const nuka_point_endpoint_term_t& t : terms) {
            const float w = t.column[0][0] + t.column[1][1] + t.column[2][2];
            if (t.kind == NUKA_CONTACT_SIDE_PARTICLE && (heaviest == nullptr || w > weight)) {
                heaviest = &t;
                weight = w;
            }
        }
        return heaviest != nullptr && particle(heaviest->index);
    };
    const auto side = [&](uint32_t kind, uint32_t index) {
        switch (kind) {
            case NUKA_CONTACT_SIDE_RIGID: return body(index);
            case NUKA_CONTACT_SIDE_LINK: return articulation(index);
            case NUKA_CONTACT_SIDE_PARTICLE: return particle(index);
            case NUKA_CONTACT_SIDE_POINT_ENDPOINT: return endpoint(index);
            default: return false;
        }
    };
    if (record.worst_kind == NUKA_TELEMETRY_ELEMENT_PARTICLE) return particle(record.worst_index);
    if (record.worst_kind != NUKA_TELEMETRY_ELEMENT_ROW) return false;
    // A particle or endpoint side pins the contact point; a body or link side only names its owner.
    const uint32_t b = record.worst_side_kind[1];
    const int a = b == NUKA_CONTACT_SIDE_PARTICLE || b == NUKA_CONTACT_SIDE_POINT_ENDPOINT ? 1 : 0;
    const bool found = side(record.worst_side_kind[a], record.worst_side_index[a]) ||
                       side(record.worst_side_kind[1 - a], record.worst_side_index[1 - a]);
    if (found) out->row = record.worst_index;
    return found;
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

EditorSim::EditorSim(WorldLayout layout, render::RenderWorld render_world, const scene::SceneIR& scene)
    : layout_(std::move(layout)), render_world_(std::move(render_world)) {
    // The heatmap's colours multiply into a white matte.
    scene::RenderMaterial heat;
    heat.base_color[0] = heat.base_color[1] = heat.base_color[2] = 1.0f;
    heat.roughness = 0.85f;
    heat_material_ = render_world_.MaterialCount();
    render_world_.materials.push_back(heat);
    // The ghost: the shell's info blue, translucent and faintly lit from within.
    scene::RenderMaterial ghost;
    ghost.base_color[0] = 0.34f;
    ghost.base_color[1] = 0.71f;
    ghost.base_color[2] = 0.94f;
    ghost.emissive[0] = 0.04f;
    ghost.emissive[1] = 0.10f;
    ghost.emissive[2] = 0.16f;
    ghost.roughness = 0.6f;
    ghost.opacity = 0.35f;
    ghost_material_ = render_world_.MaterialCount();
    render_world_.materials.push_back(ghost);
    for (const ParticleSurfaceBinding& s : layout_.surfaces) {
        render::RenderInstance inst;
        inst.entity = scene.EntityOfMedia(s.info.medium);
        inst.mesh_id = render_world_.meshes.AppendGeometry({}, render::MeshSource::PrimitiveFallback);
        inst.render_material_id = s.info.render_material_id < render_world_.default_material_id
                                      ? s.info.render_material_id
                                      : render_world_.default_material_id;
        Surface surface{render_world_.InstanceCount(), inst.mesh_id, inst.render_material_id, {}};
        surface.topology.triangles = s.triangles;
        surface.topology.normal_offset = s.info.normal_offset;
        surface.topology.smooth_iters = s.info.smooth_iters;
        surface.topology.smooth_lambda = s.info.smooth_lambda;
        surfaces_.push_back(std::move(surface));
        render_world_.instances.push_back(inst);
    }
}

std::vector<PoseWrite> EditorSim::TakePoseWrites() {
    std::vector<PoseWrite> out;
    out.swap(pose_writes_);
    return out;
}

void EditorSim::ApplyParticles(const std::vector<math::Vec3>& positions, const std::vector<float>& heat, float alert) {
    const bool heated = heat.size() == positions.size();
    for (const Surface& s : surfaces_) {
        render::MeshGeometry geo;
        runtime::soft::BuildSurfaceMesh(positions, s.topology, geo);
        if (heated) {
            geo.colors.resize(positions.size() * 3u);
            for (size_t v = 0; v < positions.size(); ++v) HeatColor(heat[v], alert, &geo.colors[v * 3u]);
        }
        render_world_.meshes.ReplaceGeometry(s.mesh, std::move(geo));
        render_world_.instances[s.instance].render_material_id = heated ? heat_material_ : s.material;
    }
}

void EditorSim::ApplyPoses(const std::vector<math::Transform>& links,
                           const std::vector<math::Transform>& bodies) {
    math::Transform pose;
    for (render::RenderInstance& inst : render_world_.instances)
        if (ResolvePose(inst.pose_source, links, bodies, &pose)) inst.world_xform = pose * inst.cached_visual_local;
    for (render::RenderCamera& cam : render_world_.cameras)
        if (ResolvePose(cam.pose_source, links, bodies, &pose)) cam.world_xform = pose * cam.cached_visual_local;
    for (render::RenderLight& light : render_world_.lights)
        if (ResolvePose(light.pose_source, links, bodies, &pose)) light.world_xform = pose * light.cached_visual_local;
}

void EditorSim::ApplyGhost(const std::vector<math::Transform>& links,
                           const std::vector<math::Transform>& bodies) {
    std::vector<render::RenderInstance>& ghost = render_world_.ghost_instances;
    ghost.clear();
    math::Transform pose;
    for (const render::RenderInstance& inst : render_world_.instances) {
        if (!ResolvePose(inst.pose_source, links, bodies, &pose)) continue;
        render::RenderInstance& g = ghost.emplace_back(inst);
        g.world_xform = pose * inst.cached_visual_local;
        g.render_material_id = ghost_material_;
    }
}

}  // namespace nuka::runtime::app::viewer
