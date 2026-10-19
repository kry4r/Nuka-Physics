#include "runtime/app/viewer/editor_world.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

static_assert(sizeof(math::Transform) == 7u * sizeof(float), "pose wire layout");
static_assert(sizeof(math::Vec3) == 3u * sizeof(float), "velocity wire layout");

// The command arrow's shaft and head widths, head length and thickness in metres; the measured arrow
// is narrower so the command shows around it where the two agree.
constexpr float kArrowShaft = 0.05f, kArrowHead = 0.15f, kArrowHeadLength = 0.13f, kArrowThick = 0.03f;
constexpr float kMeasuredWidth = 0.7f;
// Metres of arrow per m/s, the longest arrow and the slowest speed drawn. Lengths snap to a step so
// the mesh library keeps few arrows; one shorter than kArrowShortest shrinks whole.
constexpr float kArrowPerSpeed = 0.6f, kArrowMax = 1.2f, kArrowMinSpeed = 0.02f, kArrowStep = 0.02f;
constexpr float kArrowShortest = 1.5f * kArrowHeadLength;
// The measured arrow's rise over the command's, a thickness and a hair.
constexpr float kArrowRise = kArrowThick + 0.01f;
// The terrain checker's square and the lift that keeps the surface over a coplanar floor.
constexpr float kCheckerSquare = 1.0f, kTerrainLift = 0.004f;

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

// A flat block arrow along +X from the origin, centred on the XY plane: a shaft and a triangular head
// `thick` from top to bottom, flat-shaded so its edges read from afar.
render::MeshGeometry ArrowMesh(float length, float shaft, float head, float head_length, float thick) {
    const float neck = length - head_length;
    // The outline, counter-clockwise from above.
    const float outline[7][2] = {{0.0f, -0.5f * shaft}, {neck, -0.5f * shaft}, {neck, -0.5f * head}, {length, 0.0f},
                                 {neck, 0.5f * head},   {neck, 0.5f * shaft},  {0.0f, 0.5f * shaft}};
    render::MeshGeometry g;
    const auto vertex = [&](const float* p, float z, const math::Vec3& n) {
        g.positions.insert(g.positions.end(), {p[0], p[1], z});
        g.normals.insert(g.normals.end(), {n.x, n.y, n.z});
        return g.VertexCount() - 1u;
    };
    // Top and bottom: the shaft's rectangle (corners 0, 1, 5, 6) and the head's triangle (2, 3, 4).
    for (const float side : {1.0f, -1.0f}) {
        uint32_t v[7];
        for (int k = 0; k < 7; ++k) v[k] = vertex(outline[k], 0.5f * thick * side, {0.0f, 0.0f, side});
        for (const auto& t : {std::array<uint32_t, 3>{v[0], v[1], v[5]}, std::array<uint32_t, 3>{v[0], v[5], v[6]},
                              std::array<uint32_t, 3>{v[2], v[3], v[4]}}) {
            if (side > 0.0f) g.indices.insert(g.indices.end(), {t[0], t[1], t[2]});
            else g.indices.insert(g.indices.end(), {t[0], t[2], t[1]});
        }
    }
    // The walls: a quad per outline edge, facing out.
    for (int k = 0; k < 7; ++k) {
        const float* a = outline[k];
        const float* b = outline[(k + 1) % 7];
        const math::Vec3 n = math::Vec3{b[1] - a[1], a[0] - b[0], 0.0f}.Normalized();
        const uint32_t a0 = vertex(a, -0.5f * thick, n), b0 = vertex(b, -0.5f * thick, n);
        const uint32_t b1 = vertex(b, 0.5f * thick, n), a1 = vertex(a, 0.5f * thick, n);
        g.indices.insert(g.indices.end(), {a0, b0, b1, a0, b1, a1});
    }
    return g;
}

// The highest world point of local bounds `box` posed at `pose`.
float TopOf(const std::array<math::Vec3, 2>& box, const math::Transform& pose) {
    float top = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < 8; ++k)
        top = std::max(top, pose.TransformPoint({box[k & 1].x, box[(k >> 1) & 1].y, box[k >> 2].z}).z);
    return top;
}

// The terrain surface: a quad per cell with its own corners so each cell takes one checker shade,
// lit by smooth normals from central differences and brightened with height.
render::MeshGeometry TerrainMesh(const terrain::HeightField& hf) {
    constexpr float kShades[2][3] = {{0.43f, 0.46f, 0.51f}, {0.31f, 0.34f, 0.38f}};
    constexpr int kCorners[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    const int rows = static_cast<int>(hf.nrow), cols = static_cast<int>(hf.ncol);
    auto value = [&](int r, int c) {
        return hf.values[static_cast<size_t>(std::clamp(r, 0, rows - 1)) * hf.ncol + std::clamp(c, 0, cols - 1)];
    };
    auto normal = [&](int r, int c) {
        const float run_x = static_cast<float>(std::min(c + 1, cols - 1) - std::max(c - 1, 0)) * hf.cell_x;
        const float run_y = static_cast<float>(std::min(r + 1, rows - 1) - std::max(r - 1, 0)) * hf.cell_y;
        const float dx = (value(r, c + 1) - value(r, c - 1)) * hf.scale_z / run_x;
        const float dy = (value(r + 1, c) - value(r - 1, c)) * hf.scale_z / run_y;
        return math::Vec3{-dx, -dy, 1.0f}.Normalized();
    };
    render::MeshGeometry g;
    const size_t cells = static_cast<size_t>(rows - 1) * static_cast<size_t>(cols - 1);
    g.positions.reserve(cells * 12u);
    g.normals.reserve(cells * 12u);
    g.colors.reserve(cells * 12u);
    g.indices.reserve(cells * 6u);
    for (int r = 0; r + 1 < rows; ++r) {
        for (int c = 0; c + 1 < cols; ++c) {
            const float mid_x = hf.origin.x + (static_cast<float>(c) + 0.5f) * hf.cell_x;
            const float mid_y = hf.origin.y + (static_cast<float>(r) + 0.5f) * hf.cell_y;
            const int square = static_cast<int>(std::floor(mid_x / kCheckerSquare)) +
                               static_cast<int>(std::floor(mid_y / kCheckerSquare));
            const float* shade = kShades[square & 1];
            const uint32_t first = g.VertexCount();
            for (const auto& corner : kCorners) {
                const int vr = r + corner[0], vc = c + corner[1];
                const float v = value(vr, vc);
                const math::Vec3 n = normal(vr, vc);
                const float lift = 0.9f + 0.25f * v;
                g.positions.insert(g.positions.end(), {hf.origin.x + static_cast<float>(vc) * hf.cell_x,
                                                       hf.origin.y + static_cast<float>(vr) * hf.cell_y,
                                                       hf.base_z + v * hf.scale_z + kTerrainLift});
                g.normals.insert(g.normals.end(), {n.x, n.y, n.z});
                g.colors.insert(g.colors.end(), {shade[0] * lift, shade[1] * lift, shade[2] * lift});
            }
            g.indices.insert(g.indices.end(), {first, first + 1u, first + 2u, first, first + 2u, first + 3u});
        }
    }
    return g;
}

}  // namespace

void CheckResult(nuka_result_t result, const char* what) {
    if (result != NUKA_RESULT_OK)
        throw std::runtime_error(std::string(what) + ": " + ResultText(result));
}

std::string ResultText(nuka_result_t result) {
    const std::string detail = nuka_last_error_message();
    const std::string text = nuka_result_message(result);
    return detail.empty() ? text : text + ": " + detail;
}

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
    // The terrain's checker rides its vertex colours over a matte white.
    scene::RenderMaterial ground;
    ground.base_color[0] = ground.base_color[1] = ground.base_color[2] = 1.0f;
    ground.roughness = 0.9f;
    terrain_material_ = render_world_.MaterialCount();
    render_world_.materials.push_back(ground);
    // Arrows: the measured velocity in the shell's info blue and the command in its ok green. Half the colour
    // glows, so sunlit faces stay saturated and the faces still shade apart.
    constexpr float kArrowColors[2][3] = {{0.20f, 0.55f, 0.95f}, {0.18f, 0.80f, 0.36f}};
    for (int k = 0; k < 2; ++k) {
        scene::RenderMaterial arrow;
        for (int c = 0; c < 3; ++c) {
            arrow.base_color[c] = 0.45f * kArrowColors[k][c];
            arrow.emissive[c] = 0.45f * kArrowColors[k][c];
        }
        arrow.roughness = 0.7f;
        arrow_materials_[k] = render_world_.MaterialCount();
        render_world_.materials.push_back(arrow);
    }
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

void EditorSim::SetTerrain(const terrain::HeightField& terrain) {
    std::vector<render::RenderInstance>& extra = render_world_.extra_instances;
    if (terrain_shown_) extra.erase(extra.begin());
    terrain_shown_ = terrain::IsValid(terrain);
    if (!terrain_shown_) return;
    if (terrain_mesh_ == render::kNoId)
        terrain_mesh_ = render_world_.meshes.AppendGeometry({}, render::MeshSource::PrimitiveFallback);
    render_world_.meshes.ReplaceGeometry(terrain_mesh_, TerrainMesh(terrain));
    render::RenderInstance inst;
    inst.mesh_id = terrain_mesh_;
    inst.render_material_id = terrain_material_;
    extra.insert(extra.begin(), inst);
}

void EditorSim::SetArrows(const std::vector<Arrow>& arrows) {
    std::vector<render::RenderInstance>& extra = render_world_.extra_instances;
    extra.resize(terrain_shown_ ? 1u : 0u);
    for (const Arrow& a : arrows) {
        const float speed = std::hypot(a.velocity.x, a.velocity.y);
        if (speed < kArrowMinSpeed) continue;
        const long steps = std::max(1L, std::lround(std::min(speed * kArrowPerSpeed, kArrowMax) / kArrowStep));
        const float length = static_cast<float>(steps) * kArrowStep;
        const float shrink = std::min(1.0f, length / kArrowShortest);
        const float width = shrink * (a.command ? 1.0f : kMeasuredWidth);
        render::RenderInstance& inst = extra.emplace_back();
        const std::string key =
            (a.command ? "marker:arrow:command:" : "marker:arrow:measured:") + std::to_string(steps);
        inst.mesh_id = render_world_.meshes.InternPrimitive(key, [&] {
            return ArrowMesh(length, kArrowShaft * width, kArrowHead * width, kArrowHeadLength * shrink,
                             kArrowThick * shrink);
        });
        inst.render_material_id = arrow_materials_[a.command ? 1 : 0];
        inst.world_xform.position = a.at + math::Vec3{0.0f, 0.0f, a.command ? 0.0f : kArrowRise};
        inst.world_xform.rotation =
            math::Quat::FromAxisAngle(math::Vec3::UnitZ(), std::atan2(a.velocity.y, a.velocity.x));
    }
}

float EditorSim::LinkTop(const std::vector<math::Transform>& links) const {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::vector<std::array<math::Vec3, 2>> bounds(render_world_.meshes.Count(),
                                                  {math::Vec3{kInf, kInf, kInf}, math::Vec3{-kInf, -kInf, -kInf}});
    std::vector<char> known(bounds.size(), 0);
    float top = -kInf;
    for (const render::RenderInstance& inst : render_world_.instances) {
        if (inst.pose_source.kind != render::PoseSource::Kind::Link || inst.pose_source.row >= links.size() ||
            inst.mesh_id >= bounds.size())
            continue;
        std::array<math::Vec3, 2>& box = bounds[inst.mesh_id];
        if (!known[inst.mesh_id]) {
            known[inst.mesh_id] = 1;
            const std::vector<float>& p = render_world_.meshes.Geometry(inst.mesh_id).positions;
            for (size_t v = 0; v + 2 < p.size(); v += 3) {
                box[0] = {std::min(box[0].x, p[v]), std::min(box[0].y, p[v + 1]), std::min(box[0].z, p[v + 2])};
                box[1] = {std::max(box[1].x, p[v]), std::max(box[1].y, p[v + 1]), std::max(box[1].z, p[v + 2])};
            }
        }
        if (box[0].x <= box[1].x)
            top = std::max(top, TopOf(box, links[inst.pose_source.row] * inst.cached_visual_local));
    }
    return top;
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

void EditorSim::ClearEnvs() {
    render_world_.env_instances.clear();
    env_copies_.clear();
}

void EditorSim::AddEnv(uint32_t env, const std::vector<math::Transform>& links,
                       const std::vector<math::Transform>& bodies, const math::Vec3& offset) {
    math::Transform pose;
    for (size_t i = 0; i < render_world_.instances.size(); ++i) {
        const render::RenderInstance& inst = render_world_.instances[i];
        if (!ResolvePose(inst.pose_source, links, bodies, &pose)) continue;
        render::RenderInstance& copy = render_world_.env_instances.emplace_back(inst);
        copy.world_xform = pose * inst.cached_visual_local;
        copy.world_xform.position += offset;
        env_copies_.push_back({env, static_cast<uint32_t>(i)});
    }
}

}  // namespace nuka::runtime::app::viewer
