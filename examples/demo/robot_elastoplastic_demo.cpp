// A finite-force Panda gripper loads and releases a freely simulated Hencky J2 specimen.
// Captured body, joint and material states drive both views of the replay renderer.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision/mesh_surface.hpp"
#include "collision/primitive_surface.hpp"
#include "nk/material/hencky_j2.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "phi/articulation_contract.hpp"
#include "render/studio_beauty.hpp"
#include "render/scene_asset.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "world_capture_checkpoint.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace scene = nuka::scene;
namespace cook = scene::cook;
namespace render = nuka::render;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kHz = 120.0f;
constexpr double kScheduleScale = 0.5;
constexpr Vec3 kCenter{0.5033f, -0.0435f, 0.5140f};
constexpr Vec3 kSize{0.024f, 0.036f, 0.028f};
constexpr float kDensity = 1000.0f, kYoungs = 50000.0f, kPoisson = 0.25f;
constexpr float kYield = 12000.0f, kHardening = 7500.0f, kGravity = -9.81f;
constexpr float kOpen = 0.04f, kMild = 0.0165f, kStrong = 0.004f;
constexpr std::array<const char*, 9> kJointNames{
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"};
constexpr std::array<float, 9> kHome{
    -0.04080849f, 0.34862798f, -0.04117866f, -2.36771464f,
    0.03044342f, 2.70693517f, 0.67838210f, kOpen, kOpen};
constexpr std::array<float, 9> kKp{600, 600, 500, 500, 360, 240, 160, 2000, 2000};
constexpr std::array<float, 9> kKd{46, 46, 38, 38, 28, 20, 14, 12, 12};
constexpr std::array<float, 9> kLimit{87, 87, 87, 87, 12, 12, 12, 20, 20};

struct Args {
    std::filesystem::path executable;
    std::filesystem::path out = "out/robot_elastoplastic";
    std::filesystem::path source = std::filesystem::path(NUKA_SOURCE_DIR) /
        ".nuka-assets/generated/panda/panda_pick_place.nks";
    std::filesystem::path replay;
    std::filesystem::path environment = std::filesystem::path(NUKA_SOURCE_DIR) /
        "examples/assets/nuka_lab/gripper.nks";
    float dx = 0.0012f;
    uint32_t substeps = 128u, velocity_iterations = 32u, frames = 336u;
    uint32_t contact_capacity = 32768u;
    uint32_t width = 1280u, height = 800u, samples = 48u, render_stride = 1u;
    uint32_t checkpoint_interval = 12u, stop_after = ~0u;
    nk::World::ExecutionMode execution = nk::World::ExecutionMode::Graph;
    std::string overview = "three-quarter";
    std::string renderer = "rt";
    bool no_render = false, resume = false;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    args.executable = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--no-render") { args.no_render = true; continue; }
        if (key == "--resume") { args.resume = true; continue; }
        Require(i + 1 < argc, "Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--out-dir") args.out = value;
        else if (key == "--scene") args.source = value;
        else if (key == "--replay") args.replay = value;
        else if (key == "--environment") args.environment = value;
        else if (key == "--dx") args.dx = std::stof(value);
        else if (key == "--substeps") args.substeps = std::stoul(value);
        else if (key == "--velocity-iterations") args.velocity_iterations = std::stoul(value);
        else if (key == "--contact-capacity") args.contact_capacity = std::stoul(value);
        else if (key == "--frames") args.frames = std::stoul(value);
        else if (key == "--width") args.width = std::stoul(value);
        else if (key == "--height") args.height = std::stoul(value);
        else if (key == "--samples") args.samples = std::stoul(value);
        else if (key == "--render-stride") args.render_stride = std::stoul(value);
        else if (key == "--renderer") {
            Require(value == "rt" || value == "raster", "Renderer must be rt or raster");
            args.renderer = value;
        }
        else if (key == "--overview") {
            Require(value == "three-quarter" || value == "front" || value == "top",
                    "Overview must be three-quarter, front or top");
            args.overview = value;
        }
        else if (key == "--checkpoint-interval") args.checkpoint_interval = std::stoul(value);
        else if (key == "--stop-after") args.stop_after = std::stoul(value);
        else if (key == "--execution") {
            Require(value == "eager" || value == "graph", "Execution must be eager or graph");
            args.execution = value == "graph" ? nk::World::ExecutionMode::Graph : nk::World::ExecutionMode::Eager;
        }
        else throw std::runtime_error("Unknown argument " + key);
    }
    Require(std::isfinite(args.dx) && args.dx > 0.0f && args.dx <= 0.004f,
            "Grid spacing must be in (0, 0.004] m");
    Require(args.substeps && args.velocity_iterations && args.velocity_iterations <= 65535u &&
            args.frames && args.width && args.height && args.samples && args.render_stride,
            "Counts must be positive; velocity iterations must fit uint16");
    Require(!args.resume || args.checkpoint_interval != 0u, "Resume requires checkpointing");
    return args;
}

double Smooth(double time, double begin, double end, double a, double b) {
    const double u = std::clamp((time - begin) / (end - begin), 0.0, 1.0);
    return a + (b - a) * u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

float FingerTarget(double time) {
    time /= kScheduleScale;
    if (time < 0.5) return kOpen;
    if (time < 1.2) return static_cast<float>(Smooth(time, 0.5, 1.2, kOpen, kMild));
    if (time < 1.5) return kMild;
    if (time < 2.0) return static_cast<float>(Smooth(time, 1.5, 2.0, kMild, kOpen));
    if (time < 2.6) return kOpen;
    if (time < 3.5) return static_cast<float>(Smooth(time, 2.6, 3.5, kOpen, kStrong));
    if (time < 4.0) return kStrong;
    if (time < 4.8) return static_cast<float>(Smooth(time, 4.0, 4.8, kStrong, kOpen));
    return kOpen;
}

const char* Stage(double time) {
    time /= kScheduleScale;
    if (time < 0.5) return "settle";
    if (time < 1.2) return "mild_load";
    if (time < 1.5) return "mild_hold";
    if (time < 2.0) return "mild_unload";
    if (time < 2.6) return "mild_recovery";
    if (time < 3.5) return "strong_load";
    if (time < 4.0) return "strong_hold";
    if (time < 4.8) return "release";
    return "free_recovery";
}

void AddSupport(scene::SceneIR& ir, const std::string& name, Vec3 center, Vec3 half,
                scene::MaterialRecord material) {
    material.name = name + "_material";
    const auto material_id = ir.AddMaterial(material);
    scene::RigidBodyRecord body;
    body.name = name;
    body.is_static = true;
    body.mass = 0.0f;
    body.local_transform.position = center;
    const auto body_id = ir.AddRigidBody(body);
    scene::CollisionShapeRecord shape;
    shape.name = name;
    shape.body_id = body_id;
    shape.type = scene::ShapeType::Box;
    shape.half_extents = half;
    shape.friction_mu = 0.4f;
    ir.AddCollisionShape(shape);
    shape.name += "_visual";
    shape.contype = shape.conaffinity = 0u;
    shape.material_id = material_id;
    ir.AddCollisionShape(shape);
}

struct SceneModel {
    scene::SceneIR ir;
    cook::CookToModelResult cooked;
    std::array<uint32_t, kJointNames.size()> joints{};
    std::vector<float> link_mass;
    std::vector<Vec3> link_com;
    float particle_volume = 0.0f;
};

SceneModel BuildModel(const Args& args) {
    SceneModel result;
    auto& ir = result.ir;
    ir = scene::nks::Load(args.source.string());
    ir.InitialStateMut() = {};
    for (size_t i = ir.RigidBodyCount(); i-- > 0u;) {
        const auto body = ir.GetBody(static_cast<scene::BodyId>(i));
        if (body.parent_id == scene::kInvalidBody && body.name != "link0")
            Require(ir.RemoveBodySubtree(body.id), "Cannot remove scene body " + body.name);
    }
    std::array<scene::BodyId, kJointNames.size()> joint_bodies{};
    for (size_t j = 0u; j < kJointNames.size(); ++j) {
        bool found = false;
        for (uint32_t i = 0u; i < ir.JointCount(); ++i) {
            if (ir.GetJoint(i).name != kJointNames[j]) continue;
            auto& joint = ir.GetJointMut(i);
            joint.initial_position = kHome[j];
            joint_bodies[j] = joint.child_body;
            found = true;
            break;
        }
        Require(found, "Missing Panda joint " + std::string(kJointNames[j]));
    }
    const auto appearance = scene::nks::Load(args.environment.string());
    const auto material = [&](const std::string& name) {
        const auto& materials = appearance.Materials();
        const auto found = std::find_if(materials.begin(), materials.end(),
            [&](const auto& value) { return value.name == name; });
        Require(found != materials.end(), "Missing environment material " + name);
        return *found;
    };
    AddSupport(ir, "bench", {0.35f, 0.0f, 0.40f}, {0.65f, 0.45f, 0.04f}, material("bench_surface"));
    AddSupport(ir, "specimen_support", {kCenter.x, kCenter.y, 0.47f}, {0.04f, 0.05f, 0.03f},
               material("specimen_support"));
    for (uint32_t i = 0u; i < ir.ShapeCount(); ++i) {
        const auto& shape = ir.GetShape(i);
        if (shape.body_id == joint_bodies[7] || shape.body_id == joint_bodies[8])
            ir.GetShapeMut(i).friction_mu = 0.8f;
    }
    const auto scene_directory = args.resume ? args.out / "checkpoint_scene_check" : args.out;
    std::filesystem::create_directories(scene_directory);
    const auto scene_path = scene_directory / "scene.nks";
    scene::nks::Save(ir, scene_path.string());
    if (args.resume) {
        for (const char* name : {"scene.nks", "scene.nka"}) {
            Require(nuka::demo::CaptureFileFingerprint(scene_directory / name) ==
                    nuka::demo::CaptureFileFingerprint(args.out / name), "Resume scene differs from capture");
        }
        std::filesystem::remove(scene_path);
        std::filesystem::remove(scene_directory / "scene.nka");
        std::filesystem::remove(scene_directory);
    }
    result.cooked = cook::CookToModel(ir, 1);
    auto& model = result.cooked.model;
    auto& drives = model.hold_drives;
    const uint32_t links = model.capacities.links_per_env;
    drives.targets = model.articulation.initial_q;
    drives.stiffness.assign(links, 0.0f);
    drives.damping.assign(links, 0.0f);
    drives.force_limits.assign(links, 0.0f);
    for (size_t j = 0u; j < kJointNames.size(); ++j) {
        const auto* ref = result.cooked.scene_map.RefOf(ir.EntityOfBody(joint_bodies[j]));
        Require(ref != nullptr && ref->link_index < links, "Joint has no articulated link");
        const uint32_t link = result.joints[j] = ref->link_index;
        drives.targets[link] = kHome[j];
        drives.stiffness[link] = kKp[j];
        drives.damping[link] = kKd[j];
        drives.force_limits[link] = kLimit[j];
    }
    result.link_mass.assign(links, 0.0f);
    result.link_com.resize(links);
    for (const auto& body : ir.Bodies()) {
        const auto* ref = result.cooked.scene_map.RefOf(ir.EntityOfBody(body.id));
        if (ref == nullptr || ref->link_index >= links) continue;
        result.link_mass[ref->link_index] = body.mass;
        result.link_com[ref->link_index] = body.inertial_transform.position;
    }
    cook::MpmCookInput medium;
    const float spacing = args.dx * 0.5f;
    const std::array<float, 3> size{kSize.x, kSize.y, kSize.z};
    std::array<uint32_t, 3> cells{};
    std::array<float, 3> pitch{};
    for (size_t d = 0u; d < cells.size(); ++d) {
        cells[d] = std::max(1l, std::lround(size[d] / spacing));
        pitch[d] = size[d] / cells[d];
    }
    for (uint32_t z = 0u; z < cells[2]; ++z)
        for (uint32_t y = 0u; y < cells[1]; ++y)
            for (uint32_t x = 0u; x < cells[0]; ++x)
                medium.positions.push_back(kCenter + Vec3{
                    (x + 0.5f) * pitch[0] - 0.5f * kSize.x,
                    (y + 0.5f) * pitch[1] - 0.5f * kSize.y,
                    (z + 0.5f) * pitch[2] - 0.5f * kSize.z});
    const size_t count = medium.positions.size();
    const float volume = result.particle_volume = pitch[0] * pitch[1] * pitch[2];
    medium.velocities.assign(count, Vec3::Zero());
    medium.inv_mass.assign(count, 1.0f / (kDensity * volume));
    medium.vol0.assign(count, volume);
    medium.material.youngs = kYoungs;
    medium.material.poisson = kPoisson;
    medium.material.density = kDensity;
    medium.material.model_kind = nk::MpmMaterial::kHenckyJ2;
    medium.material.yield_stress = kYield;
    medium.material.hardening_modulus = kHardening;
    medium.dx = args.dx;
    medium.contact_capacity = args.contact_capacity;
    medium.grid_origin = {kCenter.x - 0.08f, kCenter.y - 0.08f, 0.44f};
    medium.grid_dims[0] = medium.grid_dims[1] = static_cast<uint32_t>(std::ceil(0.16f / args.dx)) + 1u;
    medium.grid_dims[2] = static_cast<uint32_t>(std::ceil(0.16f / args.dx)) + 1u;
    medium.floor_d = 0.44f;
    medium.floor_friction = 0.4f;
    cook::CookMpmParticles(model, 1u, medium);
    model.particles.mpm_body_friction = 0.8f;
    model.particles.mpm_body_band = 0.0002f;
    return result;
}

template <class T> void Write(std::ostream& output, const std::vector<T>& values) {
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    Require(static_cast<bool>(output), "Capture write failed");
}

template <class T> void Read(std::istream& input, std::vector<T>& values) {
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    Require(static_cast<bool>(input), "Capture read failed");
}

template <class T> std::vector<T> Download(nk::World& world, nk::FieldId field) {
    const size_t bytes = world.GetModel().capacities.ElementCount(field) * nk::LayoutOf(field).elem_size;
    std::vector<T> values(bytes / sizeof(T));
    Require(bytes == values.size() * sizeof(T), "Field element size mismatch");
    Require(world.GetData().DownloadField(field, values.data(), bytes), "Field download failed");
    return values;
}

template <class T> std::vector<T> DownloadRange(nk::World& world, nk::FieldId field,
                                              uint64_t first, uint64_t count) {
    std::vector<T> values(count);
    if (count != 0u)
        Require(world.GetData().DownloadField(field, values.data(), count * sizeof(T),
                    first * sizeof(T)), "Field range download failed");
    return values;
}

struct InterfaceMetrics {
    double normal_residual = 0.0;
    double cone_violation = 0.0;
    double penetration = 0.0;
};

InterfaceMetrics MeasureInterface(nk::World& world, const std::vector<Vec3>& positions,
                                  const std::vector<Transform>& poses, float dx) {
    const auto& model = world.GetModel();
    const auto& cap = model.capacities;
    InterfaceMetrics result;
    const nuka::collision::MeshSurfaceView mesh{model.hull_verts.data(),
        model.mesh_triangles.data(), model.mesh_bvh_nodes.data(),
        {cap.max_hull_verts, cap.max_mesh_triangles, cap.max_mesh_bvh_nodes}};
    for (uint32_t body = 0u; body < model.shape_table_rows.size(); ++body) {
        const auto& shape = model.shape_table_rows[body];
        if ((shape.contype | shape.conaffinity) == 0u) continue;
        const auto inverse = poses[body].Inverse();
        for (const auto x : positions) {
            const Vec3 local = inverse.TransformPoint(x);
            double distance = 0.0;
            if (shape.kind <= nuka::collision::kShapePlane) {
                const auto surface = nuka::collision::QueryPrimitiveSurface(shape.kind,
                    {shape.params[0], shape.params[1], shape.params[2]}, local);
                Require(surface.valid, "Invalid primitive readout geometry");
                distance = surface.distance;
            } else {
                Require(body < model.mesh_surface_info.size() &&
                    model.mesh_surface_info[body].triangle_count > 0u,
                    "Penetration readout needs an authored triangle surface");
                const auto surface = nuka::collision::QueryMeshSurface(mesh,
                    model.mesh_surface_info[body], local, dx);
                Require(surface.valid, "Invalid mesh readout geometry");
                distance = surface.distance;
            }
            result.penetration = std::max(result.penetration, -distance);
        }
    }
    const uint32_t contacts = Download<uint32_t>(world, nk::FieldId::GridContactRetained)[0];
    if (contacts == 0u) return result;
    const phi::MpmParams* params = nullptr;
    for (const auto& call : world.GetPipeline().Calls())
        if (call.op == phi::NkOp::MpmExchange) {
            params = static_cast<const phi::MpmParams*>(call.params);
            break;
        }
    Require(params != nullptr, "Missing grid contact provider");
    const uint64_t base = uint64_t(params->full_row_slot_count) * nk::kPairDrivenRowsPerSlot +
        uint64_t(params->contact_slot_base - params->full_row_slot_count) *
            nk::kPairDrivenParticleRowsPerSlot;
    const uint64_t count = uint64_t(contacts) * nk::kPairDrivenParticleRowsPerSlot;
    const uint32_t dofs = cap.dofs_per_env;
    const auto rows = DownloadRange<nk::NkRow>(world, nk::FieldId::Urows, base, count);
    const auto impulse = DownloadRange<float>(world, nk::FieldId::Lambda, base, count);
    const auto mass = DownloadRange<float>(world, nk::FieldId::RowMeff, base, count);
    const auto ja = DownloadRange<float>(world, nk::FieldId::ChainJacobian, base * dofs, count * dofs);
    const auto jb = DownloadRange<float>(world, nk::FieldId::ChainJacobianB, base * dofs, count * dofs);
    const auto qdot = Download<float>(world, nk::FieldId::QdotFlat);
    const auto grid = Download<Vec3>(world, nk::FieldId::GridVelocity);
    const auto particle_velocity = Download<Vec3>(world, nk::FieldId::ParticleVel);
    const auto ranges = Download<nk::PointEndpointRange>(world, nk::FieldId::PointEndpointRanges);
    const auto terms = Download<nk::PointEndpointTerm>(world, nk::FieldId::PointEndpointTerms);
    const auto linear = Download<Vec3>(world, nk::FieldId::BodyLinearVelocity);
    const auto angular = Download<Vec3>(world, nk::FieldId::BodyAngularVelocity);
    const auto side_velocity = [&](const nk::NkRowSide& side, const std::vector<float>& jacobian,
                                   uint64_t row) {
        if (side.kind == nk::kNkSideStatic) return 0.0;
        if (side.kind == nk::kNkSideGrid) return double(side.jlin.Dot(grid[side.index]));
        if (side.kind == nk::kNkSideParticle) return double(side.jlin.Dot(particle_velocity[side.index]));
        if (side.kind == nk::kNkSidePointEndpoint) {
            double value = 0.0;
            const auto range = ranges.at(side.index);
            for (uint32_t i = 0u; i < range.count; ++i) {
                const auto& term = terms.at(range.first + i);
                const auto& velocity = term.kind == nk::kNkSideGrid ? grid : particle_velocity;
                value += term.TransposeMultiply(side.jlin).Dot(velocity.at(term.index));
            }
            return value;
        }
        if (side.kind == nk::kNkSideRigid)
            return double(side.jlin.Dot(linear[side.index])) + side.jang.Dot(angular[side.index]);
        Require(side.kind == nk::kNkSideArtic, "Unexpected grid contact endpoint");
        double value = 0.0;
        for (uint32_t d = 0u; d < dofs; ++d)
            value += double(jacobian[row * dofs + d]) * qdot[uint64_t(side.index) * dofs + d];
        return value;
    };
    for (uint64_t row = 0u; row < count; row += nk::kPairDrivenParticleRowsPerSlot) {
        const auto& normal = rows[row];
        Require((normal.flags & nk::nk_row_flags::kVelocityOnly) != 0u && mass[row] > 0.0f,
                "Invalid grid contact row");
        const double velocity = side_velocity(normal.a, ja, row) + side_velocity(normal.b, jb, row) -
                                double(normal.rhs) * params->dt;
        const double residual = std::fabs(std::min(velocity, double(impulse[row]) / mass[row]));
        result.normal_residual = std::max(result.normal_residual, residual);
        const double tangent = std::hypot(double(impulse[row + 1u]), double(impulse[row + 2u]));
        result.cone_violation = std::max(result.cone_violation, tangent - normal.mu * impulse[row]);
    }
    return result;
}

std::vector<float> GravityFeedforward(const SceneModel& scene_model, const nk::Model& model,
                                     const std::vector<Transform>& poses) {
    const auto& articulation = model.articulation;
    std::vector<float> force(poses.size(), 0.0f);
    for (uint32_t body = 0u; body < poses.size(); ++body) {
        const Vec3 gravity{0.0f, 0.0f, kGravity * scene_model.link_mass[body]};
        const Vec3 center = poses[body].TransformPoint(scene_model.link_com[body]);
        for (uint32_t joint = body; joint < poses.size(); joint = articulation.parent_link[joint]) {
            const auto kind = static_cast<phi::ArticulationJointType>(articulation.joint_type[joint]);
            const Vec3 axis = poses[joint].TransformDirection(articulation.joint_axis[joint]);
            if (kind == phi::ArticulationJointType::Revolute)
                force[joint] -= axis.Dot((center - poses[joint].position).Cross(gravity));
            else if (kind == phi::ArticulationJointType::Prismatic)
                force[joint] -= axis.Dot(gravity);
        }
    }
    return force;
}

class Renderer {
public:
    Renderer(const Args& args, const scene::SceneIR& ir, const scene::SceneMap& map,
             uint32_t particles, float spacing) : args_(args) {
        scene_ = render::BuildStudioScene(ir.Ecs(), map,
            std::vector<nuka::runtime::soft::SurfaceTopology>{}, args.width, args.height, false);
        render::UseAuthoredSceneMaterials(scene_);
        const auto environment = scene::nks::Load(args.environment.string());
        const auto asset = render::BuildSceneRenderAsset(environment, args.environment.parent_path().string());
        render::RenderAssetBinding binding;
        render::SetSceneRenderAsset(scene_.world, binding, asset);
        render::ApplySceneLighting(scene_.options, asset);
        render::ApplySceneCamera(scene_.options, asset, args.overview);
        close_options_ = scene_.options;
        render::ApplySceneCamera(close_options_, asset, "close");
        render::AddStudioDensitySurface(scene_, ir.Ecs(), render::kNoId, spacing, 0u, particles);
        auto& surface = scene_.density_surfaces.back();
        surface.params.h = 3.0f * spacing;
        surface.material_id = render::SceneAssetMaterial(asset, binding, "gripper_sample");
        if (args.renderer == "raster") {
            raster_ = std::make_unique<render::VulkanRasterRenderer>();
            std::printf("raster device: %s\n", raster_->DeviceName().c_str());
        } else {
            renderer_ = std::make_unique<render::StudioRtRenderer>();
            Require(renderer_->ok(), "No ray tracing backend");
            renderer_->SetBeauty(true, args.samples);
        }
        std::filesystem::create_directories(args.out / "frames");
        scene::nks::Save(environment, (args.out / "render_environment.nks").string());
    }

    void Frame(uint32_t frame, const std::vector<Vec3>& positions,
               const std::vector<Transform>& links, const std::vector<Transform>& bodies) {
        if (frame % args_.render_stride != 0u) return;
        render::PublishStudioScene(scene_, links, positions, bodies);
        for (uint32_t view = 0u; view < 2u; ++view) {
            const auto& options = view == 0u ? scene_.options : close_options_;
            const auto image = raster_ ? raster_->Render(scene_.world, options) :
                                         renderer_->Render(scene_.world, options);
            Require(image.pixels.size() == size_t(args_.width) * args_.height, "Incomplete image");
            char name[64];
            std::snprintf(name, sizeof(name), "%s_%06u.ppm", view == 0u ? "wide" : "close", frame);
            std::ofstream output(args_.out / "frames" / name, std::ios::binary);
            output << "P6\n" << args_.width << ' ' << args_.height << "\n255\n";
            std::vector<uint8_t> rgb(image.pixels.size() * 3u);
            for (size_t i = 0u; i < image.pixels.size(); ++i) {
                rgb[3u * i] = image.pixels[i].r;
                rgb[3u * i + 1u] = image.pixels[i].g;
                rgb[3u * i + 2u] = image.pixels[i].b;
            }
            Write(output, rgb);
        }
    }
private:
    Args args_;
    render::StudioScene scene_;
    render::RasterOptions close_options_;
    std::unique_ptr<render::StudioRtRenderer> renderer_;
    std::unique_ptr<render::VulkanRasterRenderer> raster_;
};

double Determinant(const float* f) {
    return double(f[0]) * (double(f[4]) * f[8] - double(f[5]) * f[7]) -
           double(f[1]) * (double(f[3]) * f[8] - double(f[5]) * f[6]) +
           double(f[2]) * (double(f[3]) * f[7] - double(f[4]) * f[6]);
}

void Replay(const Args& args) {
    std::ifstream capture(args.replay / "states.bin", std::ios::binary);
    std::vector<uint32_t> header(7u);
    std::vector<float> spacing(1u);
    Read(capture, header); Read(capture, spacing);
    Require(header[0] == 0x4E554B41u && header[1] == 2u, "Unsupported capture format");
    const auto ir = scene::nks::Load((args.replay / "scene.nks").string());
    const auto cooked = cook::CookToModel(ir, 1);
    Renderer renderer(args, ir, cooked.scene_map, header[2], spacing[0]);
    std::vector<Vec3> particles(header[2]);
    std::vector<Transform> bodies(header[3]), links(header[4]);
    std::vector<float> q(header[4]);
    for (uint32_t frame = 0u; frame < header[5]; ++frame) {
        Read(capture, particles); Read(capture, bodies); Read(capture, links); Read(capture, q);
        renderer.Frame(frame, particles, links, bodies);
        if (frame % 30u == 0u) std::printf("render %u/%u\n", frame, header[5]);
    }
}

void Simulate(const Args& args, phi::Device* device, phi::Backend* backend) {
    Require(args.resume || !std::filesystem::exists(args.out / "states.bin"),
            "Capture already exists; use --resume to continue it");
    auto scene_model = BuildModel(args);
    nk::Pipeline::SolverConfig config;
    config.dt = 1.0f / kHz;
    config.substeps = args.substeps;
    config.vel_iters = static_cast<uint16_t>(args.velocity_iterations);
    nk::World world(std::move(scene_model.cooked.model), 1u, device, backend, config);
    Require(world.Ready(), "World creation: " + world.CreationError());
    Require(world.SetExecutionMode(args.execution) == phi::Status::Ok, "Execution configuration failed");
    const auto& model = world.GetModel();
    const auto& cap = model.capacities;
    const float spacing = std::cbrt(scene_model.particle_volume);
    const double volume = scene_model.particle_volume;
    const double mass = kDensity * volume;
    const uint32_t left = scene_model.joints[7], right = scene_model.joints[8];
    auto links = Download<Transform>(world, nk::FieldId::LinkPose);
    auto q = Download<float>(world, nk::FieldId::Q);
    auto target = model.hold_drives.targets;
    auto previous_q = q;
    std::vector<float> previous_effort(q.size(), 0.0f), previous_alpha(cap.particles_per_env, 0.0f);
    std::unique_ptr<Renderer> renderer;
    if (!args.no_render)
        renderer = std::make_unique<Renderer>(args, scene_model.ir, scene_model.cooked.scene_map,
                                             cap.particles_per_env, spacing);
    std::ostringstream metadata;
    metadata << std::setprecision(10) << "{\n  \"dx\": " << args.dx
        << ",\n  \"control_hz\": " << kHz << ",\n  \"substeps\": " << args.substeps
        << ",\n  \"schedule_time_scale\": " << kScheduleScale
        << ",\n  \"dt\": " << config.dt / args.substeps
        << ",\n  \"velocity_iterations\": " << args.velocity_iterations
        << ",\n  \"execution_mode\": \"" << (args.execution == nk::World::ExecutionMode::Graph ? "graph" : "eager") << '"'
        << ",\n  \"capture_frames\": " << args.frames + 1u
        << ",\n  \"particles\": " << cap.particles_per_env
        << ",\n  \"grid_contact_capacity\": " << cap.mpm_contact_capacity_per_env
        << ",\n  \"youngs\": " << kYoungs << ",\n  \"poisson\": " << kPoisson
        << ",\n  \"density\": " << kDensity << ",\n  \"yield_stress\": " << kYield
        << ",\n  \"hardening_modulus\": " << kHardening
        << ",\n  \"specimen_m\": [" << kSize.x << ',' << kSize.y << ',' << kSize.z << ']'
        << ",\n  \"particle_volume_m3\": " << volume
        << ",\n  \"finger_targets_m\": [" << kOpen << ',' << kMild << ',' << kStrong << ']'
        << ",\n  \"gravity_z\": " << kGravity
        << ",\n  \"body_band_m\": 0.0002,\n  \"finger_friction\": 0.8,"
        << "\n  \"actuation\": \"finite PD with sampled gravity feedforward\","
        << "\n  \"actuator_work\": \"trapezoidal estimate at control samples\","
        << "\n  \"angular_balance\": \"APIC angular momentum; gravity moment uses control-sample trapezoids\","
        << "\n  \"interface_readout\": \"last common substep; normal projected velocity residual; particle centers versus physical geometry\","
        << "\n  \"pinned_particles\": 0,\n  \"prescribed_joint_poses\": false\n}\n";
    const auto metadata_path = args.out / "config.json";
    if (args.resume) {
        std::ifstream previous(metadata_path);
        const std::string text((std::istreambuf_iterator<char>(previous)), std::istreambuf_iterator<char>());
        Require(text == metadata.str(), "Resume configuration differs from capture");
    } else {
        std::ofstream output(metadata_path);
        output << metadata.str();
        Require(static_cast<bool>(output), "Configuration write failed");
    }
    const uint64_t identity = nuka::demo::CaptureFileFingerprint(args.executable) ^
        nuka::demo::CaptureFileFingerprint(args.out / "scene.nks") ^
        nuka::demo::CaptureFileFingerprint(args.out / "scene.nka") ^
        nuka::demo::CaptureFileFingerprint(metadata_path);
    double actuator_work = 0.0;
    std::array<double, 3> previous_momentum{}, previous_angular{}, previous_center{};
    uint32_t first_frame = 0u;
    uint64_t checkpoint_sequence = 0u;
    const size_t recorder_bytes = sizeof(actuator_work) + sizeof(previous_momentum) +
        sizeof(previous_angular) + sizeof(previous_center) + previous_effort.size() * sizeof(float);
    if (args.resume) {
        const auto checkpoint = nuka::demo::LoadCaptureCheckpoint(args.out, world, identity,
                                                                  recorder_bytes, args.frames);
        Require(args.stop_after > checkpoint.header.frame || checkpoint.header.frame == args.frames,
                "Stop frame must follow checkpoint");
        Require(world.GetData().UploadPersistent(checkpoint.persistent), "Checkpoint restore failed");
        size_t offset = 0u;
        const auto restore = [&](void* target_bytes, size_t bytes) {
            std::memcpy(target_bytes, checkpoint.recorder.data() + offset, bytes);
            offset += bytes;
        };
        restore(&actuator_work, sizeof(actuator_work));
        restore(previous_momentum.data(), sizeof(previous_momentum));
        restore(previous_angular.data(), sizeof(previous_angular));
        restore(previous_center.data(), sizeof(previous_center));
        restore(previous_effort.data(), previous_effort.size() * sizeof(float));
        previous_q = q = Download<float>(world, nk::FieldId::Q);
        previous_alpha = Download<float>(world, nk::FieldId::ParticlePlastic);
        links = Download<Transform>(world, nk::FieldId::LinkPose);
        std::filesystem::resize_file(args.out / "states.bin", checkpoint.header.capture_bytes);
        std::filesystem::resize_file(args.out / "metrics.csv", checkpoint.header.metrics_bytes);
        first_frame = checkpoint.header.frame + 1u;
        checkpoint_sequence = checkpoint.header.sequence + 1u;
        std::printf("resume after frame %u\n", checkpoint.header.frame);
    }
    const auto open_mode = args.resume ? std::ios::app : std::ios::trunc;
    std::ofstream capture(args.out / "states.bin", std::ios::binary | open_mode);
    std::ofstream metrics(args.out / "metrics.csv", open_mode);
    if (!args.resume) {
        Write(capture, std::vector<uint32_t>{0x4E554B41u, 2u, cap.particles_per_env,
            cap.bodies_per_env, cap.links_per_env, args.frames + 1u, static_cast<uint32_t>(kHz)});
        Write(capture, std::vector<float>{spacing});
        metrics << "frame,time_s,stage,finger_target_m,left_q_m,right_q_m,width_x_m,width_y_m,height_m,"
        "center_x_m,center_y_m,center_z_m,alpha_mean,alpha_max,alpha_increment_min,"
        "det_fp_error,det_fe_min,det_fe_max,elastic_j,hardening_j,plastic_dissipation_j,"
        "kinetic_j,gravity_j,actuator_work_estimate_j,left_force_n,right_force_n,"
        "left_effort_n,right_effort_n,max_speed_m_s,grid_contact_peak,grid_contact_overflow,env_status,"
        "central_width_y_m,shape_rms_m,normal_residual_m_s,friction_cone_violation_ns,penetration_m,"
        "linear_balance_error_ns,angular_balance_error_nms\n";
    }
    metrics << std::setprecision(10);
    for (uint32_t frame = first_frame; frame <= args.frames; ++frame) {
        const double time = frame / double(kHz);
        if (frame > 0u) {
            target[left] = target[right] = FingerTarget((frame - 0.5) / kHz);
            const auto feedforward = GravityFeedforward(scene_model, model, links);
            Require(world.GetData().UploadField(nk::FieldId::DriveTarget, target.data(),
                    target.size() * sizeof(float)), "Target upload failed");
            Require(world.GetData().UploadField(nk::FieldId::JointF, feedforward.data(),
                    feedforward.size() * sizeof(float)), "Feedforward upload failed");
            Require(world.StepConfigured() == phi::Status::Ok, "World step failed");
        }
        Require(world.Synchronize() == phi::Status::Ok, "Simulation completion failed");
        const auto positions = Download<Vec3>(world, nk::FieldId::ParticlePos);
        const auto velocity = Download<Vec3>(world, nk::FieldId::ParticleVel);
        const auto elastic = Download<float>(world, nk::FieldId::ParticleF);
        const auto plastic = Download<float>(world, nk::FieldId::ParticlePlasticF);
        const auto alpha = Download<float>(world, nk::FieldId::ParticlePlastic);
        const auto affine = Download<float>(world, nk::FieldId::ParticleC);
        const auto bodies = Download<Transform>(world, nk::FieldId::BodyPose);
        const auto reaction = Download<Vec3>(world, nk::FieldId::MpmBodyReaction);
        const auto effort = Download<float>(world, nk::FieldId::ActuatorEffort);
        const auto status = Download<uint32_t>(world, nk::FieldId::EnvStatus);
        const auto peak = Download<uint64_t>(world, nk::FieldId::GridContactPeak);
        const auto overflow = Download<uint64_t>(world, nk::FieldId::GridContactOverflow);
        const auto boundary_impulse = Download<Vec3>(world, nk::FieldId::MpmBoundaryImpulse);
        const auto body_moment = Download<Vec3>(world, nk::FieldId::StepMpmBodyMoment);
        const auto boundary_moment = Download<Vec3>(world, nk::FieldId::MpmBoundaryMoment);
        links = Download<Transform>(world, nk::FieldId::LinkPose);
        q = Download<float>(world, nk::FieldId::Q);
        const auto interface = MeasureInterface(world, positions, bodies, args.dx);
        Write(capture, positions); Write(capture, bodies); Write(capture, links); Write(capture, q);
        for (size_t i = 0u; i < q.size(); ++i)
            actuator_work += 0.5 * (effort[i] + previous_effort[i]) * (q[i] - previous_q[i]);
        previous_q = q;
        previous_effort = effort;
        Vec3 lo{1.0e30f, 1.0e30f, 1.0e30f}, hi{-1.0e30f, -1.0e30f, -1.0e30f};
        std::array<double, 3> center{}, momentum{}, angular{};
        double central_lo = 1.0e30, central_hi = -1.0e30, displacement_sq = 0.0;
        double alpha_sum = 0.0, alpha_max = 0.0, increment_min = 0.0, fp_error = 0.0;
        double fe_min = 1.0e30, fe_max = 0.0, elastic_j = 0.0, hardening_j = 0.0;
        double kinetic_j = 0.0, gravity_j = 0.0, speed_max = 0.0;
        for (uint32_t p = 0u; p < positions.size(); ++p) {
            const auto x = positions[p], v = velocity[p];
            Require(std::isfinite(x.LengthSq()) && std::isfinite(v.LengthSq()), "Nonfinite particle state");
            lo.x = std::min(lo.x, x.x); lo.y = std::min(lo.y, x.y); lo.z = std::min(lo.z, x.z);
            hi.x = std::max(hi.x, x.x); hi.y = std::max(hi.y, x.y); hi.z = std::max(hi.z, x.z);
            center[0] += x.x; center[1] += x.y; center[2] += x.z;
            momentum[0] += mass * v.x; momentum[1] += mass * v.y; momentum[2] += mass * v.z;
            const double apic = 0.25 * args.dx * args.dx;
            angular[0] += mass * (double(x.y) * v.z - double(x.z) * v.y +
                apic * (double(affine[9u * p + 7u]) - affine[9u * p + 5u]));
            angular[1] += mass * (double(x.z) * v.x - double(x.x) * v.z +
                apic * (double(affine[9u * p + 2u]) - affine[9u * p + 6u]));
            angular[2] += mass * (double(x.x) * v.y - double(x.y) * v.x +
                apic * (double(affine[9u * p + 3u]) - affine[9u * p + 1u]));
            const Vec3 rest = model.particles.initial_pos[p];
            if (std::fabs(rest.x - kCenter.x) <= kSize.x * 0.25f &&
                std::fabs(rest.z - kCenter.z) <= kSize.z * 0.25f) {
                central_lo = std::min(central_lo, double(x.y));
                central_hi = std::max(central_hi, double(x.y));
            }
            alpha_sum += alpha[p]; alpha_max = std::max(alpha_max, double(alpha[p]));
            increment_min = std::min(increment_min, double(alpha[p] - previous_alpha[p]));
            fp_error = std::max(fp_error, std::fabs(Determinant(plastic.data() + 9u * p) - 1.0));
            const double determinant = Determinant(elastic.data() + 9u * p);
            fe_min = std::min(fe_min, determinant); fe_max = std::max(fe_max, determinant);
            nk::material::HenckyResponse response;
            Require(nk::material::EvaluateHenckyJ2(elastic.data() + 9u * p,
                {kYoungs, kPoisson, kYield, kHardening}, response) == nk::material::ConstitutiveStatus::Ok,
                "Invalid material state");
            elastic_j += volume * response.elastic_energy;
            hardening_j += volume * 0.5 * kHardening * alpha[p] * alpha[p];
            double affine_norm = 0.0;
            for (uint32_t k = 0u; k < 9u; ++k)
                affine_norm += double(affine[9u * p + k]) * affine[9u * p + k];
            kinetic_j += 0.5 * mass * (v.LengthSq() + 0.25 * args.dx * args.dx * affine_norm);
            gravity_j -= mass * kGravity * (x.z - 0.5f);
            speed_max = std::max(speed_max, double(v.Length()));
        }
        for (auto& value : center) value /= positions.size();
        for (uint32_t p = 0u; p < positions.size(); ++p) {
            const Vec3 difference = positions[p] - model.particles.initial_pos[p];
            const double x = difference.x - (center[0] - kCenter.x);
            const double y = difference.y - (center[1] - kCenter.y);
            const double z = difference.z - (center[2] - kCenter.z);
            displacement_sq += x * x + y * y + z * z;
        }
        std::array<double, 3> balance{}, angular_balance{};
        if (frame > 0u) {
            for (size_t axis = 0u; axis < 3u; ++axis) {
                balance[axis] = momentum[axis] - previous_momentum[axis];
                angular_balance[axis] = angular[axis] - previous_angular[axis];
            }
            const auto accumulate = [](std::array<double, 3>& target, const auto& values) {
                for (const auto value : values) {
                    target[0] += value.x; target[1] += value.y; target[2] += value.z;
                }
            };
            accumulate(balance, reaction); accumulate(balance, boundary_impulse);
            accumulate(angular_balance, body_moment); accumulate(angular_balance, boundary_moment);
            const double gravity_impulse = kGravity * mass * positions.size() / kHz;
            balance[2] -= gravity_impulse;
            angular_balance[0] -= 0.5 * (center[1] + previous_center[1]) * gravity_impulse;
            angular_balance[1] += 0.5 * (center[0] + previous_center[0]) * gravity_impulse;
        }
        previous_momentum = momentum; previous_angular = angular; previous_center = center;
        const auto norm = [](const std::array<double, 3>& value) {
            return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        };
        previous_alpha = alpha;
        const auto finger_force = [&](uint32_t link) {
            return reaction[model.articulation.link_body[link]].Dot(
                links[link].TransformDirection(model.articulation.joint_axis[link])) * kHz;
        };
        metrics << frame << ',' << time << ',' << Stage(time) << ',' << FingerTarget(time) << ','
            << q[left] << ',' << q[right] << ',' << hi.x - lo.x << ',' << hi.y - lo.y << ','
            << hi.z - lo.z << ',' << center[0] << ',' << center[1] << ',' << center[2] << ','
            << alpha_sum / positions.size() << ',' << alpha_max << ',' << increment_min << ','
            << fp_error << ',' << fe_min << ',' << fe_max << ',' << elastic_j << ',' << hardening_j << ','
            << volume * kYield * alpha_sum << ',' << kinetic_j << ',' << gravity_j << ','
            << actuator_work << ',' << finger_force(left) << ',' << finger_force(right) << ','
            << effort[left] << ',' << effort[right] << ',' << speed_max << ',' << peak[0] << ','
            << overflow[0] << ',' << status[0] << ',' << central_hi - central_lo << ','
            << std::sqrt(displacement_sq / positions.size()) << ',' << interface.normal_residual << ','
            << interface.cone_violation << ',' << interface.penetration << ','
            << norm(balance) << ',' << norm(angular_balance) << '\n';
        if (frame % 12u == 0u || status[0] != 0u) {
            metrics.flush(); capture.flush();
            std::printf("t=%.3f %s q=(%.5f,%.5f) width=%.5f alpha=%.6f force=(%.3f,%.3f) contacts=%llu status=%u\n",
                time, Stage(time), q[left], q[right], hi.y - lo.y, alpha_sum / positions.size(),
                finger_force(left), finger_force(right), static_cast<unsigned long long>(peak[0]), status[0]);
        }
        Require(status[0] == 0u, "Environment flags " + std::to_string(status[0]));
        if (renderer) renderer->Frame(frame, positions, links, bodies);
        if (args.checkpoint_interval != 0u &&
            (frame % args.checkpoint_interval == 0u || frame == args.stop_after || frame == args.frames)) {
            capture.flush(); metrics.flush();
            Require(static_cast<bool>(capture) && static_cast<bool>(metrics), "Capture flush failed");
            nuka::demo::CaptureCheckpoint checkpoint;
            checkpoint.header.identity = identity;
            checkpoint.header.sequence = checkpoint_sequence++;
            checkpoint.header.frame = frame;
            checkpoint.header.capture_bytes = static_cast<uint64_t>(capture.tellp());
            checkpoint.header.metrics_bytes = static_cast<uint64_t>(metrics.tellp());
            checkpoint.recorder.resize(recorder_bytes);
            size_t offset = 0u;
            const auto append = [&](const void* bytes, size_t count) {
                std::memcpy(checkpoint.recorder.data() + offset, bytes, count);
                offset += count;
            };
            append(&actuator_work, sizeof(actuator_work));
            append(previous_momentum.data(), sizeof(previous_momentum));
            append(previous_angular.data(), sizeof(previous_angular));
            append(previous_center.data(), sizeof(previous_center));
            append(previous_effort.data(), previous_effort.size() * sizeof(float));
            nuka::demo::SaveCaptureCheckpoint(args.out, world, checkpoint);
        }
        if (frame == args.stop_after && frame < args.frames) return;
    }
    Require(world.Reset() == phi::Status::Ok, "Reset failed");
    const auto reset_alpha = Download<float>(world, nk::FieldId::ParticlePlastic);
    const auto reset_positions = Download<Vec3>(world, nk::FieldId::ParticlePos);
    const auto reset_q = Download<float>(world, nk::FieldId::Q);
    Require(std::all_of(reset_alpha.begin(), reset_alpha.end(), [](float value) { return value == 0.0f; }),
            "Plastic state did not reset");
    Require(std::memcmp(reset_positions.data(), model.particles.initial_pos.data(),
                        reset_positions.size() * sizeof(Vec3)) == 0 &&
            reset_q == model.articulation.initial_q, "Physical state did not reset");
    Require(Download<uint64_t>(world, nk::FieldId::GridContactPeak)[0] == 0u,
            "Contact peak did not reset");
    std::ofstream completion(args.out / "completion.json");
    completion << "{\"frames\": " << args.frames + 1u << ", \"reset_passed\": true, "
        "\"material_reset_between_loads\": false, \"resume_frame\": " << first_frame
        << ", \"process_graph_replays\": " << world.GraphReplays() << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const Args args = ParseArgs(argc, argv);
        std::filesystem::create_directories(args.out);
        if (!args.replay.empty()) { Replay(args); return 0; }
        auto* device = phi::InitBestDevice();
        Require(device != nullptr, "No physics device");
        auto* backend = phi::DeviceInitBackend(device, nullptr);
        Require(backend != nullptr, "No physics backend");
        Simulate(args, device, backend);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "robot elastoplastic demo: %s\n", error.what());
        return 1;
    }
}
