// A freely falling rigid mesh indents a Hencky J2 material through the production World.
// Captured particle and rigid states also drive the headless surface renderer.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision/mesh_surface.hpp"
#include "collision/shape_kind.hpp"
#include "import/cooker/mesh_surface_cooker.hpp"
#include "import/mesh_file_loader.hpp"
#include "nk/material/hencky_j2.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "render/mesh_normals.hpp"
#include "render/studio_beauty.hpp"
#include "render/scene_asset.hpp"
#include "scene/format/nks.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/format/json.hpp"
#include "scene/scene_map.hpp"
#include "tools/perf/cuda_step_measurements.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace render = nuka::render;
namespace cook = nuka::scene::cook;
namespace collision = nuka::collision;
using nuka::math::Transform;
using nuka::math::Quat;
using nuka::math::Vec3;
using Json = nuka::scene::json::Value;

constexpr float kSampleHz = 120.0f, kBase = 0.024f;
constexpr Vec3 kSize{0.24f, 0.20f, 0.06f};
constexpr Vec3 kPlateHalf{0.18f, 0.15f, kBase * 0.5f};
constexpr float kDensity = 1000.0f, kYoungs = 50000.0f, kPoisson = 0.30f;
constexpr float kYield = 12000.0f, kHardening = 3000.0f, kGravity = -9.81f;
constexpr float kFriction = 0.20f;

struct Args {
    std::filesystem::path out = "out/elastoplastic_bunny";
    std::filesystem::path replay;
    std::filesystem::path environment = std::filesystem::path(NUKA_SOURCE_DIR) /
        "examples/assets/nuka_lab/bunny.nks";
    std::filesystem::path perf_json;
    std::filesystem::path mesh = std::filesystem::path(NUKA_SOURCE_DIR) /
        ".nuka-assets/generated/bunny_solid_120mm.obj";
    float dx = 0.005f, mass = 2.5f, drop_height = 0.18f, duration = 3.6f;
    float contact_band = 0.0f;
    uint32_t steps_per_frame = 64u, width = 1600u, height = 1000u, samples = 128u;
    uint32_t render_stride = 1u;
    bool no_render = false;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--no-render") { args.no_render = true; continue; }
        Require(i + 1 < argc, "Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--out-dir") args.out = value;
        else if (key == "--replay") args.replay = value;
        else if (key == "--environment") args.environment = value;
        else if (key == "--perf-json") args.perf_json = value;
        else if (key == "--mesh") args.mesh = value;
        else if (key == "--dx") args.dx = std::stof(value);
        else if (key == "--mass") args.mass = std::stof(value);
        else if (key == "--drop-height") args.drop_height = std::stof(value);
        else if (key == "--duration") args.duration = std::stof(value);
        else if (key == "--contact-band") args.contact_band = std::stof(value);
        else if (key == "--steps-per-frame") args.steps_per_frame = std::stoul(value);
        else if (key == "--width") args.width = std::stoul(value);
        else if (key == "--height") args.height = std::stoul(value);
        else if (key == "--samples") args.samples = std::stoul(value);
        else if (key == "--render-stride") args.render_stride = std::stoul(value);
        else throw std::runtime_error("Unknown argument " + key);
    }
    for (float value : {args.dx, args.mass, args.duration, args.drop_height})
        Require(std::isfinite(value) && value > 0.0f, "Physical inputs must be finite and positive");
    Require(args.steps_per_frame && args.width && args.height && args.samples && args.render_stride,
            "Counts must be positive");
    Require(std::isfinite(args.contact_band) && args.contact_band >= 0.0f, "Invalid contact band");
    Require(args.perf_json.empty() || args.replay.empty(), "Step timing requires live simulation");
    return args;
}

Json ReadJson(const std::filesystem::path& path) {
    std::ifstream source(path);
    Require(bool(source), "Missing " + path.string());
    std::ostringstream text; text << source.rdbuf();
    return Json::Parse(text.str());
}

void WriteJson(const std::filesystem::path& path, const Json& value) {
    std::ofstream output(path); output << value.Dump() << '\n';
    Require(bool(output), "JSON write failed");
}

Json Array(Vec3 value) {
    Json array = Json::Array();
    for (float x : {value.x, value.y, value.z}) array.PushBack(Json::Float(x));
    return array;
}

Vec3 Vector(const Json& value) {
    const auto& a = value.Elements();
    Require(a.size() == 3u, "Expected a three-component vector");
    return {a[0].AsFloat(), a[1].AsFloat(), a[2].AsFloat()};
}

template<class T> void Write(std::ostream& stream, const T* values, size_t count) {
    stream.write(reinterpret_cast<const char*>(values), std::streamsize(sizeof(T) * count));
    Require(bool(stream), "Capture write failed");
}

template<class T> void Read(std::istream& stream, T* values, size_t count) {
    stream.read(reinterpret_cast<char*>(values), std::streamsize(sizeof(T) * count));
    Require(bool(stream), "Capture read failed");
}

template<class T> void Download(nk::World& world, nk::FieldId field, std::vector<T>& values) {
    Require(world.GetData().DownloadField(field, values.data(), values.size() * sizeof(T)),
            "State readout failed");
}

render::MeshGeometry LoadMesh(const std::filesystem::path& path) {
    auto source = nuka::import::LoadObj(path.string());
    render::MeshGeometry mesh;
    mesh.positions = std::move(source.vertices);
    mesh.indices = std::move(source.indices);
    mesh.normals = render::SmoothNormals(mesh.positions, mesh.indices);
    return mesh;
}

Vec3 Vertex(const render::MeshGeometry& mesh, uint32_t index) {
    const auto* p = mesh.positions.data() + 3u * index;
    return {p[0], p[1], p[2]};
}

float Bottom(const render::MeshGeometry& mesh, const Transform& pose) {
    float result = std::numeric_limits<float>::infinity();
    for (uint32_t i = 0u; i < mesh.VertexCount(); ++i)
        result = std::min(result, pose.TransformPoint(Vertex(mesh, i)).z);
    return result;
}

struct Scene {
    nk::Model model;
    uint32_t bunny = 0u, support = 0u;
    Transform inertia;
    Vec3 principal_moments, half;
    Json asset;
};

Scene BuildModel(const Args& args, const render::MeshGeometry& mesh) {
    Scene scene;
    auto mass_path = args.mesh; mass_path.replace_extension(".json");
    scene.asset = ReadJson(mass_path);
    scene.inertia.position = Vector(scene.asset.At("center_of_mass_m"));
    const auto& q = scene.asset.At("principal_rotation_wxyz").Elements();
    Require(q.size() == 4u, "Missing principal frame rotation");
    scene.inertia.rotation = Quat{q[0].AsFloat(), q[1].AsFloat(), q[2].AsFloat(), q[3].AsFloat()};
    scene.principal_moments = Vector(scene.asset.At("principal_inertia_per_kg")) * args.mass;
    auto& model = scene.model;
    auto& cap = model.capacities;
    cap.env_count = 1u;
    nk::Model::BodyInit body;
    body.inertial_frame = scene.inertia;
    body.inv_mass = 1.0f / args.mass;
    body.inv_inertia = {1.0f / scene.principal_moments.x, 1.0f / scene.principal_moments.y,
                       1.0f / scene.principal_moments.z};
    body.pose.rotation = Quat::FromAxisAngle({0, 0, 1}, 0.18f);
    body.pose.position.z = kBase + kSize.z + args.drop_height - Bottom(mesh, body.pose);
    scene.bunny = static_cast<uint32_t>(model.body_init.size());
    model.body_init.push_back(body);
    auto surface = nuka::import::cooker::CookMeshSurfaceCached(mesh.positions.data(), mesh.VertexCount(),
        mesh.indices.data(), static_cast<uint32_t>(mesh.indices.size() / 3u), false);
    Require((surface.info.flags & collision::kMeshSurfaceClosed) != 0u,
            "The impactor must have a closed collision surface");
    model.hull_verts = mesh.positions;
    model.mesh_triangles = mesh.indices;
    model.mesh_bvh_nodes = std::move(surface.nodes);
    model.mesh_surface_info.push_back(surface.info);
    float radius = 0.0f;
    for (uint32_t i = 0u; i < mesh.VertexCount(); ++i) {
        const auto p = Vertex(mesh, i);
        scene.half.x = std::max(scene.half.x, std::abs(p.x));
        scene.half.y = std::max(scene.half.y, std::abs(p.y));
        scene.half.z = std::max(scene.half.z, std::abs(p.z));
        radius = std::max(radius, std::sqrt(p.LengthSq()));
    }
    nk::Model::PairDrivenShape shape;
    shape.kind = collision::kShapeSdfMesh;
    shape.body_id = static_cast<int32_t>(scene.bunny);
    shape.params[0] = radius;
    shape.params[1] = scene.half.x; shape.params[2] = scene.half.y; shape.params[3] = scene.half.z;
    shape.hull_vert_count = mesh.VertexCount();
    model.shape_table_rows.push_back(shape);
    model.samp_points = mesh.positions;
    model.samp_ranges = {0u, mesh.VertexCount()};
    body = nk::Model::BodyInit{};
    body.pose.position.z = kPlateHalf.z;
    scene.support = static_cast<uint32_t>(model.body_init.size());
    model.body_init.push_back(body);
    shape = nk::Model::PairDrivenShape{};
    shape.kind = collision::kShapeBox;
    shape.body_id = static_cast<int32_t>(scene.support);
    shape.params[0] = kPlateHalf.x; shape.params[1] = kPlateHalf.y; shape.params[2] = kPlateHalf.z;
    model.shape_table_rows.push_back(shape);
    model.mesh_surface_info.resize(model.body_init.size());
    model.samp_ranges.resize(model.body_init.size() * 2u);
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = cap.bodies_per_env;
    cap.max_hull_verts = mesh.VertexCount();
    cap.max_mesh_triangles = static_cast<uint32_t>(mesh.indices.size() / 3u);
    cap.max_mesh_bvh_nodes = static_cast<uint32_t>(model.mesh_bvh_nodes.size());
    cap.max_samp_points = mesh.VertexCount();
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    model.contact_family = nk::ContactFamily::PairDriven;
    const float spacing = args.dx * 0.5f;
    const uint32_t nx = std::lround(kSize.x / spacing), ny = std::lround(kSize.y / spacing),
                   nz = std::lround(kSize.z / spacing);
    Require(std::abs(nx*spacing-kSize.x) < 1e-6f && std::abs(ny*spacing-kSize.y) < 1e-6f &&
            std::abs(nz*spacing-kSize.z) < 1e-6f, "Particle spacing must divide the material dimensions");
    cook::MpmCookInput material;
    for (uint32_t z = 0u; z < nz; ++z)
        for (uint32_t y = 0u; y < ny; ++y)
            for (uint32_t x = 0u; x < nx; ++x)
                material.positions.push_back({(x+0.5f)*spacing-kSize.x*0.5f,
                    (y+0.5f)*spacing-kSize.y*0.5f, kBase+(z+0.5f)*spacing});
    const size_t count = material.positions.size();
    const float volume = spacing*spacing*spacing;
    material.velocities.assign(count, Vec3::Zero());
    material.inv_mass.assign(count, 1.0f / (kDensity*volume));
    material.vol0.assign(count, volume);
    material.material = {kYoungs, kPoisson, kDensity};
    material.material.model_kind = nk::MpmMaterial::kHenckyJ2;
    material.material.yield_stress = kYield;
    material.material.hardening_modulus = kHardening;
    material.dx = args.dx;
    material.substeps = 1u;
    material.grid_origin = {-0.20f, -0.18f, -0.02f};
    material.grid_dims[0] = static_cast<uint32_t>(std::ceil(0.40f/args.dx)) + 1u;
    material.grid_dims[1] = static_cast<uint32_t>(std::ceil(0.36f/args.dx)) + 1u;
    material.grid_dims[2] = static_cast<uint32_t>(std::ceil(0.30f/args.dx)) + 1u;
    material.floor_d = -1.0f;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(model, 1u, soft, material);
    model.particles.mpm_body_friction = kFriction;
    model.particles.mpm_body_band = args.contact_band;
    return scene;
}

render::MeshGeometry BoxMesh(Vec3 half) {
    render::MeshGeometry mesh;
    const std::array<Vec3, 8> points{{{-half.x,-half.y,-half.z}, {half.x,-half.y,-half.z},
        {half.x,half.y,-half.z}, {-half.x,half.y,-half.z}, {-half.x,-half.y,half.z},
        {half.x,-half.y,half.z}, {half.x,half.y,half.z}, {-half.x,half.y,half.z}}};
    constexpr uint32_t faces[6][4] = {{0,3,2,1}, {4,5,6,7}, {0,1,5,4},
        {1,2,6,5}, {2,3,7,6}, {3,0,4,7}};
    for (const auto& face : faces) {
        const Vec3 normal = (points[face[1]]-points[face[0]]).Cross(
            points[face[2]]-points[face[0]]).Normalized();
        const uint32_t first = mesh.VertexCount();
        for (uint32_t id : face) {
            const auto p = points[id];
            mesh.positions.insert(mesh.positions.end(), {p.x,p.y,p.z});
            mesh.normals.insert(mesh.normals.end(), {normal.x,normal.y,normal.z});
        }
        mesh.indices.insert(mesh.indices.end(), {first,first+1,first+2,first,first+2,first+3});
    }
    return mesh;
}

class Renderer {
public:
    Renderer(const Args& args, float spacing, uint32_t particles, const render::MeshGeometry& mesh)
        : args_(args) {
        nuka::scene::Registry registry;
        nuka::scene::SceneMap map;
        scene_ = render::BuildStudioScene(registry, map,
            std::vector<nuka::runtime::soft::SurfaceTopology>{}, args.width, args.height, false);
        const auto environment = nuka::scene::nks::Load(args.environment.string());
        const auto asset = render::BuildSceneRenderAsset(environment, args.environment.parent_path().string());
        render::RenderAssetBinding binding;
        render::SetSceneRenderAsset(scene_.world, binding, asset);
        render::ApplySceneLighting(scene_.options, asset);
        render::ApplySceneCamera(scene_.options, asset, "overview");
        begin_options_ = scene_.options;
        end_options_ = scene_.options;
        render::ApplySceneCamera(end_options_, asset, "close");
        render::AddStudioDensitySurface(scene_, registry, render::kNoId, spacing, 0u, particles);
        scene_.density_surfaces.back().params.h = 3.0f * spacing;
        scene_.density_surfaces.back().material_id = render::SceneAssetMaterial(asset, binding, "elastoplastic");
        const uint32_t metal_id = render::SceneAssetMaterial(asset, binding, "impact_metal");
        render::RenderInstance instance;
        instance.mesh_id = scene_.world.meshes.InternPrimitive("impact_mesh", [&] { return mesh; });
        instance.render_material_id = metal_id;
        bunny_instance_ = scene_.world.instances.size();
        scene_.world.instances.push_back(instance);
        instance.mesh_id = scene_.world.meshes.InternPrimitive("impact_support", [] { return BoxMesh(kPlateHalf); });
        instance.world_xform.position.z = kPlateHalf.z;
        scene_.world.instances.push_back(instance);
        renderer_ = std::make_unique<render::StudioRtRenderer>();
        Require(renderer_->ok(), "No ray tracing backend");
        renderer_->SetBeauty(true, args.samples);
        std::filesystem::create_directories(args.out / "frames");
        nuka::scene::nks::Save(environment, (args.out / "render_environment.nks").string());
        Json config = Json::Object();
        config.Set("width", Json::Int(args.width)); config.Set("height", Json::Int(args.height));
        config.Set("samples", Json::Int(args.samples)); config.Set("render_stride", Json::Int(args.render_stride));
        config.Set("camera_eye_begin", Array(begin_options_.camera_eye));
        config.Set("camera_target_begin", Array(begin_options_.camera_target));
        config.Set("camera_eye_end", Array(end_options_.camera_eye));
        config.Set("camera_target_end", Array(end_options_.camera_target));
        config.Set("camera_move_begin_s", Json::Float(0.45));
        config.Set("camera_move_end_s", Json::Float(1.55));
        config.Set("camera_fov_degrees", Json::Float(scene_.options.camera_fov_degrees));
        config.Set("environment_asset", Json::Str(args.environment.string()));
        config.Set("surface_kernel_spacing_ratio", Json::Float(3));
        config.Set("surface_cell_spacing_ratio", Json::Float(0.5));
        config.Set("surface_iso_fraction", Json::Float(0.5));
        config.Set("state_interpolation", Json::Bool(false));
        WriteJson(args.out / "render_config.json", config);
    }

    void Frame(uint32_t frame, const std::vector<Vec3>& positions, const Transform& bunny) {
        if (frame % args_.render_stride) return;
        float u = std::clamp((frame/kSampleHz-0.45f)/1.10f, 0.0f, 1.0f);
        u = u*u*u*(10.0f+u*(-15.0f+6.0f*u));
        scene_.options.camera_eye = begin_options_.camera_eye * (1.0f-u) + end_options_.camera_eye * u;
        scene_.options.camera_target = begin_options_.camera_target * (1.0f-u) + end_options_.camera_target * u;
        scene_.options.camera_up = (begin_options_.camera_up * (1.0f-u) + end_options_.camera_up * u).Normalized();
        render::PublishStudioScene(scene_, {}, positions);
        scene_.world.instances[bunny_instance_].world_xform = bunny;
        const auto result = renderer_->Render(scene_.world, scene_.options);
        Require(result.pixels.size() == size_t(args_.width)*args_.height, "Incomplete rendered image");
        char name[48]; std::snprintf(name, sizeof(name), "frame_%06u.ppm", frame);
        std::ofstream output(args_.out / "frames" / name, std::ios::binary);
        output << "P6\n" << args_.width << ' ' << args_.height << "\n255\n";
        std::vector<uint8_t> pixels(result.pixels.size()*3u);
        for (size_t i = 0u; i < result.pixels.size(); ++i) {
            pixels[3u*i] = result.pixels[i].r; pixels[3u*i+1u] = result.pixels[i].g;
            pixels[3u*i+2u] = result.pixels[i].b;
        }
        Write(output, pixels.data(), pixels.size());
    }
private:
    Args args_;
    render::StudioScene scene_;
    render::RasterOptions begin_options_, end_options_;
    size_t bunny_instance_ = 0u;
    std::unique_ptr<render::StudioRtRenderer> renderer_;
};

double Determinant(const float* f) {
    return double(f[0])*(double(f[4])*f[8]-double(f[5])*f[7]) -
        double(f[1])*(double(f[3])*f[8]-double(f[5])*f[6]) +
        double(f[2])*(double(f[3])*f[7]-double(f[4])*f[6]);
}

void Replay(const Args& args) {
    std::ifstream source(args.replay / "positions.bin", std::ios::binary);
    uint32_t header[5]; float spacing;
    Read(source, header, 5u); Read(source, &spacing, 1u);
    Require(header[0] == 0x4E554B41u && header[1] == 2u, "Unsupported impact capture");
    std::vector<Vec3> positions(header[2]);
    std::vector<Transform> bodies(header[3]);
    const auto config = ReadJson(args.replay / "config.json");
    const uint32_t bunny = static_cast<uint32_t>(config.At("bunny_body").AsInt());
    Require(bunny < bodies.size(), "Invalid captured body index");
    Renderer renderer(args, spacing, header[2], LoadMesh(args.replay / "bunny.obj"));
    for (uint32_t frame = 0u; frame < header[4]; ++frame) {
        Read(source, positions.data(), positions.size());
        Read(source, bodies.data(), bodies.size());
        renderer.Frame(frame, positions, bodies[bunny]);
        if (frame % 24u == 0u) std::printf("render frame %u/%u\n", frame, header[4]);
    }
}

void Simulate(const Args& args, phi::Device* device, phi::Backend* backend) {
    const auto mesh = LoadMesh(args.mesh);
    auto scene = BuildModel(args, mesh);
    const uint32_t count = scene.model.capacities.particles_per_env;
    const uint32_t body_count = scene.model.capacities.bodies_per_env;
    const uint32_t frames = std::lround(args.duration*kSampleHz);
    const float spacing = args.dx*0.5f;
    const double volume = double(spacing)*spacing*spacing, mass = kDensity*volume;
    const double dt = 1.0/(kSampleHz*args.steps_per_frame);
    const auto initial = scene.model.particles.initial_pos;
    std::vector<Transform> initial_bodies;
    for (const auto& body : scene.model.body_init) initial_bodies.push_back(body.pose);
    nk::Pipeline::SolverConfig solver;
    solver.dt = float(dt);
    nk::World world(std::move(scene.model), 1u, device, backend, solver);
    Require(world.Ready(), "World creation: " + world.CreationError());
    Require(world.FieldPtr(nk::FieldId::MpmBodyReaction) && world.FieldPtr(nk::FieldId::MpmBodyAngReaction),
            "No contact impulse readout");
    Require(world.SetExecutionMode(nk::World::ExecutionMode::Graph) == phi::Status::Ok, "Graph mode failed");
    nuka::perf::CudaStepMeasurements measurements(world, !args.perf_json.empty(),
        args.steps_per_frame, frames);
    const auto& model = world.GetModel();
    const collision::MeshSurfaceView surface{model.hull_verts.data(), model.mesh_triangles.data(),
        model.mesh_bvh_nodes.data(), {model.capacities.max_hull_verts, model.capacities.max_mesh_triangles,
                                    model.capacities.max_mesh_bvh_nodes}};
    const auto info = model.mesh_surface_info[scene.bunny];
    std::filesystem::copy_file(args.mesh, args.out / "bunny.obj", std::filesystem::copy_options::overwrite_existing);
    WriteJson(args.out / "bunny.json", scene.asset);
    Json config = Json::Object();
    config.Set("dx", Json::Float(args.dx)); config.Set("spacing", Json::Float(spacing));
    config.Set("dt", Json::Float(dt)); config.Set("sample_hz", Json::Float(kSampleHz));
    config.Set("steps_per_frame", Json::Int(args.steps_per_frame)); config.Set("particles", Json::Int(count));
    config.Set("frames", Json::Int(frames+1u)); config.Set("bunny_body", Json::Int(scene.bunny));
    config.Set("support_body", Json::Int(scene.support)); config.Set("bunny_mass_kg", Json::Float(args.mass));
    config.Set("drop_height_m", Json::Float(args.drop_height)); config.Set("specimen_m", Array(kSize));
    config.Set("base_z", Json::Float(kBase)); config.Set("density", Json::Float(kDensity));
    config.Set("youngs", Json::Float(kYoungs)); config.Set("poisson", Json::Float(kPoisson));
    config.Set("yield_stress", Json::Float(kYield)); config.Set("hardening_modulus", Json::Float(kHardening));
    config.Set("friction", Json::Float(kFriction));
    config.Set("body_contact_band", Json::Float(args.contact_band > 0.0f ? args.contact_band : args.dx));
    config.Set("gravity_z", Json::Float(kGravity)); config.Set("collision_geometry", Json::Str("closed triangle surface"));
    config.Set("source_asset", scene.asset); config.Set("mass_model", Json::Str("uniform closed mesh volume"));
    uint64_t arena_bytes[3]{};
    nk::Arena::ComputeSegments(model.capacities, arena_bytes);
    config.Set("data_arena_bytes", Json::Int(arena_bytes[0]+arena_bytes[1]+arena_bytes[2]));
    WriteJson(args.out / "config.json", config);
    std::ofstream capture(args.out / "positions.bin", std::ios::binary);
    const uint32_t header[] = {0x4E554B41u, 2u, count, body_count, frames+1u};
    Write(capture, header, 5u); Write(capture, &spacing, 1u);
    std::ofstream metrics(args.out / "metrics.csv");
    metrics << "frame,time_s,bunny_bottom_m,bunny_com_x_m,bunny_com_y_m,bunny_com_z_m,bunny_vz_m_s,"
        "bunny_speed_m_s,bunny_angular_speed_rad_s,contact_force_n,contact_impulse_z_ns,"
        "pad_top_m,center_top_m,rim_top_m,penetration_m,alpha_mean,alpha_max,alpha_increment_min,"
        "det_fp_error,det_fe_min,det_fe_max,elastic_j,hardening_j,plastic_dissipation_j,kinetic_j,"
        "gravity_j,momentum_x_ns,momentum_y_ns,momentum_z_ns,external_impulse_x_ns,"
        "external_impulse_y_ns,external_impulse_z_ns,angular_momentum_x,angular_momentum_y,"
        "angular_momentum_z,external_angular_impulse_x,external_angular_impulse_y,external_angular_impulse_z,env_status\n";
    metrics << std::setprecision(10);
    std::vector<Vec3> positions(count), velocity(count), reaction(body_count), moment(body_count);
    std::vector<Vec3> body_velocity(body_count), body_omega(body_count);
    std::vector<Transform> bodies(body_count);
    std::vector<float> elastic(count*9u), plastic(count*9u), affine(count*9u), alpha(count), previous_alpha(count);
    std::unique_ptr<Renderer> renderer;
    if (!args.no_render) renderer = std::make_unique<Renderer>(args, spacing, count, mesh);
    std::array<double,3> external{}, total_contact{};
    std::array<double,3> external_angular{}, previous_gravity_torque{};
    for (uint32_t frame = 0u; frame <= frames; ++frame) {
        double frame_impulse = 0.0;
        uint32_t status = 0u;
        if (frame > 0u) {
            measurements.BeginInterval();
            for (uint32_t s = 0u; s < args.steps_per_frame; ++s) {
                measurements.Step();
                Download(world, nk::FieldId::MpmBodyReaction, reaction);
                Download(world, nk::FieldId::MpmBodyAngReaction, moment);
                uint32_t step_status = 0u;
                Require(world.GetData().DownloadField(nk::FieldId::EnvStatus, &step_status, sizeof(step_status)),
                        "Environment status readout failed");
                status |= step_status;
                Require(status == 0u, "Environment failure flags " + std::to_string(status));
                external[0] -= reaction[scene.support].x;
                external[1] -= reaction[scene.support].y;
                external[2] += (mass*count+args.mass)*kGravity*dt - reaction[scene.support].z;
                const auto support_moment = initial_bodies[scene.support].position.Cross(reaction[scene.support]) +
                    moment[scene.support];
                external_angular[0] -= support_moment.x;
                external_angular[1] -= support_moment.y;
                external_angular[2] -= support_moment.z;
                frame_impulse += reaction[scene.bunny].z;
                total_contact[0] += reaction[scene.bunny].x;
                total_contact[1] += reaction[scene.bunny].y;
                total_contact[2] += reaction[scene.bunny].z;
            }
        }
        Require(world.Synchronize() == phi::Status::Ok, "Simulation completion failed");
        if (frame > 0u) measurements.EndInterval();
        Download(world, nk::FieldId::ParticlePos, positions); Download(world, nk::FieldId::ParticleVel, velocity);
        Download(world, nk::FieldId::ParticleF, elastic); Download(world, nk::FieldId::ParticlePlasticF, plastic);
        Download(world, nk::FieldId::ParticleC, affine); Download(world, nk::FieldId::ParticlePlastic, alpha);
        Download(world, nk::FieldId::BodyPose, bodies); Download(world, nk::FieldId::BodyLinearVelocity, body_velocity);
        Download(world, nk::FieldId::BodyAngularVelocity, body_omega);
        const auto pose = bodies[scene.bunny], inverse = pose.Inverse();
        const auto com = pose.TransformPoint(scene.inertia.position);
        const auto rotation = pose.rotation*scene.inertia.rotation;
        const auto omega = body_omega[scene.bunny], local_omega = rotation.Conjugate().Rotate(omega);
        const auto spin = rotation.Rotate({scene.principal_moments.x*local_omega.x,
            scene.principal_moments.y*local_omega.y, scene.principal_moments.z*local_omega.z});
        const auto bv = body_velocity[scene.bunny];
        const Vec3 body_angular = com.Cross(bv*args.mass)+spin;
        std::array<double,3> momentum{args.mass*bv.x,args.mass*bv.y,args.mass*bv.z};
        std::array<double,3> angular{body_angular.x,body_angular.y,body_angular.z};
        std::array<double,3> gravity_torque{args.mass*kGravity*com.y,-args.mass*kGravity*com.x,0.0};
        double kinetic = 0.5*(args.mass*bv.LengthSq()+spin.Dot(omega));
        double potential = -args.mass*kGravity*com.z, elastic_energy = 0.0, hardening_energy = 0.0;
        double alpha_sum = 0.0, alpha_max = 0.0, increment_min = 0.0, fp_error = 0.0;
        double fe_min = 1e30, fe_max = 0.0, penetration = 0.0;
        float top = -1e30f, center_top = -1e30f, rim_top = -1e30f;
        for (uint32_t p = 0u; p < count; ++p) {
            const auto x = positions[p], v = velocity[p];
            Require(std::isfinite(x.LengthSq()) && std::isfinite(v.LengthSq()), "Nonfinite particle state");
            top = std::max(top, x.z);
            const float radius = std::sqrt(x.x*x.x+x.y*x.y);
            if (radius < 0.022f) center_top = std::max(center_top, x.z);
            if (radius > 0.065f && radius < 0.095f) rim_top = std::max(rim_top, x.z);
            const auto local = inverse.TransformPoint(x);
            if (std::abs(local.x) <= scene.half.x && std::abs(local.y) <= scene.half.y &&
                std::abs(local.z) <= scene.half.z) {
                const auto point = collision::QueryMeshSurface(surface, info, local);
                Require(point.valid, "Collision surface query failed");
                penetration = std::max(penetration, double(-point.distance));
            }
            alpha_sum += alpha[p]; alpha_max = std::max(alpha_max, double(alpha[p]));
            increment_min = std::min(increment_min, double(alpha[p]-previous_alpha[p]));
            previous_alpha[p] = alpha[p];
            fp_error = std::max(fp_error, std::abs(Determinant(plastic.data()+9u*p)-1.0));
            const double determinant = Determinant(elastic.data()+9u*p);
            fe_min = std::min(fe_min, determinant); fe_max = std::max(fe_max, determinant);
            nk::material::HenckyResponse response;
            Require(nk::material::EvaluateHenckyJ2(elastic.data()+9u*p,
                {kYoungs,kPoisson,kYield,kHardening}, response) == nk::material::ConstitutiveStatus::Ok,
                "Invalid elastic state");
            elastic_energy += volume*response.elastic_energy;
            hardening_energy += volume*0.5*kHardening*alpha[p]*alpha[p];
            const float* c = affine.data()+9u*p;
            double norm = 0.0;
            for (uint32_t k = 0u; k < 9u; ++k) norm += double(c[k])*c[k];
            const double d = args.dx*args.dx*0.25;
            kinetic += 0.5*mass*(v.LengthSq()+d*norm);
            potential -= mass*kGravity*x.z;
            gravity_torque[0] += mass*kGravity*x.y;
            gravity_torque[1] -= mass*kGravity*x.x;
            momentum[0] += mass*v.x; momentum[1] += mass*v.y; momentum[2] += mass*v.z;
            const auto orbital = x.Cross(v);
            angular[0] += mass*(orbital.x+d*(c[7]-c[5]));
            angular[1] += mass*(orbital.y+d*(c[2]-c[6]));
            angular[2] += mass*(orbital.z+d*(c[3]-c[1]));
        }
        if (frame > 0u)
            for (uint32_t k = 0u; k < 3u; ++k)
                external_angular[k] += (previous_gravity_torque[k]+gravity_torque[k])/(2.0*kSampleHz);
        previous_gravity_torque = gravity_torque;
        metrics << frame << ',' << frame/double(kSampleHz) << ',' << Bottom(mesh, pose) << ','
            << com.x << ',' << com.y << ',' << com.z << ',' << bv.z << ',' << std::sqrt(bv.LengthSq()) << ','
            << std::sqrt(omega.LengthSq()) << ',' << frame_impulse*kSampleHz << ',' << total_contact[2] << ','
            << top << ',' << center_top << ',' << rim_top << ',' << penetration << ',' << alpha_sum/count << ','
            << alpha_max << ',' << increment_min << ',' << fp_error << ',' << fe_min << ',' << fe_max << ','
            << elastic_energy << ',' << hardening_energy << ',' << volume*kYield*alpha_sum << ',' << kinetic << ','
            << potential;
        for (double value : momentum) metrics << ',' << value;
        for (double value : external) metrics << ',' << value;
        for (double value : angular) metrics << ',' << value;
        for (double value : external_angular) metrics << ',' << value;
        metrics << ',' << status << '\n';
        Write(capture, positions.data(), positions.size()); Write(capture, bodies.data(), bodies.size());
        if (renderer) renderer->Frame(frame, positions, pose);
        if (frame % 24u == 0u) {
            metrics.flush(); capture.flush();
            std::printf("t=%.3f bottom=%.5f center=%.5f alpha=%.6f max=%.5f force=%.3f penetration=%.6f\n",
                frame/double(kSampleHz), Bottom(mesh, pose), center_top, alpha_sum/count, alpha_max,
                frame_impulse*kSampleHz, penetration);
        }
    }
    std::ofstream state(args.out / "material_state.bin", std::ios::binary);
    for (const auto* values : {&elastic, &plastic, &affine, &alpha}) Write(state, values->data(), values->size());
    Write(state, velocity.data(), velocity.size());
    Require(world.Reset() == phi::Status::Ok && world.Synchronize() == phi::Status::Ok, "Reset failed");
    Download(world, nk::FieldId::ParticlePlastic, alpha); Download(world, nk::FieldId::ParticlePos, positions);
    Download(world, nk::FieldId::ParticleF, elastic); Download(world, nk::FieldId::ParticlePlasticF, plastic);
    Download(world, nk::FieldId::ParticleC, affine); Download(world, nk::FieldId::ParticleVel, velocity);
    Download(world, nk::FieldId::BodyPose, bodies);
    Download(world, nk::FieldId::BodyLinearVelocity, body_velocity);
    Download(world, nk::FieldId::BodyAngularVelocity, body_omega);
    for (uint32_t p = 0u; p < count; ++p) {
        Require(alpha[p] == 0.0f && (positions[p]-initial[p]).LengthSq() == 0.0f &&
                velocity[p].LengthSq() == 0.0f, "Particle reset mismatch");
        for (uint32_t k = 0u; k < 9u; ++k)
            Require(elastic[9u*p+k] == (k%4u == 0u ? 1.0f : 0.0f) &&
                    plastic[9u*p+k] == (k%4u == 0u ? 1.0f : 0.0f) && affine[9u*p+k] == 0.0f,
                    "Material tensor reset mismatch");
    }
    for (uint32_t b = 0u; b < body_count; ++b)
        Require((bodies[b].position-initial_bodies[b].position).LengthSq() == 0.0f &&
                bodies[b].rotation == initial_bodies[b].rotation && body_velocity[b].LengthSq() == 0.0f &&
                body_omega[b].LengthSq() == 0.0f, "Rigid reset mismatch");
    Json completion = Json::Object();
    completion.Set("frames", Json::Int(frames+1u)); completion.Set("reset_passed", Json::Bool(true));
    completion.Set("body_pose_written_after_release", Json::Bool(false));
    completion.Set("material_reset_during_capture", Json::Bool(false));
    WriteJson(args.out / "completion.json", completion);
    measurements.Write(args.perf_json, dt);
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const Args args = ParseArgs(argc, argv);
        std::filesystem::create_directories(args.out);
        phi::Device* device = phi::InitBestDevice();
        phi::Backend* backend = device ? phi::DeviceInitBackend(device, nullptr) : nullptr;
        Require(backend != nullptr, "No physics backend");
        if (args.replay.empty()) Simulate(args, device, backend);
        else Replay(args);
        phi::BackendFree(backend);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "elastoplastic_bunny: %s\n", error.what());
        return 1;
    }
    return 0;
}
