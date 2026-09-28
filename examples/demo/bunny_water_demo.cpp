// A rigid Stanford bunny drops into a tank of weakly compressible MLS-MPM water.
// Captured particle and rigid states drive the headless renderer in the Nuka lab scene.

#include <algorithm>
#include <array>
#include <chrono>
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

// The tank stands on the lab deck (z = 0); its walls and floor are static collision boxes.
constexpr float kInnerX = 0.18f, kInnerY = 0.14f, kWall = 0.01f, kRim = 0.34f, kDepth = 0.18f;
constexpr float kWaterDensity = 1000.0f, kTaitGamma = 7.0f, kViscosity = 0.001f;
constexpr float kGravity = -9.81f;

struct Args {
    std::filesystem::path out = "out/bunny_water";
    std::filesystem::path replay;
    std::filesystem::path environment = std::filesystem::path(NUKA_SOURCE_DIR) /
        "examples/assets/nuka_lab/water.nks";
    std::filesystem::path mesh = std::filesystem::path(NUKA_SOURCE_DIR) /
        ".nuka-assets/generated/bunny_solid_120mm.obj";
    float dx = 0.006f, bunny_density = 1200.0f, drop_height = 0.15f, duration = 3.5f, bulk = 200000.0f;
    float sample_hz = 60.0f;
    uint32_t steps_per_frame = 240u, substeps = 1u, width = 1920u, height = 1080u, samples = 128u;
    bool implicit_stress = false, no_render = false;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--no-render") { args.no_render = true; continue; }
        if (key == "--implicit-stress") { args.implicit_stress = true; continue; }
        Require(i + 1 < argc, "Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--out-dir") args.out = value;
        else if (key == "--replay") args.replay = value;
        else if (key == "--environment") args.environment = value;
        else if (key == "--mesh") args.mesh = value;
        else if (key == "--dx") args.dx = std::stof(value);
        else if (key == "--bunny-density") args.bunny_density = std::stof(value);
        else if (key == "--drop-height") args.drop_height = std::stof(value);
        else if (key == "--bulk-modulus") args.bulk = std::stof(value);
        else if (key == "--duration") args.duration = std::stof(value);
        else if (key == "--sample-hz") args.sample_hz = std::stof(value);
        else if (key == "--steps-per-frame") args.steps_per_frame = std::stoul(value);
        else if (key == "--substeps") args.substeps = std::stoul(value);
        else if (key == "--width") args.width = std::stoul(value);
        else if (key == "--height") args.height = std::stoul(value);
        else if (key == "--samples") args.samples = std::stoul(value);
        else throw std::runtime_error("Unknown argument " + key);
    }
    for (float value : {args.dx, args.bunny_density, args.drop_height, args.duration, args.sample_hz, args.bulk})
        Require(std::isfinite(value) && value > 0.0f, "Physical inputs must be finite and positive");
    Require(args.steps_per_frame && args.substeps && args.width && args.height && args.samples,
            "Counts must be positive");
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

struct Slab { Vec3 center, half; };

// Floor first, then the four walls; the rendered tank uses the same boxes.
std::vector<Slab> TankSlabs() {
    const float z = 0.5f * kRim, x = kInnerX + 0.5f * kWall, y = kInnerY + 0.5f * kWall;
    return {{{0.0f, 0.0f, -0.02f}, {0.40f, 0.36f, 0.02f}},
            {{-x, 0.0f, z}, {0.5f * kWall, kInnerY + kWall, z}},
            {{x, 0.0f, z}, {0.5f * kWall, kInnerY + kWall, z}},
            {{0.0f, -y, z}, {kInnerX, 0.5f * kWall, z}},
            {{0.0f, y, z}, {kInnerX, 0.5f * kWall, z}}};
}

struct Scene {
    nk::Model model;
    uint32_t bunny = 0u;
    float mass = 0.0f;
    Transform inertia;
    Vec3 principal_moments;
    Json asset;
};

Scene BuildModel(const Args& args, const render::MeshGeometry& mesh) {
    Scene scene;
    auto mass_path = args.mesh; mass_path.replace_extension(".json");
    scene.asset = ReadJson(mass_path);
    scene.mass = args.bunny_density * scene.asset.At("volume_m3").AsFloat();
    scene.inertia.position = Vector(scene.asset.At("center_of_mass_m"));
    const auto& q = scene.asset.At("principal_rotation_wxyz").Elements();
    Require(q.size() == 4u, "Missing principal frame rotation");
    scene.inertia.rotation = Quat{q[0].AsFloat(), q[1].AsFloat(), q[2].AsFloat(), q[3].AsFloat()};
    scene.principal_moments = Vector(scene.asset.At("principal_inertia_per_kg")) * scene.mass;
    auto& model = scene.model;
    auto& cap = model.capacities;
    cap.env_count = 1u;
    nk::Model::BodyInit body;
    body.inertial_frame = scene.inertia;
    body.inv_mass = 1.0f / scene.mass;
    body.inv_inertia = {1.0f / scene.principal_moments.x, 1.0f / scene.principal_moments.y,
                       1.0f / scene.principal_moments.z};
    body.pose.rotation = Quat::FromAxisAngle({0, 0, 1}, 0.18f);
    body.pose.position.z = kDepth + args.drop_height - Bottom(mesh, body.pose);
    model.body_init.push_back(body);
    auto surface = nuka::import::cooker::CookMeshSurfaceCached(mesh.positions.data(), mesh.VertexCount(),
        mesh.indices.data(), static_cast<uint32_t>(mesh.indices.size() / 3u), false);
    Require((surface.info.flags & collision::kMeshSurfaceClosed) != 0u,
            "The bunny must have a closed collision surface");
    model.hull_verts = mesh.positions;
    model.mesh_triangles = mesh.indices;
    model.mesh_bvh_nodes = std::move(surface.nodes);
    model.mesh_surface_info.push_back(surface.info);
    Vec3 half;
    float radius = 0.0f;
    for (uint32_t i = 0u; i < mesh.VertexCount(); ++i) {
        const auto p = Vertex(mesh, i);
        half = {std::max(half.x, std::abs(p.x)), std::max(half.y, std::abs(p.y)), std::max(half.z, std::abs(p.z))};
        radius = std::max(radius, std::sqrt(p.LengthSq()));
    }
    nk::Model::PairDrivenShape shape;
    shape.kind = collision::kShapeSdfMesh;
    shape.body_id = 0;
    shape.params[0] = radius;
    shape.params[1] = half.x; shape.params[2] = half.y; shape.params[3] = half.z;
    shape.hull_vert_count = mesh.VertexCount();
    model.shape_table_rows.push_back(shape);
    model.samp_points = mesh.positions;
    model.samp_ranges = {0u, mesh.VertexCount()};
    for (const auto& slab : TankSlabs()) {
        body = nk::Model::BodyInit{};
        body.pose.position = slab.center;
        shape = nk::Model::PairDrivenShape{};
        shape.kind = collision::kShapeBox;
        shape.body_id = static_cast<int32_t>(model.body_init.size());
        shape.params[0] = slab.half.x; shape.params[1] = slab.half.y; shape.params[2] = slab.half.z;
        model.body_init.push_back(body);
        model.shape_table_rows.push_back(shape);
    }
    model.mesh_surface_info.resize(model.body_init.size());
    model.samp_ranges.resize(model.body_init.size() * 2u);
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = cap.bodies_per_env;
    cap.max_hull_verts = mesh.VertexCount();
    cap.max_mesh_triangles = static_cast<uint32_t>(mesh.indices.size() / 3u);
    cap.max_mesh_bvh_nodes = static_cast<uint32_t>(model.mesh_bvh_nodes.size());
    cap.max_samp_points = mesh.VertexCount();
    cap.max_contacts_per_env = 32u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    model.contact_family = nk::ContactFamily::PairDriven;
    const float spacing = args.dx * 0.5f;
    const uint32_t nx = std::lround(2.0f * kInnerX / spacing), ny = std::lround(2.0f * kInnerY / spacing),
                   nz = std::lround(kDepth / spacing);
    cook::MpmCookInput water;
    for (uint32_t z = 0u; z < nz; ++z)
        for (uint32_t y = 0u; y < ny; ++y)
            for (uint32_t x = 0u; x < nx; ++x)
                water.positions.push_back({(x + 0.5f) * spacing - kInnerX,
                    (y + 0.5f) * spacing - kInnerY, (z + 0.5f) * spacing});
    const size_t count = water.positions.size();
    const float volume = spacing * spacing * spacing;
    water.velocities.assign(count, Vec3::Zero());
    water.inv_mass.assign(count, 1.0f / (kWaterDensity * volume));
    water.vol0.assign(count, volume);
    water.material.density = kWaterDensity;
    water.material.model_kind = 3.0f;
    water.material.bulk_modulus = args.bulk;
    water.material.tait_gamma = kTaitGamma;
    water.material.viscosity = kViscosity;
    water.dx = args.dx;
    water.substeps = args.substeps;
    // Contacts form only where water meets the tank and the bunny, so the pool scales with the
    // material points rather than with the spray-covering grid.
    water.contact_capacity = static_cast<uint32_t>(count);
    water.implicit_stress = args.implicit_stress;
    // The grid covers the tank, the deck where spray over the rim lands, and the jet above it.
    const Vec3 lo{-kInnerX - kWall - 0.5f, -kInnerY - kWall - 0.5f, -3.0f * args.dx};
    const Vec3 hi{-lo.x, -lo.y, kRim + 0.35f};
    water.grid_origin = lo;
    water.grid_dims[0] = static_cast<uint32_t>(std::ceil((hi.x - lo.x) / args.dx)) + 1u;
    water.grid_dims[1] = static_cast<uint32_t>(std::ceil((hi.y - lo.y) / args.dx)) + 1u;
    water.grid_dims[2] = static_cast<uint32_t>(std::ceil((hi.z - lo.z) / args.dx)) + 1u;
    water.floor_d = 0.0f;
    // Spray that lands on the deck grips it as a wetting film instead of skating off the grid.
    water.floor_friction = 0.5f;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(model, 1u, soft, water);
    model.particles.mpm_body_friction = 0.0f;
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
        // A ray through the glass tank crosses both wall faces and the water surface.
        scene_.options.beauty_transmit_bounces = 6u;
        render::ApplySceneCamera(scene_.options, asset, "overview");
        begin_options_ = scene_.options;
        end_options_ = scene_.options;
        render::ApplySceneCamera(end_options_, asset, "close");
        render::AddStudioDensitySurface(scene_, registry, render::kNoId, spacing, 0u, particles);
        // Anisotropic kernels flatten a calm surface that isotropic kernels leave dimpled per particle.
        scene_.density_surfaces.back().params.anisotropic = true;
        scene_.density_surfaces.back().material_id = render::SceneAssetMaterial(asset, binding, "water");
        render::RenderInstance instance;
        instance.mesh_id = scene_.world.meshes.InternPrimitive("bunny", [&] { return mesh; });
        instance.render_material_id = render::SceneAssetMaterial(asset, binding, "ceramic");
        bunny_instance_ = scene_.world.instances.size();
        scene_.world.instances.push_back(instance);
        const auto slabs = TankSlabs();
        for (size_t i = 1u; i < slabs.size(); ++i) {
            instance.mesh_id = scene_.world.meshes.InternPrimitive("tank_wall" + std::to_string(i),
                [&] { return BoxMesh(slabs[i].half); });
            instance.render_material_id = render::SceneAssetMaterial(asset, binding, "tank_glass");
            instance.world_xform = Transform::Identity();
            instance.world_xform.position = slabs[i].center;
            scene_.world.instances.push_back(instance);
        }
        instance.mesh_id = scene_.world.meshes.InternPrimitive("tank_floor",
            [] { return BoxMesh({kInnerX, kInnerY, 0.0005f}); });
        instance.render_material_id = render::SceneAssetMaterial(asset, binding, "warm_shell");
        instance.world_xform = Transform::Identity();
        instance.world_xform.position.z = 0.0005f;
        scene_.world.instances.push_back(instance);
        renderer_ = std::make_unique<render::StudioRtRenderer>();
        Require(renderer_->ok(), "No ray tracing backend");
        renderer_->SetBeauty(true, args.samples);
        std::filesystem::create_directories(args.out / "frames");
        Json config = Json::Object();
        config.Set("width", Json::Int(args.width)); config.Set("height", Json::Int(args.height));
        config.Set("samples", Json::Int(args.samples));
        config.Set("environment_asset", Json::Str(args.environment.string()));
        config.Set("camera_fov_degrees", Json::Float(scene_.options.camera_fov_degrees));
        config.Set("state_interpolation", Json::Bool(false));
        WriteJson(args.out / "render_config.json", config);
    }

    void Frame(uint32_t frame, float time, const std::vector<Vec3>& positions, const Transform& bunny) {
        float u = std::clamp((time - 0.8f) / 1.6f, 0.0f, 1.0f);
        u = u*u*u*(10.0f+u*(-15.0f+6.0f*u));
        scene_.options.camera_eye = begin_options_.camera_eye * (1.0f-u) + end_options_.camera_eye * u;
        scene_.options.camera_target = begin_options_.camera_target * (1.0f-u) + end_options_.camera_target * u;
        scene_.options.camera_up = (begin_options_.camera_up * (1.0f-u) + end_options_.camera_up * u).Normalized();
        scene_.world.instances[bunny_instance_].world_xform = bunny;
        render::PublishStudioScene(scene_, {}, positions);
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
    Require(header[0] == 0x4E554B41u && header[1] == 3u, "Unsupported water capture");
    std::vector<Vec3> positions(header[2]);
    std::vector<Transform> bodies(header[3]);
    const float sample_hz = ReadJson(args.replay / "config.json").At("sample_hz").AsFloat();
    Renderer renderer(args, spacing, header[2], LoadMesh(args.replay / "bunny.obj"));
    for (uint32_t frame = 0u; frame < header[4]; ++frame) {
        Read(source, positions.data(), positions.size());
        Read(source, bodies.data(), bodies.size());
        renderer.Frame(frame, frame / sample_hz, positions, bodies[0]);
        if (frame % 15u == 0u) std::printf("render frame %u/%u\n", frame, header[4]);
    }
}

void Simulate(const Args& args, phi::Device* device, phi::Backend* backend) {
    const auto mesh = LoadMesh(args.mesh);
    auto scene = BuildModel(args, mesh);
    const uint32_t count = scene.model.capacities.particles_per_env;
    const uint32_t body_count = scene.model.capacities.bodies_per_env;
    const uint32_t frames = std::lround(args.duration * args.sample_hz);
    const float spacing = args.dx * 0.5f;
    const double dt = 1.0 / (args.sample_hz * args.steps_per_frame);
    const auto initial = scene.model.particles.initial_pos;
    const auto initial_pose = scene.model.body_init[0].pose;
    nk::Pipeline::SolverConfig solver;
    solver.dt = float(dt);
    nk::World world(std::move(scene.model), 1u, device, backend, solver);
    Require(world.Ready(), "World creation: " + world.CreationError());
    Require(world.SetExecutionMode(nk::World::ExecutionMode::Graph) == phi::Status::Ok, "Graph mode failed");
    std::filesystem::copy_file(args.mesh, args.out / "bunny.obj", std::filesystem::copy_options::overwrite_existing);
    Json config = Json::Object();
    config.Set("dx", Json::Float(args.dx)); config.Set("spacing", Json::Float(spacing));
    config.Set("dt", Json::Float(dt)); config.Set("sample_hz", Json::Float(args.sample_hz));
    config.Set("steps_per_frame", Json::Int(args.steps_per_frame)); config.Set("particles", Json::Int(count));
    config.Set("frames", Json::Int(frames + 1u)); config.Set("bunny_mass_kg", Json::Float(scene.mass));
    config.Set("bunny_density", Json::Float(args.bunny_density));
    config.Set("drop_height_m", Json::Float(args.drop_height)); config.Set("water_depth_m", Json::Float(kDepth));
    config.Set("water_density", Json::Float(kWaterDensity)); config.Set("tait_bulk_pa", Json::Float(args.bulk));
    config.Set("tait_gamma", Json::Float(kTaitGamma)); config.Set("viscosity_pa_s", Json::Float(kViscosity));
    config.Set("stress", Json::Str(args.implicit_stress ? "implicit rows" : "explicit"));
    config.Set("substeps", Json::Int(args.substeps)); config.Set("gravity_z", Json::Float(kGravity));
    config.Set("source_asset", scene.asset);
    WriteJson(args.out / "config.json", config);
    std::ofstream capture(args.out / "positions.bin", std::ios::binary);
    const uint32_t header[] = {0x4E554B41u, 3u, count, body_count, frames + 1u};
    Write(capture, header, 5u); Write(capture, &spacing, 1u);
    std::ofstream metrics(args.out / "metrics.csv");
    metrics << "frame,time_s,bunny_bottom_m,bunny_com_z_m,bunny_vz_m_s,free_fall_vz_m_s,bunny_speed_m_s,"
        "water_level_m,water_max_z_m,j_mean,j_min,j_max,env_status\n";
    metrics << std::setprecision(10);
    std::vector<Vec3> positions(count), velocity(count), body_velocity(body_count), body_omega(body_count);
    std::vector<Transform> bodies(body_count);
    std::vector<float> elastic(count * 9u), heights(count);
    std::unique_ptr<Renderer> renderer;
    if (!args.no_render) renderer = std::make_unique<Renderer>(args, spacing, count, mesh);
    const float rest_level = kDepth;
    double free_fall_error = 0.0, j_error = 0.0, peak_speed = 0.0, entry_time = -1.0, settle_speed = 0.0;
    float spill_max = -1e30f, final_bottom = 0.0f;
    uint32_t status = 0u;
    auto report_clock = std::chrono::steady_clock::now();
    for (uint32_t frame = 0u; frame <= frames; ++frame) {
        if (frame > 0u)
            for (uint32_t s = 0u; s < args.steps_per_frame; ++s) {
                const auto result = world.StepConfigured();
                Require(result == phi::Status::Ok, std::string("Step failed: ") + world.LastExecutionError().message);
            }
        Require(world.Synchronize() == phi::Status::Ok, "Simulation completion failed");
        uint32_t step_status = 0u;
        Require(world.GetData().DownloadField(nk::FieldId::EnvStatus, &step_status, sizeof(step_status)),
                "Environment status readout failed");
        status |= step_status;
        Download(world, nk::FieldId::ParticlePos, positions); Download(world, nk::FieldId::ParticleVel, velocity);
        Download(world, nk::FieldId::ParticleF, elastic); Download(world, nk::FieldId::BodyPose, bodies);
        Download(world, nk::FieldId::BodyLinearVelocity, body_velocity);
        Download(world, nk::FieldId::BodyAngularVelocity, body_omega);
        const auto pose = bodies[0];
        const auto com = pose.TransformPoint(scene.inertia.position);
        const auto bv = body_velocity[0];
        const float bottom = Bottom(mesh, pose);
        const double time = frame / double(args.sample_hz);
        double j_sum = 0.0, j_min = 1e30, j_max = 0.0;
        float top = -1e30f;
        for (uint32_t p = 0u; p < count; ++p) {
            Require(std::isfinite(positions[p].LengthSq()) && std::isfinite(velocity[p].LengthSq()),
                    "Nonfinite particle state");
            const double j = Determinant(elastic.data() + 9u * p);
            j_sum += j; j_min = std::min(j_min, j); j_max = std::max(j_max, j);
            top = std::max(top, positions[p].z);
            heights[p] = positions[p].z;
            if (std::abs(positions[p].x) > kInnerX + kWall || std::abs(positions[p].y) > kInnerY + kWall)
                spill_max = std::max(spill_max, positions[p].z);
        }
        std::nth_element(heights.begin(), heights.begin() + count * 97u / 100u, heights.end());
        const float level = heights[count * 97u / 100u];
        Require(std::isfinite(com.LengthSq()) && std::isfinite(bv.LengthSq()), "Nonfinite rigid state");
        const double free_fall = kGravity * time;
        const double speed = std::sqrt(bv.LengthSq());
        j_error = std::max(j_error, std::abs(j_sum / count - 1.0));
        if (entry_time < 0.0 && bottom > rest_level + args.dx)
            free_fall_error = std::max(free_fall_error, std::abs(bv.z - free_fall));
        else if (entry_time < 0.0) entry_time = time;
        peak_speed = std::max(peak_speed, speed);
        settle_speed = speed; final_bottom = bottom;
        metrics << frame << ',' << time << ',' << bottom << ',' << com.z << ',' << bv.z << ',' << free_fall << ','
            << speed << ',' << level << ',' << top << ',' << j_sum / count << ',' << j_min << ',' << j_max << ','
            << step_status << '\n';
        Write(capture, positions.data(), positions.size()); Write(capture, bodies.data(), bodies.size());
        if (renderer) renderer->Frame(frame, float(time), positions, pose);
        if (frame % 3u == 0u) {
            metrics.flush(); capture.flush();
            const auto now = std::chrono::steady_clock::now();
            const double step_ms = frame ? std::chrono::duration<double, std::milli>(now - report_clock).count() /
                (3.0 * args.steps_per_frame) : 0.0;
            report_clock = now;
            std::printf("t=%.3f bottom=%.4f vz=%.3f free=%.3f level=%.4f top=%.4f J=%.4f [%.3f,%.3f] status=%u "
                "step=%.1f ms\n", time, bottom, bv.z, free_fall, level, top, j_sum / count, j_min, j_max, status,
                step_ms);
            std::fflush(stdout);
        }
    }
    Require(world.Reset() == phi::Status::Ok && world.Synchronize() == phi::Status::Ok, "Reset failed");
    Download(world, nk::FieldId::ParticlePos, positions); Download(world, nk::FieldId::BodyPose, bodies);
    bool reset = (bodies[0].position - initial_pose.position).LengthSq() == 0.0f;
    for (uint32_t p = 0u; p < count; ++p) reset = reset && (positions[p] - initial[p]).LengthSq() == 0.0f;
    // Free fall is exact for the symplectic step; the dense bunny must sink and come to rest.
    const bool falls = free_fall_error < 1e-3, sinks = final_bottom < 0.01f && settle_speed < 0.05;
    const bool compressible = j_error < 0.03, clean = status == 0u;
    Json completion = Json::Object();
    completion.Set("frames", Json::Int(frames + 1u)); completion.Set("reset_passed", Json::Bool(reset));
    completion.Set("free_fall_vz_error_m_s", Json::Float(free_fall_error));
    completion.Set("water_entry_s", Json::Float(entry_time));
    completion.Set("peak_bunny_speed_m_s", Json::Float(peak_speed));
    completion.Set("final_bunny_bottom_m", Json::Float(final_bottom));
    completion.Set("final_bunny_speed_m_s", Json::Float(settle_speed));
    completion.Set("max_volume_ratio_error", Json::Float(j_error));
    completion.Set("max_spill_height_m", Json::Float(spill_max));
    completion.Set("env_status_union", Json::Int(status));
    completion.Set("physics_checks_passed", Json::Bool(reset && falls && sinks && compressible && clean));
    WriteJson(args.out / "completion.json", completion);
    std::printf("free_fall_error=%.2e entry=%.3f peak=%.3f final_bottom=%.4f final_speed=%.4f "
                "J_error=%.4f status=%u reset=%d\n", free_fall_error, entry_time, peak_speed,
                final_bottom, settle_speed, j_error, status, int(reset));
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
        std::fprintf(stderr, "bunny_water: %s\n", error.what());
        return 1;
    }
    return 0;
}
