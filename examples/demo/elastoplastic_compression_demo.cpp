#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision/primitive_surface.hpp"
#include "nk/material/hencky_j2.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "render/studio_beauty.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/scene_map.hpp"
#include "tools/perf/cuda_step_measurements.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace render = nuka::render;
namespace cook = nuka::scene::cook;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kWidth = 0.08f, kHeight = 0.10f, kBase = 0.04f;
constexpr float kDensity = 1000.0f, kGravity = -9.81f;
constexpr float kYoungs = 50000.0f, kPoisson = 0.30f;
constexpr float kYield = 12000.0f, kHardening = 3000.0f;
constexpr float kOpenGap = 0.13f, kMildGap = 0.096f, kStrongGap = 0.045f;
constexpr float kContactBand = 0.0001f;
constexpr float kSampleHz = 120.0f;
constexpr uint32_t kFrames = 576u;
constexpr Vec3 kPlateHalf{0.085f, 0.070f, 0.009f};

struct Args {
    std::filesystem::path out = "out/elastoplastic_compression";
    std::filesystem::path replay;
    std::filesystem::path perf_json;
    float dx = 0.005f;
    uint32_t steps_per_frame = 64u;
    uint32_t width = 1600u, height = 1000u, samples = 64u;
    uint32_t render_stride = 1u;
    bool no_render = false;
};

void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
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
        else if (key == "--perf-json") args.perf_json = value;
        else if (key == "--dx") args.dx = std::stof(value);
        else if (key == "--steps-per-frame") args.steps_per_frame = std::stoul(value);
        else if (key == "--width") args.width = std::stoul(value);
        else if (key == "--height") args.height = std::stoul(value);
        else if (key == "--samples") args.samples = std::stoul(value);
        else if (key == "--render-stride") args.render_stride = std::stoul(value);
        else throw std::runtime_error("Unknown argument " + key);
    }
    Require(std::isfinite(args.dx) && args.dx > 0.0f && args.dx <= 0.01f,
            "dx must be positive and no greater than 0.01 m");
    Require(args.steps_per_frame && args.width && args.height && args.samples &&
            args.render_stride, "Counts must be positive");
    Require(args.perf_json.empty() || args.replay.empty(), "Step timing requires live simulation");
    return args;
}

double Smooth(double t, double begin, double end, double a, double b) {
    const double u = std::clamp((t - begin) / (end - begin), 0.0, 1.0);
    return a + (b - a) * u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

double Gap(double t) {
    if (t < 0.30) return kOpenGap;
    if (t < 0.90) return Smooth(t, 0.30, 0.90, kOpenGap, kMildGap);
    if (t < 1.10) return kMildGap;
    if (t < 1.50) return Smooth(t, 1.10, 1.50, kMildGap, kOpenGap);
    if (t < 2.10) return kOpenGap;
    if (t < 2.90) return Smooth(t, 2.10, 2.90, kOpenGap, kStrongGap);
    if (t < 3.10) return kStrongGap;
    if (t < 3.70) return Smooth(t, 3.10, 3.70, kStrongGap, kOpenGap);
    return kOpenGap;
}

const char* Stage(double t) {
    if (t < 0.30) return "settle";
    if (t < 0.90) return "elastic_load";
    if (t < 1.10) return "elastic_hold";
    if (t < 1.50) return "elastic_unload";
    if (t < 2.10) return "elastic_recovery";
    if (t < 2.90) return "plastic_load";
    if (t < 3.10) return "plastic_hold";
    if (t < 3.70) return "plastic_unload";
    return "plastic_recovery";
}

struct SceneModel {
    nk::Model model;
    std::vector<uint32_t> plate_bodies;
};

SceneModel BuildModel(const Args& args) {
    SceneModel scene;
    auto& model = scene.model;
    model.capacities.env_count = 1u;
    for (float z : {kBase - kPlateHalf.z, kBase + kOpenGap + kPlateHalf.z}) {
        const uint32_t row = static_cast<uint32_t>(model.body_init.size());
        scene.plate_bodies.push_back(row);
        nk::Model::BodyInit body;
        body.pose.position = {0.0f, 0.0f, z};
        // Zero inverse mass with an owned shape is a prescribed rigid boundary.
        body.inv_mass = 0.0f;
        body.inv_inertia = Vec3::Zero();
        model.body_init.push_back(body);
        nk::Model::PairDrivenShape shape;
        shape.kind = nuka::collision::kShapeBox;
        shape.params[0] = kPlateHalf.x;
        shape.params[1] = kPlateHalf.y;
        shape.params[2] = kPlateHalf.z;
        shape.body_id = static_cast<int32_t>(row);
        model.shape_table_rows.push_back(shape);
    }
    auto& cap = model.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = cap.bodies_per_env;
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    model.contact_family = nk::ContactFamily::PairDriven;

    cook::MpmCookInput material;
    const float spacing = args.dx * 0.5f;
    const uint32_t nx = std::lround(kWidth / spacing), nz = std::lround(kHeight / spacing);
    Require(std::fabs(nx * spacing - kWidth) < 1.0e-6f &&
            std::fabs(nz * spacing - kHeight) < 1.0e-6f,
            "Particle spacing must divide the specimen dimensions");
    for (uint32_t z = 0u; z < nz; ++z)
        for (uint32_t y = 0u; y < nx; ++y)
            for (uint32_t x = 0u; x < nx; ++x)
                material.positions.push_back({(x + 0.5f) * spacing - kWidth * 0.5f,
                    (y + 0.5f) * spacing - kWidth * 0.5f, kBase + (z + 0.5f) * spacing});
    const size_t count = material.positions.size();
    const float volume = spacing * spacing * spacing;
    material.velocities.assign(count, Vec3::Zero());
    material.inv_mass.assign(count, 1.0f / (kDensity * volume));
    material.vol0.assign(count, volume);
    material.material.youngs = kYoungs;
    material.material.poisson = kPoisson;
    material.material.density = kDensity;
    material.material.model_kind = nk::MpmMaterial::kHenckyJ2;
    material.material.yield_stress = kYield;
    material.material.hardening_modulus = kHardening;
    material.dx = args.dx;
    material.substeps = 1u;
    material.grid_origin = {-0.16f, -0.16f, -0.02f};
    material.grid_dims[0] = static_cast<uint32_t>(std::ceil(0.32f / args.dx)) + 1u;
    material.grid_dims[1] = material.grid_dims[0];
    material.grid_dims[2] = static_cast<uint32_t>(std::ceil(0.26f / args.dx)) + 1u;
    material.floor_d = -1.0f;
    material.floor_friction = 0.0f;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(model, 1u, soft, material);
    model.particles.mpm_body_friction = 0.0f;
    model.particles.mpm_body_band = kContactBand;
    return scene;
}

template<class T> void Write(std::ostream& file, const T* data, size_t count) {
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    Require(static_cast<bool>(file), "Capture write failed");
}

template<class T> void Read(std::istream& file, T* data, size_t count) {
    file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    Require(static_cast<bool>(file), "Capture read failed");
}

template<class T> void Download(nk::World& world, nk::FieldId field, std::vector<T>& values) {
    Require(world.GetData().DownloadField(field, values.data(), values.size() * sizeof(T)),
            "World field read failed");
}

double Determinant(const float* f) {
    return double(f[0]) * (double(f[4]) * f[8] - double(f[5]) * f[7]) -
        double(f[1]) * (double(f[3]) * f[8] - double(f[5]) * f[6]) +
        double(f[2]) * (double(f[3]) * f[7] - double(f[4]) * f[6]);
}

render::MeshGeometry BoxMesh(Vec3 half, float radius = 0.0f) {
    render::MeshGeometry mesh;
    if (radius > 0.0f) {
        const float extents[] = {half.x, half.y, half.z};
        const uint32_t divisions = 12u;
        for (uint32_t axis=0u; axis<3u; ++axis) {
            const uint32_t u=(axis+1u)%3u, v=(axis+2u)%3u;
            auto coordinate = [radius, divisions](uint32_t i, float extent) {
                const uint32_t middle=divisions/2u;
                if (i<middle) return -extent+radius*float(i)/float(middle-1u);
                return extent-radius+radius*float(i-middle)/float(middle-1u);
            };
            for (float sign : {-1.0f, 1.0f}) {
                const uint32_t first=mesh.VertexCount();
                for (uint32_t j=0u; j<divisions; ++j)
                    for (uint32_t i=0u; i<divisions; ++i) {
                        float point[3]={}, core[3]={};
                        point[axis]=sign*extents[axis];
                        point[u]=coordinate(i,extents[u]);
                        point[v]=coordinate(j,extents[v]);
                        for (uint32_t k=0u; k<3u; ++k)
                            core[k]=std::clamp(point[k],-extents[k]+radius,extents[k]-radius);
                        Vec3 normal{point[0]-core[0],point[1]-core[1],point[2]-core[2]};
                        normal=normal*(1.0f/std::sqrt(normal.LengthSq()));
                        mesh.positions.insert(mesh.positions.end(), {core[0]+radius*normal.x,
                            core[1]+radius*normal.y,core[2]+radius*normal.z});
                        mesh.normals.insert(mesh.normals.end(), {normal.x,normal.y,normal.z});
                    }
                for (uint32_t j=0u; j+1u<divisions; ++j)
                    for (uint32_t i=0u; i+1u<divisions; ++i) {
                        const uint32_t a=first+j*divisions+i, b=a+1u, c=a+divisions+1u, d=a+divisions;
                        if (sign>0.0f) mesh.indices.insert(mesh.indices.end(), {a,b,c,a,c,d});
                        else mesh.indices.insert(mesh.indices.end(), {a,c,b,a,d,c});
                    }
            }
        }
        return mesh;
    }
    const std::array<Vec3, 8> points{{{-half.x,-half.y,-half.z}, {half.x,-half.y,-half.z},
        {half.x,half.y,-half.z}, {-half.x,half.y,-half.z}, {-half.x,-half.y,half.z},
        {half.x,-half.y,half.z}, {half.x,half.y,half.z}, {-half.x,half.y,half.z}}};
    constexpr uint32_t faces[6][4] = {{0,3,2,1}, {4,5,6,7}, {0,1,5,4},
        {1,2,6,5}, {2,3,7,6}, {3,0,4,7}};
    for (const auto& face : faces) {
        const Vec3 a = points[face[0]], b = points[face[1]], c = points[face[2]];
        Vec3 normal = (b - a).Cross(c - a);
        normal = normal * (1.0f / std::sqrt(normal.LengthSq()));
        const uint32_t first = mesh.VertexCount();
        for (uint32_t vertex : face) {
            const auto p = points[vertex];
            mesh.positions.insert(mesh.positions.end(), {p.x, p.y, p.z});
            mesh.normals.insert(mesh.normals.end(), {normal.x, normal.y, normal.z});
        }
        mesh.indices.insert(mesh.indices.end(), {first, first+1, first+2, first, first+2, first+3});
    }
    return mesh;
}

class Renderer {
public:
    Renderer(const Args& args, float spacing, uint32_t particles, uint32_t bodies) : args_(args) {
        nuka::scene::Registry registry;
        nuka::scene::SceneMap map;
        scene_ = render::BuildStudioScene(registry, map,
            std::vector<nuka::runtime::soft::SurfaceTopology>{}, args.width, args.height);
        for (const auto& instance : scene_.world.instances) {
            auto floor = scene_.world.meshes.Geometry(instance.mesh_id);
            for (size_t i = 0u; i < floor.positions.size(); i += 3u) {
                floor.positions[i] *= 4.0f;
                floor.positions[i+1u] *= 4.0f;
            }
            scene_.world.meshes.ReplaceGeometry(instance.mesh_id, std::move(floor));
        }
        render::AddStudioDensitySurface(scene_, registry, render::kNoId, spacing, 0u, particles);
        scene_.density_surfaces.back().params.h = 3.0f * spacing;
        auto& sample = scene_.world.materials[scene_.density_surfaces.back().material_id];
        sample.base_color[0] = 0.64f; sample.base_color[1] = 0.19f; sample.base_color[2] = 0.055f;
        sample.roughness = 0.32f;
        for (const auto& instance : scene_.world.instances) {
            auto& material=scene_.world.materials[instance.render_material_id];
            material.base_color[0]=0.025f; material.base_color[1]=0.033f; material.base_color[2]=0.045f;
            material.roughness=0.72f;
        }
        nuka::scene::RenderMaterial plate;
        plate.base_color[0] = 0.45f; plate.base_color[1] = 0.50f; plate.base_color[2] = 0.56f;
        plate.metallic = 0.90f; plate.roughness = 0.24f;
        const uint32_t plate_material = static_cast<uint32_t>(scene_.world.materials.size());
        scene_.world.materials.push_back(plate);
        const uint32_t mesh = scene_.world.meshes.InternPrimitive("compression_platen",
            [] { return BoxMesh(kPlateHalf,0.0015f); });
        for (uint32_t b = 0u; b < bodies; ++b) {
            render::RenderInstance instance;
            instance.mesh_id = mesh;
            instance.render_material_id = plate_material;
            plate_instances_.push_back(scene_.world.instances.size());
            scene_.world.instances.push_back(instance);
        }
        render::RenderInstance pedestal;
        pedestal.mesh_id=scene_.world.meshes.InternPrimitive("press_pedestal", [] {
            return BoxMesh({0.070f,0.055f,(kBase-2.0f*kPlateHalf.z)*0.5f},0.002f);
        });
        pedestal.render_material_id=plate_material;
        pedestal.world_xform.position.z=(kBase-2.0f*kPlateHalf.z)*0.5f;
        scene_.world.instances.push_back(pedestal);
        render::RenderInstance piston;
        piston.mesh_id=scene_.world.meshes.InternPrimitive("press_piston", [] {
            return BoxMesh({0.012f,0.012f,0.12f},0.003f);
        });
        piston.render_material_id=plate_material;
        piston_instance_=scene_.world.instances.size();
        scene_.world.instances.push_back(piston);
        auto& options = scene_.options;
        options.camera_eye = {0.20f, -0.48f, 0.245f};
        options.camera_target = {0.0f, 0.0f, 0.09f};
        options.camera_fov_degrees = 30.0f;
        options.sun_direction[0] = 0.25f;
        options.sun_direction[1] = -0.80f;
        options.sun_direction[2] = 0.22f;
        options.beauty_sky_fill = 0.75f;
        options.beauty_specular_env = true;
        options.sky_top[0]=0.20f; options.sky_top[1]=0.24f; options.sky_top[2]=0.31f;
        options.sky_bottom[0]=0.36f; options.sky_bottom[1]=0.40f; options.sky_bottom[2]=0.48f;
        options.ground_color[0]=0.04f; options.ground_color[1]=0.045f; options.ground_color[2]=0.055f;
        options.beauty_grade=0.20f;
        renderer_ = std::make_unique<render::StudioRtRenderer>();
        Require(renderer_->ok(), "No ray tracing backend");
        renderer_->SetBeauty(true, args.samples);
        std::filesystem::create_directories(args.out / "frames");
        std::ofstream metadata(args.out / "render_config.json");
        metadata << "{\n  \"width\": " << args.width << ",\n  \"height\": " << args.height
            << ",\n  \"samples\": " << args.samples << ",\n  \"render_stride\": " << args.render_stride
            << ",\n  \"camera_eye\": [0.20, -0.48, 0.245],\n  \"camera_target\": [0, 0, 0.09],"
            << "\n  \"camera_fov_degrees\": 30,\n  \"surface_kernel_spacing_ratio\": 3,"
            << "\n  \"surface_cell_spacing_ratio\": 0.5,\n  \"surface_iso_fraction\": 0.5,"
            << "\n  \"studio_floor_radius_m\": 32,"
            << "\n  \"state_interpolation\": false\n}\n";
    }

    void Frame(uint32_t frame, const std::vector<Vec3>& positions,
               const std::vector<Transform>& bodies) {
        if (frame % args_.render_stride) return;
        render::PublishStudioScene(scene_, {}, positions, bodies);
        for (size_t i = 0; i < bodies.size(); ++i)
            scene_.world.instances[plate_instances_[i]].world_xform = bodies[i];
        Transform piston_local=Transform::Identity();
        piston_local.position.z=kPlateHalf.z+0.12f;
        scene_.world.instances[piston_instance_].world_xform=bodies.back()*piston_local;
        const auto report = renderer_->Render(scene_.world, scene_.options);
        Require(report.pixels.size() == size_t(args_.width) * args_.height,
                "Render did not return a complete image");
        char name[48];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", frame);
        std::ofstream file(args_.out / "frames" / name, std::ios::binary);
        file << "P6\n" << args_.width << " " << args_.height << "\n255\n";
        std::vector<uint8_t> rgb(report.pixels.size() * 3u);
        for (size_t i = 0; i < report.pixels.size(); ++i) {
            rgb[3*i] = report.pixels[i].r;
            rgb[3*i+1] = report.pixels[i].g;
            rgb[3*i+2] = report.pixels[i].b;
        }
        Write(file, rgb.data(), rgb.size());
    }
private:
    Args args_;
    render::StudioScene scene_;
    std::vector<size_t> plate_instances_;
    size_t piston_instance_=0u;
    std::unique_ptr<render::StudioRtRenderer> renderer_;
};

void Replay(const Args& args) {
    std::ifstream capture(args.replay / "positions.bin", std::ios::binary);
    uint32_t header[5];
    float spacing;
    Read(capture, header, 5u); Read(capture, &spacing, 1u);
    Require(header[0] == 0x4E554B41u && header[1] == 1u, "Unsupported capture format");
    std::vector<Vec3> positions(header[2]);
    std::vector<Transform> bodies(header[3]);
    Renderer renderer(args, spacing, header[2], header[3]);
    for (uint32_t frame = 0u; frame < header[4]; ++frame) {
        Read(capture, positions.data(), positions.size());
        Read(capture, bodies.data(), bodies.size());
        renderer.Frame(frame, positions, bodies);
        if (frame % 24u == 0u) std::printf("render frame %u/%u\n", frame, header[4]);
    }
}

void Simulate(const Args& args, phi::Device* device, phi::Backend* backend) {
    auto scene = BuildModel(args);
    const uint32_t count = scene.model.capacities.particles_per_env;
    const uint32_t body_count = scene.model.capacities.bodies_per_env;
    std::vector<Transform> bodies;
    for (const auto& body : scene.model.body_init) bodies.push_back(body.pose);
    const auto initial_bodies = bodies;
    std::vector<Vec3> body_velocity(body_count), reaction(body_count);
    const double dt = 1.0 / (kSampleHz * args.steps_per_frame);
    nk::Pipeline::SolverConfig config;
    config.dt = static_cast<float>(dt);
    config.gravity[0] = 0.0f; config.gravity[1] = 0.0f; config.gravity[2] = kGravity;
    nk::World world(std::move(scene.model), 1u, device, backend, config);
    Require(world.Ready(), "World creation: " + world.CreationError());
    Require(world.FieldPtr(nk::FieldId::MpmBodyReaction) != nullptr, "No boundary impulse readout");
    Require(world.SetExecutionMode(nk::World::ExecutionMode::Graph) == phi::Status::Ok,
            "Graph configuration failed");
    nuka::perf::CudaStepMeasurements measurements(world, !args.perf_json.empty(),
        args.steps_per_frame, kFrames);

    const float spacing = args.dx * 0.5f;
    const double volume = double(spacing) * spacing * spacing;
    const double mass = kDensity * volume;
    std::vector<Vec3> positions(count), velocity(count);
    std::vector<float> elastic(count*9u), plastic(count*9u), affine(count*9u), alpha(count);
    std::vector<float> previous_alpha(count, 0.0f);
    std::unique_ptr<Renderer> renderer;
    if (!args.no_render) renderer = std::make_unique<Renderer>(args, spacing, count, body_count);
    std::ofstream capture(args.out / "positions.bin", std::ios::binary);
    const uint32_t header[] = {0x4E554B41u, 1u, count, body_count, kFrames+1u};
    Write(capture, header, 5u); Write(capture, &spacing, 1u);
    std::ofstream metadata(args.out / "config.json");
    metadata << std::setprecision(10) << "{\n  \"dx\": " << args.dx
        << ",\n  \"dt\": " << dt << ",\n  \"steps_per_frame\": " << args.steps_per_frame
        << ",\n  \"sample_hz\": " << kSampleHz << ",\n  \"particles\": " << count
        << ",\n  \"spacing\": " << spacing << ",\n  \"density\": " << kDensity
        << ",\n  \"youngs\": " << kYoungs << ",\n  \"poisson\": " << kPoisson
        << ",\n  \"yield_stress\": " << kYield << ",\n  \"hardening_modulus\": " << kHardening
        << ",\n  \"body_contact_band\": " << kContactBand
        << ",\n  \"friction\": 0,\n  \"gravity_z\": " << kGravity
        << ",\n  \"specimen_m\": [0.08, 0.08, 0.10],\n  \"base_z\": 0.04,"
        << "\n  \"gaps_m\": [" << kOpenGap << ", " << kMildGap << ", " << kStrongGap << "],"
        << "\n  \"boundary\": \"prescribed rigid platens\"\n}\n";
    std::ofstream metrics(args.out / "metrics.csv");
    metrics << "frame,time_s,stage,gap_m,height_m,width_x_m,width_y_m,z_min_m,z_max_m,"
        "alpha_mean,alpha_max,alpha_increment_min,det_fp_error,det_fe_min,det_fe_max,"
        "equivalent_stress_max_pa,elastic_j,hardening_j,plastic_dissipation_j,kinetic_j,"
        "gravity_j,boundary_work_j,top_force_n,bottom_force_n,external_impulse_z_ns,"
        "momentum_z_ns,max_speed_m_s,env_status\n";
    metrics << std::setprecision(10);
    double work = 0.0, external_impulse = 0.0;
    for (uint32_t frame = 0u; frame <= kFrames; ++frame) {
        double top_impulse = 0.0, bottom_impulse = 0.0;
        uint32_t status = 0u;
        if (frame > 0u) {
            measurements.BeginInterval();
            for (uint32_t s = 0u; s < args.steps_per_frame; ++s) {
                const uint64_t step = uint64_t(frame-1u) * args.steps_per_frame + s;
                const double begin = step * dt, end = (step+1u) * dt;
                const uint32_t top = scene.plate_bodies.back();
                bodies[top].position.z = kBase + static_cast<float>(Gap(begin)) + kPlateHalf.z;
                body_velocity[top].z = static_cast<float>((Gap(end) - Gap(begin)) / dt);
                Require(world.GetData().UploadField(nk::FieldId::BodyPose, bodies.data(),
                    bodies.size() * sizeof(Transform)), "Boundary pose upload failed");
                Require(world.GetData().UploadField(nk::FieldId::BodyLinearVelocity,
                    body_velocity.data(), body_velocity.size() * sizeof(Vec3)), "Boundary velocity upload failed");
                measurements.Step();
                Download(world, nk::FieldId::MpmBodyReaction, reaction);
                uint32_t step_status = 0u;
                Require(world.GetData().DownloadField(nk::FieldId::EnvStatus, &step_status,
                    sizeof(step_status)), "Status read failed");
                status |= step_status;
                Require(status == 0u, "Environment failure flags " + std::to_string(status));
                for (size_t b = 0; b < bodies.size(); ++b) {
                    work -= reaction[b].Dot(body_velocity[b]);
                    external_impulse -= reaction[b].z;
                }
                external_impulse += mass * count * kGravity * dt;
                top_impulse += reaction[top].z;
                bottom_impulse -= reaction[scene.plate_bodies.front()].z;
                bodies[top].position.z = kBase + static_cast<float>(Gap(end)) + kPlateHalf.z;
            }
            Require(world.GetData().UploadField(nk::FieldId::BodyPose, bodies.data(),
                bodies.size() * sizeof(Transform)), "Final boundary pose upload failed");
        }
        Require(world.Synchronize() == phi::Status::Ok, "Simulation completion failed");
        if (frame > 0u) measurements.EndInterval();
        Download(world, nk::FieldId::ParticlePos, positions);
        Download(world, nk::FieldId::ParticleVel, velocity);
        Download(world, nk::FieldId::ParticleF, elastic);
        Download(world, nk::FieldId::ParticlePlasticF, plastic);
        Download(world, nk::FieldId::ParticleC, affine);
        Download(world, nk::FieldId::ParticlePlastic, alpha);
        Vec3 lo{1e30f,1e30f,1e30f}, hi{-1e30f,-1e30f,-1e30f};
        double alpha_sum=0.0, alpha_max=0.0, increment_min=0.0, fp_error=0.0;
        double fe_min=1e30, fe_max=0.0, stress_max=0.0, elastic_energy=0.0;
        double hardening_energy=0.0, kinetic=0.0, potential=0.0, momentum=0.0, speed_max=0.0;
        for (uint32_t p = 0u; p < count; ++p) {
            const auto x = positions[p], v = velocity[p];
            Require(std::isfinite(x.x) && std::isfinite(x.y) && std::isfinite(x.z) &&
                    std::isfinite(v.LengthSq()), "Nonfinite particle state");
            lo.x=std::min(lo.x,x.x); lo.y=std::min(lo.y,x.y); lo.z=std::min(lo.z,x.z);
            hi.x=std::max(hi.x,x.x); hi.y=std::max(hi.y,x.y); hi.z=std::max(hi.z,x.z);
            alpha_sum += alpha[p]; alpha_max=std::max(alpha_max,double(alpha[p]));
            increment_min=std::min(increment_min,double(alpha[p]-previous_alpha[p]));
            previous_alpha[p]=alpha[p];
            fp_error=std::max(fp_error,std::fabs(Determinant(plastic.data()+9u*p)-1.0));
            const double det_fe=Determinant(elastic.data()+9u*p);
            fe_min=std::min(fe_min,det_fe); fe_max=std::max(fe_max,det_fe);
            nk::material::HenckyResponse response;
            Require(nk::material::EvaluateHenckyJ2(elastic.data()+9u*p,
                {kYoungs,kPoisson,kYield,kHardening},response)==nk::material::ConstitutiveStatus::Ok,
                "Invalid sampled elastic state");
            stress_max=std::max(stress_max,double(response.equivalent_stress));
            elastic_energy += volume * response.elastic_energy;
            hardening_energy += volume * 0.5 * kHardening * alpha[p] * alpha[p];
            double affine_norm=0.0;
            for (uint32_t i=0; i<9u; ++i) affine_norm += double(affine[9u*p+i])*affine[9u*p+i];
            kinetic += 0.5 * mass * (v.LengthSq() + 0.25 * args.dx * args.dx * affine_norm);
            potential -= mass * kGravity * x.z;
            momentum += mass * v.z;
            speed_max=std::max(speed_max,double(std::sqrt(v.LengthSq())));
        }
        const double time = frame / double(kSampleHz);
        metrics << frame << ',' << time << ',' << Stage(time) << ',' << Gap(time) << ','
            << hi.z-lo.z << ',' << hi.x-lo.x << ',' << hi.y-lo.y << ',' << lo.z << ',' << hi.z << ','
            << alpha_sum/count << ',' << alpha_max << ',' << increment_min << ',' << fp_error << ','
            << fe_min << ',' << fe_max << ',' << stress_max << ',' << elastic_energy << ','
            << hardening_energy << ',' << volume*kYield*alpha_sum << ',' << kinetic << ','
            << potential << ',' << work << ',' << top_impulse*kSampleHz << ','
            << bottom_impulse*kSampleHz << ',' << external_impulse << ',' << momentum << ','
            << speed_max << ',' << status << '\n';
        Write(capture, positions.data(), positions.size());
        Write(capture, bodies.data(), bodies.size());
        if (renderer) renderer->Frame(frame, positions, bodies);
        if (frame % 24u == 0u) {
            metrics.flush(); capture.flush();
            std::printf("t=%.3f %s height=%.5f alpha_mean=%.6f max=%.6f force=%.3f detFp=%.3g\n",
                time, Stage(time), hi.z-lo.z, alpha_sum/count, alpha_max,
                top_impulse*kSampleHz, fp_error);
        }
    }
    Require(world.Reset() == phi::Status::Ok && world.Synchronize() == phi::Status::Ok,
            "World reset failed");
    Download(world, nk::FieldId::ParticlePlastic, alpha);
    Download(world, nk::FieldId::ParticlePos, positions);
    Download(world, nk::FieldId::ParticleVel, velocity);
    Download(world, nk::FieldId::ParticleF, elastic);
    Download(world, nk::FieldId::ParticlePlasticF, plastic);
    Download(world, nk::FieldId::ParticleC, affine);
    Download(world, nk::FieldId::BodyPose, bodies);
    Download(world, nk::FieldId::BodyLinearVelocity, body_velocity);
    const auto& initial_positions = world.GetModel().particles.initial_pos;
    for (uint32_t p=0u; p<count; ++p) {
        Require(alpha[p]==0.0f && (positions[p]-initial_positions[p]).LengthSq()==0.0f &&
                velocity[p].LengthSq()==0.0f, "Reset did not restore particle state");
        for (uint32_t k=0u; k<9u; ++k)
            Require(elastic[9u*p+k]==(k%4u==0u ? 1.0f : 0.0f) &&
                    plastic[9u*p+k]==(k%4u==0u ? 1.0f : 0.0f) && affine[9u*p+k]==0.0f,
                    "Reset did not restore material tensors");
    }
    for (size_t b=0; b<bodies.size(); ++b)
        Require((bodies[b].position-initial_bodies[b].position).LengthSq()==0.0f &&
                body_velocity[b].LengthSq()==0.0f,
                "Reset did not restore boundary pose");
    std::ofstream completion(args.out / "completion.json");
    completion << "{\"frames\": " << kFrames+1u << ", \"reset_passed\": true, "
        "\"material_reset_between_loads\": false}\n";
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
        {
            if (args.replay.empty()) Simulate(args, device, backend);
            else Replay(args);
        }
        phi::BackendFree(backend);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "elastoplastic_compression: %s\n", error.what());
        return 1;
    }
    return 0;
}
