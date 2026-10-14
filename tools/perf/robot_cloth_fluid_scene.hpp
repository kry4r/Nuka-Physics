#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision/shape_kind.hpp"
#include "import/usd_importer.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace nuka::perf::fixture {
namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
using nuka::math::Transform;
using nuka::math::Vec3;

inline void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

inline void RequireStep(nk::World& world, const std::string& message, uint32_t index) {
    const auto result = world.Step();
    if (result.AllOk()) return;
    throw std::runtime_error(message + " at step " + std::to_string(index) +
        " (op " + std::to_string(static_cast<uint32_t>(result.failed_op)) +
        ", status " + std::to_string(static_cast<uint32_t>(result.result)) + "): " +
        world.LastExecutionError().message);
}

constexpr float kContactDMin = 0.030f;
constexpr uint32_t kClothNx = 13u;          // 13x13 = 169 cloth particles.
constexpr float kClothSpacing = 0.016f;     // ~0.19 m square (spans both front feet).
constexpr float kClothParticleMass = 0.01f; // 10 g per particle (light cloth).

constexpr uint32_t kSettleSteps = 250u;     // settle the robot stance + the media.
constexpr uint32_t kHoldSteps = 200u;       // co-step the held stance over the media.

inline nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.06f; cfg.gravity[1] = -0.04f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 48u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}
inline cook::VbdCookInput BuildCloth(float cx, float cy, float z, uint32_t nx = kClothNx,
                                     float spacing_x = kClothSpacing) {
    Require(nx >= 3u && uint64_t{nx} * nx <= std::numeric_limits<uint32_t>::max() / 6u,
            "cloth grid exceeds topology index range");
    Require(std::isfinite(spacing_x) && spacing_x > 0.0f, "cloth spacing must be positive");
    std::vector<Vec3> rest;
    rest.reserve(nx * nx);
    const float c0 = -0.5f * static_cast<float>(nx - 1u) * kClothSpacing;
    const float x0 = -0.5f * static_cast<float>(nx - 1u) * spacing_x;
    for (uint32_t j = 0; j < nx; ++j)
        for (uint32_t i = 0; i < nx; ++i)
            rest.push_back(Vec3{cx + x0 + static_cast<float>(i) * spacing_x,
                                cy + c0 + static_cast<float>(j) * kClothSpacing, z});
    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };
    std::vector<soft::ClothTriangle> tris;
    for (uint32_t j = 0; j + 1 < nx; ++j)
        for (uint32_t i = 0; i + 1 < nx; ++i) {
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j),
                                                idx(i + 1, j + 1)}});
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j + 1),
                                                idx(i, j + 1)}});
        }
    soft::ShellMaterial shell;
    shell.stretch_stiffness = 1.0e4f;
    shell.poisson = 0.3f;
    shell.bend_stiffness = 1.0e-5f;

    cook::VbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f / kClothParticleMass);
    const uint32_t last = nx - 1u;
    for (uint32_t k = 0; k < nx; ++k) {
        in.inv_mass[idx(k, 0)] = 0.0f; in.inv_mass[idx(k, last)] = 0.0f;
        in.inv_mass[idx(0, k)] = 0.0f; in.inv_mass[idx(last, k)] = 0.0f;
    }
    soft::BuildClothVertexBlocks(rest, tris, {}, std::vector<soft::ShellMaterial>(tris.size(), shell),
                                 in.vbd_elements);
    in.vbd_count = static_cast<uint32_t>(rest.size());
    for (const auto& triangle : tris)
        in.aero_triangles.push_back({triangle.v[0], triangle.v[1], triangle.v[2]});
    in.aero_drag_normal = 0.6f;
    in.aero_drag_tangent = 0.04f;
    in.aero_drag_max_dv = 0.5f;
    in.half_thickness.assign(rest.size(), 0.0f);
    in.friction.assign(rest.size(), 0.6f);  // finite mu: the foot grips/drags the cloth.
    cook::CookParticleSurface surface;
    for (const auto& triangle : tris)
        surface.triangles.insert(surface.triangles.end(), triangle.v, triangle.v + 3u);
    in.surfaces.push_back(std::move(surface));
    return in;
}

inline nuka::scene::SceneIR RobotScene(const std::filesystem::path& path, bool with_free_body = false) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(path.string());
    if (with_free_body) {
        nuka::scene::RigidBodyRecord body;
        body.name = "free_body";
        body.mass = 2.0f;
        body.inertia = {0.03f, 0.05f, 0.07f};
        body.local_transform.position = {3.0f, 2.0f, 4.0f};
        body.inertial_transform.position = {0.01f, -0.015f, 0.02f};
        nuka::scene::CollisionShapeRecord shape;
        shape.body_id = scene.AddRigidBody(body);
        shape.type = nuka::scene::ShapeType::Sphere;
        shape.radius = 0.05f;
        scene.AddCollisionShape(shape);
    }
    return scene;
}

struct SceneVisuals {
    nuka::scene::SceneIR scene;
    nuka::scene::SceneMap scene_map;
    std::vector<cook::MediaRenderSurface> material_surfaces;
};

inline nk::Model CookRobot(const std::filesystem::path& path, bool with_free_body = false,
                           SceneVisuals* visuals = nullptr) {
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    auto scene = RobotScene(path, with_free_body);
    auto cooked = cook::CookToModel(scene, 1, opt);
    nk::Model model = std::move(cooked.model);
    model.capacities.max_contacts_per_env = 32u;
    model.capacities.max_rows_per_env =
        model.capacities.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    if (visuals) {
        visuals->scene = std::move(scene);
        visuals->scene_map = std::move(cooked.scene_map);
        visuals->material_surfaces.clear();
    }
    return model;
}

struct PreparedScene {
    std::filesystem::path path;
    nk::Pipeline::SolverConfig config;
    Vec3 front_centre;
    float cloth_z;
    uint32_t front_link;
    std::vector<Transform> link_geom_local;
    std::vector<float> settled_q, settled_qdot;
};

inline PreparedScene Prepare(const std::filesystem::path& path, phi::Device* device,
                             phi::Backend* backend, const nk::Pipeline::SolverConfig& config) {
    nk::Model probe = CookRobot(path);
    const uint32_t L = probe.capacities.links_per_env;
    const std::vector<uint32_t> link_geom_kind = probe.articulation.link_geom_kind;
    const std::vector<Transform> link_geom_local = probe.articulation.link_geom_local;
    Require(L > 0u, "robot has no links");
    constexpr uint32_t kSphereGeom = nuka::collision::kShapeSphere + 1u;
    std::vector<uint32_t> foot_links;
    for (uint32_t l = 0; l < L; ++l)
        if (l < link_geom_kind.size() && link_geom_kind[l] == kSphereGeom)
            foot_links.push_back(l);
    Require(foot_links.size() >= 4u, "expected four Go2 foot spheres");

    nk::World wp(std::move(probe), 1u, device, backend, config);
    Require(wp.Ready(), wp.CreationError());
    for (uint32_t s = 0; s < kSettleSteps; ++s) RequireStep(wp, "robot calibration step failed", s);
    std::vector<Transform> link_pose(L);
    Require(wp.GetData().DownloadField(nk::FieldId::LinkPose, link_pose.data(),
                                           L * sizeof(Transform)), "calibration pose download failed");
    std::vector<float> settled_q(L), settled_qdot(L);
    Require(wp.GetData().DownloadField(nk::FieldId::Q, settled_q.data(), L * sizeof(float)),
            "calibration joint position download failed");
    Require(wp.GetData().DownloadField(nk::FieldId::Qdot, settled_qdot.data(), L * sizeof(float)),
            "calibration joint velocity download failed");
    std::vector<Vec3> foot_world;
    for (uint32_t l : foot_links)
        foot_world.push_back((link_pose[l] * link_geom_local[l]).position);
    for (size_t i = 0; i < foot_world.size(); ++i)
        std::fprintf(stderr, "[robot-coupling] foot_link=%u rest @ (%.4f,%.4f,%.4f)\n",
                     foot_links[i], foot_world[i].x, foot_world[i].y, foot_world[i].z);
    auto by_x = foot_world;
    std::sort(by_x.begin(), by_x.end(),
              [](const Vec3& a, const Vec3& c) { return a.x < c.x; });
    const float mid_x = 0.5f * (by_x.back().x + by_x.front().x);
    Vec3 front_centre{0, 0, 0}; uint32_t nf = 0;
    uint32_t front_link = foot_links[0];
    float cloth_z = 0.0f;
    for (size_t i = 0; i < foot_world.size(); ++i)
        if (foot_world[i].x > mid_x) {
            front_centre = front_centre + foot_world[i]; ++nf;
            front_link = foot_links[i]; cloth_z = foot_world[i].z;
        }
    front_centre = front_centre * (1.0f / static_cast<float>(nf));
    std::fprintf(stderr, "[robot-coupling] front_centre=(%.4f,%.4f) cloth_z=%.4f\n",
                 front_centre.x, front_centre.y, cloth_z);
    return {path, config, front_centre, cloth_z, front_link, link_geom_local, std::move(settled_q), std::move(settled_qdot)};
}

inline nk::Model CookPrepared(const PreparedScene& scene, uint32_t envs = 1u,
                              bool patch_present = true, uint32_t cloth_nx = kClothNx,
                              SceneVisuals* visuals = nullptr) {
    nk::Model model = CookRobot(scene.path, true, visuals);
    auto cloth = BuildCloth(scene.front_centre.x, scene.front_centre.y, scene.cloth_z, cloth_nx);
    if (!patch_present)
        for (auto& position : cloth.positions) position.z -= 10.0f;
    cook::CookVbdParticles(model, envs, cloth);
    model.particles.pp_contact_d_min = kContactDMin;
    for (auto& body : model.body_init)
        if (body.inv_mass > 0.0f) body.angular_velocity = {5.0f, -3.0f, 2.0f};
    return model;
}

inline std::vector<uint32_t> LatticeBoundary(const std::array<uint32_t, 3>& dimensions) {
    Require(dimensions[0] >= 2u && dimensions[1] >= 2u && dimensions[2] >= 2u &&
            uint64_t{dimensions[0]} * dimensions[1] <= UINT32_MAX / dimensions[2],
            "invalid material lattice dimensions");
    const auto index = [&](const std::array<uint32_t, 3>& point) {
        return (point[2] * dimensions[1] + point[1]) * dimensions[0] + point[0];
    };
    std::vector<uint32_t> triangles;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        const uint32_t u = (axis + 1u) % 3u, v = (axis + 2u) % 3u;
        for (uint32_t side = 0u; side < 2u; ++side)
            for (uint32_t j = 0u; j + 1u < dimensions[v]; ++j)
                for (uint32_t i = 0u; i + 1u < dimensions[u]; ++i) {
                    std::array<uint32_t, 3> point{};
                    point[axis] = side * (dimensions[axis] - 1u);
                    point[u] = i; point[v] = j;
                    const auto a = index(point);
                    ++point[u]; const auto b = index(point);
                    ++point[v]; const auto c = index(point);
                    --point[u]; const auto d = index(point);
                    if (side) triangles.insert(triangles.end(), {a, b, c, a, c, d});
                    else triangles.insert(triangles.end(), {a, c, b, a, d, c});
                }
    }
    return triangles;
}

inline nk::Model CookMpmPrepared(const PreparedScene& prepared, uint32_t envs,
                                SceneVisuals* visuals = nullptr, bool implicit_stress = false) {
    constexpr float radius = 0.028f, mass = 0.15f, dx = 0.02f;
    // The membrane starts 1 mm into the loaded front feet, so they rest on it.
    const float cloth_z = prepared.cloth_z - 0.0175f + 0.001f;
    auto scene = RobotScene(prepared.path, true);
    auto free_body = nuka::scene::kInvalidBody;
    for (const auto& body : scene.Bodies())
        if (body.name == "free_body") free_body = body.id;
    Require(free_body != nuka::scene::kInvalidBody, "free body is missing");
    auto& body = scene.GetBodyMut(free_body);
    body.mass = mass;
    body.inertia = Vec3{1, 1, 1} * (0.4f * mass * radius * radius);
    body.inertial_transform = Transform::Identity();
    body.local_transform.position = {prepared.front_centre.x + 0.039f,
                                     prepared.front_centre.y, cloth_z + radius};
    std::vector<nuka::scene::ShapeId> shapes;
    for (const auto& shape : scene.Shapes())
        if (shape.body_id == free_body) shapes.push_back(shape.id);
    for (auto id : shapes) scene.GetShapeMut(id).radius = radius;
    const float cloth_x = 0.5f * (prepared.front_centre.x + body.local_transform.position.x);
    auto impact_body = body;
    impact_body.name = "impact_body";
    impact_body.local_transform.position.x = prepared.front_centre.x - radius - dx;
    const auto impact_id = scene.AddRigidBody(impact_body);
    nuka::scene::CollisionShapeRecord impact_shape;
    impact_shape.body_id = impact_id;
    impact_shape.type = nuka::scene::ShapeType::Sphere;
    impact_shape.radius = radius;
    scene.AddCollisionShape(impact_shape);
    cook::CookToModelOptions options;
    options.contact_family = cook::CookContactFamily::PairDriven;
    auto cooked = cook::CookToModel(scene, envs, options);
    const auto* impact_ref = cooked.scene_map.RefOf(scene.EntityOfBody(impact_id));
    Require(impact_ref != nullptr, "impact body is missing from the cooked scene map");
    cooked.model.body_init.at(impact_ref->body_row).linear_velocity = {0.5f, 0.0f, 0.0f};
    nk::Model model = std::move(cooked.model);
    model.articulation.initial_q = prepared.settled_q;
    model.articulation.initial_qdot = prepared.settled_qdot;
    // The membrane reaches past both contact points, so half of each body's footprint stays over bare material.
    const float cloth_span = body.local_transform.position.x - prepared.front_centre.x + 0.25f * radius;
    auto cloth = BuildCloth(cloth_x, prepared.front_centre.y, cloth_z, kClothNx,
                            cloth_span / float(kClothNx - 1u));
    cloth.aero_drag_normal = cloth.aero_drag_tangent = cloth.aero_drag_max_dv = 0.0f;
    cook::MpmCookInput mpm;
    mpm.implicit_stress = implicit_stress;
    const Vec3 centre{prepared.front_centre.x + 0.01f, prepared.front_centre.y, cloth_z};
    const float spacing = 0.5f * dx;
    constexpr std::array<uint32_t, 3> dimensions{13u, 9u, 4u};
    for (uint32_t z = 0u; z < dimensions[2]; ++z)
        for (uint32_t y = 0u; y < dimensions[1]; ++y)
            for (uint32_t x = 0u; x < dimensions[0]; ++x)
                mpm.positions.push_back(centre + Vec3{(float(x) - 6.0f) * spacing,
                    (float(y) - 4.0f) * spacing, (float(z) - 3.5f) * spacing});
    mpm.material.youngs = 10000.0f;
    mpm.material.poisson = 0.3f;
    mpm.material.density = 1000.0f;
    const float volume = spacing * spacing * spacing;
    mpm.vol0.assign(mpm.positions.size(), volume);
    mpm.inv_mass.assign(mpm.positions.size(), 1.0f / (mpm.material.density * volume));
    mpm.velocities.assign(mpm.positions.size(), Vec3{0.0f, 0.0f, 0.5f});
    mpm.grid_origin = centre - Vec3{0.14f, 0.14f, 0.14f};
    mpm.grid_dims[0] = mpm.grid_dims[1] = mpm.grid_dims[2] = 15u;
    mpm.dx = dx;
    mpm.substeps = 4u;
    mpm.floor_d = cloth_z - 0.04f;
    mpm.floor_friction = 0.6f;
    mpm.contact_capacity = 2048u;
    cook::CookMpmVbd(model, envs, mpm, cloth);
    model.particles.pp_contact_d_min = kClothSpacing;
    model.particles.mpm_body_friction = 0.6f;
    if (visuals) {
        visuals->scene = std::move(scene);
        visuals->scene_map = std::move(cooked.scene_map);
        visuals->material_surfaces.clear();
        cook::MediaRenderSurface surface;
        surface.triangles = LatticeBoundary(dimensions);
        surface.particle_count = static_cast<uint32_t>(mpm.positions.size());
        visuals->material_surfaces.push_back(std::move(surface));
    }
    return model;
}
}  // namespace nuka::perf::fixture
