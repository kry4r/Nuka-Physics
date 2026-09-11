#pragma once

#include <algorithm>
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

constexpr float kContactDMin = 0.030f;
constexpr uint32_t kClothNx = 13u;          // 13x13 = 169 cloth particles.
constexpr float kClothSpacing = 0.016f;     // ~0.19 m square (spans both front feet).
constexpr float kClothParticleMass = 0.01f; // 10 g per particle (light cloth).
constexpr uint16_t kClothIters = 24u;       // tighter stretch solve = taut patch.
constexpr float kPoolSpacing = 0.025f;
constexpr float kPoolSupport = kPoolSpacing * 1.5f;
constexpr float kPoolRestDensity = 1000.0f;
constexpr uint32_t kPoolNx = 5u;            // 5x5 footprint.
constexpr uint32_t kPoolNz = 4u;            // 4 layers deep.

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
inline cook::XpbdCookInput BuildCloth(float cx, float cy, float z, uint32_t nx = kClothNx) {
    Require(nx >= 3u && uint64_t{nx} * nx <= std::numeric_limits<uint32_t>::max() / 6u,
            "cloth grid exceeds topology index range");
    std::vector<Vec3> rest;
    rest.reserve(nx * nx);
    const float c0 = -0.5f * static_cast<float>(nx - 1u) * kClothSpacing;
    for (uint32_t j = 0; j < nx; ++j)
        for (uint32_t i = 0; i < nx; ++i)
            rest.push_back(Vec3{cx + c0 + static_cast<float>(i) * kClothSpacing,
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
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;
    opts.bend_compliance_alpha = 1.0e-4f;
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, tris, opts, cs);

    cook::XpbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f / kClothParticleMass);
    const uint32_t last = nx - 1u;
    for (uint32_t k = 0; k < nx; ++k) {
        in.inv_mass[idx(k, 0)] = 0.0f; in.inv_mass[idx(k, last)] = 0.0f;
        in.inv_mass[idx(0, k)] = 0.0f; in.inv_mass[idx(last, k)] = 0.0f;
    }
    for (const auto& dc : cs.distance) {
        cook::CookDistanceCon c;
        c.a = dc.particle_a; c.b = dc.particle_b;
        c.rest_length = dc.rest_length; c.compliance_alpha = dc.compliance_alpha;
        in.distance.push_back(c);
    }
    for (const auto& bc : cs.bend) {
        cook::CookBendCon c;
        for (uint32_t k = 0; k < 4u; ++k) { c.p[k] = bc.particle[k]; c.k[k] = bc.k[k]; }
        c.compliance_alpha = bc.compliance_alpha;
        in.bend.push_back(c);
    }
    in.solver_iterations = kClothIters;
    for (const auto& triangle : tris)
        in.aero_triangles.push_back({triangle.v[0], triangle.v[1], triangle.v[2]});
    in.aero_drag_normal = 0.6f;
    in.aero_drag_tangent = 0.04f;
    in.aero_drag_max_dv = 0.5f;
    in.friction = 0.6f;   // finite mu: the foot grips/drags the cloth.
    return in;
}
inline cook::PbfCookInput BuildPool(float cx, float cy, float floor_z) {
    cook::PbfCookInput in;
    const float s = kPoolSpacing, h = kPoolSupport, rho0 = kPoolRestDensity;
    const float c0 = -0.5f * static_cast<float>(kPoolNx - 1u) * s;
    const float bottom = floor_z + 0.5f * s;
    for (uint32_t iz = 0; iz < kPoolNz; ++iz)
        for (uint32_t iy = 0; iy < kPoolNx; ++iy)
            for (uint32_t ix = 0; ix < kPoolNx; ++ix)
                in.positions.push_back(Vec3{cx + c0 + static_cast<float>(ix) * s,
                                            cy + c0 + static_cast<float>(iy) * s,
                                            bottom + static_cast<float>(iz) * s});
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.particle_mass = rho0 * s * s * s;
    in.rest_density = rho0;
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = 4u;
    in.clamp_overdensity = true;
    in.boundary_enabled = true;
    in.floor_z = floor_z;
    in.friction = 0.0f;   // fluid mu ~= 0: the foot slides; splash stays normal-driven.
    const float half = 0.5f * static_cast<float>(kPoolNx - 1u) * s + 3.0f * h;
    const float top = floor_z + kPoolNz * s + 4.0f * h;
    in.grid_min = Vec3{cx - half - h, cy - half - h, floor_z - h};
    auto cells = [&](float extent) {
        return static_cast<uint32_t>(std::ceil(extent / h)) + 1u;
    };
    in.grid_dims[0] = cells(2.0f * half);
    in.grid_dims[1] = cells(2.0f * half);
    in.grid_dims[2] = cells(top - (floor_z - h));
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

inline nk::Model CookRobot(const std::filesystem::path& path, bool with_free_body = false) {
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    auto scene = RobotScene(path, with_free_body);
    nk::Model model = cook::CookToModel(scene, 1, opt).model;
    model.capacities.max_contacts_per_env = 32u;
    model.capacities.max_rows_per_env =
        model.capacities.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    return model;
}

struct PreparedScene {
    std::filesystem::path path;
    nk::Pipeline::SolverConfig config;
    Vec3 front_centre, rear_foot;
    float cloth_z, pool_floor;
    uint32_t front_link, rear_link;
    std::vector<Transform> link_geom_local;
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
    for (uint32_t s = 0; s < kSettleSteps; ++s) Require(wp.Step().AllOk(), "robot calibration step failed");
    std::vector<Transform> link_pose(L);
    Require(wp.GetData().DownloadField(nk::FieldId::LinkPose, link_pose.data(),
                                           L * sizeof(Transform)), "calibration pose download failed");
    std::vector<Vec3> foot_world;
    for (uint32_t l : foot_links)
        foot_world.push_back((link_pose[l] * link_geom_local[l]).position);
    for (size_t i = 0; i < foot_world.size(); ++i)
        std::fprintf(stderr, "[robot-coupling] foot_link=%u rest @ (%.4f,%.4f,%.4f)\n",
                     foot_links[i], foot_world[i].x, foot_world[i].y, foot_world[i].z);
    auto by_x = foot_world;
    std::sort(by_x.begin(), by_x.end(),
              [](const Vec3& a, const Vec3& c) { return a.x < c.x; });
    const float front_x = by_x.back().x, rear_x = by_x.front().x;
    Vec3 front_centre{0, 0, 0}; uint32_t nf = 0;
    Vec3 rear_foot{0, 0, 0}; float rear_best = 1e9f;
    float foot_rest_z = 0.0f;
    uint32_t front_link = foot_links[0], rear_link = foot_links[0];
    for (size_t i = 0; i < foot_world.size(); ++i) {
        const Vec3& f = foot_world[i];
        foot_rest_z += f.z;
        if (f.x > 0.5f * (front_x + rear_x)) {
            front_centre = front_centre + f; ++nf; front_link = foot_links[i];
        } else if (f.x < rear_best) {
            rear_best = f.x; rear_foot = f; rear_link = foot_links[i];
        }
    }
    foot_rest_z /= static_cast<float>(foot_world.size());
    front_centre = front_centre * (1.0f / static_cast<float>(nf));
    std::fprintf(stderr,
                 "[robot-coupling] front_centre=(%.4f,%.4f) rear_foot=(%.4f,%.4f) "
                 "foot_rest_z=%.4f\n",
                 front_centre.x, front_centre.y, rear_foot.x, rear_foot.y, foot_rest_z);
    const float pool_top = foot_rest_z;
    const float pool_floor = pool_top - static_cast<float>(kPoolNz - 1u) * kPoolSpacing -
                             0.5f * kPoolSpacing;
    float front_foot_z_loaded = foot_rest_z;
    {
        nk::Model m = CookRobot(path);
        cook::PbfCookInput pool = BuildPool(rear_foot.x, rear_foot.y, pool_floor);
        cook::CookSoftFluidParticles(m, 1u, cook::XpbdCookInput{}, pool);
        m.particles.pp_contact_d_min = kContactDMin;
        nk::World w(std::move(m), 1u, device, backend, config);
        Require(w.Ready(), w.CreationError());
        for (uint32_t s = 0; s < kSettleSteps; ++s) Require(w.Step().AllOk(), "pool calibration step failed");
        std::vector<Transform> lp(L);
        Require(w.GetData().DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform)), "calibration pose download failed");
        front_foot_z_loaded = (lp[front_link] * link_geom_local[front_link]).position.z;
    }
    std::fprintf(stderr, "[robot-coupling] front_foot_z_loaded=%.4f\n",
                 front_foot_z_loaded);
    const float cloth_z = front_foot_z_loaded;


    return {path, config, front_centre, rear_foot, cloth_z, pool_floor,
            front_link, rear_link, link_geom_local};
}

inline nk::Model CookPrepared(const PreparedScene& scene, uint32_t envs = 1u,
                              bool patch_present = true, uint32_t cloth_nx = kClothNx) {
    nk::Model model = CookRobot(scene.path, true);
    auto cloth = BuildCloth(scene.front_centre.x, scene.front_centre.y, scene.cloth_z, cloth_nx);
    auto pool = BuildPool(scene.rear_foot.x, scene.rear_foot.y, scene.pool_floor);
    if (!patch_present) {
        for (auto& position : cloth.positions) position.z -= 10.0f;
        for (auto& position : pool.positions) position.z -= 10.0f;
        pool.grid_min.z -= 10.0f;
        pool.floor_z -= 10.0f;
    }
    cook::CookSoftFluidParticles(model, envs, cloth, pool);
    model.particles.pp_contact_d_min = kContactDMin;
    for (auto& body : model.body_init)
        if (body.inv_mass > 0.0f) body.angular_velocity = {5.0f, -3.0f, 2.0f};
    return model;
}
}  // namespace nuka::perf::fixture
