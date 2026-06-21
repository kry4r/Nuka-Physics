// ---------------------------------------------------------------------------
// The COUPLED world through the C ABI: a robot (Go2) + cloth (XPBD) + fluid (PBF)
// created + stepped + read through the public C entries
// (nuka_world_create_coupled_from_scene / nuka_world_step /
// nuka_world_get_buffer_view), all on the ONE general contact pipeline. This is the
// demo / RL bridge reachability floor: a coupled world is creatable + steppable +
// its particle state is readable from outside the engine.
//
// Gates:
//  (1) BRIDGE SMOKE: a coupled world steps stably; the live particle positions
//      (NUKA_FIELD_PARTICLE_POSITION) move under the robot vs a control whose media
//      are sunk far out of reach -> two-way body<->particle coupling fired through
//      the C step path. The particle field RESOLVES + is finite.
//  (2) TERRAIN + PARTICLE BUDGET (the c_abi recompute fix): a coupled world cooked
//      WITH a baked heightfield (contact_family == 1) AND particles builds + steps
//      stably (no NaN, no silent slot corruption) and still serves + moves the
//      particle field. The recompute that re-sizes the rigid budget after appending
//      terrain collidables re-applies the additive body<->particle reserve, so the
//      rigid and particle slot ranges stay disjoint (pre-fix the particle reserve
//      was clobbered -> corruption). The DISJOINT-RANGE numbers are pinned at the
//      cook level by nuka_scenario_test BodyParticleBudget.*; this gate proves the
//      end-to-end c_abi terrain+particle path is sound.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

std::string ScenePath() {
    return SourcePath("examples/scenes/go2_stand.usda").string();
}

bool SceneAvailable() {
    return std::filesystem::exists(SourcePath("examples/scenes/go2_stand.usda"));
}

struct DeviceGuard {
    nuka_device_handle handle = nullptr;
    DeviceGuard() {
        nuka_device_desc_t desc{};
        desc.gpu_index = 0u;
        desc.cuda_stream = nullptr;
        desc.backend_selection_layer_enabled = 1u;
        EXPECT_EQ(nuka_device_create(&desc, &handle), NUKA_RESULT_OK);
    }
    ~DeviceGuard() {
        if (handle != nullptr) nuka_device_destroy(handle);
    }
};

struct WorldGuard {
    nuka_world_handle handle = nullptr;
    ~WorldGuard() {
        if (handle != nullptr) nuka_world_destroy(handle);
    }
};

std::vector<float> DownloadField(nuka_world_handle world, nuka_state_field_t field,
                                 size_t* out_floats_per_element = nullptr) {
    nuka_buffer_view_t view{};
    EXPECT_EQ(nuka_world_get_buffer_view(world, field, &view), NUKA_RESULT_OK);
    std::vector<float> out;
    if (view.device_ptr == nullptr || view.element_count == 0u) return out;
    const size_t fpe = view.element_stride_bytes / sizeof(float);
    if (out_floats_per_element != nullptr) *out_floats_per_element = fpe;
    out.resize(view.element_count * fpe);
    EXPECT_EQ(cudaMemcpy(out.data(), view.device_ptr, out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// The Go2 base settles around z ~ 0.30 with its feet near z ~ 0.05; place the cloth
// under the front of the base and the fluid pool under the rear, both near the foot
// rest height so the lower hemisphere of a foot touches the medium.
// The Go2 (fixed base at z=0.445) settles with its four foot spheres (radius 0.0175)
// centred at z ~ 0.2545: the front pair at x ~ 0.062, the rear pair at x ~ -0.325
// (probed from ARTICULATION_LINK_POSE). The cloth membrane lies just under the front
// feet and the fluid pool surface just under a rear foot so each foot's lower
// hemisphere rests in its medium.
// The control sinks the media this far below the feet -- a small offset so the
// CookFluidBox floor(L/s) lattice count is invariant (a large offset cancels float
// bits in the box span and recounts the lattice). At -0.30 m the fluid surface sits
// ~ -0.045 m, far below the foot spheres at z~0.237.
constexpr float kSinkOffset = -0.30f;

nuka_coupled_particles_desc_t MediaDesc(bool at_feet, float contact_radius = 0.012f) {
    nuka_coupled_particles_desc_t p{};
    const float dz = at_feet ? 0.0f : kSinkOffset;
    const float foot_z = 0.2545f;
    p.cloth_nx = 13u;
    p.cloth_ny = 13u;
    p.cloth_spacing = 0.018f;
    p.cloth_origin_x = 0.062f;   // straddling the two front feet.
    p.cloth_origin_y = 0.0f;
    p.cloth_origin_z = (foot_z - 0.012f) + dz;  // membrane just under the foot centre.
    p.cloth_particle_mass = 0.01f;
    p.cloth_friction = 0.6f;
    p.cloth_bend_alpha = 1.0e-4f;
    p.cloth_iters = 24u;
    // A shallow pool under the rear-left foot (x ~ -0.325, y ~ 0.017); the surface at
    // ~the foot centre so the lower hemisphere is submerged from rest.
    const float pool_surf = foot_z + 0.01f;
    p.fluid_min_x = -0.39f; p.fluid_min_y = -0.06f;
    p.fluid_max_x = -0.26f; p.fluid_max_y = 0.09f;
    p.fluid_spacing = 0.022f;
    p.fluid_min_z = (foot_z - 0.10f) + dz;
    p.fluid_max_z = pool_surf + dz;
    p.fluid_rest_density = 1000.0f;
    p.fluid_floor_z = (foot_z - 0.105f) + dz;
    p.fluid_friction = 0.0f;
    p.fluid_iters = 4u;
    p.contact_radius = contact_radius;
    return p;
}

nuka_world_desc_t WorldDesc(const std::string& scene, uint32_t terrain) {
    nuka_world_desc_t d{};
    d.scene_path = scene.c_str();
    d.env_count = 1u;
    d.fixed_dt = 1.0f / 240.0f;
    d.contact_family = terrain;  // 1 == bake a heightfield collidable.
    return d;
}

// Max particle z over a particle index slice [lo, hi) -- a medium's free-surface
// height. The fluid occupies the upper slice [n_soft, P) of the [soft|fluid] layout.
float MaxZInSlice(const std::vector<float>& pos, size_t lo, size_t hi) {
    float hi_z = -1.0e9f;
    const size_t n = pos.size() / 3u;
    for (size_t i = lo; i < hi && i < n; ++i) hi_z = std::fmax(hi_z, pos[i * 3u + 2u]);
    return hi_z;
}

// Reference-relative particle L1 between an at-feet coupled world and a sunk control
// after `steps`, over ALL particles (cloth + fluid). A residual > 0 means the robot
// displaced the at-feet media (the coupling fired); ~0 means it was inert (the robot
// passed through). The cloth under the front feet is the geometrically robust signal.
double CoupledMediaRefRelL1(nuka_device_handle device, const std::string& scene,
                            float contact_radius, uint32_t steps,
                            size_t* particles_out) {
    auto run = [&](bool at_feet, std::vector<float>* out) -> bool {
        WorldGuard w;
        const nuka_world_desc_t d = WorldDesc(scene, 0u);
        const nuka_coupled_particles_desc_t p = MediaDesc(at_feet, contact_radius);
        if (nuka_world_create_coupled_from_scene(device, &d, &p, &w.handle) !=
            NUKA_RESULT_OK) {
            return false;
        }
        for (uint32_t s = 0; s < steps; ++s) {
            if (nuka_world_step(w.handle) != NUKA_RESULT_OK) return false;
        }
        size_t fpe = 0u;
        *out = DownloadField(w.handle, NUKA_FIELD_PARTICLE_POSITION, &fpe);
        return fpe == 3u && !out->empty();
    };
    std::vector<float> at_feet, control;
    if (!run(true, &at_feet) || !run(false, &control)) return -1.0;
    if (at_feet.size() != control.size()) return -1.0;
    const size_t n_total = at_feet.size() / 3u;
    if (particles_out != nullptr) *particles_out = n_total;
    // The control's media (and floor) are a constant kSinkOffset below the at-feet
    // world; subtracting it cancels the offset, so a residual is the robot's effect.
    double l1 = 0.0;
    for (size_t i = 0; i < n_total; ++i) {
        const double a = at_feet[i * 3u + 2u];
        const double c = control[i * 3u + 2u] - kSinkOffset;
        l1 += std::fabs(a - c);
    }
    return l1;
}

}  // namespace

// (1) The coupled world steps + two-way couples THROUGH the C step path: the live
// particle field moves under the robot vs a control whose media are sunk away.
TEST(CoupledWorldCAbi, CreateStepReadCouplesThroughBridge) {
    if (!SceneAvailable()) GTEST_SKIP() << "Go2 stand scene is not available";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const std::string scene = ScenePath();
    const uint32_t kSteps = 200u;

    auto run = [&](bool at_feet, std::vector<float>* particles_out,
                   size_t* fpe_out) -> bool {
        WorldGuard w;
        const nuka_world_desc_t d = WorldDesc(scene, 0u);
        const nuka_coupled_particles_desc_t p = MediaDesc(at_feet);
        const nuka_result_t rc =
            nuka_world_create_coupled_from_scene(device.handle, &d, &p, &w.handle);
        if (rc != NUKA_RESULT_OK) {
            ADD_FAILURE() << "coupled create failed: " << nuka_result_message(rc);
            return false;
        }
        // The particle field must RESOLVE on a coupled world.
        nuka_buffer_view_t pv{};
        EXPECT_EQ(nuka_world_get_buffer_view(w.handle, NUKA_FIELD_PARTICLE_POSITION,
                                             &pv),
                  NUKA_RESULT_OK);
        EXPECT_NE(pv.device_ptr, nullptr);
        EXPECT_GT(pv.element_count, 0u);
        EXPECT_EQ(pv.element_stride_bytes, 3u * sizeof(float));

        for (uint32_t s = 0; s < kSteps; ++s) {
            const nuka_result_t sr = nuka_world_step(w.handle);
            if (sr != NUKA_RESULT_OK) {
                ADD_FAILURE() << "step failed at " << s << ": "
                              << nuka_result_message(sr);
                return false;
            }
        }
        *particles_out = DownloadField(w.handle, NUKA_FIELD_PARTICLE_POSITION, fpe_out);
        // The velocity field must also resolve + be finite on a coupled world.
        const auto vel = DownloadField(w.handle, NUKA_FIELD_PARTICLE_VELOCITY);
        EXPECT_FALSE(vel.empty());
        EXPECT_TRUE(AllFinite(vel));
        return true;
    };

    std::vector<float> at_feet, control;
    size_t fpe = 0u, fpe_c = 0u;
    ASSERT_TRUE(run(/*at_feet=*/true, &at_feet, &fpe));
    ASSERT_TRUE(run(/*at_feet=*/false, &control, &fpe_c));
    ASSERT_EQ(fpe, 3u);
    ASSERT_FALSE(at_feet.empty());
    ASSERT_EQ(at_feet.size(), control.size())
        << "the two coupled worlds cooked a different particle count";
    EXPECT_TRUE(AllFinite(at_feet)) << "coupled-world particles went non-finite";
    EXPECT_TRUE(AllFinite(control)) << "control-world particles went non-finite";

    // The two worlds are IDENTICAL except a constant kSinkOffset on the control's
    // media (and floor). WITHOUT coupling the at-feet and control media would settle
    // to the SAME floor-relative state; comparing the fluid surface RELATIVE to each
    // world's floor cancels the offset, so a residual difference is the robot foot
    // displacing the at-feet pool (the two-way reaction through the C step path). The
    // cloth is perimeter-pinned in both, so the fluid slice is the free-medium signal.
    const size_t n_cloth = 13u * 13u;            // the soft slice [0, n_cloth).
    const size_t n_total = at_feet.size() / 3u;  // [n_cloth, n_total) is the fluid.
    const float at_feet_floor = 0.2545f - 0.105f;          // MediaDesc fluid_floor_z.
    const float control_floor = at_feet_floor + kSinkOffset;
    const float at_feet_surf =
        MaxZInSlice(at_feet, n_cloth, n_total) - at_feet_floor;
    const float control_surf =
        MaxZInSlice(control, n_cloth, n_total) - control_floor;
    double l1 = 0.0;  // floor-relative per-particle L1 over the fluid slice.
    for (size_t i = n_cloth; i < n_total; ++i) {
        const double a = at_feet[i * 3u + 2u] - at_feet_floor;
        const double c = control[i * 3u + 2u] - control_floor;
        l1 += std::fabs(a - c);
    }
    std::printf("[coupled] particles=%zu fluid=%zu at_feet_surf(rel)=%.4f "
                "control_surf(rel)=%.4f fluid_floor_rel_L1=%.4f\n",
                n_total, n_total - n_cloth, at_feet_surf, control_surf, l1);

    // The at-feet fluid's floor-relative state differs from the undisturbed control
    // -> the robot foot displaced the pool through the body<->particle coupling.
    EXPECT_GT(l1, 0.02)
        << "the at-feet fluid settled identically to the undisturbed control -> "
           "no body<->particle coupling acted through the C step path";
}

// (2) PART A: a coupled world WITH a baked heightfield (terrain) + particles builds
// + steps stably and serves + moves the particle field. The terrain budget recompute
// re-applies the additive body<->particle reserve so the slot ranges stay disjoint.
TEST(CoupledWorldCAbi, TerrainPlusParticlesKeepsBudgetDisjoint) {
    if (!SceneAvailable()) GTEST_SKIP() << "Go2 stand scene is not available";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const std::string scene = ScenePath();
    WorldGuard w;
    nuka_world_desc_t d = WorldDesc(scene, 1u);  // contact_family == 1: bake terrain.
    d.heightfield_terrain_type = 0u;             // flat heightfield collidable.
    const nuka_coupled_particles_desc_t p = MediaDesc(/*at_feet=*/true);
    const nuka_result_t rc =
        nuka_world_create_coupled_from_scene(device.handle, &d, &p, &w.handle);
    ASSERT_EQ(rc, NUKA_RESULT_OK) << "terrain+particle coupled create failed: "
                                  << nuka_result_message(rc);
    ASSERT_NE(w.handle, nullptr);

    // The particle field is served (the reserve was NOT clobbered by the recompute).
    nuka_buffer_view_t pv{};
    ASSERT_EQ(nuka_world_get_buffer_view(w.handle, NUKA_FIELD_PARTICLE_POSITION, &pv),
              NUKA_RESULT_OK);
    ASSERT_NE(pv.device_ptr, nullptr);
    const size_t particle_count = pv.element_count;
    ASSERT_GT(particle_count, 0u);

    const auto before = DownloadField(w.handle, NUKA_FIELD_PARTICLE_POSITION);
    const uint32_t kSteps = 200u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK)
            << "terrain+particle world step failed at " << s
            << " (slot corruption would surface here)";
    }
    const auto after = DownloadField(w.handle, NUKA_FIELD_PARTICLE_POSITION);

    // Stable: particles + robot joints stay finite (no NaN/blowup from a clobbered
    // or overlapping slot reserve).
    EXPECT_TRUE(AllFinite(after)) << "terrain+particle world went non-finite";
    const auto q = DownloadField(w.handle, NUKA_FIELD_JOINT_POSITION);
    EXPECT_TRUE(AllFinite(q)) << "the robot joints went non-finite on terrain+media";

    // The rigid contact field still resolves (the robot collides the terrain on the
    // SAME contact buffer, disjoint from the particle reserve).
    nuka_buffer_view_t cv{};
    EXPECT_EQ(nuka_world_get_buffer_view(w.handle, NUKA_FIELD_CONTACT_POINTS, &cv),
              NUKA_RESULT_OK);
    EXPECT_NE(cv.device_ptr, nullptr);

    // The media settled rather than NaN-ing or vanishing: at least one particle
    // moved (gravity + contact acted on a live, uncorrupted particle buffer).
    double moved = 0.0;
    for (size_t i = 0; i < after.size() && i < before.size(); ++i)
        moved += std::fabs(static_cast<double>(after[i]) - before[i]);
    std::printf("[coupled-terrain] particles=%zu contact_slots=%zu state_moved_L1=%.4f\n",
                particle_count, cv.element_count, moved);
    EXPECT_GT(moved, 0.0)
        << "particles never moved -> the particle buffer may be unreached/clobbered";
}

// (3) THE DEFAULT (contact_radius == 0) STILL COUPLES: the entry derives the contact
// diameter from the smaller medium spacing when 0, so a defaults-only coupled world
// displaces the pool rather than passing through it (the body<->particle op is live).
TEST(CoupledWorldCAbi, DefaultContactRadiusStillCouples) {
    if (!SceneAvailable()) GTEST_SKIP() << "Go2 stand scene is not available";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const std::string scene = ScenePath();
    size_t particles = 0u;
    const double l1 =
        CoupledMediaRefRelL1(device.handle, scene, /*contact_radius=*/0.0f,
                             /*steps=*/200u, &particles);
    std::printf("[coupled-default] particles=%zu media_ref_rel_L1=%.4f\n", particles,
                l1);
    ASSERT_GE(l1, 0.0) << "the default-radius coupled world failed to build/step";
    // Pre-fix, contact_radius == 0 left the body<->particle op inert (L1 ~ 0); the
    // spacing-derived default now produces a live two-way displacement.
    EXPECT_GT(l1, 0.01)
        << "contact_radius == 0 produced no coupling -> the documented spacing-derived "
           "default is not wired (the robot passes through the medium)";
}

// (4) BATCHED (env_count > 1): a coupled world cooks + steps N independent envs on the
// ONE pipeline. The particle field is N*P env-major and every env is bit-identical (a
// coupled world is just more bodies/particles on the same batched, D1-replicated path).
TEST(CoupledWorldCAbi, BatchedCoupledWorldReplicatesAcrossEnvs) {
    if (!SceneAvailable()) GTEST_SKIP() << "Go2 stand scene is not available";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvs = 4u;
    const std::string scene = ScenePath();  // outlive d.scene_path (a borrowed c_str).
    WorldGuard w;
    nuka_world_desc_t d = WorldDesc(scene, 0u);
    d.env_count = kEnvs;
    const nuka_coupled_particles_desc_t p = MediaDesc(/*at_feet=*/true);
    ASSERT_EQ(nuka_world_create_coupled_from_scene(device.handle, &d, &p, &w.handle),
              NUKA_RESULT_OK)
        << "a batched (env_count > 1) coupled world failed to create";

    nuka_buffer_view_t pv{};
    ASSERT_EQ(nuka_world_get_buffer_view(w.handle, NUKA_FIELD_PARTICLE_POSITION, &pv),
              NUKA_RESULT_OK);
    ASSERT_EQ(pv.element_count % kEnvs, 0u);
    const size_t per_env = pv.element_count / kEnvs;
    ASSERT_GT(per_env, 0u);

    for (uint32_t s = 0; s < 120u; ++s) {
        ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK)
            << "batched coupled world step failed at " << s;
    }
    const auto pos = DownloadField(w.handle, NUKA_FIELD_PARTICLE_POSITION);
    ASSERT_EQ(pos.size(), pv.element_count * 3u);
    EXPECT_TRUE(AllFinite(pos)) << "a batched coupled env went non-finite";

    // Every env saw the SAME scene + media at the SAME pose, so the per-env particle
    // slices must be bit-identical -- the batched path is D1-deterministic, not a
    // per-env special-case (mirrors BatchedArticulationReplication for the rigid side).
    size_t mismatches = 0u;
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        for (size_t i = 0; i < per_env * 3u; ++i) {
            if (pos[i] != pos[e * per_env * 3u + i]) ++mismatches;
        }
    }
    std::printf("[coupled-batched] envs=%u per_env=%zu cross_env_mismatches=%zu\n",
                kEnvs, per_env, mismatches);
    EXPECT_EQ(mismatches, 0u)
        << "batched coupled envs diverged -> the coupled path is not D1-replicated";
}
