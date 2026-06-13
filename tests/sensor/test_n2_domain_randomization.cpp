// ---------------------------------------------------------------------------
// p04-N2 -- per-episode domain randomization (engine-side, exit #5/#6 DR half).
// ---------------------------------------------------------------------------
//
// Covers the four spec gates:
//   (1) Deterministic sampling: same (seed, env) -> bit-exact multipliers;
//       different seed OR env -> different; all within [lo, hi].
//   (2) Application lands: after apply, link masses / gravity.z actually changed
//       by the sampled multiplier/offset (read back + verify).
//   (3) DR OFF default = no-op: a DR-disabled world is byte-unchanged by apply.
//   (4) D1 two-run byte-exact BACKWARD with DR ON (the advisor-flagged headline):
//       a single-env world with DR on (fixed seed) + apply, then a diff-sim tape
//       rollout + backward; repeated with a FRESH world + SAME seed -> grad_actions
//       + grad_parameters are byte-identical (memcmp == 0). AND DR-on grad !=
//       DR-off grad (proves it actually applied, not a silent no-op): the contact-
//       free tape responds to mass + gravity DR.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"
#include "nuka/nuka_diffsim.h"
#include "nuka/nuka_noise.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "nk/pipeline/world.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "sensor/noise/n2_domain_randomization.hpp"
#include "sensor/noise/noise_config.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace noise = nuka::sensor::noise;
namespace articulation = nuka::runtime::articulation;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

std::string Go2FloatScenePath() {
    return SourcePath("examples/scenes/go2_float.usda").string();
}

bool Go2FloatSceneAvailable() {
    return std::filesystem::exists(SourcePath("examples/scenes/go2_float.usda"));
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

struct TapeGuard {
    nuka_tape_handle handle = nullptr;
    ~TapeGuard() {
        if (handle != nullptr) nuka_tape_destroy(handle);
    }
};

nuka_result_t CreateFloatWorld(nuka_device_handle device,
                               nuka_world_handle* out) {
    const std::string scene = Go2FloatScenePath();
    nuka_world_desc_t desc{};
    desc.scene_path = scene.c_str();
    desc.env_count = 1u;
    desc.fixed_dt = 0.005f;
    return nuka_world_create_from_scene(device, &desc, out);
}

// A DR desc with all ranges set wide + fixed seed. mass + gravity drive the
// contact-free tape; friction/restitution/armature are sampled but inert there.
nuka_domain_randomization_desc_t MakeDrDesc(uint64_t seed, int enabled) {
    nuka_domain_randomization_desc_t d{};
    d.mass_mul_lo = 0.8f;
    d.mass_mul_hi = 1.2f;
    d.friction_mul_lo = 0.5f;
    d.friction_mul_hi = 1.5f;
    d.restitution_off_lo = -0.1f;
    d.restitution_off_hi = 0.1f;
    d.armature_off_lo = 0.0f;
    d.armature_off_hi = 0.05f;
    d.gravity_off_lo = -0.5f;
    d.gravity_off_hi = 0.5f;
    d.seed = seed;
    d.enabled = enabled;
    return d;
}

// Reads a global link's live device-buffer scalar mass (I[21] == the [3,3]
// block of the 6x6 spatial inertia == mass).
float DownloadLinkMass(nuka_world_handle world, uint32_t link) {
    nuka::c_abi::WorldRecord* record = nuka::c_abi::WorldTable().Get(world);
    EXPECT_NE(record, nullptr);
    // M9: the live link_inertia lives in the nk arena (the DR poke writes the
    // LinkInertia field). Read it through the SAME seam the c_abi uses.
    const uint32_t total_link_count =
        record->articulation_host.TotalLinkCount();
    const uint32_t articulation_count =
        record->articulation_host.ArticulationCount();
    const auto state = articulation::MakeArticulationDeviceStateFromViews(
        record->world->ModelViewRef(), record->world->DataViewRef(),
        total_link_count, articulation_count);
    articulation::LinkSpatialInertia inertia;
    cudaMemcpy(inertia.I, state.link_inertia[link].I, 36u * sizeof(float),
               cudaMemcpyDeviceToHost);
    return inertia.I[21];
}

float WorldGravityZ(nuka_world_handle world) {
    nuka::c_abi::WorldRecord* record = nuka::c_abi::WorldTable().Get(world);
    EXPECT_NE(record, nullptr);
    return record->step_options.gravity.z;
}

uint32_t WorldLinkCount(nuka_world_handle world) {
    nuka::c_abi::WorldRecord* record = nuka::c_abi::WorldTable().Get(world);
    EXPECT_NE(record, nullptr);
    return record->articulation_host.TotalLinkCount();
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Runs DR (set + apply BEFORE tape_create -- gravity must be set before create,
// mass takes effect before the first step) then a contact-free tape rollout +
// backward, returning [grad_actions | grad_parameters]. `dr_enabled == 0` skips
// the DR (the DR-off baseline). Fresh world per call -> identical init.
std::vector<float> RunTapeWithDr(nuka_device_handle device, uint32_t n_steps,
                                 uint64_t seed, int dr_enabled) {
    WorldGuard world;
    EXPECT_EQ(CreateFloatWorld(device, &world.handle), NUKA_RESULT_OK);

    if (dr_enabled) {
        const nuka_domain_randomization_desc_t d = MakeDrDesc(seed, 1);
        EXPECT_EQ(nuka_world_set_domain_randomization(world.handle, &d),
                  NUKA_RESULT_OK);
        EXPECT_EQ(nuka_world_apply_domain_randomization(world.handle),
                  NUKA_RESULT_OK);
    }

    nuka_tape_desc_t desc{};
    desc.checkpoint_interval = 5u;
    desc.max_tape_entries = 128u;
    desc.max_checkpoints = 64u;
    desc.recompute_on_backward = 1u;
    TapeGuard tape;
    EXPECT_EQ(nuka_tape_create(world.handle, &desc, &tape.handle), NUKA_RESULT_OK);

    const uint32_t n = nuka_tape_link_count(tape.handle);
    EXPECT_GT(n, 0u);

    for (uint32_t s = 0u; s < n_steps; ++s) {
        EXPECT_EQ(nuka_world_step_with_tape(world.handle, tape.handle),
                  NUKA_RESULT_OK);
    }
    const uint32_t step_count = nuka_tape_step_count(tape.handle);
    EXPECT_EQ(step_count, n_steps);

    std::vector<float> seed_vec(2u * n + 6u * n, 0.0f);
    for (uint32_t i = 0u; i < n; ++i) seed_vec[n + i] = 0.3f;  // dL/dqdot'

    std::vector<float> grad_actions(static_cast<size_t>(step_count) * n, 0.0f);
    std::vector<float> grad_params(n, 0.0f);
    EXPECT_EQ(nuka_tape_backward(tape.handle, seed_vec.data(),
                                 grad_actions.data(), grad_params.data()),
              NUKA_RESULT_OK);

    std::vector<float> out = grad_actions;
    out.insert(out.end(), grad_params.begin(), grad_params.end());
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) Deterministic sampling: same (seed, env) bit-exact; different -> different;
//     all multipliers within their declared [lo, hi] ranges.
// ---------------------------------------------------------------------------
TEST(N2DomainRandomization, SamplingDeterministicAndInRange) {
    noise::DomainRandomizationConfig cfg;
    cfg.enabled = true;
    cfg.seed = 0xC0FFEEu;

    const noise::SampledRandomization a0 =
        noise::SampleEpisodeRandomization(cfg, 0u);
    const noise::SampledRandomization a0_again =
        noise::SampleEpisodeRandomization(cfg, 0u);
    // Same seed + same env -> bit-exact.
    EXPECT_EQ(std::memcmp(&a0, &a0_again, sizeof(a0)), 0)
        << "same (seed, env) must produce bit-identical samples";

    const noise::SampledRandomization a1 =
        noise::SampleEpisodeRandomization(cfg, 1u);
    // Different env -> different (at least one field differs).
    EXPECT_NE(std::memcmp(&a0, &a1, sizeof(a0)), 0)
        << "different env must produce different samples";

    noise::DomainRandomizationConfig cfg2 = cfg;
    cfg2.seed = 0xBADBEEFu;
    const noise::SampledRandomization b0 =
        noise::SampleEpisodeRandomization(cfg2, 0u);
    EXPECT_NE(std::memcmp(&a0, &b0, sizeof(a0)), 0)
        << "different seed must produce different samples";

    // All within their declared ranges (default spec ranges).
    for (uint32_t env = 0u; env < 64u; ++env) {
        const noise::SampledRandomization s =
            noise::SampleEpisodeRandomization(cfg, env);
        EXPECT_GE(s.mass_multiplier, cfg.mass_multiplier.lo);
        EXPECT_LE(s.mass_multiplier, cfg.mass_multiplier.hi);
        EXPECT_GE(s.friction_multiplier, cfg.friction_multiplier.lo);
        EXPECT_LE(s.friction_multiplier, cfg.friction_multiplier.hi);
        EXPECT_GE(s.restitution_offset, cfg.restitution_offset.lo);
        EXPECT_LE(s.restitution_offset, cfg.restitution_offset.hi);
        EXPECT_GE(s.joint_armature_offset, cfg.joint_armature_offset.lo);
        EXPECT_LE(s.joint_armature_offset, cfg.joint_armature_offset.hi);
        EXPECT_GE(s.gravity_z_offset, cfg.gravity_z_offset.lo);
        EXPECT_LE(s.gravity_z_offset, cfg.gravity_z_offset.hi);
    }

    // Disabled config -> identity (mult==1, offset==0), no sampling.
    noise::DomainRandomizationConfig off;  // enabled defaults false
    const noise::SampledRandomization id =
        noise::SampleEpisodeRandomization(off, 7u);
    EXPECT_EQ(id.mass_multiplier, 1.0f);
    EXPECT_EQ(id.friction_multiplier, 1.0f);
    EXPECT_EQ(id.restitution_offset, 0.0f);
    EXPECT_EQ(id.joint_armature_offset, 0.0f);
    EXPECT_EQ(id.gravity_z_offset, 0.0f);
}

// ---------------------------------------------------------------------------
// (2) Application lands: link masses scale by the sampled multiplier and
//     gravity.z shifts by the sampled offset.
// ---------------------------------------------------------------------------
TEST(N2DomainRandomization, ApplicationLandsOnMassAndGravity) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    WorldGuard world;
    ASSERT_EQ(CreateFloatWorld(device.handle, &world.handle), NUKA_RESULT_OK);

    const uint32_t n = WorldLinkCount(world.handle);
    ASSERT_GT(n, 0u);

    // Snapshot nominal masses + gravity BEFORE apply.
    std::vector<float> nominal_mass(n, 0.0f);
    for (uint32_t l = 0u; l < n; ++l)
        nominal_mass[l] = DownloadLinkMass(world.handle, l);
    const float nominal_gz = WorldGravityZ(world.handle);

    const uint64_t seed = 424242u;
    const nuka_domain_randomization_desc_t d = MakeDrDesc(seed, 1);
    ASSERT_EQ(nuka_world_set_domain_randomization(world.handle, &d),
              NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_apply_domain_randomization(world.handle),
              NUKA_RESULT_OK);

    // The single-env world is env 0; recompute the expected sample.
    noise::DomainRandomizationConfig cfg;
    cfg.enabled = true;
    cfg.seed = seed;
    cfg.mass_multiplier = {d.mass_mul_lo, d.mass_mul_hi};
    cfg.gravity_z_offset = {d.gravity_off_lo, d.gravity_off_hi};
    const noise::SampledRandomization s =
        noise::SampleEpisodeRandomization(cfg, 0u);

    // gravity.z = nominal + offset.
    EXPECT_NEAR(WorldGravityZ(world.handle), nominal_gz + s.gravity_z_offset,
                1e-5f);

    // Each link mass = nominal * multiplier (links with nominal>0; massless /
    // fixed links stay 0). At least one link must have actually scaled.
    bool any_scaled = false;
    for (uint32_t l = 0u; l < n; ++l) {
        const float expect = nominal_mass[l] * s.mass_multiplier;
        const float got = DownloadLinkMass(world.handle, l);
        EXPECT_NEAR(got, expect, 1e-4f * (1.0f + std::fabs(expect)))
            << "link " << l << " mass mismatch after DR apply";
        if (nominal_mass[l] > 0.0f &&
            std::fabs(got - nominal_mass[l]) > 1e-6f) {
            any_scaled = true;
        }
    }
    EXPECT_TRUE(any_scaled) << "no link mass actually changed (DR silent no-op)";
    std::printf("[n2 dr] mass_mult=%.6f gravity_off=%.6f (nominal gz=%.4f)\n",
                static_cast<double>(s.mass_multiplier),
                static_cast<double>(s.gravity_z_offset),
                static_cast<double>(nominal_gz));

    // Idempotence: a second apply with the SAME seed re-randomizes around nominal
    // (no compounding) -> masses unchanged from the first apply.
    ASSERT_EQ(nuka_world_apply_domain_randomization(world.handle),
              NUKA_RESULT_OK);
    for (uint32_t l = 0u; l < n; ++l) {
        const float expect = nominal_mass[l] * s.mass_multiplier;
        EXPECT_NEAR(DownloadLinkMass(world.handle, l), expect,
                    1e-4f * (1.0f + std::fabs(expect)))
            << "link " << l << " mass compounded across applies (not idempotent)";
    }
}

// ---------------------------------------------------------------------------
// (3) DR OFF default = byte no-op: a DR-disabled world's masses + gravity are
//     byte-unchanged by apply (V1 oracle safe).
// ---------------------------------------------------------------------------
TEST(N2DomainRandomization, DisabledApplyIsByteNoOp) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    WorldGuard world;
    ASSERT_EQ(CreateFloatWorld(device.handle, &world.handle), NUKA_RESULT_OK);

    const uint32_t n = WorldLinkCount(world.handle);
    ASSERT_GT(n, 0u);
    std::vector<float> before(n, 0.0f);
    for (uint32_t l = 0u; l < n; ++l) before[l] = DownloadLinkMass(world.handle, l);
    const float gz_before = WorldGravityZ(world.handle);

    // Default world (no set) -> apply must be a no-op.
    ASSERT_EQ(nuka_world_apply_domain_randomization(world.handle), NUKA_RESULT_OK);
    // Explicitly disabled desc -> still a no-op.
    const nuka_domain_randomization_desc_t off = MakeDrDesc(999u, 0);
    ASSERT_EQ(nuka_world_set_domain_randomization(world.handle, &off),
              NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_apply_domain_randomization(world.handle), NUKA_RESULT_OK);

    for (uint32_t l = 0u; l < n; ++l) {
        uint32_t a, b;
        const float now = DownloadLinkMass(world.handle, l);
        std::memcpy(&a, &before[l], 4);
        std::memcpy(&b, &now, 4);
        EXPECT_EQ(a, b) << "link " << l << " mass changed by a DISABLED apply";
    }
    EXPECT_EQ(WorldGravityZ(world.handle), gz_before)
        << "gravity changed by a DISABLED apply";
}

// ---------------------------------------------------------------------------
// (4) ★ D1 two-run byte-exact BACKWARD with DR ON, AND DR-on grad != DR-off grad.
// ---------------------------------------------------------------------------
TEST(N2DomainRandomization, BackwardBitIdenticalTwoRunWithDrOn) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint64_t seed = 0x5EEDu;
    const uint32_t n_steps = 16u;

    // Two FRESH worlds, SAME seed, DR ON -> backward must be byte-identical.
    const std::vector<float> dr_on_1 =
        RunTapeWithDr(device.handle, n_steps, seed, /*dr_enabled=*/1);
    const std::vector<float> dr_on_2 =
        RunTapeWithDr(device.handle, n_steps, seed, /*dr_enabled=*/1);
    ASSERT_EQ(dr_on_1.size(), dr_on_2.size());

    size_t mismatches = 0u;
    for (size_t i = 0u; i < dr_on_1.size(); ++i)
        if (dr_on_1[i] != dr_on_2[i]) ++mismatches;
    std::printf("[n2 dr] D1 two-run (DR ON) mismatches=%zu of %zu\n", mismatches,
                dr_on_1.size());
    EXPECT_EQ(mismatches, 0u)
        << "DR-on backward not bit-identical across two runs (D1 broken)";
    EXPECT_TRUE(BitEqual(dr_on_1, dr_on_2));

    // DR OFF baseline -> proves DR actually changed the gradient (not a silent
    // no-op): the contact-free tape responds to mass + gravity DR.
    const std::vector<float> dr_off =
        RunTapeWithDr(device.handle, n_steps, seed, /*dr_enabled=*/0);
    ASSERT_EQ(dr_off.size(), dr_on_1.size());
    EXPECT_FALSE(BitEqual(dr_on_1, dr_off))
        << "DR-on grad == DR-off grad: DR did not actually apply to the rollout";

    float max_diff = 0.0f;
    for (size_t i = 0u; i < dr_off.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(dr_on_1[i] - dr_off[i]));
    std::printf("[n2 dr] max|grad_on - grad_off|=%.6e (proves DR applied)\n",
                static_cast<double>(max_diff));
    EXPECT_GT(max_diff, 0.0f);

    // DR OFF must itself be D1 byte-exact across two runs (sanity).
    const std::vector<float> dr_off_2 =
        RunTapeWithDr(device.handle, n_steps, seed, /*dr_enabled=*/0);
    EXPECT_TRUE(BitEqual(dr_off, dr_off_2))
        << "DR-off backward not bit-identical across two runs";
}
