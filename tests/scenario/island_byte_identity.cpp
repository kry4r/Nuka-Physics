// ---------------------------------------------------------------------------
// Dynamic connected-component island solve vs the conservative cook-time
// one-island-per-env schedule must be BYTE-IDENTICAL. The original gate only
// exercised always-grounded standing dogs; these cases drive the regimes it
// skipped, where the dynamic schedule drops an articulation tile out of every
// island mid-run:
//   * a PairDriven articulation that GAINS then LOSES contact under the
//     split-impulse position pass (pos_iters>0) -- the dropped tile must read a
//     zeroed split-impulse pseudo-velocity, not the prior step's push-out;
//   * a MULTI-ENV world with MIXED contact states across envs.
// NUKA_FORCE_STATIC_ISLANDS=1 forces the static reference; the comparison asserts
// the persisted state (base pose + q/qdot + per-link spatial velocity) matches
// bit-for-bit across the dynamic and the static schedule.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;

std::filesystem::path Go2FloatScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_float.usda";
}
bool HaveAsset() { return std::filesystem::exists(Go2FloatScenePath()); }

// K floating-base go2 composed at distinct X into ONE env template via the GENERAL
// PairDriven family. spacing < trunk width -> the trunks overlap and contact.
nk::Model CookKFloatDogs(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2FloatScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        Transform place = Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(
            scene, nuka::import::LoadUsd(Go2FloatScenePath().string()), place,
            "dog" + std::to_string(i) + "_");
    }
    nuka::scene::cook::CookToModelOptions opt;
    opt.contact_family = nuka::scene::cook::CookContactFamily::PairDriven;
    return nuka::scene::cook::CookToModel(scene, 1, opt).model;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = 0.0f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.pos_iters = 28u;  // the coupling-demo split-impulse depth (the #1/#4 surface).
    cfg.max_pairs = 64u;
    return cfg;
}

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

// The PERSISTED state the dynamic schedule could perturb, every env concatenated:
// base pose (7 f / artic), q + qdot (1 f / link), link spatial velocity (6 f / link).
std::vector<uint8_t> DumpState(nk::World& w, uint32_t artics_total,
                               uint32_t links_total) {
    std::vector<uint8_t> out;
    auto grab = [&](nk::FieldId id, size_t bytes) {
        const size_t off = out.size();
        out.resize(off + bytes);
        EXPECT_TRUE(w.GetData().DownloadField(id, out.data() + off, bytes));
    };
    grab(nk::FieldId::BasePose, static_cast<size_t>(artics_total) * 7u * sizeof(float));
    grab(nk::FieldId::Q, static_cast<size_t>(links_total) * sizeof(float));
    grab(nk::FieldId::Qdot, static_cast<size_t>(links_total) * sizeof(float));
    grab(nk::FieldId::LinkVelocity,
         static_cast<size_t>(links_total) * 6u * sizeof(float));
    return out;
}

// Build + step the scenario under the selected schedule (env var read at World
// construction), then dump the full persisted state. perturb(base_pose) runs once
// after spawn to set per-env initial conditions (mixed contact states).
template <typename Perturb>
std::vector<uint8_t> RunScenario(bool force_static, uint32_t k, float spacing,
                                 uint32_t env_count, uint32_t steps,
                                 Perturb perturb) {
    setenv("NUKA_FORCE_STATIC_ISLANDS", force_static ? "1" : "0", 1);
    Backend b = GetBackend();
    nk::Model model = CookKFloatDogs(k, spacing);
    const uint32_t artics_total = model.capacities.articulations_per_env * env_count;
    const uint32_t links_total = model.capacities.links_per_env * env_count;
    nk::World world(std::move(model), env_count, b.dev, b.backend, Cfg());
    EXPECT_TRUE(world.Ready());

    std::vector<float> base(static_cast<size_t>(artics_total) * 7u, 0.0f);
    EXPECT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base.data(),
                                              base.size() * sizeof(float)));
    perturb(base);
    EXPECT_TRUE(world.GetData().UploadField(nk::FieldId::BasePose, base.data(),
                                            base.size() * sizeof(float)));

    for (uint32_t s = 0; s < steps; ++s) EXPECT_TRUE(world.Step().AllOk()) << s;
    return DumpState(world, artics_total, links_total);
}

}  // namespace

// Trunks overlap at spawn -> contact (a nonzero split-impulse push-out is scattered)
// -> they push apart and fully separate -> every tile is contact-free (dropped from
// the dynamic schedule). The dropped tile must integrate from a ZEROED pseudo, exactly
// as the static all-tiles scatter left it; the two schedules must agree bit-for-bit.
TEST(IslandByteIdentity, ContactThenSeparateMatchesStaticSchedule) {
    if (!HaveAsset()) GTEST_SKIP() << "go2_float.usda not present";
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    auto noop = [](std::vector<float>&) {};
    const std::vector<uint8_t> dyn = RunScenario(false, 3u, 0.12f, 1u, 160u, noop);
    const std::vector<uint8_t> sta = RunScenario(true, 3u, 0.12f, 1u, 160u, noop);
    unsetenv("NUKA_FORCE_STATIC_ISLANDS");
    ASSERT_EQ(dyn.size(), sta.size());
    EXPECT_EQ(std::memcmp(dyn.data(), sta.data(), dyn.size()), 0)
        << "dynamic island solve diverged from the static schedule after a "
           "contact->airborne transition with pos_iters>0";
}

// env_count>1 with MIXED contact: env 0 keeps the overlap (contact), env>=1 shifts its
// trailing dogs far in +X (no contact ever). The dynamic schedule's per-env component
// derivation must reproduce the static per-env island bit-for-bit across all envs.
TEST(IslandByteIdentity, MultiEnvMixedContactMatchesStaticSchedule) {
    if (!HaveAsset()) GTEST_SKIP() << "go2_float.usda not present";
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const uint32_t k = 3u, env_count = 3u;
    // Push every non-first dog of env>=1 far apart so those envs never contact.
    auto perturb = [&](std::vector<float>& base) {
        const uint32_t artics_per_env = k;
        for (uint32_t e = 1; e < env_count; ++e) {
            for (uint32_t a = 1; a < artics_per_env; ++a) {
                const size_t row = (static_cast<size_t>(e) * artics_per_env + a) * 7u;
                base[row + 0] += 5.0f * static_cast<float>(a);  // +X, well clear.
            }
        }
    };
    const std::vector<uint8_t> dyn = RunScenario(false, k, 0.12f, env_count, 160u, perturb);
    const std::vector<uint8_t> sta = RunScenario(true, k, 0.12f, env_count, 160u, perturb);
    unsetenv("NUKA_FORCE_STATIC_ISLANDS");
    ASSERT_EQ(dyn.size(), sta.size());
    EXPECT_EQ(std::memcmp(dyn.data(), sta.data(), dyn.size()), 0)
        << "dynamic island solve diverged from the static schedule in a multi-env "
           "world with mixed contact states";
}
