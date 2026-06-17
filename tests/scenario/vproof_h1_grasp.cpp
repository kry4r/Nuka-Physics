// ---------------------------------------------------------------------------
// V-PROOF — H1 hulls collide on the ONE GENERAL (PairDriven) contact path.
//
// L1-c DELETED the legacy UnionCsr grasp path (BuildNkUnionModel /
// CookSceneToUnionTemplate) and with it the LEGACY-vs-GENERAL cup-trajectory
// equivalence test (GeneralTracksUnionCupTrajectory) + its union-cook helpers.
// Cup-grasping moved to RL; the union grip was the known-failing gate. What
// SURVIVES here is the general-path proof: L-RECON-B made the general
// (PairDriven) cook of the whole-body H1 scene TRACTABLE (~0.07 s) and it emits
// convex Hull shape rows for H1's meshes; L-RECON-D wires a MULTI-hull pool so
// the cvx narrowphase collides EACH shape's own hull. This test PROVES that
// wiring end to end on the general path (it does NOT assert a working grasp-hold).
//
// Asset-gated (SKIPs without .nuka-assets).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;

// The H1 cup-table scene + the integrator constants the scene steps at (g=-9.81,
// dt=1/240). L1-c: these were the kH1Union* constants in the deleted
// union_scene_constants.hpp; inlined here so the surviving general-path proof is
// self-contained (no union include).
constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";
constexpr const char* kH1MjcfPath =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
constexpr const char* kH1CupPath =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;

bool AssetsAvailable() {
    return std::filesystem::exists(kNksPath) &&
           std::filesystem::exists(kH1MjcfPath) &&
           std::filesystem::exists(kH1CupPath);
}

nk::Pipeline::SolverConfig StepCfg(float dt) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = kGravityZ;
    cfg.vel_iters = 64;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;
    return cfg;
}

// ===========================================================================
// L-RECON-D — the H1 hulls now COLLIDE on the GENERAL path. L-RECON-B made the
// general (PairDriven) cook of the whole-body union scene TRACTABLE (~0.07 s) and
// it emits convex Hull shape rows for H1's meshes. L-RECON-D wires a MULTI-hull
// pool: each cooked Hull shape's verts are packed into the concatenated
// model.hull_verts and the shape's (hull_vert_offset, hull_vert_count) slice rides
// shape_table lanes 10/11, so the cvx narrowphase collides EACH shape's own hull.
//
// This probe PROVES the wiring end to end: it cooks the H1 scene via
// CookToModel(PairDriven), asserts the multi-hull pool is populated + every hull
// row's slice is in-bounds and finite, builds an nk::World, steps it a few times,
// and shows hull-involving contacts are GENERATED on the general path (contact
// count > 0 with finite points/normals/depths) — vs the empty-pool no-op before.
// It does NOT assert a working grasp-hold (that is later collapse validation); it
// asserts hull contacts LOAD, are finite, and are deterministic (two byte-equal
// runs). Asset-gated (SKIPs without .nuka-assets); the cook is now fast.
// ===========================================================================

// Cook the H1 union scene through the GENERAL pair-driven path (now tractable).
cook::CookToModelResult CookGeneralH1() {
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    return cook::CookToModel(scene, 1, opt);
}

// Sum the active contact points across every slot of env 0 + collect (finite?)
// over the active points' position/normal/depth. slot_stride == max_contacts_per_env.
struct ContactProbe {
    uint32_t total_points = 0u;
    bool all_finite = true;
    std::vector<uint32_t> counts;   // per slot (env 0)
    std::vector<float>    pts;      // 3 * 4 per slot (env 0)
    std::vector<float>    nrm;      // 3 * 4 per slot (env 0)
    std::vector<float>    dep;      // 4 per slot (env 0)
};

ContactProbe ReadContacts(nk::World& w, uint32_t slot_stride) {
    ContactProbe p;
    p.counts.assign(slot_stride, 0u);
    p.pts.assign(static_cast<size_t>(slot_stride) * 4u * 3u, 0.0f);
    p.nrm.assign(static_cast<size_t>(slot_stride) * 4u * 3u, 0.0f);
    p.dep.assign(static_cast<size_t>(slot_stride) * 4u, 0.0f);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactCount, p.counts.data(),
                                          p.counts.size() * sizeof(uint32_t)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactPoint, p.pts.data(),
                                          p.pts.size() * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactNormal, p.nrm.data(),
                                          p.nrm.size() * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactDepth, p.dep.data(),
                                          p.dep.size() * sizeof(float)));
    for (uint32_t s = 0; s < slot_stride; ++s) {
        const uint32_t n = p.counts[s] > 4u ? 4u : p.counts[s];
        p.total_points += p.counts[s];
        for (uint32_t i = 0; i < n; ++i) {
            const size_t v = (static_cast<size_t>(s) * 4u + i);
            for (uint32_t c = 0; c < 3u; ++c) {
                if (!std::isfinite(p.pts[v * 3u + c]) ||
                    !std::isfinite(p.nrm[v * 3u + c])) p.all_finite = false;
            }
            if (!std::isfinite(p.dep[v])) p.all_finite = false;
        }
    }
    return p;
}

}  // namespace

TEST(VProofH1Grasp, GeneralPathHullContactsLoad) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present "
                        "(asset-gated)";
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // ---- (1) the general cook now packs a MULTI-hull pool (L-RECON-D) ----------
    cook::CookToModelResult cooked = CookGeneralH1();
    nk::Model& m = cooked.model;
    ASSERT_EQ(m.contact_family, nk::ContactFamily::PairDriven);
    // The pool is non-empty (the H1 meshes cooked to convex hulls, L-RECON-B) and
    // its capacity matches the packed verts (the L-RECON-D fill).
    ASSERT_GT(m.hull_verts.size(), 0u) << "general cook packed NO hull verts — the "
        "multi-hull pool fill (L-RECON-D) did not run";
    EXPECT_EQ(m.capacities.max_hull_verts,
              static_cast<uint32_t>(m.hull_verts.size() / 3u));
    // Every hull shape row's slice is in-bounds + its verts are finite, and the
    // slices reference DISTINCT regions of the concatenated pool (not one global 0).
    uint32_t hull_rows = 0u;
    uint32_t max_off = 0u;
    for (const nk::Model::PairDrivenShape& sh : m.shape_table_rows) {
        if (sh.hull_vert_count == 0u) continue;
        ++hull_rows;
        max_off = std::max(max_off, sh.hull_vert_offset);
        const size_t end =
            (static_cast<size_t>(sh.hull_vert_offset) + sh.hull_vert_count) * 3u;
        ASSERT_LE(end, m.hull_verts.size())
            << "hull row slice out of pool bounds (off=" << sh.hull_vert_offset
            << " cnt=" << sh.hull_vert_count << ")";
        for (size_t i = static_cast<size_t>(sh.hull_vert_offset) * 3u; i < end; ++i)
            ASSERT_TRUE(std::isfinite(m.hull_verts[i])) << "non-finite hull vert";
    }
    ASSERT_GT(hull_rows, 0u) << "no shape row carries a hull slice";
    // More than one hull => the offsets MUST advance past 0 (the multi-hull proof:
    // a single global hull would leave every slice at offset 0).
    if (hull_rows > 1u)
        EXPECT_GT(max_off, 0u) << "all hull slices at offset 0 — pool is not multi-hull";
    std::fprintf(stderr,
        "[VPROOF h1-grasp L-RECON-D] general cook: bodies=%u shape_rows=%zu "
        "hull_rows=%u hull_verts=%zu (max_hull_verts=%u) max_hull_off=%u\n",
        m.capacities.bodies_per_env, m.shape_table_rows.size(), hull_rows,
        m.hull_verts.size(), m.capacities.max_hull_verts, max_off);

    const uint32_t slot_stride = m.capacities.max_contacts_per_env;
    ASSERT_GT(slot_stride, 0u);

    // ---- (2) build a World on the GENERAL path + step it; hull contacts LOAD ----
    nk::Model m_run = CookGeneralH1().model;   // a fresh cook (m is consumed below).
    nk::World w(std::move(m_run), 1u, dev, backend, StepCfg(kDt));
    ASSERT_TRUE(w.Ready()) << "general H1 world failed to construct";
    uint32_t stepped_contacts = 0u;
    bool stepped_finite = true;
    for (uint32_t s = 0; s < 8u; ++s) {
        ASSERT_TRUE(w.Step().AllOk()) << "general H1 step " << s;
        const ContactProbe p = ReadContacts(w, slot_stride);
        stepped_contacts = std::max(stepped_contacts, p.total_points);
        if (!p.all_finite) stepped_finite = false;
    }
    std::fprintf(stderr,
        "[VPROOF h1-grasp L-RECON-D] general world stepped: peak hull-path "
        "contact points over 8 steps = %u (finite=%s)\n",
        stepped_contacts, stepped_finite ? "yes" : "NO");
    EXPECT_TRUE(stepped_finite) << "general-path contacts had non-finite data";
    ASSERT_GT(stepped_contacts, 0u)
        << "NO contacts generated on the general path — the H1 hulls did NOT "
           "collide (the multi-hull narrowphase wiring is a no-op). This is the "
           "exact RED L-RECON-D must turn green.";

    // ---- (3) determinism: two fresh general worlds produce byte-equal contacts --
    auto run_once = [&](std::vector<uint32_t>* counts, std::vector<float>* pts,
                        std::vector<float>* nrm, std::vector<float>* dep) -> bool {
        nk::Model md = CookGeneralH1().model;
        nk::World wd(std::move(md), 1u, dev, backend, StepCfg(kDt));
        if (!wd.Ready()) return false;
        for (uint32_t s = 0; s < 8u; ++s)
            if (!wd.Step().AllOk()) return false;
        const ContactProbe p = ReadContacts(wd, slot_stride);
        *counts = p.counts; *pts = p.pts; *nrm = p.nrm; *dep = p.dep;
        return true;
    };
    std::vector<uint32_t> c0, c1;
    std::vector<float> p0, p1, n0, n1, d0, d1;
    ASSERT_TRUE(run_once(&c0, &p0, &n0, &d0));
    ASSERT_TRUE(run_once(&c1, &p1, &n1, &d1));
    EXPECT_EQ(0, std::memcmp(c0.data(), c1.data(), c0.size() * sizeof(uint32_t)))
        << "contact COUNTS differ across runs (general path not deterministic)";
    EXPECT_EQ(0, std::memcmp(p0.data(), p1.data(), p0.size() * sizeof(float)))
        << "contact POINTS differ across runs (general path not deterministic)";
    EXPECT_EQ(0, std::memcmp(n0.data(), n1.data(), n0.size() * sizeof(float)))
        << "contact NORMALS differ across runs (general path not deterministic)";
    EXPECT_EQ(0, std::memcmp(d0.data(), d1.data(), d0.size() * sizeof(float)))
        << "contact DEPTHS differ across runs (general path not deterministic)";

    nphi::BackendFree(backend);
}
