// ---------------------------------------------------------------------------
// Tests for v0.8 C1b: MJCF importer parsing of collision/contact attributes.
//
// Covers: explicit per-geom contact attrs, the contype=0 visual-only case,
// <default>-class <geom> inheritance, <contact><exclude>, <contact><pair>,
// untouched record defaults, and parse-twice determinism. Geoms are resolved
// BY NAME (the geom-name->ShapeId map is importer-internal, not on SceneIR), so
// the test never assumes parse-order indices.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

constexpr const char* kFixture = "tests/data/contact_filtering.xml";

// Resolve a geom by its authored name to its CollisionShapeRecord. ASSERTs the
// name exists (fails the test cleanly rather than returning a bogus shape).
const nuka::scene::CollisionShapeRecord& ShapeByName(const nuka::scene::SceneIR& scene,
                                                     const std::string& name) {
    for (const auto& s : scene.Shapes()) {
        if (s.name == name) {
            return s;
        }
    }
    ADD_FAILURE() << "no shape named '" << name << "'";
    return scene.Shapes().at(0);  // unreachable on success
}

nuka::scene::ShapeId ShapeIdByName(const nuka::scene::SceneIR& scene,
                                   const std::string& name) {
    return ShapeByName(scene, name).id;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Fully-explicit geom: every C1b attr parsed into the C1a fields.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, ExplicitGeomAttrsAreParsed) {
    const auto scene = nuka::import::LoadMjcf(kFixture);
    const auto& s = ShapeByName(scene, "g_explicit");

    EXPECT_EQ(s.contype, 2u);
    EXPECT_EQ(s.conaffinity, 4u);
    EXPECT_EQ(s.condim, uint8_t(1));
    EXPECT_EQ(s.collision_group, 3);
    EXPECT_EQ(s.priority, 5);
    EXPECT_FLOAT_EQ(s.solref[0], 0.01f);
    EXPECT_FLOAT_EQ(s.solref[1], 0.8f);
    EXPECT_FLOAT_EQ(s.solimp[0], 0.8f);
    EXPECT_FLOAT_EQ(s.solimp[1], 0.9f);
    EXPECT_FLOAT_EQ(s.solimp[2], 0.002f);
    EXPECT_FLOAT_EQ(s.solimp[3], 0.4f);
    EXPECT_FLOAT_EQ(s.solimp[4], 3.0f);
    EXPECT_FLOAT_EQ(s.margin, 0.01f);
    EXPECT_FLOAT_EQ(s.gap, 0.005f);
    EXPECT_FLOAT_EQ(s.solmix, 0.5f);
    // friction="0.7 0.1 0.01" -> only the first (tangential) component kept.
    EXPECT_FLOAT_EQ(s.friction_mu, 0.7f);
}

// ---------------------------------------------------------------------------
// 2. contype=0 visual-only (the h1_with_hand finger pattern) -- load-bearing:
//    confirms 0/0 is actually read, NOT the default 1/1.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, VisualOnlyContypeZeroIsRead) {
    const auto scene = nuka::import::LoadMjcf(kFixture);
    const auto& s = ShapeByName(scene, "g_visual");
    EXPECT_EQ(s.contype, 0u);
    EXPECT_EQ(s.conaffinity, 0u);
}

// ---------------------------------------------------------------------------
// 3. <default>-class <geom> inheritance.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, GeomDefaultClassInheritance) {
    const auto scene = nuka::import::LoadMjcf(kFixture);

    // g_class: in class "c", no own contact attrs -> inherits contype=8 + solref.
    const auto& gc = ShapeByName(scene, "g_class");
    EXPECT_EQ(gc.contype, 8u);
    EXPECT_FLOAT_EQ(gc.solref[0], 0.005f);
    EXPECT_FLOAT_EQ(gc.solref[1], 0.5f);
    // condim is NOT set in class "c" -> stays the C1a record default (3).
    EXPECT_EQ(gc.condim, uint8_t(3));
    // friction never specified anywhere -> the -1 inherit-material sentinel.
    EXPECT_FLOAT_EQ(gc.friction_mu, -1.0f);

    // g_class_override: in class "c" but sets its own contype -> own value wins;
    // solref still inherited from the class.
    const auto& go = ShapeByName(scene, "g_class_override");
    EXPECT_EQ(go.contype, 10u);
    EXPECT_FLOAT_EQ(go.solref[0], 0.005f);
    EXPECT_FLOAT_EQ(go.solref[1], 0.5f);
}

// ---------------------------------------------------------------------------
// 4. <contact><exclude> -> canonical (min,max) body pair stored.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, ExcludePairIsStoredCanonical) {
    const auto scene = nuka::import::LoadMjcf(kFixture);

    // Only the resolvable exclude is stored; the unknown-body one is skipped.
    const auto& pairs = scene.ExcludePairs();
    ASSERT_EQ(pairs.size(), 1u);

    const nuka::scene::BodyId b1 = scene.GetBody(0).id;  // "b1" (first body)
    const nuka::scene::BodyId b2 = scene.GetBody(1).id;  // "b2"
    ASSERT_EQ(scene.GetBody(0).name, "b1");
    ASSERT_EQ(scene.GetBody(1).name, "b2");

    const nuka::scene::BodyId lo = b1 < b2 ? b1 : b2;
    const nuka::scene::BodyId hi = b1 < b2 ? b2 : b1;
    EXPECT_EQ(pairs[0].first, lo);
    EXPECT_EQ(pairs[0].second, hi);
}

// ---------------------------------------------------------------------------
// 5. <contact><pair> -> resolved ShapeIds + override params.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, ContactPairOverrideIsStored) {
    const auto scene = nuka::import::LoadMjcf(kFixture);

    // Only the resolvable pair is stored; the unknown-geom one is skipped.
    const auto& pairs = scene.ContactPairs();
    ASSERT_EQ(pairs.size(), 1u);
    const auto& p = pairs[0];

    EXPECT_EQ(p.geom1, ShapeIdByName(scene, "g_explicit"));
    EXPECT_EQ(p.geom2, ShapeIdByName(scene, "g_default"));
    EXPECT_FLOAT_EQ(p.friction_mu, 0.5f);
    EXPECT_EQ(p.condim, uint8_t(1));
    // Unspecified <pair> attrs keep MuJoCo's concrete pair defaults (NOT -1).
    EXPECT_FLOAT_EQ(p.solref[0], 0.02f);
    EXPECT_FLOAT_EQ(p.solref[1], 1.0f);
    EXPECT_FLOAT_EQ(p.margin, 0.0f);
    EXPECT_FLOAT_EQ(p.gap, 0.0f);
}

// ---------------------------------------------------------------------------
// 6. A geom with no contact attrs leaves every C1a record default unchanged.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, UnspecifiedAttrsKeepRecordDefaults) {
    const auto scene = nuka::import::LoadMjcf(kFixture);
    const auto& s = ShapeByName(scene, "g_default");

    EXPECT_EQ(s.contype, 1u);
    EXPECT_EQ(s.conaffinity, 1u);
    EXPECT_EQ(s.collision_group, 0);
    EXPECT_EQ(s.condim, uint8_t(3));
    EXPECT_EQ(s.priority, 0);
    EXPECT_FLOAT_EQ(s.solmix, 1.0f);
    EXPECT_FLOAT_EQ(s.margin, 0.0f);
    EXPECT_FLOAT_EQ(s.gap, 0.0f);
    EXPECT_FLOAT_EQ(s.solref[0], 0.02f);
    EXPECT_FLOAT_EQ(s.solref[1], 1.0f);
    EXPECT_FLOAT_EQ(s.solimp[0], 0.9f);
    EXPECT_FLOAT_EQ(s.solimp[1], 0.95f);
    EXPECT_FLOAT_EQ(s.solimp[2], 0.001f);
    EXPECT_FLOAT_EQ(s.solimp[3], 0.5f);
    EXPECT_FLOAT_EQ(s.solimp[4], 2.0f);
    // friction never specified -> the -1 inherit-material sentinel preserved.
    EXPECT_FLOAT_EQ(s.friction_mu, -1.0f);
}

// ---------------------------------------------------------------------------
// 7. Determinism: parse the same MJCF twice -> identical contact results.
// ---------------------------------------------------------------------------

TEST(MjcfContactFiltering, ParseTwiceIsIdentical) {
    const auto a = nuka::import::LoadMjcf(kFixture);
    const auto b = nuka::import::LoadMjcf(kFixture);

    ASSERT_EQ(a.ShapeCount(), b.ShapeCount());
    for (std::size_t i = 0; i < a.Shapes().size(); ++i) {
        const auto& sa = a.Shapes()[i];
        const auto& sb = b.Shapes()[i];
        EXPECT_EQ(sa.name, sb.name);
        EXPECT_EQ(sa.contype, sb.contype);
        EXPECT_EQ(sa.conaffinity, sb.conaffinity);
        EXPECT_EQ(sa.collision_group, sb.collision_group);
        EXPECT_EQ(sa.condim, sb.condim);
        EXPECT_EQ(sa.priority, sb.priority);
        EXPECT_FLOAT_EQ(sa.solmix, sb.solmix);
        EXPECT_FLOAT_EQ(sa.margin, sb.margin);
        EXPECT_FLOAT_EQ(sa.gap, sb.gap);
        EXPECT_FLOAT_EQ(sa.solref[0], sb.solref[0]);
        EXPECT_FLOAT_EQ(sa.solref[1], sb.solref[1]);
        for (int k = 0; k < 5; ++k) {
            EXPECT_FLOAT_EQ(sa.solimp[k], sb.solimp[k]);
        }
        EXPECT_FLOAT_EQ(sa.friction_mu, sb.friction_mu);
    }

    ASSERT_EQ(a.ExcludePairs().size(), b.ExcludePairs().size());
    for (std::size_t i = 0; i < a.ExcludePairs().size(); ++i) {
        EXPECT_EQ(a.ExcludePairs()[i], b.ExcludePairs()[i]);
    }

    ASSERT_EQ(a.ContactPairs().size(), b.ContactPairs().size());
    for (std::size_t i = 0; i < a.ContactPairs().size(); ++i) {
        const auto& pa = a.ContactPairs()[i];
        const auto& pb = b.ContactPairs()[i];
        EXPECT_EQ(pa.geom1, pb.geom1);
        EXPECT_EQ(pa.geom2, pb.geom2);
        EXPECT_EQ(pa.condim, pb.condim);
        EXPECT_FLOAT_EQ(pa.friction_mu, pb.friction_mu);
    }
}
