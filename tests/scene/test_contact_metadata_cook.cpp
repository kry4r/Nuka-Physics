// ---------------------------------------------------------------------------
// Tests for v0.8 C1a: per-shape contact metadata cook + exclude-pair storage.
//
// Covers: defaults, explicit round-trip (incl. flattened solimp ordering),
// friction-resolution precedence, exclude-pair canonicalization, and the D1
// cook-twice field-by-field determinism gate.
// ---------------------------------------------------------------------------

#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

using namespace nuka::scene;

namespace {

// Add a body + a single primitive shape (no mesh => exactly one cooked row),
// returning the source CollisionShapeRecord by value so the caller can mutate
// contact fields before adding it.
CollisionShapeRecord MakeBoxShape(BodyId body) {
    CollisionShapeRecord s;
    s.body_id = body;
    s.type    = ShapeType::Box;
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Defaults
// ---------------------------------------------------------------------------

TEST(ContactMetadataCook, DefaultsMatchMujoco) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");
    scene.AddCollisionShape(MakeBoxShape(body));

    const auto blob = CookScene(scene);
    const auto& cp = blob.contact_params;

    ASSERT_EQ(blob.shape_count, 1u);
    ASSERT_EQ(cp.contypes.size(), blob.shape_count);

    EXPECT_EQ(cp.contypes[0], 1u);
    EXPECT_EQ(cp.conaffinities[0], 1u);
    EXPECT_EQ(cp.groups[0], 0);
    EXPECT_FLOAT_EQ(cp.solref0[0], 0.02f);
    EXPECT_FLOAT_EQ(cp.solref1[0], 1.0f);
    ASSERT_EQ(cp.solimp.size(), 5u);
    EXPECT_FLOAT_EQ(cp.solimp[0], 0.9f);
    EXPECT_FLOAT_EQ(cp.solimp[1], 0.95f);
    EXPECT_FLOAT_EQ(cp.solimp[2], 0.001f);
    EXPECT_FLOAT_EQ(cp.solimp[3], 0.5f);
    EXPECT_FLOAT_EQ(cp.solimp[4], 2.0f);
    EXPECT_EQ(cp.condims[0], uint8_t(3));
    EXPECT_EQ(cp.priorities[0], 0);
    EXPECT_FLOAT_EQ(cp.solmix[0], 1.0f);
    EXPECT_FLOAT_EQ(cp.margins[0], 0.0f);
    EXPECT_FLOAT_EQ(cp.gaps[0], 0.0f);

    // No material, no per-shape override => MuJoCo default μ = 1.0.
    EXPECT_FLOAT_EQ(cp.frictions[0], 1.0f);
}

// ---------------------------------------------------------------------------
// 2. Round-trip of explicit values (incl. flattened solimp ordering)
// ---------------------------------------------------------------------------

TEST(ContactMetadataCook, ExplicitValuesRoundTrip) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");

    auto s = MakeBoxShape(body);
    s.contype         = 4u;
    s.conaffinity     = 8u;
    s.collision_group = -3;
    s.solref[0]       = 0.05f;
    s.solref[1]       = 0.7f;
    s.solimp[0]       = 0.11f;
    s.solimp[1]       = 0.22f;
    s.solimp[2]       = 0.33f;
    s.solimp[3]       = 0.44f;
    s.solimp[4]       = 0.55f;
    s.friction_mu     = 0.42f;   // explicit override (>=0)
    s.priority        = 7;
    s.solmix          = 0.25f;
    s.margin          = 0.01f;
    s.gap             = 0.02f;
    s.condim          = uint8_t(1);
    scene.AddCollisionShape(std::move(s));

    const auto blob = CookScene(scene);
    const auto& cp = blob.contact_params;

    ASSERT_EQ(blob.shape_count, 1u);
    EXPECT_EQ(cp.contypes[0], 4u);
    EXPECT_EQ(cp.conaffinities[0], 8u);
    EXPECT_EQ(cp.groups[0], -3);
    EXPECT_FLOAT_EQ(cp.solref0[0], 0.05f);
    EXPECT_FLOAT_EQ(cp.solref1[0], 0.7f);

    // Flattened solimp: shape 0's slice is [0, 5), in authoring order.
    ASSERT_EQ(cp.solimp.size(), 5u);
    EXPECT_FLOAT_EQ(cp.solimp[0], 0.11f);
    EXPECT_FLOAT_EQ(cp.solimp[1], 0.22f);
    EXPECT_FLOAT_EQ(cp.solimp[2], 0.33f);
    EXPECT_FLOAT_EQ(cp.solimp[3], 0.44f);
    EXPECT_FLOAT_EQ(cp.solimp[4], 0.55f);

    EXPECT_FLOAT_EQ(cp.frictions[0], 0.42f);
    EXPECT_EQ(cp.priorities[0], 7);
    EXPECT_FLOAT_EQ(cp.solmix[0], 0.25f);
    EXPECT_FLOAT_EQ(cp.margins[0], 0.01f);
    EXPECT_FLOAT_EQ(cp.gaps[0], 0.02f);
    EXPECT_EQ(cp.condims[0], uint8_t(1));
}

// Two shapes: confirm each shape's flattened solimp lands in its OWN [i*5,i*5+5)
// slice and rows stay parallel.
TEST(ContactMetadataCook, FlattenedSolimpMultiShapeSlices) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");

    auto s0 = MakeBoxShape(body);
    for (int k = 0; k < 5; ++k) s0.solimp[k] = static_cast<float>(k);   // 0..4
    scene.AddCollisionShape(std::move(s0));

    auto s1 = MakeBoxShape(body);
    for (int k = 0; k < 5; ++k) s1.solimp[k] = static_cast<float>(10 + k);  // 10..14
    scene.AddCollisionShape(std::move(s1));

    const auto blob = CookScene(scene);
    const auto& cp = blob.contact_params;

    ASSERT_EQ(blob.shape_count, 2u);
    ASSERT_EQ(cp.solimp.size(), 10u);
    for (int k = 0; k < 5; ++k) {
        EXPECT_FLOAT_EQ(cp.solimp[0 * 5 + k], static_cast<float>(k));
        EXPECT_FLOAT_EQ(cp.solimp[1 * 5 + k], static_cast<float>(10 + k));
    }
}

// ---------------------------------------------------------------------------
// 3. Friction-resolution precedence
// ---------------------------------------------------------------------------

TEST(ContactMetadataCook, FrictionShapeOverrideWins) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");

    MaterialRecord m;
    m.name = "mat";
    m.friction_mu = 0.3f;
    const auto mat = scene.AddMaterial(std::move(m));

    auto s = MakeBoxShape(body);
    s.material_id = mat;       // material present...
    s.friction_mu = 0.7f;      // ...but explicit per-shape override wins
    scene.AddCollisionShape(std::move(s));

    const auto blob = CookScene(scene);
    EXPECT_FLOAT_EQ(blob.contact_params.frictions[0], 0.7f);
}

TEST(ContactMetadataCook, FrictionFallsBackToMaterial) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");

    MaterialRecord m;
    m.name = "mat";
    m.friction_mu = 0.3f;
    const auto mat = scene.AddMaterial(std::move(m));

    auto s = MakeBoxShape(body);
    s.material_id = mat;
    // no per-shape override (friction_mu stays -1 sentinel)
    scene.AddCollisionShape(std::move(s));

    const auto blob = CookScene(scene);
    EXPECT_FLOAT_EQ(blob.contact_params.frictions[0], 0.3f);
}

TEST(ContactMetadataCook, FrictionDefaultsToOneWhenNoMaterialNoOverride) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");
    scene.AddCollisionShape(MakeBoxShape(body));  // no material, no override

    const auto blob = CookScene(scene);
    EXPECT_FLOAT_EQ(blob.contact_params.frictions[0], 1.0f);
}

// ---------------------------------------------------------------------------
// 4. Exclude-pair storage + canonicalization
// ---------------------------------------------------------------------------

TEST(ContactMetadataCook, ExcludePairsCanonicalizedAndStored) {
    SceneIR scene;
    scene.AddExcludePair(2, 5);   // already canonical
    scene.AddExcludePair(5, 2);   // reversed => canonicalizes to (2,5)

    const auto& pairs = scene.ExcludePairs();
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].first, 2u);
    EXPECT_EQ(pairs[0].second, 5u);
    EXPECT_EQ(pairs[1].first, 2u);
    EXPECT_EQ(pairs[1].second, 5u);
}

// ---------------------------------------------------------------------------
// 5. Determinism: cook twice, assert field-by-field equal (D1)
// ---------------------------------------------------------------------------

TEST(ContactMetadataCook, CookTwiceIsFieldByFieldIdentical) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");

    MaterialRecord m;
    m.name = "mat";
    m.friction_mu = 0.55f;
    const auto mat = scene.AddMaterial(std::move(m));

    // A few shapes with varied contact metadata + a material-backed friction.
    auto s0 = MakeBoxShape(body);
    s0.contype = 2u; s0.conaffinity = 3u; s0.collision_group = -1;
    s0.solref[0] = 0.03f; s0.solref[1] = 0.8f;
    s0.solimp[0] = 0.1f; s0.solimp[4] = 1.5f;
    s0.priority = 4; s0.solmix = 0.6f; s0.margin = 0.005f; s0.gap = 0.007f;
    s0.condim = uint8_t(1);
    s0.friction_mu = 0.9f;   // override
    scene.AddCollisionShape(std::move(s0));

    auto s1 = MakeBoxShape(body);
    s1.material_id = mat;    // inherits material μ = 0.55
    scene.AddCollisionShape(std::move(s1));

    const auto a = CookScene(scene);
    const auto b = CookScene(scene);

    const auto& x = a.contact_params;
    const auto& y = b.contact_params;

    EXPECT_EQ(a.shape_count, b.shape_count);
    EXPECT_EQ(x.contypes,      y.contypes);
    EXPECT_EQ(x.conaffinities, y.conaffinities);
    EXPECT_EQ(x.groups,        y.groups);
    EXPECT_EQ(x.solref0,       y.solref0);
    EXPECT_EQ(x.solref1,       y.solref1);
    EXPECT_EQ(x.solimp,        y.solimp);
    EXPECT_EQ(x.frictions,     y.frictions);
    EXPECT_EQ(x.condims,       y.condims);
    EXPECT_EQ(x.priorities,    y.priorities);
    EXPECT_EQ(x.solmix,        y.solmix);
    EXPECT_EQ(x.margins,       y.margins);
    EXPECT_EQ(x.gaps,          y.gaps);
}

// Cross-cut invariant: the contact-param table is exactly parallel to the
// shape table (one row per cooked shape row).
TEST(ContactMetadataCook, ContactParamTableParallelToShapeTable) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("b");
    scene.AddCollisionShape(MakeBoxShape(body));
    scene.AddCollisionShape(MakeBoxShape(body));
    scene.AddCollisionShape(MakeBoxShape(body));

    const auto blob = CookScene(scene);
    const auto& cp = blob.contact_params;
    EXPECT_EQ(blob.shape_count, 3u);
    EXPECT_EQ(cp.contypes.size(),      blob.shape_count);
    EXPECT_EQ(cp.conaffinities.size(), blob.shape_count);
    EXPECT_EQ(cp.groups.size(),        blob.shape_count);
    EXPECT_EQ(cp.solref0.size(),       blob.shape_count);
    EXPECT_EQ(cp.solref1.size(),       blob.shape_count);
    EXPECT_EQ(cp.solimp.size(),        blob.shape_count * 5u);
    EXPECT_EQ(cp.frictions.size(),     blob.shape_count);
    EXPECT_EQ(cp.condims.size(),       blob.shape_count);
    EXPECT_EQ(cp.priorities.size(),    blob.shape_count);
    EXPECT_EQ(cp.solmix.size(),        blob.shape_count);
    EXPECT_EQ(cp.margins.size(),       blob.shape_count);
    EXPECT_EQ(cp.gaps.size(),          blob.shape_count);
}
