// ---------------------------------------------------------------------------
// Tests for v0.8 C1c: the cook-time filtered-pair policy, the MuJoCo
// mj_contactParam MERGE rule, the contype/conaffinity bitmask, the system-pair
// matrix, the policy lookups, and scene_compose exclude/<pair> carry-over.
//
// The merge expected values are HAND-COMPUTED in comments at each assertion so a
// reviewer can re-derive them from the rule (research doc §8 lines 79-89).
// ---------------------------------------------------------------------------

#include "scene/contact_filter.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

using namespace nuka::scene;
using nuka::math::Transform;

namespace {

CollisionShapeRecord MakeBoxShape(BodyId body) {
    CollisionShapeRecord s;
    s.body_id = body;
    s.type    = ShapeType::Box;
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Merge -- equal priority (max condim/friction/margin/gap; solmix-weighted
//    standard solref + solimp).
// ---------------------------------------------------------------------------

TEST(FilterMerge, EqualPriorityCombine) {
    ContactParamsIn a;
    a.priority    = 0;
    a.condim      = 1;
    a.friction_mu = 0.3f;
    a.solref[0]   = 0.1f;  a.solref[1] = 2.0f;   // standard form ([0] > 0)
    for (int i = 0; i < 5; ++i) a.solimp[i] = 0.0f;
    a.solmix      = 1.0f;
    a.margin      = 0.01f; a.gap = 0.02f;

    ContactParamsIn b;
    b.priority    = 0;
    b.condim      = 3;
    b.friction_mu = 0.7f;
    b.solref[0]   = 0.2f;  b.solref[1] = 4.0f;   // standard form
    for (int i = 0; i < 5; ++i) b.solimp[i] = 1.0f;
    b.solmix      = 3.0f;
    b.margin      = 0.03f; b.gap = 0.05f;

    const MergedContactParams m = MergeContactParams(a, b);

    // condim = max(1,3) = 3; friction = max(0.3,0.7) = 0.7.
    EXPECT_EQ(m.condim, uint8_t(3));
    EXPECT_FLOAT_EQ(m.friction_mu, 0.7f);

    // mix = solmixA/(solmixA+solmixB) = 1/(1+3) = 0.25; imix = 0.75.
    // solref standard: solref[0] = 0.25*0.1 + 0.75*0.2 = 0.175;
    //                  solref[1] = 0.25*2.0 + 0.75*4.0 = 3.5.
    EXPECT_FLOAT_EQ(m.solref[0], 0.175f);
    EXPECT_FLOAT_EQ(m.solref[1], 3.5f);

    // solimp[i] = 0.25*0 + 0.75*1 = 0.75 for all i.
    for (int i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(m.solimp[i], 0.75f);

    // margin = max(0.01,0.03) = 0.03; gap = max(0.02,0.05) = 0.05.
    EXPECT_FLOAT_EQ(m.margin, 0.03f);
    EXPECT_FLOAT_EQ(m.gap, 0.05f);
}

// ---------------------------------------------------------------------------
// 2. Merge -- priority: higher-priority geom supplies ALL params wholesale.
// ---------------------------------------------------------------------------

TEST(FilterMerge, HigherPriorityWinsWholesale) {
    ContactParamsIn a;  // priority 5 => winner
    a.priority    = 5;
    a.condim      = 4;
    a.friction_mu = 0.9f;
    a.solref[0]   = 0.07f; a.solref[1] = 1.5f;
    for (int i = 0; i < 5; ++i) a.solimp[i] = 0.1f * (i + 1);  // 0.1..0.5
    a.margin      = 0.09f; a.gap = 0.08f;

    ContactParamsIn b;  // priority 0, totally different values
    b.priority    = 0;
    b.condim      = 1;
    b.friction_mu = 0.2f;
    b.solref[0]   = 0.5f;  b.solref[1] = 9.0f;
    for (int i = 0; i < 5; ++i) b.solimp[i] = 0.99f;
    b.margin      = 0.001f; b.gap = 0.002f;

    const MergedContactParams m = MergeContactParams(a, b);

    // Result must equal A's params exactly (A wins; b ignored entirely).
    EXPECT_EQ(m.condim, uint8_t(4));
    EXPECT_FLOAT_EQ(m.friction_mu, 0.9f);
    EXPECT_FLOAT_EQ(m.solref[0], 0.07f);
    EXPECT_FLOAT_EQ(m.solref[1], 1.5f);
    for (int i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(m.solimp[i], 0.1f * (i + 1));
    EXPECT_FLOAT_EQ(m.margin, 0.09f);
    EXPECT_FLOAT_EQ(m.gap, 0.08f);

    // Symmetry: swapping arguments must still yield A's params (B then wins-side).
    const MergedContactParams m2 = MergeContactParams(b, a);
    EXPECT_EQ(m2.condim, uint8_t(4));
    EXPECT_FLOAT_EQ(m2.friction_mu, 0.9f);
    EXPECT_FLOAT_EQ(m2.solref[0], 0.07f);
}

// ---------------------------------------------------------------------------
// 3. Merge -- direct solref: either side [0] <= 0 => elementwise MIN, not mix.
// ---------------------------------------------------------------------------

TEST(FilterMerge, DirectSolrefElementwiseMin) {
    ContactParamsIn a;
    a.priority  = 0;
    a.solref[0] = -3.0f;   // direct/negative form ([0] <= 0)
    a.solref[1] = -50.0f;
    a.solmix    = 1.0f;

    ContactParamsIn b;
    b.priority  = 0;
    b.solref[0] = -5.0f;   // also direct; either-side-<=0 selects MIN branch
    b.solref[1] = -10.0f;
    b.solmix    = 1.0f;

    const MergedContactParams m = MergeContactParams(a, b);
    // min(-3,-5) = -5 ; min(-50,-10) = -50.
    EXPECT_FLOAT_EQ(m.solref[0], -5.0f);
    EXPECT_FLOAT_EQ(m.solref[1], -50.0f);

    // Mixed: A standard, B direct => still MIN branch (EITHER <= 0).
    ContactParamsIn c;
    c.priority  = 0;
    c.solref[0] = 0.2f;    // standard
    c.solref[1] = 4.0f;
    const MergedContactParams m2 = MergeContactParams(c, b);
    EXPECT_FLOAT_EQ(m2.solref[0], -5.0f);   // min(0.2,-5)
    EXPECT_FLOAT_EQ(m2.solref[1], -10.0f);  // min(4.0,-10)
}

// ---------------------------------------------------------------------------
// 4. Bitmask: (contype_i & conaff_j) || (contype_j & conaff_i).
// ---------------------------------------------------------------------------

TEST(FilterBitmask, TwoWayAndOr) {
    // (contype=2,conaff=4) vs (contype=4,conaff=2):
    //   (2 & 2) = 2 nonzero => pass.
    EXPECT_TRUE(PassesContactBitmask(2u, 4u, 4u, 2u));

    // (1,1) vs (2,2): (1&2)=0 and (2&1)=0 => fail.
    EXPECT_FALSE(PassesContactBitmask(1u, 1u, 2u, 2u));

    // Default everything-collides (1,1)vs(1,1): (1&1)=1 => pass.
    EXPECT_TRUE(PassesContactBitmask(1u, 1u, 1u, 1u));

    // One-directional match: A.contype=1 & B.conaff=1 but B.contype=8 misses A.conaff=2.
    //   (1 & 1) = 1 => pass (OR over directions).
    EXPECT_TRUE(PassesContactBitmask(1u, 2u, 8u, 1u));
}

// ---------------------------------------------------------------------------
// 5. excluded_body_pairs: <exclude> + parent-child joint, sorted canonical.
// ---------------------------------------------------------------------------

TEST(FilterPolicy, ExcludedBodyPairsUnionAndLookup) {
    SceneIR scene;
    const BodyId b0 = scene.AddRigidBody("b0");
    const BodyId b1 = scene.AddRigidBody("b1");
    const BodyId b2 = scene.AddRigidBody("b2");
    const BodyId b3 = scene.AddRigidBody("b3");

    // Authored <exclude> between b0,b3 (reversed to test canonicalization).
    scene.AddExcludePair(b3, b0);
    // Parent-child joint b1 -> b2 => auto-exclude (b1,b2).
    scene.AddJoint("j", b1, b2);

    const auto blob = CookScene(scene);
    const auto& policy = blob.filter_policy;

    ASSERT_EQ(policy.excluded_body_pairs.size(), 2u);
    // Sorted canonical: (0,3) then (1,2).
    EXPECT_EQ(policy.excluded_body_pairs[0], std::make_pair(b0, b3));
    EXPECT_EQ(policy.excluded_body_pairs[1], std::make_pair(b1, b2));

    EXPECT_TRUE(IsBodyPairExcluded(policy, b0, b3));
    EXPECT_TRUE(IsBodyPairExcluded(policy, b3, b0));  // order-independent
    EXPECT_TRUE(IsBodyPairExcluded(policy, b1, b2));
    EXPECT_FALSE(IsBodyPairExcluded(policy, b0, b1));  // unrelated pair
    EXPECT_FALSE(IsBodyPairExcluded(policy, b2, b3));
}

// A floating-base / parentless joint contributes no exclude (kInvalidBody skip);
// a duplicate <exclude>+joint pair dedups to a single entry.
TEST(FilterPolicy, ParentlessJointSkippedAndDedup) {
    SceneIR scene;
    const BodyId b0 = scene.AddRigidBody("b0");
    const BodyId b1 = scene.AddRigidBody("b1");

    JointRecord freej;             // floating-base: parent stays kInvalidBody
    freej.parent_body = kInvalidBody;
    freej.child_body  = b0;
    freej.type        = JointType::Free;
    scene.AddJoint(std::move(freej));

    scene.AddJoint("hinge", b0, b1);   // auto-exclude (b0,b1)
    scene.AddExcludePair(b0, b1);      // SAME pair authored => dedups

    const auto blob = CookScene(scene);
    const auto& policy = blob.filter_policy;

    ASSERT_EQ(policy.excluded_body_pairs.size(), 1u);
    EXPECT_EQ(policy.excluded_body_pairs[0], std::make_pair(b0, b1));
    EXPECT_FALSE(IsBodyPairExcluded(policy, kInvalidBody, b0));
}

// ---------------------------------------------------------------------------
// 6. explicit_pairs: <contact><pair> shows up with correct params + lookup.
// ---------------------------------------------------------------------------

TEST(FilterPolicy, ExplicitPairsStoredAsAuthored) {
    SceneIR scene;
    const BodyId body = scene.AddRigidBody("b");
    const ShapeId g0 = scene.AddCollisionShape(MakeBoxShape(body));
    const ShapeId g1 = scene.AddCollisionShape(MakeBoxShape(body));
    const ShapeId g2 = scene.AddCollisionShape(MakeBoxShape(body));

    ContactPairOverride ov;
    ov.geom1       = g2;   // authored reversed => canonicalizes to (g0..) order
    ov.geom2       = g0;
    ov.condim      = 1;
    ov.friction_mu = 0.42f;
    ov.solref[0]   = 0.03f; ov.solref[1] = 0.8f;
    for (int k = 0; k < 5; ++k) ov.solimp[k] = 0.1f * (k + 1);
    ov.margin      = 0.005f; ov.gap = 0.006f;
    scene.AddContactPair(ov);

    const auto blob = CookScene(scene);
    const auto& policy = blob.filter_policy;
    ASSERT_EQ(policy.explicit_pairs.size(), 1u);

    // Canonical (g0,g2).
    const CookedExplicitPair& ep = policy.explicit_pairs[0];
    EXPECT_EQ(ep.geom1, g0);
    EXPECT_EQ(ep.geom2, g2);
    // Params stored verbatim (NOT merged with per-geom params).
    EXPECT_EQ(ep.params.condim, uint8_t(1));
    EXPECT_FLOAT_EQ(ep.params.friction_mu, 0.42f);
    EXPECT_FLOAT_EQ(ep.params.solref[0], 0.03f);
    EXPECT_FLOAT_EQ(ep.params.solref[1], 0.8f);
    for (int k = 0; k < 5; ++k) EXPECT_FLOAT_EQ(ep.params.solimp[k], 0.1f * (k + 1));
    EXPECT_FLOAT_EQ(ep.params.margin, 0.005f);
    EXPECT_FLOAT_EQ(ep.params.gap, 0.006f);

    // FindExplicitPair works in BOTH orders; nullptr for an unauthored pair.
    EXPECT_EQ(FindExplicitPair(policy, g0, g2), &ep);
    EXPECT_EQ(FindExplicitPair(policy, g2, g0), &ep);
    EXPECT_EQ(FindExplicitPair(policy, g0, g1), nullptr);
}

// Multiple explicit pairs are ascending-sorted by (geom1,geom2) for binary search.
TEST(FilterPolicy, ExplicitPairsSorted) {
    SceneIR scene;
    const BodyId body = scene.AddRigidBody("b");
    const ShapeId g0 = scene.AddCollisionShape(MakeBoxShape(body));
    const ShapeId g1 = scene.AddCollisionShape(MakeBoxShape(body));
    const ShapeId g2 = scene.AddCollisionShape(MakeBoxShape(body));

    auto mk = [](ShapeId a, ShapeId b) {
        ContactPairOverride o; o.geom1 = a; o.geom2 = b; return o;
    };
    // Author out of order.
    scene.AddContactPair(mk(g1, g2));
    scene.AddContactPair(mk(g0, g2));
    scene.AddContactPair(mk(g0, g1));

    const auto blob = CookScene(scene);
    const auto& ep = blob.filter_policy.explicit_pairs;
    ASSERT_EQ(ep.size(), 3u);
    EXPECT_EQ(ep[0].geom1, g0); EXPECT_EQ(ep[0].geom2, g1);
    EXPECT_EQ(ep[1].geom1, g0); EXPECT_EQ(ep[1].geom2, g2);
    EXPECT_EQ(ep[2].geom1, g1); EXPECT_EQ(ep[2].geom2, g2);

    EXPECT_NE(FindExplicitPair(blob.filter_policy, g0, g1), nullptr);
    EXPECT_NE(FindExplicitPair(blob.filter_policy, g1, g2), nullptr);
}

// ---------------------------------------------------------------------------
// System-pair matrix defaults all-enabled; set/clear is symmetric.
// ---------------------------------------------------------------------------

TEST(FilterPolicy, SystemPairMatrixDefaultsAndToggle) {
    SystemPairMatrix m;  // ctor => all true
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_TRUE(m.enabled[i][j]);

    SetSystemPairEnabled(m, SystemKind::Fluid, SystemKind::Cloth, false);
    EXPECT_FALSE(IsSystemPairEnabled(m, SystemKind::Fluid, SystemKind::Cloth));
    EXPECT_FALSE(IsSystemPairEnabled(m, SystemKind::Cloth, SystemKind::Fluid));  // symmetric
    EXPECT_TRUE(IsSystemPairEnabled(m, SystemKind::Rigid, SystemKind::Soft));    // untouched

    // The cooked policy carries the default all-enabled matrix.
    SceneIR scene;
    scene.AddRigidBody("b");
    const auto blob = CookScene(scene);
    EXPECT_TRUE(IsSystemPairEnabled(blob.filter_policy.system_pairs,
                                    SystemKind::Rigid, SystemKind::Fluid));
}

// ---------------------------------------------------------------------------
// 7. scene_compose carry-over: exclude + <pair> remapped across compose.
// ---------------------------------------------------------------------------

TEST(FilterCompose, ExcludeAndPairCarryOverRemapped) {
    // Base: 2 bodies + 1 shape + 1 exclude (b0,b1) + 1 <pair> on its shape.
    SceneIR base;
    const BodyId bb0 = base.AddRigidBody("base_b0");
    const BodyId bb1 = base.AddRigidBody("base_b1");
    const ShapeId bs0 = base.AddCollisionShape(MakeBoxShape(bb0));
    base.AddExcludePair(bb0, bb1);
    {
        ContactPairOverride o; o.geom1 = bs0; o.geom2 = bs0; o.friction_mu = 0.11f;
        base.AddContactPair(o);
    }

    // Addon: 2 bodies + 2 shapes + 1 exclude (a0,a1) + 1 <pair> on its shapes.
    SceneIR addon;
    const BodyId ab0 = addon.AddRigidBody("addon_b0");
    const BodyId ab1 = addon.AddRigidBody("addon_b1");
    const ShapeId as0 = addon.AddCollisionShape(MakeBoxShape(ab0));
    const ShapeId as1 = addon.AddCollisionShape(MakeBoxShape(ab1));
    addon.AddExcludePair(ab0, ab1);
    {
        ContactPairOverride o; o.geom1 = as0; o.geom2 = as1; o.friction_mu = 0.77f;
        addon.AddContactPair(o);
    }

    const auto boff = static_cast<uint32_t>(base.RigidBodyCount());   // 2
    const auto soff = static_cast<uint32_t>(base.ShapeCount());       // 1

    const SceneIR out = Compose(base, addon, Transform::Identity());

    // Both excludes present: base (0,1) unchanged + addon remapped (boff+0,boff+1).
    const auto& ex = out.ExcludePairs();
    ASSERT_EQ(ex.size(), 2u);
    EXPECT_EQ(ex[0], std::make_pair(bb0, bb1));
    EXPECT_EQ(ex[1], std::make_pair(boff + 0u, boff + 1u));

    // Both <pair> present: base unchanged + addon geom ids offset by soff.
    const auto& cp = out.ContactPairs();
    ASSERT_EQ(cp.size(), 2u);
    EXPECT_EQ(cp[0].geom1, bs0);
    EXPECT_EQ(cp[0].geom2, bs0);
    EXPECT_FLOAT_EQ(cp[0].friction_mu, 0.11f);
    EXPECT_EQ(cp[1].geom1, soff + as0);
    EXPECT_EQ(cp[1].geom2, soff + as1);
    EXPECT_FLOAT_EQ(cp[1].friction_mu, 0.77f);

    // End-to-end: cook the composed scene; policy reflects both remapped sets.
    const auto blob = CookScene(out);
    const auto& policy = blob.filter_policy;
    EXPECT_TRUE(IsBodyPairExcluded(policy, bb0, bb1));
    EXPECT_TRUE(IsBodyPairExcluded(policy, boff + 0u, boff + 1u));
    EXPECT_NE(FindExplicitPair(policy, soff + as0, soff + as1), nullptr);
}

// ---------------------------------------------------------------------------
// 8. Determinism: cook the same scene twice => policy field-by-field identical.
// ---------------------------------------------------------------------------

TEST(FilterPolicy, CookTwicePolicyIdentical) {
    SceneIR scene;
    const BodyId b0 = scene.AddRigidBody("b0");
    const BodyId b1 = scene.AddRigidBody("b1");
    const BodyId b2 = scene.AddRigidBody("b2");
    const ShapeId g0 = scene.AddCollisionShape(MakeBoxShape(b0));
    const ShapeId g1 = scene.AddCollisionShape(MakeBoxShape(b1));

    scene.AddExcludePair(b2, b0);
    scene.AddJoint("j", b0, b1);
    {
        ContactPairOverride o; o.geom1 = g1; o.geom2 = g0; o.friction_mu = 0.5f;
        scene.AddContactPair(o);
    }

    const auto a = CookScene(scene);
    const auto b = CookScene(scene);

    const auto& pa = a.filter_policy;
    const auto& pb = b.filter_policy;

    ASSERT_EQ(pa.excluded_body_pairs, pb.excluded_body_pairs);
    ASSERT_EQ(pa.explicit_pairs.size(), pb.explicit_pairs.size());
    for (size_t i = 0; i < pa.explicit_pairs.size(); ++i) {
        EXPECT_EQ(pa.explicit_pairs[i].geom1, pb.explicit_pairs[i].geom1);
        EXPECT_EQ(pa.explicit_pairs[i].geom2, pb.explicit_pairs[i].geom2);
        EXPECT_FLOAT_EQ(pa.explicit_pairs[i].params.friction_mu,
                        pb.explicit_pairs[i].params.friction_mu);
        EXPECT_EQ(pa.explicit_pairs[i].params.condim,
                  pb.explicit_pairs[i].params.condim);
    }
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_EQ(pa.system_pairs.enabled[i][j], pb.system_pairs.enabled[i][j]);
}
