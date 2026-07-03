// ---------------------------------------------------------------------------
// Multi-geom collidable cook: an articulation link with >1 collision primitive
// must cook EXTRA collidable body rows (proxies) that resolve to the owner link,
// while a single-geom link stays byte-identical (no proxies).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "math/transform.hpp"
#include "nk/model/model.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

namespace {

using nuka::math::Transform;
using nuka::math::Vec3;
namespace scene = nuka::scene;

// A 2-link arm (fixed base -> hinge -> link), with `n_geoms` capsule colliders on
// the link, each offset fore-aft so a multi-geom link forms a support polygon.
scene::SceneIR MakeArm(int n_geoms) {
    scene::SceneIR s;

    scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const scene::BodyId base_id = s.AddRigidBody(base);

    scene::RigidBodyRecord link;
    link.name = "link";
    link.parent_id = base_id;
    link.mass = 1.0f;
    link.inertia = Vec3{0.01f, 0.01f, 0.01f};
    link.local_transform.position = Vec3{0.0f, 0.0f, 0.2f};
    const scene::BodyId link_id = s.AddRigidBody(link);

    scene::JointRecord j;
    j.name = "j";
    j.parent_body = base_id;
    j.child_body = link_id;
    j.type = scene::JointType::Revolute;
    j.axis = Vec3{0.0f, 0.0f, 1.0f};
    s.AddJoint(j);

    for (int i = 0; i < n_geoms; ++i) {
        scene::CollisionShapeRecord c;
        c.body_id = link_id;
        c.type = scene::ShapeType::Capsule;
        c.radius = 0.02f;
        c.half_height = 0.015f;
        c.local_transform.position = Vec3{0.03f * static_cast<float>(i), 0.0f, -0.05f};
        s.AddCollisionShape(c);
    }
    return s;
}

nuka::nk::Model Cook(const scene::SceneIR& s) {
    return scene::cook::CookToModel(s, 1, scene::cook::CookToModelOptions{}).model;
}

}  // namespace

// A single-geom link cooks NO proxies: body_collidable_link stays empty/all-~0u.
TEST(MultiGeomCook, SingleGeomHasNoProxies) {
    const nuka::nk::Model m = Cook(MakeArm(1));
    for (uint32_t v : m.body_collidable_link) {
        EXPECT_EQ(v, ~uint32_t(0));
    }
    // The link carries its one primitive via link_geom (link 1 == the child link).
    ASSERT_GE(m.articulation.link_geom_kind.size(), 2u);
    EXPECT_NE(m.articulation.link_geom_kind[1], 0u);
}

// A two-geom link cooks exactly ONE proxy collidable row bound to the owner link.
TEST(MultiGeomCook, ExtraGeomBecomesProxyRow) {
    const nuka::nk::Model one = Cook(MakeArm(1));
    const nuka::nk::Model two = Cook(MakeArm(2));

    // One extra collidable body-row leaf vs the single-geom cook.
    EXPECT_EQ(two.capacities.bodies_per_env, one.capacities.bodies_per_env + 1u);

    // Exactly one proxy row, and it poses from the child link (template link 1).
    uint32_t proxy_rows = 0u, proxy_link = ~0u, proxy_row = ~0u;
    for (uint32_t b = 0; b < two.body_collidable_link.size(); ++b) {
        if (two.body_collidable_link[b] != ~uint32_t(0)) {
            ++proxy_rows;
            proxy_link = two.body_collidable_link[b];
            proxy_row = b;
        }
    }
    EXPECT_EQ(proxy_rows, 1u);
    EXPECT_EQ(proxy_link, 1u);  // the child link's template-local index.

    // The proxy row resolves its contact reaction back to the owner link.
    ASSERT_LT(proxy_row, two.body_to_link.size());
    EXPECT_EQ(two.body_to_link[proxy_row], 1u);
    // Its geom offset is the 2nd capsule's local transform (x = 0.03).
    ASSERT_LT(proxy_row, two.body_collidable_local.size());
    EXPECT_NEAR(two.body_collidable_local[proxy_row].position.x, 0.03f, 1e-6);

    // The proxy is excluded from its owner body row (never self-contacts the foot).
    const uint32_t owner_body = 1u;  // link body row.
    const uint64_t key = (static_cast<uint64_t>(owner_body < proxy_row ? owner_body : proxy_row) << 32) |
                         static_cast<uint64_t>(owner_body < proxy_row ? proxy_row : owner_body);
    bool excluded = false;
    for (uint64_t k : two.excluded_pairs) if (k == key) excluded = true;
    EXPECT_TRUE(excluded);

    // The trailing static ground row survives (last shape row is static, id -1).
    ASSERT_FALSE(two.shape_table_rows.empty());
    EXPECT_EQ(two.shape_table_rows.back().body_id, -1);
}

// Proxies of two links whose bodies are filter-excluded (parent-child) must be
// excluded from EACH OTHER, not only from the two owner body rows.
TEST(MultiGeomCook, CrossLinkProxiesInheritPairExclusion) {
    scene::SceneIR s;

    scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const scene::BodyId base_id = s.AddRigidBody(base);

    scene::BodyId prev = base_id;
    for (int k = 0; k < 2; ++k) {
        scene::RigidBodyRecord link;
        link.name = k == 0 ? "linkA" : "linkB";
        link.parent_id = prev;
        link.mass = 1.0f;
        link.inertia = Vec3{0.01f, 0.01f, 0.01f};
        link.local_transform.position = Vec3{0.0f, 0.0f, 0.2f};
        const scene::BodyId link_id = s.AddRigidBody(link);

        scene::JointRecord j;
        j.name = k == 0 ? "jA" : "jB";
        j.parent_body = prev;
        j.child_body = link_id;
        j.type = scene::JointType::Revolute;
        j.axis = Vec3{0.0f, 0.0f, 1.0f};
        s.AddJoint(j);
        prev = link_id;

        for (int i = 0; i < 2; ++i) {
            scene::CollisionShapeRecord c;
            c.body_id = link_id;
            c.type = scene::ShapeType::Capsule;
            c.radius = 0.02f;
            c.half_height = 0.015f;
            c.local_transform.position = Vec3{0.03f * static_cast<float>(i), 0.0f, -0.05f};
            s.AddCollisionShape(c);
        }
    }

    const nuka::nk::Model m = Cook(s);

    std::vector<uint32_t> proxies;
    for (uint32_t r = 0; r < m.body_collidable_link.size(); ++r) {
        if (m.body_collidable_link[r] != ~uint32_t(0)) proxies.push_back(r);
    }
    ASSERT_EQ(proxies.size(), 2u);
    ASSERT_NE(m.body_to_link[proxies[0]], m.body_to_link[proxies[1]]);

    const uint64_t key = (static_cast<uint64_t>(proxies[0]) << 32) |
                         static_cast<uint64_t>(proxies[1]);
    bool excluded = false;
    for (uint64_t k : m.excluded_pairs) if (k == key) excluded = true;
    EXPECT_TRUE(excluded) << "parent-child link proxies must inherit the pair filter";
}
