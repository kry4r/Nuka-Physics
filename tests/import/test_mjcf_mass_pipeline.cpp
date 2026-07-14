// Reusable MJCF -> SceneIR -> CookedBlob gate for body-mass source selection.

#include "import/mjcf_importer.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

nuka::scene::BodyId BodyNamed(const nuka::scene::SceneIR& scene,
                              const std::string& name) {
    for (nuka::scene::BodyId body = 0; body < scene.RigidBodyCount(); ++body) {
        if (scene.GetBody(body).name == name) return body;
    }
    return nuka::scene::kInvalidBody;
}

}  // namespace

TEST(MjcfMassPipeline, PreservesMassSourcesFromImportThroughCook) {
    using nuka::math::Vec3;
    using nuka::scene::BodyId;

    const auto scene =
        nuka::import::LoadMjcf("tests/data/mjcf_inertial_sources.xml");
    nuka::scene::CookSceneOptions options;
    options.bake_sdf = false;
    options.general_single_hull = true;
    const auto cooked = nuka::scene::CookScene(scene, options);

    ASSERT_EQ(scene.RigidBodyCount(), 3u);
    ASSERT_EQ(cooked.body_count, scene.RigidBodyCount());
    ASSERT_EQ(scene.ShapeCount(), 1u);

    const BodyId wrapper = BodyNamed(scene, "empty_wrapper");
    const BodyId explicit_body = BodyNamed(scene, "explicit_inertial");
    const BodyId geom_only = BodyNamed(scene, "geom_only");
    ASSERT_EQ(wrapper, 0u);
    ASSERT_EQ(explicit_body, 1u);
    ASSERT_EQ(geom_only, 2u);

    EXPECT_FLOAT_EQ(scene.GetBody(wrapper).mass, 0.0f);
    EXPECT_EQ(scene.GetBody(wrapper).inertia, Vec3::Zero());
    EXPECT_FLOAT_EQ(cooked.bodies.masses[wrapper], 0.0f);
    EXPECT_EQ(cooked.bodies.inertias[wrapper], Vec3::Zero());

    EXPECT_FLOAT_EQ(scene.GetBody(explicit_body).mass, 2.5f);
    EXPECT_EQ(scene.GetBody(explicit_body).inertia, Vec3(0.1f, 0.2f, 0.3f));
    EXPECT_FLOAT_EQ(cooked.bodies.masses[explicit_body], 2.5f);
    EXPECT_EQ(cooked.bodies.inertias[explicit_body], Vec3(0.1f, 0.2f, 0.3f));

    // Geom-derived inertia is not implemented; stay massless rather than leaking
    // the unrelated SceneIR construction placeholder into cooked physics.
    EXPECT_EQ(scene.GetShape(0).body_id, geom_only);
    EXPECT_FLOAT_EQ(scene.GetBody(geom_only).mass, 0.0f);
    EXPECT_EQ(scene.GetBody(geom_only).inertia, Vec3::Zero());
    EXPECT_FLOAT_EQ(cooked.bodies.masses[geom_only], 0.0f);
    EXPECT_EQ(cooked.bodies.inertias[geom_only], Vec3::Zero());
}
