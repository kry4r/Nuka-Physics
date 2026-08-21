#include "nk/contact/contact_identity.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::GetCollidableTypeInfo;
using nuka::nk::CanonicalContactDescriptor;
using nuka::nk::ContactId;

CollidableRef Ref(CollidableType type, uint32_t handle) {
    return {type, GetCollidableTypeInfo(type).react, handle};
}

CanonicalContactDescriptor Descriptor(uint32_t a, uint32_t b) {
    CanonicalContactDescriptor d;
    d.a = Ref(CollidableType::RigidBody, a);
    d.b = Ref(CollidableType::RigidBody, b);
    d.local_point_a = {0.1f, -0.2f, 0.3f};
    d.local_point_b = {-0.4f, 0.5f, -0.6f};
    d.normal = {0.0f, 0.0f, 1.0f};
    d.feature_a = 7u;
    d.feature_b = 11u;
    d.manifold_slot = 2u;
    return d;
}

TEST(ContactIdentity, CanonicalPairSwapPreservesId) {
    CanonicalContactDescriptor direct = Descriptor(3u, 8u);
    CanonicalContactDescriptor swapped = direct;
    swapped.a = direct.b;
    swapped.b = direct.a;
    swapped.local_point_a = direct.local_point_b;
    swapped.local_point_b = direct.local_point_a;
    swapped.feature_a = direct.feature_b;
    swapped.feature_b = direct.feature_a;
    swapped.normal = -direct.normal;

    EXPECT_EQ(nuka::nk::MakeContactId(direct), nuka::nk::MakeContactId(swapped));
}

TEST(ContactIdentity, StableFeaturesOverridePointMotion) {
    CanonicalContactDescriptor first = Descriptor(1u, 2u);
    CanonicalContactDescriptor moved = first;
    moved.local_point_a = {2.0f, 3.0f, 4.0f};
    moved.local_point_b = {-2.0f, -3.0f, -4.0f};

    EXPECT_EQ(nuka::nk::MakeContactId(first), nuka::nk::MakeContactId(moved));
}

TEST(ContactIdentity, MissingFeaturesUseQuantizedLocalPoints) {
    CanonicalContactDescriptor first = Descriptor(1u, 2u);
    first.feature_a = nuka::nk::kContactFeatureUnavailable;
    first.feature_b = nuka::nk::kContactFeatureUnavailable;
    CanonicalContactDescriptor within_bin = first;
    within_bin.local_point_a.x += 0.00001f;
    CanonicalContactDescriptor next_bin = first;
    next_bin.local_point_a.x += 0.001f;

    EXPECT_EQ(nuka::nk::MakeContactId(first), nuka::nk::MakeContactId(within_bin));
    EXPECT_NE(nuka::nk::MakeContactId(first), nuka::nk::MakeContactId(next_bin));
}

TEST(ContactIdentity, VersionsInvalidateId) {
    CanonicalContactDescriptor base = Descriptor(1u, 2u);
    CanonicalContactDescriptor topology = base;
    CanonicalContactDescriptor material = base;
    topology.topology_version = 1u;
    material.material_version = 1u;

    EXPECT_NE(nuka::nk::MakeContactId(base), nuka::nk::MakeContactId(topology));
    EXPECT_NE(nuka::nk::MakeContactId(base), nuka::nk::MakeContactId(material));
}

TEST(ContactIdentity, TangentBasisUsesDeterministicAxisTie) {
    nuka::math::Vec3 t1;
    nuka::math::Vec3 t2;
    nuka::nk::BuildCanonicalTangentBasis({1.0f, 1.0f, 1.0f}, &t1, &t2);
    nuka::math::Vec3 again1;
    nuka::math::Vec3 again2;
    nuka::nk::BuildCanonicalTangentBasis({1.0f, 1.0f, 1.0f}, &again1, &again2);

    EXPECT_EQ(t1, again1);
    EXPECT_EQ(t2, again2);
    EXPECT_NEAR(t1.Dot(t2), 0.0f, 1.0e-6f);
    EXPECT_NEAR(t1.LengthSq(), 1.0f, 1.0e-6f);
    EXPECT_NEAR(t2.LengthSq(), 1.0f, 1.0e-6f);
}

TEST(ContactIdentity, MatcherPrefersFeaturesAndThenNearestFallback) {
    CanonicalContactDescriptor previous[4] = {
        Descriptor(1u, 2u), Descriptor(1u, 2u),
        Descriptor(1u, 2u), Descriptor(9u, 10u)};
    previous[0].feature_a = 10u;
    previous[1].feature_a = 20u;
    previous[2].feature_a = nuka::nk::kContactFeatureUnavailable;
    previous[2].feature_b = nuka::nk::kContactFeatureUnavailable;
    previous[2].local_point_a = {1.0f, 0.0f, 0.0f};

    CanonicalContactDescriptor current[4] = {
        previous[1], previous[0], previous[2], previous[3]};
    current[2].local_point_a = {1.001f, 0.0f, 0.0f};
    nuka::nk::ContactMatch matches[4];
    nuka::nk::MatchContactManifolds<4>(previous, 4u, current, 4u, matches);

    EXPECT_EQ(matches[0].previous_index, 1u);
    EXPECT_EQ(matches[1].previous_index, 0u);
    EXPECT_EQ(matches[2].previous_index, 2u);
    EXPECT_EQ(matches[3].previous_index, 3u);
}

}  // namespace
