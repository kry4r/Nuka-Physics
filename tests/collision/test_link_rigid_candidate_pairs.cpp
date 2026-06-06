// ---------------------------------------------------------------------------
// nuka v0.8 co-residence (commit 2/3) VALIDATION GATE: the broadphase emits
// ARTICULATION-LINK <-> RIGID candidate pairs through the generalized tagged
// core, consuming commit 1's per-link world AABBs (link_aabb.hpp). VALIDATED-
// NOT-WIRED -- nothing here touches world_stepper.
//
// The generalization under test (rigid_candidate_pairs.cu):
//   * BuildCandidatePairsTagged -- the type-agnostic core: LBVH over tagged AABBs
//     -> bitmask + exclude + same-(body,type) filter -> compact (each side's
//     type/react/handle stamped from per-AABB tags, NOT hardcoded RigidBody) ->
//     sorted D1 stream. Handle split from body_id (the crux): the emit HANDLE is
//     the body-id for rigid / the LINK index for articulation; the body_id feeds
//     ONLY the same-body drop + exclude filter.
//   * BuildArticulationRigidCandidatePairs -- concatenates rigid shape AABBs
//     (RigidBody, handle=body-id) and link shape AABBs (ArticulationLink,
//     handle=link-index) and runs the core -> the cross-type stream.
//   * BuildRigidCandidatePairs -- now a thin all-RigidBody wrapper over the core;
//     must stay BYTE-IDENTICAL to C2b (the real regression guard is the existing
//     nuka_lbvh_filtered_pairs_test; here we additionally byte-compare a tagged
//     all-rigid run against the wrapper).
//
// Synthetic AABBs + tags (no cooked robot needed -- faster, like the C2b unit
// tests). Gates:
//   1. CROSS-TYPE EMISSION: an ArticulationLink AABB overlapping a RigidBody AABB
//      -> exactly one CandidatePair with the correct per-side type + handle,
//      stable_key canonical.
//   2. SAME-TYPE ALONGSIDE: rigid<->rigid and link<->link pairs emit correctly
//      next to the cross-type pair.
//   3. FILTER CROSS-TYPE (non-vacuous): a bitmask that rejects, and an excluded
//      body-pair, each DROPS the cross-type pair (toggle -> count changes).
//   4. RIGID REGRESSION: an all-RigidBody tagged set == BuildRigidCandidatePairs
//      on the same AABBs/policy (byte-compare downloaded pairs).
//   5. D1: two runs -> byte-identical downloaded pairs (memcmp).
//   6. HANDLE-GUARD: a handle >= 2^28 throws.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/candidate_pair.hpp"
#include "collision/link_aabb.hpp"
#include "collision/rigid_candidate_pairs.hpp"
#include "constraint/collidable.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "phi/device_context.hpp"
#include "scene/cooked_blob.hpp"   // scene::CookedFilterPolicy (rigid wrapper path)

#include <gtest/gtest.h>

#include <stdexcept>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace nuka;
using constraint::CollidableRef;
using constraint::CollidableType;
using constraint::ReactionProviderKind;

namespace {

// An axis-aligned box of half-extent `h` centered at `c` (the {min,max} route the
// AABB API doc sanctions), so overlaps are exactly under our control.
collision::AABB BoxAt(math::Vec3 c, float h) {
    collision::AABB box;
    box.min = c - math::Vec3{h, h, h};
    box.max = c + math::Vec3{h, h, h};
    return box;
}

ReactionProviderKind RigidReact() {
    return constraint::GetCollidableTypeInfo(CollidableType::RigidBody).react;
}
ReactionProviderKind LinkReact() {
    return constraint::GetCollidableTypeInfo(CollidableType::ArticulationLink).react;
}

// Does the downloaded stream contain exactly one pair whose two SIDES (as a set,
// order-agnostic -- a/b slot order follows AABB-index, not type) match the given
// (type,handle) endpoints? Also verifies react matches the type and the
// stable_key is the canonical MakeStableKey of the two sides.
bool HasExactlyPair(const std::vector<collision::CandidatePair>& pairs,
                    CollidableType ta, uint32_t ha,
                    CollidableType tb, uint32_t hb) {
    int matches = 0;
    for (const auto& p : pairs) {
        const bool fwd = (p.a.type == ta && p.a.handle == ha &&
                          p.b.type == tb && p.b.handle == hb);
        const bool rev = (p.a.type == tb && p.a.handle == hb &&
                          p.b.type == ta && p.b.handle == ha);
        if (!fwd && !rev) continue;
        // react must agree with type (stamped from the registry).
        EXPECT_EQ(p.a.react,
                  constraint::GetCollidableTypeInfo(p.a.type).react);
        EXPECT_EQ(p.b.react,
                  constraint::GetCollidableTypeInfo(p.b.type).react);
        // stable_key is the canonical key of the two stamped sides.
        EXPECT_EQ(p.stable_key, collision::MakeStableKey(p.a, p.b));
        ++matches;
    }
    return matches == 1;
}

// Build the mixed cross-type stream with all-permissive masks unless overridden.
// rigid: bodies at the given centers; links: link AABBs at given centers with the
// given link indices + owning body ids.
struct MixedScene {
    std::vector<collision::AABB> rigid_aabbs;
    std::vector<uint32_t>        rigid_body_ids;
    std::vector<uint32_t>        rigid_contypes;
    std::vector<uint32_t>        rigid_conaff;

    collision::LinkShapeAabbs    links;
    std::vector<uint32_t>        link_contypes;
    std::vector<uint32_t>        link_conaff;

    std::vector<std::pair<uint32_t, uint32_t>> excluded;
};

std::vector<collision::CandidatePair> RunMixed(const MixedScene& s) {
    auto context = phi::MakeDefaultDeviceContext();
    const uint32_t rc = static_cast<uint32_t>(s.rigid_aabbs.size());
    const collision::AABB* dev = nullptr;
    phi::Buffer d_rigid;
    if (rc > 0u) {
        d_rigid = phi::UploadVector(s.rigid_aabbs);
        dev = static_cast<const collision::AABB*>(d_rigid.Data());
    }
    auto stream = collision::BuildArticulationRigidCandidatePairs(
        context, dev, rc,
        s.rigid_body_ids.data(), s.rigid_contypes.data(), s.rigid_conaff.data(),
        s.links, s.link_contypes.data(), s.link_conaff.data(),
        s.excluded);
    EXPECT_EQ(stream.Count(), stream.DownloadPairs().size());
    return stream.DownloadPairs();
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1+2: CROSS-TYPE EMISSION + SAME-TYPE ALONGSIDE. Layout (half-extent 0.5,
// overlap iff centers within 1.0 on each axis):
//   rigid 0 @ x=0.0  (body 100)
//   rigid 1 @ x=0.6  (body 101)
//   link  0 @ x=0.6  (link idx 7, body 200)
//   link  1 @ x=1.1  (link idx 8, body 201)
// Overlap enumeration (|dx| < 1.0):
//   rigid0(0.0)-rigid1(0.6): 0.6 -> rigid<->rigid
//   rigid0(0.0)-link0 (0.6): 0.6 -> CROSS
//   rigid0(0.0)-link1 (1.1): 1.1 -> no
//   rigid1(0.6)-link0 (0.6): 0.0 -> CROSS
//   rigid1(0.6)-link1 (1.1): 0.5 -> CROSS
//   link0 (0.6)-link1 (1.1): 0.5 -> link<->link
// => 5 survivors: 1 rigid<->rigid, 3 cross-type, 1 link<->link.
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, CrossTypeAndSameTypeEmission) {
    MixedScene s;
    s.rigid_aabbs   = {BoxAt({0.0f, 0.0f, 0.0f}, 0.5f),
                       BoxAt({0.6f, 0.0f, 0.0f}, 0.5f)};
    s.rigid_body_ids = {100u, 101u};
    s.rigid_contypes = {1u, 1u};
    s.rigid_conaff   = {1u, 1u};

    s.links.aabbs          = {BoxAt({0.6f, 0.0f, 0.0f}, 0.5f),
                              BoxAt({1.1f, 0.0f, 0.0f}, 0.5f)};
    s.links.link_indices   = {7u, 8u};
    s.links.shape_body_ids = {200u, 201u};
    s.links.source_shape_ids = {0u, 1u};
    s.link_contypes = {1u, 1u};
    s.link_conaff   = {1u, 1u};

    const auto got = RunMixed(s);

    EXPECT_EQ(got.size(), 5u) << "expected 5 survivors across all type combos";
    // rigid<->rigid
    EXPECT_TRUE(HasExactlyPair(got, CollidableType::RigidBody, 100u,
                               CollidableType::RigidBody, 101u))
        << "rigid<->rigid pair missing";
    // CROSS-TYPE: rigid 100 -- link 7
    EXPECT_TRUE(HasExactlyPair(got, CollidableType::RigidBody, 100u,
                               CollidableType::ArticulationLink, 7u))
        << "cross-type rigid100<->link7 pair missing";
    // CROSS-TYPE: rigid 101 -- link 7
    EXPECT_TRUE(HasExactlyPair(got, CollidableType::RigidBody, 101u,
                               CollidableType::ArticulationLink, 7u))
        << "cross-type rigid101<->link7 pair missing";
    // CROSS-TYPE: rigid 101 -- link 8
    EXPECT_TRUE(HasExactlyPair(got, CollidableType::RigidBody, 101u,
                               CollidableType::ArticulationLink, 8u))
        << "cross-type rigid101<->link8 pair missing";
    // link<->link: link 7 -- link 8
    EXPECT_TRUE(HasExactlyPair(got, CollidableType::ArticulationLink, 7u,
                               CollidableType::ArticulationLink, 8u))
        << "link<->link pair missing";
}

// ---------------------------------------------------------------------------
// Gate 3a: FILTER CROSS-TYPE via bitmask (non-vacuous). A single rigid (body 100)
// overlapping a single link (idx 7, body 200). With permissive masks the cross-
// type pair emits (count 1); flipping the link's contype/conaffinity to disjoint
// bits DROPS it (count 0).
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, BitmaskDropsCrossTypePair) {
    MixedScene base;
    base.rigid_aabbs    = {BoxAt({0.0f, 0.0f, 0.0f}, 0.5f)};
    base.rigid_body_ids = {100u};
    base.rigid_contypes = {1u};
    base.rigid_conaff   = {1u};
    base.links.aabbs          = {BoxAt({0.6f, 0.0f, 0.0f}, 0.5f)};
    base.links.link_indices   = {7u};
    base.links.shape_body_ids = {200u};
    base.links.source_shape_ids = {0u};
    base.link_contypes = {1u};
    base.link_conaff   = {1u};

    // Permissive -> 1 cross-type pair.
    {
        const auto got = RunMixed(base);
        EXPECT_EQ(got.size(), 1u);
        EXPECT_TRUE(HasExactlyPair(got, CollidableType::RigidBody, 100u,
                                   CollidableType::ArticulationLink, 7u));
    }
    // Disjoint bits: rigid (1,1) vs link (2,2): (1&2)|(2&1)==0 -> DROP.
    {
        MixedScene s = base;
        s.link_contypes = {2u};
        s.link_conaff   = {2u};
        const auto got = RunMixed(s);
        EXPECT_EQ(got.size(), 0u) << "bitmask did not drop the cross-type pair";
    }
}

// ---------------------------------------------------------------------------
// Gate 3b: FILTER CROSS-TYPE via excluded body-pair (non-vacuous). Same overlap;
// adding the (rigid body 100, link body 200) pair to the exclude list DROPS the
// cross-type pair. NOTE the exclude operates on BODY ids (rigid body / link body),
// NOT the emit handles (which are body-id 100 / link-index 7) -- this proves the
// handle/body_id split: the filter uses body ids, the emit uses handles.
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, ExcludeDropsCrossTypePair) {
    MixedScene base;
    base.rigid_aabbs    = {BoxAt({0.0f, 0.0f, 0.0f}, 0.5f)};
    base.rigid_body_ids = {100u};
    base.rigid_contypes = {1u};
    base.rigid_conaff   = {1u};
    base.links.aabbs          = {BoxAt({0.6f, 0.0f, 0.0f}, 0.5f)};
    base.links.link_indices   = {7u};
    base.links.shape_body_ids = {200u};
    base.links.source_shape_ids = {0u};
    base.link_contypes = {1u};
    base.link_conaff   = {1u};

    // No exclude -> 1 pair.
    {
        const auto got = RunMixed(base);
        EXPECT_EQ(got.size(), 1u);
    }
    // Exclude (body 100, body 200) -> DROP. Canonical (min,max) ascending.
    {
        MixedScene s = base;
        s.excluded = {{100u, 200u}};
        const auto got = RunMixed(s);
        EXPECT_EQ(got.size(), 0u) << "exclude did not drop the cross-type pair";
    }
    // Control: excluding the EMIT handles (100, 7) instead of the body ids must
    // NOT drop -- proves the filter is in body-id space, distinct from handles.
    {
        MixedScene s = base;
        s.excluded = {{7u, 100u}};
        const auto got = RunMixed(s);
        EXPECT_EQ(got.size(), 1u)
            << "exclude wrongly matched on emit handles, not body ids";
    }
}

// ---------------------------------------------------------------------------
// Gate 4: RIGID REGRESSION. An all-RigidBody tagged set run through the tagged
// core produces the SAME stream as BuildRigidCandidatePairs (the wrapper) on the
// same AABBs + same exclude policy. Byte-compares the downloaded pairs. (The
// wrapper IS core+explicit-merge, so we compare with an empty explicit policy.)
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, RigidWrapperMatchesTaggedCore) {
    auto context = phi::MakeDefaultDeviceContext();

    // 5 boxes; a dense overlapping chain.
    std::vector<collision::AABB> aabbs;
    std::vector<uint32_t> body_ids;
    for (int i = 0; i < 5; ++i) {
        aabbs.push_back(BoxAt({0.4f * static_cast<float>(i), 0.0f, 0.0f}, 0.5f));
        body_ids.push_back(static_cast<uint32_t>(50 + i));
    }
    std::vector<uint32_t> contypes(5u, 1u);
    std::vector<uint32_t> conaff(5u, 1u);
    std::vector<std::pair<uint32_t, uint32_t>> excluded = {{51u, 52u}};

    phi::Buffer d_aabbs = phi::UploadVector(aabbs);
    const auto* dev = static_cast<const collision::AABB*>(d_aabbs.Data());

    // --- Wrapper path (policy carries the exclude, no explicit pairs). -------
    scene::CookedFilterPolicy policy;
    policy.excluded_body_pairs = excluded;  // canonical (min,max) ascending.
    auto wrapper_stream = collision::BuildRigidCandidatePairs(
        context, dev, 5u, body_ids.data(), contypes.data(), conaff.data(), policy);
    const auto wrapper = wrapper_stream.DownloadPairs();

    // --- Tagged-core path: all-RigidBody tags, handle==body_id, no extras. ---
    std::vector<CollidableType>       types(5u, CollidableType::RigidBody);
    std::vector<ReactionProviderKind> reacts(5u, RigidReact());
    collision::CandidateAabbTags tags;
    tags.types         = types.data();
    tags.reacts        = reacts.data();
    tags.handles       = body_ids.data();
    tags.body_ids      = body_ids.data();
    tags.contypes      = contypes.data();
    tags.conaffinities = conaff.data();
    auto core_stream = collision::BuildCandidatePairsTagged(
        context, dev, 5u, tags, excluded, {});
    const auto core = core_stream.DownloadPairs();

    ASSERT_EQ(wrapper.size(), core.size());
    ASSERT_GT(wrapper.size(), 0u) << "regression scene produced no pairs";
    EXPECT_EQ(std::memcmp(wrapper.data(), core.data(),
                          wrapper.size() * sizeof(collision::CandidatePair)), 0)
        << "rigid wrapper is not byte-identical to the tagged core (all-rigid)";
}

// ---------------------------------------------------------------------------
// Gate 4b: EXPLICIT-PAIR DEDUP WINNER (byte-layout). The rigid wrapper appends an
// explicit <pair> as an extra survivor AFTER the device survivors, so when the
// explicit pair DUPLICATES a device-detected pair (same body-pair, both overlap
// + pass bitmask + not excluded), the adjacent dedup keeps the DEVICE copy, which
// always stores the canonical (a.handle=min, b.handle=max). An explicit <pair>
// authored in DESCENDING geom order would store (max,min) -- same stable_key,
// different bytes -- so if the extras were merged BEFORE the device survivors the
// deduped pair would carry the (max,min) layout. This pins the device-first order
// (the byte-identity contract vs the pre-refactor C2b builder). NON-VACUOUS: the
// pair genuinely overlaps + passes bitmask, so the device path emits it too.
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, ExplicitPairKeepsDeviceCanonicalLayout) {
    auto context = phi::MakeDefaultDeviceContext();

    // Two overlapping boxes, bodies 10 (shape 0) and 20 (shape 1).
    std::vector<collision::AABB> aabbs = {BoxAt({0.0f, 0.0f, 0.0f}, 0.5f),
                                          BoxAt({0.6f, 0.0f, 0.0f}, 0.5f)};
    std::vector<uint32_t> body_ids = {10u, 20u};
    std::vector<uint32_t> contypes(2u, 1u);
    std::vector<uint32_t> conaff(2u, 1u);
    phi::Buffer d_aabbs = phi::UploadVector(aabbs);
    const auto* dev = static_cast<const collision::AABB*>(d_aabbs.Data());

    // Explicit <pair> with geoms ordered so the wrapper builds a DESCENDING-handle
    // extra: geom1=1 -> body 20 (a.handle), geom2=0 -> body 10 (b.handle). The
    // wrapper reads policy.explicit_pairs verbatim, so this extra is (20,10) --
    // the same body-pair the device path detects as (10,20). The dedup must keep
    // the device's (10,20) layout.
    scene::CookedFilterPolicy policy;
    scene::CookedExplicitPair ep;
    ep.geom1 = 1u;  // body 20
    ep.geom2 = 0u;  // body 10
    policy.explicit_pairs.push_back(ep);

    auto stream = collision::BuildRigidCandidatePairs(
        context, dev, 2u, body_ids.data(), contypes.data(), conaff.data(), policy);
    const auto pairs = stream.DownloadPairs();

    ASSERT_EQ(pairs.size(), 1u) << "expected exactly one deduped body-pair";
    // The surviving pair must carry the DEVICE-canonical (min,max) handle layout,
    // NOT the explicit pair's authored descending order -- proves extras append
    // after the device survivors (the dedup keeps the device copy).
    EXPECT_LT(pairs[0].a.handle, pairs[0].b.handle)
        << "deduped pair did not keep the device canonical (min,max) layout";
    EXPECT_EQ(pairs[0].a.handle, 10u);
    EXPECT_EQ(pairs[0].b.handle, 20u);
}

// ---------------------------------------------------------------------------
// Gate 5: D1. Two mixed-stream runs are byte-identical (memcmp). Uses the gate-1
// scene (rich, all type combos).
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, D1TwoRunByteExact) {
    MixedScene s;
    s.rigid_aabbs   = {BoxAt({0.0f, 0.0f, 0.0f}, 0.5f),
                       BoxAt({0.6f, 0.0f, 0.0f}, 0.5f)};
    s.rigid_body_ids = {100u, 101u};
    s.rigid_contypes = {1u, 1u};
    s.rigid_conaff   = {1u, 1u};
    s.links.aabbs          = {BoxAt({0.6f, 0.0f, 0.0f}, 0.5f),
                              BoxAt({1.1f, 0.0f, 0.0f}, 0.5f)};
    s.links.link_indices   = {7u, 8u};
    s.links.shape_body_ids = {200u, 201u};
    s.links.source_shape_ids = {0u, 1u};
    s.link_contypes = {1u, 1u};
    s.link_conaff   = {1u, 1u};

    const auto run1 = RunMixed(s);
    const auto run2 = RunMixed(s);
    ASSERT_GT(run1.size(), 0u);
    ASSERT_EQ(run1.size(), run2.size());
    EXPECT_EQ(std::memcmp(run1.data(), run2.data(),
                          run1.size() * sizeof(collision::CandidatePair)), 0)
        << "mixed cross-type stream is not two-run byte-exact (D1)";
}

// ---------------------------------------------------------------------------
// Gate 6: HANDLE-GUARD. A handle >= 2^28 (would alias the type bits) throws at
// the mint site (the tagged core). Driven through the tagged core directly with a
// single oversized handle.
// ---------------------------------------------------------------------------
TEST(LinkRigidCandidatePairs, HandleGuardThrowsOnOversizedHandle) {
    auto context = phi::MakeDefaultDeviceContext();
    std::vector<collision::AABB> aabbs = {BoxAt({0.0f, 0.0f, 0.0f}, 0.5f),
                                          BoxAt({0.4f, 0.0f, 0.0f}, 0.5f)};
    phi::Buffer d_aabbs = phi::UploadVector(aabbs);
    const auto* dev = static_cast<const collision::AABB*>(d_aabbs.Data());

    const uint32_t kBad = collision::kCandidateHandleMask + 1u;  // 2^28 exactly.
    std::vector<CollidableType>       types(2u, CollidableType::RigidBody);
    std::vector<ReactionProviderKind> reacts(2u, RigidReact());
    std::vector<uint32_t> handles  = {0u, kBad};  // second handle is oversized.
    std::vector<uint32_t> body_ids = {0u, 1u};
    std::vector<uint32_t> contypes(2u, 1u);
    std::vector<uint32_t> conaff(2u, 1u);
    collision::CandidateAabbTags tags;
    tags.types         = types.data();
    tags.reacts        = reacts.data();
    tags.handles       = handles.data();
    tags.body_ids      = body_ids.data();
    tags.contypes      = contypes.data();
    tags.conaffinities = conaff.data();

    EXPECT_THROW(
        collision::BuildCandidatePairsTagged(context, dev, 2u, tags, {}, {}),
        std::runtime_error);

    // Boundary: the maximal valid handle (2^28 - 1) must NOT throw.
    std::vector<uint32_t> ok_handles = {0u, collision::kCandidateHandleMask};
    tags.handles = ok_handles.data();
    EXPECT_NO_THROW({
        auto stream = collision::BuildCandidatePairsTagged(
            context, dev, 2u, tags, {}, {});
        (void)stream;
    });
}
