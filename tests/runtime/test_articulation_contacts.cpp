// ---------------------------------------------------------------------------
// p01-F T2 -- world-pose FK + foot-vs-ground contact detection
//
// Validates the two GPU passes in articulation_contacts.{cu,hpp}:
//
//  (A) UpdateWorldLinkPoses recomputes a world math::Transform per link from the
//      current generalized coordinates. The cooked link_pose is the FK-composed
//      world REST pose (cooker.cpp ResolveWorldTransform, all joint q = 0). So
//      with q forced to 0, the kernel's FK output must reproduce link_pose to
//      float tolerance -- an independent rest-pose sanity check.
//
//  (B) DetectFootGroundContacts is a deterministic sphere-vs-plane pass. We set
//      a static ground plane just ABOVE the Go2 feet so all four feet penetrate,
//      then assert each replicated env reports the expected number of contacts
//      (>0, <=4), contact points sit on the ground surface (z == ground), normals
//      are +Z, and depths are positive and bit-identical across replicas (D1
//      determinism).
//
// Foot-shape source: the cooked CookedShapeTable (world.template_view.shape_table)
// carries one Sphere per foot, its body_id (the calf body) and a local transform
// (identity for Go2). We map each foot's calf body -> BASE-relative global link
// index via the single-replica link_body table, then replicate that table; the
// detection kernel turns it into a per-replica global link, avoiding the trap
// that link_body is tiled WITHOUT a per-replica offset.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Test-only RAII device buffer over the phi v2 opaque Buffer*. Mirrors the legacy
// phi::Buffer surface (Data/Size/CopyFromHost/CopyToHost, move-only) so the test
// bodies stay byte-identical after the buffer sweep. Device-level DEFAULT
// (stream-0) allocation; CopyFromHost/CopyToHost run on stream 0 exactly like the
// legacy synchronous cudaMemcpy this replaces.
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    explicit OwnedDeviceBuffer(size_t bytes) : bytes_(bytes) {
        buf_ = nuka::phi::BufferAlloc(
            nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), bytes);
    }
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(OwnedDeviceBuffer&& o) noexcept
        : buf_(o.buf_), bytes_(o.bytes_) { o.buf_ = nullptr; o.bytes_ = 0u; }
    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) nuka::phi::BufferFree(buf_);
            buf_ = o.buf_; bytes_ = o.bytes_; o.buf_ = nullptr; o.bytes_ = 0u;
        }
        return *this;
    }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    const void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    size_t Size() const { return bytes_; }
    void CopyFromHost(const void* src, size_t bytes) {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferUpload(buf_, src, 0, bytes);
    }
    void CopyToHost(void* dst, size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    nuka::phi::Buffer* buf_ = nullptr;
    size_t bytes_ = 0u;
};

namespace articulation = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Independent host-side forward kinematics, built from the TRUSTED committed
// host primitives (Transform::operator*, Quat::FromAxisAngle) -- different code
// from the device QuatMul/RotateByQuat path, so matching it is a genuine
// GPU-vs-CPU cross-check of the world-pose FK. Mirrors the relative transform in
// featherstone_aba.cu's JointTransform: world[child] = world[parent] *
// {local_pose.position + parent_offset (+ axis*q prismatic),
//  local_pose.rotation * FromAxisAngle(axis, q) (revolute; identity otherwise)}.
std::vector<Transform> HostForwardKinematics(
    const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    std::vector<Transform> world(link_count);
    for (uint32_t artic = 0u; artic < host.ArticulationCount(); ++artic) {
        const uint32_t offset = host.articulation_link_offset[artic];
        const uint32_t count = host.articulation_link_count[artic];
        for (uint32_t local = 0u; local < count; ++local) {
            const uint32_t link = offset + local;
            const Transform local_pose = host.link_local_pose[link];
            const Vec3 axis = host.joint_axis[link];
            const float q = host.q[link];
            Vec3 translation = local_pose.position + host.parent_offset[link];
            Quat rotation = local_pose.rotation;
            switch (host.joint_type[link]) {
                case articulation::ArticulationJointType::Revolute:
                    rotation = (local_pose.rotation *
                                Quat::FromAxisAngle(axis, q)).Normalized();
                    break;
                case articulation::ArticulationJointType::Prismatic:
                    translation = translation + axis * q;
                    break;
                case articulation::ArticulationJointType::Fixed:
                    break;
            }
            const Transform relative{translation, rotation};
            const uint32_t parent_local = host.parent_link[link];
            world[link] = (parent_local == ~0u)
                              ? relative
                              : world[offset + parent_local] * relative;
        }
    }
    return world;
}

struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;  // base-relative
};

// Cooks a single Go2 and resolves its foot spheres to base-relative FootShapes.
CookedGo2 CookGo2() {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedGo2 result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);

    // Map each calf body_id to its base-relative global link index.
    const auto& shapes = world.template_view.shape_table;
    const uint32_t link_count = result.host.TotalLinkCount();
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = ~0u;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == ~0u) {
            continue;  // sphere not on an articulated link (e.g. base cube)
        }
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position
                                : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }
    return result;
}

} // namespace

TEST(ArticulationContacts, FkRestPoseAndDeterministicFootContacts) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t base_link_count = base.TotalLinkCount();
    ASSERT_GT(base_link_count, 0u);
    ASSERT_EQ(base.ArticulationCount(), 1u);

    // Go2 has four feet, each a sphere of radius 0.0175.
    ASSERT_EQ(cooked.feet.size(), 4u);
    const uint32_t foot_count = static_cast<uint32_t>(cooked.feet.size());
    ASSERT_LE(foot_count, articulation::kMaxFootContactsPerEnv);
    for (const auto& foot : cooked.feet) {
        EXPECT_NEAR(foot.radius, 0.0175f, 1.0e-6f);
        ASSERT_LT(foot.calf_local_link, base_link_count);
    }

    // Force q = 0 so the FK output is the rest configuration. NOTE: the cooked
    // link_pose is NOT the FK-composed world pose for this scene (the Go2 leg
    // bodies are authored as flat siblings of World, so each body's cooked pose
    // is just its own local translate, not composed down the chain) -- which is
    // precisely why this world-pose FK task exists. The independent reference is
    // a host-side FK recompute, NOT link_pose.
    for (auto& q : base.q) {
        q = 0.0f;
    }

    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // Runs the device FK on the given host state and downloads the world poses.
    auto run_device_fk = [&](const articulation::ArticulationHostState& state) {
        const uint32_t links = state.TotalLinkCount();
        auto device = articulation::UploadArticulationState(context, state);
        OwnedDeviceBuffer pose_buf(static_cast<size_t>(links) * sizeof(Transform));
        articulation::UpdateWorldLinkPoses(
            context, device.View(), static_cast<Transform*>(pose_buf.Data()));
        context.stream.Synchronize();
        std::vector<Transform> world(links);
        pose_buf.CopyToHost(world.data(), world.size() * sizeof(Transform));
        return world;
    };

    // --- (A) FK-at-q=0 reproduces the rest configuration (vs host-side FK) ------
    {
        const std::vector<Transform> got = run_device_fk(base);
        const std::vector<Transform> want = HostForwardKinematics(base);
        ASSERT_EQ(got.size(), want.size());
        const float kPosTol = 1.0e-5f;
        for (uint32_t link = 0u; link < base_link_count; ++link) {
            EXPECT_NEAR(got[link].position.x, want[link].position.x, kPosTol)
                << "link " << link;
            EXPECT_NEAR(got[link].position.y, want[link].position.y, kPosTol)
                << "link " << link;
            EXPECT_NEAR(got[link].position.z, want[link].position.z, kPosTol)
                << "link " << link;
        }
        // Sanity: at rest every foot center is at the same world z (legs are
        // mirror-symmetric in z), well above the origin -- and composed, so the
        // calf z (base 0.445 - calf 0.213 = 0.232) is NOT the raw local translate.
        const float foot_z = got[cooked.feet[0].calf_local_link].position.z;
        ASSERT_GT(foot_z, 0.05f) << "rest foot height looks wrong";
        for (const auto& foot : cooked.feet) {
            EXPECT_NEAR(got[foot.calf_local_link].position.z, foot_z, 1.0e-4f);
        }
    }

    // --- (A2) FK at NONZERO q matches host-side FK ------------------------------
    // q=0 leaves every joint/local rotation at identity, so the device
    // QuatMul / QuatFromAxisAngle / ComposeTransform-rotation paths are never
    // exercised by (A). Bend a few joints so the rotation math actually runs and
    // cross-check composed child positions (which fold in all ancestor
    // rotations) against the independent host FK. Composed positions sidestep the
    // q/-q quaternion-sign ambiguity.
    {
        auto bent = base;
        // Bend the first three non-fixed joints (a hip/thigh/calf chain on one
        // leg) with the same angles the chain-Jacobian test uses.
        const float kAngles[3] = {0.60f, 0.80f, -1.50f};
        uint32_t set = 0u;
        for (uint32_t link = 0u; link < base_link_count && set < 3u; ++link) {
            if (bent.joint_type[link] != articulation::ArticulationJointType::Fixed) {
                bent.q[link] = kAngles[set++];
            }
        }
        ASSERT_EQ(set, 3u) << "Go2 must have at least three non-fixed joints";

        const std::vector<Transform> got = run_device_fk(bent);
        const std::vector<Transform> want = HostForwardKinematics(bent);
        ASSERT_EQ(got.size(), want.size());

        // Loosened tolerance: device sinf/cosf vs host std::sin/cos differ
        // slightly and that error compounds over the chain.
        const float kPosTol = 1.0e-4f;
        bool any_moved = false;
        for (uint32_t link = 0u; link < base_link_count; ++link) {
            EXPECT_NEAR(got[link].position.x, want[link].position.x, kPosTol)
                << "bent link " << link;
            EXPECT_NEAR(got[link].position.y, want[link].position.y, kPosTol)
                << "bent link " << link;
            EXPECT_NEAR(got[link].position.z, want[link].position.z, kPosTol)
                << "bent link " << link;
            // Direct rotation check via a probe vector (avoids q/-q ambiguity).
            const Vec3 probe{0.3f, -0.4f, 0.6f};
            const Vec3 rg = got[link].rotation.Rotate(probe);
            const Vec3 rw = want[link].rotation.Rotate(probe);
            EXPECT_NEAR(rg.x, rw.x, kPosTol) << "bent link rot " << link;
            EXPECT_NEAR(rg.y, rw.y, kPosTol) << "bent link rot " << link;
            EXPECT_NEAR(rg.z, rw.z, kPosTol) << "bent link rot " << link;
        }
        // Guard against a vacuous pass: the bent pose must actually differ from
        // the rest pose somewhere down the bent leg.
        const std::vector<Transform> rest = HostForwardKinematics(base);
        for (uint32_t link = 0u; link < base_link_count; ++link) {
            if (std::fabs(got[link].position.x - rest[link].position.x) > 1.0e-3f ||
                std::fabs(got[link].position.z - rest[link].position.z) > 1.0e-3f) {
                any_moved = true;
                break;
            }
        }
        EXPECT_TRUE(any_moved) << "bent FK indistinguishable from rest; the "
                                  "rotation math was not exercised";
    }

    // --- (B) Replicate, run FK + contact detection, assert determinism ---------
    constexpr uint32_t kEnvCount = 64u;
    auto batched = articulation::ReplicateArticulationHostState(base, kEnvCount);
    ASSERT_EQ(batched.TotalLinkCount(), base_link_count * kEnvCount);

    auto batched_device = articulation::UploadArticulationState(context, batched);
    const uint32_t total_links = batched.TotalLinkCount();

    OwnedDeviceBuffer pose_buf(static_cast<size_t>(total_links) * sizeof(Transform));
    articulation::UpdateWorldLinkPoses(
        context, batched_device.View(), static_cast<Transform*>(pose_buf.Data()));

    // Ground plane just ABOVE the rest feet so all four penetrate. From the FK
    // sanity check the rest foot z is ~0.232; pick a ground above the centers.
    constexpr float kGroundHeight = 0.25f;

    OwnedDeviceBuffer feet_buf(cooked.feet.size() * sizeof(articulation::FootShape));
    feet_buf.CopyFromHost(cooked.feet.data(),
                          cooked.feet.size() * sizeof(articulation::FootShape));

    const uint32_t slot_count = kEnvCount * articulation::kMaxFootContactsPerEnv;
    OwnedDeviceBuffer link_buf(slot_count * sizeof(uint32_t));
    OwnedDeviceBuffer point_buf(slot_count * sizeof(Vec3));
    OwnedDeviceBuffer normal_buf(slot_count * sizeof(Vec3));
    OwnedDeviceBuffer depth_buf(slot_count * sizeof(float));
    OwnedDeviceBuffer count_buf(kEnvCount * sizeof(uint32_t));

    articulation::DetectFootGroundContacts(
        context,
        static_cast<const Transform*>(pose_buf.Data()),
        static_cast<const articulation::FootShape*>(feet_buf.Data()),
        foot_count,
        kEnvCount,
        base_link_count,
        kGroundHeight,
        static_cast<uint32_t*>(link_buf.Data()),
        static_cast<Vec3*>(point_buf.Data()),
        static_cast<Vec3*>(normal_buf.Data()),
        static_cast<float*>(depth_buf.Data()),
        static_cast<uint32_t*>(count_buf.Data()));
    context.stream.Synchronize();

    std::vector<uint32_t> counts(kEnvCount);
    std::vector<uint32_t> links(slot_count);
    std::vector<Vec3> points(slot_count);
    std::vector<Vec3> normals(slot_count);
    std::vector<float> depths(slot_count);
    count_buf.CopyToHost(counts.data(), counts.size() * sizeof(uint32_t));
    link_buf.CopyToHost(links.data(), links.size() * sizeof(uint32_t));
    point_buf.CopyToHost(points.data(), points.size() * sizeof(Vec3));
    normal_buf.CopyToHost(normals.data(), normals.size() * sizeof(Vec3));
    depth_buf.CopyToHost(depths.data(), depths.size() * sizeof(float));

    // Every env penetrates all four feet at rest under this ground plane.
    const uint32_t kExpectedCount = foot_count;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        EXPECT_GT(counts[env], 0u) << "env " << env;
        EXPECT_LE(counts[env], articulation::kMaxFootContactsPerEnv) << "env " << env;
        EXPECT_EQ(counts[env], kExpectedCount) << "env " << env;
    }

    // Per-contact assertions on env 0, and determinism vs every other env.
    const uint32_t env0_base = 0u;
    for (uint32_t slot = 0u; slot < counts[0]; ++slot) {
        const uint32_t i = env0_base + slot;
        // Contact point lies on the ground surface.
        EXPECT_NEAR(points[i].z, kGroundHeight, 1.0e-5f) << "slot " << slot;
        // Normal is +Z.
        EXPECT_NEAR(normals[i].x, 0.0f, 1.0e-6f);
        EXPECT_NEAR(normals[i].y, 0.0f, 1.0e-6f);
        EXPECT_NEAR(normals[i].z, 1.0f, 1.0e-6f);
        // Positive penetration depth.
        EXPECT_GT(depths[i], 0.0f) << "slot " << slot;
        // Contact link is a valid base-range calf link.
        EXPECT_LT(links[i], base_link_count) << "slot " << slot;
    }

    // Determinism: every replica's slot block is bit-identical to env 0's, with
    // contact links shifted by the replica's link-block offset.
    size_t mismatches = 0u;
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        EXPECT_EQ(counts[env], counts[0]) << "count mismatch env " << env;
        const uint32_t off = env * articulation::kMaxFootContactsPerEnv;
        for (uint32_t slot = 0u; slot < counts[0]; ++slot) {
            const uint32_t a = off + slot;
            const uint32_t b = env0_base + slot;
            if (depths[a] != depths[b] || points[a].x != points[b].x ||
                points[a].y != points[b].y || points[a].z != points[b].z ||
                normals[a].z != normals[b].z ||
                links[a] != links[b] + env * base_link_count) {
                if (mismatches < 8u) {
                    ADD_FAILURE() << "replica " << env << " slot " << slot
                                  << " diverged from env 0";
                }
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << mismatches << " contact slots diverged across "
                              << kEnvCount << " replicas";

    // --- (C) Ground BELOW all feet: no contacts, slots cleared -----------------
    // The penetrating case above keeps count==4 for every env, so it never
    // exercises the zero-count / slot-clear path. Re-run with the ground well
    // below the rest feet (z=0): every env must report zero contacts and all of
    // its slots must be cleared (link == kInvalidLink, zeroed point/normal/depth).
    constexpr float kGroundBelow = 0.0f;
    articulation::DetectFootGroundContacts(
        context,
        static_cast<const Transform*>(pose_buf.Data()),
        static_cast<const articulation::FootShape*>(feet_buf.Data()),
        foot_count,
        kEnvCount,
        base_link_count,
        kGroundBelow,
        static_cast<uint32_t*>(link_buf.Data()),
        static_cast<Vec3*>(point_buf.Data()),
        static_cast<Vec3*>(normal_buf.Data()),
        static_cast<float*>(depth_buf.Data()),
        static_cast<uint32_t*>(count_buf.Data()));
    context.stream.Synchronize();

    count_buf.CopyToHost(counts.data(), counts.size() * sizeof(uint32_t));
    link_buf.CopyToHost(links.data(), links.size() * sizeof(uint32_t));
    depth_buf.CopyToHost(depths.data(), depths.size() * sizeof(float));

    constexpr uint32_t kInvalidLink = ~0u;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        EXPECT_EQ(counts[env], 0u) << "env " << env << " should have no contacts";
    }
    for (uint32_t slot = 0u; slot < slot_count; ++slot) {
        EXPECT_EQ(links[slot], kInvalidLink) << "slot " << slot << " not cleared";
        EXPECT_EQ(depths[slot], 0.0f) << "slot " << slot << " depth not cleared";
    }
}
