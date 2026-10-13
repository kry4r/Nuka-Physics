
#include "collision/aabb.hpp"
#include "collision/link_aabb.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>
#include "nk/pipeline/world.hpp"
#include "phi/op_schema.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

namespace {

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
using nuka::collision::AABB;
using nuka::collision::ExtractLinkShapeAabbs;
using nuka::collision::LinkShapeAabbs;
using nuka::collision::ShapeWorldAabb;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

struct CookedFloat {
    articulation::ArticulationHostState host;
    nuka::scene::CookedShapeTable shapes;  // the cooked shape table (foot spheres).
};

CookedFloat CookGo2Float() {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedFloat result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    result.shapes = world.template_view.shape_table;
    return result;
}

std::vector<Transform> ForwardKinematics(cudaStream_t, int,
                                         const articulation::ArticulationHostState& host) {
    auto* device = nuka::phi::InitBestDevice();
    auto* backend = nuka::phi::DeviceInitBackend(device, nullptr);
    const auto scene = nuka::import::LoadUsd(SourcePath("examples/scenes/go2_float.usda").string());
    auto cooked = nuka::scene::cook::CookToModel(scene, 1u);
    std::vector<Transform> poses(host.TotalLinkCount());
    {
        nuka::nk::World world(std::move(cooked.model), 1u, device, backend);
        EXPECT_TRUE(world.Ready());
        EXPECT_TRUE(world.GetData().UploadField(nuka::nk::FieldId::Q,
            host.q.data(), host.q.size() * sizeof(float)));
        EXPECT_TRUE(world.GetData().UploadField(nuka::nk::FieldId::BasePose,
            host.base_pose.data(), host.base_pose.size() * sizeof(Transform)));
        nuka::phi::FkWorldPosesParams params{};
        params.articulation_count = host.ArticulationCount();
        params.total_link_count = host.TotalLinkCount();
        EXPECT_EQ(world.DispatchOp(nuka::phi::NkOp::FkWorldPoses, &params),
                  nuka::phi::Status::Ok);
        nuka::phi::BackendSynchronize(backend);
        EXPECT_TRUE(world.GetData().DownloadField(nuka::nk::FieldId::LinkPose,
            poses.data(), poses.size() * sizeof(Transform)));
    }
    nuka::phi::BackendFree(backend);
    return poses;
}

// Indices of the cooked foot SPHERE shapes (proven-available on go2_float calves).
std::vector<uint32_t> FootSphereShapes(const CookedFloat& cooked) {
    std::vector<uint32_t> out;
    for (uint32_t s = 0u; s < cooked.shapes.types.size(); ++s) {
        if (cooked.shapes.types[s] == ShapeType::Sphere) out.push_back(s);
    }
    return out;
}

// 13 fixed unit sample directions on the sphere surface (axes + sign-pattern
// oblique). NO randomness -- fixed set so the test is deterministic.
std::array<Vec3, 13> SphereSampleDirs() {
    const float k = 0.57735026919f;  // 1/sqrt(3)
    return {Vec3{ 1, 0, 0}, Vec3{-1, 0, 0}, Vec3{0,  1, 0}, Vec3{0, -1, 0},
            Vec3{ 0, 0, 1}, Vec3{ 0, 0,-1}, Vec3{k,  k, k}, Vec3{-k,-k,-k},
            Vec3{ k,-k, k}, Vec3{-k, k,-k}, Vec3{k,  k,-k}, Vec3{-k,-k, k},
            Vec3{ k,-k,-k}};
}

// --- the FK pose sweep: 4 distinct articulation poses --------------------------

void ApplyPosePerturbation(articulation::ArticulationHostState& host, int pose_idx) {
    // Base orientation: rotate about a mixed axis by a pose-dependent angle.
    const Vec3 axis{0.3f, 0.7f, 0.64807407f};  // ~unit mixed axis.
    const float base_angles[4] = {0.0f, 0.35f, -0.5f, 0.8f};
    const float base_dz[4]     = {0.0f, 0.05f, -0.07f, 0.12f};
    host.base_pose[0].rotation = Quat::FromAxisAngle(axis, base_angles[pose_idx]);
    host.base_pose[0].position.z += base_dz[pose_idx];

    const uint32_t link_count = host.TotalLinkCount();
    const float joint_angles[4] = {0.0f, 0.4f, -0.6f, 0.9f};
    if (link_count >= 3u) {
        host.q[link_count - 1u] = joint_angles[pose_idx];
        host.q[link_count - 2u] = -0.5f * joint_angles[pose_idx];
    }
}

TEST(LinkAabb, BoundsUnderFkPoseSweep) {
    const cudaStream_t context = nullptr;
    const int context_dev = 0;
    const auto base_cooked = CookGo2Float();
    const auto foot_shapes = FootSphereShapes(base_cooked);
    ASSERT_GE(foot_shapes.size(), 1u) << "go2_float must expose foot spheres";

    const auto dirs = SphereSampleDirs();
    const float kEps = 1e-4f;        // sample-inside slack.
    const float kTightEps = 1e-5f;   // sphere AABB is exact: extent == 2r.

    float worst_outside = 0.0f;
    float worst_tightness = 0.0f;
    // Per-pose world centers of foot 0, to prove the feet actually MOVE.
    std::vector<Vec3> foot0_centers;

    for (int pose = 0; pose < 4; ++pose) {
        auto cooked = base_cooked;  // fresh copy -> apply this pose's perturbation.
        ApplyPosePerturbation(cooked.host, pose);
        const auto poses = ForwardKinematics(context, context_dev, cooked.host);

        // FK link poses + the shape->body->link map feed the host convenience.
        const auto extracted =
            ExtractLinkShapeAabbs(cooked.shapes, poses, cooked.host.link_body);

        uint32_t expected = 0u;
        for (uint32_t s = 0u; s < cooked.shapes.types.size(); ++s) {
            const ShapeType t = cooked.shapes.types[s];
            if (t != ShapeType::Sphere && t != ShapeType::Box &&
                t != ShapeType::Capsule) {
                continue;
            }
            const uint32_t b = cooked.shapes.body_ids[s];
            for (uint32_t l = 0u; l < cooked.host.TotalLinkCount(); ++l) {
                if (cooked.host.link_body[l] == b) { ++expected; break; }
            }
        }
        ASSERT_EQ(extracted.aabbs.size(), expected)
            << "one AABB per link-mapped Sphere/Box/Capsule shape";
        ASSERT_GE(expected, foot_shapes.size() + 1u)
            << "expected the foot spheres AND the base cube";

        bool first_sphere = true;
        for (size_t i = 0; i < extracted.aabbs.size(); ++i) {
            const uint32_t shape = extracted.source_shape_ids[i];
            const ShapeType type = cooked.shapes.types[shape];
            const AABB& box = extracted.aabbs[i];

            // Non-vacuous (every emitted shape).
            ASSERT_LT(box.min.x, box.max.x);
            ASSERT_LT(box.min.y, box.max.y);
            ASSERT_LT(box.min.z, box.max.z);

            // Owner link world pose (shape -> body -> first link with that body).
            const uint32_t body = cooked.shapes.body_ids[shape];
            uint32_t owner = kInvalidLink;
            for (uint32_t l = 0u; l < cooked.host.TotalLinkCount(); ++l) {
                if (cooked.host.link_body[l] == body) { owner = l; break; }
            }
            ASSERT_NE(owner, kInvalidLink);
            const Transform& link = poses[owner];
            const Transform local = shape < cooked.shapes.local_transforms.size()
                                        ? cooked.shapes.local_transforms[shape]
                                        : Transform::Identity();

            if (type == ShapeType::Sphere) {
                const float r = cooked.shapes.radii[shape];
                const Vec3 center =
                    link.position + link.rotation.Rotate(local.position);
                if (first_sphere) {
                    foot0_centers.push_back(center);
                    first_sphere = false;
                }

                // EXACT tightness: a sphere's world AABB extent == 2r per axis.
                worst_tightness = std::max(worst_tightness,
                                           std::abs((box.max.x - box.min.x) - 2.0f * r));
                worst_tightness = std::max(worst_tightness,
                                           std::abs((box.max.y - box.min.y) - 2.0f * r));
                worst_tightness = std::max(worst_tightness,
                                           std::abs((box.max.z - box.min.z) - 2.0f * r));
                EXPECT_NEAR(box.max.x - box.min.x, 2.0f * r, kTightEps);
                EXPECT_NEAR(box.max.y - box.min.y, 2.0f * r, kTightEps);
                EXPECT_NEAR(box.max.z - box.min.z, 2.0f * r, kTightEps);

                // Every sampled surface point must lie inside [min,max].
                for (const Vec3& d : dirs) {
                    const Vec3 p = center + d * r;  // sphere is rotation-invariant.
                    const float ox = std::max(box.min.x - p.x, p.x - box.max.x);
                    const float oy = std::max(box.min.y - p.y, p.y - box.max.y);
                    const float oz = std::max(box.min.z - p.z, p.z - box.max.z);
                    worst_outside = std::max(worst_outside, std::max({ox, oy, oz}));
                    EXPECT_LE(p.x, box.max.x + kEps);
                    EXPECT_GE(p.x, box.min.x - kEps);
                    EXPECT_LE(p.y, box.max.y + kEps);
                    EXPECT_GE(p.y, box.min.y - kEps);
                    EXPECT_LE(p.z, box.max.z + kEps);
                    EXPECT_GE(p.z, box.min.z - kEps);
                }
            } else if (type == ShapeType::Box) {
                const Vec3 he = shape < cooked.shapes.half_extents.size()
                                    ? cooked.shapes.half_extents[shape]
                                    : Vec3::Zero();
                const Quat world_rot = (link.rotation * local.rotation).Normalized();
                const Vec3 world_center =
                    link.rotation.Rotate(local.position) + link.position;
                for (int c = 0; c < 8; ++c) {
                    const Vec3 corner{(c & 1) ? he.x : -he.x,
                                      (c & 2) ? he.y : -he.y,
                                      (c & 4) ? he.z : -he.z};
                    const Vec3 w = world_rot.Rotate(corner) + world_center;
                    const float ox = std::max(box.min.x - w.x, w.x - box.max.x);
                    const float oy = std::max(box.min.y - w.y, w.y - box.max.y);
                    const float oz = std::max(box.min.z - w.z, w.z - box.max.z);
                    worst_outside = std::max(worst_outside, std::max({ox, oy, oz}));
                    EXPECT_LE(w.x, box.max.x + kEps);
                    EXPECT_GE(w.x, box.min.x - kEps);
                    EXPECT_LE(w.y, box.max.y + kEps);
                    EXPECT_GE(w.y, box.min.y - kEps);
                    EXPECT_LE(w.z, box.max.z + kEps);
                    EXPECT_GE(w.z, box.min.z - kEps);
                }
            }
        }
    }

    // The feet must actually move across poses (else bounds pass trivially).
    ASSERT_GE(foot0_centers.size(), 2u);
    float max_motion = 0.0f;
    for (size_t i = 1; i < foot0_centers.size(); ++i) {
        max_motion = std::max(max_motion,
                              (foot0_centers[i] - foot0_centers[0]).Length());
    }
    EXPECT_GT(max_motion, 1e-2f)
        << "pose sweep must actually move the feet (got " << max_motion << ")";

    std::printf("[BoundsUnderFkPoseSweep] max sample-outside-AABB = %.3e; "
                "sphere tightness err = %.3e; max foot motion = %.4f m\n",
                worst_outside, worst_tightness, max_motion);
    EXPECT_LE(worst_outside, kEps);
    EXPECT_LE(worst_tightness, kTightEps);
}

// --- synthetic box under rotation (independent 8-corner reference) --------------

TEST(LinkAabb, BoxAabbUnderRotation) {
    const Vec3 half_extents{0.4f, 0.9f, 0.6f};  // unit-ish, distinct per axis.

    // Several nontrivial (link, local) rotation+translation cases.
    struct Case { Transform link; Transform local; };
    const std::array<Case, 4> cases = {{
        // (1) link rotated 45 deg about Z, local identity, translated.
        {Transform{Vec3{1.0f, -2.0f, 0.5f},
                   Quat::FromAxisAngle(Vec3{0, 0, 1}, 0.78539816f)},
         Transform::Identity()},
        // (2) link rotated 30 deg about a mixed axis, local identity.
        {Transform{Vec3{-0.3f, 0.2f, 1.1f},
                   Quat::FromAxisAngle(Vec3{0.5f, 0.5f, 0.70710678f}, 0.52359878f)},
         Transform::Identity()},
        // (3) link identity but LOCAL rotated 45 deg about X (proves local rot
        //     is honored, not dropped) + local offset.
        {Transform{Vec3{2.0f, 2.0f, 2.0f}, Quat::Identity()},
         Transform{Vec3{0.1f, -0.2f, 0.3f},
                   Quat::FromAxisAngle(Vec3{1, 0, 0}, 0.78539816f)}},
        // (4) BOTH link and local rotated about mixed axes + both offset.
        {Transform{Vec3{-1.0f, 0.5f, -0.5f},
                   Quat::FromAxisAngle(Vec3{0.2f, 0.9f, 0.38729833f}, 0.6f)},
         Transform{Vec3{0.05f, 0.15f, -0.1f},
                   Quat::FromAxisAngle(Vec3{0.7f, 0.1f, 0.70710678f}, 0.9f)}},
    }};

    float worst_err = 0.0f;
    bool saw_larger_than_aabb = false;
    const float kEps = 1e-5f;

    for (const Case& c : cases) {
        const AABB box = ShapeWorldAabb(ShapeType::Box, c.link, c.local,
                                        half_extents, 0.0f, 0.0f);

        // INDEPENDENT 8-corner reference (host-side Quat::Rotate is fine here --
        // this is a different code path than the module's MulQuat/RotateByQuat).
        const Quat world_rot = (c.link.rotation * c.local.rotation).Normalized();
        const Vec3 world_center =
            c.link.rotation.Rotate(c.local.position) + c.link.position;
        Vec3 rmin{ FLT_MAX,  FLT_MAX,  FLT_MAX};
        Vec3 rmax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (int i = 0; i < 8; ++i) {
            const Vec3 corner{(i & 1) ? half_extents.x : -half_extents.x,
                              (i & 2) ? half_extents.y : -half_extents.y,
                              (i & 4) ? half_extents.z : -half_extents.z};
            const Vec3 w = world_rot.Rotate(corner) + world_center;
            rmin.x = std::min(rmin.x, w.x); rmax.x = std::max(rmax.x, w.x);
            rmin.y = std::min(rmin.y, w.y); rmax.y = std::max(rmax.y, w.y);
            rmin.z = std::min(rmin.z, w.z); rmax.z = std::max(rmax.z, w.z);
        }

        worst_err = std::max({worst_err,
                              std::abs(box.min.x - rmin.x), std::abs(box.max.x - rmax.x),
                              std::abs(box.min.y - rmin.y), std::abs(box.max.y - rmax.y),
                              std::abs(box.min.z - rmin.z), std::abs(box.max.z - rmax.z)});
        EXPECT_NEAR(box.min.x, rmin.x, kEps);
        EXPECT_NEAR(box.max.x, rmax.x, kEps);
        EXPECT_NEAR(box.min.y, rmin.y, kEps);
        EXPECT_NEAR(box.max.y, rmax.y, kEps);
        EXPECT_NEAR(box.min.z, rmin.z, kEps);
        EXPECT_NEAR(box.max.z, rmax.z, kEps);

        // Rotated box's AABB must be LARGER than the axis-aligned one (same center,
        // identity rotation) along at least one axis -> proves rotation handled.
        const AABB aa = ShapeWorldAabb(
            ShapeType::Box, Transform{c.link.position, Quat::Identity()},
            Transform{c.local.position, Quat::Identity()}, half_extents, 0.0f, 0.0f);
        const Vec3 rot_ext{box.max.x - box.min.x, box.max.y - box.min.y,
                           box.max.z - box.min.z};
        const Vec3 aa_ext{aa.max.x - aa.min.x, aa.max.y - aa.min.y,
                          aa.max.z - aa.min.z};
        if (rot_ext.x > aa_ext.x + 1e-3f || rot_ext.y > aa_ext.y + 1e-3f ||
            rot_ext.z > aa_ext.z + 1e-3f) {
            saw_larger_than_aabb = true;
        }
        // The rotated AABB is never SMALLER than axis-aligned on any axis.
        EXPECT_GE(rot_ext.x, aa_ext.x - kEps);
        EXPECT_GE(rot_ext.y, aa_ext.y - kEps);
        EXPECT_GE(rot_ext.z, aa_ext.z - kEps);
    }

    EXPECT_TRUE(saw_larger_than_aabb)
        << "a rotated box must produce a larger AABB than its axis-aligned form";
    std::printf("[BoxAabbUnderRotation] max |module - 8corner ref| = %.3e\n", worst_err);
    EXPECT_LE(worst_err, kEps);
}

// --- synthetic capsule (caps contained) ----------------------------------------

TEST(LinkAabb, CapsuleAabbContainsEndpoints) {
    const float radius = 0.25f;
    const float half_height = 0.7f;
    const Transform link{Vec3{0.5f, -0.5f, 1.0f},
                         Quat::FromAxisAngle(Vec3{0.3f, 0.6f, 0.74161985f}, 0.6f)};
    const Transform local{Vec3{0.1f, 0.0f, -0.2f},
                          Quat::FromAxisAngle(Vec3{1, 0, 0}, 0.4f)};

    const AABB box = ShapeWorldAabb(ShapeType::Capsule, link, local,
                                    Vec3::Zero(), radius, half_height);
    ASSERT_LT(box.min.x, box.max.x);
    ASSERT_LT(box.min.y, box.max.y);
    ASSERT_LT(box.min.z, box.max.z);

    // The two cap centers in world follow the local Z axis through link*local.
    const Quat world_rot = (link.rotation * local.rotation).Normalized();
    const Vec3 axis = world_rot.Rotate(Vec3{0, 0, 1});
    const Vec3 center = link.rotation.Rotate(local.position) + link.position;
    const Vec3 cap_a = center + axis * half_height;
    const Vec3 cap_b = center - axis * half_height;

    const auto dirs = SphereSampleDirs();
    const float kEps = 1e-4f;
    float worst_outside = 0.0f;
    for (const Vec3& cap : {cap_a, cap_b}) {
        for (const Vec3& d : dirs) {
            const Vec3 p = cap + d * radius;
            const float ox = std::max(box.min.x - p.x, p.x - box.max.x);
            const float oy = std::max(box.min.y - p.y, p.y - box.max.y);
            const float oz = std::max(box.min.z - p.z, p.z - box.max.z);
            worst_outside = std::max(worst_outside, std::max({ox, oy, oz}));
            EXPECT_LE(p.x, box.max.x + kEps);
            EXPECT_GE(p.x, box.min.x - kEps);
            EXPECT_LE(p.y, box.max.y + kEps);
            EXPECT_GE(p.y, box.min.y - kEps);
            EXPECT_LE(p.z, box.max.z + kEps);
            EXPECT_GE(p.z, box.min.z - kEps);
        }
    }
    std::printf("[CapsuleAabbContainsEndpoints] max cap sample-outside-AABB = %.3e\n",
                worst_outside);
    EXPECT_LE(worst_outside, kEps);
}

TEST(LinkAabb, DeterministicTwoRun) {
    const cudaStream_t context = nullptr;
    const int context_dev = 0;
    const auto cooked = CookGo2Float();

    auto run = [&]() {
        const auto poses = ForwardKinematics(context, context_dev, cooked.host);
        return ExtractLinkShapeAabbs(cooked.shapes, poses, cooked.host.link_body);
    };
    const LinkShapeAabbs a = run();
    const LinkShapeAabbs b = run();

    ASSERT_EQ(a.aabbs.size(), b.aabbs.size());
    ASSERT_GE(a.aabbs.size(), 1u);
    // memcmp the raw AABB span (24*N bytes) -> byte-identical (dodges padding).
    EXPECT_EQ(0, std::memcmp(a.aabbs.data(), b.aabbs.data(),
                             a.aabbs.size() * sizeof(AABB)));
    EXPECT_EQ(a.shape_body_ids, b.shape_body_ids);
    EXPECT_EQ(a.source_shape_ids, b.source_shape_ids);
}

// --- device-callability proof (HD core on the GPU == host) ---------------------

__global__ void DeviceCoreKernel(Transform link, Transform local,
                                 Vec3 half_extents, float radius, float half_height,
                                 AABB* out /* [3]: sphere, box, capsule */) {
    const int t = threadIdx.x;
    if (t == 0) out[0] = ShapeWorldAabb(ShapeType::Sphere, link, local,
                                        half_extents, radius, half_height);
    if (t == 1) out[1] = ShapeWorldAabb(ShapeType::Box, link, local,
                                        half_extents, radius, half_height);
    if (t == 2) out[2] = ShapeWorldAabb(ShapeType::Capsule, link, local,
                                        half_extents, radius, half_height);
}

TEST(LinkAabb, DeviceCoreMatchesHost) {
    const cudaStream_t context = nullptr;
    const int context_dev = 0;

    const Transform link{Vec3{0.7f, -1.3f, 0.9f},
                         Quat::FromAxisAngle(Vec3{0.2f, 0.7f, 0.68f}, 0.55f)};
    const Transform local{Vec3{0.05f, -0.1f, 0.2f},
                          Quat::FromAxisAngle(Vec3{1, 0, 0}, 0.4f)};
    const Vec3 half_extents{0.3f, 0.5f, 0.4f};
    const float radius = 0.22f;
    const float half_height = 0.6f;

    // Host reference.
    AABB host_out[3] = {
        ShapeWorldAabb(ShapeType::Sphere, link, local, half_extents, radius, half_height),
        ShapeWorldAabb(ShapeType::Box, link, local, half_extents, radius, half_height),
        ShapeWorldAabb(ShapeType::Capsule, link, local, half_extents, radius, half_height),
    };

    // Device.
    OwnedDeviceBuffer out_buf(3 * sizeof(AABB));
    DeviceCoreKernel<<<1, 3, 0, context>>>(
        link, local, half_extents, radius, half_height,
        static_cast<AABB*>(out_buf.Data()));
    cudaStreamSynchronize(context);
    AABB dev_out[3];
    out_buf.CopyToHost(dev_out, sizeof(dev_out));

    float max_diff = 0.0f;
    for (int s = 0; s < 3; ++s) {
        max_diff = std::max({max_diff,
            std::abs(host_out[s].min.x - dev_out[s].min.x),
            std::abs(host_out[s].min.y - dev_out[s].min.y),
            std::abs(host_out[s].min.z - dev_out[s].min.z),
            std::abs(host_out[s].max.x - dev_out[s].max.x),
            std::abs(host_out[s].max.y - dev_out[s].max.y),
            std::abs(host_out[s].max.z - dev_out[s].max.z)});
    }
    std::printf("[DeviceCoreMatchesHost] max |device - host| = %.3e\n", max_diff);
    EXPECT_LE(max_diff, 1e-5f)
        << "device HD core diverges from host beyond fp-contraction tolerance";
}

}  // namespace
