// ---------------------------------------------------------------------------
// v0.7 p04 DIFFERENTIAL GATE: the self-written LBVH broadphase must produce the
// SAME candidate-pair SET as the reference SAP broadphase.
//
// Oracle split (a conscious feasibility decision -- see the p04 report):
//   * Small smoke scenes: oracle is the GPU SAP path (collision::gpu::
//     BuildCudaBroadphase). The per-shape AABBs are computed host-side (the same
//     Sphere/Plane/Box formulas the removed device AABB kernel used) and
//     uploaded once; BOTH the SAP oracle and the LBVH run over that one device
//     AABB buffer, so it stays apples-to-apples.
//   * 50k random AABBs: running GPU SAP is INFEASIBLE (its O(n^2) pair-slot
//     layout is ~1.25e9 slots ~= 10 GB). The SAP set is DEFINITIONALLY the set
//     of all overlapping {i<j} pairs, so we use a CPU brute-force O(n^2) overlap
//     oracle (cheap: ~1.25e9 compares, stores only true overlaps).
//
// Also asserts D1: two LBVH runs over the same AABBs produce a BYTE-IDENTICAL
// compact sorted pair list.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/broadphase_lbvh.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "collision/gpu/broadphase.cuh"
#include "math/transform.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer_v2.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

// Test-only RAII over the phi v2 opaque Buffer* (stream-0 DeviceBufferType;
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy). Mirrors the legacy phi::Buffer surface used by the test bodies so they
// stay byte-identical after the buffer sweep.
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    explicit OwnedDeviceBuffer(nuka::phi::Buffer* buf) : buf_(buf) {}
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(OwnedDeviceBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) nuka::phi::BufferFree(buf_);
            buf_ = o.buf_; o.buf_ = nullptr;
        }
        return *this;
    }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
private:
    nuka::phi::Buffer* buf_ = nullptr;
};

template <typename T>
OwnedDeviceBuffer UploadOwned(const std::vector<T>& values) {
    return OwnedDeviceBuffer(nuka::phi::UploadVectorV2(
        nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), values));
}

using CanonPair = std::pair<uint32_t, uint32_t>;

std::set<CanonPair> ToSet(const std::vector<collision::CollisionPair>& pairs) {
    std::set<CanonPair> out;
    for (const auto& p : pairs) {
        const uint32_t a = std::min(p.body_a, p.body_b);
        const uint32_t b = std::max(p.body_a, p.body_b);
        out.emplace(a, b);
    }
    return out;
}

// CPU brute-force overlap oracle == the SAP set by definition.
std::set<CanonPair> CpuBruteForceSet(const std::vector<collision::AABB>& aabbs) {
    std::set<CanonPair> out;
    const uint32_t n = static_cast<uint32_t>(aabbs.size());
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1u; j < n; ++j) {
            if (aabbs[i].Overlaps(aabbs[j])) {
                out.emplace(i, j);
            }
        }
    }
    return out;
}

// --- Small smoke scenes (reuse the CUDA-contacts test scenes) --------------
scene::SceneIR BuildThreeBoxScene() {
    scene::SceneIR scene;
    const math::Vec3 positions[] = {
        {0.0f, 0.0f, 0.0f}, {0.75f, 0.0f, 0.0f}, {1.5f, 0.0f, 0.0f}};
    for (math::Vec3 position : positions) {
        scene::RigidBodyRecord body;
        body.name = "box";
        body.mass = 1.0f;
        body.local_transform.position = position;
        const auto body_id = scene.AddRigidBody(std::move(body));
        scene::CollisionShapeRecord shape;
        shape.body_id = body_id;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        scene.AddCollisionShape(std::move(shape));
    }
    return scene;
}

scene::SceneIR BuildPlaneBoxScene() {
    scene::SceneIR scene;
    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));
    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    scene::RigidBodyRecord box;
    box.name = "box";
    box.mass = 1.0f;
    box.inertia = {1.0f, 1.0f, 1.0f};
    box.local_transform.position = {0.0f, 0.45f, 0.0f};
    const auto box_id = scene.AddRigidBody(std::move(box));
    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_id;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.5f, 0.5f, 0.5f};
    scene.AddCollisionShape(std::move(box_shape));
    return scene;
}

// Compute the per-shape world-space AABBs host-side using the SAME Sphere /
// Plane / Box formulas the (removed) device AABB kernel used: the world
// transform is Compose(body_pose, shape_local_transform); Sphere -> center +/-
// radius; Plane -> a wide thin slab at the plane height; Box -> 8-corner OBB
// bound. This reproduces the exact AABB values the old GPU-SAP path produced.
std::vector<collision::AABB> ComputeSceneAabbs(const scene::SceneIR& scene) {
    const runtime::BuiltWorld world = runtime::BuildWorld(scene::CookScene(scene));
    const auto& shapes = world.template_view.shape_table;
    const auto& poses = world.instance.poses;

    const uint32_t shape_count = static_cast<uint32_t>(shapes.types.size());
    std::vector<collision::AABB> aabbs(shape_count);
    for (uint32_t s = 0; s < shape_count; ++s) {
        const auto body_id = shapes.body_ids[s];
        const math::Transform world_transform =
            poses[body_id] * shapes.local_transforms[s];

        switch (shapes.types[s]) {
        case scene::ShapeType::Sphere:
            aabbs[s] = collision::AABB::FromSphere(world_transform.position,
                                                   shapes.radii[s]);
            break;
        case scene::ShapeType::Plane: {
            const float plane_y = world_transform.position.y;
            aabbs[s].min = {-1.0e6f, plane_y - 0.01f, -1.0e6f};
            aabbs[s].max = {1.0e6f, plane_y + 0.01f, 1.0e6f};
            break;
        }
        default:
            aabbs[s] = collision::AABB::FromBox(world_transform, shapes.half_extents[s]);
            break;
        }
    }
    return aabbs;
}

// Upload the host-computed scene AABBs once, then run BOTH the GPU-SAP oracle and
// the LBVH over that same device AABB buffer, and assert the reduced SETs match.
void ExpectLbvhEqualsSapOnScene(const scene::SceneIR& scene) {
    const std::vector<collision::AABB> aabbs = ComputeSceneAabbs(scene);
    const uint32_t n = static_cast<uint32_t>(aabbs.size());

    OwnedDeviceBuffer d_aabbs = UploadOwned(aabbs);
    const auto* device_aabbs = static_cast<const collision::AABB*>(d_aabbs.Data());

    auto sap = collision::gpu::BuildCudaBroadphase(device_aabbs, n);
    const auto sap_pairs = sap.DownloadPairs();

    auto lbvh = collision::gpu::BuildLbvhBroadphase(device_aabbs, n);
    const auto lbvh_pairs = lbvh.DownloadPairs();

    EXPECT_EQ(ToSet(sap_pairs), ToSet(lbvh_pairs));
}

} // namespace

TEST(LbvhVsSap, SamePairSetOnThreeBoxScene) {
    ExpectLbvhEqualsSapOnScene(BuildThreeBoxScene());
}

TEST(LbvhVsSap, SamePairSetOnPlaneBoxScene) {
    ExpectLbvhEqualsSapOnScene(BuildPlaneBoxScene());
}

TEST(LbvhVsSap, SamePairSetOn50kRandomAabbs) {
    constexpr uint32_t kCount = 50000u;
    // Deterministic random AABBs in a fixed domain. Small boxes in a large
    // volume keep the overlap fan-out modest (matches a realistic broadphase).
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> pos(-50.0f, 50.0f);
    std::uniform_real_distribution<float> ext(0.05f, 0.4f);

    std::vector<collision::AABB> aabbs(kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
        const float cx = pos(rng), cy = pos(rng), cz = pos(rng);
        const float ex = ext(rng), ey = ext(rng), ez = ext(rng);
        aabbs[i].min = {cx - ex, cy - ey, cz - ez};
        aabbs[i].max = {cx + ex, cy + ey, cz + ez};
    }

    // Upload to device.
    OwnedDeviceBuffer d_aabbs = UploadOwned(aabbs);
    const auto* device_aabbs = static_cast<const collision::AABB*>(d_aabbs.Data());

    // Generous pair cap (cheap: 16 bytes/pair). The CPU oracle confirms the true
    // count is well under this.
    auto lbvh = collision::gpu::BuildLbvhBroadphase(device_aabbs, kCount,
                                                    /*max_pairs_hint=*/4u * 1024u * 1024u);
    const auto lbvh_pairs = lbvh.DownloadPairs();

    const std::set<CanonPair> oracle = CpuBruteForceSet(aabbs);
    const std::set<CanonPair> got = ToSet(lbvh_pairs);

    ASSERT_LT(lbvh.PairCount(), lbvh.PairCapacity())
        << "pair cap too small; result was truncated";
    EXPECT_EQ(oracle, got)
        << "LBVH set (" << got.size() << ") != SAP/brute-force set ("
        << oracle.size() << ")";
}

TEST(LbvhVsSap, D1TwoRunByteExactPairList) {
    constexpr uint32_t kCount = 20000u;
    std::mt19937 rng(0x1234u);
    std::uniform_real_distribution<float> pos(-30.0f, 30.0f);
    std::uniform_real_distribution<float> ext(0.1f, 0.5f);

    std::vector<collision::AABB> aabbs(kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
        const float cx = pos(rng), cy = pos(rng), cz = pos(rng);
        const float ex = ext(rng), ey = ext(rng), ez = ext(rng);
        aabbs[i].min = {cx - ex, cy - ey, cz - ez};
        aabbs[i].max = {cx + ex, cy + ey, cz + ez};
    }
    OwnedDeviceBuffer d_aabbs = UploadOwned(aabbs);
    const auto* device_aabbs = static_cast<const collision::AABB*>(d_aabbs.Data());

    auto run1 = collision::gpu::BuildLbvhBroadphase(device_aabbs, kCount, 2u * 1024u * 1024u);
    auto run2 = collision::gpu::BuildLbvhBroadphase(device_aabbs, kCount, 2u * 1024u * 1024u);

    const auto p1 = run1.DownloadPairs();
    const auto p2 = run2.DownloadPairs();

    ASSERT_EQ(p1.size(), p2.size());
    // Byte-exact: identical compact sorted list element-for-element.
    bool identical = true;
    for (size_t i = 0; i < p1.size(); ++i) {
        if (p1[i].body_a != p2[i].body_a || p1[i].body_b != p2[i].body_b) {
            identical = false;
            break;
        }
    }
    EXPECT_TRUE(identical) << "LBVH compact pair list is not two-run byte-exact (D1)";
}
