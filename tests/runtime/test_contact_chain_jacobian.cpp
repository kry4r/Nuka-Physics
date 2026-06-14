// ---------------------------------------------------------------------------
// p01-F T3 -- full-chain contact Jacobian
//
// Validates ComputeContactChainJacobians: for a contact on a Unitree Go2 foot
// it must produce the joint-space (generalized) Jacobian over the WHOLE chain
// of ancestor joints (calf -> thigh -> hip -> root), so a contact force can be
// projected into reduced (joint) coordinates as tau_j = J_j . F.
//
// Design notes (the correctness crux is the joint-axis frame):
//  * joint_axis is stored in the joint/child-body LOCAL frame (raw USD
//    physics:axis token, never transformed -- see articulation_cooker.cpp:174,
//    world.cpp:347). The kernel must rotate it to world by link_pose.rotation.
//  * link_pose is cook-time static and is NOT updated by ABA forward kinematics
//    (AbaPass1 writes link_xup, never link_pose). So this test sets link_pose
//    explicitly to a hand-composed BENT leg configuration with a nonzero hip
//    roll -- the roll is what tilts the thigh/calf Y-axes in world and is the
//    only thing that exercises the local->world rotation.
//  * The foot collision sphere sits at the calf link origin (no local offset),
//    which coincides with the calf joint origin, so a contact placed exactly at
//    the foot gives r_calf = 0 and a structurally-zero calf torque. We offset
//    the contact below the calf so all three joints carry nonzero torque.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/world_builder.hpp"
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

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Same cook path the C ABI uses (LoadUsd -> CookScene -> BuildWorld ->
// BuildArticulationHostState).
articulation::ArticulationHostState CookGo2HostState() {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    return articulation::BuildArticulationHostState(world.template_view.articulations,
                                                    world.template_view.body_table);
}

// DOF count per joint -- mirrors the kernel's convention (Revolute/Prismatic=1,
// Fixed=0). The test's expected dof_index uses the same prefix sum.
uint32_t JointDofCount(articulation::ArticulationJointType type) {
    return type == articulation::ArticulationJointType::Fixed ? 0u : 1u;
}

} // namespace

TEST(ContactChainJacobian, Go2LegMatchesAnalyticThreeLinkChain) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    auto host = CookGo2HostState();
    const uint32_t link_count = host.TotalLinkCount();
    ASSERT_GT(link_count, 0u);
    // Single Go2 articulation in this scene.
    ASSERT_EQ(host.ArticulationCount(), 1u);
    const uint32_t offset = host.articulation_link_offset[0];

    // --- Locate a leg: a revolute LEAF (calf) and its revolute ancestors. -----
    // A leaf is a link that no other link names as parent. Walk up parent_link
    // (articulation-local) for thigh and hip; the hip's parent is the fixed root.
    std::vector<uint8_t> is_parent(link_count, 0u);
    for (uint32_t link = 0u; link < link_count; ++link) {
        const uint32_t parent_local = host.parent_link[link];
        if (parent_local != kInvalidLink) {
            is_parent[offset + parent_local] = 1u;
        }
    }

    uint32_t calf = kInvalidLink;
    for (uint32_t link = 0u; link < link_count; ++link) {
        if (is_parent[link] == 0u &&
            host.joint_type[link] == articulation::ArticulationJointType::Revolute) {
            calf = link;
            break;
        }
    }
    ASSERT_NE(calf, kInvalidLink) << "no revolute leaf (calf) found";

    const uint32_t thigh = offset + host.parent_link[calf];
    ASSERT_EQ(host.joint_type[thigh], articulation::ArticulationJointType::Revolute);
    const uint32_t hip = offset + host.parent_link[thigh];
    ASSERT_EQ(host.joint_type[hip], articulation::ArticulationJointType::Revolute);
    const uint32_t root = offset + host.parent_link[hip];
    ASSERT_EQ(host.joint_type[root], articulation::ArticulationJointType::Fixed);
    ASSERT_EQ(host.parent_link[root], kInvalidLink) << "root must have no parent";

    // The stored local joint axes (in each link's own frame).
    const Vec3 axis_hip_local = host.joint_axis[hip];
    const Vec3 axis_thigh_local = host.joint_axis[thigh];
    const Vec3 axis_calf_local = host.joint_axis[calf];

    // --- Compose a BENT leg pose by forward kinematics on the host. -----------
    // Nonzero hip ROLL tilts the thigh/calf Y-axes out of the world Y plane,
    // and the thigh/calf BEND introduces the x-lever arm. Both are required for
    // all three joint torques to be nonzero and for the axis rotation to matter.
    const float q_hip = 0.60f;     // roll about local X (large enough that the
                                   // resulting world Y-axis tilt makes the
                                   // world-vs-local discriminator unambiguous)
    const float q_thigh = 0.80f;   // pitch about local Y
    const float q_calf = -1.50f;   // pitch about local Y

    // Parent-relative link translations from the Go2 USD (identity rotations).
    const Transform base_world{{0.0f, 0.0f, 0.445f}, Quat::Identity()};
    const Transform hip_offset{{0.1934f, 0.0465f, 0.0f}, Quat::Identity()};
    const Transform thigh_offset{{0.0f, 0.0955f, 0.0f}, Quat::Identity()};
    const Transform calf_offset{{0.0f, 0.0f, -0.213f}, Quat::Identity()};

    const Transform root_world = base_world;
    const Transform hip_world =
        root_world * hip_offset *
        Transform{Vec3::Zero(), Quat::FromAxisAngle(axis_hip_local, q_hip)};
    const Transform thigh_world =
        hip_world * thigh_offset *
        Transform{Vec3::Zero(), Quat::FromAxisAngle(axis_thigh_local, q_thigh)};
    const Transform calf_world =
        thigh_world * calf_offset *
        Transform{Vec3::Zero(), Quat::FromAxisAngle(axis_calf_local, q_calf)};

    // Write the bent FK poses into the host state before upload. link_pose is
    // cook-time static, so this is the configuration the kernel will read.
    host.link_pose[root] = root_world;
    host.link_pose[hip] = hip_world;
    host.link_pose[thigh] = thigh_world;
    host.link_pose[calf] = calf_world;

    // Contact point: below the calf joint origin so r_calf != 0 (the foot sphere
    // itself sits at the calf origin, which would give a structurally-zero calf
    // torque). 0.213 m down the calf, in the calf's local frame.
    const Vec3 contact_point = calf_world.TransformPoint({0.0f, 0.0f, -0.213f});
    const Vec3 contact_normal{0.0f, 0.0f, 1.0f};  // unit vertical force direction

    // --- Analytic reference (independent flat loop, no parent walk). ----------
    // World axes via the SAME rotation the kernel must apply.
    const Vec3 axis_hip_world = hip_world.rotation.Rotate(axis_hip_local);
    const Vec3 axis_thigh_world = thigh_world.rotation.Rotate(axis_thigh_local);
    const Vec3 axis_calf_world = calf_world.rotation.Rotate(axis_calf_local);

    auto revolute_tau = [&](const Vec3& axis_world, const Vec3& origin) {
        const Vec3 lever = contact_point - origin;
        return axis_world.Cross(lever).Dot(contact_normal);
    };
    const float tau_hip = revolute_tau(axis_hip_world, hip_world.position);
    const float tau_thigh = revolute_tau(axis_thigh_world, thigh_world.position);
    const float tau_calf = revolute_tau(axis_calf_world, calf_world.position);

    // Independent closed-form geometric reference (shares no code with the
    // kernel's cross-product, so it catches a shared lever-sign / axis-order
    // convention bug). With n = z_hat, (a x r).z_hat = a_x*r_y - a_y*r_x.
    //  * hip world axis = X_hat = (1,0,0)  ->  tau = r_y = p.y - origin.y
    //    (independent of any rotation; bulletproof lever-sign check).
    //  * thigh/calf local axis = Y_hat; the leg's only out-of-Y rotation is the
    //    hip roll about X (the thigh/calf pitches are about Y and leave Y_hat
    //    invariant), so both world axes are (0, cos q_hip, sin q_hip)  ->
    //    tau = -cos(q_hip) * r_x = -cos(q_hip) * (p.x - origin.x).
    const float kClosedFormTol = 1.0e-4f;
    const float cos_hip = std::cos(q_hip);
    EXPECT_NEAR(tau_hip, contact_point.y - hip_world.position.y, kClosedFormTol)
        << "hip closed-form reference disagrees with cross-product reference";
    EXPECT_NEAR(tau_thigh, -cos_hip * (contact_point.x - thigh_world.position.x),
                kClosedFormTol)
        << "thigh closed-form reference disagrees with cross-product reference";
    EXPECT_NEAR(tau_calf, -cos_hip * (contact_point.x - calf_world.position.x),
                kClosedFormTol)
        << "calf closed-form reference disagrees with cross-product reference";

    // All three torques must be meaningfully nonzero, else the chain is not
    // actually being exercised.
    ASSERT_GT(std::fabs(tau_hip), 1.0e-3f) << "hip torque too small to validate";
    ASSERT_GT(std::fabs(tau_thigh), 1.0e-3f) << "thigh torque too small to validate";
    ASSERT_GT(std::fabs(tau_calf), 1.0e-3f) << "calf torque too small to validate";

    // Discriminator: the value the kernel WOULD produce with the un-rotated
    // LOCAL axis. With hip roll != 0 these differ measurably, so asserting the
    // kernel matches the world-axis value (and differs from the local-axis one)
    // proves the local->world rotation is actually applied.
    const float tau_thigh_local_axis =
        revolute_tau(axis_thigh_local, thigh_world.position);
    const float tau_calf_local_axis =
        revolute_tau(axis_calf_local, calf_world.position);
    ASSERT_GT(std::fabs(tau_thigh - tau_thigh_local_axis), 1.0e-3f)
        << "world vs local axis indistinguishable; test cannot prove rotation";
    ASSERT_GT(std::fabs(tau_calf - tau_calf_local_axis), 1.0e-3f)
        << "world vs local axis indistinguishable; test cannot prove rotation";

    // Expected dof_index per joint: base-inclusive prefix sum across the
    // articulation's links in [offset, j).
    auto dof_index_of = [&](uint32_t target) {
        uint32_t index = 0u;
        for (uint32_t k = offset; k < target; ++k) {
            index += JointDofCount(host.joint_type[k]);
        }
        return index;
    };
    const uint32_t dof_hip = dof_index_of(hip);
    const uint32_t dof_thigh = dof_index_of(thigh);
    const uint32_t dof_calf = dof_index_of(calf);

    // dof_stride = total articulation DOF (base-inclusive). For Go2 = 12.
    uint32_t dof_stride = 0u;
    for (uint32_t k = offset; k < offset + host.articulation_link_count[0]; ++k) {
        dof_stride += JointDofCount(host.joint_type[k]);
    }
    ASSERT_GT(dof_stride, 0u);

    // --- Upload state, set up contact buffers, run the kernel. ----------------
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto device = articulation::UploadArticulationState(context, host);

    const uint32_t contact_count = 1u;
    const uint32_t contact_link = calf;  // global link index of the contact link

    OwnedDeviceBuffer link_idx_buf(sizeof(uint32_t));
    OwnedDeviceBuffer point_buf(sizeof(Vec3));
    OwnedDeviceBuffer normal_buf(sizeof(Vec3));
    OwnedDeviceBuffer out_buf(static_cast<size_t>(contact_count) * dof_stride * sizeof(float));

    link_idx_buf.CopyFromHost(&contact_link, sizeof(uint32_t));
    point_buf.CopyFromHost(&contact_point, sizeof(Vec3));
    normal_buf.CopyFromHost(&contact_normal, sizeof(Vec3));

    articulation::ComputeContactChainJacobians(
        context,
        device.View(),
        static_cast<const uint32_t*>(link_idx_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()),
        contact_count,
        dof_stride,
        static_cast<float*>(out_buf.Data()));
    context.stream.Synchronize();

    std::vector<float> out(dof_stride, 0.0f);
    out_buf.CopyToHost(out.data(), out.size() * sizeof(float));

    // --- Assert: chain columns match analytic tau, all other columns zero. ----
    const float kTol = 1.0e-5f;
    EXPECT_NEAR(out[dof_hip], tau_hip, kTol) << "hip generalized force mismatch";
    EXPECT_NEAR(out[dof_thigh], tau_thigh, kTol) << "thigh generalized force mismatch";
    EXPECT_NEAR(out[dof_calf], tau_calf, kTol) << "calf generalized force mismatch";

    for (uint32_t col = 0u; col < dof_stride; ++col) {
        if (col == dof_hip || col == dof_thigh || col == dof_calf) {
            continue;
        }
        EXPECT_NEAR(out[col], 0.0f, kTol)
            << "non-chain column " << col << " should be untouched (zero)";
    }
}
