// ---------------------------------------------------------------------------
// v0.8 SG G0 -- DOF honesty above 18 (the whole-body H1 precondition)
//
// The articulation contact-solve machinery historically carried an 18-DOF cap
// (kMaxFactorDof / kMaxContactSolverDof = 18 = Go2's 6-DOF floating base + 12
// revolute). FactorArticulationInertiaMKernel SILENTLY CLAMPED dof to 18, so a
// >18-DOF articulation got an M^-1 tile whose leading 18x18 block is the
// inverse of the leading 18x18 block of M (== every DOF >= 18 treated as
// WELDED) and whose remaining rows/cols are ZERO (== contact impulses NEVER
// move those joints). A ~51-DOF whole-body H1 with finger contacts (chain DOF
// indices > 18) is dishonest until this is closed.
//
// These tests are the G0 gate. They are RED on the truncating HEAD and GREEN
// after the fix (cap raised to kMaxArticulationDof = 64, silent clamp replaced
// by loud host-side failure):
//
//  (a) FactorInverseCoversAllDof24FixedRoot / (b) ...25FloatingRoot
//      A SYNTHETIC serial revolute chain (no asset gating) at 24 / 25 DOF.
//      The device M^-1 must match a HOST double-precision Gauss-Jordan inverse
//      of the DOWNLOADED M (so only the factorization's honesty is under test,
//      not the CRBA), and M . M^-1 ~ I over the FULL dof block. The crisp RED
//      observable: the SPD inverse diagonal at DOF >= 18 is ZERO when clamped.
//
//  (c) EffectiveMassUsesFullChain
//      m_eff = 1/(J M^-1 J^T) for a contact on the LAST link of the 24-DOF
//      chain (chain-J support up to DOF index 23) must match the host
//      reference built from the downloaded chain-J and the host M^-1. Under
//      the clamp the device treats columns >= 18 as immobile => m_eff is
//      WRONG (too large: missing chain mobility).
//
//  (d) ImplicitDampingReachesDofAbove18
//      The general implicit joint damping qdot -= dt*(M+dt*C)^-1*(C*qdot)
//      (the c_abi single-env path, ApplyImplicitJointDamping) must apply to a
//      joint whose DOF index > 18. RED today: the host launcher rejects
//      dof_stride > 18 outright, so a 24-DOF articulation cannot get implicit
//      damping AT ALL.
//
//  (e) LoudFailBeyondMaxArticulationDof
//      Truncation beyond the supported max (64) must be a LOUD failure (the
//      launcher throws), never a clamp. RED today: 65 DOF silently clamps.
//
// All chains are built programmatically through the validated
// BuildArticulationHostState (the BuildSmallFloatingModel pattern from
// test_floating_base_aba.cpp), so no scene/cook/asset is involved.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
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

// ---------------------------------------------------------------------------
// Synthetic serial chain: one root (Fixed or FloatingBase) + `revolute_count`
// revolute links in series, axes cycling X/Y/Z, 6 cm offsets. Per-joint
// armature 0.1 keeps M well-conditioned for the float LDL^T (the distal
// diagonal never collapses), so a tight host-vs-device tolerance is honest.
// ---------------------------------------------------------------------------
articulation::ArticulationHostState BuildSerialChain(uint32_t revolute_count,
                                                     bool floating_root) {
    const uint32_t link_count = revolute_count + 1u;
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    for (uint32_t i = 0u; i < link_count; ++i) {
        topo.link_bodies.push_back(i);
        topo.parent_links.push_back(i == 0u ? kInvalidLink : i - 1u);
        if (i == 0u) {
            topo.joint_types.push_back(
                floating_root ? articulation::ArticulationJointType::FloatingBase
                              : articulation::ArticulationJointType::Fixed);
            topo.joint_axes.push_back(Vec3::UnitZ());
            topo.parent_frames.push_back(Transform::Identity());
        } else {
            topo.joint_types.push_back(articulation::ArticulationJointType::Revolute);
            const uint32_t axis = (i - 1u) % 3u;
            topo.joint_axes.push_back(axis == 0u   ? Vec3::UnitX()
                                      : axis == 1u ? Vec3::UnitY()
                                                   : Vec3::UnitZ());
            topo.parent_frames.push_back(
                Transform{Vec3{0.06f, 0.0f, 0.0f}, Quat::Identity()});
        }
        topo.child_frames.push_back(Transform::Identity());
        topo.local_poses.push_back(Transform::Identity());
        topo.inertial_frames.push_back(Transform::Identity());
        topo.initial_positions.push_back(0.0f);
        topo.joint_dampings.push_back(0.0f);
        topo.joint_armatures.push_back(i == 0u ? 0.0f : 0.1f);
        topo.masses.push_back(i == 0u ? 2.0f : 1.0f);
        topo.inertias.push_back(i == 0u ? Vec3{0.02f, 0.022f, 0.018f}
                                        : Vec3{0.010f, 0.012f, 0.008f});
    }

    nuka::scene::CookedBodyTable bodies;
    for (uint32_t i = 0u; i < link_count; ++i) {
        bodies.poses.push_back(Transform::Identity());
        bodies.local_poses.push_back(Transform::Identity());
        bodies.inertial_frames.push_back(Transform::Identity());
        bodies.masses.push_back(topo.masses[i]);
        bodies.inertias.push_back(topo.inertias[i]);
        bodies.is_static.push_back(0u);
    }
    return articulation::BuildArticulationHostState({topo}, bodies);
}

// Zig-zag bend so M is a dense, non-degenerate block (a straight chain leaves
// near-zero cross terms and singular directions).
void BendChain(articulation::ArticulationHostState* host) {
    const uint32_t n = host->TotalLinkCount();
    for (uint32_t link = 0u; link < n; ++link) {
        if (host->joint_type[link] == articulation::ArticulationJointType::Revolute) {
            host->q[link] = 0.4f * std::sin(0.9f * static_cast<float>(link) + 0.3f);
        }
    }
}

// Host double-precision Gauss-Jordan inverse (partial pivoting) of the leading
// dof x dof block of a max_dof-strided tile. Independent of the device LDL^T.
std::vector<double> HostInvert(const std::vector<float>& tile,
                               uint32_t max_dof,
                               uint32_t dof) {
    std::vector<double> a(static_cast<size_t>(dof) * dof);
    std::vector<double> inv(static_cast<size_t>(dof) * dof, 0.0);
    for (uint32_t r = 0u; r < dof; ++r) {
        for (uint32_t c = 0u; c < dof; ++c) {
            a[r * dof + c] =
                static_cast<double>(tile[static_cast<size_t>(r) * max_dof + c]);
        }
        inv[r * dof + r] = 1.0;
    }
    for (uint32_t col = 0u; col < dof; ++col) {
        uint32_t pivot = col;
        for (uint32_t r = col + 1u; r < dof; ++r) {
            if (std::fabs(a[r * dof + col]) > std::fabs(a[pivot * dof + col])) {
                pivot = r;
            }
        }
        if (pivot != col) {
            for (uint32_t c = 0u; c < dof; ++c) {
                std::swap(a[col * dof + c], a[pivot * dof + c]);
                std::swap(inv[col * dof + c], inv[pivot * dof + c]);
            }
        }
        const double diag = a[col * dof + col];
        EXPECT_GT(std::fabs(diag), 1.0e-12) << "host pivot collapsed at " << col;
        const double scale = 1.0 / diag;
        for (uint32_t c = 0u; c < dof; ++c) {
            a[col * dof + c] *= scale;
            inv[col * dof + c] *= scale;
        }
        for (uint32_t r = 0u; r < dof; ++r) {
            if (r == col) {
                continue;
            }
            const double f = a[r * dof + col];
            if (f == 0.0) {
                continue;
            }
            for (uint32_t c = 0u; c < dof; ++c) {
                a[r * dof + c] -= f * a[col * dof + c];
                inv[r * dof + c] -= f * inv[col * dof + c];
            }
        }
    }
    return inv;
}

struct DeviceMResult {
    std::vector<float> M;
    std::vector<float> Minv;
};

// Upload -> ABA Pass-1 (link_xup / motion subspaces current) -> CRBA M ->
// LDL^T inverse -> download. Mirrors RunDeviceM in test_articulation_inertia_m.
// `with_damping` folds dt*C into the joint diagonals (the implicit-damping M).
DeviceMResult RunDeviceM(cudaStream_t context, int context_dev,
                         const articulation::ArticulationHostState& host,
                         uint32_t max_dof,
                         bool with_damping = false,
                         float dt = 0.0f) {
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    auto view = device.View();
    articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, view, 0.0f);

    const uint32_t link_count = host.TotalLinkCount();
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;
    OwnedDeviceBuffer composite_buf(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia));
    OwnedDeviceBuffer m_buf(tile * sizeof(float));
    OwnedDeviceBuffer minv_buf(tile * sizeof(float));

    articulation::ComputeArticulationInertiaM(
        context, context_dev, view, max_dof,
        static_cast<articulation::LinkSpatialInertia*>(composite_buf.Data()),
        static_cast<float*>(m_buf.Data()),
        with_damping ? view.joint_damping : nullptr, dt);
    articulation::FactorArticulationInertiaM(
        context, context_dev, view, max_dof,
        static_cast<const float*>(m_buf.Data()),
        static_cast<float*>(minv_buf.Data()));
    cudaStreamSynchronize(context);

    DeviceMResult result;
    result.M.resize(tile);
    result.Minv.resize(tile);
    m_buf.CopyToHost(result.M.data(), result.M.size() * sizeof(float));
    minv_buf.CopyToHost(result.Minv.data(), result.Minv.size() * sizeof(float));
    return result;
}

// Shared (a)/(b) body: device M^-1 must cover the FULL dof block.
void CheckFactorHonesty(const articulation::ArticulationHostState& host,
                        uint32_t expected_dof) {
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(dof, expected_dof);
    const uint32_t max_dof = dof;

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    const auto res = RunDeviceM(context, context_dev, host, max_dof);

    // The CRBA must have filled the FULL dof block (sanity that the input M is
    // not itself truncated): every diagonal is strictly positive.
    for (uint32_t r = 0u; r < dof; ++r) {
        ASSERT_GT(res.M[static_cast<size_t>(r) * max_dof + r], 0.0f)
            << "CRBA M diagonal empty at DOF " << r << " (M itself truncated?)";
    }

    const auto minv_ref = HostInvert(res.M, max_dof, dof);

    // Crisp truncation observable FIRST: an SPD inverse has a strictly positive
    // diagonal at EVERY dof. The silent 18-clamp leaves rows >= 18 all-zero.
    for (uint32_t r = 0u; r < dof; ++r) {
        EXPECT_GT(res.Minv[static_cast<size_t>(r) * max_dof + r], 0.0f)
            << "M^-1 diagonal ZERO at DOF " << r
            << " => DOF truncated (silently welded) by the factorization";
    }

    // Elementwise vs the host double inverse, tolerance scaled to the inverse's
    // magnitude (float LDL^T on a well-conditioned M).
    double ref_max = 0.0;
    for (uint32_t r = 0u; r < dof; ++r) {
        for (uint32_t c = 0u; c < dof; ++c) {
            ref_max = std::max(ref_max, std::fabs(minv_ref[r * dof + c]));
        }
    }
    const double tol = 1.0e-4 * std::max(1.0, ref_max);
    double max_abs_err = 0.0;
    for (uint32_t r = 0u; r < dof; ++r) {
        for (uint32_t c = 0u; c < dof; ++c) {
            const double dev =
                static_cast<double>(res.Minv[static_cast<size_t>(r) * max_dof + c]);
            const double err = std::fabs(dev - minv_ref[r * dof + c]);
            max_abs_err = std::max(max_abs_err, err);
            EXPECT_NEAR(dev, minv_ref[r * dof + c], tol)
                << "M^-1 disagrees with the host reference at (" << r << "," << c
                << ")";
        }
    }

    // Reconstruction M . M^-1 ~ I over the FULL dof block (double accumulate).
    double recon_err = 0.0;
    for (uint32_t r = 0u; r < dof; ++r) {
        for (uint32_t c = 0u; c < dof; ++c) {
            double acc = 0.0;
            for (uint32_t k = 0u; k < dof; ++k) {
                acc += static_cast<double>(res.M[static_cast<size_t>(r) * max_dof + k]) *
                       static_cast<double>(res.Minv[static_cast<size_t>(k) * max_dof + c]);
            }
            const double expected = (r == c) ? 1.0 : 0.0;
            recon_err = std::max(recon_err, std::fabs(acc - expected));
            EXPECT_NEAR(acc, expected, 1.0e-3)
                << "M.M^-1 not identity at (" << r << "," << c << ")";
        }
    }
    std::printf("[diag] dof=%u: max|Minv_dev-Minv_ref|=%.3e (tol %.3e), "
                "max|M.Minv-I|=%.3e\n",
                dof, max_abs_err, tol, recon_err);
}

} // namespace

// (a) 24 scalar DOFs past a fixed root: every chain DOF index 0..23, six of
// them ABOVE the historical 18 cap.
TEST(DofAbove18Honesty, FactorInverseCoversAllDof24FixedRoot) {
    auto host = BuildSerialChain(24u, /*floating_root=*/false);
    BendChain(&host);
    CheckFactorHonesty(host, 24u);
}

// (b) FloatingBase root (6 DOF) + 19 revolute = 25 DOF: the whole-body shape
// (free base + long joint tail), base coupling rows included.
TEST(DofAbove18Honesty, FactorInverseCoversAllDof25FloatingRoot) {
    auto host = BuildSerialChain(19u, /*floating_root=*/true);
    BendChain(&host);
    CheckFactorHonesty(host, 25u);
}

// (c) Contact on the LAST link of the 24-DOF chain: m_eff = 1/(J M^-1 J^T)
// must use the FULL chain (J has support up to DOF index 23).
TEST(DofAbove18Honesty, EffectiveMassUsesFullChain) {
    auto host = BuildSerialChain(24u, /*floating_root=*/false);
    BendChain(&host);
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(dof, 24u);
    const uint32_t max_dof = dof;
    const uint32_t link_count = host.TotalLinkCount();
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;

    // FK world poses, then write them into link_pose (the chain-J kernel reads
    // link_pose for axis_world/lever -- the stage-4 refresh convention).
    {
        auto device = articulation::UploadArticulationState(context, context_dev, host);
        auto view = device.View();
        articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, view, 0.0f);
        OwnedDeviceBuffer pose_buf(
            static_cast<size_t>(link_count) * sizeof(Transform));
        articulation::UpdateWorldLinkPoses(context, context_dev, view,
                                           static_cast<Transform*>(pose_buf.Data()));
        cudaStreamSynchronize(context);
        std::vector<Transform> world(link_count);
        pose_buf.CopyToHost(world.data(), world.size() * sizeof(Transform));
        host.link_pose = world;
    }

    auto device = articulation::UploadArticulationState(context, context_dev, host);
    auto view = device.View();
    articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, view, 0.0f);

    OwnedDeviceBuffer composite_buf(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia));
    OwnedDeviceBuffer m_buf(tile * sizeof(float));
    OwnedDeviceBuffer minv_buf(tile * sizeof(float));
    articulation::ComputeArticulationInertiaM(
        context, context_dev, view, max_dof,
        static_cast<articulation::LinkSpatialInertia*>(composite_buf.Data()),
        static_cast<float*>(m_buf.Data()));
    articulation::FactorArticulationInertiaM(
        context, context_dev, view, max_dof, static_cast<const float*>(m_buf.Data()),
        static_cast<float*>(minv_buf.Data()));

    // Contact on the LAST link, slightly off-origin, vertical normal.
    const uint32_t last = link_count - 1u;
    const Vec3 contact_point =
        host.link_pose[last].TransformPoint({0.05f, 0.0f, -0.02f});
    const Vec3 contact_normal{0.0f, 0.0f, 1.0f};
    OwnedDeviceBuffer link_idx_buf(sizeof(uint32_t));
    OwnedDeviceBuffer point_buf(sizeof(Vec3));
    OwnedDeviceBuffer normal_buf(sizeof(Vec3));
    OwnedDeviceBuffer jac_buf(static_cast<size_t>(max_dof) * sizeof(float));
    OwnedDeviceBuffer meff_buf(sizeof(float));
    link_idx_buf.CopyFromHost(&last, sizeof(uint32_t));
    point_buf.CopyFromHost(&contact_point, sizeof(Vec3));
    normal_buf.CopyFromHost(&contact_normal, sizeof(Vec3));

    articulation::ComputeContactChainJacobians(
        context, context_dev, view, static_cast<const uint32_t*>(link_idx_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()), 1u, max_dof,
        static_cast<float*>(jac_buf.Data()));
    articulation::ComputeContactEffectiveMass(
        context, context_dev, view, static_cast<const uint32_t*>(link_idx_buf.Data()),
        static_cast<const float*>(jac_buf.Data()),
        static_cast<const float*>(minv_buf.Data()), 1u, max_dof,
        static_cast<float*>(meff_buf.Data()));
    cudaStreamSynchronize(context);

    std::vector<float> M(tile);
    std::vector<float> jac(max_dof);
    float m_eff_dev = 0.0f;
    m_buf.CopyToHost(M.data(), M.size() * sizeof(float));
    jac_buf.CopyToHost(jac.data(), jac.size() * sizeof(float));
    meff_buf.CopyToHost(&m_eff_dev, sizeof(float));

    // The chain-J must genuinely reach above DOF 18, else this gate is vacuous.
    float high_support = 0.0f;
    for (uint32_t k = 18u; k < dof; ++k) {
        high_support += jac[k] * jac[k];
    }
    ASSERT_GT(high_support, 1.0e-8f)
        << "chain-J has no support above DOF 18; the gate would be vacuous";

    // Host reference: m_eff = 1 / (J Minv_ref J^T) with the double inverse.
    const auto minv_ref = HostInvert(M, max_dof, dof);
    double denom = 0.0;
    for (uint32_t r = 0u; r < dof; ++r) {
        double row = 0.0;
        for (uint32_t c = 0u; c < dof; ++c) {
            row += minv_ref[r * dof + c] * static_cast<double>(jac[c]);
        }
        denom += static_cast<double>(jac[r]) * row;
    }
    ASSERT_GT(denom, 1.0e-9);
    const double m_eff_ref = 1.0 / denom;
    const double rel_err =
        std::fabs(static_cast<double>(m_eff_dev) - m_eff_ref) / m_eff_ref;
    std::printf("[diag] m_eff dev=%.6f ref=%.6f rel_err=%.3e\n",
                static_cast<double>(m_eff_dev), m_eff_ref, rel_err);
    EXPECT_LT(rel_err, 1.0e-3)
        << "m_eff disagrees with the full-chain host reference (dev="
        << m_eff_dev << ", ref=" << m_eff_ref
        << ") => M^-1 coupling above DOF 18 lost";
}

// (d) Implicit joint damping must reach a joint whose DOF index > 18. RED on
// HEAD: ApplyImplicitJointDamping refuses dof_stride > 18 outright (throws), so
// big articulations cannot get implicit damping at all.
TEST(DofAbove18Honesty, ImplicitDampingReachesDofAbove18) {
    auto host = BuildSerialChain(24u, /*floating_root=*/false);
    BendChain(&host);
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(dof, 24u);
    const float kDamping = 2.0f;
    const float kDt = 1.0f / 240.0f;
    const uint32_t link_count = host.TotalLinkCount();

    // Per-link damping + a non-trivial qdot on every joint.
    std::vector<uint32_t> dof_to_link;
    for (uint32_t link = 0u; link < link_count; ++link) {
        if (host.joint_type[link] == articulation::ArticulationJointType::Revolute) {
            host.joint_damping[link] = kDamping;
            host.qdot[link] = 0.3f * std::cos(0.7f * static_cast<float>(link));
            dof_to_link.push_back(link);
        }
    }
    ASSERT_EQ(dof_to_link.size(), dof);

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;

    // (M + dt*C) and its host inverse (for the reference update).
    const auto res = RunDeviceM(context, context_dev, host, dof, /*with_damping=*/true, kDt);
    const auto minv_ref = HostInvert(res.M, dof, dof);

    // Device: factor + ApplyImplicitJointDamping in one upload session.
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    auto view = device.View();
    articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, view, 0.0f);
    const size_t tile = static_cast<size_t>(dof) * dof;
    OwnedDeviceBuffer composite_buf(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia));
    OwnedDeviceBuffer m_buf(tile * sizeof(float));
    OwnedDeviceBuffer minv_buf(tile * sizeof(float));
    articulation::ComputeArticulationInertiaM(
        context, context_dev, view, dof,
        static_cast<articulation::LinkSpatialInertia*>(composite_buf.Data()),
        static_cast<float*>(m_buf.Data()), view.joint_damping, kDt);
    articulation::FactorArticulationInertiaM(
        context, context_dev, view, dof, static_cast<const float*>(m_buf.Data()),
        static_cast<float*>(minv_buf.Data()));
    articulation::ApplyImplicitJointDamping(
        context, context_dev, view, static_cast<const float*>(minv_buf.Data()),
        view.joint_damping, dof, kDt);
    cudaStreamSynchronize(context);

    articulation::ArticulationHostState download = host;
    articulation::DownloadArticulationState(device, &download);

    // Host reference: qdot_new = qdot - dt * (M+dt*C)^-1 * (C * qdot).
    std::vector<double> c_qdot(dof);
    for (uint32_t k = 0u; k < dof; ++k) {
        c_qdot[k] = static_cast<double>(kDamping) *
                    static_cast<double>(host.qdot[dof_to_link[k]]);
    }
    double max_err = 0.0;
    for (uint32_t k = 0u; k < dof; ++k) {
        double acc = 0.0;
        for (uint32_t c = 0u; c < dof; ++c) {
            acc += minv_ref[k * dof + c] * c_qdot[c];
        }
        const double expected =
            static_cast<double>(host.qdot[dof_to_link[k]]) -
            static_cast<double>(kDt) * acc;
        const double got = static_cast<double>(download.qdot[dof_to_link[k]]);
        max_err = std::max(max_err, std::fabs(got - expected));
        EXPECT_NEAR(got, expected, 1.0e-5)
            << "implicit damping wrong at DOF " << k << " (link "
            << dof_to_link[k] << ")";
        // The honesty bite: damping must actually CHANGE qdot at every DOF
        // (C is non-zero on all 24 joints).
        EXPECT_NE(got, static_cast<double>(host.qdot[dof_to_link[k]]))
            << "implicit damping left DOF " << k << " untouched";
    }
    std::printf("[diag] implicit damping max|qdot_dev-qdot_ref|=%.3e\n", max_err);
}

// (e) Beyond the supported max the engine must FAIL LOUDLY (host-side throw),
// never silently clamp. 65 DOF > kMaxArticulationDof (64).
TEST(DofAbove18Honesty, LoudFailBeyondMaxArticulationDof) {
    auto host = BuildSerialChain(65u, /*floating_root=*/false);
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(dof, 65u);

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    auto view = device.View();

    const size_t tile = static_cast<size_t>(dof) * dof;
    OwnedDeviceBuffer m_buf(tile * sizeof(float));
    OwnedDeviceBuffer minv_buf(tile * sizeof(float));
    std::vector<float> zeros(tile, 0.0f);
    m_buf.CopyFromHost(zeros.data(), zeros.size() * sizeof(float));

    EXPECT_THROW(
        articulation::FactorArticulationInertiaM(
            context, context_dev, view, dof, static_cast<const float*>(m_buf.Data()),
            static_cast<float*>(minv_buf.Data())),
        std::runtime_error)
        << "FactorArticulationInertiaM accepted " << dof
        << " DOFs without a loud failure (silent clamp?)";

    EXPECT_THROW(
        articulation::ApplyImplicitJointDamping(
            context, context_dev, view, static_cast<const float*>(minv_buf.Data()),
            view.joint_damping, dof, 1.0f / 240.0f),
        std::runtime_error)
        << "ApplyImplicitJointDamping accepted " << dof << " DOFs";
}
