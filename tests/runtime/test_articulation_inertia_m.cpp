// ---------------------------------------------------------------------------
// p01-F T4 -- joint-space inertia M (CRBA) + factorization + contact m_eff
//
// Validates the three GPU passes added to articulation_contacts.{cu,hpp}:
//
//  (1) ComputeArticulationInertiaM    -- dense symmetric M (dof x dof) via CRBA.
//  (2) FactorArticulationInertiaM     -- per-articulation LDL^T -> M^-1.
//  (3) ComputeContactEffectiveMass    -- m_eff = 1 / (J M^-1 J^T).
//
// Correctness strategy (advisor-guided): the M-vs-reference check uses the
// kinetic-energy identity  qdot^T M qdot = sum_i v_i^T I_i v_i, where v_i is ABA
// Pass-1's link_velocity (an INDEPENDENT formulation of the same quantity) and
// I_i = link_inertia, both in link-i frame. This avoids writing a second CRBA
// that could share the device path's convention bugs:
//   * random qdot                -> validates the whole quadratic form;
//   * unit probe qdot = e_i      -> M[i][i] = qdot^T M qdot (entry-level);
//   * pair qdot = e_i + e_j      -> off-diagonal M[i][j] (entry-level).
// Symmetry M == M^T and reconstruction M . M^-1 ~ I are checked directly.
//
// Determinism (check b): the M / factorization / m_eff kernels are one-block,
// single-lane, fixed-order, no-atomics -- so results are bit-identical across
// two runs and across all replicated envs (N=64). We assert both.
//
// Conditioning (check c): at a straightened-leg config M stays SPD and finite,
// and the vertical-foot m_eff stays finite/positive (the J M^-1 J^T -> 0 collapse
// is caught by the denominator clamp). Effective-mass sanity: a vertical foot
// contact yields a positive m_eff of leg-scale magnitude.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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

articulation::ArticulationHostState CookGo2HostState() {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    return articulation::BuildArticulationHostState(world.template_view.articulations,
                                                    world.template_view.body_table);
}

// Host kinetic energy reference: 0.5 * sum_i v_i^T I_i v_i, with v_i = device
// Pass-1 link_velocity and I_i = link_inertia (both per-link, link-frame). This
// shares no code with the CRBA M path, so matching 0.5 * qdot^T M qdot is a
// genuine cross-check of M.
double HostKineticEnergy(const std::vector<articulation::LinkSpatialVel>& vel,
                         const std::vector<articulation::LinkSpatialInertia>& inertia,
                         uint32_t link_begin,
                         uint32_t link_end) {
    double ke = 0.0;
    for (uint32_t link = link_begin; link < link_end; ++link) {
        const float* v = vel[link].v;
        const float* I = inertia[link].I;
        double iv[6];
        for (uint32_t r = 0u; r < 6u; ++r) {
            double s = 0.0;
            for (uint32_t c = 0u; c < 6u; ++c) {
                s += static_cast<double>(I[r * 6u + c]) * static_cast<double>(v[c]);
            }
            iv[r] = s;
        }
        for (uint32_t r = 0u; r < 6u; ++r) {
            ke += static_cast<double>(v[r]) * iv[r];
        }
    }
    return 0.5 * ke;
}

// Quadratic form 0.5 * qdot^T M qdot over the leading dof x dof block of one
// articulation's M tile (stride max_dof*max_dof, row-major).
double QuadraticForm(const std::vector<float>& M,
                     uint32_t max_dof,
                     uint32_t dof,
                     const std::vector<float>& qdot_local) {
    double sum = 0.0;
    for (uint32_t r = 0u; r < dof; ++r) {
        double row = 0.0;
        for (uint32_t c = 0u; c < dof; ++c) {
            row += static_cast<double>(M[static_cast<size_t>(r) * max_dof + c]) *
                   static_cast<double>(qdot_local[c]);
        }
        sum += static_cast<double>(qdot_local[r]) * row;
    }
    return 0.5 * sum;
}

// Maps articulation-local DOF index -> global link index (the non-fixed joint at
// that column). Lets a per-DOF probe qdot be written into the global qdot buffer.
std::vector<uint32_t> LocalDofToLink(const articulation::ArticulationHostState& host,
                                     uint32_t articulation) {
    const uint32_t offset = host.articulation_link_offset[articulation];
    const uint32_t count = host.articulation_link_count[articulation];
    std::vector<uint32_t> dof_to_link;
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (articulation::ArticulationJointDofCount(host.joint_type[link]) != 0u) {
            dof_to_link.push_back(link);
        }
    }
    return dof_to_link;
}

struct DeviceMResult {
    std::vector<float> M;
    std::vector<float> Minv;
    std::vector<articulation::LinkSpatialVel> link_velocity;
};

// Uploads `host`, runs ABA (so link_xup / motion subspaces / link_velocity are
// current for `host.qdot`), then M, then the LDL^T inverse; downloads all three.
DeviceMResult RunDeviceM(const nuka::phi::DeviceContext& context,
                         const articulation::ArticulationHostState& host,
                         uint32_t max_dof,
                         float gravity_z) {
    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);

    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t artic_count = host.ArticulationCount();
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;

    OwnedDeviceBuffer composite_buf(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia));
    OwnedDeviceBuffer m_buf(static_cast<size_t>(artic_count) * tile * sizeof(float));
    OwnedDeviceBuffer minv_buf(static_cast<size_t>(artic_count) * tile * sizeof(float));

    articulation::ComputeArticulationInertiaM(
        context, view, max_dof,
        static_cast<articulation::LinkSpatialInertia*>(composite_buf.Data()),
        static_cast<float*>(m_buf.Data()));
    articulation::FactorArticulationInertiaM(
        context, view, max_dof,
        static_cast<const float*>(m_buf.Data()),
        static_cast<float*>(minv_buf.Data()));
    context.stream.Synchronize();

    DeviceMResult result;
    result.M.resize(static_cast<size_t>(artic_count) * tile);
    result.Minv.resize(static_cast<size_t>(artic_count) * tile);
    result.link_velocity.resize(link_count);
    m_buf.CopyToHost(result.M.data(), result.M.size() * sizeof(float));
    minv_buf.CopyToHost(result.Minv.data(), result.Minv.size() * sizeof(float));

    // Download Pass-1 link velocities for the KE reference.
    articulation::ArticulationHostState download = host;
    articulation::DownloadArticulationState(device, &download);
    result.link_velocity = download.link_velocity;
    return result;
}

} // namespace

TEST(ArticulationInertiaM, CrbaMatchesKineticEnergyAndFactorizationIsDeterministic) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    auto base = CookGo2HostState();
    const uint32_t base_link_count = base.TotalLinkCount();
    ASSERT_GT(base_link_count, 0u);
    ASSERT_EQ(base.ArticulationCount(), 1u);

    const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(dof, 0u);
    const uint32_t max_dof = dof;  // single Go2: stride == dof == 12.
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;
    const auto dof_to_link = LocalDofToLink(base, 0u);
    ASSERT_EQ(dof_to_link.size(), dof);

    const float kGravity = 0.0f;  // gravity does not enter M; keep velocities clean.
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // ---------------------------------------------------------------------
    // (a) M correctness via the kinetic-energy identity, at a BENT pose.
    // ---------------------------------------------------------------------
    // Bend the first three non-fixed joints so M is a non-trivial dense block
    // (a straight rest pose would leave many cross terms near zero).
    auto bent = base;
    for (auto& q : bent.q) {
        q = 0.0f;
    }
    {
        const float kAngles[3] = {0.60f, 0.80f, -1.50f};
        for (uint32_t i = 0u; i < 3u && i < dof; ++i) {
            bent.q[dof_to_link[i]] = kAngles[i];
        }
    }

    // -- (a1) Random qdot validates the full quadratic form -------------------
    {
        auto host = bent;
        std::vector<float> qdot_local(dof, 0.0f);
        // Deterministic pseudo-random qdot in [-1, 1].
        for (uint32_t i = 0u; i < dof; ++i) {
            const float v = std::sin(1.3f * static_cast<float>(i) + 0.7f);
            qdot_local[i] = v;
            host.qdot[dof_to_link[i]] = v;
        }
        const auto res = RunDeviceM(context, host, max_dof, kGravity);

        const uint32_t offset = host.articulation_link_offset[0];
        const uint32_t count = host.articulation_link_count[0];
        const double ke_ref = HostKineticEnergy(res.link_velocity, host.link_inertia,
                                                offset, offset + count);
        const double ke_m = QuadraticForm(res.M, max_dof, dof, qdot_local);
        ASSERT_GT(ke_ref, 1.0e-4) << "reference KE too small to validate";
        const double rel_err = std::fabs(ke_m - ke_ref) / std::max(ke_ref, 1.0e-6);
        std::printf("[diag] (a1) KE: qdot^T M qdot=%.9f  sum v^T I v=%.9f  rel_err=%.3e\n",
                    ke_m, ke_ref, rel_err);
        EXPECT_LT(rel_err, 1.0e-4)
            << "qdot^T M qdot (" << ke_m << ") != sum v^T I v (" << ke_ref << ")";

        // Symmetry M == M^T (leading dof block).
        for (uint32_t r = 0u; r < dof; ++r) {
            for (uint32_t c = 0u; c < dof; ++c) {
                EXPECT_FLOAT_EQ(res.M[r * max_dof + c], res.M[c * max_dof + r])
                    << "M not symmetric at (" << r << "," << c << ")";
            }
        }
    }

    // -- (a2) Per-DOF unit probes -> diagonal; pairs -> off-diagonal ----------
    // For the whole-matrix check above we only have one M build; here we build M
    // once (pose-only, qdot ignored by M) and reconstruct entries from KE with
    // crafted qdot vectors using the SAME single M (M depends only on q).
    {
        // M depends only on the pose, so build it once with zero qdot.
        auto host = bent;
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        const auto res = RunDeviceM(context, host, max_dof, kGravity);
        const uint32_t offset = host.articulation_link_offset[0];
        const uint32_t count = host.articulation_link_count[0];

        // For a unit probe e_i, run a SEPARATE Pass-1 with that qdot to get the
        // KE reference (Pass-1's link_velocity depends on qdot, M does not).
        auto ke_for_qdot = [&](const std::vector<float>& qdot_local) {
            auto probe = host;
            for (uint32_t i = 0u; i < dof; ++i) {
                probe.qdot[dof_to_link[i]] = qdot_local[i];
            }
            const auto r = RunDeviceM(context, probe, max_dof, kGravity);
            return HostKineticEnergy(r.link_velocity, probe.link_inertia,
                                     offset, offset + count);
        };

        // Diagonal: M[i][i] = 2 * KE(e_i).
        for (uint32_t i = 0u; i < dof; ++i) {
            std::vector<float> e(dof, 0.0f);
            e[i] = 1.0f;
            const double ke = ke_for_qdot(e);
            const double m_ii = static_cast<double>(res.M[i * max_dof + i]);
            EXPECT_NEAR(m_ii, 2.0 * ke, 1.0e-4 * std::max(1.0, std::fabs(m_ii)))
                << "M[" << i << "][" << i << "] disagrees with KE probe";
            EXPECT_GT(m_ii, 0.0) << "diagonal of an SPD mass matrix must be positive";
        }

        // Off-diagonal of one leg's hip/thigh/calf sub-block:
        //   KE(e_i+e_j) = 0.5*(M_ii + M_jj) + M_ij  =>  M_ij = KE - 0.5*(M_ii+M_jj).
        const uint32_t kPairs[3][2] = {{0u, 1u}, {1u, 2u}, {0u, 2u}};
        for (const auto& pr : kPairs) {
            const uint32_t i = pr[0];
            const uint32_t j = pr[1];
            if (i >= dof || j >= dof) {
                continue;
            }
            std::vector<float> e(dof, 0.0f);
            e[i] = 1.0f;
            e[j] = 1.0f;
            const double ke = ke_for_qdot(e);
            const double m_ii = static_cast<double>(res.M[i * max_dof + i]);
            const double m_jj = static_cast<double>(res.M[j * max_dof + j]);
            const double m_ij_ref = ke - 0.5 * (m_ii + m_jj);
            const double m_ij = static_cast<double>(res.M[i * max_dof + j]);
            EXPECT_NEAR(m_ij, m_ij_ref,
                        1.0e-4 * std::max(1.0, std::fabs(m_ij_ref)))
                << "off-diagonal M[" << i << "][" << j << "] disagrees with KE probe";
        }

        // Reconstruction M . M^-1 ~ I over the leading dof block.
        for (uint32_t r = 0u; r < dof; ++r) {
            for (uint32_t c = 0u; c < dof; ++c) {
                double acc = 0.0;
                for (uint32_t k = 0u; k < dof; ++k) {
                    acc += static_cast<double>(res.M[r * max_dof + k]) *
                           static_cast<double>(res.Minv[k * max_dof + c]);
                }
                const double expected = (r == c) ? 1.0 : 0.0;
                EXPECT_NEAR(acc, expected, 1.0e-3)
                    << "M.Minv not identity at (" << r << "," << c << ")";
            }
        }
    }

    // ---------------------------------------------------------------------
    // (b) Determinism: bit-identical across two runs and across 64 replicas.
    // ---------------------------------------------------------------------
    {
        auto host = bent;
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        const auto run_a = RunDeviceM(context, host, max_dof, kGravity);
        const auto run_b = RunDeviceM(context, host, max_dof, kGravity);
        ASSERT_EQ(run_a.M.size(), run_b.M.size());
        for (size_t i = 0u; i < run_a.M.size(); ++i) {
            EXPECT_EQ(run_a.M[i], run_b.M[i]) << "M not bit-identical across runs at " << i;
        }
        for (size_t i = 0u; i < run_a.Minv.size(); ++i) {
            EXPECT_EQ(run_a.Minv[i], run_b.Minv[i])
                << "M^-1 not bit-identical across runs at " << i;
        }

        // Replicate to N=64 and assert every env's tile matches env 0's bit-for-bit.
        constexpr uint32_t kEnvCount = 64u;
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        ASSERT_EQ(batched.ArticulationCount(), kEnvCount);
        const auto batched_res = RunDeviceM(context, batched, max_dof, kGravity);
        for (uint32_t env = 1u; env < kEnvCount; ++env) {
            for (size_t i = 0u; i < tile; ++i) {
                ASSERT_EQ(batched_res.M[env * tile + i], batched_res.M[i])
                    << "replica " << env << " M tile diverged at " << i;
                ASSERT_EQ(batched_res.Minv[env * tile + i], batched_res.Minv[i])
                    << "replica " << env << " M^-1 tile diverged at " << i;
            }
        }
    }

    // ---------------------------------------------------------------------
    // (c) Conditioning at a straightened leg + effective-mass sanity.
    // ---------------------------------------------------------------------
    {
        // Straight leg: all q = 0 (Go2 rest pose has straight-ish legs). M must
        // stay SPD/finite, and a vertical foot contact must give finite +ve m_eff.
        auto host = base;
        for (auto& q : host.q) {
            q = 0.0f;
        }
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        const auto res = RunDeviceM(context, host, max_dof, kGravity);

        // No NaN/Inf anywhere in M or M^-1; diagonal strictly positive.
        for (uint32_t r = 0u; r < dof; ++r) {
            EXPECT_TRUE(std::isfinite(res.M[r * max_dof + r]));
            EXPECT_GT(res.M[r * max_dof + r], 0.0f);
            for (uint32_t c = 0u; c < dof; ++c) {
                EXPECT_TRUE(std::isfinite(res.M[r * max_dof + c]));
                EXPECT_TRUE(std::isfinite(res.Minv[r * max_dof + c]));
            }
        }

        // Reconstruction M . M^-1 ~ I at the near-singular (straight) pose. The
        // bent-pose block already checks this; repeating it here exercises the
        // factorization's conditioning guard at the degenerate config (spec c).
        double straight_recon_err = 0.0;
        for (uint32_t r = 0u; r < dof; ++r) {
            for (uint32_t c = 0u; c < dof; ++c) {
                double acc = 0.0;
                for (uint32_t k = 0u; k < dof; ++k) {
                    acc += static_cast<double>(res.M[r * max_dof + k]) *
                           static_cast<double>(res.Minv[k * max_dof + c]);
                }
                const double expected = (r == c) ? 1.0 : 0.0;
                straight_recon_err = std::max(straight_recon_err, std::fabs(acc - expected));
                EXPECT_NEAR(acc, expected, 1.0e-3)
                    << "straight-pose M.Minv not identity at (" << r << "," << c << ")";
            }
        }
        std::printf("[diag] (c) straight-pose max |M.Minv - I| = %.3e\n", straight_recon_err);

        // -- effective mass for a vertical foot contact --------------------
        // World poses for the (straight) pose, then a chain Jacobian for a
        // vertical contact at a foot, then m_eff = 1 / (J M^-1 J^T).
        auto device = articulation::UploadArticulationState(context, host);
        auto view = device.View();
        articulation::FeatherstoneAba::ComputeAccelerations(context, view, kGravity);

        OwnedDeviceBuffer pose_buf(
            static_cast<size_t>(base_link_count) * sizeof(Transform));
        articulation::UpdateWorldLinkPoses(
            context, view, static_cast<Transform*>(pose_buf.Data()));
        context.stream.Synchronize();
        std::vector<Transform> world(base_link_count);
        pose_buf.CopyToHost(world.data(), world.size() * sizeof(Transform));

        // link_pose drives the chain Jacobian's local->world axis rotation; it is
        // cook-time static, so overwrite it with the freshly-computed FK poses.
        for (uint32_t link = 0u; link < base_link_count; ++link) {
            host.link_pose[link] = world[link];
        }
        device = articulation::UploadArticulationState(context, host);
        view = device.View();
        articulation::FeatherstoneAba::ComputeAccelerations(context, view, kGravity);

        // Rebuild M / M^-1 against the FK-updated state (pose unchanged, so M is
        // the same; this keeps the M^-1 buffer consistent with `view`).
        OwnedDeviceBuffer composite_buf(
            static_cast<size_t>(base_link_count) *
                sizeof(articulation::LinkSpatialInertia));
        OwnedDeviceBuffer m_buf(tile * sizeof(float));
        OwnedDeviceBuffer minv_buf(tile * sizeof(float));
        articulation::ComputeArticulationInertiaM(
            context, view, max_dof,
            static_cast<articulation::LinkSpatialInertia*>(composite_buf.Data()),
            static_cast<float*>(m_buf.Data()));
        articulation::FactorArticulationInertiaM(
            context, view, max_dof,
            static_cast<const float*>(m_buf.Data()),
            static_cast<float*>(minv_buf.Data()));

        // Pick a revolute leaf (a calf/foot link) and place a vertical contact a
        // little below it so the chain carries a real lever arm.
        std::vector<uint8_t> is_parent(base_link_count, 0u);
        const uint32_t offset = host.articulation_link_offset[0];
        for (uint32_t link = 0u; link < base_link_count; ++link) {
            const uint32_t parent_local = host.parent_link[link];
            if (parent_local != kInvalidLink) {
                is_parent[offset + parent_local] = 1u;
            }
        }
        uint32_t foot = kInvalidLink;
        for (uint32_t link = 0u; link < base_link_count; ++link) {
            if (is_parent[link] == 0u &&
                host.joint_type[link] ==
                    articulation::ArticulationJointType::Revolute) {
                foot = link;
                break;
            }
        }
        ASSERT_NE(foot, kInvalidLink);

        const Vec3 contact_point = world[foot].TransformPoint({0.0f, 0.0f, -0.213f});
        const Vec3 contact_normal{0.0f, 0.0f, 1.0f};

        const uint32_t contact_count = 1u;
        OwnedDeviceBuffer link_idx_buf(sizeof(uint32_t));
        OwnedDeviceBuffer point_buf(sizeof(Vec3));
        OwnedDeviceBuffer normal_buf(sizeof(Vec3));
        OwnedDeviceBuffer jac_buf(static_cast<size_t>(contact_count) * max_dof *
                                      sizeof(float));
        OwnedDeviceBuffer meff_buf(contact_count * sizeof(float));
        link_idx_buf.CopyFromHost(&foot, sizeof(uint32_t));
        point_buf.CopyFromHost(&contact_point, sizeof(Vec3));
        normal_buf.CopyFromHost(&contact_normal, sizeof(Vec3));

        articulation::ComputeContactChainJacobians(
            context, view,
            static_cast<const uint32_t*>(link_idx_buf.Data()),
            static_cast<const Vec3*>(point_buf.Data()),
            static_cast<const Vec3*>(normal_buf.Data()),
            contact_count, max_dof,
            static_cast<float*>(jac_buf.Data()));
        articulation::ComputeContactEffectiveMass(
            context, view,
            static_cast<const uint32_t*>(link_idx_buf.Data()),
            static_cast<const float*>(jac_buf.Data()),
            static_cast<const float*>(minv_buf.Data()),
            contact_count, max_dof,
            static_cast<float*>(meff_buf.Data()));
        context.stream.Synchronize();

        float m_eff = 0.0f;
        meff_buf.CopyToHost(&m_eff, sizeof(float));
        std::vector<float> jac(max_dof);
        jac_buf.CopyToHost(jac.data(), jac.size() * sizeof(float));
        std::printf("[diag] (c) vertical-foot m_eff = %.6f kg (foot link %u)\n",
                    static_cast<double>(m_eff), foot);

        // m_eff is finite and strictly positive (conditioning guard works).
        EXPECT_TRUE(std::isfinite(m_eff)) << "m_eff not finite";
        EXPECT_GT(m_eff, 0.0f) << "m_eff must be positive";

        // Physically plausible magnitude: a Go2 leg is ~1-2 kg, the whole robot
        // ~15 kg. A foot's vertical effective mass should sit in a sane band, not
        // ~0 and not absurdly huge.
        EXPECT_GT(m_eff, 1.0e-3f) << "m_eff implausibly small";
        EXPECT_LT(m_eff, 1.0e3f) << "m_eff implausibly large";

        // The chain Jacobian must be non-trivial for this to be a real check.
        float jac_norm_sq = 0.0f;
        for (float v : jac) {
            jac_norm_sq += v * v;
        }
        EXPECT_GT(jac_norm_sq, 1.0e-6f) << "vertical-foot chain Jacobian collapsed";

        // -- denominator-clamp guard: a (near-)collapsed Jacobian row ---------
        // go2_stand never collapses J, so feed a near-zero J directly to prove
        // the J M^-1 J^T -> 0 path yields a large-but-finite m_eff, not an Inf
        // (the conditioning guard the spec explicitly requires, spec c).
        {
            std::vector<float> tiny_j(max_dof, 0.0f);
            tiny_j[0] = 1.0e-7f;  // J M^-1 J^T ~ Minv[0][0] * 1e-14 << denom eps.
            OwnedDeviceBuffer tiny_jac_buf(static_cast<size_t>(max_dof) * sizeof(float));
            tiny_jac_buf.CopyFromHost(tiny_j.data(), tiny_j.size() * sizeof(float));
            articulation::ComputeContactEffectiveMass(
                context, view,
                static_cast<const uint32_t*>(link_idx_buf.Data()),
                static_cast<const float*>(tiny_jac_buf.Data()),
                static_cast<const float*>(minv_buf.Data()),
                contact_count, max_dof,
                static_cast<float*>(meff_buf.Data()));
            context.stream.Synchronize();
            float clamped_m_eff = 0.0f;
            meff_buf.CopyToHost(&clamped_m_eff, sizeof(float));
            std::printf("[diag] (c) clamped (near-zero J) m_eff = %.6g kg\n",
                        static_cast<double>(clamped_m_eff));
            EXPECT_TRUE(std::isfinite(clamped_m_eff))
                << "denominator-clamp guard let m_eff go non-finite";
            EXPECT_GT(clamped_m_eff, 0.0f) << "clamped m_eff must stay positive";
            // 1 / eps with eps = kEffectiveMassDenomEpsilon (1e-9) ~ 1e9, finite.
            EXPECT_LE(clamped_m_eff,
                      1.0f / articulation::kEffectiveMassDenomEpsilon + 1.0f)
                << "clamped m_eff exceeds the 1/eps ceiling => guard not applied";
        }
    }
}
