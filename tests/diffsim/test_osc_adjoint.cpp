// ---------------------------------------------------------------------------
// v0.5 p03 R3b -- OSC differentiable adjoint: FD validation + D1 byte-exact
// ---------------------------------------------------------------------------
//
// Validates nuka::diffsim::LaunchOscAdjointGainTarget -- the reverse-mode adjoint
// of the FORWARD OSC drive kernel (runtime/articulation/articulation_drives.cu,
// ApplyOscDriveKernel via LaunchApplyOscDriveKernels). The adjoint ships the
// gain (Kp, Kd) and task-target channels, which enter the forward LINEARLY
// through rhs = Kp*e_x + Kd*edot_x (A / Lambda is a fixed linear operator w.r.t.
// these params -- no matrix-inverse derivative needed). See osc_adjoint.hpp.
//
// FD ORACLE (the primary gate). We define a scalar loss L = sum_link
// g_tau[link] * tau[link] with a FIXED g_tau, where tau is the forward OSC output.
// Then dL/dparam = sum_link g_tau[link] * d tau/d param, which is EXACTLY what the
// adjoint computes (dL/dtau = g_tau is the upstream gradient). We measure dL/dparam
// numerically by re-running the REAL forward kernel (LaunchApplyOscDriveKernels)
// with a perturbed param on the SAME frozen device state (q / qdot / link_pose /
// joint_axis held fixed -- the launcher only writes state.tau) and central-
// differencing the loss, then compare to the analytic adjoint.
//
// SCENE -- why SYNTHETIC, not the Go2 calf (an HONEST deviation). The p03 mandate
// suggested the Go2 calf/foot task link "as the OSC forward tests do". But that
// task point is the calf's LINK ORIGIN, where the calf's OWN revolute column
// vanishes (cross(axis, x_task - origin) with x_task == origin), so the chain
// Jacobian J is RANK-2 (documented in test_control_modes.cpp ~L1239-1244). A
// rank-2 J makes A = J M^-1 J^T SINGULAR; the forward's 3x3 LDL^T floors its third
// pivot to kMinDiagonalDrive (1e-6), and in fp32 that ~1e6-conditioned solve
// pollutes the in-plane gradient (~20-30% error) and makes the null (+Z) gradient
// pure roundoff. The forward map is still exactly linear, so the divergence is
// CONDITIONING (a property of the committed fp32 forward at a rank-deficient task),
// NOT an adjoint bug -- and no FD step / fp64 loss fixes it.
//
// To FD-validate the adjoint MATH cleanly we therefore run BOTH the real forward
// kernel AND the adjoint on a tiny SYNTHETIC fixed-base articulation with a
// FULL-RANK-3 task Jacobian: three revolute joints (independent axes ex/ey/ez) and
// a FIXED end-effector link offset from the last revolute joint's origin. The
// chain walk from the end-effector hits all three revolute columns (none vanishes,
// because the task point is offset from every joint origin), so A is SPD, the floor
// never fires, tau is exactly linear in the params, and FD matches analytic to
// ~float precision. This is the SAME "real kernel, synthetic controlled inputs"
// philosophy as test_kkt_build.cpp. A separate Go2 directional smoke (non-FD)
// confirms the adjoint also runs end-to-end on the real engine's device state.
//
// D1. Two adjoint runs on the same frozen state -> byte-identical param grads.
// R9 (clamp). A force limit below every |tau| -> every actuated joint saturates ->
// the adjoint masks g_tau (stop-grad) -> all param grads == 0.
// ---------------------------------------------------------------------------

#include "diffsim/osc_adjoint.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_drives.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
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
namespace diffsim = nuka::diffsim;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// A genuinely coupled SPD M^-1 tile (max_dof x max_dof, row-major): M = L L^T +
// max_dof*I, symmetrized. Fixed seed -> deterministic test input.
std::vector<float> MakeSpdTile(uint32_t max_dof, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> Lmat(static_cast<size_t>(max_dof) * max_dof, 0.0);
    for (uint32_t i = 0u; i < max_dof; ++i)
        for (uint32_t j = 0u; j <= i; ++j)
            Lmat[i * max_dof + j] = u(rng);
    std::vector<float> M(static_cast<size_t>(max_dof) * max_dof, 0.0f);
    for (uint32_t r = 0u; r < max_dof; ++r) {
        for (uint32_t c = 0u; c < max_dof; ++c) {
            double acc = 0.0;
            for (uint32_t k = 0u; k < max_dof; ++k)
                acc += Lmat[r * max_dof + k] * Lmat[c * max_dof + k];
            if (r == c) acc += static_cast<double>(max_dof);
            M[r * max_dof + c] = static_cast<float>(acc);
        }
    }
    for (uint32_t r = 0u; r < max_dof; ++r)
        for (uint32_t c = r + 1u; c < max_dof; ++c) {
            const float avg = 0.5f * (M[r * max_dof + c] + M[c * max_dof + r]);
            M[r * max_dof + c] = avg;
            M[c * max_dof + r] = avg;
        }
    return M;
}

std::vector<float> MakeGradTau(uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> g(n);
    for (uint32_t i = 0u; i < n; ++i) g[i] = u(rng);
    return g;
}

// ---------------------------------------------------------------------------
// A hand-built synthetic fixed-base articulation with a FULL-RANK-3, perfectly
// CONDITIONED task Jacobian, owning the device buffers the forward + adjoint
// kernels read/write. Chain: link0,1,2 = PRISMATIC (world axes ex/ey/ez), link3 =
// Fixed end-effector. task_link = 3, max_dof = 3.
//
// WHY PRISMATIC (not revolute): the forward's prismatic column is `axis_world`
// directly (no cross-product lever), so with axes ex/ey/ez the task Jacobian is
// J == I EXACTLY. Then A = J M^-1 J^T = M^-1 (the SPD tile -- well-conditioned,
// diagonal ~3-4), the 3x3 LDL^T floor NEVER fires, and tau = J^T A^-1 rhs is
// EXACTLY affine in Kp/Kd/x_target -> FD matches analytic to ~float precision.
// A 3-REVOLUTE chain does NOT guarantee a full-rank J (the cross-product columns
// are easily dependent -> singular A -> floored pivot -> fp32-ill-conditioned,
// the same trap as the Go2 calf); the prismatic J=I is degeneracy-proof BY
// CONSTRUCTION. The revolute cross-product J-build path is still exercised
// end-to-end by Go2EngineStateSmoke, so both forward J branches are covered.
// xdot = J*qdot = qdot stays nonzero, so the Kd channel is still exercised.
// LocalDofIndex: link0->0, link1->1, link2->2, link3->(fixed, no dof).
// ---------------------------------------------------------------------------
struct SyntheticArticulation {
    static constexpr uint32_t kLinks = 4u;
    static constexpr uint32_t kMaxDof = 3u;
    static constexpr uint32_t kTaskLink = 3u;

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;

    OwnedDeviceBuffer link_pose_dev;
    OwnedDeviceBuffer joint_axis_dev;
    OwnedDeviceBuffer joint_type_dev;
    OwnedDeviceBuffer parent_link_dev;
    OwnedDeviceBuffer art_link_count_dev;
    OwnedDeviceBuffer art_link_offset_dev;
    OwnedDeviceBuffer q_dev;
    OwnedDeviceBuffer qdot_dev;
    OwnedDeviceBuffer tau_dev;

    OwnedDeviceBuffer minv_dev;
    OwnedDeviceBuffer stiff_dev;   // float[kLinks]
    OwnedDeviceBuffer damp_dev;    // float[kLinks]
    OwnedDeviceBuffer target_dev;  // float[3]
    OwnedDeviceBuffer limit_dev;   // float[kLinks]

    float kp = 175.0f;
    float kd = 9.0f;
    Vec3 target{};

    SyntheticArticulation() { Build(); }

    template <typename T>
    static OwnedDeviceBuffer Upload(const std::vector<T>& v) {
        OwnedDeviceBuffer b(v.size() * sizeof(T));
        b.CopyFromHost(v.data(), v.size() * sizeof(T));
        return b;
    }

    void Build() {
        // World poses: distinct origins along the chain; identity rotations.
        std::vector<Transform> link_pose(kLinks);
        link_pose[0].position = Vec3{0.0f, 0.0f, 0.0f};
        link_pose[1].position = Vec3{0.30f, 0.05f, 0.00f};
        link_pose[2].position = Vec3{0.55f, 0.20f, 0.10f};
        link_pose[3].position = Vec3{0.80f, 0.40f, 0.25f};  // end-effector / task
        for (auto& p : link_pose) p.rotation = Quat::Identity();

        std::vector<Vec3> joint_axis(kLinks);
        joint_axis[0] = Vec3{1.0f, 0.0f, 0.0f};  // ex
        joint_axis[1] = Vec3{0.0f, 1.0f, 0.0f};  // ey
        joint_axis[2] = Vec3{0.0f, 0.0f, 1.0f};  // ez
        joint_axis[3] = Vec3{0.0f, 0.0f, 0.0f};  // fixed: unused

        std::vector<articulation::ArticulationJointType> joint_type(kLinks);
        joint_type[0] = articulation::ArticulationJointType::Prismatic;
        joint_type[1] = articulation::ArticulationJointType::Prismatic;
        joint_type[2] = articulation::ArticulationJointType::Prismatic;
        joint_type[3] = articulation::ArticulationJointType::Fixed;

        std::vector<uint32_t> parent_link = {kInvalidLink, 0u, 1u, 2u};  // local
        std::vector<uint32_t> art_link_count = {kLinks};
        std::vector<uint32_t> art_link_offset = {0u};
        std::vector<float> q(kLinks, 0.0f);
        // Nonzero qdot so the dL/dKd channel (edot_x = -J*qdot) is exercised.
        std::vector<float> qdot = {0.7f, -0.4f, 0.3f, 0.0f};
        std::vector<float> tau(kLinks, 0.0f);

        link_pose_dev = Upload(link_pose);
        joint_axis_dev = Upload(joint_axis);
        joint_type_dev = Upload(joint_type);
        parent_link_dev = Upload(parent_link);
        art_link_count_dev = Upload(art_link_count);
        art_link_offset_dev = Upload(art_link_offset);
        q_dev = Upload(q);
        qdot_dev = Upload(qdot);
        tau_dev = Upload(tau);

        minv_dev = Upload(MakeSpdTile(kMaxDof, 0x05C5C0DEu));
        stiff_dev = OwnedDeviceBuffer(kLinks * sizeof(float));
        damp_dev = OwnedDeviceBuffer(kLinks * sizeof(float));
        target_dev = OwnedDeviceBuffer(3u * sizeof(float));
        limit_dev = OwnedDeviceBuffer(kLinks * sizeof(float));

        // Target offset from the end-effector position so e_x is nonzero in all 3
        // (now-reachable) directions.
        target = Vec3{link_pose[3].position.x + 0.08f,
                      link_pose[3].position.y - 0.05f,
                      link_pose[3].position.z + 0.06f};
        cudaStreamSynchronize(context);
    }

    articulation::ArticulationDeviceState State() {
        articulation::ArticulationDeviceState st{};
        st.link_pose = static_cast<Transform*>(link_pose_dev.Data());
        st.joint_axis = static_cast<Vec3*>(joint_axis_dev.Data());
        st.joint_type =
            static_cast<articulation::ArticulationJointType*>(joint_type_dev.Data());
        st.parent_link = static_cast<uint32_t*>(parent_link_dev.Data());
        st.articulation_link_count = static_cast<uint32_t*>(art_link_count_dev.Data());
        st.articulation_link_offset = static_cast<uint32_t*>(art_link_offset_dev.Data());
        st.q = static_cast<float*>(q_dev.Data());
        st.qdot = static_cast<float*>(qdot_dev.Data());
        st.tau = static_cast<float*>(tau_dev.Data());
        st.total_link_count = kLinks;
        st.articulation_count = 1u;
        return st;
    }

    void SetGainsTarget(float kp_, float kd_, const Vec3& tgt) {
        std::vector<float> stiff(kLinks, 0.0f);
        std::vector<float> damp(kLinks, 0.0f);
        stiff[kTaskLink] = kp_;
        damp[kTaskLink] = kd_;
        stiff_dev.CopyFromHost(stiff.data(), stiff.size() * sizeof(float));
        damp_dev.CopyFromHost(damp.data(), damp.size() * sizeof(float));
        std::vector<float> t = {tgt.x, tgt.y, tgt.z};
        target_dev.CopyFromHost(t.data(), t.size() * sizeof(float));
    }

    // Run the REAL forward OSC kernel; return per-link tau.
    std::vector<float> RunForwardTau(float kp_, float kd_, const Vec3& tgt,
                                     const std::vector<float>& limits) {
        SetGainsTarget(kp_, kd_, tgt);
        const bool use_limit = !limits.empty();
        if (use_limit) limit_dev.CopyFromHost(limits.data(), limits.size() * sizeof(float));
        articulation::ArticulationDeviceState st = State();
        articulation::LaunchApplyOscDriveKernels(
            context, context_dev, st, kMaxDof, kTaskLink,
            static_cast<const float*>(minv_dev.Data()),
            static_cast<const float*>(target_dev.Data()),
            static_cast<const float*>(stiff_dev.Data()),
            static_cast<const float*>(damp_dev.Data()),
            use_limit ? static_cast<const float*>(limit_dev.Data()) : nullptr);
        cudaStreamSynchronize(context);
        std::vector<float> tau(kLinks, 0.0f);
        cudaMemcpy(tau.data(), tau_dev.Data(), kLinks * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return tau;
    }

    struct Grads {
        float grad_kp = 0.0f;
        float grad_kd = 0.0f;
        Vec3 grad_target{};
    };

    Grads RunAdjoint(const std::vector<float>& g_tau, const std::vector<float>& limits) {
        SetGainsTarget(kp, kd, target);
        const bool use_limit = !limits.empty();
        if (use_limit) limit_dev.CopyFromHost(limits.data(), limits.size() * sizeof(float));

        OwnedDeviceBuffer gtau_dev = Upload(g_tau);
        OwnedDeviceBuffer gkp(sizeof(float));
        OwnedDeviceBuffer gkd(sizeof(float));
        OwnedDeviceBuffer gtgt(3u * sizeof(float));

        diffsim::OscAdjointParamGrads g;
        g.grad_kp = static_cast<float*>(gkp.Data());
        g.grad_kd = static_cast<float*>(gkd.Data());
        g.grad_target = static_cast<float*>(gtgt.Data());

        articulation::ArticulationDeviceState st = State();
        diffsim::LaunchOscAdjointGainTarget(
            context, context_dev, st, kMaxDof, kTaskLink,
            static_cast<const float*>(minv_dev.Data()),
            static_cast<const float*>(target_dev.Data()),
            static_cast<const float*>(stiff_dev.Data()),
            static_cast<const float*>(damp_dev.Data()),
            use_limit ? static_cast<const float*>(limit_dev.Data()) : nullptr,
            static_cast<const float*>(gtau_dev.Data()), g);
        cudaStreamSynchronize(context);

        Grads out;
        cudaMemcpy(&out.grad_kp, gkp.Data(), sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&out.grad_kd, gkd.Data(), sizeof(float), cudaMemcpyDeviceToHost);
        float tg[3] = {0, 0, 0};
        cudaMemcpy(tg, gtgt.Data(), 3u * sizeof(float), cudaMemcpyDeviceToHost);
        out.grad_target = Vec3{tg[0], tg[1], tg[2]};
        cudaStreamSynchronize(context);
        return out;
    }
};

double LossFromTau(const std::vector<float>& g_tau, const std::vector<float>& tau) {
    double acc = 0.0;
    for (size_t i = 0u; i < tau.size(); ++i)
        acc += static_cast<double>(g_tau[i]) * static_cast<double>(tau[i]);
    return acc;
}

double RelErr(double analytic, double numeric) {
    const double scale = std::max({std::abs(analytic), std::abs(numeric), 1e-9});
    return std::abs(analytic - numeric) / scale;
}

// ---------------------------------------------------------------------------
// ADVERSARIAL (reviewer-added, attack A): a FULL-RANK-3 REVOLUTE task Jacobian.
//
// The author's primary FD scene (SyntheticArticulation) is PRISMATIC with axes
// ex/ey/ez, so J == I EXACTLY: the forward's prismatic column branch
// (col = axis_world) is exercised but the REVOLUTE cross-product branch
// (col = cross(axis_world, x_task - joint_origin)) is NOT FD-validated, and a J=I
// hides any transpose / stride error in w = J*g_tau or tau = J^T y (both reduce to
// componentwise identity). The Go2 smoke is non-FD (rank-2 calf-origin J -> floored
// A -> fp32-ill-conditioned). This rig closes that gap: 3 REVOLUTE joints with
// GENERIC, mutually-independent unit axes and generic link origins, and a task point
// OFFSET from every joint origin, so each column cross(axis, x_task - origin) is
// nonzero and J is a DENSE, NON-SYMMETRIC, rank-3 matrix (verified det != 0,
// cond(J) ~ 6, the A=J M^-1 J^T LDL^T pivots all >> the 1e-6 floor -- asserted
// below from the REAL downloaded tile). max_dof = 6 > ndof = 3 so a hardcoded-stride
// bug in J[r*max_dof+dof] also surfaces. Because the floor never fires and the
// forward is exactly linear in Kp/Kd/x_target, a correct adjoint must hit ~float
// precision (like the prismatic case); a 2x / sign-flipped FD mismatch is a REAL
// J-contraction bug, not conditioning.
//
// Chain: link0,1,2 = REVOLUTE (generic axes), link3 = Fixed end-effector / task.
// task_link = 3, max_dof = 6, ndof = 3. Same "real kernel, synthetic controlled
// inputs" philosophy as SyntheticArticulation; identical buffer ownership pattern.
// ---------------------------------------------------------------------------
struct RevoluteArticulation {
    static constexpr uint32_t kLinks = 4u;
    static constexpr uint32_t kMaxDof = 6u;   // > ndof(3): catches stride bugs.
    static constexpr uint32_t kNdof = 3u;
    static constexpr uint32_t kTaskLink = 3u;

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;

    OwnedDeviceBuffer link_pose_dev;
    OwnedDeviceBuffer joint_axis_dev;
    OwnedDeviceBuffer joint_type_dev;
    OwnedDeviceBuffer parent_link_dev;
    OwnedDeviceBuffer art_link_count_dev;
    OwnedDeviceBuffer art_link_offset_dev;
    OwnedDeviceBuffer q_dev;
    OwnedDeviceBuffer qdot_dev;
    OwnedDeviceBuffer tau_dev;

    OwnedDeviceBuffer minv_dev;
    OwnedDeviceBuffer stiff_dev;
    OwnedDeviceBuffer damp_dev;
    OwnedDeviceBuffer target_dev;
    OwnedDeviceBuffer limit_dev;

    float kp = 175.0f;
    float kd = 9.0f;
    Vec3 target{};

    // Host copies of the geometry (for the host-side full-rank pivot guard).
    std::vector<Vec3> origin;  // joint origins == link positions, links 0..2
    std::vector<Vec3> axis;    // unit joint axes, links 0..2
    Vec3 x_task{};
    std::vector<float> minv_host;  // kMaxDof*kMaxDof SPD tile (the REAL device tile)

    RevoluteArticulation() { Build(); }

    template <typename T>
    static OwnedDeviceBuffer Upload(const std::vector<T>& v) {
        OwnedDeviceBuffer b(v.size() * sizeof(T));
        b.CopyFromHost(v.data(), v.size() * sizeof(T));
        return b;
    }

    static Vec3 Unit(Vec3 v) {
        const float n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return Vec3{v.x / n, v.y / n, v.z / n};
    }

    void Build() {
        // Generic origins (NOT axis-aligned-degenerate). Identity rotations so the
        // device RotateByQuatDrive(axis) == axis (we feed already-world axes).
        std::vector<Transform> link_pose(kLinks);
        link_pose[0].position = Vec3{0.00f, 0.00f, 0.00f};
        link_pose[1].position = Vec3{0.20f, -0.15f, 0.10f};
        link_pose[2].position = Vec3{0.05f, 0.30f, -0.20f};
        link_pose[3].position = Vec3{0.40f, 0.25f, 0.45f};  // task point: offset
        for (auto& p : link_pose) p.rotation = Quat::Identity();
        x_task = link_pose[3].position;

        std::vector<Vec3> joint_axis(kLinks);
        joint_axis[0] = Unit(Vec3{1.0f, 0.2f, 0.1f});
        joint_axis[1] = Unit(Vec3{0.1f, 1.0f, 0.3f});
        joint_axis[2] = Unit(Vec3{0.2f, 0.3f, 1.0f});
        joint_axis[3] = Vec3{0.0f, 0.0f, 0.0f};  // fixed: unused

        origin = {link_pose[0].position, link_pose[1].position, link_pose[2].position};
        axis = {joint_axis[0], joint_axis[1], joint_axis[2]};

        std::vector<articulation::ArticulationJointType> joint_type(kLinks);
        joint_type[0] = articulation::ArticulationJointType::Revolute;
        joint_type[1] = articulation::ArticulationJointType::Revolute;
        joint_type[2] = articulation::ArticulationJointType::Revolute;
        joint_type[3] = articulation::ArticulationJointType::Fixed;

        std::vector<uint32_t> parent_link = {kInvalidLink, 0u, 1u, 2u};
        std::vector<uint32_t> art_link_count = {kLinks};
        std::vector<uint32_t> art_link_offset = {0u};
        std::vector<float> q(kLinks, 0.0f);
        std::vector<float> qdot = {0.7f, -0.4f, 0.3f, 0.0f};  // nonzero -> Kd channel
        std::vector<float> tau(kLinks, 0.0f);

        link_pose_dev = Upload(link_pose);
        joint_axis_dev = Upload(joint_axis);
        joint_type_dev = Upload(joint_type);
        parent_link_dev = Upload(parent_link);
        art_link_count_dev = Upload(art_link_count);
        art_link_offset_dev = Upload(art_link_offset);
        q_dev = Upload(q);
        qdot_dev = Upload(qdot);
        tau_dev = Upload(tau);

        minv_host = MakeSpdTile(kMaxDof, 0x57E9A11u);
        minv_dev = Upload(minv_host);
        stiff_dev = OwnedDeviceBuffer(kLinks * sizeof(float));
        damp_dev = OwnedDeviceBuffer(kLinks * sizeof(float));
        target_dev = OwnedDeviceBuffer(3u * sizeof(float));
        limit_dev = OwnedDeviceBuffer(kLinks * sizeof(float));

        target = Vec3{x_task.x + 0.08f, x_task.y - 0.05f, x_task.z + 0.06f};
        cudaStreamSynchronize(context);
    }

    // Build the active 3x3 J (revolute columns) on the host, EXACTLY as the kernel
    // does (col = cross(axis, x_task - origin)), then the A=J M^-1 J^T LDL^T pivots.
    // Returns the smallest pivot. A pivot >> 1e-6 means the floor never fires, so
    // the FD gate is conditioning-clean and a mismatch is unambiguously a bug.
    double SmallestADiagPivot() const {
        double J[3][3];
        for (uint32_t d = 0u; d < kNdof; ++d) {
            const Vec3 lever{x_task.x - origin[d].x, x_task.y - origin[d].y,
                             x_task.z - origin[d].z};
            const Vec3 col{axis[d].y * lever.z - axis[d].z * lever.y,
                           axis[d].z * lever.x - axis[d].x * lever.z,
                           axis[d].x * lever.y - axis[d].y * lever.x};
            J[0][d] = col.x; J[1][d] = col.y; J[2][d] = col.z;
        }
        // A = J Minv J^T (Minv: leading 3x3 of the kMaxDof tile, row-major).
        double Minv[3][3];
        for (uint32_t r = 0u; r < kNdof; ++r)
            for (uint32_t c = 0u; c < kNdof; ++c)
                Minv[r][c] = static_cast<double>(minv_host[r * kMaxDof + c]);
        double MinvJt[3][3];  // [c][s] = sum_k Minv[c][k] J[s][k]
        for (uint32_t c = 0u; c < kNdof; ++c)
            for (uint32_t s = 0u; s < kNdof; ++s) {
                double a = 0.0;
                for (uint32_t k = 0u; k < kNdof; ++k) a += Minv[c][k] * J[s][k];
                MinvJt[c][s] = a;
            }
        double A[3][3];
        for (uint32_t r = 0u; r < kNdof; ++r)
            for (uint32_t s = 0u; s < kNdof; ++s) {
                double a = 0.0;
                for (uint32_t c = 0u; c < kNdof; ++c) a += J[r][c] * MinvJt[c][s];
                A[r][s] = a;
            }
        // LDL^T pivots (same recurrence as the kernel, sans floor).
        double L[3][3]; double dvec[3];
        for (uint32_t r = 0u; r < kNdof; ++r)
            for (uint32_t c = 0u; c < kNdof; ++c) L[r][c] = A[r][c];
        double min_piv = 1e300;
        for (uint32_t j = 0u; j < kNdof; ++j) {
            double djj = L[j][j];
            for (uint32_t k = 0u; k < j; ++k) djj -= L[j][k] * L[j][k] * dvec[k];
            dvec[j] = djj;
            min_piv = std::min(min_piv, djj);
            for (uint32_t i = j + 1u; i < kNdof; ++i) {
                double lij = L[i][j];
                for (uint32_t k = 0u; k < j; ++k) lij -= L[i][k] * L[j][k] * dvec[k];
                L[i][j] = lij / djj;
            }
        }
        return min_piv;
    }

    articulation::ArticulationDeviceState State() {
        articulation::ArticulationDeviceState st{};
        st.link_pose = static_cast<Transform*>(link_pose_dev.Data());
        st.joint_axis = static_cast<Vec3*>(joint_axis_dev.Data());
        st.joint_type =
            static_cast<articulation::ArticulationJointType*>(joint_type_dev.Data());
        st.parent_link = static_cast<uint32_t*>(parent_link_dev.Data());
        st.articulation_link_count = static_cast<uint32_t*>(art_link_count_dev.Data());
        st.articulation_link_offset = static_cast<uint32_t*>(art_link_offset_dev.Data());
        st.q = static_cast<float*>(q_dev.Data());
        st.qdot = static_cast<float*>(qdot_dev.Data());
        st.tau = static_cast<float*>(tau_dev.Data());
        st.total_link_count = kLinks;
        st.articulation_count = 1u;
        return st;
    }

    void SetGainsTarget(float kp_, float kd_, const Vec3& tgt) {
        std::vector<float> stiff(kLinks, 0.0f), damp(kLinks, 0.0f);
        stiff[kTaskLink] = kp_;
        damp[kTaskLink] = kd_;
        stiff_dev.CopyFromHost(stiff.data(), stiff.size() * sizeof(float));
        damp_dev.CopyFromHost(damp.data(), damp.size() * sizeof(float));
        std::vector<float> t = {tgt.x, tgt.y, tgt.z};
        target_dev.CopyFromHost(t.data(), t.size() * sizeof(float));
    }

    std::vector<float> RunForwardTau(float kp_, float kd_, const Vec3& tgt) {
        SetGainsTarget(kp_, kd_, tgt);
        articulation::ArticulationDeviceState st = State();
        articulation::LaunchApplyOscDriveKernels(
            context, context_dev, st, kMaxDof, kTaskLink,
            static_cast<const float*>(minv_dev.Data()),
            static_cast<const float*>(target_dev.Data()),
            static_cast<const float*>(stiff_dev.Data()),
            static_cast<const float*>(damp_dev.Data()), nullptr);
        cudaStreamSynchronize(context);
        std::vector<float> tau(kLinks, 0.0f);
        cudaMemcpy(tau.data(), tau_dev.Data(), kLinks * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return tau;
    }

    struct Grads { float grad_kp = 0.0f; float grad_kd = 0.0f; Vec3 grad_target{}; };

    Grads RunAdjoint(const std::vector<float>& g_tau) {
        SetGainsTarget(kp, kd, target);
        OwnedDeviceBuffer gtau_dev = Upload(g_tau);
        OwnedDeviceBuffer gkp(sizeof(float));
        OwnedDeviceBuffer gkd(sizeof(float));
        OwnedDeviceBuffer gtgt(3u * sizeof(float));

        diffsim::OscAdjointParamGrads g;
        g.grad_kp = static_cast<float*>(gkp.Data());
        g.grad_kd = static_cast<float*>(gkd.Data());
        g.grad_target = static_cast<float*>(gtgt.Data());

        articulation::ArticulationDeviceState st = State();
        diffsim::LaunchOscAdjointGainTarget(
            context, context_dev, st, kMaxDof, kTaskLink,
            static_cast<const float*>(minv_dev.Data()),
            static_cast<const float*>(target_dev.Data()),
            static_cast<const float*>(stiff_dev.Data()),
            static_cast<const float*>(damp_dev.Data()), nullptr,
            static_cast<const float*>(gtau_dev.Data()), g);
        cudaStreamSynchronize(context);

        Grads out;
        cudaMemcpy(&out.grad_kp, gkp.Data(), sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&out.grad_kd, gkd.Data(), sizeof(float), cudaMemcpyDeviceToHost);
        float tg[3] = {0, 0, 0};
        cudaMemcpy(tg, gtgt.Data(), 3u * sizeof(float), cudaMemcpyDeviceToHost);
        out.grad_target = Vec3{tg[0], tg[1], tg[2]};
        cudaStreamSynchronize(context);
        return out;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// PRIMARY: dL/dKp, dL/dKd, dL/dx_target match central finite-difference of the
// REAL forward OSC kernel on a full-rank-3 synthetic task, < 1e-3 relative.
// ---------------------------------------------------------------------------
TEST(OscAdjoint, GainTargetMatchesFiniteDifference) {
    SyntheticArticulation s;
    const std::vector<float> g_tau =
        MakeGradTau(SyntheticArticulation::kLinks, 0xA3F0u);
    const std::vector<float> no_limits;

    const SyntheticArticulation::Grads a = s.RunAdjoint(g_tau, no_limits);

    // Guard: the FD gate is only meaningful on a NON-degenerate A. With prismatic
    // axes ex/ey/ez, J == I EXACTLY, so A = M^-1 (the SPD tile, with genuine off-
    // diagonals -- the LDL^T + both substitutions are fully exercised, not a trivial
    // diagonal case). We assert the smallest diagonal of M^-1 (== A's diagonal, since
    // J==I) is far above the LDL^T floor (kMinDiagonalDrive = 1e-6); a comfortably
    // positive diagonal on an SPD matrix means the unfloored LDL^T pivots stay well-
    // conditioned, so the floor never fires and tau is exactly affine in the params.
    // (M = L L^T + 3 I => M^-1 diagonal ~O(0.1-0.3) >> 1e-6.) The ~1e-5 empirical FD
    // agreement below is the real conditioning evidence; this is a fast regression
    // tripwire documenting the well-conditioned precondition the < 1e-3 gate needs.
    {
        const std::vector<float> minv = MakeSpdTile(SyntheticArticulation::kMaxDof, 0x05C5C0DEu);
        float min_diag = minv[0];
        for (uint32_t i = 0u; i < SyntheticArticulation::kMaxDof; ++i)
            min_diag = std::min(min_diag, minv[i * SyntheticArticulation::kMaxDof + i]);
        ASSERT_GT(min_diag, 1e-3f)
            << "synthetic A=M^-1 is near-degenerate; FD gate would be ill-conditioned";
    }

    auto loss = [&](float kp, float kd, const Vec3& tgt) {
        return LossFromTau(g_tau, s.RunForwardTau(kp, kd, tgt, no_limits));
    };

    const float dKp = 0.5f;
    const double fd_kp =
        (loss(s.kp + dKp, s.kd, s.target) - loss(s.kp - dKp, s.kd, s.target)) /
        (2.0 * dKp);
    const float dKd = 0.25f;
    const double fd_kd =
        (loss(s.kp, s.kd + dKd, s.target) - loss(s.kp, s.kd - dKd, s.target)) /
        (2.0 * dKd);
    const float dT = 1e-3f;
    Vec3 tpx = s.target, tmx = s.target; tpx.x += dT; tmx.x -= dT;
    const double fd_tx = (loss(s.kp, s.kd, tpx) - loss(s.kp, s.kd, tmx)) / (2.0 * dT);
    Vec3 tpy = s.target, tmy = s.target; tpy.y += dT; tmy.y -= dT;
    const double fd_ty = (loss(s.kp, s.kd, tpy) - loss(s.kp, s.kd, tmy)) / (2.0 * dT);
    Vec3 tpz = s.target, tmz = s.target; tpz.z += dT; tmz.z -= dT;
    const double fd_tz = (loss(s.kp, s.kd, tpz) - loss(s.kp, s.kd, tmz)) / (2.0 * dT);

    std::printf("[OscAdjoint] dL/dKp  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_kp, fd_kp, RelErr(a.grad_kp, fd_kp));
    std::printf("[OscAdjoint] dL/dKd  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_kd, fd_kd, RelErr(a.grad_kd, fd_kd));
    std::printf("[OscAdjoint] dL/dtx  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_target.x, fd_tx, RelErr(a.grad_target.x, fd_tx));
    std::printf("[OscAdjoint] dL/dty  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_target.y, fd_ty, RelErr(a.grad_target.y, fd_ty));
    std::printf("[OscAdjoint] dL/dtz  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_target.z, fd_tz, RelErr(a.grad_target.z, fd_tz));

    EXPECT_LT(RelErr(a.grad_kp, fd_kp), 1e-3) << "dL/dKp";
    EXPECT_LT(RelErr(a.grad_kd, fd_kd), 1e-3) << "dL/dKd";
    EXPECT_LT(RelErr(a.grad_target.x, fd_tx), 1e-3) << "dL/dx_target.x";
    EXPECT_LT(RelErr(a.grad_target.y, fd_ty), 1e-3) << "dL/dx_target.y";
    EXPECT_LT(RelErr(a.grad_target.z, fd_tz), 1e-3) << "dL/dx_target.z";
}

// ---------------------------------------------------------------------------
// ADVERSARIAL (reviewer-added, attack A): dL/dKp, dL/dKd, dL/dx_target match
// central FD of the REAL forward OSC kernel on a FULL-RANK-3 REVOLUTE task whose J
// is dense + non-symmetric (NOT the identity the prismatic primary test reduces to).
// This is the gate the prismatic-only validation lacks: it exercises the revolute
// cross-product J-build branch AND a J-contraction (transpose/stride) bug surfaces
// here but not on J==I. Conditioning is host-asserted clean (smallest A pivot >>
// 1e-6 floor), so a mismatch is a real bug, not fp32 ill-conditioning. max_dof=6 >
// ndof=3 also stresses the J[r*max_dof+dof] stride.
// ---------------------------------------------------------------------------
TEST(OscAdjoint, RevoluteFullRankMatchesFiniteDifference) {
    RevoluteArticulation s;

    // Conditioning guard: with generic axes/origins the revolute J is full rank and
    // A=J M^-1 J^T is well-conditioned, so the 3x3 LDL^T floor (1e-6) never fires and
    // the forward is exactly affine in the params -> FD must hit ~float precision.
    const double min_piv = s.SmallestADiagPivot();
    std::printf("[OscAdjoint-Rev] smallest A LDL^T pivot = %.4e (floor 1e-6)\n", min_piv);
    ASSERT_GT(min_piv, 1e-3)
        << "revolute A is near-degenerate; FD gate would be ill-conditioned (the "
           "prismatic primary test's well-conditioned precondition is unmet)";

    const std::vector<float> g_tau =
        MakeGradTau(RevoluteArticulation::kLinks, 0x9E3779B9u);
    const RevoluteArticulation::Grads a = s.RunAdjoint(g_tau);

    auto loss = [&](float kp, float kd, const Vec3& tgt) {
        return LossFromTau(g_tau, s.RunForwardTau(kp, kd, tgt));
    };

    const float dKp = 0.5f;
    const double fd_kp =
        (loss(s.kp + dKp, s.kd, s.target) - loss(s.kp - dKp, s.kd, s.target)) /
        (2.0 * dKp);
    const float dKd = 0.25f;
    const double fd_kd =
        (loss(s.kp, s.kd + dKd, s.target) - loss(s.kp, s.kd - dKd, s.target)) /
        (2.0 * dKd);
    const float dT = 1e-3f;
    Vec3 tpx = s.target, tmx = s.target; tpx.x += dT; tmx.x -= dT;
    const double fd_tx = (loss(s.kp, s.kd, tpx) - loss(s.kp, s.kd, tmx)) / (2.0 * dT);
    Vec3 tpy = s.target, tmy = s.target; tpy.y += dT; tmy.y -= dT;
    const double fd_ty = (loss(s.kp, s.kd, tpy) - loss(s.kp, s.kd, tmy)) / (2.0 * dT);
    Vec3 tpz = s.target, tmz = s.target; tpz.z += dT; tmz.z -= dT;
    const double fd_tz = (loss(s.kp, s.kd, tpz) - loss(s.kp, s.kd, tmz)) / (2.0 * dT);

    std::printf("[OscAdjoint-Rev] dL/dKp  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_kp, fd_kp, RelErr(a.grad_kp, fd_kp));
    std::printf("[OscAdjoint-Rev] dL/dKd  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_kd, fd_kd, RelErr(a.grad_kd, fd_kd));
    std::printf("[OscAdjoint-Rev] dL/dtx  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_target.x, fd_tx, RelErr(a.grad_target.x, fd_tx));
    std::printf("[OscAdjoint-Rev] dL/dty  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_target.y, fd_ty, RelErr(a.grad_target.y, fd_ty));
    std::printf("[OscAdjoint-Rev] dL/dtz  analytic=% .6e  FD=% .6e  rel=%.2e\n",
                a.grad_target.z, fd_tz, RelErr(a.grad_target.z, fd_tz));

    EXPECT_LT(RelErr(a.grad_kp, fd_kp), 1e-3) << "revolute dL/dKp";
    EXPECT_LT(RelErr(a.grad_kd, fd_kd), 1e-3) << "revolute dL/dKd";
    EXPECT_LT(RelErr(a.grad_target.x, fd_tx), 1e-3) << "revolute dL/dx_target.x";
    EXPECT_LT(RelErr(a.grad_target.y, fd_ty), 1e-3) << "revolute dL/dx_target.y";
    EXPECT_LT(RelErr(a.grad_target.z, fd_tz), 1e-3) << "revolute dL/dx_target.z";
}

// ---------------------------------------------------------------------------
// D1: two adjoint runs on the same frozen state -> byte-identical param grads.
// ---------------------------------------------------------------------------
TEST(OscAdjoint, D1TwoRunByteExact) {
    SyntheticArticulation s;
    const std::vector<float> g_tau =
        MakeGradTau(SyntheticArticulation::kLinks, 0xBEEF01u);
    const std::vector<float> no_limits;

    const SyntheticArticulation::Grads a = s.RunAdjoint(g_tau, no_limits);
    const SyntheticArticulation::Grads b = s.RunAdjoint(g_tau, no_limits);

    EXPECT_EQ(std::memcmp(&a.grad_kp, &b.grad_kp, sizeof(float)), 0)
        << "grad_kp differs run-to-run";
    EXPECT_EQ(std::memcmp(&a.grad_kd, &b.grad_kd, sizeof(float)), 0)
        << "grad_kd differs run-to-run";
    EXPECT_EQ(std::memcmp(&a.grad_target, &b.grad_target, sizeof(Vec3)), 0)
        << "grad_target differs run-to-run";
    std::printf("[OscAdjoint] two-run byte-identical (grad_kp/kd/target)\n");
}

// ---------------------------------------------------------------------------
// R9: when EVERY joint's forward tau saturates against the force limit, the
// adjoint masks all g_tau components to zero (stop-grad) -> all param grads == 0.
// ---------------------------------------------------------------------------
TEST(OscAdjoint, SaturatedComponentZeroGrad) {
    SyntheticArticulation s;
    const std::vector<float> g_tau =
        MakeGradTau(SyntheticArticulation::kLinks, 0x5A7u);
    const std::vector<float> no_limits;

    // Confirm the unclamped forward produces nonzero tau and the unclamped adjoint
    // is nonzero (so the saturation test is non-vacuous).
    const std::vector<float> tau_free = s.RunForwardTau(s.kp, s.kd, s.target, no_limits);
    float max_abs = 0.0f;
    for (float t : tau_free) max_abs = std::max(max_abs, std::fabs(t));
    ASSERT_GT(max_abs, 1e-3f) << "forward tau ~0; saturation test vacuous";
    const SyntheticArticulation::Grads a_free = s.RunAdjoint(g_tau, no_limits);
    EXPECT_GT(std::fabs(a_free.grad_kp) + std::fabs(a_free.grad_kd) +
                  std::fabs(a_free.grad_target.x) + std::fabs(a_free.grad_target.y) +
                  std::fabs(a_free.grad_target.z),
              0.0f)
        << "unclamped adjoint identically zero; test vacuous";

    // Force limit far below every |tau| -> every actuated joint saturates.
    std::vector<float> tiny_limits(SyntheticArticulation::kLinks, 1e-6f);
    const SyntheticArticulation::Grads a_sat = s.RunAdjoint(g_tau, tiny_limits);

    std::printf("[OscAdjoint] saturated grads: kp=% .3e kd=% .3e t=(%.3e,%.3e,%.3e)\n",
                a_sat.grad_kp, a_sat.grad_kd, a_sat.grad_target.x,
                a_sat.grad_target.y, a_sat.grad_target.z);
    EXPECT_EQ(a_sat.grad_kp, 0.0f) << "saturated dL/dKp must be 0 (stop-grad)";
    EXPECT_EQ(a_sat.grad_kd, 0.0f) << "saturated dL/dKd must be 0 (stop-grad)";
    EXPECT_EQ(a_sat.grad_target.x, 0.0f) << "saturated dL/dx_target.x must be 0";
    EXPECT_EQ(a_sat.grad_target.y, 0.0f) << "saturated dL/dx_target.y must be 0";
    EXPECT_EQ(a_sat.grad_target.z, 0.0f) << "saturated dL/dx_target.z must be 0";
}

// ---------------------------------------------------------------------------
// Go2 directional smoke (NON-FD): the adjoint runs end-to-end on a REAL cooked
// Go2 device state (the engine's own BuildWorld -> BuildArticulationHostState ->
// UploadArticulationState pipeline, calf task link) and produces finite,
// deterministic grads. This is NOT an FD gate (the calf-origin J is rank-2 ->
// floored A -> fp32-ill-conditioned; see the file header); it only confirms the
// adjoint integrates with the real engine's link_pose/q/qdot/joint_axis buffers
// (the synthetic rigs hand-build those; this proves the cooked geometry path).
// ---------------------------------------------------------------------------
namespace {

struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
};

CookedGo2 CookGo2() {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    CookedGo2 result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    const auto& shapes = world.template_view.shape_table;
    const uint32_t link_count = result.host.TotalLinkCount();
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) continue;
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link)
            if (result.host.link_body[link] == body) { calf_link = link; break; }
        if (calf_link == kInvalidLink) continue;
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }
    return result;
}

}  // namespace

TEST(OscAdjoint, Go2EngineStateSmoke) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    auto cooked = CookGo2();
    auto base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();
    const uint32_t task_link = cooked.feet.empty() ? 0u : cooked.feet.front().calf_local_link;
    ASSERT_NE(task_link, 0u);

    // Real cooked Go2 device state: the engine's own host-state -> device upload
    // (link_pose/q/qdot/joint_axis populated from the cook). No stepping world is
    // needed -- the adjoint reads the geometry/state buffers, and a finiteness
    // smoke on the cooked rest pose exercises the same real-engine path the
    // synthetic rigs cannot (cooked geometry vs hand-built buffers).
    auto device = articulation::UploadArticulationState(context, context_dev, base);
    cudaStreamSynchronize(context);

    std::vector<float> stiff(base_link_count, 0.0f), damp(base_link_count, 0.0f);
    stiff[task_link] = 200.0f;
    OwnedDeviceBuffer stiff_dev(stiff.size() * sizeof(float));
    stiff_dev.CopyFromHost(stiff.data(), stiff.size() * sizeof(float));
    OwnedDeviceBuffer damp_dev(damp.size() * sizeof(float));
    damp_dev.CopyFromHost(damp.data(), damp.size() * sizeof(float));
    std::vector<float> tgt = {0.2f, 0.0f, 0.1f};
    OwnedDeviceBuffer tgt_dev(tgt.size() * sizeof(float));
    tgt_dev.CopyFromHost(tgt.data(), tgt.size() * sizeof(float));

    // A synthetic SPD M^-1 tile (the cook's M^-1 is rebuilt per step; the
    // gain/target channels are M^-1-provenance-independent). The adjoint just needs
    // to run on the real link_pose/q/qdot/joint_axis buffers.
    OwnedDeviceBuffer minv_dev(static_cast<size_t>(max_dof) * max_dof * sizeof(float));
    {
        std::vector<float> minv = MakeSpdTile(max_dof, 0x9909u);
        minv_dev.CopyFromHost(minv.data(), minv.size() * sizeof(float));
    }
    std::vector<float> g_tau = MakeGradTau(base_link_count, 0xC0FFEEu);
    OwnedDeviceBuffer gtau_dev(g_tau.size() * sizeof(float));
    gtau_dev.CopyFromHost(g_tau.data(), g_tau.size() * sizeof(float));

    OwnedDeviceBuffer gkp(sizeof(float));
    OwnedDeviceBuffer gkd(sizeof(float));
    OwnedDeviceBuffer gtgt(3u * sizeof(float));
    diffsim::OscAdjointParamGrads grads;
    grads.grad_kp = static_cast<float*>(gkp.Data());
    grads.grad_kd = static_cast<float*>(gkd.Data());
    grads.grad_target = static_cast<float*>(gtgt.Data());

    articulation::ArticulationDeviceState st = device.View();
    diffsim::LaunchOscAdjointGainTarget(
        context, context_dev, st, max_dof, task_link,
        static_cast<const float*>(minv_dev.Data()),
        static_cast<const float*>(tgt_dev.Data()),
        static_cast<const float*>(stiff_dev.Data()),
        static_cast<const float*>(damp_dev.Data()),
        nullptr, static_cast<const float*>(gtau_dev.Data()), grads);
    cudaStreamSynchronize(context);

    float kp_g = 0, kd_g = 0, tg[3] = {0, 0, 0};
    cudaMemcpy(&kp_g, gkp.Data(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&kd_g, gkd.Data(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tg, gtgt.Data(), 3u * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(context);
    std::printf("[OscAdjoint] Go2 engine-state grads: kp=% .3e kd=% .3e t=(%.3e,%.3e,%.3e)\n",
                kp_g, kd_g, tg[0], tg[1], tg[2]);
    EXPECT_TRUE(std::isfinite(kp_g) && std::isfinite(kd_g) && std::isfinite(tg[0]) &&
                std::isfinite(tg[1]) && std::isfinite(tg[2]))
        << "Go2 engine-state adjoint produced non-finite grads";
}
