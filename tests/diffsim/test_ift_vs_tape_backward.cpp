// ---------------------------------------------------------------------------
// v0.5 p03 R3a -- IFT-at-convergence contact gradient: FD oracle + dense ref + D1
// ---------------------------------------------------------------------------
//
// Validates nuka::diffsim::IftRunner -- the IFT contact-gradient path. Given the
// downstream cotangent g = dL/dqdot_post on a CONVERGED contact step, IftRunner
// solves ONCE on the SPD Delassus  z = A^-1 (J M^-1 g),  A = J M^-1 J^T, and
// distributes
//     grad_qdot_free = g - J^T z   (GLOBAL total_link_count layout)
//     grad_b_c       = z           (contact-row / block layout)
// replacing taping the fixed-48-iteration PGS inner loop.
//
// IFT-vs-TAPE IS N/A HERE. p02's tape/backward is CONTACT-FREE -- it deliberately
// excluded contacts, deferring contact gradients to THIS phase. So there is NO
// contact-tape to compare against; the filename is kept (the plan names it) but
// the validation is finite-difference + dense reference, NOT an IFT-vs-tape check.
//
// ORACLES (sequenced):
//  (A) MatchesDenseReference -- feed a host Eigen::LDLT solve+distribute the SAME
//      GPU-BUILT A (downloaded), J, M^-1, g; compare to the GPU IFT path (< 1e-5).
//      Isolates the runner's solve+distribute from any kkt assembly discrepancy
//      (which test_kkt_build already gates). FAILURE => linear-algebra / indexing
//      bug in the runner.
//  (B) MatchesFiniteDifference -- the PRIMARY oracle. Perturb a contact-solve input
//      (a qdot_free component; and the normal-row depth -> b_c) and re-run the REAL
//      engine contact step (SolveArticulatedContactRows, the actual 48-iter PGS),
//      measure the numerical dL/dinput, compare to the IFT analytic gradient
//      (< 1e-3 relative). Active set held fixed (small eps; no slot enters/leaves,
//      no lambda_n crosses 0). FAILURE while (A) passes => PGS non-convergence
//      (R3), not a code bug.
//  (C) D1TwoRunByteExact -- two IFT backward runs, memcmp the grad buffers ->
//      byte-identical (R6 extends to the backward/solver outputs).
//
// NEAR-CONVERGENCE (R3, documented + asserted): the scenes are NORMAL-DOMINANT
// resting contacts. The pendulum is 1-DOF normal-only (one unclamped row -> the
// final normal-only sweep makes J*qdot_post == bias to machine precision, so the
// 48-iter map IS the converged fixed point). The test prints the post-solve normal
// residual |J qdot_post - bias| to evidence convergence. SLIDING / cone-boundary
// friction is genuinely nonsmooth and OUT OF SCOPE (kkt sticking-row assumption);
// the friction-coupled scene is kept normal-dominant (lambda_t small, strictly
// interior to the cone). joint_damping is OFF so qdot_free == pre-solve state.qdot
// (a clean direct FD channel).
//
// SHIPPED CHANNELS: grad_qdot_free, grad_b_c. DEFERRED: d/dM, d/dJ (the d(A^-1) =
// -A^-1 dA A^-1 + product-rule channels) -- not implemented this round.
// ---------------------------------------------------------------------------

#include "diffsim/backward_runner.hpp"
#include "diffsim/ift_runner.hpp"
#include "diffsim/kkt_builder.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace diffsim = nuka::diffsim;
namespace art = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;
constexpr uint32_t kMd = diffsim::kMaxBlockDim;          // 12
constexpr uint32_t kSlots = art::kMaxFootContactsPerEnv; // 4

// ---------------------------------------------------------------------------
// A controlled multi-DOF fixed-base revolute chain with point masses, plus a
// hand-placed downward (+Z) contact at the last link's tip. Fixed-base keeps the
// dof column -> qdot map 1:1 (no FloatingBase base DOFs), and a resting downward
// contact with light tangential coupling is normal-dominant + near-converged.
// One revolute link per DOF, all about +Y through stacked offsets along +X, so
// vertical (+Z) motion of the tip is reachable through every DOF.
// ---------------------------------------------------------------------------
art::ArticulationHostState BuildChain(uint32_t n_dof, float seg_len, float mass) {
    art::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    for (uint32_t i = 0u; i < n_dof; ++i) {
        topo.link_bodies.push_back(i);
        topo.parent_links.push_back(i == 0u ? kInvalidLink : (i - 1u));
        topo.joint_types.push_back(art::ArticulationJointType::Revolute);
        topo.joint_axes.push_back(Vec3{0.0f, 1.0f, 0.0f});  // about +Y
        // Each child is offset +X by seg_len from its parent's frame.
        const Transform parent_frame =
            (i == 0u) ? Transform::Identity()
                      : Transform({seg_len, 0.0f, 0.0f}, Quat::Identity());
        topo.parent_frames.push_back(parent_frame);
        topo.child_frames.push_back(Transform::Identity());
        topo.local_poses.push_back(Transform::Identity());
        // Point mass at the link's own tip (+X seg_len in the link frame).
        topo.inertial_frames.push_back(
            Transform({seg_len, 0.0f, 0.0f}, Quat::Identity()));
        topo.initial_positions.push_back(0.0f);
        topo.joint_dampings.push_back(0.0f);   // damping OFF (clean qdot_free channel)
        topo.joint_armatures.push_back(0.0f);
        topo.masses.push_back(mass);
        topo.inertias.push_back(Vec3{0.0f, 0.0f, 0.0f});  // pure point mass
    }

    nuka::scene::CookedBodyTable bodies;
    for (uint32_t i = 0u; i < n_dof; ++i) {
        bodies.poses.push_back(Transform::Identity());
        bodies.local_poses.push_back(Transform::Identity());
        bodies.inertial_frames.push_back(Transform::Identity());
        bodies.masses.push_back(mass);
        bodies.inertias.push_back(Vec3{0.0f, 0.0f, 0.0f});
        bodies.inv_masses.push_back(mass > 0.0f ? 1.0f / mass : 0.0f);
        bodies.inv_inertias.push_back(Vec3{0.0f, 0.0f, 0.0f});
        bodies.is_static.push_back(0u);
    }
    return art::BuildArticulationHostState({topo}, bodies);
}

// Local DOF -> global link map (the host twin of the runner's dof_to_link; used
// to set per-DOF qdot_free and read per-DOF grads). Fixed-base -> one DOF/link.
std::vector<uint32_t> DofToLink(const art::ArticulationHostState& host) {
    const uint32_t offset = host.articulation_link_offset[0];
    const uint32_t count = host.articulation_link_count[0];
    std::vector<uint32_t> map;
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (art::ArticulationJointDofCount(host.joint_type[link]) != 0u) {
            map.push_back(link);
        }
    }
    return map;
}

// Everything one forward contact step produces + the device buffers the IFT path
// reuses. The device buffers are kept ALIVE (the IFT inputs borrow their pointers).
struct ContactStep {
    // Live device world (topology + the post-solve state).
    art::ArticulationDeviceBuffers buffers;
    art::ArticulationDeviceState view;
    // Owned device contact buffers (IFT borrows these).
    nuka::phi::Buffer rows_buf, jn_buf, jt1_buf, jt2_buf, minv_buf;
    // Host readback.
    std::vector<float> qdot_free;   // pre-solve (== host.qdot input)
    std::vector<float> qdot_post;   // post-solve
    std::vector<art::ArticulatedContactRow> rows;
    std::vector<float> minv;        // per-articulation M^-1 tile (stride dof^2)
    std::vector<float> jn, jt1, jt2;  // per-slot dof-wide chain rows
    uint32_t dof = 0u;
    uint32_t slot_count = 0u;
};

// Runs the REAL engine contact forward (M -> M^-1 -> J(n,t1,t2) -> m_eff ->
// assemble -> SolveArticulatedContactRows). `host.qdot` IS qdot_free. friction
// off-by-default-OFF would make t-rows inert; we keep the real mu so the friction
// rows are present (sticking, normal-dominant). joint_damping is null (OFF).
ContactStep RunContactStep(const nuka::phi::DeviceContext& ctx,
                           const art::ArticulationHostState& host,
                           const std::vector<uint32_t>& link,
                           const std::vector<Vec3>& point,
                           const std::vector<Vec3>& normal,
                           const std::vector<float>& depth, float dt,
                           float friction_coefficient) {
    ContactStep s;
    const uint32_t dof = art::ArticulationDofCount(host, 0u);
    s.dof = dof;
    const uint32_t max_dof = dof;
    const uint32_t env = 1u;
    const uint32_t slot_count = env * kSlots;
    s.slot_count = slot_count;
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;

    s.buffers = art::UploadArticulationState(ctx, host);
    s.view = s.buffers.View();
    art::FeatherstoneAba::ComputeAccelerations(ctx, s.view, 0.0f);

    nuka::phi::Buffer composite(
        static_cast<size_t>(host.TotalLinkCount()) * sizeof(art::LinkSpatialInertia),
        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m_buf(tile * sizeof(float), nuka::phi::MemoryKind::Device);
    s.minv_buf = nuka::phi::Buffer(tile * sizeof(float), nuka::phi::MemoryKind::Device);
    art::ComputeArticulationInertiaM(
        ctx, s.view, max_dof,
        static_cast<art::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m_buf.Data()));
    art::FactorArticulationInertiaM(ctx, s.view, max_dof,
                                    static_cast<const float*>(m_buf.Data()),
                                    static_cast<float*>(s.minv_buf.Data()));

    nuka::phi::Buffer link_buf(slot_count * sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer point_buf(slot_count * sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer normal_buf(slot_count * sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer depth_buf(slot_count * sizeof(float), nuka::phi::MemoryKind::Device);
    link_buf.CopyFromHost(link.data(), slot_count * sizeof(uint32_t));
    point_buf.CopyFromHost(point.data(), slot_count * sizeof(Vec3));
    normal_buf.CopyFromHost(normal.data(), slot_count * sizeof(Vec3));
    depth_buf.CopyFromHost(depth.data(), slot_count * sizeof(float));

    nuka::phi::Buffer t1_buf(slot_count * sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer t2_buf(slot_count * sizeof(Vec3), nuka::phi::MemoryKind::Device);
    art::ComputeContactTangentBasis(ctx,
                                    static_cast<const uint32_t*>(link_buf.Data()),
                                    static_cast<const Vec3*>(normal_buf.Data()), env,
                                    static_cast<Vec3*>(t1_buf.Data()),
                                    static_cast<Vec3*>(t2_buf.Data()));

    const size_t jac_len = static_cast<size_t>(slot_count) * max_dof;
    s.jn_buf = nuka::phi::Buffer(jac_len * sizeof(float), nuka::phi::MemoryKind::Device);
    s.jt1_buf = nuka::phi::Buffer(jac_len * sizeof(float), nuka::phi::MemoryKind::Device);
    s.jt2_buf = nuka::phi::Buffer(jac_len * sizeof(float), nuka::phi::MemoryKind::Device);
    art::ComputeContactChainJacobians(ctx, s.view,
                                      static_cast<const uint32_t*>(link_buf.Data()),
                                      static_cast<const Vec3*>(point_buf.Data()),
                                      static_cast<const Vec3*>(normal_buf.Data()),
                                      slot_count, max_dof,
                                      static_cast<float*>(s.jn_buf.Data()));
    art::ComputeContactChainJacobians(ctx, s.view,
                                      static_cast<const uint32_t*>(link_buf.Data()),
                                      static_cast<const Vec3*>(point_buf.Data()),
                                      static_cast<const Vec3*>(t1_buf.Data()),
                                      slot_count, max_dof,
                                      static_cast<float*>(s.jt1_buf.Data()));
    art::ComputeContactChainJacobians(ctx, s.view,
                                      static_cast<const uint32_t*>(link_buf.Data()),
                                      static_cast<const Vec3*>(point_buf.Data()),
                                      static_cast<const Vec3*>(t2_buf.Data()),
                                      slot_count, max_dof,
                                      static_cast<float*>(s.jt2_buf.Data()));

    nuka::phi::Buffer meff_n(slot_count * sizeof(float), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer meff_t1(slot_count * sizeof(float), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer meff_t2(slot_count * sizeof(float), nuka::phi::MemoryKind::Device);
    art::ComputeContactEffectiveMass(ctx, s.view,
                                     static_cast<const uint32_t*>(link_buf.Data()),
                                     static_cast<const float*>(s.jn_buf.Data()),
                                     static_cast<const float*>(s.minv_buf.Data()),
                                     slot_count, max_dof,
                                     static_cast<float*>(meff_n.Data()));
    art::ComputeContactEffectiveMass(ctx, s.view,
                                     static_cast<const uint32_t*>(link_buf.Data()),
                                     static_cast<const float*>(s.jt1_buf.Data()),
                                     static_cast<const float*>(s.minv_buf.Data()),
                                     slot_count, max_dof,
                                     static_cast<float*>(meff_t1.Data()));
    art::ComputeContactEffectiveMass(ctx, s.view,
                                     static_cast<const uint32_t*>(link_buf.Data()),
                                     static_cast<const float*>(s.jt2_buf.Data()),
                                     static_cast<const float*>(s.minv_buf.Data()),
                                     slot_count, max_dof,
                                     static_cast<float*>(meff_t2.Data()));

    s.rows_buf = nuka::phi::Buffer(slot_count * sizeof(art::ArticulatedContactRow),
                                   nuka::phi::MemoryKind::Device);
    art::AssembleArticulatedContactRows(
        ctx, s.view, static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()),
        static_cast<const float*>(depth_buf.Data()),
        static_cast<const float*>(meff_n.Data()),
        static_cast<const float*>(meff_t1.Data()),
        static_cast<const float*>(meff_t2.Data()), env, max_dof,
        static_cast<art::ArticulatedContactRow*>(s.rows_buf.Data()));

    s.qdot_free = host.qdot;

    nuka::phi::Buffer lambda_buf(slot_count * 3u * sizeof(float),
                                 nuka::phi::MemoryKind::Device);
    std::vector<float> lambda0(slot_count * 3u, 0.0f);
    lambda_buf.CopyFromHost(lambda0.data(), lambda0.size() * sizeof(float));

    art::SolveArticulatedContactRows(
        ctx, s.view,
        static_cast<const art::ArticulatedContactRow*>(s.rows_buf.Data()),
        static_cast<const float*>(s.jn_buf.Data()),
        static_cast<const float*>(s.jt1_buf.Data()),
        static_cast<const float*>(s.jt2_buf.Data()),
        static_cast<const float*>(s.minv_buf.Data()), env, max_dof, dt,
        static_cast<float*>(lambda_buf.Data()), friction_coefficient);
    ctx.stream.Synchronize();

    art::ArticulationHostState dl = host;
    art::DownloadArticulationState(s.buffers, &dl);
    s.qdot_post = dl.qdot;
    s.rows.resize(slot_count);
    s.rows_buf.CopyToHost(s.rows.data(),
                          s.rows.size() * sizeof(art::ArticulatedContactRow));
    s.minv.resize(tile);
    s.minv_buf.CopyToHost(s.minv.data(), s.minv.size() * sizeof(float));
    const size_t jl = jac_len;
    s.jn.resize(jl); s.jt1.resize(jl); s.jt2.resize(jl);
    s.jn_buf.CopyToHost(s.jn.data(), jl * sizeof(float));
    s.jt1_buf.CopyToHost(s.jt1.data(), jl * sizeof(float));
    s.jt2_buf.CopyToHost(s.jt2.data(), jl * sizeof(float));
    return s;
}

// Runs the GPU IFT path on a finished ContactStep with cotangent g (global).
// Returns grad_qdot_free (global) + grad_b_c (block-major stride kMd).
struct IftOut {
    std::vector<float> grad_qdot_free;  // total_link_count
    std::vector<float> grad_b_c;        // block_count * kMd
};
IftOut RunIft(const nuka::phi::DeviceContext& ctx, ContactStep& s,
              const std::vector<float>& g) {
    const uint32_t n_link = s.view.total_link_count;
    nuka::phi::Buffer g_buf(n_link * sizeof(float), nuka::phi::MemoryKind::Device);
    g_buf.CopyFromHost(g.data(), n_link * sizeof(float));
    nuka::phi::Buffer gqf_buf(n_link * sizeof(float), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer gbc_buf(static_cast<size_t>(s.view.articulation_count) * kMd *
                                  sizeof(float),
                              nuka::phi::MemoryKind::Device);

    diffsim::IftContactInputs in;
    in.rows = static_cast<const art::ArticulatedContactRow*>(s.rows_buf.Data());
    in.jac_normal = static_cast<const float*>(s.jn_buf.Data());
    in.jac_tangent1 = static_cast<const float*>(s.jt1_buf.Data());
    in.jac_tangent2 = static_cast<const float*>(s.jt2_buf.Data());
    in.m_inv = static_cast<const float*>(s.minv_buf.Data());
    in.state = s.view;
    in.block_count = s.view.articulation_count;
    in.dof_stride = s.dof;

    diffsim::IftContactGrads grads;
    grads.grad_qdot_free = static_cast<float*>(gqf_buf.Data());
    grads.grad_b_c = static_cast<float*>(gbc_buf.Data());

    diffsim::IftRunner runner(ctx);
    runner.Run(in, static_cast<const float*>(g_buf.Data()), grads);
    ctx.stream.Synchronize();

    IftOut out;
    out.grad_qdot_free.resize(n_link);
    out.grad_b_c.resize(static_cast<size_t>(s.view.articulation_count) * kMd);
    gqf_buf.CopyToHost(out.grad_qdot_free.data(), out.grad_qdot_free.size() * sizeof(float));
    gbc_buf.CopyToHost(out.grad_b_c.data(), out.grad_b_c.size() * sizeof(float));
    return out;
}

// Host active-row gather, EXACTLY the kkt_builder / runner order: slots ascending,
// normal -> t1 -> t2 per engaged slot (engaged := row_type != kRowInactive AND
// depth > 0). Returns the n x dof J matrix (active rows) for one articulation.
Eigen::MatrixXd HostActiveJ(const ContactStep& s) {
    std::vector<Eigen::RowVectorXd> rows;
    for (uint32_t slot = 0u; slot < kSlots; ++slot) {
        const art::ArticulatedContactRow& row = s.rows[slot];
        const bool engaged =
            (row.row_type != art::ContactRowType::kRowInactive) && (row.depth > 0.0f);
        if (!engaged) continue;
        for (uint32_t dir = 0u; dir < 3u; ++dir) {
            const std::vector<float>& jb = (dir == 0u) ? s.jn : (dir == 1u) ? s.jt1 : s.jt2;
            Eigen::RowVectorXd r(s.dof);
            for (uint32_t c = 0u; c < s.dof; ++c)
                r(c) = static_cast<double>(jb[static_cast<size_t>(slot) * s.dof + c]);
            rows.push_back(r);
        }
    }
    Eigen::MatrixXd J(rows.size(), s.dof);
    for (size_t i = 0u; i < rows.size(); ++i) J.row(i) = rows[i];
    return J;
}

Eigen::MatrixXd HostMinv(const ContactStep& s) {
    Eigen::MatrixXd M(s.dof, s.dof);
    for (uint32_t r = 0u; r < s.dof; ++r)
        for (uint32_t c = 0u; c < s.dof; ++c)
            M(r, c) = static_cast<double>(s.minv[static_cast<size_t>(r) * s.dof + c]);
    return M;
}

// A single resting downward contact at the chain tip. ground/depth chosen so the
// contact is well inside penetration (bias well above the slop clamp) and the foot
// is normal-dominant.
struct Scene {
    art::ArticulationHostState host;
    std::vector<uint32_t> link;
    std::vector<Vec3> point, normal;
    std::vector<float> depth;
    std::vector<uint32_t> dof_to_link;
    float dt = 0.01f;
    float mu = art::kContactFriction;
};

Scene MakeChainScene(uint32_t n_dof, float depth_m, float mu,
                     bool nonplanar = false) {
    Scene sc;
    const float seg = 0.5f;
    sc.host = BuildChain(n_dof, seg, /*mass=*/1.5f);
    sc.dof_to_link = DofToLink(sc.host);
    // Tip of the last link in the link frame is (seg,0,0); at q=0 the chain lies
    // along +X so the tip world pos is ((n_dof+1)*seg? ) -- exact value irrelevant
    // to the contact math (J is built from the contact point + link pose). Place a
    // +Z normal contact at the last revolute link.
    const uint32_t last = n_dof - 1u;
    sc.link.assign(kSlots, kInvalidLink);
    sc.point.assign(kSlots, Vec3::Zero());
    sc.normal.assign(kSlots, Vec3::Zero());
    sc.depth.assign(kSlots, 0.0f);
    sc.link[0] = last;
    // Contact point = last link's tip in world (q=0: stacked +X). Use the FK tip.
    float tip_x = 0.0f;
    for (uint32_t i = 0u; i <= last; ++i) tip_x += seg;  // parent offsets
    tip_x += seg;  // mass/tip offset within the last link frame
    // PLANAR (default): all joint axes +Y, offsets +X, normal +Z -> the mechanism
    // lives in the XZ plane, so the foot has zero Y-velocity and the t2 tangent
    // (Y direction) row is identically zero (J_t2 == 0). This is the right shape
    // for the FD test (the 48-iter PGS converges exactly -> a clean gate), but it
    // does NOT exercise the t2 friction column. NONPLANAR offsets the contact point
    // in +Y and tilts the normal, so the foot's Y-velocity couples through the
    // chain (J_t2 != 0) and A is a genuine coupled 3x3 SPD block -- used ONLY by the
    // dense-reference oracle (which does not depend on PGS convergence, so no R3 risk).
    if (nonplanar) {
        sc.point[0] = Vec3{tip_x, 0.30f, 0.0f};  // +Y offset -> foot Y-velocity lives
        Vec3 nrm{0.2f, 0.1f, 1.0f};               // tilted off +Z (still mostly up)
        const float m = std::sqrt(nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z);
        sc.normal[0] = Vec3{nrm.x / m, nrm.y / m, nrm.z / m};  // unit
    } else {
        sc.point[0] = Vec3{tip_x, 0.0f, 0.0f};
        sc.normal[0] = Vec3{0.0f, 0.0f, 1.0f};
    }
    sc.depth[0] = depth_m;
    sc.mu = mu;
    return sc;
}

}  // namespace

// ---------------------------------------------------------------------------
// (A) Dense reference: GPU IFT vs host Eigen::LDLT solve+distribute, < 1e-5.
//     Fed the GPU-built A (downloaded) so a kkt discrepancy can't masquerade.
// ---------------------------------------------------------------------------
TEST(IftContactGradient, MatchesDenseReference) {
    const auto ctx = nuka::phi::MakeDefaultDeviceContext();
    // A 3-DOF chain with a NON-PLANAR contact (+Y point offset + tilted normal) so
    // ALL THREE rows are live: 1 normal + 2 friction (t1 AND t2), and the Delassus
    // is a genuinely coupled 3x3 SPD block (Jn.M^-1.Jt^T != 0, J_t2 != 0). The
    // dense-reference oracle does NOT depend on PGS convergence, so the non-planar
    // (slower-converging) geometry carries no R3 risk here -- it only stresses the
    // full solve+distribute over all three friction columns.
    Scene sc = MakeChainScene(/*n_dof=*/3u, /*depth=*/0.02f,
                              /*mu=*/art::kContactFriction, /*nonplanar=*/true);
    for (auto& v : sc.host.qdot) v = 0.0f;
    sc.host.qdot[sc.dof_to_link[0]] = 0.05f;  // small qdot_free -> active set stable

    ContactStep s = RunContactStep(ctx, sc.host, sc.link, sc.point, sc.normal,
                                   sc.depth, sc.dt, sc.mu);

    // Confirm the scene genuinely exercises the t2 column (else the 3x3 coupling
    // claim is vacuous). Both tangent rows must have non-trivial Jacobians.
    {
        float maxt1 = 0.0f, maxt2 = 0.0f;
        for (uint32_t k = 0u; k < s.dof; ++k) {
            maxt1 = std::max(maxt1, std::fabs(s.jt1[k]));
            maxt2 = std::max(maxt2, std::fabs(s.jt2[k]));
        }
        std::printf("[Ift] nonplanar scene: max|J_t1|=%.4f max|J_t2|=%.4f (both > 0)\n",
                    static_cast<double>(maxt1), static_cast<double>(maxt2));
        ASSERT_GT(maxt1, 1e-3f);
        ASSERT_GT(maxt2, 1e-3f) << "t2 column is dead -> dense-ref does not exercise it";
    }

    // Cotangent g = random fixed weights on qdot_post (global layout).
    std::mt19937 rng(0x1F7u);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> g(s.view.total_link_count, 0.0f);
    for (uint32_t k = 0u; k < s.dof; ++k) g[sc.dof_to_link[k]] = u(rng);

    const IftOut gpu = RunIft(ctx, s, g);

    // Host dense reference. Download A (GPU-built) for the second oracle of the
    // SOLVE itself, but for the distribute reference assemble A from J,M^-1 too.
    const Eigen::MatrixXd J = HostActiveJ(s);
    const Eigen::MatrixXd Minv = HostMinv(s);
    const uint32_t n = static_cast<uint32_t>(J.rows());
    ASSERT_GT(n, 0u);
    const Eigen::MatrixXd A = J * Minv * J.transpose();  // SPD Delassus

    // Cross-check our A against the GPU kkt builder's A (downloaded).
    {
        diffsim::BatchedDenseSpdSystem sys;
        diffsim::ContactDelassusBuffers bufs;
        diffsim::BuildContactDelassusSystem(
            ctx, static_cast<const art::ArticulatedContactRow*>(s.rows_buf.Data()),
            static_cast<const float*>(s.jn_buf.Data()),
            static_cast<const float*>(s.jt1_buf.Data()),
            static_cast<const float*>(s.jt2_buf.Data()),
            static_cast<const float*>(s.minv_buf.Data()), 1u, s.dof, sys, bufs);
        ctx.stream.Synchronize();
        std::vector<float> Adev(kMd * kMd, 0.0f);
        bufs.values.CopyToHost(Adev.data(), Adev.size() * sizeof(float));
        double maxd = 0.0, sca = 1e-12;
        for (uint32_t i = 0u; i < n; ++i)
            for (uint32_t j = 0u; j < n; ++j) sca = std::max(sca, std::abs(A(i, j)));
        for (uint32_t i = 0u; i < n; ++i)
            for (uint32_t j = 0u; j < n; ++j)
                maxd = std::max(maxd, std::abs(static_cast<double>(Adev[i * kMd + j]) - A(i, j)));
        std::printf("[Ift] host-A vs GPU-A max abs diff = %.3e (scale %.3e)\n", maxd, sca);
        EXPECT_LT(maxd, 1e-5 * sca);
    }

    // g in local-dof order.
    Eigen::VectorXd g_local(s.dof);
    for (uint32_t k = 0u; k < s.dof; ++k)
        g_local(k) = static_cast<double>(g[sc.dof_to_link[k]]);

    // z = A^-1 (J M^-1 g);  grad_qdot_free = g - J^T z;  grad_b_c = z.
    const Eigen::VectorXd rhs = J * (Minv * g_local);
    const Eigen::VectorXd z = A.ldlt().solve(rhs);
    const Eigen::VectorXd gqf = g_local - J.transpose() * z;

    double max_qf = 0.0, max_bc = 0.0;
    double sca_qf = 1e-9, sca_bc = 1e-9;
    for (uint32_t k = 0u; k < s.dof; ++k) sca_qf = std::max(sca_qf, std::abs(gqf(k)));
    for (uint32_t r = 0u; r < n; ++r) sca_bc = std::max(sca_bc, std::abs(z(r)));
    for (uint32_t k = 0u; k < s.dof; ++k) {
        const double dev = static_cast<double>(gpu.grad_qdot_free[sc.dof_to_link[k]]);
        max_qf = std::max(max_qf, std::abs(dev - gqf(k)));
    }
    for (uint32_t r = 0u; r < n; ++r) {
        const double dev = static_cast<double>(gpu.grad_b_c[r]);
        max_bc = std::max(max_bc, std::abs(dev - z(r)));
    }
    std::printf("[Ift] dense-ref: max |grad_qdot_free diff| = %.3e (scale %.3e)\n", max_qf, sca_qf);
    std::printf("[Ift] dense-ref: max |grad_b_c diff|       = %.3e (scale %.3e)\n", max_bc, sca_bc);
    EXPECT_LT(max_qf, 1e-5 * sca_qf) << "GPU IFT grad_qdot_free vs host Eigen::LDLT";
    EXPECT_LT(max_bc, 1e-5 * sca_bc) << "GPU IFT grad_b_c vs host Eigen::LDLT";

    // grad_b_c padding (rows >= n) must be exactly 0.
    for (uint32_t r = n; r < kMd; ++r)
        EXPECT_EQ(gpu.grad_b_c[r], 0.0f) << "grad_b_c padding nonzero at row " << r;
}

// ---------------------------------------------------------------------------
// (B) PRIMARY oracle: IFT vs finite-difference through the REAL engine contact
//     step, < 1e-3 relative. Validates qdot_free and b_c (via depth) channels.
// ---------------------------------------------------------------------------
TEST(IftContactGradient, MatchesFiniteDifference) {
    const auto ctx = nuka::phi::MakeDefaultDeviceContext();
    // 2-DOF chain: a coupled Delassus (n=3) but still normal-dominant + fast PGS
    // convergence (light tangential coupling). qdot_free small so the active set /
    // the lambda_n>0 set is stable under the FD perturbation.
    Scene sc = MakeChainScene(/*n_dof=*/2u, /*depth=*/0.02f, /*mu=*/art::kContactFriction);
    for (auto& v : sc.host.qdot) v = 0.0f;
    sc.host.qdot[sc.dof_to_link[0]] = 0.03f;
    sc.host.qdot[sc.dof_to_link[1]] = -0.02f;

    ContactStep s0 = RunContactStep(ctx, sc.host, sc.link, sc.point, sc.normal,
                                    sc.depth, sc.dt, sc.mu);

    // Near-convergence evidence: the post-solve normal residual |J_n qdot_post - bias|
    // must be tiny (the IFT linearizes the CONVERGED fixed point; the forward runs a
    // FIXED 48 iters). Also report the tangential residual (sticking -> ~0).
    {
        const float bias = art::kBaumgarteBeta / sc.dt *
                           std::fmax(sc.depth[0] - art::kPenetrationSlop, 0.0f);
        float jv_n = 0.0f, jv_t1 = 0.0f, jv_t2 = 0.0f;
        for (uint32_t k = 0u; k < s0.dof; ++k) {
            const float qd = s0.qdot_post[sc.dof_to_link[k]];
            jv_n += s0.jn[k] * qd;
            jv_t1 += s0.jt1[k] * qd;
            jv_t2 += s0.jt2[k] * qd;
        }
        std::printf("[Ift] near-converged: J_n qdot_post=%.6e bias=%.6e |residual|=%.3e\n",
                    static_cast<double>(jv_n), static_cast<double>(bias),
                    static_cast<double>(std::fabs(jv_n - bias)));
        std::printf("[Ift] tangential residual J_t1=%.3e J_t2=%.3e (sticking -> ~0)\n",
                    static_cast<double>(jv_t1), static_cast<double>(jv_t2));
        EXPECT_LT(std::fabs(jv_n - bias), 1e-4f)
            << "48-iter PGS not converged on the test scene -> FD gate invalid";
    }

    // Scalar loss L = sum_k w_k * qdot_post[dof_to_link[k]]  =>  g = dL/dqdot_post = w.
    std::mt19937 rng(0xABCDu);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> g(s0.view.total_link_count, 0.0f);
    std::vector<float> w(s0.dof);
    for (uint32_t k = 0u; k < s0.dof; ++k) { w[k] = u(rng); g[sc.dof_to_link[k]] = w[k]; }

    const IftOut ift = RunIft(ctx, s0, g);

    auto loss = [&](const ContactStep& s) {
        double L = 0.0;
        for (uint32_t k = 0u; k < s.dof; ++k)
            L += static_cast<double>(w[k]) * static_cast<double>(s.qdot_post[sc.dof_to_link[k]]);
        return L;
    };

    // -- qdot_free channel: central FD on each pre-solve qdot component. --
    const float eps = 1e-4f;
    double max_rel_qf = 0.0;
    for (uint32_t k = 0u; k < s0.dof; ++k) {
        art::ArticulationHostState hp = sc.host;
        art::ArticulationHostState hm = sc.host;
        hp.qdot[sc.dof_to_link[k]] += eps;
        hm.qdot[sc.dof_to_link[k]] -= eps;
        ContactStep sp = RunContactStep(ctx, hp, sc.link, sc.point, sc.normal,
                                        sc.depth, sc.dt, sc.mu);
        ContactStep sm = RunContactStep(ctx, hm, sc.link, sc.point, sc.normal,
                                        sc.depth, sc.dt, sc.mu);
        const double fd = (loss(sp) - loss(sm)) / (2.0 * static_cast<double>(eps));
        const double an = static_cast<double>(ift.grad_qdot_free[sc.dof_to_link[k]]);
        const double rel = std::abs(fd - an) / std::max(std::abs(fd), 1e-6);
        max_rel_qf = std::max(max_rel_qf, rel);
        std::printf("[Ift] FD qdot_free[%u]: fd=%.6e ift=%.6e rel=%.3e\n", k, fd, an, rel);
    }
    EXPECT_LT(max_rel_qf, 1e-3) << "IFT grad_qdot_free vs FD through the real contact step";

    // -- b_c channel (normal row only): FD via depth. The normal-row target is
    //    b_c = kBaumgarteBeta/dt * max(depth - slop, 0), so d b_c / d depth =
    //    kBaumgarteBeta/dt (away from the slop clamp). dL/d depth = grad_b_c[normal]
    //    * (kBaumgarteBeta/dt). The normal row is row 0 of the (single) engaged slot
    //    in the kkt order. Friction b_c=0 is not reachable via depth -- the dense
    //    reference covers the full b_c vector. --
    {
        const float chain = art::kBaumgarteBeta / sc.dt;  // d b_c_normal / d depth
        std::vector<float> dp = sc.depth, dm = sc.depth;
        dp[0] += eps; dm[0] -= eps;
        ContactStep sp = RunContactStep(ctx, sc.host, sc.link, sc.point, sc.normal,
                                        dp, sc.dt, sc.mu);
        ContactStep sm = RunContactStep(ctx, sc.host, sc.link, sc.point, sc.normal,
                                        dm, sc.dt, sc.mu);
        const double fd_depth = (loss(sp) - loss(sm)) / (2.0 * static_cast<double>(eps));
        const double an_depth = static_cast<double>(ift.grad_b_c[0]) *
                                static_cast<double>(chain);  // row 0 = normal
        const double rel = std::abs(fd_depth - an_depth) /
                           std::max(std::abs(fd_depth), 1e-6);
        std::printf("[Ift] FD b_c(depth): fd=%.6e ift*chain=%.6e rel=%.3e (chain=%.3f)\n",
                    fd_depth, an_depth, rel, static_cast<double>(chain));
        EXPECT_LT(rel, 1e-3) << "IFT grad_b_c (normal row) vs FD through depth";
    }
}

// ---------------------------------------------------------------------------
// (C) D1: two IFT backward runs -> byte-identical grad buffers (R6 extends to the
//     backward/solver outputs).
// ---------------------------------------------------------------------------
TEST(IftContactGradient, D1TwoRunByteExact) {
    const auto ctx = nuka::phi::MakeDefaultDeviceContext();
    Scene sc = MakeChainScene(/*n_dof=*/3u, /*depth=*/0.02f, /*mu=*/art::kContactFriction);
    for (auto& v : sc.host.qdot) v = 0.0f;
    sc.host.qdot[sc.dof_to_link[0]] = 0.04f;
    sc.host.qdot[sc.dof_to_link[1]] = -0.01f;

    ContactStep s = RunContactStep(ctx, sc.host, sc.link, sc.point, sc.normal,
                                   sc.depth, sc.dt, sc.mu);
    std::mt19937 rng(0x1234u);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> g(s.view.total_link_count, 0.0f);
    for (uint32_t k = 0u; k < s.dof; ++k) g[sc.dof_to_link[k]] = u(rng);

    const IftOut a = RunIft(ctx, s, g);
    const IftOut b = RunIft(ctx, s, g);
    ASSERT_EQ(a.grad_qdot_free.size(), b.grad_qdot_free.size());
    ASSERT_EQ(a.grad_b_c.size(), b.grad_b_c.size());
    const int cmp_qf = std::memcmp(a.grad_qdot_free.data(), b.grad_qdot_free.data(),
                                   a.grad_qdot_free.size() * sizeof(float));
    const int cmp_bc = std::memcmp(a.grad_b_c.data(), b.grad_b_c.data(),
                                   a.grad_b_c.size() * sizeof(float));
    EXPECT_EQ(cmp_qf, 0) << "D1 violated: grad_qdot_free differs across two IFT runs";
    EXPECT_EQ(cmp_bc, 0) << "D1 violated: grad_b_c differs across two IFT runs";
    std::printf("[Ift] D1 two-run byte-identical (grad_qdot_free + grad_b_c)\n");
}

// ---------------------------------------------------------------------------
// (D) Seam: the gated BackwardRunner entry (default ContactFree throws; ContactIft
//     routes to IftRunner and matches the direct IftRunner output).
// ---------------------------------------------------------------------------
TEST(IftContactGradient, BackwardRunnerSeamGate) {
    const auto ctx = nuka::phi::MakeDefaultDeviceContext();
    Scene sc = MakeChainScene(/*n_dof=*/2u, /*depth=*/0.02f, /*mu=*/art::kContactFriction);
    for (auto& v : sc.host.qdot) v = 0.0f;
    sc.host.qdot[sc.dof_to_link[0]] = 0.02f;

    ContactStep s = RunContactStep(ctx, sc.host, sc.link, sc.point, sc.normal,
                                   sc.depth, sc.dt, sc.mu);
    std::vector<float> g(s.view.total_link_count, 0.0f);
    for (uint32_t k = 0u; k < s.dof; ++k) g[sc.dof_to_link[k]] = 0.5f - 0.1f * k;

    // Direct IftRunner reference.
    const IftOut direct = RunIft(ctx, s, g);

    // The seam: a default (ContactFree) BackwardRunner must REJECT the IFT entry;
    // after SetGradientMode(ContactIft) it routes to IftRunner with identical math.
    // BackwardRunner needs a RecomputeOrchestrator; we only exercise the standalone
    // seam method (ApplyIftContactSeed), which does not touch the orchestrator's
    // tape -- so a minimal orchestrator over the same world suffices (zeroed drive
    // gains / mass params; never stepped).
    const uint32_t n_links = s.view.total_link_count;
    const std::vector<float> zeros(n_links, 0.0f);
    std::vector<diffsim::LinkMassParams> mass(n_links);
    diffsim::RecomputeOrchestrator orch(ctx, s.view, diffsim::RolloutParams{},
                                        zeros, zeros, zeros, mass);
    diffsim::BackwardRunner runner(ctx, orch);
    EXPECT_EQ(runner.gradient_mode(), diffsim::GradientMode::ContactFree);

    const uint32_t n_link = s.view.total_link_count;
    nuka::phi::Buffer g_buf(n_link * sizeof(float), nuka::phi::MemoryKind::Device);
    g_buf.CopyFromHost(g.data(), n_link * sizeof(float));
    nuka::phi::Buffer gqf_buf(n_link * sizeof(float), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer gbc_buf(static_cast<size_t>(s.view.articulation_count) * kMd *
                                  sizeof(float),
                              nuka::phi::MemoryKind::Device);

    diffsim::IftContactInputs in;
    in.rows = static_cast<const art::ArticulatedContactRow*>(s.rows_buf.Data());
    in.jac_normal = static_cast<const float*>(s.jn_buf.Data());
    in.jac_tangent1 = static_cast<const float*>(s.jt1_buf.Data());
    in.jac_tangent2 = static_cast<const float*>(s.jt2_buf.Data());
    in.m_inv = static_cast<const float*>(s.minv_buf.Data());
    in.state = s.view;
    in.block_count = s.view.articulation_count;
    in.dof_stride = s.dof;

    // Default mode -> rejected.
    EXPECT_THROW(runner.ApplyIftContactSeed(in, static_cast<const float*>(g_buf.Data()),
                                            static_cast<float*>(gqf_buf.Data()),
                                            static_cast<float*>(gbc_buf.Data())),
                 std::runtime_error);

    // Opt in -> routes to IFT.
    runner.SetGradientMode(diffsim::GradientMode::ContactIft);
    runner.ApplyIftContactSeed(in, static_cast<const float*>(g_buf.Data()),
                               static_cast<float*>(gqf_buf.Data()),
                               static_cast<float*>(gbc_buf.Data()));
    ctx.stream.Synchronize();

    std::vector<float> seam_qf(n_link, 0.0f);
    gqf_buf.CopyToHost(seam_qf.data(), seam_qf.size() * sizeof(float));
    for (uint32_t i = 0u; i < n_link; ++i)
        EXPECT_FLOAT_EQ(seam_qf[i], direct.grad_qdot_free[i])
            << "seam grad_qdot_free != direct IftRunner at " << i;
    std::printf("[Ift] seam: default rejects, ContactIft matches direct IftRunner\n");
}
