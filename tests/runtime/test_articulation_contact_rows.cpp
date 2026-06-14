// ---------------------------------------------------------------------------
// p01-F T5 -- articulated contact rows + block-per-articulation PGS solve
//
// Validates the contact<->ABA coupling added to articulation_contacts.{cu,hpp}:
//
//  AssembleArticulatedContactRows  -- per active contact: chain-Jacobian row,
//                                     m_eff, normal, depth, tangent basis.
//  SolveArticulatedContactRows     -- fused block-per-articulation PGS that
//                                     solves the rows and applies the joint-space
//                                     impulse Delta_qdot = M^-1 J^T lambda back
//                                     into state.qdot.
//
// There is NO contacts-on oracle, and "4096 envs identical" only proves
// replication+determinism (a uniform wiring bug passes). So every gate measures
// something the wiring cannot fake:
//
//  (1) PRIMARY magnitude anchor -- a hand-computable 1-DOF revolute pendulum.
//      M = m L^2, J = -L, m_eff = m, so lambda = m*bias, Delta_qdot = -bias/L.
//      Asserted to tight tolerance. Plus a separating-velocity sub-case that
//      forces the clamp (lambda == 0) -- the cheapest detector of a bias-sign
//      bug (lambda = m_eff*(bias - jv), row_solver convention).
//  (2) Independent FK normal-velocity check on Go2 -- recompute the contact
//      point's relative normal velocity from the link's spatial velocity (a
//      fresh ABA Pass-1 on the solved qdot+), NOT through the assembled J row.
//      Asserts it equals the Baumgarte target (residual ~ 0).
//  (3) Block-diagonal structural check -- inject ONE contact, solve, assert the
//      impulse lands ONLY on the contact leg's ancestor DOFs (derived from an
//      independent parent_link walk) and the other DOFs are bit-identical. This
//      catches a dof-column->global-qdot permutation, THE risk of this task.
//  (4) Anti-vacuous-pass guard (lambda_n > 0, sustained contact) + stability
//      (bounded penetration/energy over K steps) + determinism (bit-identical
//      across two runs and 64 replicas, cold-started).
//  (5) Friction cone (|lambda_t| <= mu*lambda_n) + lateral-push tangential
//      damping.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
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
#include "runtime/world_builder.hpp"
#include "scene/cooked_blob.hpp"
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

// ===========================================================================
// Shared device-buffer plumbing for the T5 pipeline.
// ===========================================================================

// Holds the device buffers + view for one uploaded host state; rebuilt whenever
// the host state changes so the view always matches the buffers.
struct DeviceArticulation {
    articulation::ArticulationDeviceBuffers buffers;
    articulation::ArticulationDeviceState view;
};

DeviceArticulation Upload(cudaStream_t context, int context_dev,
                          const articulation::ArticulationHostState& host) {
    DeviceArticulation world;
    world.buffers = articulation::UploadArticulationState(context, context_dev, host);
    world.view = world.buffers.View();
    return world;
}

// Result of one full T5 contact-coupling step on a host state.
struct SolveResult {
    std::vector<float> qdot_free;   // qdot after velocity-integration, pre-solve.
    std::vector<float> qdot_post;   // qdot after the contact solve.
    std::vector<float> lambda;      // [slot*3] = {normal, t1, t2}.
    std::vector<articulation::ArticulatedContactRow> rows;
    std::vector<uint32_t> contact_link;
    std::vector<float> contact_depth;
    std::vector<Vec3> contact_normal;
    std::vector<float> minv;        // per-articulation M^-1 tiles.
    std::vector<float> normal_jac;  // per-slot normal chain Jacobian.
    std::vector<float> tangent1_jac;  // per-slot tangent-1 chain Jacobian.
    std::vector<float> tangent2_jac;  // per-slot tangent-2 chain Jacobian.
    uint32_t slot_count = 0u;
};

// Runs the full T5 pipeline against pre-detected contacts:
//   ComputeAccelerations -> M -> Factor -> tangent basis ->
//   J(normal/t1/t2) -> m_eff(normal/t1/t2) -> assemble -> solve(+apply).
// `host.qdot` is treated as qdot_free (the test seeds it directly; no integration
// step needed -- the kernel reads qdot as the free velocity and writes back the
// corrected one). Contacts are supplied directly (env-major, fixed stride) so a
// gate can inject exactly one. `dt` drives the Baumgarte bias.
SolveResult RunSolve(cudaStream_t context, int context_dev,
                     const articulation::ArticulationHostState& host,
                     uint32_t env_count,
                     uint32_t max_dof,
                     const std::vector<uint32_t>& contact_link,
                     const std::vector<Vec3>& contact_point,
                     const std::vector<Vec3>& contact_normal,
                     const std::vector<float>& contact_depth,
                     float gravity_z,
                     float dt,
                     bool warm_start,
                     float friction_coefficient = articulation::kContactFriction) {
    SolveResult result;
    const uint32_t artic_count = host.ArticulationCount();
    const uint32_t slot_count = env_count * articulation::kMaxFootContactsPerEnv;
    result.slot_count = slot_count;
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;

    auto world = Upload(context, context_dev, host);
    articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, world.view, gravity_z);

    // M and M^-1.
    OwnedDeviceBuffer composite_buf(
        static_cast<size_t>(host.TotalLinkCount()) *
            sizeof(articulation::LinkSpatialInertia));
    OwnedDeviceBuffer m_buf(static_cast<size_t>(artic_count) * tile * sizeof(float));
    OwnedDeviceBuffer minv_buf(static_cast<size_t>(artic_count) * tile * sizeof(float));
    articulation::ComputeArticulationInertiaM(
        context, context_dev, world.view, max_dof,
        static_cast<articulation::LinkSpatialInertia*>(composite_buf.Data()),
        static_cast<float*>(m_buf.Data()));
    articulation::FactorArticulationInertiaM(
        context, context_dev, world.view, max_dof,
        static_cast<const float*>(m_buf.Data()),
        static_cast<float*>(minv_buf.Data()));

    // Contact-slot device buffers.
    OwnedDeviceBuffer link_buf(slot_count * sizeof(uint32_t));
    OwnedDeviceBuffer point_buf(slot_count * sizeof(Vec3));
    OwnedDeviceBuffer normal_buf(slot_count * sizeof(Vec3));
    OwnedDeviceBuffer depth_buf(slot_count * sizeof(float));
    link_buf.CopyFromHost(contact_link.data(), slot_count * sizeof(uint32_t));
    point_buf.CopyFromHost(contact_point.data(), slot_count * sizeof(Vec3));
    normal_buf.CopyFromHost(contact_normal.data(), slot_count * sizeof(Vec3));
    depth_buf.CopyFromHost(contact_depth.data(), slot_count * sizeof(float));

    // Tangent basis per slot.
    OwnedDeviceBuffer t1_buf(slot_count * sizeof(Vec3));
    OwnedDeviceBuffer t2_buf(slot_count * sizeof(Vec3));
    articulation::ComputeContactTangentBasis(
        context, context_dev,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()),
        env_count,
        static_cast<Vec3*>(t1_buf.Data()),
        static_cast<Vec3*>(t2_buf.Data()));

    // Chain Jacobians for the normal and the two tangent directions (reuse the
    // proven J builder, feeding the tangent as the "normal").
    const size_t jac_len = static_cast<size_t>(slot_count) * max_dof;
    OwnedDeviceBuffer jn_buf(jac_len * sizeof(float));
    OwnedDeviceBuffer jt1_buf(jac_len * sizeof(float));
    OwnedDeviceBuffer jt2_buf(jac_len * sizeof(float));
    articulation::ComputeContactChainJacobians(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()),
        slot_count, max_dof, static_cast<float*>(jn_buf.Data()));
    articulation::ComputeContactChainJacobians(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(t1_buf.Data()),
        slot_count, max_dof, static_cast<float*>(jt1_buf.Data()));
    articulation::ComputeContactChainJacobians(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(t2_buf.Data()),
        slot_count, max_dof, static_cast<float*>(jt2_buf.Data()));

    // Effective masses for the three row directions.
    OwnedDeviceBuffer meff_n_buf(slot_count * sizeof(float));
    OwnedDeviceBuffer meff_t1_buf(slot_count * sizeof(float));
    OwnedDeviceBuffer meff_t2_buf(slot_count * sizeof(float));
    articulation::ComputeContactEffectiveMass(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const float*>(jn_buf.Data()),
        static_cast<const float*>(minv_buf.Data()),
        slot_count, max_dof, static_cast<float*>(meff_n_buf.Data()));
    articulation::ComputeContactEffectiveMass(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const float*>(jt1_buf.Data()),
        static_cast<const float*>(minv_buf.Data()),
        slot_count, max_dof, static_cast<float*>(meff_t1_buf.Data()));
    articulation::ComputeContactEffectiveMass(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const float*>(jt2_buf.Data()),
        static_cast<const float*>(minv_buf.Data()),
        slot_count, max_dof, static_cast<float*>(meff_t2_buf.Data()));

    // Assemble rows.
    OwnedDeviceBuffer rows_buf(
        slot_count * sizeof(articulation::ArticulatedContactRow));
    articulation::AssembleArticulatedContactRows(
        context, context_dev, world.view,
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()),
        static_cast<const float*>(depth_buf.Data()),
        static_cast<const float*>(meff_n_buf.Data()),
        static_cast<const float*>(meff_t1_buf.Data()),
        static_cast<const float*>(meff_t2_buf.Data()),
        env_count, max_dof,
        static_cast<articulation::ArticulatedContactRow*>(rows_buf.Data()));

    // Snapshot qdot_free (== current state.qdot, pre-solve) from the host input.
    result.qdot_free = host.qdot;

    // Lambda buffer. Always zeroed before the solve, so reading it as a seed is a
    // cold start regardless of `warm_start`; passing it lets us READ BACK the
    // converged impulses (the kernel writes lambda only when the buffer is given).
    // `warm_start` here exercises the persisted-buffer code path; for a freshly
    // zeroed buffer the seed is 0 either way.
    OwnedDeviceBuffer lambda_buf(slot_count * 3u * sizeof(float));
    std::vector<float> lambda_init(slot_count * 3u, 0.0f);
    lambda_buf.CopyFromHost(lambda_init.data(), lambda_init.size() * sizeof(float));
    (void)warm_start;

    // Solve (fused apply: modifies state.qdot in place; writes lambda for read-back).
    articulation::SolveArticulatedContactRows(
        context, context_dev, world.view,
        static_cast<const articulation::ArticulatedContactRow*>(rows_buf.Data()),
        static_cast<const float*>(jn_buf.Data()),
        static_cast<const float*>(jt1_buf.Data()),
        static_cast<const float*>(jt2_buf.Data()),
        static_cast<const float*>(minv_buf.Data()),
        env_count, max_dof, dt,
        static_cast<float*>(lambda_buf.Data()),
        friction_coefficient);
    cudaStreamSynchronize(context);

    // Download outputs.
    articulation::ArticulationHostState download = host;
    articulation::DownloadArticulationState(world.buffers, &download);
    result.qdot_post = download.qdot;

    result.lambda.resize(slot_count * 3u);
    lambda_buf.CopyToHost(result.lambda.data(), result.lambda.size() * sizeof(float));
    result.rows.resize(slot_count);
    rows_buf.CopyToHost(result.rows.data(),
                        result.rows.size() * sizeof(articulation::ArticulatedContactRow));
    result.contact_link = contact_link;
    result.contact_depth = contact_depth;
    result.contact_normal = contact_normal;
    result.minv.resize(static_cast<size_t>(artic_count) * tile);
    minv_buf.CopyToHost(result.minv.data(), result.minv.size() * sizeof(float));
    result.normal_jac.resize(jac_len);
    jn_buf.CopyToHost(result.normal_jac.data(), result.normal_jac.size() * sizeof(float));
    result.tangent1_jac.resize(jac_len);
    jt1_buf.CopyToHost(result.tangent1_jac.data(), result.tangent1_jac.size() * sizeof(float));
    result.tangent2_jac.resize(jac_len);
    jt2_buf.CopyToHost(result.tangent2_jac.data(), result.tangent2_jac.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Gate 1: a single-link 1-DOF revolute pendulum with a hand-computable answer.
// ===========================================================================
//
// One revolute root link, axis = +Y through the origin (the pin = fixed base).
// A point mass m sits at the tip (L,0,0). A vertical (+Z) contact at the tip:
//   J     = dot(cross(Yhat, (L,0,0)), Zhat) = dot((0,0,-L),(0,0,1)) = -L
//   M     = S^T Ic S = m L^2   (rotational inertia about Y of a mass at (L,0,0))
//   m_eff = 1 / (J M^-1 J^T) = 1 / (L^2/(m L^2)) = m
// With qdot_free = 0 and one unclamped normal row at one iteration:
//   lambda    = m_eff*(bias - jv) = m*bias   (jv = J*0 = 0)
//   Delta_qdot= M^-1 J^T lambda = (1/(m L^2))(-L)(m bias) = -bias / L
//   J*qdot+   = (-L)(-bias/L) = bias  (residual hits the Baumgarte target)
articulation::ArticulationHostState BuildPendulum(float mass, float length) {
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u};
    topo.parent_links = {kInvalidLink};  // root: parent = sentinel.
    topo.joint_types = {articulation::ArticulationJointType::Revolute};
    topo.joint_axes = {Vec3{0.0f, 1.0f, 0.0f}};      // revolute about +Y.
    topo.parent_frames = {Transform::Identity()};    // joint origin at world origin.
    topo.child_frames = {Transform::Identity()};
    topo.local_poses = {Transform::Identity()};      // link frame == joint frame.
    topo.inertial_frames = {Transform({length, 0.0f, 0.0f}, Quat::Identity())};
    topo.initial_positions = {0.0f};
    topo.joint_dampings = {0.0f};
    topo.joint_armatures = {0.0f};                   // no armature -> M = m L^2 exactly.
    topo.masses = {mass};
    topo.inertias = {Vec3{0.0f, 0.0f, 0.0f}};        // pure point mass at the COM.

    // A minimal CookedBodyTable; BuildArticulationHostState prefers the topology
    // mass/inertia/inertial_frame fields, so the body table only needs a pose.
    nuka::scene::CookedBodyTable bodies;
    bodies.poses = {Transform::Identity()};
    bodies.local_poses = {Transform::Identity()};
    bodies.inertial_frames = {Transform::Identity()};
    bodies.masses = {mass};
    bodies.inertias = {Vec3{0.0f, 0.0f, 0.0f}};
    bodies.inv_masses = {mass > 0.0f ? 1.0f / mass : 0.0f};
    bodies.inv_inertias = {Vec3{0.0f, 0.0f, 0.0f}};
    bodies.is_static = {0u};

    return articulation::BuildArticulationHostState({topo}, bodies);
}

// ===========================================================================
// Go2 cooking + foot-shape resolution (mirrors the T2/T4 tests).
// ===========================================================================

struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;  // base-relative
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
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == kInvalidLink) {
            continue;
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

// Maps articulation-local DOF index -> global link (the non-fixed joint at that
// column). Used to set per-DOF qdot and to read back per-DOF Delta_qdot.
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

// Independent FK contact-point velocity along the normal, recomputed from a link's
// spatial velocity (NOT through the assembled J). link_velocity is body-frame
// spatial [omega; v_O]; world contact-point velocity:
//   omega_w = R*omega; vO_w = R*v_O; v_w = vO_w + omega_w x (p - o); dot(v_w, n).
float ContactNormalVelocityFK(const Transform& link_world,
                              const articulation::LinkSpatialVel& vel,
                              Vec3 contact_point,
                              Vec3 normal) {
    const Vec3 omega_body{vel.v[0], vel.v[1], vel.v[2]};
    const Vec3 vo_body{vel.v[3], vel.v[4], vel.v[5]};
    const Vec3 omega_w = link_world.rotation.Rotate(omega_body);
    const Vec3 vo_w = link_world.rotation.Rotate(vo_body);
    const Vec3 lever = contact_point - link_world.position;
    const Vec3 v_w = vo_w + omega_w.Cross(lever);
    return v_w.Dot(normal);
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1 -- PRIMARY hand-computable magnitude anchor.
// ---------------------------------------------------------------------------
TEST(ArticulationContactRows, PendulumHandComputedLambdaAndImpulse) {
    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    const float kMass = 2.0f;
    const float kLength = 0.5f;
    const float kDt = 0.01f;
    const float kDepth = 0.02f;  // penetration depth (m).

    auto host = BuildPendulum(kMass, kLength);
    ASSERT_EQ(host.ArticulationCount(), 1u);
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(dof, 1u);
    const uint32_t max_dof = dof;

    // The contact is at the tip (L,0,0) with a +Z normal. The pendulum link frame
    // is the joint frame (identity local pose), so at q=0 the tip is at (L,0,0).
    const Vec3 contact_point{kLength, 0.0f, 0.0f};
    const Vec3 normal{0.0f, 0.0f, 1.0f};

    std::vector<uint32_t> link(articulation::kMaxFootContactsPerEnv, kInvalidLink);
    std::vector<Vec3> point(articulation::kMaxFootContactsPerEnv, Vec3::Zero());
    std::vector<Vec3> norm(articulation::kMaxFootContactsPerEnv, Vec3::Zero());
    std::vector<float> depth(articulation::kMaxFootContactsPerEnv, 0.0f);
    link[0] = 0u;  // the single revolute link.
    point[0] = contact_point;
    norm[0] = normal;
    depth[0] = kDepth;

    // -- (1a) qdot_free = 0: a clean push-out. ------------------------------
    {
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        const auto res = RunSolve(context, context_dev, host, 1u, max_dof, link, point, norm,
                                  depth, 0.0f, kDt, /*warm_start=*/false);

        // Hand values.
        const float bias =
            articulation::kBaumgarteBeta / kDt *
            std::fmax(kDepth - articulation::kPenetrationSlop, 0.0f);
        const float expect_lambda = kMass * bias;
        const float expect_dqdot = -bias / kLength;

        const float got_lambda = res.lambda[0];   // slot 0, normal.
        const float got_dqdot = res.qdot_post[0] - res.qdot_free[0];

        std::printf("[diag] (1a) M=%.6f J=%.6f m_eff=%.6f bias=%.6f\n",
                    static_cast<double>(res.minv.empty() ? 0.0f : 1.0f / res.minv[0]),
                    static_cast<double>(res.normal_jac[0]),
                    static_cast<double>(res.rows[0].effective_mass),
                    static_cast<double>(bias));
        std::printf("[diag] (1a) lambda: solver=%.6f hand=%.6f | dqdot: solver=%.6f hand=%.6f\n",
                    static_cast<double>(got_lambda), static_cast<double>(expect_lambda),
                    static_cast<double>(got_dqdot), static_cast<double>(expect_dqdot));

        // Independent geometry checks on the assembled row.
        EXPECT_NEAR(res.normal_jac[0], -kLength, 1.0e-4f) << "chain J != -L";
        EXPECT_NEAR(res.rows[0].effective_mass, kMass, 1.0e-3f * kMass)
            << "m_eff != m";  // M carries +1e-6 epsilon, use relative tol.

        // Primary magnitude asserts.
        EXPECT_GT(got_lambda, 0.0f) << "normal impulse must push out (lambda > 0)";
        EXPECT_NEAR(got_lambda, expect_lambda, 1.0e-3f * expect_lambda);
        EXPECT_NEAR(got_dqdot, expect_dqdot, 1.0e-3f * std::fabs(expect_dqdot));

        // Post-solve residual must hit the Baumgarte target: J*qdot+ ~ bias.
        const float jv_post = res.normal_jac[0] * res.qdot_post[0];
        std::printf("[diag] (1a) post residual J*qdot+ = %.6f (target bias = %.6f)\n",
                    static_cast<double>(jv_post), static_cast<double>(bias));
        EXPECT_NEAR(jv_post, bias, 1.0e-3f * bias);
    }

    // -- (1b) Separating velocity forces the clamp (lambda == 0). -----------
    // This is the real test of the bias sign. Drive the tip OUT of the ground
    // (qdot such that jv = J*qdot >> bias). Then lambda = m*(bias - jv) < 0 ->
    // clamp to 0 -> no impulse -> qdot unchanged. A '-bias' sign bug would still
    // produce a plausible nonzero lambda in (1a) but a wrong lambda > 0 here.
    {
        // J = -L, normal +Z. A negative qdot rotates the tip upward (+Z), giving
        // a large positive separating velocity jv = J*qdot = (-L)*qdot.
        const float kQdotFree = -5.0f;  // jv = (-0.5)*(-5) = 2.5 m/s separating.
        for (auto& v : host.qdot) {
            v = kQdotFree;
        }
        const auto res = RunSolve(context, context_dev, host, 1u, max_dof, link, point, norm,
                                  depth, 0.0f, kDt, /*warm_start=*/false);
        const float jv_free = res.normal_jac[0] * kQdotFree;
        const float bias =
            articulation::kBaumgarteBeta / kDt *
            std::fmax(kDepth - articulation::kPenetrationSlop, 0.0f);
        std::printf("[diag] (1b) separating jv_free=%.6f bias=%.6f -> lambda=%.6f (expect 0)\n",
                    static_cast<double>(jv_free), static_cast<double>(bias),
                    static_cast<double>(res.lambda[0]));
        ASSERT_GT(jv_free, bias) << "test setup: foot must separate faster than bias";
        EXPECT_EQ(res.lambda[0], 0.0f) << "separating contact must clamp lambda to 0";
        EXPECT_FLOAT_EQ(res.qdot_post[0], kQdotFree)
            << "clamped (zero) impulse must leave qdot unchanged";
    }
}

// ---------------------------------------------------------------------------
// Gates 2-5 on the Go2 require the cooked scene.
// ---------------------------------------------------------------------------
TEST(ArticulationContactRows, Go2CouplingFkResidualStructureDeterminismFriction) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t base_link_count = base.TotalLinkCount();
    ASSERT_GT(base_link_count, 0u);
    ASSERT_EQ(base.ArticulationCount(), 1u);
    ASSERT_EQ(cooked.feet.size(), 4u);

    const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(dof, 0u);
    const uint32_t max_dof = dof;
    const float kDt = 0.005f;
    const float kGravity = 9.81f;

    // Topology dump: link -> {joint_type, parent_local, dof_index}.
    {
        const uint32_t offset = base.articulation_link_offset[0];
        const uint32_t count = base.articulation_link_count[0];
        std::printf("[diag] (topo) link_count=%u dof=%u\n", count, dof);
        for (uint32_t l = 0u; l < count; ++l) {
            const uint32_t link = offset + l;
            uint32_t didx = 0u;
            for (uint32_t k = offset; k < link; ++k) {
                didx += articulation::ArticulationJointDofCount(base.joint_type[k]);
            }
            std::printf("[diag] (topo) link %u type=%u parent_local=%d dof_index=%u\n",
                        link, static_cast<uint32_t>(base.joint_type[link]),
                        static_cast<int>(base.parent_link[link]), didx);
        }
        for (const auto& foot : cooked.feet) {
            std::printf("[diag] (topo) foot calf_local_link=%u\n", foot.calf_local_link);
        }
        std::printf("[diag] (topo) cooked stance base.q = [");
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            std::printf("%.4f ", static_cast<double>(base.q[l]));
        }
        std::printf("]\n");
    }

    // Helper: refresh link_pose with the FK-from-q world poses (so the chain
    // Jacobian, which reads state.link_pose, is consistent with M). Mirrors the
    // inertia_m test (lines 429-432).
    auto refresh_link_pose = [&](articulation::ArticulationHostState& host) {
        auto world = Upload(context, context_dev, host);
        articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, world.view, kGravity);
        OwnedDeviceBuffer pose_buf(
            static_cast<size_t>(host.TotalLinkCount()) * sizeof(Transform));
        articulation::UpdateWorldLinkPoses(
            context, context_dev, world.view, static_cast<Transform*>(pose_buf.Data()));
        cudaStreamSynchronize(context);
        std::vector<Transform> world_pose(host.TotalLinkCount());
        pose_buf.CopyToHost(world_pose.data(), world_pose.size() * sizeof(Transform));
        for (uint32_t link = 0u; link < host.TotalLinkCount(); ++link) {
            host.link_pose[link] = world_pose[link];
        }
        return world_pose;
    };

    // Detects foot contacts against a ground plane and returns the env-major
    // fixed-stride slot arrays for `env_count` (replicated) environments.
    struct Contacts {
        std::vector<uint32_t> link;
        std::vector<Vec3> point;
        std::vector<Vec3> normal;
        std::vector<float> depth;
    };
    auto detect = [&](const articulation::ArticulationHostState& host,
                      uint32_t env_count, float ground_height) {
        auto world = Upload(context, context_dev, host);
        const uint32_t links = host.TotalLinkCount();
        OwnedDeviceBuffer pose_buf(static_cast<size_t>(links) * sizeof(Transform));
        articulation::UpdateWorldLinkPoses(
            context, context_dev, world.view, static_cast<Transform*>(pose_buf.Data()));
        OwnedDeviceBuffer feet_buf(cooked.feet.size() * sizeof(articulation::FootShape));
        feet_buf.CopyFromHost(cooked.feet.data(),
                              cooked.feet.size() * sizeof(articulation::FootShape));
        const uint32_t slot_count = env_count * articulation::kMaxFootContactsPerEnv;
        OwnedDeviceBuffer link_buf(slot_count * sizeof(uint32_t));
        OwnedDeviceBuffer point_buf(slot_count * sizeof(Vec3));
        OwnedDeviceBuffer normal_buf(slot_count * sizeof(Vec3));
        OwnedDeviceBuffer depth_buf(slot_count * sizeof(float));
        OwnedDeviceBuffer count_buf(env_count * sizeof(uint32_t));
        articulation::DetectFootGroundContacts(
            context, context_dev, static_cast<const Transform*>(pose_buf.Data()),
            static_cast<const articulation::FootShape*>(feet_buf.Data()),
            static_cast<uint32_t>(cooked.feet.size()), env_count, base_link_count,
            ground_height,
            static_cast<uint32_t*>(link_buf.Data()),
            static_cast<Vec3*>(point_buf.Data()),
            static_cast<Vec3*>(normal_buf.Data()),
            static_cast<float*>(depth_buf.Data()),
            static_cast<uint32_t*>(count_buf.Data()));
        cudaStreamSynchronize(context);
        Contacts c;
        c.link.resize(slot_count);
        c.point.resize(slot_count);
        c.normal.resize(slot_count);
        c.depth.resize(slot_count);
        link_buf.CopyToHost(c.link.data(), slot_count * sizeof(uint32_t));
        point_buf.CopyToHost(c.point.data(), slot_count * sizeof(Vec3));
        normal_buf.CopyToHost(c.normal.data(), slot_count * sizeof(Vec3));
        depth_buf.CopyToHost(c.depth.data(), slot_count * sizeof(float));
        return c;
    };

    // Press the feet into the ground. CRITICAL: keep the cooked bent stand stance
    // (thigh ~0.8, calf ~-1.5 per leg) -- do NOT straighten the legs to q=0. At
    // q=0 the only DOF that moves a foot vertically (+Z normal) is the hip-roll
    // joint, and the Y-tangent friction row routes through that SAME single DOF;
    // the normal and friction rows then fight over one DOF, PGS oscillates, the
    // foot never acquires vertical velocity, and the FK normal-velocity residual
    // (gate 2) stays at the full bias (a rank-deficient/singular contact pose, NOT
    // a wiring bug -- the t1 chain Jacobian shows thigh/calf DOFs are reachable).
    // At the bent stance the thigh/calf produce vertical foot motion through DOFs
    // that do not conflict with the friction rows, so the normal constraint is
    // satisfiable and the residual collapses. The cooked feet sit at z ~0.2966
    // (radius 0.0175, foot bottom ~0.279); a ground at 0.31 penetrates all four
    // feet by ~0.031 m.
    const float kGround = 0.31f;

    // -----------------------------------------------------------------------
    // (4-anti-vacuous) Sustained active contact + lambda_n > 0, single env.
    // -----------------------------------------------------------------------
    auto pressed = base;
    for (auto& v : pressed.qdot) {
        v = 0.0f;
    }
    refresh_link_pose(pressed);
    const auto c1 = detect(pressed, 1u, kGround);
    uint32_t active = 0u;
    for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
        if (c1.link[s] != kInvalidLink) {
            ++active;
        }
    }
    ASSERT_GT(active, 0u) << "feet must reach the ground for the gates to be meaningful";
    std::printf("[diag] (4) active foot contacts = %u\n", active);

    {
        const auto res = RunSolve(context, context_dev, pressed, 1u, max_dof, c1.link, c1.point,
                                  c1.normal, c1.depth, kGravity, kDt,
                                  /*warm_start=*/false);
        uint32_t loaded = 0u;
        for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
            if (res.rows[s].row_type == articulation::ContactRowType::kRowNormal) {
                EXPECT_GT(res.lambda[s * 3u], 0.0f)
                    << "active foot " << s << " must carry load (lambda_n > 0)";
                if (res.lambda[s * 3u] > 0.0f) {
                    ++loaded;
                }
            }
        }
        // Dump foot-0 normal chain Jacobian + qdot+ over all 12 DOFs.
        {
            std::printf("[diag] (2-dbg) foot0 normal J = [");
            for (uint32_t k = 0u; k < max_dof; ++k) {
                std::printf("%.4f ", static_cast<double>(res.normal_jac[k]));
            }
            std::printf("]\n[diag] (2-dbg) foot0 t1 J = [");
            for (uint32_t k = 0u; k < max_dof; ++k) {
                std::printf("%.4f ", static_cast<double>(res.tangent1_jac[k]));
            }
            std::printf("]\n[diag] (2-dbg) foot0 t2 J = [");
            for (uint32_t k = 0u; k < max_dof; ++k) {
                std::printf("%.4f ", static_cast<double>(res.tangent2_jac[k]));
            }
            std::printf("]\n[diag] (2-dbg) qdot+ = [");
            for (uint32_t l = 0u; l < base_link_count; ++l) {
                std::printf("%.6g ", static_cast<double>(res.qdot_post[l]));
            }
            std::printf("]\n[diag] (2-dbg) lambda_n/t1/t2 foot0 = %.6g %.6g %.6g  m_eff_n=%.6g\n",
                        static_cast<double>(res.lambda[0]),
                        static_cast<double>(res.lambda[1]),
                        static_cast<double>(res.lambda[2]),
                        static_cast<double>(res.rows[0].effective_mass));
        }
        std::printf("[diag] (4) feet carrying load (lambda_n>0) = %u\n", loaded);
        EXPECT_GT(loaded, 0u);

        // ----------------------------------------------------------------
        // (2) Independent FK normal-velocity residual on the solved qdot+.
        // ----------------------------------------------------------------
        // Re-run ABA Pass-1 on qdot+ (recomputes link_velocity via the spatial
        // path -- different code from the scalar chain J) and check each active
        // contact's contact-point normal velocity equals the Baumgarte target.
        auto post = pressed;
        post.qdot = res.qdot_post;
        const auto world_pose = refresh_link_pose(post);
        auto world = Upload(context, context_dev, post);
        articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, world.view, kGravity);
        cudaStreamSynchronize(context);
        articulation::ArticulationHostState dl = post;
        articulation::DownloadArticulationState(world.buffers, &dl);

        float max_residual = 0.0f;
        for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
            if (res.rows[s].row_type != articulation::ContactRowType::kRowNormal) {
                continue;
            }
            const uint32_t link = c1.link[s];
            const float vn = ContactNormalVelocityFK(
                world_pose[link], dl.link_velocity[link], c1.point[s], c1.normal[s]);
            const float bias =
                articulation::kBaumgarteBeta / kDt *
                std::fmax(c1.depth[s] - articulation::kPenetrationSlop, 0.0f);
            const float residual = std::fabs(vn - bias);
            max_residual = std::max(max_residual, residual);
            const auto& lv = dl.link_velocity[link];
            std::printf("[diag] (2) foot %u link %u qdot+=%.5f lv=[%.4f %.4f %.4f | %.4f %.4f %.4f] "
                        "FK v_n=%.6f bias=%.6f residual=%.3e\n",
                        s, link, static_cast<double>(res.qdot_post[link]),
                        static_cast<double>(lv.v[0]), static_cast<double>(lv.v[1]),
                        static_cast<double>(lv.v[2]), static_cast<double>(lv.v[3]),
                        static_cast<double>(lv.v[4]), static_cast<double>(lv.v[5]),
                        static_cast<double>(vn), static_cast<double>(bias),
                        static_cast<double>(residual));
        }
        // PGS with friction does not drive every normal row to exact equality in
        // a finite sweep, but the residual must be small relative to the bias.
        EXPECT_LT(max_residual, 5.0e-2f)
            << "FK normal-velocity residual too large -> J/M^-1/dof-map wiring bug";
    }

    // -----------------------------------------------------------------------
    // (3) Block-diagonal structural check: ONE injected contact -> impulse on
    //     exactly that leg's ancestor DOFs (independent parent_link walk).
    // -----------------------------------------------------------------------
    {
        auto host = pressed;
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        refresh_link_pose(host);

        // Pick the first active contact link from the detection above.
        uint32_t inject_link = kInvalidLink;
        Vec3 inject_point{};
        float inject_depth = 0.0f;
        for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
            if (c1.link[s] != kInvalidLink) {
                inject_link = c1.link[s];
                inject_point = c1.point[s];
                inject_depth = c1.depth[s];
                break;
            }
        }
        ASSERT_NE(inject_link, kInvalidLink);

        std::vector<uint32_t> link(articulation::kMaxFootContactsPerEnv, kInvalidLink);
        std::vector<Vec3> point(articulation::kMaxFootContactsPerEnv, Vec3::Zero());
        std::vector<Vec3> norm(articulation::kMaxFootContactsPerEnv, Vec3::Zero());
        std::vector<float> depth(articulation::kMaxFootContactsPerEnv, 0.0f);
        link[0] = inject_link;
        point[0] = inject_point;
        norm[0] = Vec3{0.0f, 0.0f, 1.0f};
        depth[0] = inject_depth;

        const auto res = RunSolve(context, context_dev, host, 1u, max_dof, link, point, norm,
                                  depth, kGravity, kDt, /*warm_start=*/false);

        // Independent topology walk: the ancestor links of inject_link up to the
        // root are the ONLY DOFs whose qdot may change. No reuse of the solver's
        // dof->qdot map.
        const uint32_t offset = host.articulation_link_offset[0];
        std::vector<uint8_t> may_change(base_link_count, 0u);
        uint32_t walk = inject_link;
        while (walk != kInvalidLink) {
            if (articulation::ArticulationJointDofCount(host.joint_type[walk]) != 0u) {
                may_change[walk] = 1u;
            }
            const uint32_t parent_local = host.parent_link[walk];
            walk = (parent_local == kInvalidLink) ? kInvalidLink : (offset + parent_local);
        }

        const auto dof_to_link = LocalDofToLink(host, 0u);
        uint32_t changed_in_leg = 0u;
        uint32_t changed_outside = 0u;
        for (uint32_t link_idx = 0u; link_idx < base_link_count; ++link_idx) {
            if (articulation::ArticulationJointDofCount(host.joint_type[link_idx]) == 0u) {
                continue;
            }
            const bool changed = (res.qdot_post[link_idx] != res.qdot_free[link_idx]);
            if (may_change[link_idx]) {
                if (changed) {
                    ++changed_in_leg;
                }
            } else {
                // Other legs must be bit-identical (M is block-diagonal per leg).
                EXPECT_EQ(res.qdot_post[link_idx], res.qdot_free[link_idx])
                    << "impulse leaked to a non-contact-leg DOF (link " << link_idx
                    << ") -> dof-column->qdot permutation bug";
                if (changed) {
                    ++changed_outside;
                }
            }
        }
        std::printf("[diag] (3) single contact: changed-in-leg=%u changed-outside=%u "
                    "(leg ancestor DOFs)\n", changed_in_leg, changed_outside);
        (void)dof_to_link;
        EXPECT_GT(changed_in_leg, 0u) << "contact must move its own leg";
        EXPECT_EQ(changed_outside, 0u);
    }

    // -----------------------------------------------------------------------
    // (4-stability) Press feet, run K steps: bounded penetration + finite qddot.
    // -----------------------------------------------------------------------
    {
        auto host = pressed;
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        const uint32_t kSteps = 30u;
        float max_depth_seen = 0.0f;
        float first_total_depth = -1.0f;
        float last_total_depth = 0.0f;
        for (uint32_t step = 0u; step < kSteps; ++step) {
            refresh_link_pose(host);
            const auto c = detect(host, 1u, kGround);
            float total_depth = 0.0f;
            for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
                if (c.link[s] != kInvalidLink) {
                    total_depth += c.depth[s];
                    max_depth_seen = std::max(max_depth_seen, c.depth[s]);
                }
            }
            if (first_total_depth < 0.0f) {
                first_total_depth = total_depth;
            }
            last_total_depth = total_depth;

            const auto res = RunSolve(context, context_dev, host, 1u, max_dof, c.link, c.point,
                                      c.normal, c.depth, kGravity, kDt,
                                      /*warm_start=*/false);
            // qdot+ must be finite.
            for (float v : res.qdot_post) {
                ASSERT_TRUE(std::isfinite(v)) << "non-finite qdot at step " << step;
            }
            // Apply the solved qdot, then integrate q (semi-implicit Euler).
            host.qdot = res.qdot_post;
            for (uint32_t l = 0u; l < base_link_count; ++l) {
                host.q[l] += host.qdot[l] * kDt;
                ASSERT_TRUE(std::isfinite(host.q[l]));
            }
        }
        std::printf("[diag] (4-stability) total depth first=%.6f last=%.6f max=%.6f over %u steps\n",
                    static_cast<double>(first_total_depth),
                    static_cast<double>(last_total_depth),
                    static_cast<double>(max_depth_seen), kSteps);
        // Penetration must stay bounded (not blow up). Allow generous headroom
        // over the initial penetration; the key is it does not diverge.
        EXPECT_LT(max_depth_seen, 0.5f) << "penetration grew unbounded";
        EXPECT_LT(last_total_depth, first_total_depth + 0.1f)
            << "penetration trending upward unbounded";
    }

    // -----------------------------------------------------------------------
    // (4-determinism) Bit-identical across two runs and across 64 replicas.
    // -----------------------------------------------------------------------
    {
        auto host = pressed;
        for (auto& v : host.qdot) {
            v = 0.0f;
        }
        refresh_link_pose(host);
        const auto c = detect(host, 1u, kGround);
        const auto run_a = RunSolve(context, context_dev, host, 1u, max_dof, c.link, c.point,
                                    c.normal, c.depth, kGravity, kDt,
                                    /*warm_start=*/false);
        const auto run_b = RunSolve(context, context_dev, host, 1u, max_dof, c.link, c.point,
                                    c.normal, c.depth, kGravity, kDt,
                                    /*warm_start=*/false);
        ASSERT_EQ(run_a.qdot_post.size(), run_b.qdot_post.size());
        for (size_t i = 0u; i < run_a.qdot_post.size(); ++i) {
            EXPECT_EQ(run_a.qdot_post[i], run_b.qdot_post[i])
                << "qdot+ not bit-identical across runs at " << i;
        }
        for (size_t i = 0u; i < run_a.lambda.size(); ++i) {
            EXPECT_EQ(run_a.lambda[i], run_b.lambda[i])
                << "lambda not bit-identical across runs at " << i;
        }

        // 64 replicas: every env's solved qdot+ / lambda matches env 0 bit-for-bit.
        constexpr uint32_t kEnvCount = 64u;
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        ASSERT_EQ(batched.ArticulationCount(), kEnvCount);
        refresh_link_pose(batched);
        const auto cb = detect(batched, kEnvCount, kGround);
        const auto rb = RunSolve(context, context_dev, batched, kEnvCount, max_dof, cb.link,
                                 cb.point, cb.normal, cb.depth, kGravity, kDt,
                                 /*warm_start=*/false);
        size_t qdot_mismatches = 0u;
        for (uint32_t env = 1u; env < kEnvCount; ++env) {
            for (uint32_t l = 0u; l < base_link_count; ++l) {
                if (rb.qdot_post[env * base_link_count + l] != rb.qdot_post[l]) {
                    ++qdot_mismatches;
                }
            }
        }
        const uint32_t per_env_slots = articulation::kMaxFootContactsPerEnv * 3u;
        size_t lambda_mismatches = 0u;
        for (uint32_t env = 1u; env < kEnvCount; ++env) {
            for (uint32_t i = 0u; i < per_env_slots; ++i) {
                if (rb.lambda[env * per_env_slots + i] != rb.lambda[i]) {
                    ++lambda_mismatches;
                }
            }
        }
        std::printf("[diag] (4-determinism) replica qdot mismatches=%zu lambda mismatches=%zu\n",
                    qdot_mismatches, lambda_mismatches);
        EXPECT_EQ(qdot_mismatches, 0u);
        EXPECT_EQ(lambda_mismatches, 0u);
    }

    // -----------------------------------------------------------------------
    // (5) Friction isolation: stick vs slip, validated by INDEPENDENT FK.
    // -----------------------------------------------------------------------
    //
    // WHY THE OLD GATE WAS ILL-POSED (and is replaced): it pressed the feet 2.5cm
    // into the ground, so Baumgarte pushed each foot OUT at ~1.23 m/s in one step.
    // On the bent Go2 leg the normal-pushout and a friction tangent SHARE the thigh
    // DOF (Jn.M^-1.Jt^T ~ -1.26), so the mandatory normal pushout necessarily swings
    // the foot tangentially -- AT mu=0 the foot already slides ~0.95 m/s. The old
    // assertion |v_t_post| <= |v_t_free(contact-free)| omitted that unavoidable swing
    // and was unsatisfiable for correct physics. It is a test bug, not a solver bug.
    //
    // WHY THE EARLIER "contraction = Jt.M^-1.Jt^T * m_eff == 1" CHECK WAS CIRCULAR:
    // m_eff_t is DEFINED as 1/(Jt.M^-1.Jt^T), so the product is 1 for ANY tangent
    // Jacobian -- including a geometrically wrong one. The tangent-J GEOMETRY was
    // never validated. This gate validates it by measuring v_t through INDEPENDENT
    // forward kinematics (link spatial velocity + cross product at the contact
    // point, via ContactNormalVelocityFK with a tangent direction), NOT through the
    // assembled tangent J row -- so a wrong-direction/wrong-scale Jt is exposed.
    //
    // WELL-POSED DESIGN:
    //  * ONE injected contact on leg 0 at depth = kPenetrationSlop/2: the contact is
    //    active (depth>0) but the Baumgarte bias is EXACTLY 0, so there is no
    //    normal-push-induced swing baked into the comparison.
    //  * lambda_n is engaged ONLY by an into-ground (downward) foot velocity we set
    //    in qdot (gravity does NOT enter qdot_free in this harness -- the kernel
    //    seeds qdot_work directly from state.qdot). A purely tangential push would
    //    give lambda_n ~ 0 and a vacuous cone; the downward component is mandatory.
    //  * We compare WITH friction (mu=0.8) vs WITHOUT friction (mu=0) on the SAME
    //    qdot, NOT vs the contact-free velocity. Both runs share the same downward
    //    push and (nearly) the same lambda_n, so any normal-push-induced swing
    //    cancels in the comparison and only the friction effect remains.
    //  * TWO sub-cases driven by the DOWNWARD-vs-TANGENTIAL split (stick/slip is a
    //    ratio m_eff_t*|jv_t| vs mu*m_eff_n*|jv_n|, not a magnitude):
    //      STICK: big downward push + small tangential push -> friction stays
    //             INTERIOR (|lambda_t| < mu*lambda_n strictly) and drives v_t -> ~0.
    //             This is what catches sign/skew errors in the UNCLAMPED friction
    //             update and validates the tangent-J geometry/direction.
    //      SLIP:  modest downward push + large tangential push -> friction SATURATES
    //             (|lambda_t| == mu*lambda_n within tol); v_t is measurably reduced
    //             but still nonzero. Saturation exercises the cone scale.
    //    (Pure SCALE error in Jt cancels in the unclamped update via m_eff_t, but
    //    surfaces in the slip saturation; DIRECTION/skew error surfaces in the stick
    //    v_t->0. The two together make the gate non-fakeable.)
    //  * Anti-vacuous guards asserted BEFORE the comparisons: lambda_n > 0 (contact
    //    engaged) and a genuine frictionless slide exists (|v_t_frictionless| above a
    //    threshold for slip; nonzero tangential input for stick).
    {
        // Leg-0 DOFs: dof0->link1 (hip-roll), dof1->link2 (thigh), dof2->link3
        // (calf). From the (2-dbg) dump the leg-0 contact geometry is
        //   Jn  ~ [+0.0955, +0.1528, 0, ...]   (hip-roll + thigh move foot in +Z)
        //   Jt1 ~ [0, -0.1350, +0.0134, ...]   (thigh dominates t1, calf nearly pure)
        //   Jt2 ~ [+0.1350, 0, 0, ...]         (hip-roll is pure t2)
        // so a NEGATIVE thigh rate drives the foot DOWN (engages lambda_n) and a
        // calf rate produces an almost-pure t1 slide. We shape the two sub-cases by
        // the magnitude split between the (downward) thigh rate and the (tangential)
        // calf rate.
        const auto dof_to_link = LocalDofToLink(base, 0u);
        ASSERT_GE(dof_to_link.size(), 3u);
        const uint32_t link_hip = dof_to_link[0];
        const uint32_t link_thigh = dof_to_link[1];
        const uint32_t link_calf = dof_to_link[2];

        // Build the single-contact slot arrays for leg 0 at a slop/2 depth.
        auto build_pressed = pressed;
        for (auto& v : build_pressed.qdot) {
            v = 0.0f;
        }
        const auto world_pose0 = refresh_link_pose(build_pressed);
        const auto cseed = detect(build_pressed, 1u, kGround);
        uint32_t inject_link = kInvalidLink;
        Vec3 inject_point{};
        for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
            if (cseed.link[s] != kInvalidLink) {
                inject_link = cseed.link[s];
                inject_point = cseed.point[s];
                break;
            }
        }
        ASSERT_NE(inject_link, kInvalidLink);
        // Resting contact: depth just above slop is treated as zero-bias because
        // bias = beta/dt * max(depth - slop, 0); use slop/2 so the row is active
        // (depth>0) yet the Baumgarte target is exactly 0.
        const float kRestDepth = articulation::kPenetrationSlop * 0.5f;

        std::vector<uint32_t> link(articulation::kMaxFootContactsPerEnv, kInvalidLink);
        std::vector<Vec3> point(articulation::kMaxFootContactsPerEnv, Vec3::Zero());
        std::vector<Vec3> norm(articulation::kMaxFootContactsPerEnv, Vec3::Zero());
        std::vector<float> depth(articulation::kMaxFootContactsPerEnv, 0.0f);
        link[0] = inject_link;
        point[0] = inject_point;
        norm[0] = Vec3{0.0f, 0.0f, 1.0f};
        depth[0] = kRestDepth;

        // Runs the resting-contact friction scenario for one (downward, tangential)
        // velocity split at a given mu, and reports the FK-measured tangential
        // velocity + the converged impulses. world_pose is reused for FK.
        struct FrictionDiag {
            float lambda_n = 0.0f;
            float lambda_t1 = 0.0f;
            float lambda_t2 = 0.0f;
            float cone = 0.0f;
            float vt_t1 = 0.0f;     // FK tangential velocity along t1 (post-solve).
            float vt_t2 = 0.0f;     // FK tangential velocity along t2 (post-solve).
            float vt_mag = 0.0f;    // sqrt(vt_t1^2 + vt_t2^2).
            float jvt1 = 0.0f;      // tangential velocity through the ASSEMBLED J row.
            float jvt2 = 0.0f;      // (cross-check vs FK -> isolates geometry vs solve).
            Vec3 t1{};
            Vec3 t2{};
        };
        auto run_friction = [&](float thigh_rate, float calf_rate, float hip_rate,
                                float mu) -> FrictionDiag {
            auto host = build_pressed;
            for (auto& v : host.qdot) {
                v = 0.0f;
            }
            host.qdot[link_thigh] = thigh_rate;
            host.qdot[link_calf] = calf_rate;
            host.qdot[link_hip] = hip_rate;
            // Keep link_pose consistent with the q used everywhere (already set by
            // world_pose0; re-set to be explicit since we mutate host).
            for (uint32_t l = 0u; l < host.TotalLinkCount(); ++l) {
                host.link_pose[l] = world_pose0[l];
            }
            const auto res = RunSolve(context, context_dev, host, 1u, max_dof, link, point, norm,
                                      depth, kGravity, kDt, /*warm_start=*/false, mu);
            FrictionDiag d;
            d.lambda_n = res.lambda[0];
            d.lambda_t1 = res.lambda[1];
            d.lambda_t2 = res.lambda[2];
            d.t1 = res.rows[0].tangent1;
            d.t2 = res.rows[0].tangent2;
            d.cone = mu * std::fmax(d.lambda_n, 0.0f);

            // INDEPENDENT FK: re-run ABA Pass-1 on qdot+ and measure the contact
            // point's velocity along t1 and t2 (NOT through the assembled J).
            auto post = host;
            post.qdot = res.qdot_post;
            for (uint32_t l = 0u; l < post.TotalLinkCount(); ++l) {
                post.link_pose[l] = world_pose0[l];
            }
            auto world = Upload(context, context_dev, post);
            articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, world.view, kGravity);
            cudaStreamSynchronize(context);
            articulation::ArticulationHostState dl = post;
            articulation::DownloadArticulationState(world.buffers, &dl);
            d.vt_t1 = ContactNormalVelocityFK(world_pose0[inject_link],
                                              dl.link_velocity[inject_link],
                                              inject_point, d.t1);
            d.vt_t2 = ContactNormalVelocityFK(world_pose0[inject_link],
                                              dl.link_velocity[inject_link],
                                              inject_point, d.t2);
            d.vt_mag = std::sqrt(d.vt_t1 * d.vt_t1 + d.vt_t2 * d.vt_t2);
            // Cross-check: tangential velocity through the ASSEMBLED tangent J row
            // on qdot+ (slot 0, jac base 0). If this agrees with the FK number the
            // tangent J is geometrically consistent and any residual is solve-side;
            // if it disagrees (J~0 but FK!=0) the tangent geometry is wrong.
            for (uint32_t k = 0u; k < dof_to_link.size(); ++k) {
                d.jvt1 += res.tangent1_jac[k] * res.qdot_post[dof_to_link[k]];
                d.jvt2 += res.tangent2_jac[k] * res.qdot_post[dof_to_link[k]];
            }
            return d;
        };

        auto report = [&](const char* tag, float mu, const FrictionDiag& d) {
            std::printf("[diag] (5-%s mu=%.1f) lambda_n=%.6f |lt1|=%.6f |lt2|=%.6f "
                        "mu*lambda_n=%.6f  v_t1=%.6f v_t2=%.6f |v_t|=%.6f "
                        "jvt(throughJ)=%.6f %.6f\n",
                        tag, static_cast<double>(mu),
                        static_cast<double>(d.lambda_n),
                        static_cast<double>(std::fabs(d.lambda_t1)),
                        static_cast<double>(std::fabs(d.lambda_t2)),
                        static_cast<double>(d.cone),
                        static_cast<double>(d.vt_t1),
                        static_cast<double>(d.vt_t2),
                        static_cast<double>(d.vt_mag),
                        static_cast<double>(d.jvt1),
                        static_cast<double>(d.jvt2));
        };

        const float kMu = articulation::kContactFriction;  // 0.8

        // ----------------------- STICK sub-case ----------------------------
        // Big downward push (thigh) + small tangential push (calf): the impulse
        // needed to arrest the slide is well within mu*lambda_n, so friction is
        // INTERIOR and the foot sticks (v_t -> ~0). Catches direction/skew errors.
        {
            const float kThighDown = -6.0f;  // negative thigh => foot moves -Z.
            const float kCalfTan = 0.3f;     // small tangential slide (calf ~ pure t1).
            const auto d0 = run_friction(kThighDown, kCalfTan, 0.0f, 0.0f);
            const auto df = run_friction(kThighDown, kCalfTan, 0.0f, kMu);
            report("stick", 0.0f, d0);
            report("stick", kMu, df);

            // Anti-vacuous: contact engaged and a real frictionless slide exists.
            ASSERT_GT(df.lambda_n, 0.0f) << "stick: contact must carry load (lambda_n>0)";
            ASSERT_GT(std::fabs(d0.vt_t1), 1.0e-3f)
                << "stick: a genuine frictionless tangential slide must exist";

            // Friction is INTERIOR (strictly inside the cone) -- the defining
            // property of sticking; validates the UNCLAMPED friction update.
            const float cone = df.cone;
            EXPECT_LT(std::fabs(df.lambda_t1), cone)
                << "stick: tangent1 impulse must be strictly interior (|lt1| < mu*ln)";
            EXPECT_LT(std::fabs(df.lambda_t2), cone)
                << "stick: tangent2 impulse must be strictly interior (|lt2| < mu*ln)";

            // The foot STICKS: friction drives the tangential velocity to ~0. Tight
            // tolerance -- a wrong-direction tangent J would leave residual slide.
            EXPECT_LT(df.vt_mag, 1.0e-2f)
                << "stick: friction must drive tangential foot velocity to ~0";
            // And it is a real reduction vs frictionless.
            EXPECT_LT(df.vt_mag, 0.25f * d0.vt_mag)
                << "stick: friction must dramatically reduce the slide";
        }

        // ----------------------- SLIP sub-case ------------------------------
        // Modest downward push (just enough for lambda_n>0) + large tangential push:
        // the demanded friction impulse exceeds mu*lambda_n, friction SATURATES, the
        // slide is reduced but not eliminated. Exercises the cone scale.
        {
            const float kThighDown = -1.5f;  // modest downward => small lambda_n.
            const float kCalfTan = 8.0f;     // large tangential slide.
            const auto d0 = run_friction(kThighDown, kCalfTan, 0.0f, 0.0f);
            const auto df = run_friction(kThighDown, kCalfTan, 0.0f, kMu);
            report("slip", 0.0f, d0);
            report("slip", kMu, df);

            // Anti-vacuous: contact engaged and a substantial frictionless slide.
            ASSERT_GT(df.lambda_n, 0.0f) << "slip: contact must carry load (lambda_n>0)";
            ASSERT_GT(d0.vt_mag, 0.1f)
                << "slip: a substantial frictionless slide must exist to oppose";

            // Friction SATURATES: the dominant tangent rides the cone boundary.
            const float cone = df.cone;
            const float tol = 1.0e-3f * std::fmax(cone, 1.0f);
            const float lt_max = std::fmax(std::fabs(df.lambda_t1), std::fabs(df.lambda_t2));
            EXPECT_GT(lt_max, cone - tol)
                << "slip: friction must saturate (a tangent impulse rides the cone)";

            // Friction MEASURABLY reduces the slide, but does NOT eliminate it
            // (it cannot, since the demand exceeds the cone).
            EXPECT_LT(df.vt_mag, 0.85f * d0.vt_mag)
                << "slip: friction must measurably reduce the slide";
            EXPECT_GT(df.vt_mag, 1.0e-2f)
                << "slip: saturated friction cannot fully stop the slide";
        }

        // ----------------------- Friction cone + determinism ----------------
        // Cone holds in a representative scenario, and the solve is bit-identical
        // across two runs (D1 determinism, also exercised under the mu toggle).
        {
            const auto a = run_friction(-3.0f, 2.0f, 0.0f, kMu);
            const auto b = run_friction(-3.0f, 2.0f, 0.0f, kMu);
            const float cone = a.cone;
            const float tol = 1.0e-5f * std::fmax(cone, 1.0f);
            EXPECT_LE(std::fabs(a.lambda_t1), cone + tol) << "tangent1 outside cone";
            EXPECT_LE(std::fabs(a.lambda_t2), cone + tol) << "tangent2 outside cone";
            EXPECT_EQ(a.lambda_n, b.lambda_n) << "lambda_n not deterministic";
            EXPECT_EQ(a.lambda_t1, b.lambda_t1) << "lambda_t1 not deterministic";
            EXPECT_EQ(a.lambda_t2, b.lambda_t2) << "lambda_t2 not deterministic";
            EXPECT_EQ(a.vt_t1, b.vt_t1) << "v_t1 not deterministic";
            std::printf("[diag] (5-cone) lambda_n=%.6f |lt1|=%.6f |lt2|=%.6f cone=%.6f\n",
                        static_cast<double>(a.lambda_n),
                        static_cast<double>(std::fabs(a.lambda_t1)),
                        static_cast<double>(std::fabs(a.lambda_t2)),
                        static_cast<double>(cone));
        }
    }
}
