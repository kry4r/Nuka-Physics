// ---------------------------------------------------------------------------
// v0.8 C4c -- ReactionProvider validation (rigid invM / articulation chain-J /
//             particle invM / static null)
//
// Validates src/constraint/reaction_provider.hpp: the §0.5 reaction-provider
// interface (effective_inv_mass + apply_impulse), three real reaction models
// plus the immovable StaticNull, each MIRRORING an existing production math path
// so the C5 wiring is behavior-preserving.
//
// The heart of the test is the INDEPENDENT hand-oracle for each model:
//
//  * ArticulationChainJ -- a hand-built ONE-revolute-DOF pendulum (Fixed root +
//    one Revolute link, COM at the joint origin so M is exactly the axis inertia
//    I_a with NO parallel-axis term). The contact point is placed OFF the link
//    origin (the C4c arbitrary-contact-point exercise). With the joint origin at
//    the world origin and a unit revolute axis `a`:
//        J = dot( a x (p - origin), n )     (scalar chain Jacobian, 1 DOF)
//        M = I_a   (verified against the device CRBA build before trusting it)
//        effective_inv_mass = J M^-1 J^T = J^2 / I_a
//        apply_impulse(dl): dqdot = M^-1 J^T dl = (J / I_a) * dl
//    We compute J, I_a, J^2/I_a and (J/I_a)*dl BY HAND (see the arithmetic in
//    the test body) and assert the provider reproduces them.
//
//  * RigidInvMass -- a box with known mass m and principal inertia (Ix,Iy,Iz).
//    invM = diag(1/m I3, 1/Ix,1/Iy,1/Iz) (the maximal diagonal path row_solver.cu
//    uses). effective_inv_mass = (1/m)|Jlin|^2 + sum Jang.k^2/Ik; apply Dv/Dw by
//    hand. (Diagonal inv_inertia, NOT R I^-1 R^T -- the behavior-preservation
//    note in the header.)
//
//  * ParticleInvMass -- scalar 1/m: effective = (1/m)|Jlin|^2; Dv = dl/m Jlin.
//
//  * StaticNull -- effective_inv_mass == 0 exactly; apply_impulse is a no-op.
//
//  * D1: every effective_inv_mass + every applied delta is byte-identical across
//    two runs (the providers reuse already-D1 paths; no new nondeterminism).
//
// The articulation path round-trips the device CRBA M^-1 + chain-J builders (the
// proven, already-D1 production kernels the provider consumes), so this is a .cu
// CUDA-language target.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"
#include "constraint/reaction_provider.hpp"
#include "constraint/row.hpp"
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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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
namespace constraint = nuka::constraint;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

// ---------------------------------------------------------------------------
// Hand-build a single-revolute-DOF pendulum ArticulationHostState.
//   link 0: Fixed root at the world origin (identity pose).
//   link 1: Revolute about `axis` (link-frame), child of link 0, joint origin
//           at the world origin, COM AT the joint origin (inertial_frame =
//           Identity) so the spatial inertia carries NO parallel-axis term and
//           CRBA M (1x1) == the axis component of the diagonal inertia.
// All arrays are filled directly (no cooker) so the inertia + axis are exact and
// known. q is set so the revolute link's world pose is identity (q controls the
// joint rotation; the link_pose we set explicitly is what the chain-J reads).
// ---------------------------------------------------------------------------
articulation::ArticulationHostState BuildRevolutePendulum(Vec3 axis,
                                                          Vec3 diagonal_inertia,
                                                          float mass) {
    articulation::ArticulationHostState s;
    s.articulation_link_offset = {0u};
    s.articulation_link_count = {2u};
    s.base_pose = {Transform::Identity()};

    const uint32_t link_count = 2u;
    for (uint32_t link = 0u; link < link_count; ++link) {
        const bool is_root = (link == 0u);
        // Root: a massless Fixed link (no DOF, no inertia). Child: the revolute
        // link with the prescribed inertia + COM at the joint origin.
        const float link_mass = is_root ? 0.0f : mass;
        const Vec3 link_inertia = is_root ? Vec3::Zero() : diagonal_inertia;
        s.link_inertia.push_back(articulation::MakeSpatialInertia(
            link_mass, link_inertia, Transform::Identity()));
        s.link_velocity.emplace_back();
        s.link_acceleration.emplace_back();
        s.link_xup.emplace_back();
        s.link_velocity_bias.emplace_back();
        s.link_articulated_inertia.emplace_back();
        s.link_bias_force.emplace_back();
        s.link_u_spatial.emplace_back();
        s.joint_motion_subspace.emplace_back();
        // Both links at the world origin (identity world pose). The chain-J reads
        // link_pose (cook-time static) for the local->world axis rotation + the
        // revolute lever origin; identity keeps the oracle arithmetic clean.
        s.link_pose.push_back(Transform::Identity());
        s.link_local_pose.push_back(Transform::Identity());
        s.link_inertial_frame.push_back(Transform::Identity());
        s.q.push_back(0.0f);
        s.qdot.push_back(0.0f);
        s.qddot.push_back(0.0f);
        s.tau.push_back(0.0f);
        s.joint_damping.push_back(0.0f);
        s.joint_armature.push_back(0.0f);
        s.joint_diagonal.push_back(0.0f);
        s.joint_force.push_back(0.0f);
        s.joint_axis.push_back(is_root ? Vec3::UnitZ() : axis);
        s.parent_offset.push_back(Vec3::Zero());
        s.joint_type.push_back(is_root ? articulation::ArticulationJointType::Fixed
                                       : articulation::ArticulationJointType::Revolute);
        s.parent_link.push_back(is_root ? kInvalidLink : 0u);
        s.link_body.push_back(link);
        s.link_to_articulation.push_back(0u);
    }
    return s;
}

// Device round-trip: upload, ABA Pass-1 (link_xup / motion subspaces current),
// CRBA M, LDL^T inverse, chain-J for the given contact, effective mass. Returns
// the downloaded M tile, M^-1 tile, chain-J row, and the device m_eff.
struct DevicePendulum {
    std::vector<float> M;        // [dof*dof]
    std::vector<float> Minv;     // [dof*dof]
    std::vector<float> chain_j;  // [dof]
    float device_m_eff = 0.0f;   // 1 / (J M^-1 J^T) from the production kernel
    uint32_t dof = 0u;
};

DevicePendulum RunDevicePendulum(const nuka::phi::DeviceContext& context,
                                 const articulation::ArticulationHostState& host,
                                 uint32_t contact_link,
                                 Vec3 contact_point,
                                 Vec3 contact_normal) {
    DevicePendulum out;
    const uint32_t dof = articulation::ArticulationDofCount(host, 0u);
    out.dof = dof;
    const uint32_t max_dof = dof;
    const size_t tile = static_cast<size_t>(max_dof) * max_dof;

    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, 0.0f);

    OwnedDeviceBuffer composite_buf(
        static_cast<size_t>(host.TotalLinkCount()) *
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

    const uint32_t contact_count = 1u;
    OwnedDeviceBuffer link_idx_buf(sizeof(uint32_t));
    OwnedDeviceBuffer point_buf(sizeof(Vec3));
    OwnedDeviceBuffer normal_buf(sizeof(Vec3));
    OwnedDeviceBuffer jac_buf(static_cast<size_t>(max_dof) * sizeof(float));
    OwnedDeviceBuffer meff_buf(sizeof(float));
    link_idx_buf.CopyFromHost(&contact_link, sizeof(uint32_t));
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

    out.M.resize(tile);
    out.Minv.resize(tile);
    out.chain_j.resize(max_dof);
    m_buf.CopyToHost(out.M.data(), out.M.size() * sizeof(float));
    minv_buf.CopyToHost(out.Minv.data(), out.Minv.size() * sizeof(float));
    jac_buf.CopyToHost(out.chain_j.data(), out.chain_j.size() * sizeof(float));
    meff_buf.CopyToHost(&out.device_m_eff, sizeof(float));
    return out;
}

} // namespace

// ===========================================================================
// RigidInvMass -- analytic J M^-1 J^T + Dv/Dw vs row_solver's diagonal path.
// ===========================================================================
TEST(ReactionProviders, RigidInvMassMatchesAnalyticAndApplies) {
    // Box: mass 4 kg, principal inertia (2,3,5) kg.m^2. invM diagonal:
    //   inv_mass = 0.25, inv_inertia = (0.5, 1/3, 0.2).
    const float mass = 4.0f;
    const Vec3 inertia{2.0f, 3.0f, 5.0f};
    const float inv_mass = 1.0f / mass;
    const Vec3 inv_inertia{1.0f / inertia.x, 1.0f / inertia.y, 1.0f / inertia.z};

    Vec3 linear_velocity = Vec3::Zero();
    Vec3 angular_velocity = Vec3::Zero();

    constraint::RigidReactionState rigid;
    rigid.inv_mass = inv_mass;
    rigid.inv_inertia = inv_inertia;
    rigid.velocity = &linear_velocity;
    rigid.angular_velocity = &angular_velocity;

    constraint::ReactionProvider provider;
    provider.views.rigid = &rigid;
    provider.views.rigid_count = 1u;

    constraint::CollidableRef ref;
    ref.type = constraint::CollidableType::RigidBody;
    ref.react = constraint::ReactionProviderKind::RigidInvMass;
    ref.handle = 0u;

    // A general (non-axis-aligned) Jacobian so every diagonal term is exercised.
    constraint::RowJacobian6 j;
    j.linear = Vec3{0.6f, -0.8f, 0.0f};   // |Jlin|^2 = 0.36 + 0.64 = 1.0
    j.angular = Vec3{0.3f, 0.5f, -0.7f};

    // --- Hand oracle: effective_inv_mass = (1/m)|Jlin|^2 + sum Jang.k^2 invI.k.
    const float expected_eff =
        inv_mass * (j.linear.x * j.linear.x + j.linear.y * j.linear.y +
                    j.linear.z * j.linear.z) +
        j.angular.x * j.angular.x * inv_inertia.x +
        j.angular.y * j.angular.y * inv_inertia.y +
        j.angular.z * j.angular.z * inv_inertia.z;

    const float eff = provider.effective_inv_mass(ref, j);
    EXPECT_FLOAT_EQ(eff, expected_eff);

    // --- apply_impulse: Dv = dl * inv_mass * Jlin ; Dw.k = dl * Jang.k * invI.k.
    const float dlambda = 1.7f;
    provider.apply_impulse(ref, j, dlambda);
    EXPECT_FLOAT_EQ(linear_velocity.x, dlambda * inv_mass * j.linear.x);
    EXPECT_FLOAT_EQ(linear_velocity.y, dlambda * inv_mass * j.linear.y);
    EXPECT_FLOAT_EQ(linear_velocity.z, dlambda * inv_mass * j.linear.z);
    EXPECT_FLOAT_EQ(angular_velocity.x, dlambda * j.angular.x * inv_inertia.x);
    EXPECT_FLOAT_EQ(angular_velocity.y, dlambda * j.angular.y * inv_inertia.y);
    EXPECT_FLOAT_EQ(angular_velocity.z, dlambda * j.angular.z * inv_inertia.z);

    // Static rigid (inv_mass == 0) must be a no-op even with RigidInvMass kind.
    Vec3 sv = Vec3::Zero();
    Vec3 sw = Vec3::Zero();
    constraint::RigidReactionState immovable;
    immovable.inv_mass = 0.0f;
    immovable.velocity = &sv;
    immovable.angular_velocity = &sw;
    constraint::ReactionProvider immovable_provider;
    immovable_provider.views.rigid = &immovable;
    immovable_provider.views.rigid_count = 1u;
    EXPECT_FLOAT_EQ(immovable_provider.effective_inv_mass(ref, j), 0.0f);
    immovable_provider.apply_impulse(ref, j, dlambda);
    EXPECT_EQ(sv, Vec3::Zero());
    EXPECT_EQ(sw, Vec3::Zero());
}

// ===========================================================================
// ParticleInvMass -- scalar 1/m, translational only.
// ===========================================================================
TEST(ReactionProviders, ParticleInvMassScalarAndApplies) {
    const float mass = 0.5f;
    const float inv_mass = 1.0f / mass;  // 2.0
    Vec3 velocity = Vec3::Zero();

    constraint::ParticleReactionState particle;
    particle.inv_mass = inv_mass;
    particle.velocity = &velocity;

    constraint::ReactionProvider provider;
    provider.views.particle = &particle;
    provider.views.particle_count = 1u;

    constraint::CollidableRef ref;
    ref.type = constraint::CollidableType::Particle;
    ref.react = constraint::ReactionProviderKind::ParticleInvMass;
    ref.handle = 0u;

    constraint::RowJacobian6 j;
    j.linear = Vec3{0.0f, 0.0f, 1.0f};         // |Jlin|^2 = 1
    j.angular = Vec3{9.0f, -9.0f, 9.0f};       // IGNORED by a particle

    // effective_inv_mass = inv_mass * |Jlin|^2 (angular Jacobian must NOT enter).
    EXPECT_FLOAT_EQ(provider.effective_inv_mass(ref, j), inv_mass * 1.0f);

    const float dlambda = 3.0f;
    provider.apply_impulse(ref, j, dlambda);
    EXPECT_FLOAT_EQ(velocity.x, dlambda * inv_mass * j.linear.x);
    EXPECT_FLOAT_EQ(velocity.y, dlambda * inv_mass * j.linear.y);
    EXPECT_FLOAT_EQ(velocity.z, dlambda * inv_mass * j.linear.z);  // 3*2*1 = 6
    EXPECT_FLOAT_EQ(velocity.z, 6.0f);
}

// ===========================================================================
// StaticNull -- zero reaction, no-op apply.
// ===========================================================================
TEST(ReactionProviders, StaticNullIsZeroAndNoOp) {
    constraint::ReactionProvider provider;  // no views resolved for static.

    constraint::CollidableRef ref;
    ref.type = constraint::CollidableType::StaticWorld;
    ref.react = constraint::ReactionProviderKind::StaticNull;
    ref.handle = 0u;

    constraint::RowJacobian6 j;
    j.linear = Vec3{1.0f, 2.0f, 3.0f};
    j.angular = Vec3{4.0f, 5.0f, 6.0f};

    EXPECT_FLOAT_EQ(provider.effective_inv_mass(ref, j), 0.0f);
    // apply_impulse must not crash / touch anything (no DOFs).
    provider.apply_impulse(ref, j, 99.0f);
    SUCCEED();
}

// ===========================================================================
// ArticulationChainJ -- 1-DOF pendulum vs the hand-computed J M^-1 J^T.
//   * contact point OFF the link origin (the C4c arbitrary-point exercise).
//   * M verified == I_a before the contraction is trusted.
// ===========================================================================
TEST(ReactionProviders, ArticulationChainJMatchesHandComputedPendulum) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // Revolute axis = world X; diagonal inertia (Ix,Iy,Iz) = (7, 11, 13).
    // The single DOF is rotation about X, so the hand CRBA M (1x1) == Ix = 7.
    const Vec3 axis{1.0f, 0.0f, 0.0f};
    const Vec3 diagonal_inertia{7.0f, 11.0f, 13.0f};
    const float mass = 2.0f;  // COM at joint origin -> mass adds NO parallel-axis.
    const float I_a = diagonal_inertia.x;  // hand M = 7.0

    auto host = BuildRevolutePendulum(axis, diagonal_inertia, mass);
    const uint32_t revolute_link = 1u;  // global link index of the revolute DOF.

    // Contact point OFF the link origin: (0, 0.4, 0.9). The joint origin is the
    // world origin (link_pose identity), so lever = point - origin = the point.
    // Normal n = world +Z.
    const Vec3 contact_point{0.0f, 0.4f, 0.9f};
    const Vec3 contact_normal{0.0f, 0.0f, 1.0f};

    // --- HAND ORACLE -------------------------------------------------------
    // axis_world = R_link * axis = identity * X = (1,0,0).
    // lever = point - joint_origin = (0, 0.4, 0.9).
    // a x lever = (1,0,0) x (0,0.4,0.9) = (0*0.9 - 0*0.4, 0*0 - 1*0.9, 1*0.4 - 0*0)
    //           = (0, -0.9, 0.4).
    // J = dot(a x lever, n) = dot((0,-0.9,0.4),(0,0,1)) = 0.4.
    const Vec3 a_world = axis;  // identity rotation
    const Vec3 lever = contact_point;  // origin at world origin
    const Vec3 axis_cross_lever = a_world.Cross(lever);
    const float J_hand = axis_cross_lever.Dot(contact_normal);
    ASSERT_NEAR(J_hand, 0.4f, 1.0e-6f) << "hand Jacobian arithmetic check";
    // effective_inv_mass = J^2 / I_a = 0.16 / 7 = 0.0228571...
    const float expected_eff = (J_hand * J_hand) / I_a;
    // apply_impulse(dl): dqdot = (J / I_a) * dl.
    const float dlambda = 2.5f;
    const float expected_dqdot = (J_hand / I_a) * dlambda;

    // --- DEVICE BUILD (the proven CRBA M^-1 + chain-J the provider consumes) -
    const auto dev = RunDevicePendulum(context, host, revolute_link,
                                       contact_point, contact_normal);
    ASSERT_EQ(dev.dof, 1u) << "pendulum must have exactly one DOF";

    // Robustness: the device CRBA M MUST equal the hand I_a before we trust the
    // contraction. This catches a motion-subspace / inertia-frame setup error.
    std::printf("[diag] pendulum device M=%.6f (hand I_a=%.6f)  chain_J=%.6f "
                "(hand J=%.6f)  device_m_eff=%.6f\n",
                static_cast<double>(dev.M[0]), static_cast<double>(I_a),
                static_cast<double>(dev.chain_j[0]), static_cast<double>(J_hand),
                static_cast<double>(dev.device_m_eff));
    ASSERT_NEAR(dev.M[0], I_a, 1.0e-4f)
        << "device CRBA M disagrees with hand I_a; setup error";
    // M^-1 (1x1) == 1/I_a.
    ASSERT_NEAR(dev.Minv[0], 1.0f / I_a, 1.0e-5f);
    // The device chain Jacobian must match the hand J (the arbitrary point made
    // it through the builder).
    ASSERT_NEAR(dev.chain_j[0], J_hand, 1.0e-5f)
        << "device chain-J disagrees with hand J at the off-origin contact point";

    // --- The PROVIDER under test ------------------------------------------
    float qdot = 0.0f;  // live joint velocity the provider will mutate.
    constraint::ArticulationReactionState artic;
    artic.chain_jacobian = dev.chain_j.data();
    artic.inertia_M_inv = dev.Minv.data();
    artic.qdot = &qdot;
    artic.dof_stride = dev.dof;

    constraint::ReactionProvider provider;
    provider.views.articulation = &artic;
    provider.views.articulation_count = 1u;

    constraint::CollidableRef ref;
    ref.type = constraint::CollidableType::ArticulationLink;
    ref.react = constraint::ReactionProviderKind::ArticulationChainJ;
    ref.handle = 0u;  // index into the resolved articulation binding array.

    // (RowJacobian6 is ignored by the articulation path -- it uses the resolved
    // chain row -- so pass a sentinel to prove it is not consulted.)
    constraint::RowJacobian6 sentinel;
    sentinel.linear = Vec3{123.0f, 456.0f, 789.0f};
    sentinel.angular = Vec3{-1.0f, -2.0f, -3.0f};

    const float eff = provider.effective_inv_mass(ref, sentinel);
    EXPECT_NEAR(eff, expected_eff, 1.0e-5f)
        << "articulation effective_inv_mass != hand J^2/I_a";
    // Also equals 1/device_m_eff (the production kernel reciprocates this term).
    EXPECT_NEAR(eff, 1.0f / dev.device_m_eff, 1.0e-4f)
        << "effective_inv_mass != 1/(production m_eff)";

    provider.apply_impulse(ref, sentinel, dlambda);
    EXPECT_NEAR(qdot, expected_dqdot, 1.0e-5f)
        << "articulation apply_impulse != hand (J/I_a)*dl";

    std::printf("[diag] artic eff_invM=%.8f (hand %.8f)  dqdot=%.8f (hand %.8f)\n",
                static_cast<double>(eff), static_cast<double>(expected_eff),
                static_cast<double>(qdot), static_cast<double>(expected_dqdot));
}

// ===========================================================================
// D1: 2-run byte-identity of every effective_inv_mass AND every applied delta.
// ===========================================================================
TEST(ReactionProviders, DeterministicAcrossTwoRuns) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // Build the pendulum + device M^-1/chain-J once (the device build is the only
    // potential nondeterminism source; the providers themselves are pure host
    // arithmetic, so we re-run the whole device pipeline twice and compare).
    const Vec3 axis{1.0f, 0.0f, 0.0f};
    const Vec3 diagonal_inertia{7.0f, 11.0f, 13.0f};
    const float mass = 2.0f;
    auto host = BuildRevolutePendulum(axis, diagonal_inertia, mass);
    const Vec3 contact_point{0.0f, 0.4f, 0.9f};
    const Vec3 contact_normal{0.0f, 0.0f, 1.0f};

    auto run_once = [&]() {
        const auto dev = RunDevicePendulum(context, host, 1u, contact_point,
                                           contact_normal);
        struct Outcome {
            float rigid_eff;
            Vec3 rigid_dv;
            Vec3 rigid_dw;
            float particle_eff;
            Vec3 particle_dv;
            float artic_eff;
            float artic_dqdot;
            std::vector<float> chain_j;
            std::vector<float> minv;
        } o{};
        o.chain_j = dev.chain_j;
        o.minv = dev.Minv;

        // Rigid.
        Vec3 lv = Vec3::Zero();
        Vec3 av = Vec3::Zero();
        constraint::RigidReactionState rigid{0.25f, Vec3{0.5f, 1.0f / 3.0f, 0.2f},
                                             &lv, &av};
        constraint::ReactionProvider rp;
        rp.views.rigid = &rigid;
        rp.views.rigid_count = 1u;
        constraint::CollidableRef rref;
        rref.react = constraint::ReactionProviderKind::RigidInvMass;
        rref.handle = 0u;
        constraint::RowJacobian6 j;
        j.linear = Vec3{0.6f, -0.8f, 0.0f};
        j.angular = Vec3{0.3f, 0.5f, -0.7f};
        o.rigid_eff = rp.effective_inv_mass(rref, j);
        rp.apply_impulse(rref, j, 1.7f);
        o.rigid_dv = lv;
        o.rigid_dw = av;

        // Particle.
        Vec3 pv = Vec3::Zero();
        constraint::ParticleReactionState particle{2.0f, &pv};
        constraint::ReactionProvider pp;
        pp.views.particle = &particle;
        pp.views.particle_count = 1u;
        constraint::CollidableRef pref;
        pref.react = constraint::ReactionProviderKind::ParticleInvMass;
        pref.handle = 0u;
        constraint::RowJacobian6 pj;
        pj.linear = Vec3{0.0f, 0.0f, 1.0f};
        o.particle_eff = pp.effective_inv_mass(pref, pj);
        pp.apply_impulse(pref, pj, 3.0f);
        o.particle_dv = pv;

        // Articulation.
        float qdot = 0.0f;
        constraint::ArticulationReactionState artic{dev.chain_j.data(),
                                                    dev.Minv.data(), &qdot, dev.dof};
        constraint::ReactionProvider ap;
        ap.views.articulation = &artic;
        ap.views.articulation_count = 1u;
        constraint::CollidableRef aref;
        aref.react = constraint::ReactionProviderKind::ArticulationChainJ;
        aref.handle = 0u;
        constraint::RowJacobian6 dummy;
        o.artic_eff = ap.effective_inv_mass(aref, dummy);
        ap.apply_impulse(aref, dummy, 2.5f);
        o.artic_dqdot = qdot;
        return o;
    };

    const auto a = run_once();
    const auto b = run_once();

    EXPECT_EQ(a.rigid_eff, b.rigid_eff);
    EXPECT_EQ(a.rigid_dv, b.rigid_dv);
    EXPECT_EQ(a.rigid_dw, b.rigid_dw);
    EXPECT_EQ(a.particle_eff, b.particle_eff);
    EXPECT_EQ(a.particle_dv, b.particle_dv);
    EXPECT_EQ(a.artic_eff, b.artic_eff);
    EXPECT_EQ(a.artic_dqdot, b.artic_dqdot);
    // The device build feeding the articulation provider must also be byte-exact.
    ASSERT_EQ(a.chain_j.size(), b.chain_j.size());
    for (size_t i = 0u; i < a.chain_j.size(); ++i) {
        EXPECT_EQ(a.chain_j[i], b.chain_j[i]) << "chain-J not bit-identical at " << i;
    }
    ASSERT_EQ(a.minv.size(), b.minv.size());
    for (size_t i = 0u; i < a.minv.size(); ++i) {
        EXPECT_EQ(a.minv[i], b.minv[i]) << "M^-1 not bit-identical at " << i;
    }
}
