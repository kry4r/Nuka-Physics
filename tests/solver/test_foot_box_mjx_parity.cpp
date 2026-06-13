// ---------------------------------------------------------------------------
// v0.8 W1b -- the CO-RESIDENT FREE-BOX foot-contact STEP-0 MJX-parity capstone.
// ---------------------------------------------------------------------------
// THE BOX-SIDE REALITY ANCHOR TO MUJOCO. The unified spine's co-resident two-way
// contact (an articulation foot <-> a movable rigid box) was validated by
// exact-solve oracles + vs the production kernel + hand-derivation, but the BOX
// side had NEVER been anchored against MuJoCo -- only against legacy-Nuka. This
// test closes that: at a FIXED penetrating contact state (a go2 FL foot pressed
// 4 mm into a FREE rigid box, qvel=0, all tau=0), ONE mjx.forward gives the
// ground-truth qacc for ALL DOFs -- the articulation base 6 + legs 12 AND the
// box's 6 free DOFs -- and Nuka's unified pipeline (foot<->box ->
// EmitCompliantContactRows -> UnifiedSolve) must reproduce:
//   PRIMARY   : the box 6-DOF qacc (3 linear + 3 angular) vs MJX.
//   SECONDARY : the base-6 qacc vs MJX (already anchored by C5c-2 vs a STATIC
//               ground; here confirmed it stays anchored with the box present).
// This is the ONLY end-to-end MuJoCo anchor for the co-resident scene.
//
// WHY THE BOX qacc IS A SHARP GATE (the angular component is the grasp signal).
// The box is a FREE body; with no contact its qacc is pure gravity (lin (0,0,
// -9.81), ang 0). The contact adds: (a) a downward LINEAR accel = f_n/m_box
// (the box is pushed away from the foot above it), and (b) -- crucially -- an
// ANGULAR accel from the OFF-COM (x-lever) contact: alpha = (r x n) . I^-1 . f.
// MJX's box angular qacc is ~-4.0 rad/s^2 about y. A WRONG box-side reaction
// (missing the box angular Jacobian r x n, a wrong box inertia, or a dropped box
// term) shows up directly in the box qacc -> the gate BITES (demonstrated below
// by an inertia-perturbation bite-check that drives the error >> tolerance).
//
// MODEL ALIGNMENT (the task's #1 risk -- box mass+inertia). The MJX golden is a
// SOURCE TRANSLATION of the SAME go2_float.usda Nuka cooks (generator --mode
// floating-foot-box-contact-step), so the ARTICULATION mass matrix M is shared by
// construction (the C5c-2 rationale). The BOX is pinned: the MJCF sets an EXPLICIT
// <inertial mass=BOX_MASS diaginertia=I> as the SOLE inertia source (the box geom
// is mass=0 density=0 geometry), and this test duplicates BOX_MASS / BOX_HALF_
// EXTENTS / the derived diagonal inertia EXACTLY. The golden RECORDS the box mass
// + inertia MJX actually built; GATE 1 asserts our pinned values equal them, so an
// inertia mismatch is caught BEFORE the qacc compare (a qacc miss from an inertial
// mismatch is not a real bug -- the alignment must be exact).
//
// THE VELOCITY-IMPULSE <-> ACCELERATION IDENTITY (the C5c-2 seed). MJX solves
// contact at the ACCELERATION level (its force counteracts BOTH the penetration
// aref AND the free into-contact acceleration). A zero-velocity seed drops the
// latter. So we seed every DOF with its unconstrained one-step velocity
// qdot_seed = qvel0 + a_free*dt (qvel0=0): the articulation seed is the free
// floating-base ABA qacc * dt (flat-18), the BOX seed is gravity*dt (lin (0,0,
// g*dt), ang 0). After the solve, qacc = qdot_post/dt for BOTH the articulation
// (-> base-6) and the box (-> box-6).
//
// SINGLE-STEP, not a rollout (two contact solvers diverge within steps; one
// forward step at a fixed state is the only well-defined MJX<->Nuka comparison).
// VALIDATED-NOT-WIRED: no production stepper/integrator/solver/golden is touched.
// New golden go2_foot_box_contact_step.bin (generated, NOT hand-edited).
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"
#include "collision/candidate_pair.hpp"
#include "collision/contact_stream_driver.hpp"
#include "collision/link_aabb.hpp"
#include "collision/rigid_candidate_pairs.hpp"
#include "constraint/contact_manifold.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "constraint/row_builder.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/buffer_transfer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/rigid/body_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "solver/solver_config.hpp"  // SolverConfig (iters/dt knobs; legacy UnifiedSolve dropped M9 T11)

#include "foot_chain_jacobian.hpp"  // shared FK-refresh-correct ComputeFootChainJ18

#include <gtest/gtest.h>

#include "nk_solve_harness.hpp"  // M9 T11: the ONLY solve path -- nk SolveRowsBlockIsland

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace amf = nuka::collision::amf;
using nuka::collision::AABB;
using nuka::collision::BuildContactManifolds;
using nuka::collision::CandidatePair;
using nuka::collision::LinkShapeAabbs;
using nuka::collision::ResolvedShape;
using nuka::collision::ShapeResolver;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ContactRowSides;
using nuka::constraint::ReactionProviderKind;
using nuka::constraint::RowArticulationRefs;
using nuka::constraint::RowArticulationSide;
using nuka::constraint::RowBuffers;
using nuka::constraint::RowJacobian6;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;
using nuka::scene::ShapeType;

constexpr uint32_t kInvalidLink = ~0u;
constexpr uint32_t kDof = 18u;  // 6-DOF floating base + 12 revolute legs.
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;  // MUST match the generator's model.opt.timestep.
constexpr uint32_t kBoxRigidBodyId = 9000u;  // distinct from every go2 link body id.

// ---------------------------------------------------------------------------
// THE BOX (pinned EXACTLY to the generator's BOX_MASS / BOX_HALF_EXTENTS). A solid
// box of half-extents h, mass m has diagonal inertia
//   I = (m/3)*(h_y^2+h_z^2, h_x^2+h_z^2, h_x^2+h_y^2).
// ASYMMETRIC half-extents -> three DISTINCT principal inertias (so a wrong-axis
// box inertia or a dropped box angular Jacobian shows up in the box qacc). GATE 1
// asserts these equal what the golden recorded MJX building. NOT hand-faked.
//
// SCOPE: the box diagonal inv_inertia is compared in the BOX BODY FRAME against
// MJX's body-frame angular qacc; this is valid here ONLY because the box quat is
// IDENTITY (body frame == world frame). A rotated box would need frame-aware
// inertia handling (R I^-1 R^T) -- out of W1b scope (the contact state pins the
// box at identity orientation).
// ---------------------------------------------------------------------------
constexpr float kBoxMass = 2.0f;
constexpr Vec3 kBoxHalf{0.06f, 0.05f, 0.04f};
constexpr float kBoxInvMass = 1.0f / kBoxMass;
Vec3 BoxDiagInertia() {
    return Vec3{(kBoxMass / 3.0f) * (kBoxHalf.y * kBoxHalf.y + kBoxHalf.z * kBoxHalf.z),
                (kBoxMass / 3.0f) * (kBoxHalf.x * kBoxHalf.x + kBoxHalf.z * kBoxHalf.z),
                (kBoxMass / 3.0f) * (kBoxHalf.x * kBoxHalf.x + kBoxHalf.y * kBoxHalf.y)};
}
Vec3 BoxInvInertia() {
    const Vec3 i = BoxDiagInertia();
    return Vec3{1.0f / i.x, 1.0f / i.y, 1.0f / i.z};
}

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}
std::filesystem::path GoldenPath(const char* filename) {
    if (const char* dir = std::getenv("NUKA_GOLDEN_DIR")) {
        if (dir[0] != '\0') return std::filesystem::path(dir) / filename;
    }
    return SourcePath("tests/oracle/golden") / filename;
}

// ---------------------------------------------------------------------------
// Bespoke loader for the W1b foot-box-contact-step golden (magic "NUKAFBOX").
// Layout mirrors generate_mjx_golden.py generate_foot_box_contact_step.
// ---------------------------------------------------------------------------
struct FootBoxConfig {
    std::vector<float> qpos;     // nq (base 7 + legs 12 + box 7)
    std::vector<float> qvel;     // nv (=0)
    std::vector<float> tau;      // nv (=0)
    std::vector<float> qacc_on;  // nv
    std::vector<float> qacc_off; // nv
    uint32_t base_dof = 0u;      // go2 base freejoint qvel start (0)
    uint32_t box_dof = 0u;       // box freejoint qvel start
    std::array<float, 3> box_pos{};
    std::array<float, 4> box_quat{};  // w-first
    float box_mass = 0.0f;
    std::array<float, 3> box_inertia{};  // MJX-built diagonal inertia
    uint32_t ncon = 0u;
    std::vector<std::array<float, 7>> contacts;  // [pos(3), normal(3), dist]
    std::vector<float> normal_force;
    std::vector<float> invweight;
};
struct FootBoxGolden {
    uint32_t nq = 0u, nv = 0u, max_contacts = 0u;
    std::string model_name;
    std::vector<FootBoxConfig> configs;
};

uint32_t ReadU32(std::ifstream& in) {
    uint32_t v = 0u;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("foot-box golden: truncated header");
    return v;
}
float ReadF32(const std::vector<float>& buf, size_t& off) { return buf[off++]; }

FootBoxGolden LoadFootBoxGolden(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open foot-box golden: " + path.string());
    std::array<char, 8> magic{};
    in.read(magic.data(), 8);
    if (!in || std::string_view(magic.data(), 8) != "NUKAFBOX")
        throw std::runtime_error("bad foot-box golden magic: " + path.string());
    const uint32_t version = ReadU32(in);
    if (version != 1u) throw std::runtime_error("unsupported foot-box golden version");
    FootBoxGolden g;
    const uint32_t config_count = ReadU32(in);
    g.nq = ReadU32(in);
    g.nv = ReadU32(in);
    g.max_contacts = ReadU32(in);
    const uint32_t name_len = ReadU32(in);
    g.model_name.resize(name_len);
    if (name_len > 0u) in.read(g.model_name.data(), name_len);

    // Per-record float layout (must match the generator's np.concatenate order):
    //   qpos(nq), qvel(nv), tau(nv), qacc_on(nv), qacc_off(nv),
    //   base_dof, box_dof (2), box_pos(3), box_quat(4), box_mass(1),
    //   box_inertia(3), ncon(1), contacts(maxc*7), normal_force(maxc),
    //   invweight(maxc).
    const size_t rec_floats = static_cast<size_t>(g.nq) + g.nv * 4u + 2u + 3u + 4u +
                              1u + 3u + 1u + static_cast<size_t>(g.max_contacts) * 7u +
                              g.max_contacts + g.max_contacts;
    for (uint32_t c = 0u; c < config_count; ++c) {
        std::vector<float> rec(rec_floats);
        in.read(reinterpret_cast<char*>(rec.data()),
                static_cast<std::streamsize>(rec_floats * sizeof(float)));
        if (!in) throw std::runtime_error("foot-box golden: truncated record");
        size_t off = 0u;
        FootBoxConfig cfg;
        cfg.qpos.assign(rec.begin(), rec.begin() + g.nq); off += g.nq;
        cfg.qvel.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.tau.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.qacc_on.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.qacc_off.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.base_dof = static_cast<uint32_t>(ReadF32(rec, off));
        cfg.box_dof = static_cast<uint32_t>(ReadF32(rec, off));
        for (uint32_t j = 0u; j < 3u; ++j) cfg.box_pos[j] = ReadF32(rec, off);
        for (uint32_t j = 0u; j < 4u; ++j) cfg.box_quat[j] = ReadF32(rec, off);
        cfg.box_mass = ReadF32(rec, off);
        for (uint32_t j = 0u; j < 3u; ++j) cfg.box_inertia[j] = ReadF32(rec, off);
        cfg.ncon = static_cast<uint32_t>(ReadF32(rec, off));
        cfg.contacts.resize(g.max_contacts);
        for (uint32_t k = 0u; k < g.max_contacts; ++k)
            for (uint32_t j = 0u; j < 7u; ++j) cfg.contacts[k][j] = ReadF32(rec, off);
        cfg.normal_force.resize(g.max_contacts);
        for (uint32_t k = 0u; k < g.max_contacts; ++k) cfg.normal_force[k] = ReadF32(rec, off);
        cfg.invweight.resize(g.max_contacts);
        for (uint32_t k = 0u; k < g.max_contacts; ++k) cfg.invweight[k] = ReadF32(rec, off);
        g.configs.push_back(std::move(cfg));
    }
    return g;
}

// ---------------------------------------------------------------------------
// Quaternion helpers (mirror featherstone_oracle_harness.cpp -- the validated
// floating-base frame bridge). v' = R(q) v (w-first Hamilton).
// ---------------------------------------------------------------------------
Vec3 RotateByQuat(const Quat& q, const Vec3& v) {
    const float n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    Quat u = q;
    if (n2 > 1.0e-12f) {
        const float inv = 1.0f / std::sqrt(n2);
        u.w *= inv; u.x *= inv; u.y *= inv; u.z *= inv;
    }
    const Vec3 qv{u.x, u.y, u.z};
    const auto cross = [](const Vec3& a, const Vec3& b) {
        return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    };
    const Vec3 t{2.0f * cross(qv, v).x, 2.0f * cross(qv, v).y, 2.0f * cross(qv, v).z};
    const Vec3 c = cross(qv, t);
    return Vec3{v.x + u.w * t.x + c.x, v.y + u.w * t.y + c.y, v.z + u.w * t.z + c.z};
}
Vec3 RotateByQuatInverse(const Quat& q, const Vec3& v) {
    return RotateByQuat(Quat{q.w, -q.x, -q.y, -q.z}, v);
}

// Nuka base spatial 6-vector [omega(0:3) body; v_lin(3:6) body] -> MJX free-joint
// convention [v_lin(0:3) WORLD, omega(3:6) BODY] (the omega x v term vanishes at
// zero base velocity). Identical to the C5c-2 conversion.
std::array<float, 6> NukaSpatialToMjx(const float spatial[6], const Quat& base_rot,
                                      const Vec3& omega_body, const Vec3& v_lin_body) {
    const Vec3 omega_dot{spatial[0], spatial[1], spatial[2]};
    const Vec3 a_lin_body_spatial{spatial[3], spatial[4], spatial[5]};
    const Vec3 omega_cross_v{
        omega_body.y * v_lin_body.z - omega_body.z * v_lin_body.y,
        omega_body.z * v_lin_body.x - omega_body.x * v_lin_body.z,
        omega_body.x * v_lin_body.y - omega_body.y * v_lin_body.x};
    const Vec3 a_lin_body{a_lin_body_spatial.x + omega_cross_v.x,
                          a_lin_body_spatial.y + omega_cross_v.y,
                          a_lin_body_spatial.z + omega_cross_v.z};
    const Vec3 a_lin_world = RotateByQuat(base_rot, a_lin_body);
    return {a_lin_world.x, a_lin_world.y, a_lin_world.z,
            omega_dot.x, omega_dot.y, omega_dot.z};
}

// ---------------------------------------------------------------------------
// Cooked go2_float (reuses the C5c-1/co-residence cook). feet + shapes + mass.
// ---------------------------------------------------------------------------
struct CookedFloat {
    articulation::ArticulationHostState host;
    nuka::scene::CookedShapeTable shapes;
    std::vector<articulation::FootShape> feet;
    std::vector<float> link_mass;
};

CookedFloat CookGo2Float() {
    const auto scene = nuka::import::LoadUsd(SourcePath("examples/scenes/go2_float.usda").string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    CookedFloat result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    result.shapes = world.template_view.shape_table;
    const uint32_t link_count = result.host.TotalLinkCount();
    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link)
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    const auto& shapes = world.template_view.shape_table;
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

std::vector<Transform> ForwardKinematics(const nuka::phi::DeviceContext& context,
                                         const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

Vec3 FootWorldCenter(const std::vector<Transform>& poses, const articulation::FootShape& foot) {
    const Transform calf = poses[foot.calf_local_link];
    return calf.position + calf.rotation.Rotate(foot.local_offset);
}

// 18x18 M^-1 via the production CRBA (reuse, not hand-rolled).
std::vector<float> ComputeInverseInertia18(const nuka::phi::DeviceContext& context,
                                           const articulation::ArticulationHostState& host) {
    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    const uint32_t link_count = host.TotalLinkCount();
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, kGravityZ);
    nuka::phi::Buffer composite(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia),
        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m(static_cast<size_t>(kDof) * kDof * sizeof(float), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m_inv(static_cast<size_t>(kDof) * kDof * sizeof(float), nuka::phi::MemoryKind::Device);
    articulation::ComputeArticulationInertiaM(
        context, view, kDof,
        static_cast<articulation::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m.Data()));
    articulation::FactorArticulationInertiaM(
        context, view, kDof, static_cast<const float*>(m.Data()),
        static_cast<float*>(m_inv.Data()));
    context.stream.Synchronize();
    std::vector<float> out(static_cast<size_t>(kDof) * kDof);
    m_inv.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

amf::PrimParams MakeFootPrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;
    return p;
}
amf::PrimParams MakeBoxPrim(const Vec3& center, const Vec3& half) {
    amf::PrimParams p;
    p.half_extents = half;
    p.frame.t = center;  // identity frame -> axis-aligned box (box_quat is identity).
    return p;
}
AABB BoxWorldAabb(const Vec3& center, const Vec3& half) {
    AABB box;
    box.min = center - half;
    box.max = center + half;
    return box;
}
CollidableRef MakeFootRef(uint32_t calf_link) {
    CollidableRef ref;
    ref.type = CollidableType::ArticulationLink;
    ref.react = ReactionProviderKind::ArticulationChainJ;
    ref.handle = calf_link;
    return ref;
}

// Free floating-base ABA at the host state, returning the flat-18 a_free in Nuka
// native packing [omega_dot body(0:3); a_lin body(3:6); leg qddot(6:17)]. Mirrors
// the C5c-2 ComputeFreeAba.
struct FreeAba {
    std::array<float, 6> base_spatial{};
    std::vector<float> leg_qddot;
    std::array<float, kDof> flat{};
};
FreeAba ComputeFreeAba(const nuka::phi::DeviceContext& context,
                       articulation::ArticulationHostState host,  // by value.
                       const Quat& base_rot) {
    const uint32_t root = host.articulation_link_offset[0];
    auto device = articulation::UploadArticulationState(context, host);
    articulation::FeatherstoneAba::ComputeAccelerations(context, device.View(), kGravityZ);
    context.stream.Synchronize();
    articulation::DownloadArticulationState(device, &host);
    FreeAba out;
    const uint32_t link_count = host.articulation_link_count[0];
    const uint32_t leg_dofs = link_count - 1u;
    out.leg_qddot.resize(leg_dofs);
    for (uint32_t i = 0u; i < leg_dofs; ++i) out.leg_qddot[i] = host.qddot[root + 1u + i];
    float a0[6];
    for (uint32_t i = 0u; i < 6u; ++i) a0[i] = host.link_acceleration[root].a[i];
    const Vec3 g_world{0.0f, 0.0f, -kGravityZ};  // GravityAcceleration: out[5] = -g_z.
    const Vec3 g_body = RotateByQuatInverse(base_rot, g_world);
    out.base_spatial = {a0[0], a0[1], a0[2], a0[3] - g_body.x, a0[4] - g_body.y, a0[5] - g_body.z};
    for (uint32_t i = 0u; i < 6u; ++i) out.flat[i] = out.base_spatial[i];
    for (uint32_t i = 0u; i < leg_dofs && (6u + i) < kDof; ++i) out.flat[6u + i] = out.leg_qddot[i];
    return out;
}

// ---------------------------------------------------------------------------
// One config's full unified-pipeline result (the foot<->box co-resident solve).
// ---------------------------------------------------------------------------
struct ParityResult {
    bool pair_found = false;
    uint32_t row_count = 0u;
    float contact_depth = 0.0f;
    Vec3 contact_normal{}, contact_point{};
    std::array<float, 6> base6_mjx{};       // Nuka qacc_total base-6 in MJX frame.
    std::array<float, 6> base6_free_mjx{};  // free-ABA base-6 (GATE 0).
    std::array<float, 6> box6_mjx{};        // box qacc [lin world(3); ang body(3)].
    std::array<float, 6> box6_free_mjx{};   // free box qacc (gravity only).
    std::array<float, kDof> qacc_total{};   // articulation qacc (NUKA flat packing).
    float lambda = 0.0f, foot_force = 0.0f;
    double exact_lambda = 0.0;              // DOUBLE-precision (A+R) solve.
    Vec3 box_lin_after{}, box_ang_after{};
    BodyState box_state{};
};

// Build the foot<->box co-resident state from the golden, then:
//   free ABA -> seed qdot = a_free*dt; box seed = gravity*dt; solve; qacc=dv/dt.
// box_inertia_scale lets the bite-check perturb the box inertia (default 1.0).
ParityResult RunParity(const nuka::phi::DeviceContext& context, CookedFloat cooked,
                       const FootBoxGolden& g, const FootBoxConfig& cfg,
                       const nuka::solver::SolverConfig& config,
                       float box_inertia_scale = 1.0f) {
    ParityResult result;
    auto& host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t link_count = host.articulation_link_count[0];
    const uint32_t leg_dofs = link_count - 1u;

    // --- Set the host state from the golden (base = identity quat in qpos). -----
    Quat base_rot{cfg.qpos[3], cfg.qpos[4], cfg.qpos[5], cfg.qpos[6]};
    {
        const float n = std::sqrt(base_rot.w * base_rot.w + base_rot.x * base_rot.x +
                                  base_rot.y * base_rot.y + base_rot.z * base_rot.z);
        const float inv = n > 0.0f ? 1.0f / n : 1.0f;
        base_rot.w *= inv; base_rot.x *= inv; base_rot.y *= inv; base_rot.z *= inv;
    }
    host.base_pose[0].position = Vec3{cfg.qpos[0], cfg.qpos[1], cfg.qpos[2]};
    host.base_pose[0].rotation = base_rot;
    for (auto& v : host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (uint32_t i = 0u; i < leg_dofs; ++i) {
        host.q[root + 1u + i] = cfg.qpos[7u + i];
        host.qdot[root + 1u + i] = 0.0f;
        host.tau[root + 1u + i] = cfg.tau[6u + i];  // all zero for W1b.
    }
    host.q[root] = 0.0f; host.qdot[root] = 0.0f; host.tau[root] = 0.0f;

    const Vec3 omega_body{0.0f, 0.0f, 0.0f}, v_lin_body{0.0f, 0.0f, 0.0f};

    // --- Free ABA (a_free + GATE-0 base-6); free box qacc = gravity only. -------
    const FreeAba afree = ComputeFreeAba(context, host, base_rot);
    result.base6_free_mjx = NukaSpatialToMjx(afree.base_spatial.data(), base_rot, omega_body, v_lin_body);
    result.box6_free_mjx = {0.0f, 0.0f, kGravityZ, 0.0f, 0.0f, 0.0f};

    // --- FK -> the contacting foot (FL = foot[0]); the box from the golden. -----
    const auto poses = ForwardKinematics(context, host);
    const std::vector<float> minv = ComputeInverseInertia18(context, host);
    const articulation::FootShape& foot = cooked.feet[0];
    const Vec3 foot_center = FootWorldCenter(poses, foot);
    const Vec3 box_center{cfg.box_pos[0], cfg.box_pos[1], cfg.box_pos[2]};

    // --- (1) per-link AABBs + the box AABB -> (2) cross-type broadphase. --------
    const LinkShapeAabbs links =
        nuka::collision::ExtractLinkShapeAabbs(cooked.shapes, poses, host.link_body);
    std::vector<uint32_t> link_contypes(links.aabbs.size(), 1u);
    std::vector<uint32_t> link_conaff(links.aabbs.size(), 1u);
    std::vector<AABB> rigid_aabbs = {BoxWorldAabb(box_center, kBoxHalf)};
    std::vector<uint32_t> rigid_body_ids = {kBoxRigidBodyId};
    std::vector<uint32_t> rigid_contypes = {1u}, rigid_conaff = {1u};
    nuka::phi::Buffer d_rigid = nuka::phi::UploadVector(rigid_aabbs);
    const AABB* d_rigid_ptr = static_cast<const AABB*>(d_rigid.Data());
    auto stream = nuka::collision::BuildArticulationRigidCandidatePairs(
        context, d_rigid_ptr, static_cast<uint32_t>(rigid_aabbs.size()),
        rigid_body_ids.data(), rigid_contypes.data(), rigid_conaff.data(),
        links, link_contypes.data(), link_conaff.data(), /*excluded_body_pairs=*/{});
    const auto pairs = stream.DownloadPairs();

    CandidatePair foot_box_pair;
    for (const auto& p : pairs) {
        const bool a_link = p.a.type == CollidableType::ArticulationLink &&
                            p.a.handle == foot.calf_local_link;
        const bool b_box = p.b.type == CollidableType::RigidBody && p.b.handle == kBoxRigidBodyId;
        const bool a_box = p.a.type == CollidableType::RigidBody && p.a.handle == kBoxRigidBodyId;
        const bool b_link = p.b.type == CollidableType::ArticulationLink &&
                            p.b.handle == foot.calf_local_link;
        if ((a_link && b_box) || (a_box && b_link)) { foot_box_pair = p; result.pair_found = true; break; }
    }
    if (!result.pair_found) return result;

    // --- (3) narrowphase: drive ONLY (foot, box) -> exactly one manifold. -------
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::ArticulationLink && ref.handle == foot.calf_local_link) {
            out->type = ShapeType::Sphere; out->prim = MakeFootPrim(foot_center, foot.radius); return true;
        }
        if (ref.type == CollidableType::RigidBody && ref.handle == kBoxRigidBodyId) {
            out->type = ShapeType::Box; out->prim = MakeBoxPrim(box_center, kBoxHalf); return true;
        }
        return false;
    };
    std::vector<CandidatePair> drive_pairs = {foot_box_pair};
    std::vector<ContactManifold> manifolds;
    BuildContactManifolds(drive_pairs, resolve, &manifolds);
    if (manifolds.size() != 1u || manifolds[0].point_count == 0u) return result;
    result.contact_depth = manifolds[0].points[0].penetration;
    result.contact_normal = manifolds[0].points[0].normal;
    result.contact_point = manifolds[0].points[0].position;

    // --- (4) compliant row + sides (condim=1). invweight pinned to MJX's so R
    // matches MuJoCo exactly (the C5c-2 isolation of the recoil coupling). -------
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = cfg.invweight.empty() ? 1.0f : cfg.invweight[0];
    inputs.dt = kDt;
    inputs.condim = 1u;
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    result.row_count = rows.RowCount();
    if (rows.RowCount() != 1u || sides.size() != 1u) return result;

    const ContactRowSides& s = sides[0];
    int box_local = -1, foot_local = -1;
    if (s.a.react == ReactionProviderKind::ArticulationChainJ &&
        s.b.react == ReactionProviderKind::RigidInvMass) { foot_local = 0; box_local = 1; }
    else if (s.b.react == ReactionProviderKind::ArticulationChainJ &&
             s.a.react == ReactionProviderKind::RigidInvMass) { foot_local = 1; box_local = 0; }
    if (foot_local < 0 || box_local < 0) return result;

    const RowJacobian6 j_foot = rows.JacobianForRowBody(0u, static_cast<uint32_t>(foot_local));
    RowJacobian6 j_box = rows.JacobianForRowBody(0u, static_cast<uint32_t>(box_local));
    // The unified solver fills the box angular Jacobian r x n IN-KERNEL; mirror it
    // here for the exact-solve oracle (A_box includes (r x n).I^-1.(r x n)).
    j_box.angular = (result.contact_point - box_center).Cross(j_box.linear);

    // --- the 18-wide foot chain-J on the foot's contact normal. -----------------
    const Vec3 foot_normal = j_foot.linear;
    const std::vector<float> chain_j = nuka::test::ComputeFootChainJ18(
        context, host, poses, foot.calf_local_link, result.contact_point, foot_normal, kDof);
    std::vector<float> chain_jacobians = chain_j;

    // --- art_refs + body_indices (box -> BodyState 0; foot -> coloring key). -----
    const uint32_t kArtIndex = 0u, body_count = 1u;
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    rows.body_indices[2u * 0u + static_cast<uint32_t>(box_local)] = 0u;
    rows.body_indices[2u * 0u + static_cast<uint32_t>(foot_local)] = body_count + kArtIndex;
    RowArticulationSide foot_side{kArtIndex, 0u}, none_side{};
    art_refs[0].a = (foot_local == 0) ? foot_side : none_side;
    art_refs[0].b = (foot_local == 1) ? foot_side : none_side;

    // --- THE SEED: unconstrained one-step velocity qdot = a_free*dt (artic) AND
    // the box at gravity*dt (its free one-step velocity). qvel0 = 0 both sides. ---
    std::vector<float> qdot(kDof, 0.0f);
    for (uint32_t i = 0u; i < kDof; ++i) qdot[i] = afree.flat[i] * kDt;

    BodyState box;
    box.inv_mass = kBoxInvMass;
    const Vec3 inv_inertia = BoxInvInertia();
    box.inv_inertia = Vec3{inv_inertia.x / box_inertia_scale, inv_inertia.y / box_inertia_scale,
                           inv_inertia.z / box_inertia_scale};
    box.position = box_center;
    box.orientation = Quat::Identity();
    box.linear_velocity = Vec3{0.0f, 0.0f, kGravityZ * kDt};  // free box one-step seed.
    box.angular_velocity = Vec3::Zero();
    std::vector<BodyState> bodies = {box};

    // --- PRIMARY DIAG: exact 1x1 (A+R) lambda = b solve (double). ----------------
    {
        double a_art = 0.0;
        for (uint32_t r = 0u; r < kDof; ++r) {
            double rw = 0.0;
            for (uint32_t c = 0u; c < kDof; ++c)
                rw += static_cast<double>(minv[r * kDof + c]) * static_cast<double>(chain_j[c]);
            a_art += static_cast<double>(chain_j[r]) * rw;
        }
        double a_box = static_cast<double>(box.inv_mass) *
                       (static_cast<double>(j_box.linear.x) * j_box.linear.x +
                        static_cast<double>(j_box.linear.y) * j_box.linear.y +
                        static_cast<double>(j_box.linear.z) * j_box.linear.z);
        a_box += static_cast<double>(j_box.angular.x) * j_box.angular.x * box.inv_inertia.x;
        a_box += static_cast<double>(j_box.angular.y) * j_box.angular.y * box.inv_inertia.y;
        a_box += static_cast<double>(j_box.angular.z) * j_box.angular.z * box.inv_inertia.z;
        const double A = a_art + a_box;
        const double R = static_cast<double>(rows.rows[0].compliance_alpha);
        double jv = 0.0;
        for (uint32_t r = 0u; r < kDof; ++r)
            jv += static_cast<double>(chain_j[r]) * static_cast<double>(qdot[r]);
        jv += static_cast<double>(j_box.linear.x) * box.linear_velocity.x +
              static_cast<double>(j_box.linear.y) * box.linear_velocity.y +
              static_cast<double>(j_box.linear.z) * box.linear_velocity.z;
        jv += static_cast<double>(j_box.angular.x) * box.angular_velocity.x +
              static_cast<double>(j_box.angular.y) * box.angular_velocity.y +
              static_cast<double>(j_box.angular.z) * box.angular_velocity.z;
        const double b = static_cast<double>(rows.rows[0].rhs) * static_cast<double>(kDt) - jv;
        result.exact_lambda = (A + R) > 1.0e-12 ? std::max(0.0, b / (A + R)) : 0.0;
    }

    // --- (5) the unified two-way solve: M9 T11 nk-ONLY -- the SAME assembled
    // inputs run through the nk SolveRowsBlockIsland op; the SAME gates judge.
    // The legacy solver::UnifiedSolve arm (deleted in M9 T11-core) is gone --
    // this is the M4 re-point made the sole path.
    {
        const auto nk_res = nk_harness::NkSolveRows(
            rows, sides, art_refs, chain_jacobians, minv, qdot, &bodies,
            kDof, static_cast<uint16_t>(config.velocity_iterations), kDt);
        EXPECT_TRUE(nk_res.ok) << "nk solve harness failed";
        if (nk_res.ok) {
            qdot = nk_res.qdot;
            for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
                rows.rows[r].lambda = nk_res.lambda[r];
            }
        }
    }

    // --- qacc_total = qdot_post/dt (articulation -> base-6); box dv/dt (box-6). --
    for (uint32_t i = 0u; i < kDof; ++i) result.qacc_total[i] = qdot[i] / kDt;
    float base_spatial_total[6];
    for (uint32_t i = 0u; i < 6u; ++i) base_spatial_total[i] = result.qacc_total[i];
    result.base6_mjx = NukaSpatialToMjx(base_spatial_total, base_rot, omega_body, v_lin_body);

    result.box_lin_after = bodies[0].linear_velocity;
    result.box_ang_after = bodies[0].angular_velocity;
    result.box_state = bodies[0];
    // Box qacc: dv/dt over the WHOLE step (seed was at qvel0=0). The box has identity
    // orientation, so body frame == world frame: MJX's [v_lin world; omega body] maps
    // directly. The linear seed (gravity*dt) is folded into the post velocity, so
    // dv/dt is the FULL constrained qacc (matching MJX's qacc directly).
    result.box6_mjx = {bodies[0].linear_velocity.x / kDt, bodies[0].linear_velocity.y / kDt,
                       bodies[0].linear_velocity.z / kDt, bodies[0].angular_velocity.x / kDt,
                       bodies[0].angular_velocity.y / kDt, bodies[0].angular_velocity.z / kDt};

    result.lambda = rows.rows[0].lambda;
    result.foot_force = rows.rows[0].lambda / kDt;
    return result;
}

nuka::solver::SolverConfig MakeConfig(uint32_t iters) {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = iters;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}
float MaxAbs6(const std::array<float, 6>& a, const std::array<float, 6>& b) {
    float m = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
float Rel6(const std::array<float, 6>& nuka, const std::array<float, 6>& mjx, float scale = 1.0f) {
    float m = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) {
        const float den = std::max(scale, std::fabs(mjx[i]));
        m = std::max(m, std::fabs(nuka[i] - mjx[i]) / den);
    }
    return m;
}

}  // namespace

// ===========================================================================
// GATE 1 (contact-state + box-inertia alignment) + GATE 0 (free dynamics) +
// GATE 2 (PRIMARY box-6 qacc; SECONDARY base-6 qacc) vs MJX.
// ===========================================================================
TEST(FootBoxMjxParity, BoxAndBaseSixContactQaccMatchesMjx) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_box_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path))
        GTEST_SKIP() << "foot-box golden not present (run generator --mode floating-foot-box-contact-step)";

    const auto golden = LoadFootBoxGolden(golden_path);
    ASSERT_EQ(golden.nq, 26u);  // base 7 + legs 12 + box 7.
    ASSERT_EQ(golden.nv, 24u);  // base 6 + legs 12 + box 6.
    ASSERT_GT(golden.configs.size(), 0u);

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    ASSERT_EQ(cooked.host.ArticulationCount(), 1u);
    ASSERT_FALSE(cooked.feet.empty());

    const auto cfg_default = MakeConfig(10u);
    const auto cfg_high = MakeConfig(200u);

    for (size_t ci = 0u; ci < golden.configs.size(); ++ci) {
        const auto& cfg = golden.configs[ci];
        ASSERT_EQ(cfg.base_dof, 0u);
        ASSERT_EQ(cfg.box_dof, 18u);

        // ---- GATE 1a: box mass + inertia alignment (the #1 risk). The Nuka box
        // params MUST equal what MJX built from the <geom box>. A qacc miss from an
        // inertial mismatch is not a real bug -- catch it BEFORE the qacc compare.
        const Vec3 nuka_box_inertia = BoxDiagInertia();
        EXPECT_NEAR(cfg.box_mass, kBoxMass, 1.0e-6f) << "box mass mismatch Nuka vs MJX";
        EXPECT_NEAR(cfg.box_inertia[0], nuka_box_inertia.x, 1.0e-6f) << "box inertia x mismatch";
        EXPECT_NEAR(cfg.box_inertia[1], nuka_box_inertia.y, 1.0e-6f) << "box inertia y mismatch";
        EXPECT_NEAR(cfg.box_inertia[2], nuka_box_inertia.z, 1.0e-6f) << "box inertia z mismatch";
        std::printf("[diag] GATE1a box: MJX mass=%.6f inertia=(%.6f,%.6f,%.6f)  "
                    "Nuka mass=%.6f inertia=(%.6f,%.6f,%.6f)\n",
                    cfg.box_mass, cfg.box_inertia[0], cfg.box_inertia[1], cfg.box_inertia[2],
                    kBoxMass, nuka_box_inertia.x, nuka_box_inertia.y, nuka_box_inertia.z);

        // ---- GATE 1b: the contact STATE the golden recorded. MJX uses the geom1->
        // geom2 normal frame; with the box as geom2 it points DOWN into the box.
        EXPECT_EQ(cfg.ncon, 1u) << "config " << ci << " MJX ncon != 1 (single foot<->box)";
        const auto& con = cfg.contacts[0];
        EXPECT_NEAR(std::fabs(con[5]), 1.0f, 1.0e-4f) << "config " << ci << " normal not +/-Z";
        EXPECT_LT(con[6], 0.0f) << "config " << ci << " contact not penetrating (dist >= 0)";
        std::printf("[diag] GATE1b MJX contact: pos=(%.4f,%.4f,%.4f) normal=(%.2f,%.2f,%.2f) "
                    "dist=%.5f  f_n=%.4f N\n",
                    con[0], con[1], con[2], con[3], con[4], con[5], con[6], cfg.normal_force[0]);

        const auto def = RunParity(context, cooked, golden, cfg, cfg_default);
        const auto high = RunParity(context, cooked, golden, cfg, cfg_high);
        EXPECT_TRUE(high.pair_found) << "broadphase did not emit the (foot, box) pair";
        ASSERT_EQ(high.row_count, 1u) << "config " << ci << " Nuka produced != 1 contact row";

        // GATE 1b (Nuka <-> MJX cross-assert): the contact STATE Nuka assembles
        // MUST match the golden's -- a qacc match on a WRONG contact state is
        // meaningless. The MJX dist is signed-negative (penetrating); Nuka's depth
        // is positive penetration, so depth == -dist. Cross-check the contact-point
        // x,y AND the depth. We DELIBERATELY do NOT cross-assert the contact-point z:
        // Nuka places the manifold point on the box TOP FACE (box_center.z + half_z),
        // MJX at the penetration MIDPOINT ((foot_bottom + box_top)/2), so the two
        // differ by exactly penetration/2 (~2 mm) -- a pure CONVENTION difference,
        // not an error. It is provably irrelevant to the compared qacc: the box
        // torque is r x F with F purely -Z, so torque_y = -r_x*F_z drops r_z entirely
        // (this is why box angular_y matches at 6.7e-4 despite the z gap); the
        // foot-side chain-J uses the box-surface point and the 1.3e-4 base-6
        // agreement absorbs it.
        EXPECT_GT(high.contact_depth, 0.0f) << "config " << ci << " Nuka foot did not penetrate box";
        EXPECT_NEAR(high.contact_depth, -con[6], 1.0e-4f)
            << "config " << ci << " Nuka penetration depth != MJX |dist|";
        EXPECT_NEAR(high.contact_point.x, con[0], 1.0e-3f)
            << "config " << ci << " Nuka contact point x != MJX";
        EXPECT_NEAR(high.contact_point.y, con[1], 1.0e-3f)
            << "config " << ci << " Nuka contact point y != MJX";
        EXPECT_NEAR(std::fabs(high.contact_normal.z), 1.0f, 1.0e-4f)
            << "config " << ci << " Nuka contact normal not +/-Z";
        std::printf("[diag] Nuka contact: normal=(%.2f,%.2f,%.2f) point=(%.4f,%.4f,%.4f) depth=%.5f "
                    "(MJX point z=%.4f differs by ~pen/2 = box-face-vs-midpoint convention)\n",
                    high.contact_normal.x, high.contact_normal.y, high.contact_normal.z,
                    high.contact_point.x, high.contact_point.y, high.contact_point.z,
                    high.contact_depth, con[2]);

        // ---- GATE 0: free dynamics tolerance floor (contact OFF) on BOTH sides.
        std::array<float, 6> base_off{}, box_off{};
        for (uint32_t i = 0u; i < 6u; ++i) {
            base_off[i] = cfg.qacc_off[cfg.base_dof + i];
            box_off[i] = cfg.qacc_off[cfg.box_dof + i];
        }
        const float gate0_base = MaxAbs6(high.base6_free_mjx, base_off);
        const float gate0_box = MaxAbs6(high.box6_free_mjx, box_off);
        std::printf("[diag] GATE0 free base-6 vs OFF=%.3e  free box-6 vs OFF=%.3e\n",
                    gate0_base, gate0_box);
        EXPECT_LT(gate0_base, 5.0e-3f) << "free-base ABA base-6 diverges from MJX qacc_OFF";
        EXPECT_LT(gate0_box, 1.0e-4f) << "free box qacc != gravity (box model mismatch)";

        // ---- GATE 2 PRIMARY: the BOX 6-DOF qacc vs MJX. ----------------------
        std::array<float, 6> box_on{}, base_on{};
        for (uint32_t i = 0u; i < 6u; ++i) {
            box_on[i] = cfg.qacc_on[cfg.box_dof + i];
            base_on[i] = cfg.qacc_on[cfg.base_dof + i];
        }
        const float box6_def = Rel6(def.box6_mjx, box_on);
        const float box6_high = Rel6(high.box6_mjx, box_on);
        std::printf("[diag] BOX-6 qacc Nuka(high)=(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    high.box6_mjx[0], high.box6_mjx[1], high.box6_mjx[2],
                    high.box6_mjx[3], high.box6_mjx[4], high.box6_mjx[5]);
        std::printf("[diag] BOX-6 qacc MJX_ON   =(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    box_on[0], box_on[1], box_on[2], box_on[3], box_on[4], box_on[5]);
        std::printf("[diag] config %zu BOX-6 REL: default(10)=%.3e  high(200)=%.3e\n",
                    ci, box6_def, box6_high);

        // ---- GATE 2 SECONDARY: the BASE-6 qacc vs MJX (still anchored). -------
        const float base6_high = Rel6(high.base6_mjx, base_on);
        std::printf("[diag] BASE-6 qacc Nuka(high)=(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    high.base6_mjx[0], high.base6_mjx[1], high.base6_mjx[2],
                    high.base6_mjx[3], high.base6_mjx[4], high.base6_mjx[5]);
        std::printf("[diag] BASE-6 qacc MJX_ON   =(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    base_on[0], base_on[1], base_on[2], base_on[3], base_on[4], base_on[5]);
        std::printf("[diag] config %zu BASE-6 REL high(200)=%.3e\n", ci, base6_high);

        // The exact-solve corroboration (INPUT correctness, kernel-independent).
        std::printf("[diag] config %zu kernel lambda=%.6f exact (A+R)=%.6f  foot f=%.4f N "
                    "(MJX efc=%.4f N)\n",
                    ci, high.lambda, high.exact_lambda, high.foot_force, cfg.normal_force[0]);
        EXPECT_GT(high.exact_lambda, 0.0) << "the assembled (A+R) system has no positive impulse";
        const double kernel_vs_exact =
            std::fabs(static_cast<double>(high.lambda) - high.exact_lambda) /
            std::max(1.0, std::fabs(high.exact_lambda));
        EXPECT_LT(kernel_vs_exact, 1.0e-4)
            << "config " << ci << " kernel lambda diverged from the exact (A+R) solve";

        // PRIMARY box-6 bound: input-fidelity-limited (same recoil-path fidelity as
        // C5c-2's base-6, ~1e-3..1e-4). The box angular component (~-4 rad/s^2 about
        // y) is the grasp signal -- a dropped box angular Jacobian or wrong inertia
        // blows this up (the bite-check below proves it).
        EXPECT_LT(box6_high, 5.0e-3f)
            << "config " << ci << " PRIMARY: box 6-DOF qacc drifted from MJX "
               "(box-side reaction / inertia / angular-Jacobian error)";
        EXPECT_LE(box6_high, box6_def + 1.0e-3f)
            << "config " << ci << " box-6 did not converge with PGS iters (structural bug)";

        // SECONDARY base-6 bound (the C5c-2 anchor, with the box present).
        EXPECT_LT(base6_high, 2.0e-3f)
            << "config " << ci << " SECONDARY: base-6 qacc drifted from MJX with the box present";
    }
}

// ===========================================================================
// THE GATE BITES: perturbing the box inertia 5% pushes the box-6 qacc error well
// over tolerance. Proves the comparison is LOAD-BEARING on the box-side reaction
// (specifically the box inertia -> the angular qacc), not vacuously satisfied.
// (Production code is NOT touched -- the perturbation is a test-side input only.)
// ===========================================================================
TEST(FootBoxMjxParity, BoxInertiaPerturbationFailsTheGate) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_box_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path)) GTEST_SKIP() << "foot-box golden not present";

    const auto golden = LoadFootBoxGolden(golden_path);
    ASSERT_GT(golden.configs.size(), 0u);
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    ASSERT_FALSE(cooked.feet.empty());
    const auto& cfg = golden.configs[0];
    const auto config = MakeConfig(200u);

    std::array<float, 6> box_on{};
    for (uint32_t i = 0u; i < 6u; ++i) box_on[i] = cfg.qacc_on[cfg.box_dof + i];

    const auto correct = RunParity(context, cooked, golden, cfg, config, /*scale=*/1.0f);
    const auto wrong = RunParity(context, cooked, golden, cfg, config, /*scale=*/1.05f);
    const float err_correct = Rel6(correct.box6_mjx, box_on);
    const float err_wrong = Rel6(wrong.box6_mjx, box_on);
    std::printf("[diag] BITE: box-6 rel err correct=%.3e  +5%%-inertia=%.3e (tol=5e-3)\n",
                err_correct, err_wrong);
    EXPECT_LT(err_correct, 5.0e-3f) << "correct box inertia should pass the gate";
    EXPECT_GT(err_wrong, 5.0e-3f)
        << "a 5% box-inertia perturbation should FAIL the box-6 gate -- the comparison "
           "is not load-bearing on the box-side reaction (vacuous gate)";
    // The angular qacc is where the box inertia bites hardest; show it explicitly.
    std::printf("[diag] BITE: box ang qacc_y correct=%.4f  +5%%=%.4f  MJX=%.4f\n",
                correct.box6_mjx[4], wrong.box6_mjx[4], box_on[4]);
    EXPECT_GT(std::fabs(wrong.box6_mjx[4] - box_on[4]),
              std::fabs(correct.box6_mjx[4] - box_on[4]) + 1.0e-2f)
        << "the box angular qacc should diverge under a wrong box inertia";
}

// ===========================================================================
// Determinism (D1): two identical parity runs -> byte-identical box-6 + base-6.
// ===========================================================================
TEST(FootBoxMjxParity, ParityRunDeterministic) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_box_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path)) GTEST_SKIP() << "foot-box golden not present";

    const auto golden = LoadFootBoxGolden(golden_path);
    ASSERT_GT(golden.configs.size(), 0u);
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    ASSERT_FALSE(cooked.feet.empty());
    const auto config = MakeConfig(64u);

    const auto a = RunParity(context, cooked, golden, golden.configs[0], config);
    const auto b = RunParity(context, cooked, golden, golden.configs[0], config);
    EXPECT_EQ(std::memcmp(a.box6_mjx.data(), b.box6_mjx.data(), 6u * sizeof(float)), 0)
        << "two runs produced different box-6 qacc (nondeterministic)";
    EXPECT_EQ(std::memcmp(a.base6_mjx.data(), b.base6_mjx.data(), 6u * sizeof(float)), 0)
        << "two runs produced different base-6 qacc (nondeterministic)";
    EXPECT_EQ(std::memcmp(&a.box_state, &b.box_state, sizeof(BodyState)), 0)
        << "two runs produced a different box BodyState (nondeterministic)";
    std::printf("[diag] (D1) box-6 + base-6 + box BodyState byte-identical across 2 runs\n");
}

// ===========================================================================
// M4 RE-POINT (now the SOLE path, M9 T11) — the foot x box co-residence golden
// gates through the nk SolveRowsBlockIsland op (plan M4 "test_foot_box...
// (oracle 口径) 重指"). RunParity already runs ONLY the nk solve; this arm
// re-holds the MJX bounds (box-6 rel 5e-3, base-6 rel 2e-3, kernel-vs-exact
// 1e-4). The legacy nk-vs-UnifiedSolve cross-check was dropped with the class.
// ===========================================================================
TEST(FootBoxMjxParity, NkSolveRowsBlockIslandMatchesMjx) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_box_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path)) GTEST_SKIP() << "foot-box golden not present";

    const auto golden = LoadFootBoxGolden(golden_path);
    ASSERT_GT(golden.configs.size(), 0u);
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const auto cooked = CookGo2Float();
    const auto cfg_high = MakeConfig(200u);

    for (size_t ci = 0u; ci < golden.configs.size(); ++ci) {
        const auto& cfg = golden.configs[ci];
        const auto nk = RunParity(context, cooked, golden, cfg, cfg_high,
                                  /*box_inertia_scale=*/1.0f);

        std::array<float, 6> box_on{}, base_on{};
        for (uint32_t i = 0u; i < 6u; ++i) {
            box_on[i] = cfg.qacc_on[cfg.box_dof + i];
            base_on[i] = cfg.qacc_on[cfg.base_dof + i];
        }
        const float box6 = Rel6(nk.box6_mjx, box_on);
        const float base6 = Rel6(nk.base6_mjx, base_on);
        const double kernel_vs_exact =
            std::fabs(static_cast<double>(nk.lambda) - nk.exact_lambda) /
            std::max(1.0, std::fabs(nk.exact_lambda));
        std::printf("[nk re-point] config %zu BOX-6 REL=%.3e BASE-6 REL=%.3e "
                    "kernel-vs-exact=%.3e\n",
                    ci, box6, base6, kernel_vs_exact);
        EXPECT_LT(box6, 5.0e-3f) << "config " << ci << " nk box-6 drifted vs MJX";
        EXPECT_LT(base6, 2.0e-3f) << "config " << ci << " nk base-6 drifted vs MJX";
        EXPECT_LT(kernel_vs_exact, 1.0e-4)
            << "config " << ci << " nk lambda diverged from the exact (A+R) solve";
    }
}
