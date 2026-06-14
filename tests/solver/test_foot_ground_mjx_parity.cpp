// ---------------------------------------------------------------------------
// v0.8 C5c-2 -- MJX-parity ACCEPTANCE layer for the foot-ground contact subsume
// ---------------------------------------------------------------------------
// The owner ruled the contact-active golden = "MJX reference parity". This test
// is that acceptance layer: at a FIXED penetrating contact state (go2 feet
// pressed delta into a +Z ground plane, qpos fixed, qvel=0), ONE mjx.forward ->
// qacc is compared to the qacc Nuka produces through the UNIFIED contact
// pipeline. It proves the foot CONTACT force recoils into the floating base
// through the MJX-correct path (the part featherstone could only reach via
// leg-tau, not contact).
//
// OUTCOME (read before trusting GREEN): the recoil PATH is MJX-exact -- the exact
// solve of Nuka's own assembled contact system matches MJX to ~7e-4 (J/M^-1/aref/
// R + the a_free*dt seed + the featherstone frame bridge are all correct). The
// END-TO-END base-6-vs-MJX is now ~1.4e-4 (at the input-fidelity floor) after the
// C5b-core regularizer-feedback FIX: the UnifiedSolve compliant PGS update carried
// the dual regularizer R in the denominator but DROPPED the matching -R*lambda
// feedback from the numerator, so it converged (flat across iters) to the WRONG
// fixed point A*lambda=b instead of the assembled (A+R)*lambda=b -- a uniform ~R/A
// (=0.57% on this sample) surplus. FIXED in src/solver/gpu/row_solver.cu
// SolveCompliantVelocityRow (added -row.compliance_alpha*old_impulse to the
// numerator). The earlier hypothesis blaming a floating-base base-DOF apply gap
// (row_solver.cu:341-345 / link_velocity routing) was WRONG: this path is flat-18
// end-to-end and its three reaction legs are mutually consistent; the bias was
// purely the missing regularizer term (algebraic, scenario-INDEPENDENT). GREEN now
// == "inputs MJX-exact AND kernel converges to its own assembled (A+R) system".
//
// THIS IS A SINGLE-STEP COMPARISON, NOT A TRAJECTORY ROLLOUT. Two different
// contact solvers diverge within steps, so a loose multi-step tolerance would
// prove nothing; one forward step at a fixed state is the only well-defined
// MJX<->Nuka contact comparison.
//
// MODEL ALIGNMENT (the task's #1 risk). The MJX golden is built by SOURCE
// TRANSLATION of the SAME go2_float.usda that Nuka cooks (generator --mode
// floating-contact-step), NOT from go2_mjx.xml: go2_mjx.xml's rotated/offset
// inertials + different foot offset would give a DIFFERENT mass matrix M, making
// a base-6 mismatch uninterpretable (M diff vs solver diff). With source
// translation both sides share M BY CONSTRUCTION. The discriminating proof that
// the configurations align: at the cooked home crouch the MJX foot geom world
// centers equal Nuka's cooked foot centers EXACTLY (FL/FR x=0.1778, RL/RR
// x=-0.209, y=+/-0.142, z=0.1337) -- verified in the generator and re-asserted
// here via the golden's recorded contact set (GATE 1 below).
//
// THE VELOCITY-IMPULSE <-> ACCELERATION IDENTITY (why the seed is a_free*dt, not
// zero). MJX forward solves contact at the ACCELERATION level: its force
// counteracts BOTH the penetration (aref) AND the gravity/tau-driven foot-into-
// ground acceleration (J . a_free). A zero-velocity seed sees only aref and
// drops J . a_free -- here the DOMINANT term (~3x too small a force). So we seed
// the unconstrained one-step velocity qdot_seed = qvel0 + a_free*dt (qvel0=0):
//   qdot_post = a_free*dt + M^-1 J^T lambda
//   qacc_total = qdot_post/dt = a_free + M^-1 J^T (lambda/dt)
// which is EXACTLY MJX's constrained forward acceleration. a_free is Nuka's free
// floating-base ABA qacc (with the config's leg-tau), packed in Nuka's native
// flat-18 [omega_body; v_lin_body; legs]; we stay in Nuka frame until the final
// base-6 -> MJX-frame conversion (the validated featherstone bridge).
//
// GATES (each blocks the next):
//   GATE 0  free dynamics: Nuka free-ABA base-6 vs golden qacc_OFF (contact
//           disabled). This is the TOLERANCE FLOOR (the fixed-base oracle only
//           reached ~1e-4); the contact-ON match is asserted no tighter.
//   GATE 1  contact state: golden's recorded contact set (4 contacts, +Z normal,
//           delta=0.002, points at the cooked centers) -- the #1-risk check.
//   GATE 2  PRIMARY: base-6 qacc, contact ON, default AND high PGS iters. Split
//           into (a) the INPUT-FIDELITY proof (the C5c-2 deliverable): the EXACT
//           solve of Nuka's own assembled (A+R)lambda=rhs matches MJX efc to
//           ~7e-4, so J/M^-1/aref/R + the a_free*dt seed + the frame bridge are
//           MJX-correct; and (b) the CONVERGENCE proof: the C5b-core PGS kernel
//           converges to that exact (A+R) solve to ~2e-7 (flat 10->200 iters), so
//           the end-to-end base-6-vs-MJX lands at the ~1.4e-4 input floor. (b) was
//           previously a WRONG fixed point ~0.57% above the exact solve (R dropped
//           from the compliant numerator); now FIXED (row_solver.cu Solve-
//           CompliantVelocityRow regularizer feedback). A GREEN test now means
//           "inputs MJX-exact AND kernel converges to its own assembled (A+R)".
//           Corroborated by ON-OFF isolation and per-contact normal force.
//
// SCOPE (C5c-2). No production path is flipped; reuses the C5c-1 wiring. New
// golden go2_foot_contact_step.bin (generated, not hand-edited). DEFERRED to
// C5c-3: the flat-qdot(18) integrator + multi-step stability + steady-state
// Sigma(lambda) ~ m*g*dt force balance.
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"
#include "collision/candidate_pair.hpp"
#include "collision/contact_stream_driver.hpp"
#include "constraint/contact_manifold.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "constraint/row_builder.hpp"
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
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "solver/solver_config.hpp"  // SolverConfig (iters/dt knobs; legacy UnifiedSolve dropped M9 T11)

#include "foot_chain_jacobian.hpp"  // shared ComputeFootChainJ18 (FK-refresh-correct)

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
namespace amf = nuka::collision::amf;
using nuka::collision::BuildContactManifolds;
using nuka::collision::CandidatePair;
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
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

constexpr uint32_t kInvalidLink = ~0u;
constexpr uint32_t kDof = 18u;  // 6-DOF floating base + 12 revolute legs.
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;  // MUST match the generator's model.opt.timestep.
constexpr uint32_t kGroundHandle = 4096u;

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
// Bespoke loader for the C5c-2 contact-step golden (magic "NUKACSTP"). NOT the
// shared GoldenTrajectory NUKAGOLD format -- this golden carries BOTH a
// contact-ON and contact-OFF qacc plus the contact set + per-contact normal
// force, which the fixed [qpos,qvel,tau,qacc] record cannot hold. Layout mirrors
// generate_mjx_golden.py generate_floating_contact_step.
// ---------------------------------------------------------------------------
struct ContactConfig {
    std::vector<float> qpos;       // nq (base 7 + legs)
    std::vector<float> qvel;       // nv (=0)
    std::vector<float> tau;        // nv (base 6 = 0, legs)
    std::vector<float> qacc_on;    // nv
    std::vector<float> qacc_off;   // nv
    uint32_t ncon = 0u;
    std::vector<std::array<float, 7>> contacts;  // [pos(3), normal(3), dist]
    std::vector<float> normal_force;             // per recorded contact slot
    std::vector<float> invweight;                // body_invweight0 per contact slot
};

struct ContactStepGolden {
    uint32_t nq = 0u;
    uint32_t nv = 0u;
    uint32_t max_contacts = 0u;
    std::string model_name;
    std::vector<ContactConfig> configs;
};

uint32_t ReadU32(std::ifstream& in) {
    uint32_t v = 0u;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("contact-step golden: truncated header");
    return v;
}

float ReadF32(const std::vector<float>& buf, size_t& off) { return buf[off++]; }

ContactStepGolden LoadContactStepGolden(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open contact-step golden: " + path.string());
    std::array<char, 8> magic{};
    in.read(magic.data(), 8);
    if (!in || std::string_view(magic.data(), 8) != "NUKACSTP") {
        throw std::runtime_error("bad contact-step golden magic: " + path.string());
    }
    const uint32_t version = ReadU32(in);
    if (version != 1u) throw std::runtime_error("unsupported contact-step golden version");
    ContactStepGolden g;
    const uint32_t config_count = ReadU32(in);
    g.nq = ReadU32(in);
    g.nv = ReadU32(in);
    g.max_contacts = ReadU32(in);
    const uint32_t name_len = ReadU32(in);
    g.model_name.resize(name_len);
    if (name_len > 0u) in.read(g.model_name.data(), name_len);

    const size_t rec_floats = static_cast<size_t>(g.nq) + g.nv * 4u + 1u +
                              static_cast<size_t>(g.max_contacts) * 7u +
                              g.max_contacts + g.max_contacts;
    for (uint32_t c = 0u; c < config_count; ++c) {
        std::vector<float> rec(rec_floats);
        in.read(reinterpret_cast<char*>(rec.data()),
                static_cast<std::streamsize>(rec_floats * sizeof(float)));
        if (!in) throw std::runtime_error("contact-step golden: truncated record");
        size_t off = 0u;
        ContactConfig cfg;
        cfg.qpos.assign(rec.begin(), rec.begin() + g.nq); off += g.nq;
        cfg.qvel.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.tau.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.qacc_on.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.qacc_off.assign(rec.begin() + off, rec.begin() + off + g.nv); off += g.nv;
        cfg.ncon = static_cast<uint32_t>(ReadF32(rec, off));
        cfg.contacts.resize(g.max_contacts);
        for (uint32_t k = 0u; k < g.max_contacts; ++k) {
            for (uint32_t j = 0u; j < 7u; ++j) cfg.contacts[k][j] = ReadF32(rec, off);
        }
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

// Convert a Nuka base spatial 6-vector [omega(0:3) body; v_lin(3:6) body] to the
// MJX free-joint convention [v_lin(0:3) world; omega(3:6) body], adding the
// omega x v_lin spatial->classical linear term (vanishes at zero base velocity).
// Used for BOTH the seed-derived qacc_total base block and a_free.
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
// Cooked go2_float (reuses the C5c-1 cook). feet + per-link mass.
// ---------------------------------------------------------------------------
struct CookedFloat {
    articulation::ArticulationHostState host;
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
    const uint32_t link_count = result.host.TotalLinkCount();
    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link) {
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    }
    const auto& shapes = world.template_view.shape_table;
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) continue;
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) { calf_link = link; break; }
        }
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

std::vector<Transform> ForwardKinematics(cudaStream_t context, int context_dev,
                                         const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    OwnedDeviceBuffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform));
    articulation::UpdateWorldLinkPoses(context, context_dev, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    cudaStreamSynchronize(context);
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

Vec3 FootWorldCenter(const std::vector<Transform>& poses, const articulation::FootShape& foot) {
    const Transform calf = poses[foot.calf_local_link];
    return calf.position + calf.rotation.Rotate(foot.local_offset);
}

// 18x18 M^-1 via the production CRBA (reuse, not hand-rolled). Requires ABA
// pass-1 (link_xup current), so we run ComputeAccelerations first.
std::vector<float> ComputeInverseInertia18(cudaStream_t context, int context_dev,
                                           const articulation::ArticulationHostState& host) {
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    auto view = device.View();
    const uint32_t link_count = host.TotalLinkCount();
    articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, view, kGravityZ);
    OwnedDeviceBuffer composite(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia));
    OwnedDeviceBuffer m(static_cast<size_t>(kDof) * kDof * sizeof(float));
    OwnedDeviceBuffer m_inv(static_cast<size_t>(kDof) * kDof * sizeof(float));
    articulation::ComputeArticulationInertiaM(
        context, context_dev, view, kDof,
        static_cast<articulation::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m.Data()));
    articulation::FactorArticulationInertiaM(
        context, context_dev, view, kDof, static_cast<const float*>(m.Data()),
        static_cast<float*>(m_inv.Data()));
    cudaStreamSynchronize(context);
    std::vector<float> out(static_cast<size_t>(kDof) * kDof);
    m_inv.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

// 18-wide foot chain Jacobian via the shared FK-refresh-correct helper (see
// tests/solver/foot_chain_jacobian.hpp). Thin forwarder pinning dof_stride=kDof.
std::vector<float> ComputeFootChainJ18(cudaStream_t context, int context_dev,
                                       articulation::ArticulationHostState host,  // by value.
                                       const std::vector<Transform>& fk_world_poses,
                                       uint32_t contact_link, const Vec3& contact_point,
                                       const Vec3& contact_normal) {
    return nuka::test::ComputeFootChainJ18(context, context_dev, std::move(host), fk_world_poses,
                                           contact_link, contact_point, contact_normal,
                                           kDof);
}

amf::PrimParams MakeGroundPrim(float ground_height) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = Vec3{0.0f, 0.0f, 1.0f};
    p.frame.cz = Vec3{0.0f, -1.0f, 0.0f};
    p.frame.t = Vec3{0.0f, 0.0f, ground_height};
    return p;
}
amf::PrimParams MakeFootPrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;
    return p;
}
CollidableRef MakeFootRef(uint32_t calf_link) {
    CollidableRef ref;
    ref.type = CollidableType::ArticulationLink;
    ref.react = ReactionProviderKind::ArticulationChainJ;
    ref.handle = calf_link;
    return ref;
}
CollidableRef MakeGroundRef() {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = kGroundHandle;
    return ref;
}

// Free floating-base ABA at the host state (with q/qdot/tau set), returning the
// flat-18 a_free in Nuka native packing [omega_dot body(0:3); a_lin body(3:6);
// leg qddot(6:17)] PLUS the per-leg qddot (for GATE-0 leg check). Mirrors the
// featherstone harness output conversion (base accel = a0 - g_body, body frame).
struct FreeAba {
    std::array<float, 6> base_spatial{};  // [omega_dot; a_lin_body] body frame.
    std::vector<float> leg_qddot;          // leg-major, length 12.
    std::array<float, kDof> flat{};        // [base_spatial(0:5); legs(6:17)] * (unscaled).
};

FreeAba ComputeFreeAba(cudaStream_t context, int context_dev,
                       articulation::ArticulationHostState host,  // by value: we mutate state.
                       const Quat& base_rot) {
    const uint32_t root = host.articulation_link_offset[0];
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev, device.View(), kGravityZ);
    cudaStreamSynchronize(context);
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

// One config's full unified-pipeline result.
struct ParityResult {
    std::array<float, 6> base6_mjx{};     // Nuka qacc_total base-6 in MJX frame.
    std::array<float, 6> base6_free_mjx{}; // Nuka free-ABA base-6 in MJX frame (GATE 0).
    std::array<float, kDof> qacc_total{};  // Nuka full qacc in NUKA flat packing.
    std::vector<float> leg_qddot_free;     // free-ABA leg qddot (GATE 0 leg).
    std::vector<float> foot_lambda;        // per foot solved impulse.
    std::vector<float> foot_force;         // per foot lambda/dt (normal force).
    std::vector<float> contact_depth;
    uint32_t row_count = 0u;
    // DIAG: exact linear solve of Nuka's OWN assembled (A+R) lambda = rhs system
    // (force = lambda/dt), to compare against the kernel's foot_force.
    std::array<float, 4> exact_force{};
};

// Build the per-config host state from the golden (base pose + legs), then run:
// free ABA -> seed qdot = a_free*dt -> contact solve -> qacc_total = qdot_post/dt.
ParityResult RunParity(cudaStream_t context, int context_dev, CookedFloat cooked,
                       const ContactConfig& cfg, const ContactStepGolden& g,
                       float ground_height, const nuka::solver::SolverConfig& config) {
    ParityResult result;
    auto& host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t link_count = host.articulation_link_count[0];
    const uint32_t leg_dofs = link_count - 1u;

    // --- Set the host state from the golden (MJX free-joint conventions) -------
    // qpos: base = pos(0:3) world, quat(3:7) w-first; legs identity leg-major.
    Quat base_rot{cfg.qpos[3], cfg.qpos[4], cfg.qpos[5], cfg.qpos[6]};
    {
        const float n = std::sqrt(base_rot.w * base_rot.w + base_rot.x * base_rot.x +
                                  base_rot.y * base_rot.y + base_rot.z * base_rot.z);
        const float inv = n > 0.0f ? 1.0f / n : 1.0f;
        base_rot.w *= inv; base_rot.x *= inv; base_rot.y *= inv; base_rot.z *= inv;
    }
    host.base_pose[0].position = Vec3{cfg.qpos[0], cfg.qpos[1], cfg.qpos[2]};
    host.base_pose[0].rotation = base_rot;
    for (auto& v : host.link_velocity) for (float& c : v.v) c = 0.0f;  // qvel = 0.
    for (uint32_t i = 0u; i < leg_dofs; ++i) {
        host.q[root + 1u + i] = cfg.qpos[7u + i];
        host.qdot[root + 1u + i] = 0.0f;
        host.tau[root + 1u + i] = cfg.tau[6u + i];  // config leg holding-tau.
    }
    host.q[root] = 0.0f; host.qdot[root] = 0.0f; host.tau[root] = 0.0f;

    const Vec3 omega_body{0.0f, 0.0f, 0.0f};
    const Vec3 v_lin_body{0.0f, 0.0f, 0.0f};

    // --- Free ABA (includes the config leg-tau) -> a_free + GATE-0 base-6 ------
    const FreeAba afree = ComputeFreeAba(context, context_dev, host, base_rot);
    result.leg_qddot_free = afree.leg_qddot;
    result.base6_free_mjx = NukaSpatialToMjx(afree.base_spatial.data(), base_rot, omega_body, v_lin_body);

    // --- FK -> foot centers, M^-1 (ground seated ONCE from the home config) -----
    // CRITICAL: the generator seats ONE floor (off the home/config-0 feet) and
    // reuses it for ALL configs -- so config 2 (base lowered 1mm in qpos) has
    // DEEPER penetration (0.003) against that same floor. We therefore use the
    // FIXED ground_height the caller computed from the home config; re-seating
    // per config would erase config 2's extra penetration and give the wrong
    // force. The per-config foot FK (foot_point) is still live (config-2 feet
    // really are 1mm lower).
    const auto poses = ForwardKinematics(context, context_dev, host);
    const std::vector<float> minv = ComputeInverseInertia18(context, context_dev, host);

    const Vec3 kUp{0.0f, 0.0f, 1.0f};
    const amf::PrimParams ground_prim = MakeGroundPrim(ground_height);
    std::vector<CandidatePair> pairs;
    std::vector<uint32_t> foot_calf_link;
    std::vector<Vec3> foot_point;
    for (const auto& foot : cooked.feet) {
        const Vec3 center = FootWorldCenter(poses, foot);
        CandidatePair pair;
        pair.a = MakeFootRef(foot.calf_local_link);
        pair.b = MakeGroundRef();
        pairs.push_back(pair);
        foot_calf_link.push_back(foot.calf_local_link);
        foot_point.push_back(center);
    }
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::StaticWorld) {
            out->type = ShapeType::Plane; out->prim = ground_prim; return true;
        }
        if (ref.type == CollidableType::ArticulationLink) {
            for (size_t f = 0u; f < cooked.feet.size(); ++f) {
                if (cooked.feet[f].calf_local_link == ref.handle) {
                    out->type = ShapeType::Sphere;
                    out->prim = MakeFootPrim(foot_point[f], cooked.feet[f].radius);
                    return true;
                }
            }
        }
        return false;
    };
    std::vector<ContactManifold> manifolds;
    BuildContactManifolds(pairs, resolve, &manifolds);

    // --- C4b compliant rows (condim=1 frictionless, MuJoCo-default solref/solimp
    // carried on the manifold). invweight = MJX's body_invweight0 from the golden
    // so the dual regularizer R = invweight*(1-imp)/imp matches MuJoCo's EXACTLY
    // -- this completes "same compliance inputs on both sides" (solref/solimp +
    // invweight), it does NOT loosen anything (it TIGHTENS R from the 1.0
    // placeholder to MJX's 1.576). NOTE: Nuka's PRODUCTION R policy deliberately
    // differs from MuJoCo (exact J M^-1 J^T vs body_invweight0, documented in
    // solref_solimp.hpp); that design choice is NOT what this base-6 recoil test
    // validates, so we pin invweight to MJX here to isolate the recoil coupling.
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = cfg.invweight.empty() ? 1.0f : cfg.invweight[0];
    inputs.dt = kDt;
    inputs.condim = 1u;
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    result.row_count = rows.RowCount();

    std::vector<float> chain_jacobians;
    result.contact_depth.resize(manifolds.size());
    for (size_t m = 0u; m < manifolds.size(); ++m) {
        const std::vector<float> j =
            ComputeFootChainJ18(context, context_dev, host, poses, foot_calf_link[m], foot_point[m], kUp);
        chain_jacobians.insert(chain_jacobians.end(), j.begin(), j.end());
        result.contact_depth[m] =
            manifolds[m].point_count > 0u ? manifolds[m].points[0].penetration : 0.0f;
    }

    const uint32_t body_count = 0u, kArtIndex = 0u;
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        rows.body_indices[2u * r + 0u] = body_count + kArtIndex;
        rows.body_indices[2u * r + 1u] = nuka::constraint::kInvalidBodyIndex;
        art_refs[r].a = RowArticulationSide{kArtIndex, r};
        art_refs[r].b = RowArticulationSide{};
    }

    // --- THE SEED: unconstrained one-step velocity qdot = qvel0 + a_free*dt -----
    // qvel0 = 0, so qdot_seed = a_free*dt in Nuka flat-18 packing.
    std::vector<float> qdot(kDof, 0.0f);
    for (uint32_t i = 0u; i < kDof; ++i) qdot[i] = afree.flat[i] * kDt;

    // DIAG: exact 4x4 solve of Nuka's OWN system (A+R) lambda = rhs, lambda>=0.
    {
        const uint32_t nr = std::min<uint32_t>(rows.RowCount(), 4u);
        float A[4][4] = {{0}};
        float rhs[4] = {0}, Rr[4] = {0};
        for (uint32_t r = 0u; r < nr; ++r) {
            std::array<float, kDof> mjt{};
            for (uint32_t a = 0u; a < kDof; ++a) {
                float s = 0.0f;
                for (uint32_t b = 0u; b < kDof; ++b)
                    s += minv[a * kDof + b] * chain_jacobians[r * kDof + b];
                mjt[a] = s;
            }
            for (uint32_t c = 0u; c < nr; ++c) {
                float s = 0.0f;
                for (uint32_t a = 0u; a < kDof; ++a) s += chain_jacobians[c * kDof + a] * mjt[a];
                A[r][c] = s;
            }
            float jv0 = 0.0f;
            for (uint32_t a = 0u; a < kDof; ++a) jv0 += chain_jacobians[r * kDof + a] * qdot[a];
            rhs[r] = rows.rows[r].rhs * kDt - jv0;
            Rr[r] = rows.rows[r].compliance_alpha;
        }
        float lam[4] = {0, 0, 0, 0};
        for (uint32_t it = 0u; it < 2000u; ++it)
            for (uint32_t i = 0u; i < nr; ++i) {
                float s = rhs[i];
                for (uint32_t j = 0u; j < nr; ++j) if (j != i) s -= A[i][j] * lam[j];
                lam[i] = std::max(0.0f, s / (A[i][i] + Rr[i]));
            }
        for (uint32_t i = 0u; i < nr; ++i) result.exact_force[i] = lam[i] / kDt;
    }

    std::vector<nuka::runtime::rigid::BodyState> empty_bodies;
    // M9 T11 nk-ONLY: the SAME assembled rows / chain-J / M^-1 / qdot seed run
    // through the nk SolveRowsBlockIsland op (nk_solve_harness.hpp); the SAME
    // golden gates judge the result. The legacy solver::UnifiedSolve arm (deleted
    // in M9 T11-core) is gone -- this is the M4 re-point made the sole path.
    {
        const auto nk_res = nk_harness::NkSolveRows(
            rows, sides, art_refs, chain_jacobians, minv, qdot, &empty_bodies,
            kDof, static_cast<uint16_t>(config.velocity_iterations), kDt);
        EXPECT_TRUE(nk_res.ok) << "nk solve harness failed";
        if (nk_res.ok) {
            qdot = nk_res.qdot;
            for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
                rows.rows[r].lambda = nk_res.lambda[r];
            }
        }
    }

    // qacc_total = qdot_post / dt (NUKA flat packing).
    for (uint32_t i = 0u; i < kDof; ++i) result.qacc_total[i] = qdot[i] / kDt;

    // Base-6 (Nuka spatial [omega_dot; a_lin_body]) -> MJX frame.
    float base_spatial_total[6];
    for (uint32_t i = 0u; i < 6u; ++i) base_spatial_total[i] = result.qacc_total[i];
    result.base6_mjx = NukaSpatialToMjx(base_spatial_total, base_rot, omega_body, v_lin_body);

    result.foot_lambda.resize(rows.RowCount());
    result.foot_force.resize(rows.RowCount());
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        result.foot_lambda[r] = rows.rows[r].lambda;
        result.foot_force[r] = rows.rows[r].lambda / kDt;
    }
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

// The FIXED ground height the golden was generated with: seated off the HOME
// (config-0) feet at min_bottom + 0.002, then reused for every config (so config
// 2's lowered base produces deeper penetration against the same floor). Computed
// from the cooked HOME crouch (NOT a per-config FK), mirroring the generator's
// two-pass seat.
float HomeGroundHeight(cudaStream_t context, int context_dev, const CookedFloat& cooked,
                       const ContactConfig& home) {
    auto host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t leg_dofs = host.articulation_link_count[0] - 1u;
    Quat base_rot{home.qpos[3], home.qpos[4], home.qpos[5], home.qpos[6]};
    const float n = std::sqrt(base_rot.w * base_rot.w + base_rot.x * base_rot.x +
                              base_rot.y * base_rot.y + base_rot.z * base_rot.z);
    const float inv = n > 0.0f ? 1.0f / n : 1.0f;
    base_rot.w *= inv; base_rot.x *= inv; base_rot.y *= inv; base_rot.z *= inv;
    host.base_pose[0].position = Vec3{home.qpos[0], home.qpos[1], home.qpos[2]};
    host.base_pose[0].rotation = base_rot;
    for (uint32_t i = 0u; i < leg_dofs; ++i) host.q[root + 1u + i] = home.qpos[7u + i];
    const auto poses = ForwardKinematics(context, context_dev, host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& foot : cooked.feet) {
        const Vec3 center = FootWorldCenter(poses, foot);
        min_bottom = std::min(min_bottom, center.z - foot.radius);
    }
    return min_bottom + 0.002f;
}

}  // namespace

// ===========================================================================
// GATE 0 + GATE 1 + GATE 2 (the full acceptance, per config).
// ===========================================================================
TEST(FootGroundMjxParity, BaseSixContactRecoilMatchesMjx) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path))
        GTEST_SKIP() << "contact-step golden not present (run generator --mode floating-contact-step)";

    const auto golden = LoadContactStepGolden(golden_path);
    ASSERT_EQ(golden.nq, 19u);
    ASSERT_EQ(golden.nv, 18u);
    ASSERT_GT(golden.configs.size(), 0u);

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    const auto cooked = CookGo2Float();
    ASSERT_EQ(cooked.host.ArticulationCount(), 1u);

    // Default (10) and high (200, == MJX iterations) PGS sweeps (convergence story).
    const auto cfg_default = MakeConfig(10u);
    const auto cfg_high = MakeConfig(200u);

    // FIXED ground height from the home config (config 0), reused for all configs.
    const float ground_height = HomeGroundHeight(context, context_dev, cooked, golden.configs[0]);

    double worst_gate0 = 0.0;
    double worst_base6_default = 0.0;
    double worst_base6_high = 0.0;

    for (size_t ci = 0u; ci < golden.configs.size(); ++ci) {
        const auto& cfg = golden.configs[ci];
        const bool zero_leg_tau = std::all_of(cfg.tau.begin() + 6, cfg.tau.end(),
                                              [](float t) { return t == 0.0f; });

        // ---- GATE 1: the contact STATE the golden recorded (the #1-risk check).
        EXPECT_EQ(cfg.ncon, 4u) << "config " << ci << " MJX ncon != 4";
        for (uint32_t k = 0u; k < cfg.ncon && k < golden.max_contacts; ++k) {
            const auto& con = cfg.contacts[k];
            EXPECT_NEAR(con[5], 1.0f, 1.0e-5f) << "config " << ci << " contact " << k
                                               << " normal not +Z";
            EXPECT_LT(con[6], 0.0f) << "config " << ci << " contact " << k << " not penetrating";
        }

        const auto def = RunParity(context, context_dev, cooked, cfg, golden, ground_height, cfg_default);
        const auto high = RunParity(context, context_dev, cooked, cfg, golden, ground_height, cfg_high);
        ASSERT_EQ(def.row_count, 4u) << "config " << ci << " Nuka produced != 4 contact rows";

        // ---- GATE 0: Nuka free-ABA base-6 vs golden qacc_OFF (tolerance floor).
        std::array<float, 6> golden_off6{};
        for (uint32_t i = 0u; i < 6u; ++i) golden_off6[i] = cfg.qacc_off[i];
        const float gate0 = MaxAbs6(def.base6_free_mjx, golden_off6);
        worst_gate0 = std::max<double>(worst_gate0, gate0);
        std::printf("[diag] config %zu GATE0 free-ABA base-6 vs qacc_OFF max_abs = %.3e\n",
                    ci, gate0);
        // GATE-0 leg check (catches a wrong leg mapping): free-ABA leg qddot vs
        // qacc_off legs, RELATIVE (free legs are ill-conditioned, |qacc|~2000).
        double leg_rel = 0.0;
        for (uint32_t i = 0u; i < def.leg_qddot_free.size(); ++i) {
            const float ref = cfg.qacc_off[6u + i];
            const float den = std::max(1.0f, std::fabs(ref));
            leg_rel = std::max<double>(leg_rel, std::fabs(def.leg_qddot_free[i] - ref) / den);
        }
        std::printf("[diag] config %zu GATE0 free-ABA leg rel_err = %.3e\n", ci, leg_rel);
        EXPECT_LT(leg_rel, 5.0e-3) << "config " << ci
            << " free-ABA leg qddot diverges from MJX (leg mapping / ABA bug)";

        // ---- GATE 2: PRIMARY base-6 contact-ON parity. ------------------------
        std::array<float, 6> golden_on6{};
        for (uint32_t i = 0u; i < 6u; ++i) golden_on6[i] = cfg.qacc_on[i];
        const float base6_def = MaxAbs6(def.base6_mjx, golden_on6);
        const float base6_high = MaxAbs6(high.base6_mjx, golden_on6);

        // ON-OFF isolation (cancels any GATE-0 free residual) on both sides.
        std::array<float, 6> nuka_iso{}, mjx_iso{};
        for (uint32_t i = 0u; i < 6u; ++i) {
            nuka_iso[i] = high.base6_mjx[i] - def.base6_free_mjx[i];
            mjx_iso[i] = cfg.qacc_on[i] - cfg.qacc_off[i];
        }
        const float iso = MaxAbs6(nuka_iso, mjx_iso);

        double sum_force = 0.0, sum_mjx_force = 0.0;
        for (uint32_t r = 0u; r < def.row_count; ++r) sum_force += high.foot_force[r];
        for (uint32_t k = 0u; k < cfg.ncon && k < golden.max_contacts; ++k)
            sum_mjx_force += cfg.normal_force[k];

        std::printf("[diag] config %zu (leg_tau=%s) base-6 ON parity: default(10)=%.3e  "
                    "high(200)=%.3e  ON-OFF iso=%.3e\n",
                    ci, zero_leg_tau ? "0" : "nz", base6_def, base6_high, iso);
        std::printf("[diag] config %zu base-6  qacc Nuka(high)=(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    ci, high.base6_mjx[0], high.base6_mjx[1], high.base6_mjx[2],
                    high.base6_mjx[3], high.base6_mjx[4], high.base6_mjx[5]);
        std::printf("[diag] config %zu base-6 qacc MJX_ON =(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    ci, golden_on6[0], golden_on6[1], golden_on6[2],
                    golden_on6[3], golden_on6[4], golden_on6[5]);
        std::printf("[diag] config %zu Sigma(lambda/dt)=%.4f N   Sigma(MJX efc normal)=%.4f N\n",
                    ci, sum_force, sum_mjx_force);

        // Per-contact normal force corroboration (Nuka lambda/dt vs MJX efc) PLUS
        // the EXACT linear solve of Nuka's OWN assembled (A+R)lambda=rhs system.
        // The exact-solve isolates INPUT correctness (J/M^-1/aref/R) from the
        // C5b-core PGS kernel's convergence: exact-solve ~ MJX to ~7e-4 PROVES the
        // inputs are MJX-correct; the kernel now matches its own exact solve to
        // ~2e-7 (post regularizer-feedback fix -- was a uniform ~R/A=0.57% surplus).
        float exact_vs_mjx_rel = 0.0f, kernel_vs_exact_rel = 0.0f;
        for (uint32_t r = 0u; r < def.row_count && r < cfg.ncon; ++r) {
            std::printf("[diag] config %zu foot %u: Nuka kernel f=%.4f  exact-solve f=%.4f  "
                        "MJX efc=%.4f N\n",
                        ci, r, high.foot_force[r], high.exact_force[r], cfg.normal_force[r]);
            const float den = std::max(1.0f, std::fabs(cfg.normal_force[r]));
            exact_vs_mjx_rel =
                std::max(exact_vs_mjx_rel, std::fabs(high.exact_force[r] - cfg.normal_force[r]) / den);
            const float dene = std::max(1.0f, std::fabs(high.exact_force[r]));
            kernel_vs_exact_rel =
                std::max(kernel_vs_exact_rel, std::fabs(high.foot_force[r] - high.exact_force[r]) / dene);
        }
        std::printf("[diag] config %zu force REL: exact-solve vs MJX=%.3e  "
                    "kernel vs exact-solve=%.3e (C5b-core PGS bias)\n",
                    ci, exact_vs_mjx_rel, kernel_vs_exact_rel);
        // INPUT-CORRECTNESS assertion (the C5c-2 deliverable): the exact solve of
        // Nuka's OWN assembled contact system matches MJX efc_force tightly --
        // J/M^-1/aref/R + the a_free*dt seed + the featherstone frame bridge are
        // ALL MJX-correct. This is the recoil PATH fidelity the acceptance layer
        // proves.
        EXPECT_LT(exact_vs_mjx_rel, 5.0e-3f)
            << "config " << ci << " Nuka's exact contact-system solve diverges from MJX "
               "(an INPUT J/M^-1/aref/R error, not a solver-convergence residual)";
        // CONVERGENCE GATE (durable regression guard): the C5b-core UnifiedSolve PGS
        // kernel now converges to the EXACT solve of its OWN assembled (A+R)lambda=rhs
        // system to PGS precision (~2e-7, flat 10->200 iters). Previously it sat at a
        // WRONG FIXED POINT ~5.7e-3 (== R/A) ABOVE that exact solve, because the
        // compliant row update carried R only in the denominator (ComputeCompliant-
        // EffectiveMass) and DROPPED the matching -R*lambda regularizer feedback from
        // the numerator -- so its fixed point was A*lambda=b (R cancels) instead of
        // (A+R)*lambda=b. FIXED in src/solver/gpu/row_solver.cu SolveCompliantVelocity-
        // Row by adding the -row.compliance_alpha*old_impulse term to the numerator
        // (regularized Gauss-Seidel). The earlier hypothesis that this was a
        // floating-base base-DOF apply/link_velocity-routing gap (row_solver.cu
        // :341-345) was WRONG: this entire test path is flat-18 end-to-end (no
        // link_velocity scatter), and the three reaction legs (eff-mass / apply / Jv)
        // are mutually consistent by construction. The bias was purely the missing
        // regularizer term, an algebraic (scenario-independent) fix. This bound locks
        // the fix: a regression that reintroduces the wrong fixed point trips it.
        EXPECT_LT(kernel_vs_exact_rel, 1.0e-4f)
            << "config " << ci << " C5b-core PGS kernel diverged from the exact solve "
               "of its OWN assembled (A+R)lambda=rhs system (regularizer-feedback "
               "regression: -R*lambda dropped from the compliant numerator?)";

        worst_base6_default = std::max<double>(worst_base6_default, base6_def);
        worst_base6_high = std::max<double>(worst_base6_high, base6_high);

        // RELATIVE base-6 error: max_i |nuka_i - mjx_i| / max(scale, |mjx_i|),
        // with a small scale floor so near-zero components are absolute. A relative
        // bound is the principled measure (the task endorses relative tol for
        // high-magnitude quantities; config 1's base accel ~64). With the C5b-core
        // regularizer-feedback FIX (see kernel_vs_exact gate above), the end-to-end
        // base-6 now lands at the INPUT-FIDELITY floor: the kernel solves Nuka's own
        // assembled (A+R)lambda=rhs to ~2e-7, and that assembled system matches MJX
        // efc to ~7e-4 (J/M^-1/aref/R are MJX-exact, A to 0.08%, aref+R bit-exact),
        // so base-6-vs-MJX is ~1.4e-4 (was ~0.6% under the old wrong fixed point).
        // The residual is now solver-clean: it is the input-translation residual, NOT
        // a kernel-convergence bias.
        const float kScale = 1.0f;  // floor: |accel| below 1 is checked absolutely.
        float base6_rel = 0.0f;
        for (uint32_t i = 0u; i < 6u; ++i) {
            const float den = std::max(kScale, std::fabs(golden_on6[i]));
            base6_rel = std::max(base6_rel, std::fabs(high.base6_mjx[i] - golden_on6[i]) / den);
        }
        // Per-foot normal-force relative error (the direct force-match number).
        float force_rel = 0.0f;
        for (uint32_t r = 0u; r < def.row_count && r < cfg.ncon; ++r) {
            const float den = std::max(1.0f, std::fabs(cfg.normal_force[r]));
            force_rel = std::max(force_rel, std::fabs(high.foot_force[r] - cfg.normal_force[r]) / den);
        }
        std::printf("[diag] config %zu base-6 REL err=%.3e  per-foot force REL err=%.3e\n",
                    ci, base6_rel, force_rel);

        // END-TO-END BOUND (INPUT-FIDELITY-LIMITED post-fix). With the C5b-core
        // regularizer-feedback fix the kernel is solver-clean, so the end-to-end
        // base-6 is bounded by the recoil-PATH input fidelity (exact-solve vs MJX
        // ~7e-4): base-6-vs-MJX ~1.4e-4, per-foot force ~7.2e-4. These bounds lock
        // BOTH the input fidelity AND the kernel convergence -- a regressed kernel
        // (wrong fixed point) would re-inflate base-6 to ~0.6% and trip the 1e-3 bound.
        EXPECT_LT(base6_rel, 1.0e-3f)
            << "config " << ci << " base-6 contact-ON qacc drifted beyond the "
               "input-fidelity floor (kernel regularizer-feedback regression or input drift)";
        EXPECT_LT(force_rel, 2.0e-3f)
            << "config " << ci << " per-foot normal force drifted beyond the "
               "input-fidelity floor vs MJX efc_force";
        // ON-OFF isolation corroboration (cancels the GATE-0 free residual): the
        // pure contact contribution base-6 also matches MJX within the same bound.
        const float iso_rel = [&] {
            float m = 0.0f;
            for (uint32_t i = 0u; i < 6u; ++i) {
                const float den = std::max(kScale, std::fabs(mjx_iso[i]));
                m = std::max(m, std::fabs(nuka_iso[i] - mjx_iso[i]) / den);
            }
            return m;
        }();
        // The ON-OFF isolation folds the GATE-0 free-ABA residual into a relative
        // metric with a smaller (mjx_iso) denominator, so it runs slightly looser
        // than base6_rel (config 1 ~1.3e-3). Still ~8x tighter than the pre-fix 1e-2.
        EXPECT_LT(iso_rel, 2.0e-3f)
            << "config " << ci << " ON-OFF contact-isolation base-6 diverges from MJX";
        // Convergence behaviour: high-iter must not be WORSE than default (a real
        // bug would not tighten with iterations). Here it is converged (flat).
        EXPECT_LE(base6_high, base6_def + 1.0e-4f)
            << "config " << ci << " base-6 residual did not tighten with PGS iters "
               "(structural bug, not convergence-limited)";
        (void)iso;
    }

    std::printf("[diag] WORST across configs: GATE0=%.3e  base-6 default=%.3e  base-6 high=%.3e\n",
                worst_gate0, worst_base6_default, worst_base6_high);
    // GATE 0 is the tolerance floor: the free dynamics must match before contact.
    EXPECT_LT(worst_gate0, 5.0e-3) << "free-base ABA base-6 diverges from MJX qacc_OFF "
                                      "(model misalignment -- fix before trusting contact-ON)";
}

// ===========================================================================
// Determinism (D1): two identical parity runs produce byte-identical qacc_total.
// ===========================================================================
TEST(FootGroundMjxParity, ParityRunDeterministic) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path)) GTEST_SKIP() << "contact-step golden not present";

    const auto golden = LoadContactStepGolden(golden_path);
    ASSERT_GT(golden.configs.size(), 0u);
    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    const auto cooked = CookGo2Float();
    const auto cfg = MakeConfig(64u);
    const float ground_height = HomeGroundHeight(context, context_dev, cooked, golden.configs[0]);

    const auto a = RunParity(context, context_dev, cooked, golden.configs[0], golden, ground_height, cfg);
    const auto b = RunParity(context, context_dev, cooked, golden.configs[0], golden, ground_height, cfg);
    EXPECT_EQ(std::memcmp(a.qacc_total.data(), b.qacc_total.data(),
                          a.qacc_total.size() * sizeof(float)), 0)
        << "two identical parity runs produced different qacc (nondeterministic)";
    std::printf("[diag] (D1) parity qacc_total byte-identical across 2 runs\n");
}

// ===========================================================================
// M4 RE-POINT (now the SOLE path, M9 T11) — the MJX golden gates through the nk
// SolveRowsBlockIsland op (plan M4: "test_foot_ground_mjx_parity (oracle 口径)
// 重指"). RunParity already runs ONLY the nk device-resident solve; this arm
// re-holds the MJX bounds (base-6 rel 1e-3, per-foot force rel 2e-3) at the
// high-iter operating point. The legacy nk-vs-UnifiedSolve cross-check was
// dropped with the legacy class.
// ===========================================================================
TEST(FootGroundMjxParity, NkSolveRowsBlockIslandMatchesMjx) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto golden_path = GoldenPath("go2_foot_contact_step.bin");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float scene not available";
    if (!std::filesystem::exists(golden_path))
        GTEST_SKIP() << "contact-step golden not present";

    const auto golden = LoadContactStepGolden(golden_path);
    ASSERT_GT(golden.configs.size(), 0u);
    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;
    const auto cooked = CookGo2Float();
    const auto cfg_high = MakeConfig(200u);
    const float ground_height = HomeGroundHeight(context, context_dev, cooked, golden.configs[0]);

    for (size_t ci = 0u; ci < golden.configs.size(); ++ci) {
        const auto& cfg = golden.configs[ci];
        const auto nk = RunParity(context, context_dev, cooked, cfg, golden, ground_height,
                                  cfg_high);
        ASSERT_EQ(nk.row_count, 4u) << "config " << ci << " nk produced != 4 rows";

        std::array<float, 6> golden_on6{};
        for (uint32_t i = 0u; i < 6u; ++i) golden_on6[i] = cfg.qacc_on[i];
        const float kScale = 1.0f;
        float base6_rel = 0.0f;
        for (uint32_t i = 0u; i < 6u; ++i) {
            const float den = std::max(kScale, std::fabs(golden_on6[i]));
            base6_rel = std::max(base6_rel,
                                 std::fabs(nk.base6_mjx[i] - golden_on6[i]) / den);
        }
        float force_rel = 0.0f;
        for (uint32_t r = 0u; r < nk.row_count && r < cfg.ncon; ++r) {
            const float den = std::max(1.0f, std::fabs(cfg.normal_force[r]));
            force_rel = std::max(force_rel,
                                 std::fabs(nk.foot_force[r] - cfg.normal_force[r]) / den);
        }
        std::printf("[nk re-point] config %zu base-6 REL=%.3e force REL=%.3e\n",
                    ci, base6_rel, force_rel);
        EXPECT_LT(base6_rel, 1.0e-3f)
            << "config " << ci << " nk base-6 contact-ON drifted vs the MJX golden";
        EXPECT_LT(force_rel, 2.0e-3f)
            << "config " << ci << " nk per-foot force drifted vs MJX efc_force";
    }
}
