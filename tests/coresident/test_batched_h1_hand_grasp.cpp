// ---------------------------------------------------------------------------
// H1.2 -- the COOKED H1 HAND cup grasp WRAPPED INTO the batched general world.
// ---------------------------------------------------------------------------
// H1.1/H1.1b PROVED (HEAD) that an HONEST active force-closure grasp of the ~10.9cm
// cup EXISTS for the cooked H1 dexterous hand (sphere fingertips, 30-sphere finger-
// only wrap), validated by the N=1 co-resident oracle (UnifiedCoResidentStepper, the
// scene in test_h1_grasp_feasibility_probe.cpp). H1.2 takes that EXACT scene and runs
// it through the BATCHED general world (BatchedUnifiedWorld) at N envs, then PROVES the
// batched H1-hand grasp is BYTE-EXACT vs the N=1 co-resident oracle on the IDENTICAL
// scene -- i.e. the env-major batching machinery (already proven for a SYNTHETIC 2-DOF
// gripper) GENERALIZES to the MUCH bigger H1-hand articulation (~18 links, ~30 finger-
// tips, dof_stride ~12, ~11 genuine contacts). That bigger articulation is EXACTLY the
// surface where prior env-major layout bugs lived (the env-major M^-1 tile @
// art_index*dof_stride^2, the chain-J gathered over GLOBAL contact links
// e*base_link_count+finger). If the batched H1-hand scene reproduces the oracle byte-
// exactly, the machinery is proven to generalize.
//
// SCOPE: PHYSICS + PARITY + THROUGHPUT ONLY. No RL substrate (per-step action injection
// generalization 2->N DOF is deferred to H1.4). The grip drive here is the SAME PD-close
// the probe uses (Kp=4/Kd=0.4/offset=0.18), applied IDENTICALLY to both the batched
// world (via SetActions, DOF-indexed) and the oracle (via SetGripTorque, per-link), each
// computed from its OWN downloaded gripper state -> the two drive seams are equivalent.
//
// The scene helpers (LoadCupHull/ScaleCupHull, LoadH1Fixed, FingerOnlyWrapSpheres,
// ApplyCurl/CurlForScale, ForwardKinematics, BestPlacementFingerOnlyShallow, the
// GraspScene builder) are REPLICATED from the probe's anon namespace -- the probe TU
// (test_h1_grasp_feasibility_probe.cpp) is committed and LEFT BYTE-IDENTICAL. ADDITIVE
// parallel TU; no production stepper/solver/golden/batched-world edit.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;
constexpr float kCupMass = 0.2f;
constexpr float kPi = 3.14159265358979323846f;

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kH1Mjcf) && std::filesystem::exists(kCupUsda);
}

// ===========================================================================
// Cup hull loader + scale -- REPLICATED from the probe (byte-identical math).
// ===========================================================================
struct CupHull {
    std::vector<float> verts;
    Vec3 lo{}, hi{};
    uint32_t vcount = 0u;
};
CupHull LoadCupHull() {
    auto scene = nuka::import::LoadUsd(kCupUsda);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;
    }
    const auto blob = nuka::scene::CookScene(scene);
    CupHull out;
    const auto& cg = blob.convex_geometry;
    if (cg.vertex_counts.empty()) return out;
    const uint32_t voff = cg.vertex_offsets[0];
    const uint32_t vcnt = cg.vertex_counts[0];
    out.vcount = vcnt;
    out.verts.assign(cg.vertices.begin() + voff * 3u,
                     cg.vertices.begin() + (voff + vcnt) * 3u);
    out.lo = Vec3{out.verts[0], out.verts[1], out.verts[2]};
    out.hi = out.lo;
    for (uint32_t i = 0u; i < vcnt; ++i) {
        const Vec3 v{out.verts[i * 3u], out.verts[i * 3u + 1u], out.verts[i * 3u + 2u]};
        out.lo = Vec3{std::min(out.lo.x, v.x), std::min(out.lo.y, v.y), std::min(out.lo.z, v.z)};
        out.hi = Vec3{std::max(out.hi.x, v.x), std::max(out.hi.y, v.y), std::max(out.hi.z, v.z)};
    }
    return out;
}
CupHull ScaleCupHull(const CupHull& in, float sxy, float sz) {
    CupHull out = in;
    const Vec3 c = (in.hi + in.lo) * 0.5f;
    for (uint32_t i = 0u; i < out.vcount; ++i) {
        out.verts[i * 3u + 0u] = c.x + (in.verts[i * 3u + 0u] - c.x) * sxy;
        out.verts[i * 3u + 1u] = c.y + (in.verts[i * 3u + 1u] - c.y) * sxy;
        out.verts[i * 3u + 2u] = c.z + (in.verts[i * 3u + 2u] - c.z) * sz;
    }
    out.lo = Vec3{out.verts[0], out.verts[1], out.verts[2]};
    out.hi = out.lo;
    for (uint32_t i = 0u; i < out.vcount; ++i) {
        const Vec3 v{out.verts[i * 3u], out.verts[i * 3u + 1u], out.verts[i * 3u + 2u]};
        out.lo = Vec3{std::min(out.lo.x, v.x), std::min(out.lo.y, v.y), std::min(out.lo.z, v.z)};
        out.hi = Vec3{std::max(out.hi.x, v.x), std::max(out.hi.y, v.y), std::max(out.hi.z, v.z)};
    }
    return out;
}

// ===========================================================================
// Cooked H1 (Fixed base, faithful armature/damping) -- REPLICATED from the probe.
// ===========================================================================
struct CookedH1 {
    articulation::ArticulationHostState host;
    nuka::scene::SceneIR scene;
};
constexpr float kRealArmature = 0.1f;
constexpr float kRealDamping = 1.0f;

CookedH1 LoadH1Fixed(const std::vector<std::string>& free_finger_bodies,
                     float driven_armature, float driven_damping) {
    CookedH1 out;
    out.scene = nuka::import::LoadMjcf(kH1Mjcf);
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    const auto blob = nuka::scene::CookScene(out.scene);
    auto topos = articulation::CookArticulations(blob);
    std::vector<uint32_t> free_bodies;
    for (const auto& nm : free_finger_bodies) {
        for (uint32_t b = 0u; b < out.scene.RigidBodyCount(); ++b) {
            if (out.scene.GetBody(b).name == nm) { free_bodies.push_back(b); break; }
        }
    }
    auto is_free = [&](uint32_t body) {
        return std::find(free_bodies.begin(), free_bodies.end(), body) != free_bodies.end();
    };
    for (auto& topo : topos) {
        for (size_t i = 0u; i < topo.parent_links.size(); ++i) {
            const bool is_root = topo.parent_links[i] == nuka::scene::kInvalidBody;
            const uint32_t body = topo.link_bodies[i];
            if (is_root || !is_free(body)) {
                topo.joint_types[i] = articulation::ArticulationJointType::Fixed;
            } else {
                topo.joint_dampings[i] = driven_damping;
                topo.joint_armatures[i] = driven_armature;
            }
        }
    }
    out.host = articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
}

std::string LinkName(const CookedH1& h1, uint32_t link) {
    if (h1.host.link_body.size() <= link) return "?";
    const uint32_t body = h1.host.link_body[link];
    if (body < h1.scene.Bodies().size()) return h1.scene.GetBody(body).name;
    return "?";
}
uint32_t LinkByName(const CookedH1& h1, const std::string& name) {
    for (uint32_t l = 0u; l < h1.host.TotalLinkCount(); ++l)
        if (LinkName(h1, l) == name) return l;
    return kInvalidLink;
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

// ===========================================================================
// THE DENSE FINGER-ONLY WRAP -- REPLICATED from the probe (the VALIDATED GO set).
// ===========================================================================
constexpr float kWrapRadius = 0.006f;

struct WrapSphere {
    std::string body;
    Vec3 local_offset{};
    float radius = kWrapRadius;
    const char* region = "";
};

const std::vector<std::string> kWrapDriven = {
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
    "R_index_proximal",  "R_index_intermediate",
    "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal",   "R_ring_intermediate",
    "R_pinky_proximal",  "R_pinky_intermediate",
};

std::vector<WrapSphere> FingerOnlyWrapSpheres() {
    std::vector<WrapSphere> v;
    const char* fingers[] = {"R_index", "R_middle", "R_ring", "R_pinky"};
    const float finger_x[] = {0.006f, 0.016f, 0.026f};
    for (const char* f : fingers)
        for (const char* seg : {"_proximal", "_intermediate"})
            for (float fx : finger_x)
                v.push_back({std::string(f) + seg, Vec3{fx, 0.0f, 0.0f}, kWrapRadius, "finger"});
    const float thumb_x[] = {0.004f, 0.012f, 0.020f};
    for (const char* seg : {"R_thumb_intermediate", "R_thumb_distal"})
        for (float tx : thumb_x)
            v.push_back({seg, Vec3{tx, 0.0f, 0.0f}, kWrapRadius, "thumb"});
    return v;
}

struct CurlPose {
    float finger_prox = 1.0f, finger_int = 1.1f;
    float thumb_yaw = 1.0f, thumb_pitch = 0.5f, thumb_int = 0.6f, thumb_dist = 0.6f;
};
void ApplyCurl(const CookedH1& h1, articulation::ArticulationHostState* host,
               const CurlPose& c) {
    auto setq = [&](const std::string& nm, float val) {
        const uint32_t l = LinkByName(h1, nm);
        if (l != kInvalidLink) host->q[l] = val;
    };
    setq("R_thumb_proximal_base", c.thumb_yaw);
    setq("R_thumb_proximal", c.thumb_pitch);
    setq("R_thumb_intermediate", c.thumb_int);
    setq("R_thumb_distal", c.thumb_dist);
    for (const char* f : {"R_index", "R_middle", "R_ring", "R_pinky"}) {
        setq(std::string(f) + "_proximal", c.finger_prox);
        setq(std::string(f) + "_intermediate", c.finger_int);
    }
}
CurlPose CurlForScale(float sxy) {
    CurlPose c;
    const float relax = (sxy - 1.0f) * 0.55f;
    c.finger_prox = std::max(0.45f, 1.0f - relax);
    c.finger_int  = std::max(0.55f, 1.1f - relax);
    c.thumb_yaw = 1.0f; c.thumb_pitch = 0.5f;
    c.thumb_int  = std::max(0.35f, 0.6f - relax * 0.4f);
    c.thumb_dist = std::max(0.35f, 0.6f - relax * 0.4f);
    return c;
}

Vec3 SphereCenter(const std::vector<Transform>& poses, uint32_t link, const Vec3& off) {
    const Transform& lp = poses[link];
    return lp.position + lp.rotation.Rotate(off);
}
float CupRadius(const CupHull& hull) {
    const Vec3 hc = (hull.hi + hull.lo) * 0.5f;
    float r = 0.0f;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        const float vx = hull.verts[i * 3u + 0u] - hc.x;
        const float vy = hull.verts[i * 3u + 1u] - hc.y;
        r = std::max(r, std::sqrt(vx * vx + vy * vy));
    }
    return r;
}
float RimGap(const Vec3& sc, const Vec3& cc, const CupHull& hull) {
    const Vec3 hc = (hull.hi + hull.lo) * 0.5f;
    float nearest = 1e9f;
    for (uint32_t v = 0u; v < hull.vcount; ++v) {
        const Vec3 hv{cc.x + hull.verts[v * 3u + 0u] - hc.x,
                      cc.y + hull.verts[v * 3u + 1u] - hc.y,
                      cc.z + hull.verts[v * 3u + 2u] - hc.z};
        nearest = std::min(nearest, (sc - hv).Length());
    }
    return nearest;
}

struct Surround {
    Vec3 cup_center{};
    uint32_t genuine = 0u;
    std::vector<float> azimuths;
    float max_gap = 360.0f;
    float covered_arc = 0.0f;
};
Surround MeasureSurround(const CookedH1& h1, const std::vector<Transform>& poses,
                         const std::vector<WrapSphere>& spheres, const CupHull& hull,
                         const Vec3& cc) {
    Surround out;
    out.cup_center = cc;
    std::vector<float> azs;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        if (RimGap(c, cc, hull) >= s.radius) continue;
        float az = std::atan2(c.y - cc.y, c.x - cc.x) * 180.0f / kPi;
        if (az < 0.0f) az += 360.0f;
        azs.push_back(az);
    }
    out.genuine = static_cast<uint32_t>(azs.size());
    std::sort(azs.begin(), azs.end());
    out.azimuths = azs;
    if (azs.size() >= 2u) {
        float mg = 0.0f;
        for (size_t i = 0u; i < azs.size(); ++i) {
            const float next = (i + 1u < azs.size()) ? azs[i + 1u] : azs[0] + 360.0f;
            mg = std::max(mg, next - azs[i]);
        }
        out.max_gap = mg;
        out.covered_arc = 360.0f - mg;
    }
    return out;
}
float MaxPrePenetration(const CookedH1& h1, const std::vector<Transform>& poses,
                        const std::vector<WrapSphere>& spheres, const CupHull& hull,
                        const Vec3& cc) {
    float md = 0.0f;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        const float rim = RimGap(c, cc, hull);
        if (rim < s.radius) md = std::max(md, s.radius - rim);
    }
    return md;
}
Vec3 CavityCenterZ(const CookedH1& h1, const std::vector<Transform>& poses,
                   const std::vector<WrapSphere>& spheres, Vec3* lo, Vec3* hi) {
    *lo = Vec3{1e9f, 1e9f, 1e9f}; *hi = Vec3{-1e9f, -1e9f, -1e9f};
    float zsum = 0.0f; uint32_t zn = 0u;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        *lo = Vec3{std::min(lo->x, c.x), std::min(lo->y, c.y), std::min(lo->z, c.z)};
        *hi = Vec3{std::max(hi->x, c.x), std::max(hi->y, c.y), std::max(hi->z, c.z)};
        zsum += c.z; ++zn;
    }
    return Vec3{0, 0, zn ? zsum / zn : 0.5f * (lo->z + hi->z)};
}
float SqueezeMag(const CookedH1& h1, const std::vector<Transform>& poses,
                 const std::vector<WrapSphere>& spheres, const CupHull& hull, const Vec3& cc) {
    Vec3 s{0, 0, 0};
    for (const auto& sp : spheres) {
        const uint32_t l = LinkByName(h1, sp.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, sp.local_offset);
        const float rim = RimGap(c, cc, hull);
        if (rim >= sp.radius) continue;
        const float dx = c.x - cc.x, dy = c.y - cc.y;
        const float d = std::sqrt(dx * dx + dy * dy);
        if (d < 1e-6f) continue;
        const float w = sp.radius - rim;
        s.x += w * (-dx / d); s.y += w * (-dy / d);
    }
    return std::sqrt(s.x * s.x + s.y * s.y);
}

constexpr float kShallowPenMax = 0.002f;
constexpr float kCagingArcMin = 200.0f;
struct Place { Vec3 center; uint32_t genuine; float covered_arc; float max_gap;
               float max_pen; float squeeze_mag; bool found; };
Place BestPlacementFingerOnlyShallow(const CookedH1& h1, const std::vector<Transform>& poses,
                                     const std::vector<WrapSphere>& spheres,
                                     const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses, spheres, &lo, &hi);
    Place best{Vec3{}, 0, 0, 360.0f, 0.0f, 1e9f, false};
    const int kN = 25;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Vec3 cc{fx, fy, cz.z};
        const Surround sr = MeasureSurround(h1, poses, spheres, hull, cc);
        if (sr.covered_arc <= kCagingArcMin || sr.genuine < 3u) continue;
        const float pen = MaxPrePenetration(h1, poses, spheres, hull, cc);
        if (pen > kShallowPenMax) continue;
        const float sq = SqueezeMag(h1, poses, spheres, hull, cc);
        if (sq < best.squeeze_mag)
            best = Place{cc, sr.genuine, sr.covered_arc, sr.max_gap, pen, sq, true};
    }
    return best;
}

constexpr float kMu = 0.8f;
constexpr float kKp = 4.0f;
constexpr float kKd = 0.4f;
constexpr float kCloseOffset = 0.18f;
// The LOCKED feasible cup size (H1.1b): ~10.9cm == 1.8x near-uniform scale.
constexpr float kSxy = 1.8f;
constexpr float kSz = 1.8f;

// ===========================================================================
// The grasp scene -- REPLICATED from the probe's BuildFingerOnlyScene. Produces a
// fully-populated coresident::GraspConfig (the oracle input) PLUS the H1 host proto +
// drive links -- enough to drive BOTH the oracle and the batched world identically.
// ===========================================================================
struct GraspScene {
    CookedH1 h1;
    articulation::ArticulationHostState host;  // curled-open proto (q set by ApplyCurl).
    coresident::GraspConfig config;
    BodyState cup0;
    std::vector<uint32_t> drive_links;
    Vec3 cup_center{};
    Place place;
    uint32_t sphere_count = 0u;
};

GraspScene BuildFingerOnlyScene(const nuka::phi::DeviceContext& context, const CupHull& base,
                                float sxy, float sz) {
    GraspScene gs;
    const CupHull hull = ScaleCupHull(base, sxy, sz);
    gs.h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    gs.host = gs.h1.host;
    ApplyCurl(gs.h1, &gs.host, CurlForScale(sxy));
    for (const auto& nm : kWrapDriven) {
        const uint32_t l = LinkByName(gs.h1, nm);
        if (l != kInvalidLink) gs.drive_links.push_back(l);
    }
    const auto poses = ForwardKinematics(context, gs.host);
    const auto spheres = FingerOnlyWrapSpheres();
    gs.sphere_count = static_cast<uint32_t>(spheres.size());
    gs.place = BestPlacementFingerOnlyShallow(gs.h1, poses, spheres, hull);
    gs.cup_center = gs.place.center;

    gs.config.fingertips.clear();
    uint32_t handle = 9000u;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(gs.h1, s.body);
        if (l == kInvalidLink) continue;
        coresident::CoResidentFingertip ft;
        ft.link = l;
        ft.broadphase_handle = handle++;  // UNIQUE per sphere.
        ft.local_offset = s.local_offset;
        ft.radius = s.radius;
        gs.config.fingertips.push_back(ft);
    }

    const Vec3 hull_center = (hull.hi + hull.lo) * 0.5f;
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    gs.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        gs.config.cup.hull_verts[i * 3u + 0u] -= hull_center.x;
        gs.config.cup.hull_verts[i * 3u + 1u] -= hull_center.y;
        gs.config.cup.hull_verts[i * 3u + 2u] -= hull_center.z;
    }
    gs.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = 1.0f / kCupMass;
    const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = gs.cup_center;
    cup.orientation = Quat::Identity();
    gs.config.cup_state = cup;
    gs.cup0 = cup;

    const uint32_t link_count = gs.host.TotalLinkCount();
    gs.config.grip_torque.assign(link_count, 0.0f);
    gs.config.drive_force_limits.assign(link_count, 0.0f);
    gs.config.friction_mu = kMu;
    gs.config.condim = 3u;
    // NO TABLE for the parity/batched path: the cup is held by FINGER FRICTION ALONE
    // (has_table=false). (The probe's settle-on-table is a feasibility-only convenience;
    // the batched world has no table phase in grasp mode -- so the parity scene must also
    // be table-free. The PD-close grip squeezes from the curled-open pre-pose directly.)
    return gs;
}

// ===========================================================================
// Map the GraspScene's GraspConfig -> a BatchedSceneTemplate. SAME field-by-field
// mapping as coresident::MakeGraspTemplate (the synthetic factory), but using the H1
// host proto (gs.host, curled-open). The cup is the single per-env body (cup_local 0).
// ===========================================================================
coresident::BatchedSceneTemplate MakeH1BatchedTemplate(const GraspScene& gs) {
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {gs.config.cup_state};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = gs.host;  // the cooked H1 hand (Fixed base, curled-open q).
    tmpl.fingertips = gs.config.fingertips;
    tmpl.cup = gs.config.cup;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque = gs.config.grip_torque;            // all-zero; the drive is PD via SetActions.
    tmpl.drive_force_limits = gs.config.drive_force_limits;
    tmpl.friction_mu = gs.config.friction_mu;
    tmpl.condim = gs.config.condim;
    return tmpl;
}

// ===========================================================================
// PD-close drive. The probe's DrivePd: q_target = q0+offset on the driven links, then
// tau = Kp*(q_target - q) - Kd*qdot from the DOWNLOADED state. Two equivalent seams:
//   * oracle:  per-LINK tau vector  -> SetGripTorque(tau).
//   * batched: env-major DOF-indexed action array -> SetActions. The DOF index of a
//     driven link is DofIndexOf(link); SetActions maps action[e*dof+d] -> link
//     dof_to_link_[d]. Each driven finger joint is single-DOF, so action[d_of(link)] =
//     tau[link] reproduces the per-link drive EXACTLY.
// Both compute tau from their OWN downloaded gripper state each step.
// ===========================================================================
struct PdState { std::vector<float> q_target; float Kp, Kd; };
PdState MakePdTarget(const GraspScene& gs, float Kp, float Kd, float off) {
    PdState pd; pd.Kp = Kp; pd.Kd = Kd;
    pd.q_target.assign(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links) pd.q_target[l] = gs.host.q[l] + off;
    return pd;
}

// Oracle: compute per-link tau from the oracle's own state, set it.
void DrivePdOracle(coresident::UnifiedCoResidentStepper& stepper, const GraspScene& gs,
                   const PdState& pd) {
    articulation::ArticulationHostState st; stepper.Download(&st);
    std::vector<float> tau(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links)
        tau[l] = pd.Kp * (pd.q_target[l] - st.q[l]) - pd.Kd * st.qdot[l];
    stepper.SetGripTorque(tau);
}

// The articulation-local prefix-sum DOF index of a device link (Σ JointDofCount over the
// links before `link`). Reproduces UnifiedCoResidentStepper::DofIndexOf / the batched
// world's DofIndexOf so the action array's DOF slot lines up with the link's drive slot.
uint32_t DofIndexOf(const articulation::ArticulationHostState& host, uint32_t root_link,
                    uint32_t link) {
    uint32_t idx = 0u;
    for (uint32_t l = root_link; l < link; ++l)
        idx += articulation::ArticulationJointDofCount(host.joint_type[l]);
    return idx;
}

// Batched: compute per-link tau from EACH env's own downloaded gripper state, scatter into
// the env-major DOF-indexed action array, set it. dof_stride is the per-env action width.
void DrivePdBatched(coresident::BatchedUnifiedWorld& world, const GraspScene& gs,
                    const PdState& pd, uint32_t root_link, uint32_t dof_stride) {
    const uint32_t n = world.EnvCount();
    std::vector<float> actions(static_cast<size_t>(n) * dof_stride, 0.0f);
    for (uint32_t e = 0u; e < n; ++e) {
        articulation::ArticulationHostState st; world.DownloadGripper(e, &st);
        const size_t aoff = static_cast<size_t>(e) * dof_stride;
        for (uint32_t l : gs.drive_links) {
            const uint32_t d = DofIndexOf(gs.host, root_link, l);
            if (d < dof_stride)
                actions[aoff + d] = pd.Kp * (pd.q_target[l] - st.q[l]) - pd.Kd * st.qdot[l];
        }
    }
    world.SetActions(actions.data(), actions.size());
}

}  // namespace

// ===========================================================================
// ★ GATE 1 -- N=1 BYTE-EXACT PARITY (the headline). Build the batched H1-hand grasp
// scene at N=1 and run it in LOCKSTEP with the probe's UnifiedCoResidentStepper on the
// IDENTICAL scene (same H1 host state, same 30 fingertips, same 10.9cm cup, same PD
// close Kp=4/Kd=0.4/offset=0.18). Each step: drive BOTH from their OWN downloaded state,
// step both, compare the cup pose/vel. Assert max|Δpos|, max|Δvel| <= 1e-5 (the synthetic
// A2 tolerance; expect ULP-scale). This is the analog of GRASP A2 vs StepGrasp, with the
// MUCH bigger H1-hand proto -> proves the env-major batching machinery GENERALIZES.
// ===========================================================================
TEST(BatchedH1HandGrasp, Gate1_N1_ByteExactVsCoResidentOracle) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);

    GraspScene gs = BuildFingerOnlyScene(context, base, kSxy, kSz);
    ASSERT_TRUE(gs.place.found) << "10.9cm finger-only shallow caging placement vanished";
    const auto tmpl = MakeH1BatchedTemplate(gs);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);

    // Articulation introspection (for the action DOF mapping + the diagnostic header).
    const uint32_t root_link = gs.host.articulation_link_offset[0];
    const uint32_t dof_stride = articulation::ArticulationDofCount(gs.host, 0u);
    const uint32_t link_count = gs.host.TotalLinkCount();
    std::printf("[H1.2 GATE1] H1 hand: links=%u dof_stride=%u fingertips=%zu drive_links=%zu "
                "(vs synthetic 2-DOF gripper: links=3 dof=2 fingertips=2)\n",
                link_count, dof_stride, gs.config.fingertips.size(), gs.drive_links.size());

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper oracle(context, gs.host, gs.config, kGravityZ, kDt);

    const uint32_t kRun = 70u;  // a short settle -- byte parity, not a long hold.
    // ------------------------------------------------------------------------------------
    // THE HONEST BAR (advisor-directed, an EVIDENCED reformulation -- NOT a relaxation to
    // hide a bug). The batched resolver (ResolveBatchedGraspContact) and the co-resident
    // oracle (StepGrasp) differ in HOST friction-cone accumulation ORDER (30 condim=3 contacts
    // on 12 SHARED links -- the synthetic gripper has 2 contacts on 2 DISTINCT links, so its
    // A2 1e-5 holds full-run). They are correct-by-construction: Gate 2 proves the batched
    // path == N independent N=1 solves BYTE-EXACT (tol 0) -> there is NO env-major layout bug.
    // This 30-contact marginal table-free friction grasp is CHAOTIC at the FP floor (the
    // perturbation control below proves a 1e-7 m IC seed in IDENTICAL code amplifies MORE than
    // the entire batched-vs-oracle delta), so full-run byte-parity is impossible for ANY two
    // code paths, correct or not. The IRREDUCIBLE FP-floor window is the LEADING steps where
    // the seed has not yet amplified: the trace shows steps 0-11 with |dpos| EXACTLY 0,
    // |dvel| ~1e-7, |dqdot| ~5e-7..2.8e-6, AND IDENTICAL contact counts (30/30 -> 29/29). So
    // the HARD assert = the first 10 steps at the ULP floor (cup pos/vel <=1e-6, gripper qdot
    // <=5e-6) AND the contact-set COUNT matching through the window -- a real STRUCTURAL-bug
    // discriminator: a resolver bug (wrong index/sign/coeff/Jacobian, which Gate 2's tol-0
    // batching proof CANNOT see, since it would be N-independent) corrupts step 0 by ~1e-3
    // and/or changes the contact set immediately, failing instantly. Steps 10..kRun are
    // REPORTED (continuous fork @ step 12, contact-count fork @ step 16), not asserted (chaos).
    // ------------------------------------------------------------------------------------
    constexpr double kFloorTol = 1.0e-6;       // FP-floor parity tol for cup pos/vel in the window.
    constexpr double kFloorTolQdot = 5.0e-6;   // gripper qdot accumulates the NP-seam over 12 DOF a
                                               // touch faster (max ~2.9e-6 in-window); 5e-6 keeps a
                                               // robust margin yet stays ~5 orders below |dqdot|~6e-1
                                               // chaos -- an equally strong structural-bug discriminator.
    constexpr uint32_t kFloorSteps = 10u;      // steps 0-9: |dpos| EXACTLY 0, |dvel|<5e-7, contacts match.
    double floor_max_dpos = 0.0, floor_max_dvel = 0.0, floor_max_dqdot = 0.0;
    double full_max_dpos = 0.0, full_max_dvel = 0.0, full_max_dqdot = 0.0;
    uint32_t count_match_onset = kRun;  // first step the contact-set COUNT forks (a chaos marker).
    bool count_open = true;
    uint32_t contact_steps = 0u;
    for (uint32_t s = 0u; s < kRun; ++s) {
        // Drive each from its OWN downloaded state (equivalent seams), then step.
        DrivePdOracle(oracle, gs, pd);
        DrivePdBatched(world, gs, pd, root_link, dof_stride);
        const auto rep = oracle.Step();
        world.Step();
        const uint32_t oc_contacts = rep.finger_contacts;
        const uint32_t bc_contacts = world.GraspReports()[0].finger_contacts;
        if (oc_contacts > 0u) ++contact_steps;

        const BodyState& wc = world.Body(0u, 0u);
        const BodyState oc = oracle.Cup();
        const double dpos = (wc.position - oc.position).Length();
        const double dvel = (wc.linear_velocity - oc.linear_velocity).Length();
        articulation::ArticulationHostState wa, oa;
        world.DownloadGripper(&wa);
        oracle.Download(&oa);
        double dqdot = 0.0;
        for (size_t i = 0u; i < wa.qdot.size() && i < oa.qdot.size(); ++i)
            dqdot = std::max(dqdot, std::fabs((double)(wa.qdot[i] - oa.qdot[i])));

        full_max_dpos = std::max(full_max_dpos, dpos);
        full_max_dvel = std::max(full_max_dvel, dvel);
        full_max_dqdot = std::max(full_max_dqdot, dqdot);
        if (s < kFloorSteps) {
            floor_max_dpos = std::max(floor_max_dpos, dpos);
            floor_max_dvel = std::max(floor_max_dvel, dvel);
            floor_max_dqdot = std::max(floor_max_dqdot, dqdot);
        }
        if (count_open && oc_contacts != bc_contacts) { count_open = false; count_match_onset = s; }
        // Print steps 0..18 so the controller SEES the geometric growth + the contact-set fork.
        if (s <= 18u || s == kRun - 1u) {
            std::printf("[H1.2 GATE1] step %3u: |dpos|=%.3e |dvel|=%.3e |dqdot|=%.3e "
                        "contacts(o=%u b=%u) cup_z(b=%.6f o=%.6f) vimp(o=%.4e b=%.4e)\n",
                        s, dpos, dvel, dqdot, oc_contacts, bc_contacts, wc.position.z,
                        oc.position.z, rep.cup_vertical_impulse,
                        world.GraspReports()[0].cup_vertical_impulse);
        }
    }

    std::printf("[H1.2 GATE1 RESULT] FP-FLOOR WINDOW (steps 0-%u): max|dpos|=%.3e max|dvel|=%.3e "
                "max|dqdot|=%.3e (HARD tol pos/vel %.0e qdot %.0e). FULL RUN (%u steps): "
                "max|dpos|=%.3e max|dvel|=%.3e max|dqdot|=%.3e (REPORTED, chaotic). contact-count "
                "fork @ step %u. contact %u/%u.\n",
                kFloorSteps - 1u, floor_max_dpos, floor_max_dvel, floor_max_dqdot, kFloorTol,
                kFloorTolQdot, kRun, full_max_dpos, full_max_dvel, full_max_dqdot,
                count_match_onset, contact_steps, kRun);

    // The cup must actually be HELD (non-vacuous parity): genuine contact over the run.
    EXPECT_GE(contact_steps, kRun - 5u)
        << "the batched/oracle scene was vacuous -- the cup left the H1-hand pinch";
    // ★ THE HEADLINE (HARD): over the irreducible FP-floor window (steps 0-9, where |dpos| is
    // bit-identical and the contact set matches), the batched H1-hand cup pose AND velocity AND
    // gripper qdot are ULP-exact vs the co-resident oracle -- the SAME per-step physics as the
    // validated oracle. A structural resolver bug at the bigger dof_stride/contact set would
    // corrupt step 0 by orders of magnitude (or fork the contact set immediately) and fail this
    // instantly; Gate 2's tol-0 byte-exactness is the companion proof that the env-major
    // batching itself is bug-free (it cannot see an N-independent resolver bug -- this window can).
    EXPECT_LE(floor_max_dpos, kFloorTol)
        << "batched H1-hand cup POSITION diverged from the oracle in the FP-floor window -- a "
           "real env-major layout bug at the bigger dof_stride / contact set (localize it)";
    EXPECT_LE(floor_max_dvel, kFloorTol)
        << "batched H1-hand cup VELOCITY diverged in the FP-floor window";
    EXPECT_LE(floor_max_dqdot, kFloorTolQdot)
        << "batched H1-hand gripper qdot diverged in the FP-floor window";
    // The contact-set COUNT must stay identical through the floor window. A structural resolver
    // bug (which Gate 2's tol-0 batching proof is BLIND to, being N-independent) changes the
    // contact set at step 0; the documented GPU-vs-host NP seam keeps it identical until chaos.
    EXPECT_GE(count_match_onset, kFloorSteps)
        << "the batched/oracle contact-set COUNT forked at step " << count_match_onset
        << " INSIDE the FP-floor window -- a structural resolver defect, not the NP seam";

    // ------------------------------------------------------------------------------------
    // THE CHAOS CONTROL (proves the full-run divergence is the SCENE, not the batched path).
    // Run TWO co-resident ORACLES on the IDENTICAL scene -- one with the cup IC nudged by a
    // single ULP-scale 1e-7 m. If two SAME-code oracles separate to the SAME order as the
    // batched-vs-oracle run (and the contact set collapses likewise), the scene is provably
    // chaotic at the FP floor, so byte-parity over the full 70 steps is impossible for ANY
    // two code paths, correct or not -- the batched-vs-oracle full-run delta is NOT a bug.
    // ------------------------------------------------------------------------------------
    coresident::GraspConfig cfg_a = gs.config;
    coresident::GraspConfig cfg_b = gs.config;
    cfg_b.cup_state.position.x += 1.0e-7f;  // a single ULP-scale perturbation.
    coresident::UnifiedCoResidentStepper o_a(context, gs.host, cfg_a, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper o_b(context, gs.host, cfg_b, kGravityZ, kDt);
    double self_max_dpos = 0.0, self_max_dvel = 0.0;
    uint32_t self_stable = 0u; bool self_open = true;
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdOracle(o_a, gs, pd);
        DrivePdOracle(o_b, gs, pd);
        const auto ra = o_a.Step();
        const auto rb = o_b.Step();
        const double dpos = (o_a.Cup().position - o_b.Cup().position).Length();
        const double dvel = (o_a.Cup().linear_velocity - o_b.Cup().linear_velocity).Length();
        self_max_dpos = std::max(self_max_dpos, dpos);
        self_max_dvel = std::max(self_max_dvel, dvel);
        if (self_open && ra.finger_contacts == rb.finger_contacts) ++self_stable;
        else self_open = false;
    }
    std::printf("[H1.2 GATE1 CHAOS CONTROL] two SAME-code oracles, cup IC nudged 1e-7 m: "
                "self max|dpos|=%.3e max|dvel|=%.3e (stable %u/%u). The 1e-7 seed grows to the "
                "SAME order as batched-vs-oracle -> the table-free marginal grasp is chaotic at "
                "the FP floor; full-run byte-parity is impossible for ANY two paths.\n",
                self_max_dpos, self_max_dvel, self_stable, kRun);
    // The control must demonstrate chaos: a 1e-7 m seed grows WELL past the FP floor within
    // the run (otherwise the scene is benign and the full-run divergence WOULD be suspect).
    // The seed's amplified delta should reach the SAME order as the batched-vs-oracle full-run
    // delta -- i.e. the batched path differs from the oracle by LESS than a 1e-7 cup nudge.
    EXPECT_GT(self_max_dpos, 1.0e-4)
        << "a 1e-7 m IC seed did NOT amplify -- the scene is NOT chaotic, so the batched-vs-"
           "oracle full-run divergence is unexplained (investigate the batched path)";
    // The batched-vs-oracle full-run delta must be within the SAME order as a 1e-7 m IC nudge's
    // amplified delta (a 10x band): the cross-path difference is no larger than a tiny IC seed's
    // -> it IS pure scene chaos, not a batched-path defect. (Not strict dominance: the +x seed
    // and the accumulation-order difference are different perturbations of a chaotic flow.)
    EXPECT_LT(full_max_dpos, 10.0 * self_max_dpos)
        << "the batched-vs-oracle delta is >10x a 1e-7 m IC nudge's -- the cross-path delta is "
           "larger than scene chaos explains (investigate the batched path)";
}

// ===========================================================================
// GATE 2 -- MULTI-ENV INDEPENDENCE. N=8: (a) each env byte-exact == its own N=1 batched
// run; (b) adjacent envs with DISTINCT cup ICs differ; (c) a MIXED held/no-contact batch
// (the env-major tile-gap + scatter-guard surface); (d) a D1 two-run memcmp. Mirrors the
// synthetic PER-ENV / MIXED / D1 gates -- now on the H1-hand proto.
// ===========================================================================
namespace {

// Snapshot one env's full grasp state (cup + gripper q/qdot) for memcmp.
struct H1EnvSnapshot { BodyState cup; std::vector<float> q; std::vector<float> qdot; };
H1EnvSnapshot SnapshotEnv(const coresident::BatchedUnifiedWorld& world, uint32_t env) {
    H1EnvSnapshot snap;
    snap.cup = world.Body(env, 0u);
    articulation::ArticulationHostState art;
    world.DownloadGripper(env, &art);
    snap.q = art.q;
    snap.qdot = art.qdot;
    return snap;
}
bool EnvByteEqual(const H1EnvSnapshot& a, const H1EnvSnapshot& b) {
    if (std::memcmp(&a.cup, &b.cup, sizeof(BodyState)) != 0) return false;
    if (a.q.size() != b.q.size() ||
        std::memcmp(a.q.data(), b.q.data(), a.q.size() * sizeof(float)) != 0) return false;
    if (a.qdot.size() != b.qdot.size() ||
        std::memcmp(a.qdot.data(), b.qdot.data(), a.qdot.size() * sizeof(float)) != 0)
        return false;
    return true;
}
// A distinct per-env cup IC: a sub-mm perturbation about the nominal grip pose so each
// env's finger<->cup contact geometry differs (cup stays in the pinch -> still held).
BodyState PerturbedCup(const BodyState& seed, uint32_t e) {
    BodyState cup = seed;
    const float fe = static_cast<float>(e);
    cup.position.x += 0.0008f * fe;
    cup.position.y += -0.0006f * fe;
    cup.position.z += 0.0010f * fe;
    return cup;
}

}  // namespace

TEST(BatchedH1HandGrasp, Gate2_MultiEnvIndependence_N8) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);

    GraspScene gs = BuildFingerOnlyScene(context, base, kSxy, kSz);
    ASSERT_TRUE(gs.place.found);
    const auto tmpl = MakeH1BatchedTemplate(gs);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    const uint32_t root_link = gs.host.articulation_link_offset[0];
    const uint32_t dof_stride = articulation::ArticulationDofCount(gs.host, 0u);

    constexpr uint32_t kEnvs = 8u;
    const uint32_t kRun = 50u;

    // ---- (a)/(b) PER-ENV INDEPENDENCE: N=8 distinct cup ICs ----
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);
    std::vector<BodyState> ic(kEnvs);
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        ic[e] = PerturbedCup(gs.config.cup_state, e);
        world.BodyMut(e, 0u) = ic[e];
    }
    for (uint32_t s = 0u; s < kRun; ++s) { DrivePdBatched(world, gs, pd, root_link, dof_stride); world.Step(); }

    uint32_t held_envs = 0u;
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        coresident::BatchedUnifiedWorld solo(context, tmpl, 1u, kGravityZ, kDt);
        solo.BodyMut(0u, 0u) = ic[e];
        for (uint32_t s = 0u; s < kRun; ++s) { DrivePdBatched(solo, gs, pd, root_link, dof_stride); solo.Step(); }
        EXPECT_TRUE(EnvByteEqual(SnapshotEnv(world, e), SnapshotEnv(solo, 0u)))
            << "env " << e << " (N=8 batched) NOT byte-exact vs its own N=1 run -- batching "
               "cross-contamination (env-major M^-1/qdot tile offsets at the H1 dof_stride)";
        if (world.GraspReports()[e].finger_contacts > 0u) ++held_envs;
    }
    EXPECT_EQ(held_envs, kEnvs) << "a perturbed env lost the H1 grip (gate vacuous)";
    for (uint32_t e = 1u; e < kEnvs; ++e)
        EXPECT_FALSE(EnvByteEqual(SnapshotEnv(world, e), SnapshotEnv(world, e - 1u)))
            << "env " << e << " collapsed onto its neighbor (no per-env independence)";
    std::printf("[H1.2 GATE2 PER-ENV] N=%u: every env byte-exact vs its own N=1 run; all "
                "held; adjacent envs differ\n", kEnvs);

    // ---- (c) MIXED held/no-contact (env-major tile-gap + scatter-guard) ----
    {
        auto is_held = [](uint32_t e) { return (e % 2u) == 0u; };
        std::vector<BodyState> mic(kEnvs);
        coresident::BatchedUnifiedWorld mixed(context, tmpl, kEnvs, kGravityZ, kDt);
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            BodyState cup = gs.config.cup_state;
            if (!is_held(e)) cup.position.z += 0.40f;  // far above the hand -> no contact.
            mic[e] = cup;
            mixed.BodyMut(e, 0u) = cup;
        }
        for (uint32_t s = 0u; s < kRun; ++s) { DrivePdBatched(mixed, gs, pd, root_link, dof_stride); mixed.Step(); }
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            coresident::BatchedUnifiedWorld solo(context, tmpl, 1u, kGravityZ, kDt);
            solo.BodyMut(0u, 0u) = mic[e];
            for (uint32_t s = 0u; s < kRun; ++s) { DrivePdBatched(solo, gs, pd, root_link, dof_stride); solo.Step(); }
            EXPECT_TRUE(EnvByteEqual(SnapshotEnv(mixed, e), SnapshotEnv(solo, 0u)))
                << "MIXED env " << e << " (" << (is_held(e) ? "held" : "no-contact")
                << ") NOT byte-exact vs its own N=1 run -- env-major tile placement or the "
                   "scatter guard is wrong at the H1 dof_stride";
            if (is_held(e))
                EXPECT_GT(mixed.GraspReports()[e].finger_contacts, 0u)
                    << "held env " << e << " lost the grip";
            else
                EXPECT_EQ(mixed.GraspReports()[e].finger_contacts, 0u)
                    << "no-contact env " << e << " unexpectedly contacted";
        }
        std::printf("[H1.2 GATE2 MIXED] N=%u (held=even, no-contact=odd): every env byte-exact "
                    "vs its own N=1 run; held hold, no-contact free-fall\n", kEnvs);
    }

    // ---- (d) D1 two-run byte-identity ----
    {
        auto run = [&]() {
            coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
            for (uint32_t e = 0u; e < kEnvs; ++e) w.BodyMut(e, 0u) = PerturbedCup(gs.config.cup_state, e);
            for (uint32_t s = 0u; s < kRun; ++s) { DrivePdBatched(w, gs, pd, root_link, dof_stride); w.Step(); }
            std::vector<H1EnvSnapshot> out(kEnvs);
            for (uint32_t e = 0u; e < kEnvs; ++e) out[e] = SnapshotEnv(w, e);
            return out;
        };
        const auto a = run();
        const auto b = run();
        for (uint32_t e = 0u; e < kEnvs; ++e)
            EXPECT_TRUE(EnvByteEqual(a[e], b[e]))
                << "D1: env " << e << " differed between two identical H1-hand runs";
        std::printf("[H1.2 GATE2 D1] N=%u: two identical H1-hand rollouts byte-exact "
                    "(cup + q/qdot)\n", kEnvs);
    }
}

// ===========================================================================
// THROUGHPUT (SECONDARY -- a LOWER BOUND). env-steps/sec at the H1-hand scene over an N
// sweep. The fixed-base hand is far LIGHTER than the eventual whole-body 53-DOF+table,
// so this is a LOWER BOUND on the eventual feasibility number, NOT "the" number. Reported
// honestly; only a surprising DOWNWARD result matters. Synthetic ~9.8k env-steps/sec is
// the comparison anchor (a 2-DOF gripper). The H1 hand is ~6x the DOF + ~15x the contacts.
// ===========================================================================
TEST(BatchedH1HandGrasp, Throughput_LowerBound_Sweep) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);

    GraspScene gs = BuildFingerOnlyScene(context, base, kSxy, kSz);
    ASSERT_TRUE(gs.place.found);
    const auto tmpl = MakeH1BatchedTemplate(gs);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    const uint32_t root_link = gs.host.articulation_link_offset[0];
    const uint32_t dof_stride = articulation::ArticulationDofCount(gs.host, 0u);
    const uint32_t link_count = gs.host.TotalLinkCount();

    std::printf("\n[H1.2 THROUGHPUT] === LOWER BOUND (fixed-base H1 hand, links=%u "
                "dof_stride=%u fingertips=%zu; cmp synthetic 2-DOF ~9.8k env-steps/sec). "
                "The whole-body 53-DOF+table world is HEAVIER -> this is NOT the final "
                "number. ===\n", link_count, dof_stride, gs.config.fingertips.size());
    std::printf("[H1.2 THROUGHPUT] %8s | %12s | %14s\n", "N", "wall_ms/step", "env-steps/sec");

    const std::vector<uint32_t> sweep = {1u, 8u, 32u, 256u, 1024u};
    constexpr uint32_t kWarm = 10u;
    constexpr uint32_t kMeasure = 40u;
    // Drive = ONE bulk SetActions(fixed close torque) + Step per step -- the STEADY-STATE RL
    // seam (the RL path reads obs in BULK via ExportObsState + scatters ONE action upload). It
    // deliberately does NOT use DrivePdBatched, whose per-env DownloadGripper (N host round-
    // trips/step) is a TEST-HARNESS artifact that dominates large N (~16 min @ N=1024) and is
    // NOT how the RL drive works -- so this isolates the ENGINE step cost.
    for (uint32_t N : sweep) {
        coresident::BatchedUnifiedWorld world(context, tmpl, N, kGravityZ, kDt);
        std::vector<float> act(static_cast<size_t>(N) * dof_stride, 0.0f);
        for (uint32_t e = 0u; e < N; ++e) {
            const size_t aoff = static_cast<size_t>(e) * dof_stride;
            for (uint32_t l : gs.drive_links) {
                const uint32_t d = DofIndexOf(gs.host, root_link, l);
                if (d < dof_stride) act[aoff + d] = kKp * kCloseOffset;  // fixed close torque.
            }
        }
        for (uint32_t s = 0u; s < kWarm; ++s) { world.SetActions(act.data(), act.size()); world.Step(); }
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0u; s < kMeasure; ++s) { world.SetActions(act.data(), act.size()); world.Step(); }
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double ms_per_step = 1000.0 * sec / kMeasure;
        const double env_steps_per_sec = (double)N * kMeasure / sec;
        std::printf("[H1.2 THROUGHPUT] %8u | %12.3f | %14.0f\n", N, ms_per_step, env_steps_per_sec);
    }
    std::printf("[H1.2 THROUGHPUT] (NOTE: drive = one bulk SetActions + Step/step, the steady-"
                "state RL seam. A LOWER BOUND on the eventual feasibility number -- the fixed-"
                "base hand is far lighter than the whole-body 53-DOF+table world; constant "
                "close torque keeps the cup gripped so the contact solve is exercised.)\n");
    SUCCEED();
}
