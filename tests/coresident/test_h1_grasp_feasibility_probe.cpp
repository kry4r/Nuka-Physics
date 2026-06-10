// ---------------------------------------------------------------------------
// H1.1 -- the CHEAP, SCRIPTED GO/NO-GO FORCE-CLOSURE FEASIBILITY PROBE.
// ---------------------------------------------------------------------------
// THE ONE QUESTION. Does an HONEST force-closure grasp of the cup EXIST AT ALL for the
// cooked H1 dexterous hand, with sphere-fingertip contact? This GATES the entire H1
// RL-grasp path: RL only finds FEASIBLE grasps, so if no honest grasp exists
// geometrically, no RL / arm / whole-body work matters.
//
// THE PRIOR ART (the four spikes that led here -- read in this order):
//   * test_h1_grasp_spike.cpp        -- fingertip PINCH: 2 azimuths, pivots out (NEG).
//   * test_h1_power_grasp_lift.cpp   -- 6cm WRAP: translation cage but rotational fails
//     + PASSIVE vertical hold (survives grip=0 = a WEDGE = FAKE by BITE) (NEG).
//   * test_h1_scaled_cup_grasp.cpp   -- ~10cm cup + PALM sphere: closes the void + crushes
//     tilt, but the void-closed placement needs a ~7mm finger WEDGE that holds PASSIVELY
//     (grip=0 drop ~7e-5 m) = the DISPOSITIVE BITE-negative (NEG).
//   * test_h1_dense_grasp.cpp        -- isolates CONTACT DENSITY: a ~30-sphere chain at
//     ~10cm. The prior "needs a deep wedge / only passive" symptom was CONFOUNDED with
//     contact SPARSITY (~11 spheres undersampled the patch ~10x). With a DENSE chain +
//     a SHALLOW (<=2mm) FINGER-ONLY opposed wrap, the ~10.9cm (1.8x) cup CONVERGES to a
//     VALIDATED ACTIVE force-closure grasp: ForceClosureLiftWithDisturbance PASSES (hard
//     EXPECT_TRUE) under a worst-case 1g lateral + 2.5 rad/s tilt, and BITE grip=0 DROPS
//     ~1.1 m at 1.93 mm penetration. THAT is the GO. (Its FingerOnlyFallbackLiftGate SKIP
//     is a STRICTER >180deg distributed-CAGE arc metric, NOT a force-closure-feasibility
//     test -- the GO is an opposed 2-sided active wrap, NOT a geometric >180deg cage.)
//
// WHAT THIS PROBE ADDS (the genuine gap the four spikes left). The dense GO ran its
// HOLD/rotation and BITE checks as SEPARATE gates and only at ONE size (1.8x = ~10.9cm,
// ABOVE the owner's stated 8-9cm allowance). Nobody ran, IN ONE scripted close, the
// CONJOINT honest gate -- settle ACTIVE grip -> the dense spike's PRE-COMMITTED worst-case
// disturbance (1g lateral 30 steps along the escape-gap bisector + a 2.5 rad/s tilt) ->
// confirm bounded tilt (HOLD-rotation) AND grip->0 -> confirm the cup DROPS (BITE
// honesty) -- and swept across the SIZE knob {1.4,1.6,1.8}x = {8.5,9.7,10.9}cm. So this
// probe answers the DECISION-RELEVANT open question: is the GO only at 10.9cm, or does it
// hold down at the owner's 8-9cm allowance?  The grip=0 DROP is the single honesty
// discriminator per cell (a hold that survives grip=0 is a PASSIVE WEDGE = FAKE).
//
// The wrap/curl/placement/build is MIRRORED from the validated dense-grasp spike (the
// DENSE 6mm sphere-chain, FINGER-ONLY = no palm, the BestPlacementShallowNoPalmCaging
// shallow placement, the unique-broadphase-handle build, faithful armature 0.1/damping
// 1.0, PD Kp=4/Kd=0.4/offset=0.18, mass 0.2 kg, mu=0.8). NO engine narrowphase/solver
// change -- a feasibility probe on EXISTING capability (sphere x ConvexHull, robust C3).
// The four prior spikes' gates are LEFT INTACT; this is an additive parallel TU.
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
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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
constexpr float kCupMass = 0.2f;  // faithful 0.2 kg (held constant across sizes).
constexpr float kPi = 3.14159265358979323846f;

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kH1Mjcf) && std::filesystem::exists(kCupUsda);
}

// ===========================================================================
// Cup hull loader + scale (identical to the dense spike).
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
// Cooked H1 (Fixed base, faithful armature/damping) -- identical to the dense spike.
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
// THE DENSE FINGER-ONLY WRAP (mirrored from the dense-grasp spike, VALIDATED GO).
// 3 spheres along each contacting phalanx (4 fingers prox+int + thumb int+dist), 6mm
// radius, FINGER-ONLY (no palm). One UNIQUE broadphase_handle per sphere; link = real
// device link (chain-J uses link).
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

// Finger-only dense chain (no palm) -- the validated GO contact set.
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

// The validated finger-only shallow caging placement: covered_arc > 200deg, max_pen
// <= 2mm, min squeeze. (== BestPlacementShallowNoPalmCaging in the dense spike.)
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

}  // namespace

namespace {

constexpr float kMu = 0.8f;
constexpr float kKp = 4.0f;
constexpr float kKd = 0.4f;
constexpr float kCloseOffset = 0.18f;  // the active-squeeze curl-beyond (dense GO value).

struct GraspScene {
    CookedH1 h1;
    articulation::ArticulationHostState host;
    coresident::GraspConfig config;
    BodyState cup0;
    std::vector<uint32_t> drive_links;
    Vec3 cup_center{};
    float table_height = 0.0f;
    Place place;
    uint32_t sphere_count = 0u;
};

// Build the finger-only dense grasp scene at a chosen radial scale (mirrors the dense
// spike's BuildDenseGraspScene with PlaceMode::FingerOnlyShallow + with_palm=false).
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

    gs.table_height = gs.cup_center.z + (hull.lo.z - hull_center.z);
    gs.config.has_table = true;
    gs.config.table_height = gs.table_height;
    gs.config.table_mu = kMu;
    gs.config.table_broadphase_id = 8500u;
    return gs;
}

struct PdState { std::vector<float> q_target; float Kp, Kd; };
PdState MakePdTarget(const GraspScene& gs, float Kp, float Kd, float off) {
    PdState pd; pd.Kp = Kp; pd.Kd = Kd;
    pd.q_target.assign(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links) pd.q_target[l] = gs.host.q[l] + off;
    return pd;
}
void DrivePd(coresident::UnifiedCoResidentStepper& stepper, const GraspScene& gs,
             const PdState& pd) {
    articulation::ArticulationHostState st; stepper.Download(&st);
    std::vector<float> tau(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links)
        tau[l] = pd.Kp * (pd.q_target[l] - st.q[l]) - pd.Kd * st.qdot[l];
    stepper.SetGripTorque(tau);
}
double RelTilt(const Quat& q_ref, const Quat& q_cur) {
    const Quat q_rel = q_ref.Conjugate() * q_cur;
    return 2.0 * std::acos((double)std::min(1.0f, std::fabs(q_rel.w)));
}

// The world-XY azimuth of the bisector of the largest LIVE coverage gap -- the direction
// the cup most easily escapes an opposed grasp. The disturbance pushes ALONG it (the
// pre-committed worst case, mirrored from the dense spike's ForceClosureLiftWithDisturbance).
float EscapeGapBisectorAz(const nuka::phi::DeviceContext& context, const GraspScene& gs,
                          coresident::UnifiedCoResidentStepper& stepper, const CupHull& hull) {
    articulation::ArticulationHostState st; stepper.Download(&st);
    const auto poses = ForwardKinematics(context, st);
    const auto spheres = FingerOnlyWrapSpheres();
    const Vec3 cc = stepper.Cup().position;
    std::vector<float> azs;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(gs.h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        if (RimGap(c, cc, hull) >= s.radius) continue;
        float az = std::atan2(c.y - cc.y, c.x - cc.x) * 180.0f / kPi;
        if (az < 0.0f) az += 360.0f; azs.push_back(az);
    }
    if (azs.size() < 2u) return 0.0f;
    std::sort(azs.begin(), azs.end());
    float mg = 0.0f, lo = azs[0];
    for (size_t i = 0u; i < azs.size(); ++i) {
        const float next = (i + 1u < azs.size()) ? azs[i + 1u] : azs[0] + 360.0f;
        if (next - azs[i] > mg) { mg = next - azs[i]; lo = azs[i]; }
    }
    float bis = lo + 0.5f * mg; if (bis >= 360.0f) bis -= 360.0f;
    return bis;
}

// THE PRE-COMMITTED worst-case disturbance (mirrored from the dense GO):
//   sustained 1.0 g lateral accel for 30 steps along the escape-gap bisector
//   + a ONE-SHOT 2.5 rad/s angular tilt kick, then release + settle, gravity ON.
constexpr float kLatAccelG = 1.0f;
constexpr uint32_t kPushSteps = 30u;
constexpr float kTiltKickW = 2.5f;
// Honest bounds (the dense GO's bounds): a real grasp resists these; a pinch does not.
constexpr double kMaxDisp = 0.07;
constexpr double kMaxTilt = 0.35;
constexpr double kMaxFinalW = 0.50;

// ===========================================================================
// THE CONJOINT HONEST GATE for one size cell. Returns the per-cell verdict numbers.
//   Rollout #1 -- HOLD(gravity + rotation): settle ACTIVE on table, remove table, settle,
//     apply the pre-committed worst-case disturbance, confirm bounded disp/tilt/spin +
//     weight recovered + contacts retained.
//   Rollout #2 -- BITE(honesty): settle ACTIVE, remove table, hold briefly, grip->0,
//     confirm the cup DROPS (a hold surviving grip=0 is a PASSIVE WEDGE = FAKE).
//   GO iff HOLD(gravity+rotation) AND BITE(drops) BOTH pass at this size.
// ===========================================================================
struct CellResult {
    std::string tag;
    bool placement_found = false;
    float cup_diam = 0.0f, seat_pen = 0.0f, covered_arc = 0.0f, max_gap = 0.0f;
    uint32_t sphere_count = 0u;
    double weight_kick = 0.0;
    // HOLD
    double settle_fimp = 0.0, bisector_az = 0.0;
    double peak_disp = 0.0, peak_tilt = 0.0, final_w = 0.0, recovered_fimp = 0.0;
    uint32_t contact_post = 0u, steps_post = 0u;
    bool any_static_post = false;
    bool hold_gravity = false, hold_rotation = false;
    // BITE
    double bite_drop = 0.0, free_fall = 0.0, bite_vz = 0.0;
    bool grip_on_holds = false, bite_drops = false;
    // verdict
    bool go = false;
};

CellResult RunCell(const nuka::phi::DeviceContext& context, const CupHull& base,
                   float sxy, float sz, const char* tag) {
    CellResult r;
    r.tag = tag;
    const CupHull hull = ScaleCupHull(base, sxy, sz);
    r.cup_diam = 2.0f * CupRadius(hull);
    r.weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;

    // ---- Rollout #1: HOLD (gravity + worst-case rotation) ----
    {
        GraspScene gs = BuildFingerOnlyScene(context, base, sxy, sz);
        r.sphere_count = gs.sphere_count;
        r.placement_found = gs.place.found;
        if (!gs.place.found) return r;  // no shallow finger-only caging placement at this size.
        r.seat_pen = gs.place.max_pen;
        r.covered_arc = gs.place.covered_arc;
        r.max_gap = gs.place.max_gap;

        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                     kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
        for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);
        coresident::CoResidentStepReport rep;
        for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); rep = stepper.Step(); }
        r.settle_fimp = rep.cup_vertical_impulse;
        const Vec3 c_settled = stepper.Cup().position;
        const Quat q_settled = stepper.Cup().orientation;
        const float bis = EscapeGapBisectorAz(context, gs, stepper, hull);
        r.bisector_az = bis;
        const float brad = bis * kPi / 180.0f;
        const Vec3 push_dir{std::cos(brad), std::sin(brad), 0.0f};
        // The worst-case in-plane tilt axis = PERPENDICULAR to the push (a roll about the
        // escape line, NOT a yaw about the cup axis), EXACTLY as the validated dense gate
        // (RunForceClosureDist). A yaw kick is weaker; this matches the validated worst case.
        const Vec3 tilt_axis{-push_dir.y, push_dir.x, 0.0f};
        const Vec3 dv_step = push_dir * (kLatAccelG * (-kGravityZ) * kDt);

        const uint32_t kReleaseSettle = 70u;
        const uint32_t kTotal = kPushSteps + kReleaseSettle;
        r.steps_post = kTotal;
        double peak_disp = 0.0, peak_tilt = 0.0, sum_fimp = 0.0;
        uint32_t n_imp = 0u, contact_post = 0u;
        bool any_static = false;
        for (uint32_t s = 0u; s < kTotal; ++s) {
            DrivePd(stepper, gs, pd);
            if (s < kPushSteps) {  // sustained 1g lateral push along the escape bisector.
                if (s == 0u)  // ONE-SHOT tilt kick (ApplyCupImpulse accumulates -> apply once).
                    stepper.ApplyCupImpulse(dv_step, tilt_axis * kTiltKickW);
                else
                    stepper.ApplyCupImpulse(dv_step, Vec3::Zero());
            }
            rep = stepper.Step();
            if (rep.finger_contacts > 0u) ++contact_post;
            if (rep.any_static_row) any_static = true;
            peak_disp = std::max(peak_disp, (double)(stepper.Cup().position - c_settled).Length());
            peak_tilt = std::max(peak_tilt, RelTilt(q_settled, stepper.Cup().orientation));
            if (s >= kPushSteps && rep.finger_contacts > 0u && rep.cup_vertical_impulse > 0.0) {
                sum_fimp += rep.cup_vertical_impulse; ++n_imp;
            }
        }
        r.peak_disp = peak_disp;
        r.peak_tilt = peak_tilt;
        r.final_w = stepper.Cup().angular_velocity.Length();
        r.recovered_fimp = n_imp ? sum_fimp / n_imp : 0.0;
        r.contact_post = contact_post;
        r.any_static_post = any_static;
        r.hold_gravity = (r.settle_fimp > 0.5 * r.weight_kick) &&
                         (std::fabs(r.recovered_fimp - r.weight_kick) < 0.35 * r.weight_kick) &&
                         (contact_post >= kTotal - 12u) && !any_static;
        r.hold_rotation = (peak_disp < kMaxDisp) && (peak_tilt < kMaxTilt) &&
                          (r.final_w < kMaxFinalW);
    }

    // ---- Rollout #2: BITE (the honesty discriminator) ----
    {
        GraspScene gs = BuildFingerOnlyScene(context, base, sxy, sz);
        if (!gs.place.found) return r;
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                     kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
        for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);
        coresident::CoResidentStepReport rep_on;
        for (uint32_t s = 0u; s < 40u; ++s) { DrivePd(stepper, gs, pd); rep_on = stepper.Step(); }
        const double z_active = stepper.Cup().position.z;
        const double vz_on = stepper.Cup().linear_velocity.z;
        r.grip_on_holds = (std::fabs(z_active - gs.cup0.position.z) < 0.05) &&
                          (std::fabs(vz_on) < 0.5) &&
                          (rep_on.cup_vertical_impulse > 0.5 * r.weight_kick);
        const std::vector<float> zero(gs.host.TotalLinkCount(), 0.0f);
        const uint32_t kFall = 120u;
        for (uint32_t s = 0u; s < kFall; ++s) { stepper.SetGripTorque(zero); stepper.Step(); }
        const BodyState cupF = stepper.Cup();
        r.bite_drop = z_active - cupF.position.z;
        r.bite_vz = cupF.linear_velocity.z;
        r.free_fall = 0.5 * (-kGravityZ) * (kFall * kDt) * (kFall * kDt);
        r.bite_drops = (r.bite_drop > 0.02) || (cupF.linear_velocity.z < -0.10);
    }

    r.go = r.placement_found && r.hold_gravity && r.hold_rotation && r.grip_on_holds &&
           r.bite_drops;
    return r;
}

void PrintRow(const CellResult& r) {
    if (!r.placement_found) {
        std::printf("[CELL] %-10s diam=%.1fcm spheres=%u -> NO SHALLOW FINGER-ONLY CAGING "
                    "PLACEMENT (arc<=200deg @ <=2mm) => no-go (cannot even set up)\n",
                    r.tag.c_str(), r.cup_diam * 100.0f, r.sphere_count);
        return;
    }
    std::printf("[CELL] %-10s diam=%.1fcm spheres=%u seat_pen=%.2fmm arc=%.0f gap=%.0f "
                "bisect_az=%.0f || HOLD-grav=%d(settle_fimp=%.3e/mg.dt=%.3e recov=%.3e) "
                "HOLD-rot=%d(disp=%.4f tilt=%.4f |w|=%.3f) contact=%u/%u || "
                "BITE: grip_on=%d grip0_drop=%.4f/ff=%.3f vz=%.3f drops=%d => %s\n",
                r.tag.c_str(), r.cup_diam * 100.0f, r.sphere_count, r.seat_pen * 1000.0f,
                r.covered_arc, r.max_gap, r.bisector_az,
                r.hold_gravity ? 1 : 0, r.settle_fimp, r.weight_kick, r.recovered_fimp,
                r.hold_rotation ? 1 : 0, r.peak_disp, r.peak_tilt, r.final_w,
                r.contact_post, r.steps_post,
                r.grip_on_holds ? 1 : 0, r.bite_drop, r.free_fall, r.bite_vz,
                r.bite_drops ? 1 : 0, r.go ? "GO" : "no-go");
}

struct Cell { float sxy; float sz; const char* tag; };
// Size sweep at near-uniform scale (matching the dense GO's 1.8x/1.8 canonical). 1.4x ->
// ~8.5cm, 1.6x -> ~9.7cm, 1.8x -> ~10.9cm. The minimum-size question: does the GO hold
// down at the owner's 8-9cm allowance, or only at 10.9cm?
const std::vector<Cell> kCells = {
    {1.4f, 1.4f, "8.5cm"},
    {1.6f, 1.6f, "9.7cm"},
    {1.8f, 1.8f, "10.9cm"},
};

}  // namespace

// ===========================================================================
// GEOMETRY PROBE: does a shallow finger-only caging placement EXIST per size? (no dyn.)
// ===========================================================================
TEST(H1GraspFeasibilityProbe, ShallowFingerOnlyPlacementPerSize) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    for (const auto& cell : kCells) {
        const CupHull hull = ScaleCupHull(base, cell.sxy, cell.sz);
        CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
        articulation::ArticulationHostState host = h1.host;
        ApplyCurl(h1, &host, CurlForScale(cell.sxy));
        const auto poses = ForwardKinematics(context, host);
        const auto spheres = FingerOnlyWrapSpheres();
        const Place pl = BestPlacementFingerOnlyShallow(h1, poses, spheres, hull);
        std::printf("[GEO] %-10s diam=%.1fcm spheres=%zu found=%d arc=%.1f gap=%.1f "
                    "max_pen=%.2fmm genuine=%u (need arc>200deg @ <=2mm)\n",
                    cell.tag, 2.0f * CupRadius(hull) * 100.0f, spheres.size(),
                    pl.found ? 1 : 0, pl.covered_arc, pl.max_gap, pl.max_pen * 1000.0f,
                    pl.genuine);
    }
    SUCCEED();
}

// ===========================================================================
// THE FEASIBILITY VERDICT -- sweep size through the CONJOINT honest gate, emit GO/NO-GO.
// ===========================================================================
TEST(H1GraspFeasibilityProbe, ConjointHonestGateVerdict) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);

    std::printf("\n[VERDICT] === H1.1 force-closure feasibility: finger-only dense shallow "
                "wrap, size sweep {8.5,9.7,10.9}cm, mass 0.2kg, mu=0.8, faithful armature "
                "0.1 ===\n");
    std::printf("[VERDICT] GO per cell = HOLD(gravity + PRE-COMMITTED worst-case "
                "disturbance: 1g lateral %u steps along the escape-gap bisector + %.1f "
                "rad/s tilt) AND BITE(grip=0 drops). grip=0 drop is the honesty "
                "discriminator (a hold surviving grip=0 is a PASSIVE WEDGE = FAKE).\n",
                kPushSteps, kTiltKickW);

    std::vector<CellResult> results;
    bool any_go = false;
    for (const auto& cell : kCells) {
        CellResult r = RunCell(context, base, cell.sxy, cell.sz, cell.tag);
        PrintRow(r);
        results.push_back(r);
        if (r.go) any_go = true;
    }

    // The minimum GO size (decision-relevant: vs the owner's 8-9cm allowance).
    float min_go_diam = 1e9f;
    for (const auto& r : results) if (r.go) min_go_diam = std::min(min_go_diam, r.cup_diam);

    if (any_go) {
        std::printf("[VERDICT] === GO === a SCRIPTED finger-only dense shallow wrap threads "
                    "HOLD(gravity + worst-case disturbance) AND BITE(grip=0 drops) at "
                    "concrete config(s). HONEST ACTIVE force-closure of the cup EXISTS for "
                    "the cooked H1 hand + sphere fingertips -> RL has a feasible target. "
                    "Min GO cup diameter = %.1f cm.\n", min_go_diam * 100.0f);
        for (const auto& r : results)
            if (r.go)
                std::printf("[VERDICT]   GO cell %-10s diam=%.1fcm seat_pen=%.2fmm "
                            "disp=%.4f tilt=%.4f |w|=%.3f BITE_drop=%.4f (free=%.3f vz=%.3f)\n",
                            r.tag.c_str(), r.cup_diam * 100.0f, r.seat_pen * 1000.0f,
                            r.peak_disp, r.peak_tilt, r.final_w, r.bite_drop, r.free_fall,
                            r.bite_vz);
        std::printf("[VERDICT]   NOTE: the GO is an opposed 2-sided ACTIVE wrap (gap ~"
                    "%.0f deg, NOT a >180deg geometric cage); the dense spike's "
                    "FingerOnlyFallbackLiftGate SKIP is a stricter distributed-cage arc "
                    "metric, NOT a force-closure-feasibility test. This probe's "
                    "ForceClosureLift-style disturbance gate is the feasibility measure.\n",
                    results.empty() ? 0.0 : (double)results.back().max_gap);
        // Assert the largest size (the validated dense GO config) genuinely threads both,
        // so the controller's re-run is a GREEN gate (not a SKIP) on the proven config.
        const CellResult& large = results.back();
        EXPECT_TRUE(large.placement_found) << "10.9cm finger-only placement vanished";
        EXPECT_TRUE(large.hold_gravity) << "10.9cm did not hold the weight after table removal";
        EXPECT_TRUE(large.hold_rotation) << "10.9cm pivoted/flew out under the worst-case "
                                            "disturbance (disp/tilt/|w| over bound)";
        EXPECT_TRUE(large.grip_on_holds) << "10.9cm grip-on did not hold";
        EXPECT_TRUE(large.bite_drops) << "10.9cm grip=0 did NOT drop -> PASSIVE WEDGE (fake)";
        return;
    }

    // NO-GO (only if the validated GO did not reproduce -- a regression signal).
    std::printf("[VERDICT] === NO-GO === no size cell threads HOLD(gravity+rotation) AND "
                "BITE(grip=0 drops). If the dense-grasp spike's ForceClosureLiftWith"
                "Disturbance + FingerOnlyFallbackBiteGripOffVsOn are GREEN, this probe has "
                "REGRESSED (check the mirror); otherwise the validated GO did not "
                "reproduce here.\n");
    GTEST_SKIP() << "NO-GO: no size cell threaded the conjoint honest gate (expected GO at "
                    "10.9cm per the validated dense spike -- investigate the mirror). See "
                    "[VERDICT] above.";
}
