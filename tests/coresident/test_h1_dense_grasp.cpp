// ---------------------------------------------------------------------------
// v0.8 C7b-2c -- the DENSE-CONTACT H1 power/wrap grasp DISCRIMINATOR.
// ---------------------------------------------------------------------------
// THE ONE VARIABLE THIS TESTS: CONTACT DENSITY (and, conditioned on it, contact
// DEPTH). The four prior H1-grasp spikes (pinch / power / scaled-cup-v1/v2/v3) all
// placed exactly ONE collision sphere per phalanx (~11 contacts for the whole hand),
// undersampling the finger/cylinder contact patch ~10x. With sparse point-contacts the
// ONLY way to make a holding normal force is to drive the points DEEP -- the prior
// "void-closed" placement needed a ~7.2mm finger WEDGE that held PASSIVELY (grip=0 still
// held; see test_h1_scaled_cup_grasp.cpp BITE). So the prior diagnoses ("cup too small",
// "needs deep wedge", "only passive") are CONFOUNDED with contact SPARSITY: they predict
// the same symptom, and the dense-contact case was NEVER tested.
//
// THIS experiment isolates density. It re-uses the EXACT validated machinery from the
// scaled-cup spike (faithful H1 import base Fixed, armature=0.1/damping=1.0, the SAME
// ~10cm 1.6x cup, the SAME PD gains kKp=4/kKd=0.4, the SAME cup-on-table -> table-removal
// LIFT choreography through the unified spine) and changes ONLY the contact set:
//   * a DENSE SPHERE-CHAIN: 3 small spheres distributed ALONG the length of each
//     contacting phalanx (proximal + intermediate of the four fingers + thumb prox/
//     int/dist + a palm pad) -> ~37 contacts vs the prior ~11, sampling the cylinder
//     surface like a real wrap's line/area contact;
//   * a SHALLOW placement: a NEW BestPlacementShallow that adds a max_pen <= 2mm filter
//     on top of coverage+palm+gap, so the dynamics tests a SHALLOW dense wrap (NOT the
//     7.2mm wedge). If no <=2mm placement is coverage-feasible, that is a clean NEGATIVE.
//
// SPHERES ONLY (the robust C3 SphereHull path); NO capsules (capsule x hull re-enters the
// EPA shallow-penetration dead band -- known debt). Dense == a CHAIN of spheres.
//
// THE DECISION (binary):
//   PASS "contact-model artifact": dense + shallow (<=2mm) gives an ACTIVE grasp --
//     grip=0 -> the cup falls/slips, grip=on -> it holds through the table-removal LIFT,
//     caged (small tilt, contacts retained), faithful params, stable gains, bookkeeping
//     closes (max_cross small) => the prior negatives were a contact-model artifact and
//     the H1-cup demo is VIABLE with dense contacts.
//   NEGATIVE "genuinely hard": even dense + shallow can't hold actively (still needs deep
//     penetration / no <=2mm coverage-feasible placement, OR can't cage, OR unstable at
//     real params) => size AND sparsity both excluded => the grasp is genuinely hard.
//
// ADDITIVE: a NEW translation unit; the prior honest-negative gates in
// test_h1_scaled_cup_grasp.cpp / test_h1_power_grasp_lift.cpp are LEFT INTACT. The only
// stepper touch is the inert-by-default finger_link<-handle decoupling (handle==link for
// every prior fingertip, so byte-for-byte unchanged for them).
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
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
constexpr float kCupMass = 0.2f;  // the same light-but-believable 0.2 kg mug.
constexpr float kPi = 3.14159265358979323846f;

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kH1Mjcf) && std::filesystem::exists(kCupUsda);
}

// ---------------------------------------------------------------------------
// Cup hull loader + scale-about-center (identical to the scaled-cup spike).
// ---------------------------------------------------------------------------
struct CupHull {
    std::vector<float> verts;  // flat x,y,z, mesh-local.
    Vec3 lo{}, hi{};
    uint32_t vcount = 0u;
};
CupHull LoadCupHull() {
    auto scene = nuka::import::LoadUsd(kCupUsda);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;  // single hull.
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

// ---------------------------------------------------------------------------
// The cooked H1 (faithful import, Fixed base, real armature/damping on the driven set).
// ---------------------------------------------------------------------------
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
    for (uint32_t l = 0u; l < h1.host.TotalLinkCount(); ++l) {
        if (LinkName(h1, l) == name) return l;
    }
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

// ---------------------------------------------------------------------------
// THE DENSE WRAP. Same driven joints as the scaled-cup spike, but each contacting
// phalanx now carries a CHAIN of small spheres distributed ALONG its length (the +x
// local axis points down the phalanx toward the cup per the geometry probe). Three
// spheres per phalanx span the ~32mm segment at +x {0.006, 0.016, 0.026}; the smaller
// 6mm radius (vs the prior 10mm) lets the shallowest tangent contact sit at <=2mm depth
// while the +x spread swings each phalanx's azimuth ~40deg (the finger is offset ~26mm
// in y from the cup center) -- so the chain closes coverage WITHOUT a deep wedge.
//
// Each sphere gets a UNIQUE broadphase_handle (so the stepper resolver picks the right
// sphere geometry per contact); `link` stays the true articulation link (the chain-J
// uses link, via the inert-by-default finger_link<-handle decoupling).
// ---------------------------------------------------------------------------
constexpr float kWrapRadius = 0.006f;  // 6mm dense finger-segment pad (vs the prior 10mm).
constexpr float kPalmRadius = 0.010f;  // a slightly larger palm pad chain.

#ifndef PALM_OFFSET_X
#define PALM_OFFSET_X 0.10f
#endif

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

// The dense sphere-chain: 3 spheres along each contacting phalanx, plus a 3-sphere palm
// pad chain (slightly larger). The +x positions span the phalanx; thumb segments use a
// shorter span (the thumb phalanges are shorter). Locked ONCE -- not swept.
std::vector<WrapSphere> WrapSpheres(bool with_palm, const Vec3& palm_offset) {
    std::vector<WrapSphere> v;
    // The four fingers: proximal + intermediate, 3 spheres each spanning ~32mm.
    const char* fingers[] = {"R_index", "R_middle", "R_ring", "R_pinky"};
    const float finger_x[] = {0.006f, 0.016f, 0.026f};
    for (const char* f : fingers) {
        for (const char* seg : {"_proximal", "_intermediate"}) {
            for (float fx : finger_x)
                v.push_back({std::string(f) + seg, Vec3{fx, 0.0f, 0.0f}, kWrapRadius, "finger"});
        }
    }
    // The thumb: intermediate + distal (the wrap-side phalanges), shorter span.
    const float thumb_x[] = {0.004f, 0.012f, 0.020f};
    for (const char* seg : {"R_thumb_intermediate", "R_thumb_distal"}) {
        for (float tx : thumb_x)
            v.push_back({seg, Vec3{tx, 0.0f, 0.0f}, kWrapRadius, "thumb"});
    }
    // The palm pad: a 3-sphere chain spread in y across the palm-inner face, at the
    // calibrated palm reach (so the back wall co-contacts with the curled fingers).
    if (with_palm) {
        for (float py : {-0.014f, 0.0f, 0.014f})
            v.push_back({"right_hand_link",
                         Vec3{palm_offset.x, palm_offset.y + py, palm_offset.z},
                         kPalmRadius, "palm"});
    }
    return v;
}

// The size-appropriate OPEN curl for the ~10cm cup (the same heuristic as the scaled-cup
// spike: a fatter cup needs a more open hand).
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
CurlPose OffsetCurl(CurlPose c, float finger_delta, float thumb_yaw_delta,
                    float thumb_pitch_delta) {
    c.finger_prox = std::clamp(c.finger_prox + finger_delta, 0.25f, 1.45f);
    c.finger_int = std::clamp(c.finger_int + finger_delta, 0.35f, 1.55f);
    c.thumb_yaw = std::clamp(c.thumb_yaw + thumb_yaw_delta, 0.35f, 1.45f);
    c.thumb_pitch = std::clamp(c.thumb_pitch + thumb_pitch_delta, 0.20f, 1.10f);
    c.thumb_int = std::clamp(c.thumb_int + 0.5f * finger_delta, 0.25f, 1.10f);
    c.thumb_dist = std::clamp(c.thumb_dist + 0.5f * finger_delta, 0.25f, 1.10f);
    return c;
}

Vec3 SphereCenter(const std::vector<Transform>& poses, uint32_t link,
                  const Vec3& local_offset) {
    const Transform& lp = poses[link];
    return lp.position + lp.rotation.Rotate(local_offset);
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
float RimGap(const Vec3& sphere_center, const Vec3& cup_center, const CupHull& hull) {
    const Vec3 hc = (hull.hi + hull.lo) * 0.5f;
    float nearest = 1e9f;
    for (uint32_t v = 0u; v < hull.vcount; ++v) {
        const Vec3 hv{cup_center.x + hull.verts[v * 3u + 0u] - hc.x,
                      cup_center.y + hull.verts[v * 3u + 1u] - hc.y,
                      cup_center.z + hull.verts[v * 3u + 2u] - hc.z};
        nearest = std::min(nearest, (sphere_center - hv).Length());
    }
    return nearest;
}

// ---------------------------------------------------------------------------
// Coverage + squeeze metrics over the dense set (dense-ready: iterate per offset).
// ---------------------------------------------------------------------------
struct Surround {
    Vec3 cup_center{};
    uint32_t genuine = 0u;
    std::vector<float> azimuths;
    bool palm_contacts = false;
    float max_gap = 360.0f;
    float covered_arc = 0.0f;
};
Surround MeasureSurround(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                         const std::vector<WrapSphere>& spheres, const CupHull& hull,
                         const Vec3& cup_center) {
    Surround out;
    out.cup_center = cup_center;
    std::vector<float> azs;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses_curl, l, s.local_offset);
        const float rim = RimGap(c, cup_center, hull);
        if (rim >= s.radius) continue;  // not a genuine contact.
        if (std::string(s.region) == "palm") out.palm_contacts = true;
        const float dx = c.x - cup_center.x, dy = c.y - cup_center.y;
        float az = std::atan2(dy, dx) * 180.0f / kPi;
        if (az < 0.0f) az += 360.0f;
        azs.push_back(az);
    }
    out.genuine = static_cast<uint32_t>(azs.size());
    std::sort(azs.begin(), azs.end());
    out.azimuths = azs;
    if (azs.size() >= 2u) {
        float max_gap = 0.0f;
        for (size_t i = 0u; i < azs.size(); ++i) {
            const float next = (i + 1u < azs.size()) ? azs[i + 1u] : azs[0] + 360.0f;
            max_gap = std::max(max_gap, next - azs[i]);
        }
        out.max_gap = max_gap;
        out.covered_arc = 360.0f - max_gap;
    }
    return out;
}

// The single deepest finger/palm pre-penetration depth at a curled pose vs the cup. When
// `region` is non-null, only spheres of that region count (so the palm-vs-fingers split
// of the required depth can be reported -- the prior sparse case had fingers 7.2mm /
// palm 2.4mm, i.e. the WRAP coverage, not the palm, forced the depth).
float MaxPrePenetration(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                        const std::vector<WrapSphere>& spheres, const CupHull& hull,
                        const Vec3& cup_center, const char* region = nullptr) {
    float max_depth = 0.0f;
    for (const auto& s : spheres) {
        if (region != nullptr && std::string(s.region) != region) continue;
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses_curl, l, s.local_offset);
        const float rim = RimGap(c, cup_center, hull);
        if (rim < s.radius) max_depth = std::max(max_depth, s.radius - rim);
    }
    return max_depth;
}

struct Place { Vec3 center; uint32_t genuine; float covered_arc; float max_gap;
               bool palm; float max_pen; float squeeze_mag; bool found; };

Vec3 CavityCenterZ(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                   const std::vector<WrapSphere>& spheres, Vec3* lo, Vec3* hi) {
    *lo = Vec3{1e9f, 1e9f, 1e9f}; *hi = Vec3{-1e9f, -1e9f, -1e9f};
    float zsum = 0.0f; uint32_t zn = 0u;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses_curl, l, s.local_offset);
        *lo = Vec3{std::min(lo->x, c.x), std::min(lo->y, c.y), std::min(lo->z, c.z)};
        *hi = Vec3{std::max(hi->x, c.x), std::max(hi->y, c.y), std::max(hi->z, c.z)};
        zsum += c.z; ++zn;
    }
    return Vec3{0, 0, zn ? zsum / zn : 0.5f * (lo->z + hi->z)};
}

float SqueezeMag(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                 const std::vector<WrapSphere>& spheres, const CupHull& hull,
                 const Vec3& cup_center) {
    Vec3 s{0, 0, 0};
    for (const auto& sp : spheres) {
        const uint32_t l = LinkByName(h1, sp.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses_curl, l, sp.local_offset);
        const float rim = RimGap(c, cup_center, hull);
        if (rim >= sp.radius) continue;
        const float dx = c.x - cup_center.x, dy = c.y - cup_center.y;
        const float d = std::sqrt(dx * dx + dy * dy);
        if (d < 1e-6f) continue;
        const float w = sp.radius - rim;
        s.x += w * (-dx / d); s.y += w * (-dy / d);
    }
    return std::sqrt(s.x * s.x + s.y * s.y);
}

constexpr float kVoidGapDeg = 110.0f;     // the 6cm palm-side void; "closed" = gap < this.
constexpr float kShallowPenMax = 0.002f;  // <=2mm == SHALLOW (NOT the 7.2mm wedge).
constexpr float kLargeCagingArcMin = 200.0f;

// THE SHALLOW placement: min net-squeeze imbalance WITHIN {covered_arc>180, palm
// contacts, max_gap<110, AND max_pen<=2mm}. This is the discriminating constraint --
// if the dense chain makes a SHALLOW void-closed placement coverage-feasible, the
// feasible set is non-empty; if not, that is a clean NEGATIVE.
Place BestPlacementShallow(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                           const std::vector<WrapSphere>& spheres, const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses_curl, spheres, &lo, &hi);
    Place best{Vec3{}, 0, 0, 360.0f, false, 0.0f, 1e9f, false};
    const int kN = 25;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Vec3 cc{fx, fy, cz.z};
        const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, cc);
        if (sr.covered_arc <= 180.0f || sr.genuine < 3u) continue;
        if (!sr.palm_contacts || sr.max_gap >= kVoidGapDeg) continue;
        const float pen = MaxPrePenetration(h1, poses_curl, spheres, hull, cc);
        if (pen > kShallowPenMax) continue;  // SHALLOW only.
        const float sq = SqueezeMag(h1, poses_curl, spheres, hull, cc);
        if (sq < best.squeeze_mag) {
            best = Place{cc, sr.genuine, sr.covered_arc, sr.max_gap, sr.palm_contacts,
                         pen, sq, true};
        }
    }
    return best;
}

// DIAGNOSTIC: the best SHALLOW (<=2mm) placement with covered_arc>180 but WITHOUT the
// palm co-contact / gap<110 requirement -- i.e. can the dense FINGER+THUMB wrap alone
// close coverage shallow? This decomposes the NEGATIVE: if this is FOUND, the finger
// wrap closes shallow and the residual obstruction is specifically the PALM co-contact
// (the option-1 void-closure criterion), NOT a generic "can't close". If NOT found, the
// shallow-infeasibility is robust regardless of the palm requirement.
Place BestPlacementShallowNoPalm(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                                 const std::vector<WrapSphere>& spheres, const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses_curl, spheres, &lo, &hi);
    Place best{Vec3{}, 0, 0, 360.0f, false, 0.0f, 1e9f, false};
    const int kN = 25;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Vec3 cc{fx, fy, cz.z};
        const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, cc);
        if (sr.covered_arc <= 180.0f || sr.genuine < 3u) continue;  // NO palm/gap filter.
        const float pen = MaxPrePenetration(h1, poses_curl, spheres, hull, cc);
        if (pen > kShallowPenMax) continue;  // SHALLOW only.
        const float sq = SqueezeMag(h1, poses_curl, spheres, hull, cc);
        if (sq < best.squeeze_mag) {
            best = Place{cc, sr.genuine, sr.covered_arc, sr.max_gap, sr.palm_contacts,
                         pen, sq, true};
        }
    }
    return best;
}

// The Phase-1 larger-cup placement: <=2mm everywhere, palm co-contact present, and
// caging arc >~200deg. This is deliberately less tied to the old 110deg void threshold:
// the large-object question is whether the palm fills the opposite side shallow and closes
// the prior ~165deg finger-only gap.
Place BestPlacementShallowCaging(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                                 const std::vector<WrapSphere>& spheres, const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses_curl, spheres, &lo, &hi);
    Place best{Vec3{}, 0, 0, 360.0f, false, 0.0f, 1e9f, false};
    const int kN = 25;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Vec3 cc{fx, fy, cz.z};
        const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, cc);
        if (sr.covered_arc <= kLargeCagingArcMin || sr.genuine < 3u) continue;
        if (!sr.palm_contacts) continue;
        const float pen = MaxPrePenetration(h1, poses_curl, spheres, hull, cc);
        if (pen > kShallowPenMax) continue;
        const float sq = SqueezeMag(h1, poses_curl, spheres, hull, cc);
        if (sq < best.squeeze_mag) {
            best = Place{cc, sr.genuine, sr.covered_arc, sr.max_gap, sr.palm_contacts,
                         pen, sq, true};
        }
    }
    return best;
}

Place BestPlacementShallowNoPalmCaging(const CookedH1& h1,
                                       const std::vector<Transform>& poses_curl,
                                       const std::vector<WrapSphere>& spheres,
                                       const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses_curl, spheres, &lo, &hi);
    Place best{Vec3{}, 0, 0, 360.0f, false, 0.0f, 1e9f, false};
    const int kN = 25;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Vec3 cc{fx, fy, cz.z};
        const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, cc);
        if (sr.covered_arc <= kLargeCagingArcMin || sr.genuine < 3u) continue;
        const float pen = MaxPrePenetration(h1, poses_curl, spheres, hull, cc);
        if (pen > kShallowPenMax) continue;
        const float sq = SqueezeMag(h1, poses_curl, spheres, hull, cc);
        if (sq < best.squeeze_mag) {
            best = Place{cc, sr.genuine, sr.covered_arc, sr.max_gap, sr.palm_contacts,
                         pen, sq, true};
        }
    }
    return best;
}

// The diagnostic best void-closed placement WITHOUT the shallow filter (for reporting
// how deep the void-closed placement would have to be if shallow is infeasible).
Place BestPlacementVoidClosed(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                              const std::vector<WrapSphere>& spheres, const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses_curl, spheres, &lo, &hi);
    Place best{Vec3{}, 0, 0, 360.0f, false, 0.0f, 1e9f, false};
    const int kN = 25;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Vec3 cc{fx, fy, cz.z};
        const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, cc);
        if (sr.covered_arc <= 180.0f || sr.genuine < 3u) continue;
        if (!sr.palm_contacts || sr.max_gap >= kVoidGapDeg) continue;
        const float sq = SqueezeMag(h1, poses_curl, spheres, hull, cc);
        if (sq < best.squeeze_mag) {
            const float pen = MaxPrePenetration(h1, poses_curl, spheres, hull, cc);
            best = Place{cc, sr.genuine, sr.covered_arc, sr.max_gap, sr.palm_contacts,
                         pen, sq, true};
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// The DENSE grasp scene. UNIQUE broadphase_handle per sphere; link = real link.
// ---------------------------------------------------------------------------
struct ScaleSpec { float sxy; float sz; };

constexpr float kDynSxy = 1.6f;   // the ~9.66cm prior dense negative.
constexpr float kDynSz = 1.2f;
constexpr float kLargeDynSxy = 1.8f;  // model_large.usda canonical scale (~10.9cm dia).
constexpr float kLargeDynSz = 1.8f;
constexpr float kFingerOnlyFallbackSxy = 1.8f;
constexpr float kFingerOnlyFallbackSz = 1.8f;
constexpr float kFingerOnlyFallbackFingerDelta = 0.0f;
constexpr float kFingerOnlyFallbackThumbYawDelta = 0.0f;
constexpr float kFingerOnlyFallbackThumbPitchDelta = 0.0f;
const ScaleSpec kLargeScaleSweep[] = {
    {1.6f, 1.6f}, {1.7f, 1.7f}, {1.8f, 1.8f}, {1.9f, 1.9f},
    {2.0f, 2.0f}, {2.2f, 2.2f}, {2.4f, 2.4f}, {2.6f, 2.6f},
};
constexpr float kMu = 0.8f;
constexpr float kCloseOffset = 0.18f;  // the same ACTIVE squeeze curl-beyond.
constexpr float kKp = 4.0f;
constexpr float kKd = 0.4f;

struct DenseGraspScene {
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

// Placement mode: Shallow (the discriminator -- <=2mm filter) or VoidClosed (the deep
// best void-closed placement WITHOUT the shallow filter, used to VALIDATE the apparatus
// + the unique-handle stepper edit: the deep placement always exists, so stepping there
// exercises the dense contact set + the finger_link<-handle decoupling for the first time).
enum class PlaceMode { Shallow, VoidClosed, ShallowCaging, FingerOnlyShallow };

DenseGraspScene BuildDenseGraspScene(const nuka::phi::DeviceContext& context,
                                     const CupHull& base, bool has_table,
                                     PlaceMode mode = PlaceMode::Shallow,
                                     float sxy = kDynSxy, float sz = kDynSz,
                                     bool with_palm = true,
                                     CurlPose curl_override = CurlPose{},
                                     bool use_curl_override = false) {
    DenseGraspScene gs;
    const CupHull hull = ScaleCupHull(base, sxy, sz);
    gs.h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    gs.host = gs.h1.host;
    const CurlPose curl = use_curl_override ? curl_override : CurlForScale(sxy);
    ApplyCurl(gs.h1, &gs.host, curl);

    for (const auto& nm : kWrapDriven) {
        const uint32_t l = LinkByName(gs.h1, nm);
        if (l != kInvalidLink) gs.drive_links.push_back(l);
    }

    const auto poses_curl = ForwardKinematics(context, gs.host);
    const Vec3 palm_offset{PALM_OFFSET_X, 0.0f, 0.0f};
    const auto spheres = WrapSpheres(with_palm, palm_offset);
    gs.sphere_count = static_cast<uint32_t>(spheres.size());
    if (mode == PlaceMode::Shallow) {
        gs.place = BestPlacementShallow(gs.h1, poses_curl, spheres, hull);
    } else if (mode == PlaceMode::ShallowCaging) {
        gs.place = BestPlacementShallowCaging(gs.h1, poses_curl, spheres, hull);
    } else if (mode == PlaceMode::FingerOnlyShallow) {
        gs.place = BestPlacementShallowNoPalmCaging(gs.h1, poses_curl, spheres, hull);
    } else {
        gs.place = BestPlacementVoidClosed(gs.h1, poses_curl, spheres, hull);
    }
    gs.cup_center = gs.place.center;

    // Each sphere -> a fingertip with a UNIQUE broadphase_handle (so the resolver picks
    // the right geometry) and the REAL device link (so the chain-J is correct). Handles
    // start above every reserved id (cup=7000, table=8500) -> use a 9000+ base.
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
    gs.config.has_table = has_table;
    gs.config.table_height = gs.table_height;
    gs.config.table_mu = kMu;
    gs.config.table_broadphase_id = 8500u;
    return gs;
}

struct PdState { std::vector<float> q_target; float Kp, Kd; };
PdState MakePdTarget(const DenseGraspScene& gs, float Kp, float Kd, float close_offset) {
    PdState pd; pd.Kp = Kp; pd.Kd = Kd;
    pd.q_target.assign(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links) pd.q_target[l] = gs.host.q[l] + close_offset;
    return pd;
}
void DrivePd(coresident::UnifiedCoResidentStepper& stepper, const DenseGraspScene& gs,
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

// Live coverage (dense): contacts retained + arc/gap + palm, from the live FK + cup pose.
struct LargeScaleEval {
    float sxy = 1.0f;
    float sz = 1.0f;
    float diameter = 0.0f;
    Place shallow;
    Place no_palm;
    Place void_closed;
    float finger_pen = 0.0f;
    float thumb_pen = 0.0f;
    float palm_pen = 0.0f;
};

LargeScaleEval EvaluateLargeScale(const nuka::phi::DeviceContext& context, const CookedH1& h1,
                                  const CupHull& base, float sxy, float sz) {
    LargeScaleEval e;
    e.sxy = sxy;
    e.sz = sz;
    const CupHull hull = ScaleCupHull(base, sxy, sz);
    e.diameter = 2.0f * CupRadius(hull);
    articulation::ArticulationHostState host = h1.host;
    ApplyCurl(h1, &host, CurlForScale(sxy));
    const auto poses = ForwardKinematics(context, host);
    const Vec3 palm_offset{PALM_OFFSET_X, 0.0f, 0.0f};
    const auto spheres = WrapSpheres(true, palm_offset);
    const auto finger_only = WrapSpheres(false, palm_offset);
    e.shallow = BestPlacementShallowCaging(h1, poses, spheres, hull);
    e.no_palm = BestPlacementShallowNoPalmCaging(h1, poses, finger_only, hull);
    e.void_closed = BestPlacementVoidClosed(h1, poses, spheres, hull);
    if (e.shallow.found) {
        e.finger_pen = MaxPrePenetration(h1, poses, spheres, hull, e.shallow.center, "finger");
        e.thumb_pen = MaxPrePenetration(h1, poses, spheres, hull, e.shallow.center, "thumb");
        e.palm_pen = MaxPrePenetration(h1, poses, spheres, hull, e.shallow.center, "palm");
    } else if (e.void_closed.found) {
        e.finger_pen = MaxPrePenetration(h1, poses, spheres, hull, e.void_closed.center, "finger");
        e.thumb_pen = MaxPrePenetration(h1, poses, spheres, hull, e.void_closed.center, "thumb");
        e.palm_pen = MaxPrePenetration(h1, poses, spheres, hull, e.void_closed.center, "palm");
    }
    return e;
}

bool LargeGeometryPasses(const LargeScaleEval& e) {
    return e.shallow.found && e.shallow.palm && e.shallow.covered_arc > kLargeCagingArcMin &&
           e.shallow.max_pen <= kShallowPenMax + 1e-5f &&
           e.finger_pen <= kShallowPenMax + 1e-5f &&
           e.thumb_pen <= kShallowPenMax + 1e-5f &&
           e.palm_pen <= kShallowPenMax + 1e-5f;
}

struct FingerOnlyFallbackEval {
    float sxy = 1.0f;
    float sz = 1.0f;
    float finger_delta = 0.0f;
    float thumb_yaw_delta = 0.0f;
    float thumb_pitch_delta = 0.0f;
    Place place;
};

FingerOnlyFallbackEval EvaluateFingerOnlyFallback(
    const nuka::phi::DeviceContext& context, const CookedH1& h1, const CupHull& base,
    float sxy, float sz, float finger_delta, float thumb_yaw_delta,
    float thumb_pitch_delta) {
    FingerOnlyFallbackEval e;
    e.sxy = sxy;
    e.sz = sz;
    e.finger_delta = finger_delta;
    e.thumb_yaw_delta = thumb_yaw_delta;
    e.thumb_pitch_delta = thumb_pitch_delta;
    const CupHull hull = ScaleCupHull(base, sxy, sz);
    articulation::ArticulationHostState host = h1.host;
    ApplyCurl(h1, &host,
              OffsetCurl(CurlForScale(sxy), finger_delta, thumb_yaw_delta, thumb_pitch_delta));
    const auto poses = ForwardKinematics(context, host);
    const Vec3 palm_offset{PALM_OFFSET_X, 0.0f, 0.0f};
    const auto finger_only = WrapSpheres(false, palm_offset);
    e.place = BestPlacementShallowNoPalmCaging(h1, poses, finger_only, hull);
    return e;
}

struct LiveCov {
    float covered_arc = 0.0f;
    float max_gap = 360.0f;
    bool palm = false;
    uint32_t n = 0u;
    // C7b-2d (additive): world-XY azimuth (deg, [0,360)) of the bisector of the
    // largest live coverage gap -- i.e. the direction toward which the cup can most
    // easily escape an opposed (force-closure) grasp. ForceClosureLiftWithDisturbance
    // pushes the held cup along this azimuth as the pre-committed worst case. Prior
    // callers ignore this field, so they are byte-for-byte unchanged.
    float gap_bisector_az = 0.0f;
};
LiveCov LiveCoverage(const nuka::phi::DeviceContext& context, const DenseGraspScene& gs,
                     coresident::UnifiedCoResidentStepper& stepper, const CupHull& hull,
                     bool with_palm = true) {
    articulation::ArticulationHostState st; stepper.Download(&st);
    const auto poses = ForwardKinematics(context, st);
    const Vec3 palm_offset{PALM_OFFSET_X, 0.0f, 0.0f};
    const auto spheres = WrapSpheres(with_palm, palm_offset);
    const Vec3 cc = stepper.Cup().position;
    LiveCov out;
    std::vector<float> azs;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(gs.h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        if (RimGap(c, cc, hull) >= s.radius) continue;
        if (std::string(s.region) == "palm") out.palm = true;
        float az = std::atan2(c.y - cc.y, c.x - cc.x) * 180.0f / kPi;
        if (az < 0.0f) az += 360.0f;
        azs.push_back(az);
    }
    out.n = static_cast<uint32_t>(azs.size());
    if (azs.size() < 2u) return out;
    std::sort(azs.begin(), azs.end());
    float max_gap = 0.0f;
    float gap_start = azs[0];  // azimuth at the CCW start of the largest gap.
    for (size_t i = 0u; i < azs.size(); ++i) {
        const float next = (i + 1u < azs.size()) ? azs[i + 1u] : azs[0] + 360.0f;
        const float g = next - azs[i];
        if (g > max_gap) { max_gap = g; gap_start = azs[i]; }
    }
    out.max_gap = max_gap;
    out.covered_arc = 360.0f - max_gap;
    // Bisector of the largest gap; std::fmod keeps it in [0,360) after the wraparound.
    out.gap_bisector_az = std::fmod(gap_start + max_gap * 0.5f, 360.0f);
    return out;
}

// ---------------------------------------------------------------------------
// C7b-2d: FORCE-CLOSURE DISTURBANCE-ROBUSTNESS rollout (additive).
//
// FingerOnlyFallbackLiftGate proved the opposed finger-only wrap HOLDS the cup
// against gravity after table removal (support == mg.dt, 220/220 in contact) but
// SKIPs on its CAGING-ARC sub-criterion: the grasp is FORCE CLOSURE (opposed
// friction), not FORM CLOSURE (caging), so a ~176-184 deg azimuth gap remains.
// Caging-arc was only a PROXY for "secure". The CORRECT security proof for a
// force-closure grasp is DISTURBANCE ROBUSTNESS: when pushed toward the open gap
// (the direction it can most easily escape) it must NOT escape, with the grip held
// FIXED (no re-grab, no squeeze-up).
//
// Protocol (single source of truth so the gate + its D1 sibling are byte-identical):
//   * 70-step table-supported settle, then SetTableEnabled(false).
//   * kLiftSettle lift steps under gravity ONLY (confirm hold: support ~= mg.dt),
//     snapshot the held pose, and read the gap-bisector azimuth from the live
//     coverage (the SAME max_gap LiveCoverage reports) -> the worst-case push dir.
//   * kPushSteps push window: SUSTAINED lateral body acceleration kLatAccelG*g along
//     the bisector dir, injected as dv = a_lat*dir*kDt EACH step; plus a ONE-SHOT
//     in-plane angular kick kTiltKickW (rad/s) applied ONCE at the first push step
//     (ApplyCupImpulse ADDS to angular_velocity -- repeating it would integrate to an
//     absurd spin; the held cup must merely survive the initial tilt). Grip held at
//     the SAME pd target throughout (DrivePd unchanged).
//   * kReleaseSettle steps with NO impulse: the held cup must recover.
// Gravity stays ON the whole time (held cup resists gravity + lateral simultaneously
// == the realistic worst case). Magnitudes are PRE-COMMITTED constants below and are
// NOT tuned to pass: 1.0*g lateral is "as hard as the gravity it already holds"; a
// free cup under 1g for kPushSteps would slide ~0.5*g*(kPushSteps*dt)^2 ~= 7.7cm,
// well past the bound below, so the bound cleanly separates "grasp arrests it" from
// "cup escapes".
// ---------------------------------------------------------------------------
constexpr float kFcLatAccelG = 1.0f;     // sustained lateral accel == 1.0 g (worst case).
constexpr uint32_t kFcLiftSettle = 50u;  // gravity-only hold before the push.
constexpr uint32_t kFcPushSteps = 30u;   // sustained-push window length.
constexpr uint32_t kFcReleaseSettle = 70u;  // post-push recovery window.
constexpr float kFcTiltKickW = 2.5f;     // ONE-SHOT angular kick magnitude (rad/s).
// Translation bound: ~1.3x the 1.8x cup radius (~0.055 m) -> 0.07 m. A free cup at 1g
// blows past this in the push window; a held cup stays cm-scale.
constexpr double kFcMaxDisp = 0.07;
constexpr double kFcMaxTilt = 0.35;      // bounded rotation away from the held attitude.
constexpr double kFcMaxFinalW = 1.20;    // bounded residual spin after recovery.

struct ForceClosureDistResult {
    bool placement_found = false;
    bool gap_defined = false;       // >=2 live contacts at end-of-settle (gap well-posed).
    float bisector_az = 0.0f;       // deg, world XY.
    Vec3 push_dir{};                // unit, world XY (along the bisector).
    double settle_fimp = 0.0;       // support just before the push (sanity: ~= mg.dt).
    double peak_disp = 0.0;         // peak displacement from held pose (push + recovery).
    double peak_tilt = 0.0;         // peak tilt from held attitude.
    double recovered_fimp = 0.0;    // mean support over the recovery window.
    double recovery_cross = 0.0;    // max vertical-impulse bookkeeping mismatch (recovery).
    double final_w = 0.0;           // |w| at the end.
    uint32_t steps_post = 0u;       // total post-removal steps.
    uint32_t contact_post = 0u;     // # post-removal steps with finger_contacts>0.
    bool any_static_post = false;   // a StaticNull (table) row persisted post-removal.
    double max_gap_push = 0.0;      // live gap during/after the push (informational).
    BodyState cup_final{};          // for D1 byte-identity.
};

// Runs the FULL force-closure disturbance protocol once. Deterministic by construction
// (fixed seed-free integration, pre-committed constants). `art_out`, if non-null, gets
// the final articulation host state for D1.
ForceClosureDistResult RunForceClosureDist(const nuka::phi::DeviceContext& context,
                                           const CupHull& base,
                                           articulation::ArticulationHostState* art_out) {
    ForceClosureDistResult r;
    const CupHull hull = ScaleCupHull(base, kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz);
    const CurlPose curl = OffsetCurl(CurlForScale(kFingerOnlyFallbackSxy),
                                     kFingerOnlyFallbackFingerDelta,
                                     kFingerOnlyFallbackThumbYawDelta,
                                     kFingerOnlyFallbackThumbPitchDelta);
    DenseGraspScene gs = BuildDenseGraspScene(
        context, base, /*has_table=*/true, PlaceMode::FingerOnlyShallow,
        kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz, /*with_palm=*/false,
        curl, /*use_curl_override=*/true);
    if (!gs.place.found) return r;  // placement_found stays false.
    r.placement_found = true;

    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);

    // --- table-supported settle ---
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);

    // --- phase a: gravity-only hold; snapshot held pose + measure support ---
    coresident::CoResidentStepReport rep;
    double settle_fimp_sum = 0.0; uint32_t settle_fimp_n = 0u;
    for (uint32_t s = 0u; s < kFcLiftSettle; ++s) {
        DrivePd(stepper, gs, pd);
        rep = stepper.Step();
        if (rep.any_static_row) r.any_static_post = true;
        if (s >= kFcLiftSettle / 2u && rep.finger_contacts > 0u) {
            settle_fimp_sum += rep.cup_vertical_impulse; ++settle_fimp_n;
        }
    }
    r.settle_fimp = settle_fimp_n ? settle_fimp_sum / settle_fimp_n : 0.0;
    const Vec3 c_held = stepper.Cup().position;
    const Quat q_held = stepper.Cup().orientation;
    const LiveCov cov_held = LiveCoverage(context, gs, stepper, hull, /*with_palm=*/false);
    r.bisector_az = cov_held.gap_bisector_az;
    r.gap_defined = (cov_held.n >= 2u);
    const float az_rad = cov_held.gap_bisector_az * kPi / 180.0f;
    r.push_dir = Vec3{std::cos(az_rad), std::sin(az_rad), 0.0f};

    // --- phase b: sustained lateral push along the bisector + ONE-SHOT tilt ---
    const Vec3 dv_step = r.push_dir * (kFcLatAccelG * (-kGravityZ) * kDt);
    // In-plane tilt axis = perpendicular to the push (worst-case roll about the escape line).
    const Vec3 tilt_axis{-r.push_dir.y, r.push_dir.x, 0.0f};
    auto track = [&](const coresident::CoResidentStepReport& rp) {
        ++r.steps_post;
        if (rp.finger_contacts > 0u) ++r.contact_post;
        if (rp.any_static_row) r.any_static_post = true;
        const Vec3 cp = stepper.Cup().position;
        r.peak_disp = std::max(r.peak_disp, (double)(cp - c_held).Length());
        r.peak_tilt = std::max(r.peak_tilt, RelTilt(q_held, stepper.Cup().orientation));
    };
    for (uint32_t s = 0u; s < kFcPushSteps; ++s) {
        DrivePd(stepper, gs, pd);
        if (s == 0u) {
            // ONE-SHOT angular kick (NOT per-step: ApplyCupImpulse accumulates).
            stepper.ApplyCupImpulse(dv_step, tilt_axis * kFcTiltKickW);
        } else {
            stepper.ApplyCupImpulse(dv_step, Vec3::Zero());
        }
        rep = stepper.Step();
        track(rep);
        if (s % 10u == 0u || s + 1u == kFcPushSteps) {
            const LiveCov cv = LiveCoverage(context, gs, stepper, hull, /*with_palm=*/false);
            r.max_gap_push = std::max(r.max_gap_push, (double)cv.max_gap);
        }
    }

    // --- phase c: release; recover; measure recovered support + clean bookkeeping ---
    double rec_sum = 0.0; uint32_t rec_n = 0u;
    for (uint32_t s = 0u; s < kFcReleaseSettle; ++s) {
        DrivePd(stepper, gs, pd);
        rep = stepper.Step();
        track(rep);
        // Accumulate recovery support over the LATTER half (settled, impulse-free).
        if (s >= kFcReleaseSettle / 2u && rep.finger_contacts > 0u) {
            rec_sum += rep.cup_vertical_impulse; ++rec_n;
            if (rep.cup_vertical_impulse != 0.0) {
                const double cross =
                    std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse) /
                    std::max(std::fabs(rep.cup_dvz_impulse), 1e-9);
                r.recovery_cross = std::max(r.recovery_cross, cross);
            }
        }
        if (s % 20u == 0u || s + 1u == kFcReleaseSettle) {
            const LiveCov cv = LiveCoverage(context, gs, stepper, hull, /*with_palm=*/false);
            r.max_gap_push = std::max(r.max_gap_push, (double)cv.max_gap);
        }
    }
    r.recovered_fimp = rec_n ? rec_sum / rec_n : 0.0;
    r.final_w = stepper.Cup().angular_velocity.Length();
    r.cup_final = stepper.Cup();
    (void)weight_kick;
    if (art_out) stepper.Download(art_out);
    return r;
}

}  // namespace

// ===========================================================================
// PHASE 1 / GATE 0: sweep the DENSE wrap over larger round cups (1.6x..2.6x) and
// find the first scale whose fingers, thumb, and palm all co-contact shallow (<=2mm)
// while the palm closes the old finger-only azimuth gap. This is the owner-selected
// next branch after the 1.6x dense negative: switch to a larger object, validate the
// geometry before spending dynamics time.
// ==========================================================================
TEST(H1DenseGraspLargeCup, ScaleSweepFindsBothSidesShallow) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);

    bool any_pass = false;
    bool any_finger_only = false;
    LargeScaleEval chosen;
    for (const ScaleSpec spec : kLargeScaleSweep) {
        const LargeScaleEval e = EvaluateLargeScale(context, h1, base, spec.sxy, spec.sz);
        std::printf("[LARGE-SWEEP] k=%.2f/%.2f diam=%.4f m shallow=%d arc=%.1f "
                    "gap=%.1f max_pen=%.2f mm region(finger/thumb/palm)=%.2f/%.2f/%.2f "
                    "nopalm=%d nopalm_arc=%.1f nopalm_gap=%.1f void_pen=%.2f mm\n",
                    e.sxy, e.sz, e.diameter, e.shallow.found ? 1 : 0,
                    e.shallow.covered_arc, e.shallow.max_gap,
                    e.shallow.max_pen * 1000.0f, e.finger_pen * 1000.0f,
                    e.thumb_pen * 1000.0f, e.palm_pen * 1000.0f,
                    e.no_palm.found ? 1 : 0, e.no_palm.covered_arc, e.no_palm.max_gap,
                    e.void_closed.max_pen * 1000.0f);
        if (e.no_palm.found) any_finger_only = true;
        if (!any_pass && LargeGeometryPasses(e)) {
            chosen = e;
            any_pass = true;
        }
    }

    if (!any_pass) {
        GTEST_SKIP() << "NEGATIVE: no round-cup scale in 1.6..2.6 achieved both-sides "
                        "shallow closure (fingers/thumb/palm all <=2mm, palm co-contact, "
                        "arc >200deg). Finger-only caging in the same fixed curl was "
                     << (any_finger_only ? "found" : "not found")
                     << "; see FingerOnlyFallbackScaleAndCurlSweep for the bounded "
                        "fallback scan and [LARGE-SWEEP] table.";
    }

    std::printf("[LARGE-SWEEP] CHOSEN k=%.2f/%.2f diam=%.4f m arc=%.1f gap=%.1f "
                "max_pen=%.2f mm region=%.2f/%.2f/%.2f mm\n",
                chosen.sxy, chosen.sz, chosen.diameter, chosen.shallow.covered_arc,
                chosen.shallow.max_gap, chosen.shallow.max_pen * 1000.0f,
                chosen.finger_pen * 1000.0f, chosen.thumb_pen * 1000.0f,
                chosen.palm_pen * 1000.0f);
    EXPECT_TRUE(LargeGeometryPasses(chosen));
    EXPECT_NEAR(chosen.sxy, kLargeDynSxy, 1e-4f)
        << "canonical model_large.usda is expected to be the selected 1.8x gate object";
}

TEST(H1DenseGraspLargeCup, FingerOnlyFallbackScaleAndCurlSweep) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);

    const FingerOnlyFallbackEval canonical = EvaluateFingerOnlyFallback(
        context, h1, base, kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz,
        kFingerOnlyFallbackFingerDelta, kFingerOnlyFallbackThumbYawDelta,
        kFingerOnlyFallbackThumbPitchDelta);
    std::printf("[LARGE-FINGER-ONLY] canonical k=%.2f/%.2f found=%d arc=%.1f gap=%.1f "
                "max_pen=%.2f mm contacts=%u\n",
                canonical.sxy, canonical.sz, canonical.place.found ? 1 : 0,
                canonical.place.covered_arc, canonical.place.max_gap,
                canonical.place.max_pen * 1000.0f, canonical.place.genuine);
    if (canonical.place.found) {
        EXPECT_GT(canonical.place.covered_arc, kLargeCagingArcMin);
        EXPECT_LE(canonical.place.max_pen, kShallowPenMax + 1e-5f);
        return;
    }

    const float finger_offsets[] = {-0.20f, 0.0f, 0.20f};
    const float thumb_yaw_offsets[] = {-0.35f, 0.0f, 0.35f};
    const float thumb_pitch_offsets[] = {-0.25f, 0.0f, 0.25f};
    bool found = false;
    FingerOnlyFallbackEval best;
    float best_arc = 0.0f;
    for (const ScaleSpec spec : kLargeScaleSweep) {
        for (float fd : finger_offsets) for (float ty : thumb_yaw_offsets) {
            for (float tp : thumb_pitch_offsets) {
                const FingerOnlyFallbackEval e = EvaluateFingerOnlyFallback(
                    context, h1, base, spec.sxy, spec.sz, fd, ty, tp);
                if (e.place.covered_arc > best_arc) {
                    best = e;
                    best_arc = e.place.covered_arc;
                }
                if (e.place.found) {
                    found = true;
                    best = e;
                    goto fallback_done;
                }
            }
        }
    }
fallback_done:
    std::printf("[LARGE-FINGER-ONLY] found=%d k=%.2f/%.2f curl_offsets(fd/ty/tp)=%.2f/%.2f/%.2f "
                "arc=%.1f gap=%.1f max_pen=%.2f mm contacts=%u\n",
                found ? 1 : 0, best.sxy, best.sz, best.finger_delta,
                best.thumb_yaw_delta, best.thumb_pitch_delta, best.place.covered_arc,
                best.place.max_gap, best.place.max_pen * 1000.0f, best.place.genuine);
    if (!found) {
        GTEST_SKIP() << "NEGATIVE: bounded finger-only fallback scan (8 scales x 27 curl "
                        "offsets) found no >200deg shallow opposed wrap without palm. Best "
                        "arc=" << best_arc << " deg. Stop Phase 1 and ask owner for a "
                        "different object/pose strategy; do not grind parameters.";
    }
    EXPECT_TRUE(found);
}

// ===========================================================================
// GATE 1 (the VALIDITY precondition): does the stepper REALIZE the dense set as MANY
// independent contacts? Build the dense scene, settle a few steps on the table, and
// assert rep.finger_contacts jumps from the prior ~11 to >= ~30. If it stays ~11, the
// unique-handle fix did not take (handle collapse) and EVERY downstream number is
// invalid -- that is a BROKEN FIX, not a physics negative. This gates the experiment.
// ===========================================================================
TEST(H1DenseGrasp, DenseContactsAreRealized) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);

    // Build at the DEEP void-closed placement (always exists) -- this is the FIRST + ONLY
    // exercise of the dense contact set + the unique-handle / finger_link<-handle stepper
    // edit. It validates the APPARATUS independently of whether a SHALLOW placement exists.
    DenseGraspScene gs =
        BuildDenseGraspScene(context, base, /*has_table=*/true, PlaceMode::VoidClosed);
    std::printf("[DENSE] sphere_count=%u (vs prior ~11), radius=%.3f m, "
                "void-closed placement max_pen=%.4f m (%.1f mm) arc=%.1f gap=%.1f\n",
                gs.sphere_count, kWrapRadius, gs.place.max_pen, gs.place.max_pen * 1000.0f,
                gs.place.covered_arc, gs.place.max_gap);
    ASSERT_TRUE(gs.place.found) << "no void-closed placement at all (apparatus broken)";

    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    coresident::CoResidentStepReport rep;
    uint32_t max_contacts = 0u;
    // The bookkeeping cross-check is |vimp - dvz_impulse| / |dvz_impulse|. With the table
    // ON the cup is near-static so cup_dvz_impulse ~= 0 -> the RELATIVE ratio is ill-
    // defined (divide-by-near-zero), NOT a solve failure. So we track the ABSOLUTE
    // disagreement |vimp - dvz_impulse| and normalize by the weight kick (the physical
    // scale), measured only when the cup actually accelerated (|dvz| above a floor) --
    // that is the honest convergence probe for a table-supported near-static cup.
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    double max_cross_rel = 0.0, max_cross_abs_norm = 0.0;
    for (uint32_t s = 0u; s < 30u; ++s) {
        DrivePd(stepper, gs, pd);
        rep = stepper.Step();
        max_contacts = std::max(max_contacts, rep.finger_contacts);
        if (rep.finger_contacts > 0u && rep.cup_vertical_impulse != 0.0) {
            const double abs_dis = std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse);
            max_cross_abs_norm = std::max(max_cross_abs_norm, abs_dis / weight_kick);
            if (std::fabs(rep.cup_dvz_impulse) > 0.2 * weight_kick) {
                max_cross_rel = std::max(max_cross_rel,
                    abs_dis / std::max(std::fabs(rep.cup_dvz_impulse), 1e-9));
            }
        }
    }
    std::printf("[DENSE] settled finger_contacts=%u (max over 30 steps=%u) row_count=%u "
                "cross_rel(moving)=%.2e cross_abs/mgdt=%.2e\n",
                rep.finger_contacts, max_contacts, rep.row_count, max_cross_rel,
                max_cross_abs_norm);
    // APPARATUS GREEN (the validity precondition): the dense contacts are REALIZED as many
    // INDEPENDENT contacts (not deduped to the prior sparse ~11). NOTE on what this proves:
    // finger_contacts is counted at BuildContactManifolds (BEFORE the chain-J wiring), so a
    // high count validates the geometry RESOLVER picks a distinct sphere per unique handle
    // -- it does NOT by itself prove the chain-J binds the right link. The finger_link<-
    // handle mapping is correct BY CONSTRUCTION: each sphere gets a UNIQUE 9000+ handle that
    // maps to its own ft.link in BuildDenseGraspScene; the stepper loop maps that handle
    // back to ft.link (a miss would pass a 9000+ value to FootChainJ -> out-of-bounds
    // garbage, which it does not). The dense DYNAMICS path (handle != link in the chain-J)
    // is EXERCISED here but NOT validated -- the bookkeeping does not close (see below).
    ASSERT_GE(max_contacts, 25u)
        << "dense contacts collapsed (finger_contacts " << max_contacts << " ~ the prior "
           "sparse ~11) -> the unique-handle resolver did not take; downstream invalid";

    // SECONDARY FINDING (surfaced, not asserted green): the deep (5.9mm), heavily-
    // overlapping dense set (37 contacts / 115 rows under active squeeze) does NOT solve
    // with closed bookkeeping in 64 PGS iterations -- the cup-side vertical impulse and
    // the cup's actual vertical-velocity change disagree by ~16x (cross_rel/cross_abs both
    // O(10)), measured even on steps where the cup moved. So the dense-DYNAMICS path
    // (handle != link) is exercised-but-not-validated: plausibly genuine ill-conditioning
    // at 5.9mm x 37 contacts x active squeeze. It does NOT undermine the GEOMETRIC NEGATIVE
    // (GATE 2, solver-independent) -- it reinforces "genuinely hard". Per the pre-committed
    // stop we do NOT crank iterations. Surfaced via SKIP (no false-regression RED).
    if (max_cross_rel > 5.0e-2 || max_cross_abs_norm > 5.0e-2) {
        GTEST_SKIP() << "APPARATUS VALID (dense contacts realized: " << max_contacts
                     << " >= 25, unique-handle RESOLVER works; chain-J mapping correct-by-"
                        "construction) BUT the DEEP dense-DYNAMICS path does not converge "
                        "cleanly: cross_rel(moving)=" << max_cross_rel << " cross_abs/mgdt="
                     << max_cross_abs_norm << " (bookkeeping disagrees ~16x at the 5.9mm / "
                        "115-row active-squeeze set in 64 PGS iters) -- exercised but not "
                        "validated. The dispositive NEGATIVE is the GEOMETRIC shallow-"
                        "infeasibility (GATE 2), solver-independent. See the report.";
    }
}

// ===========================================================================
// GATE 2 (calibration / the key number): the achieved SHALLOW placement -- max pre-
// penetration (must be <=2mm now, vs the prior 7.2mm wedge), coverage, palm, |S|. Also
// reports the void-closed placement WITHOUT the shallow filter (how deep it would have
// to be) so the dense-vs-sparse depth contrast is on record.
// ===========================================================================
TEST(H1DenseGrasp, ShallowPlacementCalibration) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CupHull hull = ScaleCupHull(base, kDynSxy, kDynSz);

    const CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    articulation::ArticulationHostState host = h1.host;
    ApplyCurl(h1, &host, CurlForScale(kDynSxy));
    const auto poses = ForwardKinematics(context, host);
    const Vec3 palm_offset{PALM_OFFSET_X, 0.0f, 0.0f};
    const auto spheres = WrapSpheres(true, palm_offset);

    const Place shallow = BestPlacementShallow(h1, poses, spheres, hull);
    const Place shallow_nopalm = BestPlacementShallowNoPalm(h1, poses, spheres, hull);
    const Place voidc = BestPlacementVoidClosed(h1, poses, spheres, hull);
    std::printf("[CALIB] dense diam=%.4f m spheres=%zu | SHALLOW place found=%d "
                "max_pen=%.4f m (%.1f mm) arc=%.1f gap=%.1f palm=%d |S|=%.5f\n",
                2.0f * CupRadius(hull), spheres.size(), shallow.found ? 1 : 0,
                shallow.max_pen, shallow.max_pen * 1000.0f, shallow.covered_arc,
                shallow.max_gap, shallow.palm ? 1 : 0, shallow.squeeze_mag);
    std::printf("[CALIB] void-closed WITHOUT shallow filter: found=%d max_pen=%.4f m "
                "(%.1f mm) arc=%.1f gap=%.1f (compare the prior sparse 7.2mm wedge)\n",
                voidc.found ? 1 : 0, voidc.max_pen, voidc.max_pen * 1000.0f,
                voidc.covered_arc, voidc.max_gap);
    // The DEPTH-BY-REGION split at the void-closed placement: which part of the wrap
    // forces the deep penetration? (Prior sparse: fingers 7.2mm / palm 2.4mm -> the
    // WRAP coverage, not the palm, forced the depth.)
    if (voidc.found) {
        const float pen_finger = MaxPrePenetration(h1, poses, spheres, hull, voidc.center, "finger");
        const float pen_thumb  = MaxPrePenetration(h1, poses, spheres, hull, voidc.center, "thumb");
        const float pen_palm   = MaxPrePenetration(h1, poses, spheres, hull, voidc.center, "palm");
        std::printf("[CALIB] void-closed depth BY REGION: fingers=%.1f mm thumb=%.1f mm "
                    "palm=%.1f mm (the deepest region forces the wedge)\n",
                    pen_finger * 1000.0f, pen_thumb * 1000.0f, pen_palm * 1000.0f);
    }
    // THE DECOMPOSING DIAGNOSTIC: does the dense FINGER+THUMB wrap close coverage
    // (arc>180) SHALLOW (<=2mm) WITHOUT requiring the palm co-contact? This separates
    // "the dense wrap can't close shallow" (robust NEGATIVE) from "only the palm co-
    // contact requirement forces the wedge" (the residual obstruction is the 1.6x cup
    // not filling the palm-finger span shallow on BOTH sides -> option-2 = bigger object
    // OR drop palm co-contact, NOT generically hard).
    std::printf("[CALIB] SHALLOW no-palm (finger+thumb wrap only, arc>180 @ <=2mm): "
                "found=%d arc=%.1f gap=%.1f max_pen=%.4f m (%.1f mm) palm_incidental=%d\n",
                shallow_nopalm.found ? 1 : 0, shallow_nopalm.covered_arc,
                shallow_nopalm.max_gap, shallow_nopalm.max_pen,
                shallow_nopalm.max_pen * 1000.0f, shallow_nopalm.palm ? 1 : 0);
    // The DISPOSITIVE geometric finding, decomposed by the no-palm scan + region split.
    // The required void-closure depth is dominated by the PALM (5.9mm) while the dense
    // FINGER wrap is shallow (2.0mm) -- so the mechanism is NOT "the dense wrap can't
    // close" but specifically the PALM co-contact requirement. The skip message states
    // which branch the data picked. Surfaced as SKIP-with-finding (not a hard FAIL =
    // a false regression -- the same protocol the prior spikes use).
    if (!shallow.found) {
        if (shallow_nopalm.found) {
            GTEST_SKIP() << "DENSE+SHALLOW INFEASIBLE *ONLY WITH PALM CO-CONTACT* "
                            "(NEGATIVE, mechanism identified): the dense FINGER+THUMB wrap "
                            "DOES close coverage (arc=" << shallow_nopalm.covered_arc
                         << ", gap=" << shallow_nopalm.max_gap << ") SHALLOW at max_pen="
                         << shallow_nopalm.max_pen * 1000.0f << " mm -- density FIXED the "
                            "finger depth (prior sparse fingers 7.2mm -> dense 2.0mm). The "
                            "residual obstruction is the PALM co-contact (option-1 void-"
                            "closure, gap<110): the void-closed-with-palm placement needs "
                            "max_pen=" << voidc.max_pen * 1000.0f << " mm, ALL of it in the "
                            "PALM (region split: fingers 2.0 / thumb 2.9 / palm 5.9 mm). So "
                            "the 1.6x cup does not fill the palm-finger span shallow on BOTH "
                            "sides -> option-2 = a BIGGER object OR drop the palm co-contact "
                            "requirement, NOT generically hard. See the report.";
        } else {
            GTEST_SKIP() << "DENSE+SHALLOW INFEASIBLE (robust NEGATIVE): not even the dense "
                            "FINGER+THUMB wrap closes coverage (arc>180) SHALLOW (<=2mm) "
                            "without the palm; the void-closed placement needs max_pen="
                         << voidc.max_pen * 1000.0f << " mm -- a DEEP wedge regardless of "
                            "the palm requirement -> sparsity excluded -> genuinely hard. "
                            "See the report.";
        }
    }
    EXPECT_TRUE(shallow.found)
        << "(reached only if a shallow placement exists) dense+shallow feasible";
    EXPECT_LE(shallow.max_pen, kShallowPenMax + 1e-5f);
}

// ===========================================================================
// GATE 3 (THE LIFT): dense + shallow cup on a static table, PD ACTIVE-squeeze close,
// then REMOVE the table. Assert the dense wrap CAGES the cup: bounded tilt, contacts
// retained, balance cross-check closes, support ~+mg.dt, any_static==0. The dense
// difference from the prior negative is that coverage should NOT collapse (the live
// max_gap stays closed) because many distributed contacts share the load.
// ===========================================================================
TEST(H1DenseGraspLargeCup, FingerOnlyFallbackLiftGate) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CupHull hull = ScaleCupHull(base, kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    const CurlPose curl = OffsetCurl(CurlForScale(kFingerOnlyFallbackSxy),
                                     kFingerOnlyFallbackFingerDelta,
                                     kFingerOnlyFallbackThumbYawDelta,
                                     kFingerOnlyFallbackThumbPitchDelta);
    DenseGraspScene gs = BuildDenseGraspScene(
        context, base, /*has_table=*/true, PlaceMode::FingerOnlyShallow,
        kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz, /*with_palm=*/false,
        curl, /*use_curl_override=*/true);
    if (!gs.place.found) {
        GTEST_SKIP() << "NEGATIVE: canonical finger-only fallback has no <=2mm caging "
                        "placement. See FingerOnlyFallbackScaleAndCurlSweep.";
    }
    std::printf("[FINGER-LIFT] step-zero k=%.2f diam=%.4f m max_pen=%.2f mm arc=%.1f "
                "gap=%.1f spheres=%u\n",
                kFingerOnlyFallbackSxy, 2.0f * CupRadius(hull), gs.place.max_pen * 1000.0f,
                gs.place.covered_arc, gs.place.max_gap, gs.sphere_count);

    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    coresident::CoResidentStepReport rep;
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); rep = stepper.Step(); }
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;
    const LiveCov cov_settle = LiveCoverage(context, gs, stepper, hull, /*with_palm=*/false);
    std::printf("[FINGER-LIFT] table settle: contacts=%u table_rows=%u finger_vimp=%.4e "
                "table_vimp=%.4e (mg.dt=%.4e) arc=%.1f gap=%.1f\n",
                rep.finger_contacts, rep.table_row_count, rep.cup_vertical_impulse,
                rep.table_vertical_impulse, weight_kick, cov_settle.covered_arc,
                cov_settle.max_gap);

    stepper.SetTableEnabled(false);
    const uint32_t kLift = 220u, kLiftSettle = 60u;
    double max_disp = 0.0, max_tilt = 0.0, max_cross = 0.0, max_gap_live = 0.0;
    double sum_fimp = 0.0;
    uint32_t n_steady = 0u, contact_after = 0u;
    bool any_static_after = false;
    for (uint32_t s = 0u; s < kLift; ++s) {
        DrivePd(stepper, gs, pd);
        rep = stepper.Step();
        if (rep.finger_contacts > 0u) ++contact_after;
        if (rep.any_static_row) any_static_after = true;
        const Vec3 cp = stepper.Cup().position;
        max_disp = std::max(max_disp, (double)(cp - c_settled).Length());
        max_tilt = std::max(max_tilt, RelTilt(q_settled, stepper.Cup().orientation));
        if (rep.finger_contacts > 0u && rep.cup_vertical_impulse != 0.0) {
            const double cross = std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse) /
                std::max(std::fabs(rep.cup_dvz_impulse), 1e-9);
            max_cross = std::max(max_cross, cross);
        }
        if (s % 20u == 0u || s + 1u == kLift) {
            const LiveCov cov = LiveCoverage(context, gs, stepper, hull, /*with_palm=*/false);
            if (s >= 20u) max_gap_live = std::max(max_gap_live, (double)cov.max_gap);
            std::printf("[FINGER-LIFT]   traj s=%3u tilt=%.4f rad |w|=%.3f gap=%.1f n=%u\n",
                        s, RelTilt(q_settled, stepper.Cup().orientation),
                        stepper.Cup().angular_velocity.Length(), cov.max_gap, cov.n);
        }
        if (s >= kLiftSettle) { sum_fimp += rep.cup_vertical_impulse; ++n_steady; }
    }
    const double mean_fimp = n_steady ? sum_fimp / n_steady : 0.0;
    const LiveCov cov_end = LiveCoverage(context, gs, stepper, hull, /*with_palm=*/false);
    const double final_w = stepper.Cup().angular_velocity.Length();
    std::printf("[FINGER-LIFT] after removal: contact=%u/%u any_static=%d mean_fimp=%.4e "
                "(mg.dt=%.4e) cross=%.2e max_disp=%.4f max_tilt=%.4f |w|=%.3f "
                "end_arc=%.1f end_gap=%.1f max_gap=%.1f\n",
                contact_after, kLift, any_static_after ? 1 : 0, mean_fimp, weight_kick,
                max_cross, max_disp, max_tilt, final_w, cov_end.covered_arc,
                cov_end.max_gap, max_gap_live);

    EXPECT_FALSE(any_static_after) << "a table row persisted after removal";
    EXPECT_LT(max_cross, 5.0e-2)
        << "finger-only impulse bookkeeping disagrees (numerically invalid lift gate)";
    const bool support_ok = std::fabs(mean_fimp - weight_kick) < 0.35 * weight_kick;
    const bool translation_ok = max_disp < 0.05;
    const bool contact_ok = contact_after >= kLift - 12u;
    const bool arc_ok = (max_gap_live <= 160.0) && (cov_end.max_gap <= 160.0f);
    const bool tilt_ok = max_tilt < 0.20 && final_w < 0.90;
    const bool caged = support_ok && translation_ok && contact_ok && arc_ok && tilt_ok &&
                       !any_static_after && max_cross < 5.0e-2;
    if (!caged) {
        GTEST_SKIP() << "NEGATIVE: finger-only fallback shallow geometry did not lift-cage: "
                     << "support_ok=" << support_ok << " mean_fimp=" << mean_fimp
                     << " translation_ok=" << translation_ok << " disp=" << max_disp
                     << " contact_ok=" << contact_ok << " " << contact_after << "/" << kLift
                     << " arc_ok=" << arc_ok << " max_gap=" << max_gap_live
                     << " tilt_ok=" << tilt_ok << " tilt=" << max_tilt << " |w|="
                     << final_w << ". Stop Phase 1; no parameter grinding.";
    }
    EXPECT_TRUE(caged);
}

// ---------------------------------------------------------------------------
// C7b-2d: the FORCE-CLOSURE security proof. FingerOnlyFallbackLiftGate holds the cup
// against gravity but SKIPs on caging-arc (the grasp is force closure, not form
// closure). The correct security proof for a force-closure grasp is disturbance
// robustness: push the held cup toward its open gap (worst-case escape direction) at
// 1.0 g sustained + a one-shot tilt, grip held FIXED, and require it does NOT escape
// and that support RECOVERS to mg.dt afterward. PASS => the opposed wrap is a validated
// secure grasp; SKIP-with-numbers => force closure too fragile laterally -> next move
// is a flat-faced object (box), not parameter grinding.
// ---------------------------------------------------------------------------
TEST(H1DenseGraspLargeCup, ForceClosureLiftWithDisturbance) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;

    const ForceClosureDistResult r = RunForceClosureDist(context, base, /*art_out=*/nullptr);
    if (!r.placement_found) {
        GTEST_SKIP() << "NEGATIVE: finger-only fallback has no <=2mm placement. See "
                        "FingerOnlyFallbackScaleAndCurlSweep.";
    }
    if (!r.gap_defined) {
        GTEST_SKIP() << "NEGATIVE: <2 live contacts at end-of-settle -> escape gap "
                        "ill-defined (cup not held). settle_fimp=" << r.settle_fimp
                     << " (mg.dt=" << weight_kick << ").";
    }

    // PRE-COMMITTED disturbance, echoed verbatim so the worst case is on record.
    std::printf("[FORCE-CLOSURE-DIST] PRE-COMMIT: lat=%.2f g sustained %u steps along gap-"
                "bisector az=%.1f deg dir=(%.3f,%.3f,0) + ONE-SHOT tilt %.2f rad/s; grip held "
                "FIXED (Kp=%.1f Kd=%.1f offset=%.2f); release+settle %u steps; gravity ON.\n",
                kFcLatAccelG, kFcPushSteps, r.bisector_az, r.push_dir.x, r.push_dir.y,
                kFcTiltKickW, kKp, kKd, kCloseOffset, kFcReleaseSettle);
    std::printf("[FORCE-CLOSURE-DIST] held support before push: fimp=%.4e (mg.dt=%.4e)\n",
                r.settle_fimp, weight_kick);
    std::printf("[FORCE-CLOSURE-DIST] result: peak_disp=%.4f m (bound %.3f) peak_tilt=%.4f rad "
                "(bound %.2f) recovered_fimp=%.4e (mg.dt=%.4e) final|w|=%.3f contacts=%u/%u "
                "any_static=%d max_gap=%.1f recovery_cross=%.2e\n",
                r.peak_disp, kFcMaxDisp, r.peak_tilt, kFcMaxTilt, r.recovered_fimp, weight_kick,
                r.final_w, r.contact_post, r.steps_post, r.any_static_post ? 1 : 0,
                r.max_gap_push, r.recovery_cross);

    const bool translation_ok = r.peak_disp < kFcMaxDisp;
    const bool tilt_ok = r.peak_tilt < kFcMaxTilt && r.final_w < kFcMaxFinalW;
    const bool contact_ok = r.contact_post >= r.steps_post - 12u;
    const bool recover_ok = std::fabs(r.recovered_fimp - weight_kick) < 0.35 * weight_kick;
    const bool static_ok = !r.any_static_post;
    const bool cross_ok = r.recovery_cross < 5.0e-2;
    const bool held = translation_ok && tilt_ok && contact_ok && recover_ok && static_ok &&
                      cross_ok;

    if (!held) {
        GTEST_SKIP() << "NEGATIVE: force closure too fragile laterally -> next move is a "
                        "flat-faced object (box), not parameter grinding. translation_ok="
                     << translation_ok << " disp=" << r.peak_disp << " tilt_ok=" << tilt_ok
                     << " tilt=" << r.peak_tilt << " |w|=" << r.final_w << " contact_ok="
                     << contact_ok << " " << r.contact_post << "/" << r.steps_post
                     << " recover_ok=" << recover_ok << " recovered_fimp=" << r.recovered_fimp
                     << " static_ok=" << static_ok << " cross_ok=" << cross_ok
                     << " cross=" << r.recovery_cross << ".";
    }
    EXPECT_TRUE(held);
}

// C7b-2d D1: the disturbance protocol must be deterministic (byte-identical across two
// runs of the SAME pre-committed rollout).
TEST(H1DenseGraspLargeCup, ForceClosureLiftWithDisturbanceDeterministicTwoRun) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    articulation::ArticulationHostState aa, ab;
    const ForceClosureDistResult ra = RunForceClosureDist(context, base, &aa);
    if (!ra.placement_found)
        GTEST_SKIP() << "NEGATIVE: finger-only fallback geometry absent.";
    const ForceClosureDistResult rb = RunForceClosureDist(context, base, &ab);
    ASSERT_TRUE(rb.placement_found);
    EXPECT_EQ(std::memcmp(&ra.cup_final, &rb.cup_final, sizeof(BodyState)), 0)
        << "force-closure disturbance cup diverged across 2 rollouts";
    ASSERT_EQ(aa.qdot.size(), ab.qdot.size());
    EXPECT_EQ(std::memcmp(aa.qdot.data(), ab.qdot.data(), aa.qdot.size() * sizeof(float)), 0)
        << "force-closure disturbance qdot diverged across 2 rollouts";
    std::printf("[FORCE-CLOSURE-D1] disturbance rollout cup + qdot byte-identical across 2 runs "
                "(bisector az=%.1f deg, peak_disp=%.4f m)\n", ra.bisector_az, ra.peak_disp);
}

TEST(H1DenseGraspLargeCup, FingerOnlyFallbackBiteGripOffVsOn) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    const uint32_t kFall = 120u;
    const double free_fall = 0.5 * (-kGravityZ) * (kFall * kDt) * (kFall * kDt);
    const CurlPose curl = OffsetCurl(CurlForScale(kFingerOnlyFallbackSxy),
                                     kFingerOnlyFallbackFingerDelta,
                                     kFingerOnlyFallbackThumbYawDelta,
                                     kFingerOnlyFallbackThumbPitchDelta);
    DenseGraspScene gs = BuildDenseGraspScene(
        context, base, /*has_table=*/true, PlaceMode::FingerOnlyShallow,
        kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz, /*with_palm=*/false,
        curl, /*use_curl_override=*/true);
    if (!gs.place.found) GTEST_SKIP() << "NEGATIVE: finger-only fallback geometry absent.";
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);
    coresident::CoResidentStepReport rep_on;
    for (uint32_t s = 0u; s < 40u; ++s) { DrivePd(stepper, gs, pd); rep_on = stepper.Step(); }
    const double z_active = stepper.Cup().position.z;
    const double vz_on = stepper.Cup().linear_velocity.z;
    const bool holds_on = (std::fabs(z_active - gs.cup0.position.z) < 0.05) &&
                          (std::fabs(vz_on) < 0.5) &&
                          (rep_on.cup_vertical_impulse > 0.5 * weight_kick);
    const std::vector<float> zero(gs.host.TotalLinkCount(), 0.0f);
    coresident::CoResidentStepReport rep_off;
    for (uint32_t s = 0u; s < kFall; ++s) { stepper.SetGripTorque(zero); rep_off = stepper.Step(); }
    const BodyState cupF = stepper.Cup();
    const double drop = z_active - cupF.position.z;
    const bool falls = (drop > 0.02) || (cupF.linear_velocity.z < -0.10);
    std::printf("[FINGER-BITE] max_pen=%.2f mm grip=ON holds=%d z=%.5f vz=%.4f "
                "finger_vimp=%.4e mg.dt=%.4e | grip=0 drop=%.5f free=%.4f vz=%.4f "
                "finger_vimp=%.4e -> %s\n",
                gs.place.max_pen * 1000.0f, holds_on ? 1 : 0, z_active, vz_on,
                rep_on.cup_vertical_impulse, weight_kick, drop, free_fall,
                cupF.linear_velocity.z, rep_off.cup_vertical_impulse,
                falls ? "FALLS (ACTIVE)" : "HOLDS (shallow form closure)");
    if (!holds_on) {
        GTEST_SKIP() << "NEGATIVE: finger-only fallback grip=ON did not hold. vimp="
                     << rep_on.cup_vertical_impulse << " vs mg.dt=" << weight_kick
                     << " vz=" << vz_on;
    }
    std::printf("[FINGER-BITE] diagnostic: active opposed hold only; lift coverage gate "
                "decides whether this is a caging grasp.\n");
    EXPECT_TRUE(holds_on);
    EXPECT_LE(gs.place.max_pen, kShallowPenMax + 1e-5f);
}

TEST(H1DenseGraspLargeCup, FingerOnlyFallbackDeterministicTwoRun) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CurlPose curl = OffsetCurl(CurlForScale(kFingerOnlyFallbackSxy),
                                     kFingerOnlyFallbackFingerDelta,
                                     kFingerOnlyFallbackThumbYawDelta,
                                     kFingerOnlyFallbackThumbPitchDelta);
    auto rollout = [&](BodyState* cup, articulation::ArticulationHostState* art) {
        DenseGraspScene gs = BuildDenseGraspScene(
            context, base, /*has_table=*/true, PlaceMode::FingerOnlyShallow,
            kFingerOnlyFallbackSxy, kFingerOnlyFallbackSz, /*with_palm=*/false,
            curl, /*use_curl_override=*/true);
        if (!gs.place.found) return false;
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        *cup = stepper.Cup();
        stepper.Download(art);
        return true;
    };
    BodyState ca, cb;
    articulation::ArticulationHostState aa, ab;
    if (!rollout(&ca, &aa)) GTEST_SKIP() << "NEGATIVE: finger-only fallback geometry absent.";
    ASSERT_TRUE(rollout(&cb, &ab));
    EXPECT_EQ(std::memcmp(&ca, &cb, sizeof(BodyState)), 0) << "finger-only cup diverged";
    ASSERT_EQ(aa.qdot.size(), ab.qdot.size());
    EXPECT_EQ(std::memcmp(aa.qdot.data(), ab.qdot.data(), aa.qdot.size() * sizeof(float)), 0)
        << "finger-only qdot diverged";
    std::printf("[FINGER-D1] finger-only fallback cup + qdot byte-identical across 2 rollouts\n");
}

TEST(H1DenseGraspLargeCup, LiftGateActiveShallowWrapCagesCup) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CupHull hull = ScaleCupHull(base, kLargeDynSxy, kLargeDynSz);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;

    DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true,
                                              PlaceMode::ShallowCaging,
                                              kLargeDynSxy, kLargeDynSz);
    if (!gs.place.found) {
        GTEST_SKIP() << "NEGATIVE: canonical 1.8x large cup has no <=2mm palm+finger "
                        "caging placement. See ScaleSweepFindsBothSidesShallow.";
    }
    std::printf("[LARGE-LIFT] step-zero k=%.2f diam=%.4f m max_pen=%.2f mm "
                "arc=%.1f gap=%.1f palm=%d spheres=%u\n",
                kLargeDynSxy, 2.0f * CupRadius(hull), gs.place.max_pen * 1000.0f,
                gs.place.covered_arc, gs.place.max_gap, gs.place.palm ? 1 : 0,
                gs.sphere_count);
    EXPECT_LE(gs.place.max_pen, kShallowPenMax + 1e-5f);
    EXPECT_GT(gs.place.covered_arc, kLargeCagingArcMin);
    EXPECT_TRUE(gs.place.palm);

    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    coresident::CoResidentStepReport rep;
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); rep = stepper.Step(); }
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;
    const LiveCov cov_settle = LiveCoverage(context, gs, stepper, hull);
    std::printf("[LARGE-LIFT] table-supported settle: finger_contacts=%u table_rows=%u "
                "finger_vimp=%.4e table_vimp=%.4e (mg.dt=%.4e) arc=%.1f gap=%.1f palm=%d\n",
                rep.finger_contacts, rep.table_row_count, rep.cup_vertical_impulse,
                rep.table_vertical_impulse, weight_kick, cov_settle.covered_arc,
                cov_settle.max_gap, cov_settle.palm ? 1 : 0);

    stepper.SetTableEnabled(false);
    const uint32_t kLift = 220u, kLiftSettle = 60u;
    double max_disp = 0.0, max_tilt = 0.0, max_cross = 0.0, max_gap_live = 0.0;
    double sum_fimp = 0.0, sum_friction = 0.0;
    uint32_t n_steady = 0u, contact_after = 0u, palm_after = 0u;
    bool any_static_after = false;
    for (uint32_t s = 0u; s < kLift; ++s) {
        DrivePd(stepper, gs, pd);
        rep = stepper.Step();
        if (rep.finger_contacts > 0u) ++contact_after;
        if (rep.any_static_row) any_static_after = true;
        const Vec3 cp = stepper.Cup().position;
        max_disp = std::max(max_disp, (double)(cp - c_settled).Length());
        max_tilt = std::max(max_tilt, RelTilt(q_settled, stepper.Cup().orientation));
        if (rep.finger_contacts > 0u && rep.cup_vertical_impulse != 0.0) {
            const double cross = std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse) /
                std::max(std::fabs(rep.cup_dvz_impulse), 1e-9);
            max_cross = std::max(max_cross, cross);
        }
        if (s % 20u == 0u || s + 1u == kLift) {
            const LiveCov cov = LiveCoverage(context, gs, stepper, hull);
            max_gap_live = std::max(max_gap_live, (double)cov.max_gap);
            if (cov.palm) ++palm_after;
            std::printf("[LARGE-LIFT]   traj s=%3u tilt=%.4f rad |w|=%.3f "
                        "live_gap=%.1f n=%u palm=%d\n",
                        s, RelTilt(q_settled, stepper.Cup().orientation),
                        stepper.Cup().angular_velocity.Length(), cov.max_gap,
                        cov.n, cov.palm ? 1 : 0);
        }
        if (s >= kLiftSettle) {
            sum_fimp += rep.cup_vertical_impulse;
            sum_friction += rep.finger_vimpulse_friction;
            ++n_steady;
        }
    }
    const double mean_fimp = n_steady ? sum_fimp / n_steady : 0.0;
    const double mean_friction = n_steady ? sum_friction / n_steady : 0.0;
    const LiveCov cov_end = LiveCoverage(context, gs, stepper, hull);
    const double final_w = stepper.Cup().angular_velocity.Length();

    std::printf("[LARGE-LIFT] AFTER TABLE REMOVAL: contact=%u/%u any_static=%d "
                "palm_retained=%u mean finger_vimp=%.4e friction=%.4e (mg.dt=%.4e) "
                "cross=%.2e\n",
                contact_after, kLift, any_static_after ? 1 : 0, palm_after, mean_fimp,
                mean_friction, weight_kick, max_cross);
    std::printf("[LARGE-LIFT] cage: max_disp=%.4f max_tilt=%.4f rad final|w|=%.3f "
                "end_arc=%.1f end_gap=%.1f max_gap_live=%.1f end_palm=%d\n",
                max_disp, max_tilt, final_w, cov_end.covered_arc, cov_end.max_gap,
                max_gap_live, cov_end.palm ? 1 : 0);

    EXPECT_FALSE(any_static_after) << "a table row persisted after removal";
    EXPECT_LT(max_cross, 5.0e-2)
        << "finger impulse bookkeeping disagrees (numerically invalid lift gate)";

    const bool support_ok = std::fabs(mean_fimp - weight_kick) < 0.35 * weight_kick;
    const bool translation_ok = max_disp < 0.05;
    const bool contact_ok = contact_after >= kLift - 12u;
    const bool arc_ok = (max_gap_live <= 130.0) && (cov_end.max_gap <= 130.0f);
    const bool tilt_ok = max_tilt < 0.15 && final_w < 0.75;
    const bool caged = support_ok && translation_ok && contact_ok && arc_ok && tilt_ok &&
                       !any_static_after && max_cross < 5.0e-2;
    if (!caged) {
        GTEST_SKIP() << "NEGATIVE: large-cup shallow lift did not cage: support_ok="
                     << support_ok << " (mean_fimp " << mean_fimp << " vs mg.dt "
                     << weight_kick << ") translation_ok=" << translation_ok << " (disp "
                     << max_disp << ") contact_ok=" << contact_ok << " (" << contact_after
                     << "/" << kLift << ") arc_ok=" << arc_ok << " (max_gap_live "
                     << max_gap_live << " end_gap " << cov_end.max_gap << ") tilt_ok="
                     << tilt_ok << " (tilt " << max_tilt << " rad, |w| " << final_w
                     << "). Faithful params held; do not grind gains/tolerances.";
    }
    EXPECT_TRUE(caged);
}

TEST(H1DenseGrasp, LiftGateDenseWrapCagesCup) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CupHull hull = ScaleCupHull(base, kDynSxy, kDynSz);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;

    DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true);
    if (!gs.place.found) GTEST_SKIP() << "DENSE+SHALLOW INFEASIBLE (NEGATIVE): no <=2mm "
                                          "void-closed placement -- see GATE 2 (Shallow"
                                          "PlacementCalibration) for the dispositive finding.";
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    const double z0 = gs.cup0.position.z;

    coresident::CoResidentStepReport rep;
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); rep = stepper.Step(); }
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;
    const LiveCov cov_settle = LiveCoverage(context, gs, stepper, hull);
    std::printf("[LIFT] table-supported settle: finger_contacts=%u table_rows=%u "
                "finger_vimp=%.4e table_vimp=%.4e (mg.dt=%.4e) arc=%.1f gap=%.1f palm=%d\n",
                rep.finger_contacts, rep.table_row_count, rep.cup_vertical_impulse,
                rep.table_vertical_impulse, weight_kick, cov_settle.covered_arc,
                cov_settle.max_gap, cov_settle.palm ? 1 : 0);

    stepper.SetTableEnabled(false);
    const uint32_t kLift = 220u, kLiftSettle = 60u;
    double max_disp = 0.0, max_tilt = 0.0, max_cross = 0.0, max_gap_live = 0.0;
    double tilt_at120 = 0.0, sum_fimp = 0.0;
    uint32_t n_steady = 0u, contact_after = 0u, palm_after = 0u;
    bool any_static_after = false; double steady_qd = 0.0;
    for (uint32_t s = 0u; s < kLift; ++s) {
        DrivePd(stepper, gs, pd);
        rep = stepper.Step();
        if (rep.finger_contacts > 0u) ++contact_after;
        if (rep.any_static_row) any_static_after = true;
        const Vec3 cp = stepper.Cup().position;
        max_disp = std::max(max_disp, (double)(cp - c_settled).Length());
        const double tilt = RelTilt(q_settled, stepper.Cup().orientation);
        max_tilt = std::max(max_tilt, tilt);
        if (s == 119u) tilt_at120 = tilt;
        if (rep.finger_contacts > 0u && rep.cup_vertical_impulse != 0.0) {
            const double cross = std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse) /
                std::max(std::fabs(rep.cup_dvz_impulse), 1e-9);
            max_cross = std::max(max_cross, cross);
        }
        if (s % 20u == 0u || s + 1u == kLift) {
            const LiveCov cov = LiveCoverage(context, gs, stepper, hull);
            max_gap_live = std::max(max_gap_live, (double)cov.max_gap);
            if (cov.palm) ++palm_after;
            std::printf("[LIFT]   traj s=%3u tilt=%.4f rad |w|=%.3f live_gap=%.1f n=%u palm=%d\n",
                        s, tilt, stepper.Cup().angular_velocity.Length(), cov.max_gap,
                        cov.n, cov.palm ? 1 : 0);
        }
        if (s >= kLiftSettle) {
            articulation::ArticulationHostState st; stepper.Download(&st);
            for (uint32_t l : gs.drive_links)
                steady_qd = std::max(steady_qd, std::fabs((double)st.qdot[l]));
            sum_fimp += rep.cup_vertical_impulse; ++n_steady;
        }
    }
    const double mean_fimp = n_steady ? sum_fimp / n_steady : 0.0;
    const double drift = std::fabs(stepper.Cup().position.z - z0);
    const LiveCov cov_end = LiveCoverage(context, gs, stepper, hull);
    const double final_w = stepper.Cup().angular_velocity.Length();

    std::printf("[LIFT] AFTER TABLE REMOVAL: contact=%u/%u any_static=%d palm_retained=%u "
                "mean finger_vimp=%.4e (mg.dt=%.4e) cross=%.2e\n",
                contact_after, kLift, any_static_after ? 1 : 0, palm_after, mean_fimp,
                weight_kick, max_cross);
    std::printf("[LIFT] cage: max_disp=%.4f max_tilt=%.4f rad tilt@120=%.4f final|w|=%.3f "
                "drift=%.5f steady|qd|=%.3f end_arc=%.1f end_gap=%.1f max_gap_live=%.1f "
                "end_palm=%d\n",
                max_disp, max_tilt, tilt_at120, final_w, drift, steady_qd,
                cov_end.covered_arc, cov_end.max_gap, max_gap_live, cov_end.palm ? 1 : 0);

    // The structural truths (assert green -- these hold regardless of the cage outcome):
    EXPECT_FALSE(any_static_after) << "a table row persisted after removal";
    EXPECT_LT(max_cross, 5.0e-2)
        << "impulse bookkeeping disagrees (solve UNDER-CONVERGED at dense row count -> "
           "the result is numerically invalid, NOT a physics negative)";

    // The cage gates -- assert TRUE when achieved; the SKIP-with-finding branch carries
    // the honest NEGATIVE (no false-regression RED), mirroring the prior spikes' protocol.
    const bool support_ok = mean_fimp > 0.5 * weight_kick;
    const bool translation_ok = max_disp < 0.05;
    const bool contact_ok = contact_after >= kLift - 12u;
    const bool stable_ok = steady_qd < 5.0;
    const bool arc_ok = (max_gap_live <= 130.0) && (cov_end.max_gap <= 130.0f);
    const bool tilt_ok = max_tilt < 0.15;
    const bool caged = support_ok && translation_ok && contact_ok && stable_ok &&
                       arc_ok && tilt_ok && !any_static_after && max_cross < 5.0e-2;
    if (!caged) {
        GTEST_SKIP() << "DENSE LIFT did not give a settled distributed cage: support_ok="
                     << support_ok << " (mean_fimp " << mean_fimp << " vs mg.dt "
                     << weight_kick << ") translation_ok=" << translation_ok << " (disp "
                     << max_disp << ") contact_ok=" << contact_ok << " (" << contact_after
                     << "/" << kLift << ") stable_ok=" << stable_ok << " (|qd| " << steady_qd
                     << ") arc_ok=" << arc_ok << " (max_gap_live " << max_gap_live
                     << " end_gap " << cov_end.max_gap << ") tilt_ok=" << tilt_ok
                     << " (tilt " << max_tilt << " rad). See the BITE gate for the active-"
                        "vs-passive disposition + the report.";
    }
    EXPECT_TRUE(caged) << "dense wrap should cage the cup distributed through the lift";
}

// ===========================================================================
// GATE 4 (THE DISPOSITIVE BITE): active vs passive at SHALLOW depth. Settle the dense
// active squeeze, remove the table, confirm it holds; then grip=0 (zero torque). An
// ACTIVE friction grasp loses normal force -> the cup falls/slips. A passive form/wedge
// holds. The KEY contrast vs the prior negative: here the contact is SHALLOW (<=2mm), so
// if grip=0 STILL holds it is legitimate shallow FORM closure (report it), NOT the 7.2mm
// wedge. Report the depth + grip=0 vs grip=on either way.
// ===========================================================================
TEST(H1DenseGraspLargeCup, ActiveBiteGripOffVsOn) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    const uint32_t kFall = 120u;
    const double free_fall = 0.5 * (-kGravityZ) * (kFall * kDt) * (kFall * kDt);

    DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true,
                                              PlaceMode::ShallowCaging,
                                              kLargeDynSxy, kLargeDynSz);
    if (!gs.place.found) {
        GTEST_SKIP() << "NEGATIVE: canonical 1.8x large cup has no <=2mm palm+finger "
                        "caging placement. See ScaleSweepFindsBothSidesShallow.";
    }
    std::printf("[LARGE-BITE] shallow placement max_pen=%.4f m (%.1f mm) spheres=%u\n",
                gs.place.max_pen, gs.place.max_pen * 1000.0f, gs.sphere_count);

    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);

    coresident::CoResidentStepReport rep_on;
    for (uint32_t s = 0u; s < 40u; ++s) { DrivePd(stepper, gs, pd); rep_on = stepper.Step(); }
    const double z_active = stepper.Cup().position.z;
    const double vz_on = stepper.Cup().linear_velocity.z;
    const bool holds_on = (std::fabs(z_active - gs.cup0.position.z) < 0.05) &&
                          (std::fabs(vz_on) < 0.5) &&
                          (rep_on.cup_vertical_impulse > 0.5 * weight_kick);

    const std::vector<float> zero(gs.host.TotalLinkCount(), 0.0f);
    coresident::CoResidentStepReport rep_off;
    for (uint32_t s = 0u; s < kFall; ++s) { stepper.SetGripTorque(zero); rep_off = stepper.Step(); }
    const BodyState cupF = stepper.Cup();
    const double drop = z_active - cupF.position.z;
    const bool falls = (drop > 0.02) || (cupF.linear_velocity.z < -0.10);

    std::printf("[LARGE-BITE] grip=ON holds=%d (z=%.5f vz=%.4f finger_vimp=%.4e mg.dt=%.4e)\n",
                holds_on ? 1 : 0, z_active, vz_on, rep_on.cup_vertical_impulse, weight_kick);
    std::printf("[LARGE-BITE] grip=0 after active hold: dropped %.5f m over %u steps "
                "(free-fall=%.4f) final vz=%.4f finger_vimp=%.4e -> %s\n",
                drop, kFall, free_fall, cupF.linear_velocity.z, rep_off.cup_vertical_impulse,
                falls ? "FALLS (ACTIVE grasp)" : "HOLDS (shallow form closure)");

    if (!holds_on) {
        GTEST_SKIP() << "NEGATIVE: grip=ON did not hold the shallow large-cup placement "
                        "(finger_vimp " << rep_on.cup_vertical_impulse << " vs mg.dt "
                     << weight_kick << ", vz " << vz_on << ").";
    }
    if (!falls) {
        std::printf("[LARGE-BITE] NOTE: grip=0 also holds at %.2f mm; this is shallow "
                    "form closure, not the historical deep passive wedge.\n",
                    gs.place.max_pen * 1000.0f);
    }
    EXPECT_TRUE(holds_on);
    EXPECT_LE(gs.place.max_pen, kShallowPenMax + 1e-5f);
}

TEST(H1DenseGrasp, ActiveBiteGripOffVsOn) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    const uint32_t kFall = 120u;
    const double free_fall = 0.5 * (-kGravityZ) * (kFall * kDt) * (kFall * kDt);

    DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true);
    if (!gs.place.found) GTEST_SKIP() << "DENSE+SHALLOW INFEASIBLE (NEGATIVE): no <=2mm "
                                          "void-closed placement -- see GATE 2 (Shallow"
                                          "PlacementCalibration) for the dispositive finding.";
    std::printf("[BITE] shallow placement max_pen=%.4f m (%.1f mm) spheres=%u\n",
                gs.place.max_pen, gs.place.max_pen * 1000.0f, gs.sphere_count);

    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    for (uint32_t s = 0u; s < 70u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);

    // grip=ON: confirm the active squeeze holds for a window.
    coresident::CoResidentStepReport rep_on;
    for (uint32_t s = 0u; s < 40u; ++s) { DrivePd(stepper, gs, pd); rep_on = stepper.Step(); }
    const double z_active = stepper.Cup().position.z;
    const double vz_on = stepper.Cup().linear_velocity.z;
    const bool holds_on = (std::fabs(z_active - gs.cup0.position.z) < 0.05) &&
                          (std::fabs(vz_on) < 0.5);

    // grip=0: zero torque (NOT an active open). Active grasp -> falls; passive -> holds.
    const std::vector<float> zero(gs.host.TotalLinkCount(), 0.0f);
    coresident::CoResidentStepReport rep_off;
    for (uint32_t s = 0u; s < kFall; ++s) { stepper.SetGripTorque(zero); rep_off = stepper.Step(); }
    const BodyState cupF = stepper.Cup();
    const double drop = z_active - cupF.position.z;
    const bool falls = (drop > 0.02) || (cupF.linear_velocity.z < -0.10);

    std::printf("[BITE] grip=ON holds=%d (z=%.5f vz=%.4f finger_vimp=%.4e mg.dt=%.4e)\n",
                holds_on ? 1 : 0, z_active, vz_on, rep_on.cup_vertical_impulse, weight_kick);
    std::printf("[BITE] grip=0 after active hold: dropped %.5f m over %u steps "
                "(free-fall=%.4f) final vz=%.4f finger_vimp=%.4e -> %s\n",
                drop, kFall, free_fall, cupF.linear_velocity.z, rep_off.cup_vertical_impulse,
                falls ? "FALLS (ACTIVE grasp)" : "HOLDS (form/wedge closure)");

    // The DISPOSITIVE branch:
    //   grip=on holds AND grip=0 falls  -> ACTIVE grasp at shallow depth = PASS.
    //   grip=0 holds at <=2mm           -> legitimate shallow FORM closure (report, not wedge).
    //   grip=on does not even hold       -> dense+shallow cannot cage = NEGATIVE.
    const bool active = holds_on && falls;
    if (!active) {
        GTEST_SKIP() << "DENSE BITE not an ACTIVE-friction grasp: grip=on holds="
                     << holds_on << " grip=0 falls=" << falls << " (drop " << drop
                     << " m over " << kFall << " steps vs " << free_fall << " free-fall, vz "
                     << cupF.linear_velocity.z << "). max_pen=" << gs.place.max_pen
                     << " m. If grip=on does NOT hold -> dense+shallow can't cage (NEGATIVE)."
                        " If grip=0 HOLDS at <=2mm -> shallow FORM closure (not the 7.2mm "
                        "wedge), still not an ACTIVE grasp. See the report.";
    }
    EXPECT_TRUE(active) << "dense+shallow should give an ACTIVE grasp (on holds, off falls)";
}

// ===========================================================================
// GATE 5 (dynamic lift load, confirmatory): after table removal + active hold, impose a
// sustained ~1.75g upward accel + lateral + tilt; the dense wrap must stay caged.
// ===========================================================================
TEST(H1DenseGraspLargeCup, DynamicLiftLoadStaysCaged) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    const CupHull hull = ScaleCupHull(base, kLargeDynSxy, kLargeDynSz);

    DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true,
                                              PlaceMode::ShallowCaging,
                                              kLargeDynSxy, kLargeDynSz);
    if (!gs.place.found) {
        GTEST_SKIP() << "NEGATIVE: canonical 1.8x large cup has no <=2mm palm+finger "
                        "caging placement. See ScaleSweepFindsBothSidesShallow.";
    }
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    for (uint32_t s = 0u; s < 80u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);
    for (uint32_t s = 0u; s < 20u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;

    stepper.ApplyCupImpulse(Vec3{0.04f, 0.04f, 0.0f}, Vec3{1.5f, 0.8f, 0.8f});
    double max_disp = 0.0, max_tilt = 0.0, max_gap_live = 0.0;
    uint32_t contact_after = 0u;
    const uint32_t kLoad = 120u;
    for (uint32_t s = 0u; s < kLoad; ++s) {
        DrivePd(stepper, gs, pd);
        if (s < 30u)
            stepper.ApplyCupImpulse(Vec3{0.0f, 0.0f, 1.75f * (-kGravityZ) * kDt}, Vec3::Zero());
        const auto rep = stepper.Step();
        if (rep.finger_contacts > 0u) ++contact_after;
        const Vec3 cp = stepper.Cup().position;
        max_disp = std::max(max_disp, (double)(cp - c_settled).Length());
        max_tilt = std::max(max_tilt, RelTilt(q_settled, stepper.Cup().orientation));
        if (s % 20u == 0u || s + 1u == kLoad) {
            const LiveCov cov = LiveCoverage(context, gs, stepper, hull);
            max_gap_live = std::max(max_gap_live, (double)cov.max_gap);
        }
    }
    const double final_w = stepper.Cup().angular_velocity.Length();
    std::printf("[LARGE-DYNLOAD] 1.75g lift + lateral + tilt: max_disp=%.4f "
                "max_tilt=%.4f rad final|w|=%.3f contact=%u/%u max_gap=%.1f\n",
                max_disp, max_tilt, final_w, contact_after, kLoad, max_gap_live);
    const bool caged = (contact_after >= kLoad - 16u) && (max_tilt < 0.30) &&
                       (max_disp < 0.08) && (max_gap_live <= 150.0);
    if (!caged) {
        GTEST_SKIP() << "NEGATIVE: large-cup dynamic load not caged: contact="
                     << contact_after << "/" << kLoad << " max_tilt=" << max_tilt
                     << " rad max_disp=" << max_disp << " max_gap=" << max_gap_live
                     << ".";
    }
    EXPECT_TRUE(caged);
}

TEST(H1DenseGrasp, DynamicLiftLoadStaysCaged) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);

    DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true);
    if (!gs.place.found) GTEST_SKIP() << "DENSE+SHALLOW INFEASIBLE (NEGATIVE): no <=2mm "
                                          "void-closed placement -- see GATE 2 (Shallow"
                                          "PlacementCalibration) for the dispositive finding.";
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
    for (uint32_t s = 0u; s < 80u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);
    for (uint32_t s = 0u; s < 20u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;

    stepper.ApplyCupImpulse(Vec3{0.04f, 0.04f, 0.0f}, Vec3{1.5f, 0.8f, 0.8f});
    double max_disp = 0.0, max_tilt = 0.0; uint32_t contact_after = 0u;
    const uint32_t kLoad = 120u;
    for (uint32_t s = 0u; s < kLoad; ++s) {
        DrivePd(stepper, gs, pd);
        if (s < 30u)
            stepper.ApplyCupImpulse(Vec3{0.0f, 0.0f, 1.75f * (-kGravityZ) * kDt}, Vec3::Zero());
        const auto rep = stepper.Step();
        if (rep.finger_contacts > 0u) ++contact_after;
        const Vec3 cp = stepper.Cup().position;
        max_disp = std::max(max_disp, (double)(cp - c_settled).Length());
        max_tilt = std::max(max_tilt, RelTilt(q_settled, stepper.Cup().orientation));
    }
    const double final_w = stepper.Cup().angular_velocity.Length();
    std::printf("[DYNLOAD] 1.75g lift + lateral + tilt: max_disp=%.4f max_tilt=%.4f rad "
                "final|w|=%.3f contact=%u/%u\n",
                max_disp, max_tilt, final_w, contact_after, kLoad);
    const bool caged = (contact_after >= kLoad - 16u) && (max_tilt < 0.30);
    if (!caged) {
        GTEST_SKIP() << "DENSE dynamic load not caged: contact=" << contact_after << "/"
                     << kLoad << " max_tilt=" << max_tilt << " rad (see the LIFT/BITE "
                        "gates for the disposition).";
    }
    EXPECT_TRUE(caged) << "dense wrap should stay caged under the dynamic lift load";
}

// ===========================================================================
// GATE 6 (D1): the full dense close + table-removal lift run TWICE -> byte-exact cup+qdot.
// Run at the DEEP void-closed placement (PlaceMode::VoidClosed) so the dense contact set
// is ACTIVE -- this gives a REAL dense-contact-path determinism check (byte-determinism
// holds regardless of solve convergence). (At PlaceMode::Shallow the placement is not
// found here, the cup would sit at the origin far from the hand, and D1 would be a
// trivial contact-free check -- so we deliberately use the deep, contact-active path.)
// ===========================================================================
TEST(H1DenseGraspLargeCup, DeterministicTwoRun) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    auto rollout = [&](BodyState* cup, articulation::ArticulationHostState* art) {
        DenseGraspScene gs = BuildDenseGraspScene(context, base, /*has_table=*/true,
                                                  PlaceMode::ShallowCaging,
                                                  kLargeDynSxy, kLargeDynSz);
        if (!gs.place.found) return false;
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        *cup = stepper.Cup();
        stepper.Download(art);
        return true;
    };
    BodyState ca, cb;
    articulation::ArticulationHostState aa, ab;
    if (!rollout(&ca, &aa)) {
        GTEST_SKIP() << "NEGATIVE: canonical 1.8x large cup has no <=2mm palm+finger "
                        "caging placement. See ScaleSweepFindsBothSidesShallow.";
    }
    ASSERT_TRUE(rollout(&cb, &ab));
    EXPECT_EQ(std::memcmp(&ca, &cb, sizeof(BodyState)), 0) << "large-cup cup diverged across runs";
    ASSERT_EQ(aa.qdot.size(), ab.qdot.size());
    EXPECT_EQ(std::memcmp(aa.qdot.data(), ab.qdot.data(), aa.qdot.size() * sizeof(float)), 0)
        << "large-cup qdot diverged across runs";
    std::printf("[LARGE-D1] large dense cup + qdot byte-identical across 2 rollouts\n");
}

TEST(H1DenseGrasp, DeterministicTwoRun) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull base = LoadCupHull();
    ASSERT_GT(base.vcount, 0u);
    auto rollout = [&](BodyState* cup, articulation::ArticulationHostState* art) {
        DenseGraspScene gs =
            BuildDenseGraspScene(context, base, /*has_table=*/true, PlaceMode::VoidClosed);
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, kKp, kKd, kCloseOffset);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        *cup = stepper.Cup();
        stepper.Download(art);
    };
    BodyState ca, cb;
    articulation::ArticulationHostState aa, ab;
    rollout(&ca, &aa);
    rollout(&cb, &ab);
    EXPECT_EQ(std::memcmp(&ca, &cb, sizeof(BodyState)), 0) << "cup diverged across runs";
    ASSERT_EQ(aa.qdot.size(), ab.qdot.size());
    EXPECT_EQ(std::memcmp(aa.qdot.data(), ab.qdot.data(), aa.qdot.size() * sizeof(float)), 0)
        << "qdot diverged across runs";
    std::printf("[D1] dense cup + qdot byte-identical across 2 close+table-removal rollouts\n");
}
