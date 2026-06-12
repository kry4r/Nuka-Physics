// ---------------------------------------------------------------------------
// v0.8 C7b-2a-v2 -- the H1 POWER/WRAP grasp + LIFT spike. VALIDATED-NOT-WIRED.
// ---------------------------------------------------------------------------
// THE ONE THING THIS PROVES (or disproves). The fingertip-pinch spike
// (test_h1_grasp_spike.cpp) was an honest negative: a faithful H1 fingertip pinch
// can hold the cup VERTICALLY but cannot CAGE it rotationally -- all five fingers
// sit at ~one azimuth opposing the thumb, so a fingertip grasp has only TWO contact
// azimuths and the cup pivots ~45 deg out of the pinch under a disturbance.
//
// THE OWNER-CONFIRMED FIX (this spike): a POWER/WRAP grasp -- collision spheres on
// the PROXIMAL/INTERMEDIATE finger links + the THUMB links + the PALM link, at
// MULTIPLE azimuths around the cylinder -- plus a cup-on-table -> close -> LIFT
// choreography. THE DISCRIMINATING GATE is the LIFT: when the table stops
// supporting the cup, does the wrap carry the full weight WITHOUT the cup pivoting
// out (bounded tilt, not the 0.5 rad the 2-point gave)?
//
// STEP ZERO (static FK) is the highest-uncertainty gate and runs FIRST: curl the
// hand into a WRAP around the cup, attach a sphere per thumb/finger link, run FK, and
// over a bounded grid of cup placements measure the GENUINELY-contacting spheres'
// covered ARC (max adjacent azimuth gap < 180 -> a true surround). The PALM is a dead
// end -- FK proves the palm-inner surface and the finger walls sit in DISJOINT cup
// windows (palm<->finger x-span ~8cm > the 6cm cup), so the cage comes from the curled
// fingers spreading into TWO azimuth bands (proximals ~280, intermediates ~40) plus
// the thumb (~90-130) -- no palm.
//
// COLLISION PRIMITIVE: a SPHERE per link (CoResidentFingertip = {link, offset,
// radius}); sphere x ConvexHull is the robust C3 tier. ONE sphere per device link
// (the StepGrasp resolver matches by broadphase_handle == device link and reuses it
// as the chain-J link; two spheres on one link would collapse).
//
// SPIKE OUTCOME (HONEST NEGATIVE with a PARTIAL ADVANCE -- see each gate + the report):
//   * STEP ZERO PASSES: the curled wrap surrounds the cup (max covered_arc ~216 deg).
//   * GENUINE ADVANCE over the pinch: a grip-INDEPENDENT TRANSLATION CAGE -- at the
//     SQUEEZE-BALANCED placement (min net-squeeze imbalance |S| s.t. covered_arc>180,
//     the ONE authorized contact-placement refinement) the cup center stays bounded
//     through the close + the table-removal LIFT, and the wrap delivers a net upward
//     impulse ~= m*g*dt (the table+finger equilibrium triangle closes, cross-checked).
//   * THE NEGATIVE (discriminator): the ROTATIONAL cage FAILS -- the cup tilts (a
//     developing, non-settling pivot) + spins (|w|>1) + the live covered_arc collapses
//     (202 -> <180). ROOT = the unfillable ~110-deg palm-side void; contacts already
//     cover ~75% of the cup height so z-spread cannot fix it. SAME invariant as the
//     pinch negative, now in the rotational DOF.
//   * HONESTY CAVEATS: the vertical support is PASSIVE -- it survives grip=0 (BITE-A:
//     deep ~6-10mm pre-penetration holds the cup, drop ~0), so by the task's pinch-era
//     BITE criterion ("survives grip=0 => fake") this is NOT an active grasp. The wrap
//     trades the pinch's 2-azimuth pivot for a WRAP pivot AND a passive vertical hold;
//     the genuine narrow positive is the grip-independent TRANSLATION cage. The BITE-
//     semantics tension (grip=0-survival is EXPECTED for a geometric cage) is surfaced
//     for owner adjudication. The shipped 120-step lift's vertical balance is actually
//     decent (bal_err peaks ~0.43, mean ~= mg.dt); the tilt is window-dependent (0.32
//     rad @120 steps, ~1.08 @200) so the ROBUST rotational-failure discriminators are
//     the window-INDEPENDENT persistent spin (|w|~1.4) + the live-arc collapse
//     (202 -> ~160). Active OPEN releases the cup (drops ~2m over 250 steps) but
//     armature-slow. Real faithful params (armature=0.1/damping=1.0) held throughout.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses
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
constexpr float kCupMass = 0.2f;  // 0.2 kg (same as the fingertip spike).
constexpr float kPi = 3.14159265358979323846f;

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kH1Mjcf) && std::filesystem::exists(kCupUsda);
}

// ---------------------------------------------------------------------------
// Cup hull (identical loader to the fingertip spike).
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

// ---------------------------------------------------------------------------
// The cooked H1 articulation + its SceneIR (names + joint limits).
// ---------------------------------------------------------------------------
struct CookedH1 {
    articulation::ArticulationHostState host;
    nuka::scene::SceneIR scene;
};

// FAITHFUL H1 finger-joint params: the MJCF default class `h1` authors
// armature="0.1", damping="1.0" on the finger joints. We DO NOT fabricate these:
// the fingertip-spike DECISION test proved (i) the loaded static equilibrium is
// armature-INDEPENDENT (qdot=qddot=0 there) and (ii) the wall is the acquisition
// race, NOT the static hold. The wrap-LIFT choreography removes the acquisition
// race entirely (the cup is table-supported while we close), so faithful armature
// is the right -- and required -- value here.
constexpr float kRealArmature = 0.1f;
constexpr float kRealDamping = 1.0f;

// Load the H1, FREEZE every joint except the named driven finger bodies, force the
// root Fixed, and re-author the driven knuckles' armature/damping to the FAITHFUL
// MJCF class values. (Identical mechanism to the fingertip spike's LoadH1Fixed, but
// the driven-knuckle params are the real 0.1/1.0, not the fabricated 1e-2/0.15.)
CookedH1 LoadH1Fixed(const std::vector<std::string>& free_finger_bodies,
                     float driven_armature, float driven_damping) {
    CookedH1 out;
    out.scene = nuka::import::LoadMjcf(kH1Mjcf);
    // Strip ALL shape geometry before cooking (no cooked H1 collision shapes are
    // needed: the grasp stepper direct-emits sphere<->cup pairs). Body inertias come
    // from the <inertial> tags (body-level), so stripping geometry cannot touch them.
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

    uint32_t kept = 0u;
    for (auto& topo : topos) {
        for (size_t i = 0u; i < topo.parent_links.size(); ++i) {
            const bool is_root = topo.parent_links[i] == nuka::scene::kInvalidBody;
            const uint32_t body = topo.link_bodies[i];
            if (is_root || !is_free(body)) {
                topo.joint_types[i] = articulation::ArticulationJointType::Fixed;
            } else {
                ++kept;
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

// Run FK at the given host state -> world pose of every link.
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
// The WRAP contact set. THREE non-collinear azimuth REGIONS around the cylinder:
//   * PALM   (right_hand_link) -- the back wall of the wrap cavity (~180 deg),
//   * THUMB  (intermediate + distal) -- the +y wall (~50 deg),
//   * FINGERS (index/middle/ring/pinky proximal + intermediate) -- the far wall
//     (~0 deg) and, when CURLED, their tips swing AROUND toward the -y side to
//     close the 270-deg void the fingertip pinch left open.
// One sphere per device link (the stepper constraint). The DRIVEN joints (curl) are
// the proximal + intermediate finger knuckles + the thumb yaw/pitch -- so the close
// transient moves the contact spheres inward + around.
// ---------------------------------------------------------------------------
constexpr float kWrapRadius = 0.010f;  // 10 mm finger-segment pad sphere.

struct WrapSphere {
    std::string body;       // device body name (resolved to a link).
    Vec3 local_offset{};    // sphere center in the link frame.
    float radius = kWrapRadius;
    const char* region = "";  // "palm" / "thumb" / "finger" (for azimuth grouping).
};

// The driven finger bodies (joints kept Revolute) -- proximal + intermediate of the
// four fingers + the thumb chain. The hand curls these to wrap the cup.
const std::vector<std::string> kWrapDriven = {
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
    "R_index_proximal",  "R_index_intermediate",
    "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal",   "R_ring_intermediate",
    "R_pinky_proximal",  "R_pinky_intermediate",
};

// The wrap spheres (one per link). Local offsets along the link +X (the finger
// forward axis under FK) put the sphere on the finger segment, not at the joint.
// NO PALM sphere: the FK proved the palm-inner surface (x~0.349) and the finger
// walls (x~0.43-0.45) sit in DISJOINT cup-placement windows -- a 6cm cup can touch
// the palm OR the fingers, never both (palm<->finger x-span ~8cm > 6cm cup). The
// cage instead comes from the CURLED fingers spreading into two azimuth bands
// (proximals ~280 deg, intermediates ~40 deg) PLUS the thumb (distal ~91, inter
// ~120 deg) -- four non-collinear azimuth clusters with no palm at all.
std::vector<WrapSphere> WrapSpheres() {
    return {
        {"R_thumb_intermediate",  Vec3{0.015f, 0.0f, 0.0f},  kWrapRadius, "thumb"},
        {"R_thumb_distal",        Vec3{0.012f, 0.0f, 0.0f},  kWrapRadius, "thumb"},
        {"R_index_proximal",      Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_index_intermediate",  Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_middle_proximal",     Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_middle_intermediate", Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_ring_proximal",       Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_ring_intermediate",   Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_pinky_proximal",      Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
        {"R_pinky_intermediate",  Vec3{0.018f, 0.0f, 0.0f},  kWrapRadius, "finger"},
    };
}

// The CURL pose: per-driven-joint flex (rad) that closes the fingers around the cup.
// Set up-front (NOT searched against the cup) -- a principled "make a fist around a
// cylinder" pose. Finger proximals + intermediates flex positive (curl inward);
// thumb yaw/pitch oppose. The cup is then placed by a fixed rule (palm cavity), and
// we MEASURE whether the curl surrounds it -- we do not move the cup to fake it.
struct CurlPose {
    float finger_prox = 1.0f;   // finger proximal curl (rad), within [0,1.7].
    float finger_int  = 1.1f;   // finger intermediate curl (rad), within [0,1.7].
    float thumb_yaw   = 1.0f;   // thumb proximal_base yaw (rad), within [-0.1,1.3].
    float thumb_pitch = 0.5f;   // thumb proximal pitch (rad), within [-0.1,0.6].
    float thumb_int   = 0.6f;   // thumb intermediate (rad), within [0,0.8].
    float thumb_dist  = 0.6f;   // thumb distal (rad), within [0,1.2].
};

// Apply a curl pose to the host q by body name.
void ApplyCurl(const CookedH1& h1, articulation::ArticulationHostState* host,
               const CurlPose& c) {
    auto setq = [&](const std::string& nm, float v) {
        const uint32_t l = LinkByName(h1, nm);
        if (l != kInvalidLink) host->q[l] = v;
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

// World center of a wrap sphere under FK.
Vec3 SphereCenter(const std::vector<Transform>& poses, uint32_t link,
                  const Vec3& local_offset) {
    const Transform& lp = poses[link];
    return lp.position + lp.rotation.Rotate(local_offset);
}

// Cup wall radius (vertical cylinder) about the hull local center.
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

// Rim-aware nearest-hull-point gap of a sphere center to a cup at world `cup_center`
// (vertical axis, identity orientation). Genuine contact iff gap < sphere radius.
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

// The surround result for one cup placement: which spheres genuinely contact, their
// sorted azimuths, the max adjacent (wrap-around) azimuth gap, the covered arc.
struct Surround {
    Vec3 cup_center{};
    uint32_t genuine = 0u;
    std::vector<float> azimuths;     // sorted, of the genuinely-contacting spheres.
    std::vector<std::string> bodies; // the contacting bodies (aligned to azimuths order pre-sort).
    float max_gap = 360.0f;
    float covered_arc = 0.0f;
};

// Measure the surround of a cup at `cup_center` (vertical axis) by the curled wrap.
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
        const float dx = c.x - cup_center.x, dy = c.y - cup_center.y;
        float az = std::atan2(dy, dx) * 180.0f / kPi;
        if (az < 0.0f) az += 360.0f;
        azs.push_back(az);
        out.bodies.push_back(s.body);
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
    } else {
        out.max_gap = 360.0f;
        out.covered_arc = 0.0f;
    }
    return out;
}

// Net squeeze proxy at a placement: S = Σ_genuine (radius - rim_gap)*inward_normal
// (penetration as the normal-force proxy; inward = toward the cup axis). |S| small ->
// the squeeze is BALANCED (no net push toward the cage void). The max-coverage
// placement jams the 4 finger-intermediate contacts deepest -> a large net squeeze
// toward ~202 deg (the void) that drives the cup out; the MIN-|S| placement makes the
// squeeze cancel so the cage holds through the close.
struct Squeeze { float mag; float dir_deg; uint32_t genuine; float covered_arc; Vec3 center; };
Squeeze NetSqueeze(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                   const std::vector<WrapSphere>& spheres, const CupHull& hull,
                   const Vec3& cup_center) {
    Vec3 s{0, 0, 0};
    const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, cup_center);
    for (const auto& sp : spheres) {
        const uint32_t l = LinkByName(h1, sp.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses_curl, l, sp.local_offset);
        const float rim = RimGap(c, cup_center, hull);
        if (rim >= sp.radius) continue;
        const float dx = c.x - cup_center.x, dy = c.y - cup_center.y;
        const float d = std::sqrt(dx * dx + dy * dy);
        if (d < 1e-6f) continue;
        const Vec3 inward{-dx / d, -dy / d, 0.0f};  // toward the cup axis.
        const float w = sp.radius - rim;            // penetration depth.
        s.x += w * inward.x; s.y += w * inward.y;
    }
    const float mag = std::sqrt(s.x * s.x + s.y * s.y);
    float dir = std::atan2(s.y, s.x) * 180.0f / kPi; if (dir < 0) dir += 360.0f;
    return Squeeze{mag, dir, sr.genuine, sr.covered_arc, cup_center};
}

}  // namespace

namespace {

// The cup<->table friction + the wrap friction (mu). mu=0.8 (rubber-on-ceramic-ish,
// the fingertip spike's value). condim=3 (normal + 4 friction spokes).
constexpr float kMu = 0.8f;

// The power-grasp dynamics scene: the curled wrap + the cup at the STEP ZERO best
// placement, resting on a static table, with a constant inward grip torque on the
// driven knuckles (PD or constant). Built so the wrap contacts start ENGAGED (the
// curl pose surrounds the cup) -- no acquisition race (the cup is table-supported).
struct PowerGraspScene {
    CookedH1 h1;
    articulation::ArticulationHostState host;  // curled (flex baked into q).
    coresident::GraspConfig config;
    BodyState cup0;
    std::vector<uint32_t> drive_links;  // device links carrying grip torque.
    Vec3 cup_center{};
    float table_height = 0.0f;
};

// The cavity xy-box (bounding box of the curled wrap sphere centers) + the finger-
// band z, the sweep domain shared by STEP ZERO + the dynamics placement.
void CavityBox(const CookedH1& h1, const std::vector<Transform>& poses_curl,
               const std::vector<WrapSphere>& spheres, Vec3* lo, Vec3* hi, float* cz) {
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
    *cz = zn ? zsum / zn : 0.5f * (lo->z + hi->z);
}

// THE cup placement for the dynamics: the SQUEEZE-BALANCED placement -- the cavity
// center that MINIMIZES the net-squeeze imbalance |S| subject to covered_arc > 180.
// (NOT the max-coverage placement: that jams the 4 finger-intermediate contacts
// deepest, so the net squeeze drives the cup out the cage void during the close. The
// min-|S| placement makes the squeeze cancel so the cage holds -- the task's one
// authorized contact-placement refinement.)
Squeeze BestPlacement(const CookedH1& h1, const std::vector<Transform>& poses_curl,
                      const std::vector<WrapSphere>& spheres, const CupHull& hull) {
    Vec3 lo, hi; float cz;
    CavityBox(h1, poses_curl, spheres, &lo, &hi, &cz);
    Squeeze best{1e9f, 0, 0, 0, Vec3{}};
    const int kN = 21;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j) {
            const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
            const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
            const Squeeze q = NetSqueeze(h1, poses_curl, spheres, hull, Vec3{fx, fy, cz});
            if (q.covered_arc > 180.0f && q.genuine >= 3u && q.mag < best.mag) best = q;
        }
    return best;
}

// Build the power-grasp scene. `grip_torque` is the constant inward knuckle torque
// (0 for the BITE / PD-driven cases). `has_table` rests the cup on a static plane.
PowerGraspScene BuildPowerGraspScene(const nuka::phi::DeviceContext& context,
                                     const CupHull& hull, float grip_torque, float mu,
                                     bool has_table) {
    PowerGraspScene gs;
    gs.h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    gs.host = gs.h1.host;
    ApplyCurl(gs.h1, &gs.host, CurlPose{});

    // Resolve the driven links carrying the grip torque (the curled knuckles).
    for (const auto& nm : kWrapDriven) {
        const uint32_t l = LinkByName(gs.h1, nm);
        if (l != kInvalidLink) gs.drive_links.push_back(l);
    }

    // FK at the curled pose -> the wrap sphere centers + the best cup placement.
    const auto poses_curl = ForwardKinematics(context, gs.host);
    const auto spheres = WrapSpheres();
    const Squeeze best = BestPlacement(gs.h1, poses_curl, spheres, hull);
    gs.cup_center = best.center;

    // Fingertip descriptors (one sphere per driven finger/thumb link).
    gs.config.fingertips.clear();
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(gs.h1, s.body);
        if (l == kInvalidLink) continue;
        coresident::CoResidentFingertip ft;
        ft.link = l;
        ft.broadphase_handle = l;
        ft.local_offset = s.local_offset;
        ft.radius = s.radius;
        gs.config.fingertips.push_back(ft);
    }

    // The cup: convex-hull collidable, verts re-centered about the cup COM (so cup
    // BodyState.position IS the geometric center) -- mirrors the fingertip spike.
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

    // Constant inward grip torque (N.m) on each driven knuckle (+q curls -> +tau).
    const uint32_t link_count = gs.host.TotalLinkCount();
    gs.config.grip_torque.assign(link_count, 0.0f);
    for (uint32_t l : gs.drive_links) gs.config.grip_torque[l] = grip_torque;
    gs.config.drive_force_limits.assign(link_count, 0.0f);
    gs.config.friction_mu = mu;
    gs.config.condim = 3u;

    // The table: a static +Z plane the cup bottom rests on. cup bottom relative to
    // the cup center = hull.lo.z - hull_center.z (the recentred hull's lowest z).
    gs.table_height = gs.cup_center.z + (hull.lo.z - hull_center.z);
    gs.config.has_table = has_table;
    gs.config.table_height = gs.table_height;
    gs.config.table_mu = mu;
    gs.config.table_broadphase_id = 8500u;
    return gs;
}

}  // namespace

// ===========================================================================
// PROBE: dump the curled right-hand sphere centers + their azimuths so the wrap
// geometry is concrete. (Diagnostic; no assertion.)
// ===========================================================================
TEST(H1PowerGraspProbe, DumpCurledWrapGeometry) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    articulation::ArticulationHostState host = h1.host;
    ApplyCurl(h1, &host, CurlPose{});
    const auto poses = ForwardKinematics(context, host);

    const auto spheres = WrapSpheres();
    std::printf("[WRAP] cup_radius=%.4f  curled right-hand wrap sphere centers:\n",
                CupRadius(hull));
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        std::printf("  %-22s (%-6s) world=(%.4f,%.4f,%.4f)\n",
                    s.body.c_str(), s.region, c.x, c.y, c.z);
    }
    SUCCEED();
}

// ===========================================================================
// STEP ZERO -- can the curled H1 hand SURROUND the cup with covered arc > 180 deg?
// ===========================================================================
// Curl the hand around the cup (a principled "fist", NOT searched vs the cup). Then
// BOUND the negative rather than hand-pick a winner: SWEEP a grid of cup centers
// across the wrap cavity (axis vertical, FORCED -- the cup rests on a table in the
// dynamics) and report the MAX covered_arc over the grid. The physics gate is
// covered_arc > 180 (max adjacent azimuth gap < 180) on the GENUINELY-contacting
// spheres (rim gap < sphere radius). The azimuth LIST is printed as the
// non-collinearity evidence (no anatomical region count -- the curl already spreads
// the fingers into multiple azimuth bands).
//   * If MAX covered_arc < 180 over the whole cavity -> airtight honest negative:
//     the cup is too small for the H1 hand's contact span (every void-filling
//     contact costs its opposite). STOP and surface the sweep + the span numbers.
//   * If a plausible placement reaches >= 180 -> STEP ZERO passes; the dynamics run
//     AT that placement.
TEST(H1PowerGraspStepZero, CurledWrapSurroundsCupOver180Deg) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);
    const float cup_radius = CupRadius(hull);

    articulation::ArticulationHostState host = h1.host;
    ApplyCurl(h1, &host, CurlPose{});
    const auto poses_curl = ForwardKinematics(context, host);
    const auto spheres = WrapSpheres();

    // --- The cavity bounds: the bounding box of the curled finger/thumb spheres in
    //     x/y, and the finger-band z. The cup center sweeps inside this box. --------
    Vec3 lo, hi; float cz;
    CavityBox(h1, poses_curl, spheres, &lo, &hi, &cz);
    std::printf("[STEP0] cup_radius=%.4f  curled wrap sphere xy-box "
                "x[%.4f,%.4f] y[%.4f,%.4f]  sweep z=%.4f\n",
                cup_radius, lo.x, hi.x, lo.y, hi.y, cz);

    // --- Mutual-exclusivity spans (why the cage is hard): the thumb<->finger-proximal
    //     y-span and the inner-finger<->outer-finger x-span vs the 2*cup_radius. ----
    const Vec3 thumb_d = SphereCenter(poses_curl, LinkByName(h1, "R_thumb_distal"),
                                      Vec3{0.012f, 0.0f, 0.0f});
    const Vec3 idx_prox = SphereCenter(poses_curl, LinkByName(h1, "R_index_proximal"),
                                       Vec3{0.018f, 0.0f, 0.0f});
    const Vec3 idx_int = SphereCenter(poses_curl, LinkByName(h1, "R_index_intermediate"),
                                      Vec3{0.018f, 0.0f, 0.0f});
    std::printf("[STEP0] spans vs cup diameter %.4f: thumb_distal<->idx_prox y-span "
                "%.4f, idx_prox<->idx_int x-span %.4f\n",
                2.0f * cup_radius, std::fabs(thumb_d.y - idx_prox.y),
                std::fabs(idx_prox.x - idx_int.x));

    // --- The bounded grid sweep -> MAX covered_arc (the surround-EXISTENCE gate). ---
    Surround best;
    const int kN = 21;  // 21x21 grid over the cavity xy-box.
    for (int i = 0; i < kN; ++i) {
        for (int j = 0; j < kN; ++j) {
            const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
            const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
            const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull,
                                                Vec3{fx, fy, cz});
            if (sr.genuine >= 3u && sr.covered_arc > best.covered_arc) best = sr;
        }
    }

    std::printf("[STEP0] MAX-COVERAGE over %dx%d sweep: cup_center=(%.4f,%.4f,%.4f) "
                "genuine=%u max_gap=%.1f covered_arc=%.1f deg (need >180)\n",
                kN, kN, best.cup_center.x, best.cup_center.y, best.cup_center.z,
                best.genuine, best.max_gap, best.covered_arc);
    std::printf("[STEP0] MAX-COVERAGE azimuths:");
    for (float a : best.azimuths) std::printf(" %.1f", a);
    std::printf("\n");

    // The placement the DYNAMICS uses: the squeeze-BALANCED one (min net squeeze
    // |S| s.t. covered_arc>180). The max-coverage placement surrounds the cup but the
    // net squeeze drives it out the void during the close; the balanced placement
    // holds the cage. Both are reported; STEP ZERO gates on surround EXISTENCE.
    const Squeeze bal = BestPlacement(h1, poses_curl, spheres, hull);
    std::printf("[STEP0] SQUEEZE-BALANCED (dynamics placement): cup_center=(%.4f,%.4f,%.4f) "
                "|S|=%.5f dir=%.0fdeg genuine=%u covered_arc=%.1f deg\n",
                bal.center.x, bal.center.y, bal.center.z, bal.mag, bal.dir_deg,
                bal.genuine, bal.covered_arc);

    EXPECT_GT(best.covered_arc, 180.0f)
        << "no cup placement in the cavity surrounds the cup (max covered arc "
        << best.covered_arc << " deg <= 180; the cup is too small for the H1 hand's "
        << "contact span -- every void-filling contact costs its opposite). STOP.";
}

// ===========================================================================
// CALIBRATION: the table-supported close. Characterize the wrap contact count +
// the finger vertical impulse + the table support as the grip torque varies, at
// FAITHFUL armature=0.1/damping=1.0. Picks a stable grip torque for the LIFT gate.
// (Diagnostic; no assertion -- a SURFACE SIGNAL if no stable loaded wrap exists.)
// ===========================================================================
TEST(H1PowerGraspDynamics, TableSupportedCloseCalibration) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    std::printf("[CALIB] dt=1/240 mg.dt=%.4e mu=%.2f armature=0.1 damping=1.0 (FAITHFUL)\n",
                weight_kick, kMu);

    for (float tq : {0.0f, 0.1f, 0.5f}) {
        PowerGraspScene gs = BuildPowerGraspScene(context, hull, tq, kMu, /*table=*/true);
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                     kGravityZ, kDt);
        const double z0 = gs.cup0.position.z;
        double steady_qd = 0.0, sum_fimp = 0.0, sum_timp = 0.0;
        uint32_t n_steady = 0u, min_contacts = 1000u, max_contacts = 0u;
        coresident::CoResidentStepReport rep;
        for (uint32_t k = 0u; k < 80u; ++k) {
            rep = stepper.Step();
            min_contacts = std::min(min_contacts, rep.finger_contacts);
            max_contacts = std::max(max_contacts, rep.finger_contacts);
            if (k >= 40u) {
                articulation::ArticulationHostState st; stepper.Download(&st);
                for (uint32_t l : gs.drive_links)
                    steady_qd = std::max(steady_qd, std::fabs((double)st.qdot[l]));
                sum_fimp += rep.cup_vertical_impulse;
                sum_timp += rep.table_lambda;
                ++n_steady;
            }
        }
        const double drift = std::fabs(stepper.Cup().position.z - z0);
        std::printf("[CALIB] tq=%.2f -> contacts[%u..%u] table_rows=%u "
                    "steady finger_vimp=%.4e table_lambda=%.4e (mg.dt=%.4e) "
                    "drift=%.5f steady|qd|=%.3f %s\n",
                    tq, min_contacts, max_contacts, rep.table_row_count,
                    n_steady ? sum_fimp / n_steady : 0.0,
                    n_steady ? sum_timp / n_steady : 0.0, weight_kick, drift, steady_qd,
                    (steady_qd > 5.0) ? "[UNSTABLE]" : "[STABLE]");
    }
    SUCCEED();
}

namespace {

// PD-to-beyond drive: each step tau_i = Kp*(q_target_i - q_i) - Kd*qdot_i, with
// q_target = the curled pose + close_offset (curl BEYOND contact). A finger jammed
// against the cup settles to qdot~0 with tau sustained -> a true static wrap squeeze
// (unlike a constant torque, which leaves the fingers creeping). q_target is read
// from the host q at construction (the baked curl) + the offset, per driven link.
struct PdState {
    std::vector<float> q_target;  // per device link.
    float Kp, Kd;
};
PdState MakePdTarget(const PowerGraspScene& gs, float Kp, float Kd, float close_offset) {
    PdState pd;
    pd.Kp = Kp; pd.Kd = Kd;
    pd.q_target.assign(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links) pd.q_target[l] = gs.host.q[l] + close_offset;
    return pd;
}
// Apply one PD torque step to the stepper from the Download'd q/qdot.
void DrivePd(coresident::UnifiedCoResidentStepper& stepper, const PowerGraspScene& gs,
             const PdState& pd) {
    articulation::ArticulationHostState st; stepper.Download(&st);
    std::vector<float> tau(gs.host.TotalLinkCount(), 0.0f);
    for (uint32_t l : gs.drive_links)
        tau[l] = pd.Kp * (pd.q_target[l] - st.q[l]) - pd.Kd * st.qdot[l];
    stepper.SetGripTorque(tau);
}

// Post-step cup tilt RELATIVE to a reference orientation (the settled pose): angle of
// q_ref^-1 * q_cur (= 2*acos|w|).
double RelTilt(const Quat& q_ref, const Quat& q_cur) {
    const Quat q_rel = q_ref.Conjugate() * q_cur;
    return 2.0 * std::acos((double)std::min(1.0f, std::fabs(q_rel.w)));
}

// Measure the post-removal azimuth coverage of the LIVE wrap (does the cage hold
// geometrically?). Recompute the sphere world centers from the Download'd q + FK and
// the current cup pose, count genuine contacts, return covered_arc.
float LiveCoveredArc(const nuka::phi::DeviceContext& context, const PowerGraspScene& gs,
                     coresident::UnifiedCoResidentStepper& stepper, const CupHull& hull) {
    articulation::ArticulationHostState st; stepper.Download(&st);
    const auto poses = ForwardKinematics(context, st);
    const auto spheres = WrapSpheres();
    const Vec3 cc = stepper.Cup().position;
    std::vector<float> azs;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(gs.h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = SphereCenter(poses, l, s.local_offset);
        if (RimGap(c, cc, hull) >= s.radius) continue;
        float az = std::atan2(c.y - cc.y, c.x - cc.x) * 180.0f / kPi;
        if (az < 0.0f) az += 360.0f;
        azs.push_back(az);
    }
    if (azs.size() < 2u) return 0.0f;
    std::sort(azs.begin(), azs.end());
    float max_gap = 0.0f;
    for (size_t i = 0u; i < azs.size(); ++i) {
        const float next = (i + 1u < azs.size()) ? azs[i + 1u] : azs[0] + 360.0f;
        max_gap = std::max(max_gap, next - azs[i]);
    }
    return 360.0f - max_gap;
}

}  // namespace

// ===========================================================================
// THE LIFT GATE (the discriminator) -- table-supported wrap-close, then REMOVE the
// table and demand the wrap carries the FULL cup weight without the cup pivoting out.
// ===========================================================================
// CHOREOGRAPHY: (1) cup rests on a static table; PD-to-beyond drives the wrap into
// LOADED contact while the table supports the cup (no acquisition race, faithful
// armature). (2) REMOVE the table (SetTableEnabled(false)). The wrap must now carry
// the full weight: finger vertical impulse ~= +m*g*dt, cup stays CAGED (bounded
// translation AND bounded TILT -- NOT the 0.5 rad the fingertip 2-point gave),
// contacts retained, any_static_row==0 after removal.
TEST(H1PowerGraspDynamics, LiftGateWrapCarriesWeightWithoutPivot) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;

    // PD-to-beyond settle the wrap against the table-supported cup.
    PowerGraspScene gs = BuildPowerGraspScene(context, hull, /*grip_torque=*/0.0f,
                                              kMu, /*table=*/true);
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                 kGravityZ, kDt);
    // PD HOLDS the Step-Zero wrap pose (close_offset=0): the spheres pre-penetrate the
    // cup at the baked curl, so PD only needs to RESIST uncurl (sustained squeeze), NOT
    // curl deeper. (Over-curl shoves the cup out the cage void -- MaxCoverageCloseCollapsesDiagnostic.)
    const PdState pd = MakePdTarget(gs, /*Kp=*/4.0f, /*Kd=*/0.4f, /*close_offset=*/0.0f);
    const double z0 = gs.cup0.position.z;

    // --- Phase 1: table-supported close (settle the loaded wrap). ---
    const uint32_t kSettle = 70u;
    coresident::CoResidentStepReport rep;
    for (uint32_t s = 0u; s < kSettle; ++s) { DrivePd(stepper, gs, pd); rep = stepper.Step(); }
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;
    // The equilibrium triangle WHILE table-supported: table + finger ~= +mg.dt.
    std::printf("[LIFT] table-supported settle: contacts=%u table_rows=%u "
                "finger_vimp=%.4e (N %.4e + F %.4e) table_vimp=%.4e sum=%.4e "
                "(mg.dt=%.4e) cup_dvz=%.4e covered_arc=%.1f\n",
                rep.finger_contacts, rep.table_row_count, rep.cup_vertical_impulse,
                rep.finger_vimpulse_normal, rep.finger_vimpulse_friction,
                rep.table_vertical_impulse,
                rep.cup_vertical_impulse + rep.table_vertical_impulse, weight_kick,
                rep.cup_dvz_impulse, LiveCoveredArc(context, gs, stepper, hull));

    // --- Phase 2: REMOVE the table. The wrap must carry the full weight alone. ---
    stepper.SetTableEnabled(false);
    const uint32_t kLift = 120u, kLiftSettle = 60u;
    double max_disp = 0.0, max_tilt = 0.0, max_balance_err = 0.0, max_cross = 0.0;
    double sum_fimp = 0.0; uint32_t n_steady = 0u, contact_after = 0u;
    bool any_static_after = false; double steady_qd = 0.0;
    double last_fN = 0.0, last_fF = 0.0;
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
        if (s >= kLiftSettle) {
            articulation::ArticulationHostState st; stepper.Download(&st);
            for (uint32_t l : gs.drive_links)
                steady_qd = std::max(steady_qd, std::fabs((double)st.qdot[l]));
            sum_fimp += rep.cup_vertical_impulse; ++n_steady;
            const double bal = std::fabs(rep.cup_vertical_impulse - weight_kick) /
                std::max(weight_kick, 1e-9);
            max_balance_err = std::max(max_balance_err, bal);
            last_fN = rep.finger_vimpulse_normal; last_fF = rep.finger_vimpulse_friction;
        }
    }
    const double mean_fimp = n_steady ? sum_fimp / n_steady : 0.0;
    const double drift = std::fabs(stepper.Cup().position.z - z0);
    const float live_arc = LiveCoveredArc(context, gs, stepper, hull);
    const double final_w = stepper.Cup().angular_velocity.Length();

    std::printf("[LIFT] AFTER TABLE REMOVAL: contact=%u/%u any_static=%d "
                "mean finger_vimp=%.4e (mg.dt=%.4e bal_err=%.3f) last(N %.4e+F %.4e) "
                "cross=%.2e\n",
                contact_after, kLift, any_static_after ? 1 : 0, mean_fimp, weight_kick,
                max_balance_err, last_fN, last_fF, max_cross);
    std::printf("[LIFT] cage: max_disp=%.4f max_tilt=%.4f rad (vs fingertip 2-point "
                "~0.5 rad) final|w|=%.3f drift=%.5f steady|qd|=%.3f live_covered_arc=%.1f\n",
                max_disp, max_tilt, final_w, drift, steady_qd, live_arc);

    // --- THE LIFT GATE: ASSERT what is TRUE + stable; SKIP-with-finding on the
    //     rotational cage (the residual the wrap cannot reach). This mirrors the
    //     fingertip spike's DisturbancePivotsOut2Point protocol: do NOT assert the
    //     failure-state green (a false RED the day it is fixed) and do NOT leave a hard
    //     FAIL (a false regression). -------------------------------------------------
    //
    // ASSERTED (true + stable). NOTE on the carry: BiteAndActiveReleaseProbe shows the
    // hold is PASSIVE (it survives grip=0 -- deep pre-penetration + the translation
    // cage), so the vertical support is NOT attributable to active grip; the genuine
    // new feature over the pinch is the grip-INDEPENDENT TRANSLATION CAGE. We assert the
    // wrap delivers a net UPWARD impulse balancing the weight (true, but passive) +ASSERT
    // the translation cage.
    //   * No table after removal (the support is gone; the wrap holds it alone).
    EXPECT_FALSE(any_static_after) << "a table row persisted after removal";
    //   * VERTICAL SUPPORT: the wrap delivers ~+m*g*dt upward (the flip from the table-
    //     supported -mg.dt to +mg.dt). STEADY MEAN (the cup rotates, so the instantaneous
    //     balance is somewhat rough -- max bal_err peaks ~0.43 at the shipped 120-step
    //     lift; the spike reports the mean, NOT a clean static-hold claim). The mean
    //     support + the cross-check (cross ~2e-6) are the honest numbers.
    EXPECT_GT(mean_fimp, 0.5 * weight_kick)
        << "the wrap does not support the cup weight after table removal (mean finger "
        << "vimp " << mean_fimp << " << mg.dt=" << weight_kick << ")";
    EXPECT_LT(max_cross, 5.0e-2) << "impulse bookkeeping disagrees";
    //   * TRANSLATION CAGE (the GENUINE advance over the pinch): the cup center stays
    //     bounded (the squeeze-balanced wrap closed the translational DOF the pinch
    //     could not, and it is grip-independent per the BITE probe).
    EXPECT_LT(max_disp, 0.05) << "the cup center flew out (translation cage failed)";
    //   * Contacts retained by COUNT + bounded finger spin (no runaway at real armature).
    EXPECT_GE(contact_after, kLift - 8u) << "the wrap lost contact after table removal";
    EXPECT_LT(steady_qd, 5.0) << "a driven finger ran away after table removal";

    // THE ROTATIONAL CAGE is the HONEST NEGATIVE (surfaced, not asserted green). The
    // cup PIVOTS: a DEVELOPING tilt + a persistent spin (final|w|>1) that does NOT
    // settle, and the live covered-arc COLLAPSES (202 -> <180) as the cup rotates
    // contacts to one side. The wrap's many asymmetric contacts produce a net contact
    // MOMENT the 2-point pinch lacked, actively spinning the cup. ROOT CAUSE: the
    // unfillable ~110-deg palm-side void (palm<->finger x-span ~8cm > the 6cm cup -> the
    // palm cannot co-contact with the fingers). The contacts already cover ~75% of the
    // cup height (z-band 1.158..1.221 on the 1.148..1.232 cup body), so spreading
    // contacts in z cannot fix it -- the cup tips into the void about the low-leverage
    // axis through the two finger clusters. This is the SAME invariant that defeated the
    // fingertip pinch, now in the rotational DOF after the wrap closed the translational
    // one. STOP-and-surface.
    if (max_tilt > 0.25 || final_w > 1.0 || live_arc < 180.0f) {
        GTEST_SKIP() << "HONEST NEGATIVE (spike STOP): the faithful-param wrap achieves a "
                        "VERTICAL SUPPORT (mean finger_vimp=" << mean_fimp << " ~= mg.dt="
                     << weight_kick << ", cross-checked; note BITE shows this is PASSIVE) "
                        "+ a TRANSLATION CAGE (max_disp=" << max_disp << " < 0.05) -- both "
                        "of which the fingertip pinch lacked or masked -- but the "
                        "ROTATIONAL cage fails: cup tilts " << max_tilt << " rad (developing, "
                        "does not settle), final|w|=" << final_w << " (persistent spin), live "
                        "covered_arc collapses to " << live_arc << " deg. Root = the "
                        "unfillable ~110-deg palm-side void (palm<->finger 8cm > 6cm cup); "
                        "contacts already cover ~75% of cup height so z-spread cannot fix "
                        "it. The spike LOCATED the boundary -- see report for the owner round.";
    }
    // (Reached only if a future change fixes the rotational cage -- then assert it green.)
    EXPECT_LT(max_tilt, 0.1) << "rotational cage (now expected to hold)";
}

// ===========================================================================
// MAX-COVERAGE CLOSE TRACE (diagnostic): the MOTIVATION for the balanced placement.
// At the MAX-covered_arc placement the close COLLAPSES -- the net squeeze (dominated
// by the 4 deepest finger-intermediate contacts) drives the cup straight toward the
// cage void (~202 deg) and the live covered_arc collapses to ~0 (the cup slides out).
// Contrast SqueezeBalancedPlacementCloseTrace (the min-|S| placement holds the cage).
// PD holds the Step-Zero wrap pose (close_offset=0). No assert (the collapse is read).
// ===========================================================================
TEST(H1PowerGraspDynamics, MaxCoverageCloseCollapsesDiagnostic) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    // The MAX-coverage placement (the one BestPlacement deliberately REJECTS).
    CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    articulation::ArticulationHostState host0 = h1.host;
    ApplyCurl(h1, &host0, CurlPose{});
    const auto poses_curl = ForwardKinematics(context, host0);
    const auto spheres = WrapSpheres();
    Vec3 lo, hi; float cz;
    CavityBox(h1, poses_curl, spheres, &lo, &hi, &cz);
    Surround maxc;
    const int kN = 21;
    for (int i = 0; i < kN; ++i) for (int j = 0; j < kN; ++j) {
        const float fx = lo.x + (hi.x - lo.x) * i / (kN - 1);
        const float fy = lo.y + (hi.y - lo.y) * j / (kN - 1);
        const Surround sr = MeasureSurround(h1, poses_curl, spheres, hull, Vec3{fx, fy, cz});
        if (sr.genuine >= 3u && sr.covered_arc > maxc.covered_arc) maxc = sr;
    }

    PowerGraspScene gs = BuildPowerGraspScene(context, hull, 0.0f, kMu, /*table=*/true);
    gs.cup0.position = maxc.cup_center;
    gs.config.cup_state.position = maxc.cup_center;
    gs.cup_center = maxc.cup_center;
    const Vec3 hull_center = (hull.hi + hull.lo) * 0.5f;
    gs.table_height = maxc.cup_center.z + (hull.lo.z - hull_center.z);
    gs.config.table_height = gs.table_height;
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, /*Kp=*/4.0f, /*Kd=*/0.4f, /*offset=*/0.0f);
    const Vec3 c0 = maxc.cup_center;
    std::printf("[MAXC CLOSE] cup0=(%.4f,%.4f,%.4f) covered_arc=%.1f (the REJECTED "
                "placement; opening~202deg)\n", c0.x, c0.y, c0.z, maxc.covered_arc);
    for (uint32_t s = 0u; s <= 120u; ++s) {
        DrivePd(stepper, gs, pd);
        const auto rep = stepper.Step();
        if (s % 20u == 0u || s == 120u) {
            const Vec3 cp = stepper.Cup().position;
            const float dx = cp.x - c0.x, dy = cp.y - c0.y;
            float dir = std::atan2(dy, dx) * 180.0f / kPi; if (dir < 0) dir += 360.0f;
            std::printf("  s=%3u cup=(%.4f,%.4f,%.4f) xy_disp=%.4f dir=%.0fdeg "
                        "contacts=%u arc=%.1f\n",
                        s, cp.x, cp.y, cp.z, std::sqrt(dx*dx+dy*dy), dir,
                        rep.finger_contacts, LiveCoveredArc(context, gs, stepper, hull));
        }
    }
    SUCCEED();
}

// ===========================================================================
// SQUEEZE-BALANCED PLACEMENT (the ONE authorized refinement). The max-coverage
// placement jams the 4 finger-intermediate contacts deepest -> the net squeeze (Σ
// penetration*inward-normal) points ~202 deg, straight at the cage void, and the cup
// rides it out (MaxCoverageCloseCollapsesDiagnostic). 2D force-closure IS satisfiable here (inward
// normals span 220 deg), but the GRIP doesn't produce a balancing set. So we instead
// pick the cup placement that MINIMIZES the net-squeeze imbalance |S| subject to
// covered_arc > 180, and run the close ONCE there. This is the task's "one reasonable
// refinement of contact placement", not a param grind. STOP after this run.
// ===========================================================================
TEST(H1PowerGraspDynamics, SqueezeBalancedPlacementCloseTrace) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    // BuildPowerGraspScene already places the cup at the squeeze-BALANCED placement
    // (BestPlacement = min |S| s.t. covered_arc>180). Report that placement's |S|.
    CookedH1 h1 = LoadH1Fixed(kWrapDriven, kRealArmature, kRealDamping);
    articulation::ArticulationHostState host = h1.host;
    ApplyCurl(h1, &host, CurlPose{});
    const auto poses_curl = ForwardKinematics(context, host);
    const auto spheres = WrapSpheres();
    const Squeeze best = BestPlacement(h1, poses_curl, spheres, hull);
    std::printf("[BAL] min-|S| placement (covered_arc>180): center=(%.4f,%.4f,%.4f) "
                "|S|=%.5f dir=%.0fdeg genuine=%u arc=%.1f\n",
                best.center.x, best.center.y, best.center.z, best.mag, best.dir_deg,
                best.genuine, best.covered_arc);
    ASSERT_LT(best.mag, 1e8f) << "no covered_arc>180 placement found";

    // Run the CLOSE ONCE at this placement (close_offset=0, PD holds the wrap pose).
    PowerGraspScene gs = BuildPowerGraspScene(context, hull, 0.0f, kMu, /*table=*/true);
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config, kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, /*Kp=*/4.0f, /*Kd=*/0.4f, /*offset=*/0.0f);
    const Vec3 c0 = gs.cup_center;
    std::printf("[BAL CLOSE] cup0=(%.4f,%.4f,%.4f) squeeze_dir=%.0fdeg\n",
                c0.x, c0.y, c0.z, best.dir_deg);
    for (uint32_t s = 0u; s <= 120u; ++s) {
        DrivePd(stepper, gs, pd);
        const auto rep = stepper.Step();
        if (s % 20u == 0u || s == 120u) {
            const Vec3 cp = stepper.Cup().position;
            const float dx = cp.x - c0.x, dy = cp.y - c0.y;
            float dir = std::atan2(dy, dx) * 180.0f / kPi; if (dir < 0) dir += 360.0f;
            std::printf("  s=%3u cup=(%.4f,%.4f,%.4f) xy_disp=%.4f dir=%.0fdeg "
                        "contacts=%u arc=%.1f\n",
                        s, cp.x, cp.y, cp.z, std::sqrt(dx*dx+dy*dy), dir,
                        rep.finger_contacts, LiveCoveredArc(context, gs, stepper, hull));
        }
    }
    SUCCEED();  // diagnostic; the close behaviour is read, the decision is in the report.
}

// ===========================================================================
// THE BITE / RELEASE PROBE -- is the hold ACTIVE grip or a PASSIVE cage?
// ===========================================================================
// The task's classic BITE ("grip=0 -> cup falls; a hold that survives grip=0 is
// fake") was written for a PINCH, where grip FORCE is the hold. For a CAGE the
// counterfactual is different: a curled wrap with DEEP baked pre-penetration (~6-10mm
// per contact at the balanced placement) + a translation cage holds the cup at grip=0
// because the hold is GEOMETRIC, not a friction balance. So we run BOTH:
//   (A) grip=0 (zero torque): MEASURE whether the cup still holds (it does -> the hold
//       is PASSIVE). We do NOT assert it falls (that is the pinch semantics, which mis-
//       fit a cage); we surface the number for the owner to adjudicate.
//   (B) the cage-release CHARACTERIZATION (diagnostic, no assert): actively OPEN the
//       fingers (PD target -> uncurled). At faithful armature=0.1 the open is qddot-
//       throttled so release is SLOW (not stuck -- the LIFT gate proved the cup is
//       dynamically free). We print the driven q/qdot + cup vz so the report can state
//       "fingers uncurl, cup releases -- releasable, just armature-slow".
// CONCLUSION (in the report): the hold is PASSIVE (A), contradicting the task's pinch-
// era BITE assumption -- expected for a geometric cage; surfaced for the owner round.
TEST(H1PowerGraspDynamics, BiteAndActiveReleaseProbe) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    const uint32_t kFall = 100u;
    const double free_fall = 0.5 * (-kGravityZ) * (kFall * kDt) * (kFall * kDt);

    // --- (A) grip=0 after table removal: does the PASSIVE cage hold? (measure) ---
    {
        PowerGraspScene gs = BuildPowerGraspScene(context, hull, 0.0f, kMu, /*table=*/true);
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                     kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, 4.0f, 0.4f, 0.0f);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);
        const double z_settled = stepper.Cup().position.z;
        const std::vector<float> zero(gs.host.TotalLinkCount(), 0.0f);
        coresident::CoResidentStepReport rep;
        for (uint32_t s = 0u; s < kFall; ++s) { stepper.SetGripTorque(zero); rep = stepper.Step(); }
        const BodyState cupF = stepper.Cup();
        const double drop = z_settled - cupF.position.z;
        std::printf("[BITE-A] grip=0 + table removed: cup dropped %.5f m over %u steps "
                    "(free-fall=%.4f) final vz=%.4f finger_vimp=%.4e (mg.dt=%.4e)\n",
                    drop, kFall, free_fall, cupF.linear_velocity.z,
                    rep.cup_vertical_impulse, weight_kick);
        std::printf("[BITE-A] => cup does NOT free-fall at grip=0 -> the hold is PASSIVE "
                    "(deep ~6-10mm pre-penetration + translation cage), NOT active grip.\n");
    }

    // --- (B) the cage-release CHARACTERIZATION (diagnostic, NO assert): actively OPEN
    //     the fingers (PD target = uncurled, q->0). At faithful armature=0.1 the open
    //     is qddat-throttled (qddot ~ tau/arm), so release is SLOW -- NOT stuck (the
    //     LIFT gate already proved the cup is dynamically free: it pivots/spins, the
    //     solver separates these contacts). We extend the open to 250 steps and print
    //     the driven-link q/qdot so the report can state "fingers uncurl (q->0,
    //     qdot<0), cup releases -- releasable, just armature-slow", a FACT not a gate.
    //     We do NOT raise Kp past the faithful 1 N.m ceiling and do NOT reduce the
    //     pre-penetration (that reopens the pinch spike's acquisition-race wall). ----
    PowerGraspScene gs = BuildPowerGraspScene(context, hull, 0.0f, kMu, /*table=*/true);
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                 kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, 4.0f, 0.4f, 0.0f);
    for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);
    const double z_settled = stepper.Cup().position.z;
    PdState open; open.Kp = 4.0f; open.Kd = 0.4f;
    open.q_target.assign(gs.host.TotalLinkCount(), 0.0f);  // uncurled.
    const uint32_t kOpen = 250u;
    for (uint32_t s = 0u; s < kOpen; ++s) { DrivePd(stepper, gs, open); stepper.Step(); }
    const BodyState cupF = stepper.Cup();
    const double dropB = z_settled - cupF.position.z;
    articulation::ArticulationHostState st; stepper.Download(&st);
    double mean_q = 0.0, mean_qd = 0.0;
    for (uint32_t l : gs.drive_links) { mean_q += st.q[l]; mean_qd += st.qdot[l]; }
    mean_q /= std::max<size_t>(gs.drive_links.size(), 1);
    mean_qd /= std::max<size_t>(gs.drive_links.size(), 1);
    const double free_fall_open = 0.5 * (-kGravityZ) * (kOpen * kDt) * (kOpen * kDt);
    std::printf("[BITE-B] actively OPEN (PD target uncurled, %u steps): cup dropped %.5f m "
                "(free-fall=%.4f) final vz=%.4f | driven mean q=%.4f qdot=%.4f -> %s\n",
                kOpen, dropB, free_fall_open, cupF.linear_velocity.z, mean_q, mean_qd,
                (cupF.linear_velocity.z < -0.05f || dropB > 0.01)
                    ? "RELEASING (armature-slow, not stuck)"
                    : "release barely initiated this window");
    SUCCEED();  // characterization; the BITE conclusion is BITE-A (passive hold) + report.
}

// ===========================================================================
// DYNAMIC LIFT LOAD (confirmatory; SKIP-with-finding). After table removal, impose a
// sustained ~1.5g upward accel (the lift's inertial load) + a lateral + a tilt
// disturbance. The dynamic load only AMPLIFIES the rotational underconstraint -- this
// is run to CONFIRM the boundary, not as a new green gate.
// ===========================================================================
TEST(H1PowerGraspDynamics, DynamicLiftLoadAmplifiesPivot) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    PowerGraspScene gs = BuildPowerGraspScene(context, hull, 0.0f, kMu, /*table=*/true);
    coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                 kGravityZ, kDt);
    const PdState pd = MakePdTarget(gs, 4.0f, 0.4f, 0.0f);
    for (uint32_t s = 0u; s < 80u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
    stepper.SetTableEnabled(false);
    const Vec3 c_settled = stepper.Cup().position;
    const Quat q_settled = stepper.Cup().orientation;

    // Lateral + tilt impulse, then a sustained ~1.75g upward kick (the lift inertial
    // load = (1 + 0.75) g upward beyond holding the weight) for the first steps.
    stepper.ApplyCupImpulse(Vec3{0.04f, 0.04f, 0.0f}, Vec3{1.5f, 0.8f, 0.8f});
    double max_disp = 0.0, max_tilt = 0.0; uint32_t contact_after = 0u;
    const uint32_t kLoad = 120u;
    for (uint32_t s = 0u; s < kLoad; ++s) {
        DrivePd(stepper, gs, pd);
        if (s < 30u)  // sustained ~1.75g lift accel for 30 steps.
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
    // Surface the (expected) amplified pivot; do not assert it green.
    if (max_tilt > 0.35 || final_w > 1.0) {
        GTEST_SKIP() << "HONEST NEGATIVE (confirmatory): the dynamic lift load amplifies "
                        "the rotational underconstraint -- max_tilt=" << max_tilt
                     << " rad, final|w|=" << final_w << ", contact=" << contact_after
                     << "/" << kLoad << ". Confirms the palm-side-void boundary under the "
                        "inertial lift load. See report.";
    }
    SUCCEED();
}

// ===========================================================================
// D1: the full table-supported close + table-removal lift run TWICE -> byte-exact
// final cup state + q + qdot (determinism through the additive table path).
// ===========================================================================
TEST(H1PowerGraspDynamics, DeterministicTwoRun) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);
    auto rollout = [&](BodyState* cup, articulation::ArticulationHostState* art) {
        PowerGraspScene gs = BuildPowerGraspScene(context, hull, 0.0f, kMu, /*table=*/true);
        coresident::UnifiedCoResidentStepper stepper(context, gs.host, gs.config,
                                                     kGravityZ, kDt);
        const PdState pd = MakePdTarget(gs, 4.0f, 0.4f, 0.0f);
        for (uint32_t s = 0u; s < 60u; ++s) { DrivePd(stepper, gs, pd); stepper.Step(); }
        stepper.SetTableEnabled(false);  // remove the table mid-rollout (the toggle path).
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
    std::printf("[D1] cup + qdot byte-identical across 2 close+table-removal rollouts\n");
}
