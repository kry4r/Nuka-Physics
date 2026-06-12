#pragma once
// ---------------------------------------------------------------------------
// v0.8 C7b-2 SHOWCASE -- SHARED H1-cup demo helpers (the reusable extraction).
// ---------------------------------------------------------------------------
// This header EXTRACTS the proven helper bodies that both the cup-sequence demo
// test and future demo/RT-render code need: the H1 floating-base cook, the leg
// resolution + obs builder + torque limits, the forward-kinematics / CoM / tilt
// probes, the standing stance + foot constants, and the cup convex-hull loader.
//
// EVERY function body here is COPIED VERBATIM from the already-proven sources --
//   tests/coresident/test_h1_bridge_spike.cpp   (the standing setup, Gate-2)
//   tests/coresident/test_h1_dense_grasp.cpp    (the cup hull helpers)
// -- and is only re-homed (marked `inline`, placed in `namespace nuka::demo`).
// NOTHING here is new physics; it is the de-duplication of validated code so the
// demo TU and later showcase code share ONE copy. The original test files are
// LEFT UNTOUCHED (they keep their own anon-namespace copies).
//
// Scene-coupled helpers (the live-state obs assembler `Gate2Obs`, the scene
// builder) are NOT extracted here -- they depend on the test's scene struct and
// stay in the consuming TU.
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
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace nuka::demo {

namespace articulation = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

// ---- core constants (== bridge spike) -------------------------------------
constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 0.005f;

// The shared obs/action contract dimensions (== export_h1_tiny_actor_bridge.py).
constexpr uint32_t kObsDim = 32u;
constexpr uint32_t kActDim = 10u;

// poly center PINNED scalar (full-cook seat-time value; printed by costand SEAT).
constexpr float kPolyCx = 0.0770f;
constexpr float kPolyCy = 0.0f;

// The 10 contract leg JOINTS in the pinned order (cooked names append "_link").
inline const std::array<std::string, 10> kLegLinkNames = {
    "left_hip_yaw_link",  "left_hip_roll_link",  "left_hip_pitch_link",
    "left_knee_link",     "left_ankle_link",     "right_hip_yaw_link",
    "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link",
    "right_ankle_link"};

// ---- physical MJCF torque limits (N*m) per leg slot, contract order -------
inline float LegLimit(uint32_t slot) {
    // slots: 0 hip_yaw,1 roll,2 pitch,3 knee,4 ankle (x2 for right).
    const uint32_t j = slot % 5u;
    if (j == 3u) return 300.0f;  // knee.
    if (j == 4u) return 40.0f;   // ankle.
    return 200.0f;               // hip yaw/roll/pitch.
}

// ===========================================================================
// COOK helpers (mirror the bridge spike's LoadFloating, by name).
// ===========================================================================
struct Cooked {
    articulation::ArticulationHostState host;
    nuka::scene::SceneIR scene;
};

inline Cooked LoadFloating(const std::string& mjcf) {
    Cooked out;
    out.scene = nuka::import::LoadMjcf(mjcf);
    // Drop mesh geometry (dodge the V-HACD wall; we only need inertia + topology).
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    const auto blob = nuka::scene::CookScene(out.scene);
    auto topos = articulation::CookArticulations(blob);
    out.host = articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
}

inline std::string LinkName(const Cooked& c, uint32_t link) {
    if (c.host.link_body.size() <= link) return "?";
    const uint32_t body = c.host.link_body[link];
    if (body < c.scene.Bodies().size()) return c.scene.GetBody(body).name;
    return "?";
}
inline uint32_t LinkByName(const Cooked& c, const std::string& name) {
    for (uint32_t l = 0u; l < c.host.TotalLinkCount(); ++l)
        if (LinkName(c, l) == name) return l;
    return kInvalidLink;
}

// Resolve the 10 contract leg links (by name) into device link indices.
inline std::array<uint32_t, 10> ResolveLegLinks(const Cooked& c) {
    std::array<uint32_t, 10> out{};
    for (uint32_t s = 0u; s < 10u; ++s) out[s] = LinkByName(c, kLegLinkNames[s]);
    return out;
}

// ===========================================================================
// The SHARED obs-builder (== bridge spike BuildObs).
//   obs[0:4]   = quat (qw,qx,qy,qz)
//   obs[4:10]  = base spatial vel (omega-first) [wx,wy,wz, vx,vy,vz]
//   obs[10:12] = (base_x - poly_cx, base_y - poly_cy)
//   obs[12:22] = leg qpos in leg-index order
//   obs[22:32] = leg qvel in leg-index order
// ===========================================================================
inline std::vector<float> BuildObs(const std::array<float, 4>& quat,
                                   const std::array<float, 6>& spatial_vel,
                                   float base_x, float base_y,
                                   const std::array<float, 10>& leg_qpos,
                                   const std::array<float, 10>& leg_qvel,
                                   const std::array<uint32_t, 10>& leg_index) {
    std::vector<float> obs(kObsDim, 0.0f);
    for (uint32_t i = 0u; i < 4u; ++i) obs[i] = quat[i];
    for (uint32_t i = 0u; i < 6u; ++i) obs[4u + i] = spatial_vel[i];
    obs[10] = base_x - kPolyCx;
    obs[11] = base_y - kPolyCy;
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = leg_index[s];   // the leg-index map (identity normally).
        obs[12u + s] = leg_qpos[j];
        obs[22u + s] = leg_qvel[j];
    }
    return obs;
}

// The identity leg-index map (the correct one: reduced 1..10 == full by-name).
inline std::array<uint32_t, 10> IdentityLegIndex() {
    std::array<uint32_t, 10> m{};
    for (uint32_t s = 0u; s < 10u; ++s) m[s] = s;
    return m;
}

// ===========================================================================
// FK / CoM / tilt probes (== bridge spike Piece-C scaffold).
// ===========================================================================
inline std::vector<Transform> ForwardKinematics(
    const nuka::phi::DeviceContext& context,
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

struct CoMModel {
    std::vector<float> mass;
    std::vector<Vec3> com_local;
    double total = 0.0;
};
inline CoMModel BuildCoMModel(const articulation::ArticulationHostState& host) {
    CoMModel m;
    const uint32_t n = host.TotalLinkCount();
    m.mass.resize(n);
    m.com_local.resize(n);
    for (uint32_t l = 0u; l < n; ++l) {
        m.mass[l] = host.link_inertia[l].I[3u * 6u + 3u];
        m.com_local[l] = host.link_inertial_frame[l].position;
        m.total += m.mass[l];
    }
    return m;
}
inline Vec3 WholeBodyCoM(const CoMModel& m, const std::vector<Transform>& poses) {
    Vec3 acc{0, 0, 0};
    for (uint32_t l = 0u; l < poses.size(); ++l) {
        const Vec3 cw = poses[l].position + poses[l].rotation.Rotate(m.com_local[l]);
        acc += cw * m.mass[l];
    }
    return acc * static_cast<float>(1.0 / m.total);
}
inline double TiltDeg(const Quat& q) {
    const double gz = 1.0 - 2.0 * (static_cast<double>(q.x) * q.x +
                                   static_cast<double>(q.y) * q.y);
    return std::acos(std::max(-1.0, std::min(1.0, gz))) * 180.0 / M_PI;
}

// ===========================================================================
// Stance + foot constants (== bridge spike Piece-C scaffold).
// ===========================================================================
constexpr float kStanceHipPitch = -0.40f;
constexpr float kStanceKnee = 0.70f;
constexpr float kStanceAnkle = -0.30f;
constexpr float kKpHold = 50.0f, kKdHold = 4.0f;
constexpr float kFootSphereR = 0.025f;
constexpr float kFootBottomZ = -0.055f;
constexpr float kFootToeX = 0.10f;
constexpr float kFootHeelX = -0.10f;

// Hold-PD torque clamp per non-leg actuated DOF (== bridge spike HoldLimitFor).
inline float HoldLimitFor(const std::string& nm) {
    if (nm.find("torso") != std::string::npos) return 200.0f;
    if (nm.find("elbow") != std::string::npos) return 18.0f;
    if (nm.find("shoulder_yaw") != std::string::npos) return 18.0f;
    if (nm.find("shoulder") != std::string::npos) return 40.0f;
    return 1.0f;
}

// ===========================================================================
// Cup convex-hull loader + scale-about-center (== dense-grasp spike).
// ===========================================================================
struct CupHull {
    std::vector<float> verts;  // flat x,y,z, mesh-local.
    Vec3 lo{}, hi{};
    uint32_t vcount = 0u;
};
inline CupHull LoadCupHull(const std::string& cup_usda) {
    auto scene = nuka::import::LoadUsd(cup_usda);
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
inline CupHull ScaleCupHull(const CupHull& in, float sxy, float sz) {
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
// GRASP-POSE machinery (== test_h1_dense_grasp.cpp anon-namespace helpers).
// ---------------------------------------------------------------------------
// COPIED VERBATIM from tests/coresident/test_h1_dense_grasp.cpp (the proven dense
// wrap + the force-closure GATE3 placement search) and re-homed here (marked
// `inline`, namespace nuka::demo) so the R2+ reach/grasp demo rungs can place the
// cup where the curled right hand nestles it. NOTHING here is new physics; it is the
// de-duplication of the dense-grasp helpers. The dense-grasp test keeps its OWN
// anon-namespace copies (LEFT UNTOUCHED). These operate on FK world poses so they
// work on EITHER the Fixed-base dense cook or the floating-base demo cook.
// ===========================================================================
constexpr float kDemoPi = 3.14159265358979323846f;
constexpr float kWrapRadius = 0.006f;  // 6mm dense finger-segment pad (== dense spike).
constexpr float kPalmRadius = 0.010f;  // a slightly larger palm pad chain.
constexpr float kPalmOffsetX = 0.10f;  // == dense spike PALM_OFFSET_X.
constexpr float kVoidGapDeg = 110.0f;        // the palm-side void; "closed" = gap < this.
constexpr float kShallowPenMax = 0.002f;     // <=2mm == SHALLOW (NOT a deep wedge).
constexpr float kLargeCagingArcMin = 200.0f; // force-closure caging-arc floor.

struct WrapSphere {
    std::string body;
    Vec3 local_offset{};
    float radius = kWrapRadius;
    const char* region = "";
};

// The dense sphere-chain: 3 spheres along each contacting phalanx + a 3-sphere palm
// pad (== dense-grasp WrapSpheres, verbatim).
inline std::vector<WrapSphere> WrapSpheres(bool with_palm, const Vec3& palm_offset) {
    std::vector<WrapSphere> v;
    const char* fingers[] = {"R_index", "R_middle", "R_ring", "R_pinky"};
    const float finger_x[] = {0.006f, 0.016f, 0.026f};
    for (const char* f : fingers) {
        for (const char* seg : {"_proximal", "_intermediate"}) {
            for (float fx : finger_x)
                v.push_back({std::string(f) + seg, Vec3{fx, 0.0f, 0.0f}, kWrapRadius, "finger"});
        }
    }
    const float thumb_x[] = {0.004f, 0.012f, 0.020f};
    for (const char* seg : {"R_thumb_intermediate", "R_thumb_distal"}) {
        for (float tx : thumb_x)
            v.push_back({seg, Vec3{tx, 0.0f, 0.0f}, kWrapRadius, "thumb"});
    }
    if (with_palm) {
        for (float py : {-0.014f, 0.0f, 0.014f})
            v.push_back({"right_hand_link",
                         Vec3{palm_offset.x, palm_offset.y + py, palm_offset.z},
                         kPalmRadius, "palm"});
    }
    return v;
}

// The size-appropriate OPEN curl for the ~10cm cup (== dense-grasp CurlPose/CurlForScale).
struct CurlPose {
    float finger_prox = 1.0f, finger_int = 1.1f;
    float thumb_yaw = 1.0f, thumb_pitch = 0.5f, thumb_int = 0.6f, thumb_dist = 0.6f;
};
inline void ApplyCurl(const Cooked& h1, articulation::ArticulationHostState* host,
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
inline CurlPose CurlForScale(float sxy) {
    CurlPose c;
    const float relax = (sxy - 1.0f) * 0.55f;
    c.finger_prox = std::max(0.45f, 1.0f - relax);
    c.finger_int  = std::max(0.55f, 1.1f - relax);
    c.thumb_yaw = 1.0f; c.thumb_pitch = 0.5f;
    c.thumb_int  = std::max(0.35f, 0.6f - relax * 0.4f);
    c.thumb_dist = std::max(0.35f, 0.6f - relax * 0.4f);
    return c;
}

inline Vec3 SphereCenter(const std::vector<Transform>& poses, uint32_t link,
                         const Vec3& local_offset) {
    const Transform& lp = poses[link];
    return lp.position + lp.rotation.Rotate(local_offset);
}
// Surface-to-surface distance of a sphere center to the cup hull (== dense RimGap):
// nearest distance to a hull vertex (cup at world cup_center, hull mesh-local-centered).
inline float RimGap(const Vec3& sphere_center, const Vec3& cup_center, const CupHull& hull) {
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

// Coverage over the dense set (== dense-grasp Surround/MeasureSurround).
struct Surround {
    Vec3 cup_center{};
    uint32_t genuine = 0u;
    std::vector<float> azimuths;
    bool palm_contacts = false;
    float max_gap = 360.0f;
    float covered_arc = 0.0f;
};
inline Surround MeasureSurround(const Cooked& h1, const std::vector<Transform>& poses_curl,
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
        float az = std::atan2(dy, dx) * 180.0f / kDemoPi;
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
inline float MaxPrePenetration(const Cooked& h1, const std::vector<Transform>& poses_curl,
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
inline Vec3 CavityCenterZ(const Cooked& h1, const std::vector<Transform>& poses_curl,
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
inline float SqueezeMag(const Cooked& h1, const std::vector<Transform>& poses_curl,
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

struct Place {
    Vec3 center{}; uint32_t genuine = 0u; float covered_arc = 0.0f; float max_gap = 360.0f;
    bool palm = false; float max_pen = 0.0f; float squeeze_mag = 1e9f; bool found = false;
};

// The force-closure GATE3 placement search (== dense-grasp BestPlacementShallowNoPalmCaging):
// finds the cup center nestled in the curled hand with covered_arc>200deg + ALL contacts <=2mm
// (SHALLOW), minimizing the net squeeze imbalance. NO palm requirement (the validated
// FingerOnlyShallow opposed-wrap criterion). cup_center is in WORLD coords (FK is world).
inline Place BestPlacementShallowNoPalmCaging(const Cooked& h1,
                                              const std::vector<Transform>& poses_curl,
                                              const std::vector<WrapSphere>& spheres,
                                              const CupHull& hull) {
    Vec3 lo, hi;
    const Vec3 cz = CavityCenterZ(h1, poses_curl, spheres, &lo, &hi);
    Place best;
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

}  // namespace nuka::demo
