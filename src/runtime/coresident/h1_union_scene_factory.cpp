// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- H1 whole-body UNION scene factory (v0.8 G2).
//
// EVERY authoring body below is REPRODUCED VERBATIM (same float expressions,
// same call order, same constants) from the test-side proven sources:
//   tests/coresident/h1_demo_shared.hpp          (the cook / FK / CoM / cup-hull
//                                                 / wrap / placement helpers)
//   tests/coresident/h1_union_scene_shared.hpp   (the G1a-G1d union scene
//                                                 authoring + drive/CoP seams)
// promoted into libnuka so the C-ABI union world builds the SAME scene the 13
// nuka_batched_h1_union_world_test gates were proven on. src cannot include a
// tests/ header, so this is a copy -- guarded by the BYTE-EQUALITY gate in
// tests/coresident/test_h1_union_scene_factory.cpp (factory template ==
// shared-header template, field for field). NO new physics.
// ---------------------------------------------------------------------------

#include "runtime/coresident/h1_union_scene_factory.hpp"

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/canonical_types.hpp"  // DecomposeMode
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::runtime::coresident {

namespace {

namespace articulation = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = kH1UnionGravityZ;
constexpr float kDt = kH1UnionDt;

// The H1.2 / H1.1b LOCKED constants (== h1_union_scene_shared.hpp).
constexpr float kCupMass = kH1UnionCupMass;
constexpr float kMu = 0.8f;
constexpr float kSxy = 1.8f;
constexpr float kSz = 1.8f;
constexpr float kCupFarOffsetX = 5.0f;

// The H1.1/H1.2 proven grip-close PD + driven-finger joint properties.
constexpr float kGripKp = 4.0f;
constexpr float kGripKd = 0.4f;
constexpr float kCloseOffset = kH1UnionCloseOffset;
constexpr float kRealArmature = 0.1f;
constexpr float kRealDamping = 1.0f;

const char* const kWrapDriven[] = {
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
    "R_index_proximal",  "R_index_intermediate",
    "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal",   "R_ring_intermediate",
    "R_pinky_proximal",  "R_pinky_intermediate",
};

// ===========================================================================
// h1_demo_shared.hpp bodies (verbatim; namespace nuka::demo -> here).
// ===========================================================================
const std::array<std::string, 10> kLegLinkNames = {
    "left_hip_yaw_link",  "left_hip_roll_link",  "left_hip_pitch_link",
    "left_knee_link",     "left_ankle_link",     "right_hip_yaw_link",
    "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link",
    "right_ankle_link"};

float LegLimit(uint32_t slot) {
    const uint32_t j = slot % 5u;
    if (j == 3u) return 300.0f;  // knee.
    if (j == 4u) return 40.0f;   // ankle.
    return 200.0f;               // hip yaw/roll/pitch.
}

struct Cooked {
    articulation::ArticulationHostState host;
    nuka::scene::SceneIR scene;
};

Cooked LoadFloating(const std::string& mjcf) {
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

std::string LinkName(const Cooked& c, uint32_t link) {
    if (c.host.link_body.size() <= link) return "?";
    const uint32_t body = c.host.link_body[link];
    if (body < c.scene.Bodies().size()) return c.scene.GetBody(body).name;
    return "?";
}
uint32_t LinkByName(const Cooked& c, const std::string& name) {
    for (uint32_t l = 0u; l < c.host.TotalLinkCount(); ++l)
        if (LinkName(c, l) == name) return l;
    return kInvalidLink;
}

std::array<uint32_t, 10> ResolveLegLinks(const Cooked& c) {
    std::array<uint32_t, 10> out{};
    for (uint32_t s = 0u; s < 10u; ++s) out[s] = LinkByName(c, kLegLinkNames[s]);
    return out;
}

std::vector<Transform> ForwardKinematics(
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
CoMModel BuildCoMModel(const articulation::ArticulationHostState& host) {
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
Vec3 WholeBodyCoM(const CoMModel& m, const std::vector<Transform>& poses) {
    Vec3 acc{0, 0, 0};
    for (uint32_t l = 0u; l < poses.size(); ++l) {
        const Vec3 cw = poses[l].position + poses[l].rotation.Rotate(m.com_local[l]);
        acc += cw * m.mass[l];
    }
    return acc * static_cast<float>(1.0 / m.total);
}

// Stance + foot constants (== bridge spike Piece-C scaffold).
constexpr float kStanceHipPitch = -0.40f;
constexpr float kStanceKnee = 0.70f;
constexpr float kStanceAnkle = -0.30f;
constexpr float kKpHold = 50.0f, kKdHold = 4.0f;
constexpr float kFootSphereR = 0.025f;
constexpr float kFootBottomZ = -0.055f;
constexpr float kFootToeX = 0.10f;
constexpr float kFootHeelX = -0.10f;

float HoldLimitFor(const std::string& nm) {
    if (nm.find("torso") != std::string::npos) return 200.0f;
    if (nm.find("elbow") != std::string::npos) return 18.0f;
    if (nm.find("shoulder_yaw") != std::string::npos) return 18.0f;
    if (nm.find("shoulder") != std::string::npos) return 40.0f;
    return 1.0f;
}

struct CupHull {
    std::vector<float> verts;  // flat x,y,z, mesh-local.
    Vec3 lo{}, hi{};
    uint32_t vcount = 0u;
};
CupHull LoadCupHull(const std::string& cup_usda) {
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

// GRASP-POSE machinery (== test_h1_dense_grasp.cpp via h1_demo_shared.hpp).
constexpr float kDemoPi = 3.14159265358979323846f;
constexpr float kWrapRadius = 0.006f;
constexpr float kPalmRadius = 0.010f;
constexpr float kShallowPenMax = 0.002f;
constexpr float kLargeCagingArcMin = 200.0f;

struct WrapSphere {
    std::string body;
    Vec3 local_offset{};
    float radius = kWrapRadius;
    const char* region = "";
};

std::vector<WrapSphere> WrapSpheres(bool with_palm, const Vec3& palm_offset) {
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

struct CurlPose {
    float finger_prox = 1.0f, finger_int = 1.1f;
    float thumb_yaw = 1.0f, thumb_pitch = 0.5f, thumb_int = 0.6f, thumb_dist = 0.6f;
};
void ApplyCurl(const Cooked& h1, articulation::ArticulationHostState* host,
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

Vec3 SphereCenter(const std::vector<Transform>& poses, uint32_t link,
                  const Vec3& local_offset) {
    const Transform& lp = poses[link];
    return lp.position + lp.rotation.Rotate(local_offset);
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

struct Surround {
    Vec3 cup_center{};
    uint32_t genuine = 0u;
    std::vector<float> azimuths;
    bool palm_contacts = false;
    float max_gap = 360.0f;
    float covered_arc = 0.0f;
};
Surround MeasureSurround(const Cooked& h1, const std::vector<Transform>& poses_curl,
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
float MaxPrePenetration(const Cooked& h1, const std::vector<Transform>& poses_curl,
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
Vec3 CavityCenterZ(const Cooked& h1, const std::vector<Transform>& poses_curl,
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
float SqueezeMag(const Cooked& h1, const std::vector<Transform>& poses_curl,
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

Place BestPlacementShallowNoPalmCaging(const Cooked& h1,
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

// ===========================================================================
// h1_union_scene_shared.hpp bodies (verbatim).
// ===========================================================================
Cooked LoadFloatingGraspCook(const std::string& mjcf) {
    Cooked out;
    out.scene = nuka::import::LoadMjcf(mjcf);
    // Drop mesh geometry (== LoadFloating: dodge V-HACD; inertia + topology only).
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    const auto blob = nuka::scene::CookScene(out.scene);
    auto topos = nuka::runtime::articulation::CookArticulations(blob);
    // Resolve the 12 wrap-driven finger BODIES by name (== LoadH1Fixed's seam).
    std::vector<uint32_t> driven_bodies;
    for (const char* nm : kWrapDriven) {
        for (uint32_t b = 0u; b < out.scene.RigidBodyCount(); ++b) {
            if (out.scene.GetBody(b).name == nm) {
                driven_bodies.push_back(b);
                break;
            }
        }
    }
    auto is_driven = [&](uint32_t body) {
        return std::find(driven_bodies.begin(), driven_bodies.end(), body) !=
               driven_bodies.end();
    };
    for (auto& topo : topos) {
        for (size_t i = 0u; i < topo.link_bodies.size(); ++i) {
            if (!is_driven(topo.link_bodies[i])) continue;
            topo.joint_dampings[i] = kRealDamping;
            topo.joint_armatures[i] = kRealArmature;
        }
    }
    out.host = nuka::runtime::articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
}

struct UnionScene {
    Cooked h1;
    articulation::ArticulationHostState host;  // the floating proto (zeroed vels).
    GraspConfig config;                        // oracle input (cup FAR -> no contact).
    BodyState cup0;
    uint32_t root_link = 0u;
    uint32_t dof_stride = 0u;
    uint32_t base_dof = 0u;
    uint32_t link_count = 0u;
    StandGraspConfig sg;
    bool has_feet = false;
    double total_mass = 0.0;
    Transform seat_pose{};
    float poly_cx = 0.0f;
    Place place{};
    std::vector<uint32_t> grip_links;
};

UnionScene BuildFloatingNoContactScene(const std::string& mjcf,
                                       const std::string& cup_usda,
                                       bool grasp_cook = false) {
    UnionScene sc;
    sc.h1 = grasp_cook ? LoadFloatingGraspCook(mjcf) : LoadFloating(mjcf);
    sc.host = sc.h1.host;
    // Quiescent IC (== the cup-sequence demo's floating seed): zero velocities,
    // identity base rotation; q stays at the cook rest pose.
    for (auto& v : sc.host.link_velocity)
        for (float& c : v.v) c = 0.0f;
    for (float& qd : sc.host.qdot) qd = 0.0f;
    sc.host.base_pose[0].rotation = Quat::Identity();

    sc.root_link = sc.host.articulation_link_offset[0];
    sc.dof_stride = articulation::ArticulationDofCount(sc.host, 0u);
    sc.base_dof =
        articulation::ArticulationJointDofCount(sc.host.joint_type[sc.root_link]);
    sc.link_count = sc.host.TotalLinkCount();

    // The 30-sphere finger-only wrap (the H1.1 GO set) resolved BY NAME on the
    // floating cook -- handles 9000+ (the canonical disjoint handle map).
    const auto spheres = WrapSpheres(false, Vec3{});
    uint32_t handle = 9000u;
    for (const auto& s : spheres) {
        const uint32_t l = LinkByName(sc.h1, s.body);
        if (l == kInvalidLink) continue;
        CoResidentFingertip ft;
        ft.link = l;
        ft.broadphase_handle = handle++;
        ft.local_offset = s.local_offset;
        ft.radius = s.radius;
        sc.config.fingertips.push_back(ft);
    }

    // The 10.9cm cup hull (COM-centered mesh-local verts), parked FAR to +X.
    const CupHull base = LoadCupHull(cup_usda);
    const CupHull hull = ScaleCupHull(base, kSxy, kSz);
    const Vec3 hull_center = (hull.hi + hull.lo) * 0.5f;
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    sc.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        sc.config.cup.hull_verts[i * 3u + 0u] -= hull_center.x;
        sc.config.cup.hull_verts[i * 3u + 1u] -= hull_center.y;
        sc.config.cup.hull_verts[i * 3u + 2u] -= hull_center.z;
    }
    sc.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = 1.0f / kCupMass;
    const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = sc.host.base_pose[0].position + Vec3{kCupFarOffsetX, 0.0f, 0.0f};
    cup.orientation = Quat::Identity();
    sc.config.cup_state = cup;
    sc.cup0 = cup;

    sc.config.grip_torque.assign(sc.link_count, 0.0f);
    sc.config.drive_force_limits.assign(sc.link_count, 0.0f);
    sc.config.friction_mu = kMu;
    sc.config.condim = 3u;
    return sc;
}

BatchedSceneTemplate MakeUnionTemplate(const UnionScene& sc) {
    BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {sc.config.cup_state};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = sc.host;  // the FLOATING-base whole-body H1.
    tmpl.fingertips = sc.config.fingertips;
    tmpl.cup = sc.config.cup;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque = sc.config.grip_torque;  // all-zero; drive is via SetActions.
    tmpl.drive_force_limits = sc.config.drive_force_limits;
    tmpl.friction_mu = sc.config.friction_mu;
    tmpl.condim = sc.config.condim;
    tmpl.has_feet = sc.has_feet;
    if (sc.has_feet) {
        tmpl.feet = sc.sg.feet;
        tmpl.ground = sc.sg.ground;
        tmpl.foot_mu = sc.sg.foot_mu;
    }
    tmpl.has_table = sc.sg.has_table;
    if (sc.sg.has_table) {
        tmpl.table_height = sc.sg.table_height;
        tmpl.table_mu = sc.sg.table_mu;
        tmpl.table_broadphase_id = sc.sg.table_broadphase_id;
        tmpl.cup_table_proxy_half = sc.sg.cup_table_proxy_half;
        tmpl.cup_table_proxy_offset = sc.sg.cup_table_proxy_offset;
        tmpl.cup_table_proxy_id = sc.sg.cup_table_proxy_id;
    }
    return tmpl;
}

UnionScene BuildFeetGroundScene(const nuka::phi::DeviceContext& context,
                                const std::string& mjcf, const std::string& cup_usda,
                                bool grasp_cook = false) {
    UnionScene sc = BuildFloatingNoContactScene(mjcf, cup_usda, grasp_cook);

    // Bent stance on both legs (hip_pitch/knee/ankle; yaw/roll = 0).
    const auto legs = ResolveLegLinks(sc.h1);
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        float v = 0.0f;
        if (j == 2u) v = kStanceHipPitch;
        else if (j == 3u) v = kStanceKnee;
        else if (j == 4u) v = kStanceAnkle;
        if (legs[s] != kInvalidLink) sc.host.q[legs[s]] = v;
    }

    // Feet: toe/heel spheres per ankle, handles 12000+.
    uint32_t foot_handle = 12000u;
    for (uint32_t a : {legs[4], legs[9]}) {  // left_ankle, right_ankle.
        for (float x : {kFootToeX, kFootHeelX}) {
            CoResidentFootSphere fs;
            fs.link = a;
            fs.broadphase_handle = foot_handle++;
            fs.local_offset = Vec3{x, 0.0f, kFootBottomZ};
            fs.radius = kFootSphereR;
            sc.sg.feet.push_back(fs);
        }
    }

    // Seat the base so the LOWEST foot bottom rests 2mm into the ground.
    const auto poses = ForwardKinematics(context, sc.host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& fs : sc.sg.feet) {
        const Vec3 c =
            poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        min_bottom = std::min(min_bottom, c.z - fs.radius);
    }
    constexpr float kGroundZ = 0.0f;
    constexpr float kRestPen = 0.002f;
    sc.host.base_pose[0].position.z -= (min_bottom - (kGroundZ - kRestPen));
    sc.seat_pose = sc.host.base_pose[0];
    sc.sg.ground.height = kGroundZ;
    sc.sg.ground.broadphase_id = 8000u;
    sc.sg.foot_mu = 0.8f;
    float cx_sum = 0.0f;
    for (const auto& fs : sc.sg.feet) {
        const Vec3 c =
            poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        cx_sum += c.x;
    }
    sc.poly_cx = cx_sum / static_cast<float>(sc.sg.feet.size());

    // Mirror the grasp side into the StandGraspConfig (the oracle input).
    sc.sg.fingertips = sc.config.fingertips;
    sc.sg.cup = sc.config.cup;
    sc.sg.cup_state = sc.config.cup_state;
    sc.sg.finger_mu = sc.config.friction_mu;
    sc.sg.condim = sc.config.condim;
    sc.sg.has_table = false;
    sc.sg.drive_torque.assign(sc.link_count, 0.0f);
    sc.sg.drive_force_limits.assign(sc.link_count, 0.0f);

    sc.has_feet = true;
    sc.total_mass = BuildCoMModel(sc.host).total;
    return sc;
}

// ===========================================================================
// The PD drive set + CoP balance law (verbatim from the shared header; the
// settle pre-roll consumes them; the factory then exports drive tables built
// from the same entries).
// ===========================================================================
struct DriveLink {
    uint32_t link = kInvalidLink;
    float target = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float tlim = 0.0f;
    bool grip = false;
};

std::vector<DriveLink> BuildStanceHoldDriveSet(const UnionScene& sc) {
    std::vector<DriveLink> out;
    const auto legs = ResolveLegLinks(sc.h1);
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        if (j == 4u) continue;  // ankles are CoP-driven (CopCtl), not posture PD.
        DriveLink d;
        d.link = legs[s];
        if (j == 0u) {        // hip_yaw.
            d.target = 0.0f; d.kp = 150.0f; d.kd = 8.0f; d.tlim = 200.0f;
        } else if (j == 1u) { // hip_roll.
            d.target = 0.0f; d.kp = 200.0f; d.kd = 12.0f; d.tlim = 200.0f;
        } else if (j == 2u) { // hip_pitch.
            d.target = kStanceHipPitch; d.kp = 200.0f; d.kd = 14.0f; d.tlim = 200.0f;
        } else {              // knee.
            d.target = kStanceKnee; d.kp = 300.0f; d.kd = 18.0f; d.tlim = 300.0f;
        }
        out.push_back(d);
    }
    auto is_leg = [&](uint32_t l) {
        for (uint32_t s = 0u; s < 10u; ++s)
            if (legs[s] == l) return true;
        return false;
    };
    for (uint32_t l = 0u; l < sc.link_count; ++l) {
        if (l == sc.root_link || is_leg(l)) continue;
        if (articulation::ArticulationJointDofCount(sc.host.joint_type[l]) == 0u) continue;
        DriveLink d;
        d.link = l;
        d.target = sc.host.q[l];  // the seated rest q.
        d.kp = kKpHold;
        d.kd = kKdHold;
        d.tlim = HoldLimitFor(LinkName(sc.h1, l));
        out.push_back(d);
    }
    return out;
}

std::vector<DriveLink> BuildGraspStanceDriveSet(const UnionScene& sc,
                                                float close_offset = kCloseOffset) {
    std::vector<DriveLink> out = BuildStanceHoldDriveSet(sc);
    auto is_grip = [&](uint32_t l) {
        for (uint32_t g : sc.grip_links)
            if (g == l) return true;
        return false;
    };
    const uint32_t wrist = LinkByName(sc.h1, "right_hand_link");
    for (DriveLink& d : out) {
        if (d.link == wrist) {
            // The wrist's REAL MJCF motor limit (right_hand_joint ctrlrange +/-6
            // N*m) instead of HoldLimitFor's 1 N*m finger default -- see the
            // shared-header honesty note (G1c).
            d.tlim = 6.0f;
            continue;
        }
        if (!is_grip(d.link)) continue;
        d.target = sc.host.q[d.link] + close_offset;  // close past the curled pre-pose.
        d.kp = kGripKp;
        d.kd = kGripKd;
        d.tlim = 0.0f;  // unclamped (the H1.1/H1.2 close drive has no clamp).
        d.grip = true;
    }
    return out;
}

struct CopCtl {
    const nuka::phi::DeviceContext* context = nullptr;
    CoMModel com;
    std::array<uint32_t, 2> ankle_links{};
    float poly_cx = 0.0f;
    float kp = 320.0f, kd = 50.0f;   // the costand STAGE3 headline gains.
    float ankle_kd = 1.5f;           // small ankle joint damping (kAnkleKd).
    float tlim = 40.0f;              // the REAL ankle torque limit.
    uint32_t settle = 15u;           // engage after this many drive calls.
    std::vector<Vec3> com_prev;      // per-env previous whole-body CoM.
    std::vector<uint32_t> calls;     // per-env drive-call counter.

    CopCtl(const nuka::phi::DeviceContext& ctx, const UnionScene& sc, uint32_t envs)
        : context(&ctx),
          com(BuildCoMModel(sc.host)),
          poly_cx(sc.poly_cx),
          com_prev(envs, Vec3{}),
          calls(envs, 0u) {
        const auto legs = ResolveLegLinks(sc.h1);
        ankle_links = {legs[4], legs[9]};
    }

    float AnkleTorque(const articulation::ArticulationHostState& st, uint32_t e) {
        const auto poses = ForwardKinematics(*context, st);
        const Vec3 c = WholeBodyCoM(com, poses);
        const uint32_t k = calls[e]++;
        const float com_vx = (k == 0u) ? 0.0f : (c.x - com_prev[e].x) / kDt;
        com_prev[e] = c;
        if (k < settle) return 0.0f;
        return kp * (c.x - poly_cx) + kd * com_vx;
    }
};

uint32_t DofIndexOf(const articulation::ArticulationHostState& host,
                    uint32_t root_link, uint32_t link) {
    uint32_t idx = 0u;
    for (uint32_t l = root_link; l < link; ++l)
        idx += articulation::ArticulationJointDofCount(host.joint_type[l]);
    return idx;
}

// Oracle drive seam (verbatim except the ADDITIVE `ankle_tau_out` recorder: the
// per-ankle applied torque is COPIED out after computation -- no float changes).
void DrivePdOracle(UnifiedCoResidentStepper& stepper, const UnionScene& sc,
                   const std::vector<DriveLink>& drive, float target_scale,
                   CopCtl* cop = nullptr, float* ankle_tau_out = nullptr) {
    articulation::ArticulationHostState st;
    stepper.Download(&st);
    std::vector<float> tau(sc.link_count, 0.0f);
    for (const DriveLink& d : drive) {
        const float tgt = d.grip ? d.target : target_scale * d.target;
        float u = d.kp * (tgt - st.q[d.link]) - d.kd * st.qdot[d.link];
        if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
        tau[d.link] = u;
    }
    if (cop != nullptr) {
        const float tau_cop = cop->AnkleTorque(st, 0u);
        uint32_t a = 0u;
        for (uint32_t l : cop->ankle_links) {
            float u = tau_cop - cop->ankle_kd * st.qdot[l];
            u = std::max(-cop->tlim, std::min(cop->tlim, u));
            tau[l] = u;
            if (ankle_tau_out != nullptr && a < 2u) ankle_tau_out[a] = u;
            ++a;
        }
    }
    stepper.SetGripTorque(tau);
}

// ===========================================================================
// G1c/G1d scene assembly (verbatim; the settle pre-roll lives here).
// ===========================================================================
constexpr uint32_t kGraspSettleSteps = 200u;  // ~0.83s: stance+arm quasi-static.

UnionScene BuildGraspStandScene(const nuka::phi::DeviceContext& context,
                                const std::string& mjcf, const std::string& cup_usda,
                                float* ankle_tau_out) {
    UnionScene sc = BuildFeetGroundScene(context, mjcf, cup_usda, /*grasp_cook=*/true);

    // Curl the right hand to the H1.1 curled-open pre-pose.
    ApplyCurl(sc.h1, &sc.host, CurlForScale(kSxy));

    // The 12 grip-driven links, resolved by name on the floating cook.
    for (const char* nm : kWrapDriven) {
        const uint32_t l = LinkByName(sc.h1, nm);
        if (l != kInvalidLink) sc.grip_links.push_back(l);
    }
    const uint32_t hand = LinkByName(sc.h1, "right_hand_link");
    if (hand == kInvalidLink) return sc;  // place.found stays false -> loud throw.

    // The cup placement, searched on the SEAT + curled FK.
    const auto seat_poses = ForwardKinematics(context, sc.host);
    const Transform hand_seat = seat_poses[hand];
    {
        const auto spheres = WrapSpheres(false, Vec3{});
        const CupHull base = LoadCupHull(cup_usda);
        const CupHull hull = ScaleCupHull(base, kSxy, kSz);
        sc.place = BestPlacementShallowNoPalmCaging(sc.h1, seat_poses, spheres, hull);
    }

    // SETTLE pre-roll (cup STILL parked 5m away -> exactly the G1b feet-only
    // regime). Deterministic -> the scene is identical across every consumer.
    {
        UnifiedCoResidentStepper settle(context, sc.host, sc.sg, kGravityZ, kDt);
        CopCtl cop(context, sc, 1u);
        const auto hold = BuildGraspStanceDriveSet(sc, /*close_offset=*/0.0f);
        for (uint32_t s = 0u; s < kGraspSettleSteps; ++s) {
            DrivePdOracle(settle, sc, hold, 1.0f, &cop, ankle_tau_out);
            settle.Step();
        }
        articulation::ArticulationHostState settled;
        settle.Download(&settled);
        sc.host.q = settled.q;
        sc.host.qdot = settled.qdot;
        sc.host.link_velocity = settled.link_velocity;
        sc.host.base_pose[0] = settled.base_pose[0];
    }

    // Carry the searched placement through the settle by the HAND-FRAME delta.
    if (sc.place.found) {
        const auto settled_poses = ForwardKinematics(context, sc.host);
        const Transform hand_settled = settled_poses[hand];
        const Quat dq = hand_settled.rotation * hand_seat.rotation.Conjugate();
        const Vec3 local = hand_seat.rotation.Conjugate().Rotate(
            sc.place.center - hand_seat.position);
        sc.place.center = hand_settled.position + hand_settled.rotation.Rotate(local);
        sc.config.cup_state.position = sc.place.center;
        sc.config.cup_state.orientation = dq;  // near-identity settle tilt.
    }

    // Move the cup INTO the hand on BOTH sides (oracle config + template seed).
    sc.sg.cup_state = sc.config.cup_state;
    sc.cup0 = sc.config.cup_state;
    return sc;
}

constexpr float kTableRestPen = 0.002f;

UnionScene BuildGraspStandTableScene(const nuka::phi::DeviceContext& context,
                                     const std::string& mjcf,
                                     const std::string& cup_usda,
                                     float* ankle_tau_out) {
    UnionScene sc = BuildGraspStandScene(context, mjcf, cup_usda, ankle_tau_out);
    if (!sc.place.found) return sc;  // caller throws loudly.
    const CupHull base = LoadCupHull(cup_usda);
    const CupHull hull = ScaleCupHull(base, kSxy, kSz);
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    sc.sg.has_table = true;
    sc.sg.cup_table_proxy_half = half;  // bbox proxy, centered (offset 0).
    sc.sg.cup_table_proxy_offset = Vec3{0.0f, 0.0f, 0.0f};
    sc.sg.cup_table_proxy_id = 7001u;
    sc.sg.table_broadphase_id = 8500u;
    sc.sg.table_mu = 0.6f;
    sc.sg.table_height = sc.cup0.position.z - half.z + kTableRestPen;
    return sc;
}

// ===========================================================================
// G2 factory-only metadata assembly (NOT part of the byte-equality surface).
// ===========================================================================
// The ankle POSTURE stand-in gains for the exported drive tables (the CoP law
// needs an FK CoM probe a python caller does not have). kp must beat the
// inverted-pendulum gravity stiffness m*g*h (~460 N*m/rad for the settled H1);
// the clamp is the REAL ankle motor limit.
constexpr float kAnklePostureKp = 400.0f;
constexpr float kAnklePostureKd = 30.0f;
constexpr float kAnkleTorqueLimit = 40.0f;

std::vector<H1UnionDriveEntry> ExportDriveTable(
    const UnionScene& sc, const std::vector<DriveLink>& drive,
    const std::array<uint32_t, 2>& ankle_links, const float* ankle_tau) {
    std::vector<H1UnionDriveEntry> out;
    out.reserve(drive.size() + 2u);
    for (const DriveLink& d : drive) {
        H1UnionDriveEntry e;
        e.link = d.link;
        e.dof = DofIndexOf(sc.host, sc.root_link, d.link);
        e.target = d.target;
        e.kp = d.kp;
        e.kd = d.kd;
        e.tlim = d.tlim;
        e.grip = d.grip ? 1u : 0u;
        out.push_back(e);
    }
    // Ankle posture entries: target biased by the settle's final CoP torque so
    // the PD reproduces the equilibrium feedforward at the settled pose.
    for (uint32_t a = 0u; a < 2u; ++a) {
        const uint32_t l = ankle_links[a];
        if (l == kInvalidLink) continue;
        H1UnionDriveEntry e;
        e.link = l;
        e.dof = DofIndexOf(sc.host, sc.root_link, l);
        e.target = sc.host.q[l] + ankle_tau[a] / kAnklePostureKp;
        e.kp = kAnklePostureKp;
        e.kd = kAnklePostureKd;
        e.tlim = kAnkleTorqueLimit;
        e.grip = 0u;
        out.push_back(e);
    }
    return out;
}

}  // namespace

H1UnionScene BuildH1UnionScene(const phi::DeviceContext& context,
                               const std::string& h1_mjcf,
                               const std::string& cup_usda) {
    if (!std::filesystem::exists(h1_mjcf)) {
        throw std::runtime_error("BuildH1UnionScene: H1 MJCF not found at '" +
                                 h1_mjcf + "' (run from the repo root or pass an "
                                 "absolute path)");
    }
    if (!std::filesystem::exists(cup_usda)) {
        throw std::runtime_error("BuildH1UnionScene: cup USDA not found at '" +
                                 cup_usda + "'");
    }

    H1UnionScene out;
    UnionScene sc = BuildGraspStandTableScene(context, h1_mjcf, cup_usda,
                                              out.ankle_settle_tau);
    if (!sc.place.found) {
        throw std::runtime_error(
            "BuildH1UnionScene: the cup placement search FAILED (no shallow "
            "force-closure caging pose on the seat FK) -- the scene would be "
            "degenerate; refusing to construct");
    }
    if (sc.config.fingertips.size() < 30u || sc.sg.feet.size() != 4u) {
        throw std::runtime_error(
            "BuildH1UnionScene: wrap/feet authoring incomplete (fingertips=" +
            std::to_string(sc.config.fingertips.size()) + ", feet=" +
            std::to_string(sc.sg.feet.size()) + ")");
    }

    out.tmpl = MakeUnionTemplate(sc);
    out.dof_stride = sc.dof_stride;
    out.base_dof = sc.base_dof;
    out.root_link = sc.root_link;
    out.link_count = sc.link_count;
    out.total_mass = sc.total_mass;
    out.poly_cx = sc.poly_cx;
    out.seat_pose = sc.seat_pose;
    out.place_found = sc.place.found;
    out.table_height = sc.sg.table_height;
    out.cup_mass = kCupMass;
    out.grip_links = sc.grip_links;
    out.grip_dofs.clear();
    for (uint32_t l : sc.grip_links)
        out.grip_dofs.push_back(DofIndexOf(sc.host, sc.root_link, l));

    // The three reference choreography tables, built from the SETTLED host q
    // (the same call the G1d gates make post-settle) + the ankle posture rows.
    CopCtl cop_meta(context, sc, 1u);  // ankle link resolution only (no FK call).
    out.drive_hold = ExportDriveTable(sc, BuildGraspStanceDriveSet(sc, 0.0f),
                                      cop_meta.ankle_links, out.ankle_settle_tau);
    out.drive_rest = ExportDriveTable(sc,
                                      BuildGraspStanceDriveSet(sc, kH1UnionRestBackOffset),
                                      cop_meta.ankle_links, out.ankle_settle_tau);
    out.drive_close = ExportDriveTable(sc, BuildGraspStanceDriveSet(sc, kCloseOffset),
                                       cop_meta.ankle_links, out.ankle_settle_tau);

    // Per-DOF |torque| limits: legs at the MJCF motor limits, wrist 6 N*m, the
    // rest at the HoldLimitFor envelope; the 6 dead base columns stay 0.
    out.dof_torque_limit.assign(sc.dof_stride, 0.0f);
    const auto legs = ResolveLegLinks(sc.h1);
    const uint32_t wrist_r = LinkByName(sc.h1, "right_hand_link");
    const uint32_t wrist_l = LinkByName(sc.h1, "left_hand_link");
    for (uint32_t l = sc.root_link + 1u; l < sc.link_count; ++l) {
        if (articulation::ArticulationJointDofCount(sc.host.joint_type[l]) == 0u)
            continue;
        const uint32_t d = DofIndexOf(sc.host, sc.root_link, l);
        if (d >= sc.dof_stride) continue;
        float lim = HoldLimitFor(LinkName(sc.h1, l));
        for (uint32_t s = 0u; s < 10u; ++s) {
            if (legs[s] == l) {
                lim = LegLimit(s);
                break;
            }
        }
        if (l == wrist_r || l == wrist_l) lim = 6.0f;
        out.dof_torque_limit[d] = lim;
    }
    return out;
}

}  // namespace nuka::runtime::coresident
