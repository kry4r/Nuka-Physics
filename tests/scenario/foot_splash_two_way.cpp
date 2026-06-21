// ---------------------------------------------------------------------------
// PBF FLUID <-> rigid foot TWO-WAY coupling on the ONE general row solver.
//
// A PBF fluid pool (a lattice of fluid particles confined by four static walls + a
// z-up boundary floor over a ground plane) settles, then a rigid sphere "foot" is
// driven down into it. The fluid particles are SPHERE collidables flowing
// through the SAME body<->particle row solver a foot uses on the ground, the SAME
// row a cloth or tet particle uses: detection -> body<->particle narrowphase ->
// emit -> SolveRowsBlockIsland -> PBF finalize compose. There is NO bespoke fluid
// coupler and NO grafted buoyancy/drag force -- the resistance EMERGES from the
// non-penetration rows plus PBF incompressibility (the displaced fluid's density
// projection pushes back). A particle is a particle to the contact row; the medium
// difference is DATA (the fluid's mu ~= 0 lets the foot slide tangentially while
// the normal row still brakes penetration), never code.
//
// The two-way effect, proven numerically against a no-fluid control:
//   (a) SPLASH: the foot plunging into the pool displaces fluid up/aside -- the
//       fluid surface rises measurably above its settled rest surface.
//   (b) TWO-WAY: the foot is resisted/supported by the fluid -- a body<->particle
//       normal row carries a body-side lambda > 0 (the reaction the fluid pushes
//       back into the foot) AND a dynamic foot released onto the pool decelerates,
//       coming to rest well above a no-fluid control that free-falls to the floor.
//   Every coupling row asserted has exactly one side kind == kNkSideParticle, so
//   the contact is the general body<->particle path, not a special-cased coupler.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "collision/shape_kind.hpp"
#include "math/quat.hpp"  // ground-plane rotation (+Y -> +Z).
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = nuka::collision::kShapeBox;
constexpr uint32_t kKindSphere = nuka::collision::kShapeSphere;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// Fluid pool geometry. A wide lattice resting on a z-up boundary floor at z=0
// (coincident with a ground plane), confined laterally by four static walls so it
// settles into a coherent pool. A real ground plane under the whole scene gives the
// no-fluid control a floor to rest on, so the with/without-fluid foot rest is a
// clean physical comparison.
constexpr float kSpacing = 0.035f;             // fluid particle spacing.
constexpr float kSupportRadius = kSpacing * 1.5f;
constexpr float kRestDensity = 1000.0f;
constexpr uint32_t kPoolNx = 6u;               // footprint particles (x,y).
constexpr uint32_t kPoolNz = 8u;               // pool depth in particles (deep pool).
constexpr float kFloorZ = 0.0f;                // ground plane + PBF boundary (bottom).
constexpr float kPoolBottomZ = kFloorZ + 0.5f * kSpacing;  // first layer above floor.
// The pool footprint is centred on the origin; the walls sit just outside it.
constexpr float kHalfFoot = 0.5f * static_cast<float>(kPoolNx - 1u) * kSpacing;
constexpr float kWallGap = 0.6f * kSpacing;    // wall stand-off from the outer fluid.
constexpr float kWallInner = kHalfFoot + kWallGap;  // inner face of each wall.
constexpr float kWallHalfThick = 0.02f;
constexpr float kWallHalfHeight = 0.24f;       // walls taller than the settled pool.
constexpr float kContactDMin = kSpacing;       // particle sphere radius = d_min/2.
constexpr float kFootRadius = 0.05f;
// Bodies: ground plane (0) + 4 walls (1..4) + the foot (5).
constexpr uint32_t kFootBody = 5u;
constexpr uint32_t kSettleSteps = 900u;

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    // A big velocity budget: the foot:fluid mass gap needs PGS iterations to
    // transmit momentum without the foot tunnelling through the pool.
    cfg.vel_iters = 48u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}

// A static z-up ground plane at z=0 (the amf plane normal is local +Y, rotated to
// world +Z). The pool rests on it; the no-fluid control foot lands on it.
void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.rotation = Quat::FromAxisAngle(Vec3{1, 0, 0}, 1.57079632679f);
    bi.inv_mass = 0.0f;
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindPlane;
    sh.params[0] = 0.0f; sh.params[1] = 0.0f; sh.params[2] = 0.0f;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// A static box wall (immovable, one-way by mass). The pool's lateral container.
void AddStaticBox(nk::Model& m, const Vec3& pos, const Vec3& half, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = 0.0f;
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half.x; sh.params[1] = half.y; sh.params[2] = half.z;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// A sphere body collidable. inv_mass == 0 -> kinematic (driven by upload);
// inv_mass > 0 -> a free dynamic body the contact reaction can move.
void AddSphere(nk::Model& m, const Vec3& pos, float radius, float inv_mass,
               int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = inv_mass;
    if (inv_mass > 0.0f) {
        const float ii = inv_mass / (0.4f * radius * radius);  // solid-sphere diag.
        bi.inv_inertia = Vec3{ii, ii, ii};
    } else {
        bi.inv_inertia = Vec3{0, 0, 0};
    }
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindSphere;
    sh.params[0] = radius; sh.params[1] = radius; sh.params[2] = radius;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// The PBF pool input: a lattice over the boundary floor + the grid AABB sized to
// cover the settled pool AND the splash headroom (a particle outside the grid finds
// no neighbours, so the dims must enclose the whole working volume).
cook::PbfCookInput BuildPool() {
    cook::PbfCookInput in;
    const float s = kSpacing, h = kSupportRadius, rho0 = kRestDensity;
    const float mass = rho0 * s * s * s;
    const float c0 = -kHalfFoot;
    for (uint32_t iz = 0; iz < kPoolNz; ++iz)
        for (uint32_t iy = 0; iy < kPoolNx; ++iy)
            for (uint32_t ix = 0; ix < kPoolNx; ++ix)
                in.positions.push_back(Vec3{c0 + ix * s, c0 + iy * s,
                                            kPoolBottomZ + iz * s});
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.particle_mass = mass;
    in.rest_density = rho0;
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = 4u;
    in.clamp_overdensity = true;
    in.boundary_enabled = true;
    in.floor_z = kFloorZ;
    in.friction = 0.0f;  // fluid mu ~= 0: the foot slides; splash stays normal-driven.
    // Grid AABB: span the walled footprint + vertical headroom for the splash + the
    // plunging foot. cell_size == support_radius, so dims = ceil(extent / h) + pad.
    const float span_xy = 2.0f * (kWallInner + kWallHalfThick) + 2.0f * h;
    const float top_z = kWallHalfHeight * 2.0f + kFootRadius + 4.0f * h;
    const float lo = -(kWallInner + kWallHalfThick) - h;
    in.grid_min = Vec3{lo, lo, kFloorZ - h};
    auto cells = [&](float extent) {
        return static_cast<uint32_t>(std::ceil(extent / h)) + 1u;
    };
    in.grid_dims[0] = cells(span_xy);
    in.grid_dims[1] = cells(span_xy);
    in.grid_dims[2] = cells(top_z - (kFloorZ - h));
    return in;
}

// Cook the ground plane + four walls + the foot + the PBF pool into ONE Model.
// CookPbfParticles sets mode = Pbf (the single-system fluid path that exercises the
// PBF finalize compose) and grows the rigid budget additively for the body<->particle
// reserve. foot_inv_mass picks kinematic(0)/dynamic(>0).
nk::Model BuildScene(const Vec3& foot_pos, float foot_inv_mass, bool fluid_present) {
    nk::Model m;
    const Vec3 wx_half{kWallHalfThick, kWallInner + kWallHalfThick, kWallHalfHeight};
    const Vec3 wy_half{kWallInner + kWallHalfThick, kWallHalfThick, kWallHalfHeight};
    const float wz = kFloorZ + kWallHalfHeight;
    AddGroundPlane(m, 0);                                                         // floor.
    AddStaticBox(m, Vec3{-(kWallInner + kWallHalfThick), 0.0f, wz}, wx_half, 1);  // -x.
    AddStaticBox(m, Vec3{+(kWallInner + kWallHalfThick), 0.0f, wz}, wx_half, 2);  // +x.
    AddStaticBox(m, Vec3{0.0f, -(kWallInner + kWallHalfThick), wz}, wy_half, 3);  // -y.
    AddStaticBox(m, Vec3{0.0f, +(kWallInner + kWallHalfThick), wz}, wy_half, 4);  // +y.
    AddSphere(m, foot_pos, kFootRadius, foot_inv_mass, 5);                        // foot.

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;   // rigid budget (cook sizing).
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::PbfCookInput pool = BuildPool();
    if (!fluid_present)
        for (Vec3& p : pool.positions) p.z -= 10.0f;  // sink the pool out of reach.
    cook::CookPbfParticles(m, 1u, pool);    // appends the fluid + grows the budget.
    m.particles.pp_contact_d_min = kContactDMin;  // body<->fluid sphere radius.
    return m;
}

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17],
// b.kind [24] b.index [25].
struct RowSides { uint32_t a_kind, b_kind; bool active; float upper; };
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.upper = r[7];
    s.a_kind = u(16); s.b_kind = u(24);
    return s;
}

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(),
                              P * sizeof(Vec3));
}

// Max particle z over the whole pool -- the fluid free surface height.
float SurfaceMaxZ(const std::vector<Vec3>& pos) {
    float hi = -1.0e9f;
    for (const Vec3& p : pos) hi = std::max(hi, p.z);
    return hi;
}

// Min particle z over the disc of radius `reach` about the foot xy -- the pocket
// the foot pushes the fluid out of (the surface dips directly under the foot).
float MinZUnderFoot(const std::vector<Vec3>& pos, const Vec3& foot_xy, float reach) {
    float lo = 1.0e9f;
    for (const Vec3& p : pos) {
        const float dx = p.x - foot_xy.x, dy = p.y - foot_xy.y;
        if (dx * dx + dy * dy <= reach * reach) lo = std::min(lo, p.z);
    }
    return lo;
}

// Per-body byte offsets into the env-major Data fields (the foot is body kFootBody).
constexpr uint64_t kPoseOff = static_cast<uint64_t>(kFootBody) * sizeof(Transform);
constexpr uint64_t kVec3Off = static_cast<uint64_t>(kFootBody) * sizeof(Vec3);

// Drive the kinematic foot to `target` this step: set its pose + zero its velocity
// (a scripted kinematic body, the viewer_move_entity primitive).
void DriveFoot(nk::World& w, const Vec3& target) {
    Transform tf = Transform::Identity();
    tf.position = target;
    const Vec3 zero = Vec3::Zero();
    w.GetData().UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
    w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
    w.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);
}

// Scan THIS step's rows for a body<->particle normal reaction; updates the deepest
// body-side lambda seen + the count of body<->particle rows. Asserts each coupling
// row has exactly one particle side (the general path, never a two-particle row).
void ScanCouplingRows(nk::Data& d, uint32_t rows, std::vector<float>* urows,
                      std::vector<float>* lambda, float* max_lambda,
                      uint32_t* particle_rows) {
    d.DownloadField(nk::FieldId::Urows, urows->data(), urows->size() * sizeof(float));
    d.DownloadField(nk::FieldId::Lambda, lambda->data(), lambda->size() * sizeof(float));
    for (uint32_t row = 0u; row < rows; ++row) {
        const RowSides rs = DecodeRow(*urows, row);
        if (!rs.active) continue;
        const bool a_part = rs.a_kind == nk::kNkSideParticle;
        const bool b_part = rs.b_kind == nk::kNkSideParticle;
        if (!(a_part || b_part)) continue;
        EXPECT_FALSE(a_part && b_part) << "row has two particle sides";
        ++(*particle_rows);
        if (rs.upper > 1.0e30f)  // a normal row (friction spokes cap upper to 0).
            *max_lambda = std::max(*max_lambda, (*lambda)[row]);
    }
}

}  // namespace

// (a) SPLASH + (b) reaction: a KINEMATIC foot driven down into the settled pool
// displaces fluid (the surface rises above rest, the pocket under the foot drops)
// AND a body<->particle normal row carries a body-side lambda > 0. The foot's
// motion is scripted; the fluid's response flows entirely through the coupling rows.
TEST(FootSplashTwoWay, KinematicFootDisplacesFluidAndIsResisted) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const Vec3 foot_xy{0.0f, 0.0f, 0.0f};
    const float park = kWallHalfHeight * 2.0f + kFootRadius + 0.08f;  // above the pool.
    nk::Model m = BuildScene(Vec3{foot_xy.x, foot_xy.y, park}, 0.0f, /*fluid=*/true);
    const uint32_t rows = m.capacities.max_rows_per_env;
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    // SETTLE: foot parked high, the pool relaxes to a flat free surface.
    for (uint32_t s = 0; s < kSettleSteps; ++s) { DriveFoot(w, Vec3{0, 0, park}); w.Step(); }
    std::vector<Vec3> settled;
    DownloadParticles(w, &settled);
    const float rest_surface = SurfaceMaxZ(settled);
    const float reach = kFootRadius + kSpacing;
    const float rest_under = MinZUnderFoot(settled, foot_xy, reach);

    // PLUNGE: drive the foot down so its lower hemisphere enters the fluid (its
    // centre dips to ~one foot-radius below the rest surface), over ~250 steps,
    // hold ~200. The displaced fluid is pushed up/aside.
    const float bottom = rest_surface - kFootRadius;
    std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
    std::vector<float> lambda(rows, 0.0f);
    float max_body_lambda = 0.0f;
    uint32_t particle_rows_seen = 0u;
    float splash_surface = -1.0e9f;     // highest free surface seen during the plunge.
    nk::Data& d = w.GetData();
    for (uint32_t s = 0; s < 250u; ++s) {
        const float t = static_cast<float>(s) / 249.0f;
        DriveFoot(w, Vec3{0, 0, park + (bottom - park) * t});
        w.Step();
        ScanCouplingRows(d, rows, &urows, &lambda, &max_body_lambda,
                         &particle_rows_seen);
        std::vector<Vec3> p; DownloadParticles(w, &p);
        splash_surface = std::max(splash_surface, SurfaceMaxZ(p));
    }
    for (uint32_t s = 0; s < 200u; ++s) {
        DriveFoot(w, Vec3{0, 0, bottom});
        w.Step();
        ScanCouplingRows(d, rows, &urows, &lambda, &max_body_lambda,
                         &particle_rows_seen);
        std::vector<Vec3> p; DownloadParticles(w, &p);
        splash_surface = std::max(splash_surface, SurfaceMaxZ(p));
    }
    std::vector<Vec3> pressed; DownloadParticles(w, &pressed);
    const float pressed_under = MinZUnderFoot(pressed, foot_xy, reach);

    std::fprintf(stderr,
                 "[foot-splash] rest_surface=%.4f splash_surface=%.4f (rise=%.4f) "
                 "rest_under=%.4f pressed_under=%.4f (drop=%.4f)\n",
                 rest_surface, splash_surface, splash_surface - rest_surface,
                 rest_under, pressed_under, rest_under - pressed_under);
    std::fprintf(stderr,
                 "[foot-splash] body_lambda=%.6f particle_rows_seen=%u\n",
                 max_body_lambda, particle_rows_seen);

    for (const Vec3& p : pressed)
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));

    // (a) SPLASH: the foot displaced fluid -- the free surface rose above rest OR the
    // pocket directly under the foot dropped below the rest surface (displaced aside).
    const float rise = splash_surface - rest_surface;
    const float pocket_drop = rest_under - pressed_under;
    EXPECT_TRUE(rise > 0.005f || pocket_drop > 0.005f)
        << "the foot did not displace the fluid (no splash rise, no pocket drop)";
    // (b) TWO-WAY reaction: a body<->particle normal row carried a real push-out.
    EXPECT_GT(particle_rows_seen, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_GT(max_body_lambda, 0.0f)
        << "the fluid produced no body-side reaction impulse on the foot";
}

// (b) reaction direction, second proof: a DYNAMIC foot released onto the pool is
// resisted/supported by the fluid -- it decelerates and rests well above a no-fluid
// control that falls to the bare ground plane. The arrest EMERGES from the contact
// rows + PBF incompressibility; there is NO buoyancy/drag force grafted onto the row.
TEST(FootSplashTwoWay, DynamicFootIsDeceleratedByFluidVsControl) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const Vec3 foot_xy{0.0f, 0.0f, 0.0f};
    const float inv_mass = 1.0f / 0.30f;  // 0.30 kg dynamic foot (light enough to float).

    // Drops a DYNAMIC foot onto the pool centre and returns its final z + min z during
    // the drop + the deepest body<->particle normal reaction. With `fluid_present`
    // false the pool is sunk far below so the foot falls through the empty container to
    // rest on the ground plane (z ~= kFootRadius).
    auto drop_foot = [&](bool fluid_present, float* out_lambda,
                         uint32_t* out_rows, float* out_min_z) -> float {
        nk::Model m = BuildScene(Vec3{foot_xy.x, foot_xy.y, 1.0f}, inv_mass,
                                 fluid_present);
        const uint32_t rows = m.capacities.max_rows_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return 1e9f;
        nk::Data& d = w.GetData();
        // Settle the pool while the foot is parked high (kinematic-style hold via a
        // velocity zero each step is not needed: a dynamic foot at z=1 just falls, so
        // we hold it by re-uploading its pose during the settle).
        Transform hold = Transform::Identity();
        hold.position = Vec3{foot_xy.x, foot_xy.y, 1.0f};
        const Vec3 zero = Vec3::Zero();
        for (uint32_t s = 0; s < kSettleSteps; ++s) {
            d.UploadField(nk::FieldId::BodyPose, &hold, sizeof(Transform), kPoseOff);
            d.UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
            w.Step();
        }
        std::vector<Vec3> settled; DownloadParticles(w, &settled);
        const float surface = fluid_present ? SurfaceMaxZ(settled) : kFloorZ;
        // Release the (still dynamic) foot just above the surface, at rest, and drop it.
        const float release_z = surface + kFootRadius + 0.06f;
        Transform tf = Transform::Identity();
        tf.position = Vec3{foot_xy.x, foot_xy.y, release_z};
        d.UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
        d.UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
        d.UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);

        std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
        std::vector<float> lambda(rows, 0.0f);
        float max_lambda = 0.0f;
        uint32_t particle_rows = 0u;
        float min_z = 1.0e9f;
        for (uint32_t s = 0; s < 360u; ++s) {
            w.Step();
            ScanCouplingRows(d, rows, &urows, &lambda, &max_lambda, &particle_rows);
            Transform got = Transform::Identity();
            d.DownloadField(nk::FieldId::BodyPose, &got, sizeof(Transform), kPoseOff);
            min_z = std::min(min_z, got.position.z);
        }
        Transform got = Transform::Identity();
        d.DownloadField(nk::FieldId::BodyPose, &got, sizeof(Transform), kPoseOff);
        if (out_lambda) *out_lambda = max_lambda;
        if (out_rows) *out_rows = particle_rows;
        if (out_min_z) *out_min_z = min_z;
        return got.position.z;
    };

    float body_particle_lambda = 0.0f;
    uint32_t particle_rows_seen = 0u;
    float foot_min_z_with_fluid = 1e9f;
    const float foot_z_with_fluid =
        drop_foot(true, &body_particle_lambda, &particle_rows_seen,
                  &foot_min_z_with_fluid);
    float ctrl_min_z = 1e9f;
    const float foot_z_no_fluid = drop_foot(false, nullptr, nullptr, &ctrl_min_z);

    std::fprintf(stderr,
                 "[foot-splash] foot_z_with_fluid=%.4f foot_z_no_fluid=%.4f "
                 "foot_min_z_with_fluid=%.4f body_lambda=%.6f particle_rows_seen=%u\n",
                 foot_z_with_fluid, foot_z_no_fluid, foot_min_z_with_fluid,
                 body_particle_lambda, particle_rows_seen);

    // The reaction direction: a body<->particle normal row carried a real push-out.
    EXPECT_GT(particle_rows_seen, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_GT(body_particle_lambda, 0.0f)
        << "the fluid produced no body-side reaction impulse on the foot";
    // The foot was DECELERATED/supported by the fluid: it came to rest measurably
    // above the no-fluid control, which free-falls to the floor (z ~= kFootRadius).
    EXPECT_GT(foot_z_with_fluid, foot_z_no_fluid + 0.02f)
        << "the fluid did not resist the foot (it sank as far as the no-fluid control)";
}
