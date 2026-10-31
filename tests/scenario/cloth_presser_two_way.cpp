// Cloth deformation, recovery, and rigid reaction share the general row solver.
// A free drape spans two ridges; a presser supplies kinematic and dynamic loads.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "collision/shape_kind.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = nuka::collision::kShapeBox;
constexpr uint32_t kKindSphere = nuka::collision::kShapeSphere;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// Cloth + scene geometry. An unpinned sheet across two parallel low ridges settles
// like a hammock: it rests on the ridge tops, its center sagging into the gap.
constexpr uint32_t kGridN = 24u;          // 24x24 = 576 particles.
constexpr float kSpacing = 0.025f;        // patch ~0.575 m square.
constexpr float kClothZ = 0.16f;          // flat lay height just above the ridge tops.
constexpr float kParticleMass = 0.01f;    // 10 g per particle (light cloth).
constexpr float kContactDMin = 0.030f;    // particle sphere radius = d_min/2 = 0.015.
constexpr float kRidgeHalfX = 0.12f;      // each ridge half-extents (tops at z=0.10).
constexpr float kRidgeHalfY = 0.30f;
constexpr float kRidgeHalfZ = 0.05f;
constexpr float kRidgeTopZ = kRidgeHalfZ * 2.0f;  // ridges sit on the ground -> 0.10.
constexpr float kPresserRadius = 0.06f;
constexpr float kRidgeCenterX = kRidgeHalfX + kPresserRadius + kContactDMin;
constexpr uint32_t kSettleSteps = 800u;
constexpr uint32_t kPresserBody = 3u;     // ground(0) + ridge x2(1,2) + presser(3).

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
    cfg.vel_iters = 32u;
    cfg.velocity_iterations_override = true;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}

// A static body collidable (immovable, one-way by mass).
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

// A static Z-up half-space supports the hanging cloth edges at z=0.
void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
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

// A sphere body collidable. inv_mass == 0 makes it kinematic (driven by upload),
// inv_mass > 0 makes it a free dynamic body the contact reaction can move.
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

// Unpinned cloth drapes over the ridges as shell and hinge vertex blocks.
cook::VbdCookInput BuildCloth() {
    std::vector<Vec3> rest;
    rest.reserve(kGridN * kGridN);
    const float c0 = -0.5f * static_cast<float>(kGridN - 1u) * kSpacing;
    for (uint32_t j = 0; j < kGridN; ++j) {
        for (uint32_t i = 0; i < kGridN; ++i) {
            rest.push_back(Vec3{c0 + static_cast<float>(i) * kSpacing,
                                c0 + static_cast<float>(j) * kSpacing, kClothZ});
        }
    }
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridN + i; };
    std::vector<soft::ClothTriangle> tris;
    for (uint32_t j = 0; j + 1 < kGridN; ++j) {
        for (uint32_t i = 0; i + 1 < kGridN; ++i) {
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j),
                                                idx(i + 1, j + 1)}});
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j + 1),
                                                idx(i, j + 1)}});
        }
    }
    soft::ShellMaterial shell;
    shell.stretch_stiffness = 1.0e4f;  // near-inextensible stretch.
    shell.poisson = 0.3f;
    shell.bend_stiffness = 1.0e-5f;    // soft bend so it drapes.

    cook::VbdCookInput in;
    in.positions = rest;
    // A tiny downward seed so the flat sheet commits to the ridge symmetrically.
    in.velocities.assign(rest.size(), Vec3{0.0f, 0.0f, -0.05f});
    in.inv_mass.assign(rest.size(), 1.0f / kParticleMass);  // UNPINNED free drape.
    soft::BuildClothVertexBlocks(rest, tris, {}, {}, std::vector<soft::ShellMaterial>(tris.size(), shell),
                                 in.vbd_elements);
    in.vbd_count = static_cast<uint32_t>(rest.size());
    in.half_thickness.assign(rest.size(), 0.5f * kContactDMin);
    in.friction.assign(rest.size(), 0.6f);
    return in;
}

// The rigid and cloth geometry use one model and a shared contact row budget.
// Zero inverse mass drives the presser kinematically; positive mass permits reaction.
nk::Model BuildScene(const Vec3& presser_pos, float presser_inv_mass,
                     float presser_radius = kPresserRadius) {
    nk::Model m;
    const Vec3 ridge_half{kRidgeHalfX, kRidgeHalfY, kRidgeHalfZ};
    AddGroundPlane(m, 0);                                                 // ground.
    AddStaticBox(m, Vec3{-kRidgeCenterX, 0.0f, kRidgeHalfZ}, ridge_half, 1);  // ridge -x.
    AddStaticBox(m, Vec3{+kRidgeCenterX, 0.0f, kRidgeHalfZ}, ridge_half, 2);  // ridge +x.
    AddSphere(m, presser_pos, presser_radius, presser_inv_mass, 3);       // presser.

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;   // rigid budget (cook sizing).
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    const cook::VbdCookInput cloth = BuildCloth();
    cook::CookVbdParticles(m, 1u, cloth);    // appends cloth + grows the budget.
    m.particles.pp_contact_d_min = kContactDMin;
    return m;
}

struct PresserReaction {
    float maximum_normal_impulse = 0.0f;
    uint32_t particle_rows = 0u;
};

PresserReaction ReadPresserReaction(nk::World& w, std::vector<nk::NkRow>& rows,
                                    std::vector<float>& lambda) {
    const uint32_t count = w.GetModel().capacities.max_rows_per_env;
    rows.resize(count);
    lambda.resize(count);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Urows, rows.data(),
                                          rows.size() * sizeof(nk::NkRow)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Lambda, lambda.data(),
                                          lambda.size() * sizeof(float)));
    PresserReaction reaction;
    for (uint32_t i = 0u; i < count; ++i) {
        const nk::NkRow& row = rows[i];
        if (!(row.flags & nk::nk_row_flags::kActive)) continue;
        const bool a_particle = row.a.kind == nk::kNkSideParticle;
        const bool b_particle = row.b.kind == nk::kNkSideParticle;
        if (!(a_particle || b_particle)) continue;
        EXPECT_FALSE(a_particle && b_particle) << "row has two particle sides";
        const nk::NkRowSide& body = a_particle ? row.b : row.a;
        if (body.kind != nk::kNkSideRigid || body.index != kPresserBody) continue;
        ++reaction.particle_rows;
        if (row.flags & nk::nk_row_flags::kBlockNormal)
            reaction.maximum_normal_impulse =
                std::max(reaction.maximum_normal_impulse, lambda[i]);
    }
    return reaction;
}

// Min particle z over the cloth whose xy lies within reach of the presser sphere
// (the pressed pocket), so the metric tracks the dip, not the wider drape.
float MinZUnderPresser(const std::vector<Vec3>& pos, const Vec3& presser_xy) {
    float min_z = 1.0e9f;
    const float reach = kPresserRadius + 4.0f * kSpacing;
    for (const Vec3& p : pos) {
        const float dx = p.x - presser_xy.x, dy = p.y - presser_xy.y;
        if (dx * dx + dy * dy <= reach * reach) min_z = std::min(min_z, p.z);
    }
    return min_z;
}

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(),
                              P * sizeof(Vec3));
}

// Per-body byte offsets into the env-major Data fields (DownloadField/UploadField
// take a BYTE offset; the presser is body row kPresserBody in env 0).
constexpr uint64_t kPoseOff = static_cast<uint64_t>(kPresserBody) * sizeof(Transform);
constexpr uint64_t kVec3Off = static_cast<uint64_t>(kPresserBody) * sizeof(Vec3);
constexpr uint64_t kFloatOff = static_cast<uint64_t>(kPresserBody) * sizeof(float);

// Kinematic surface velocity matches the uploaded displacement over this timestep.
void DrivePresser(nk::World& w, const Vec3& target) {
    Transform previous;
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose, &previous,
                                          sizeof(Transform), kPoseOff));
    Transform tf = Transform::Identity();
    tf.position = target;
    const Vec3 velocity = (target - previous.position) / Cfg().dt;
    const Vec3 zero = Vec3::Zero();
    ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff));
    ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &velocity, sizeof(Vec3), kVec3Off));
    ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off));
}

}  // namespace

// A kinematic presser indents the settled drape, then lifts to permit recovery.
TEST(ClothPresserTwoWay, ClothDipsUnderPresserThenRecovers) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const Vec3 presser_xy{0.0f, 0.0f, 0.0f};
    const float top = 0.40f;                 // parked well above the cloth.
    nk::Model m = BuildScene(Vec3{presser_xy.x, presser_xy.y, top}, 0.0f);
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    // SETTLE: presser parked high, cloth drapes over the ridge.
    for (uint32_t s = 0; s < kSettleSteps; ++s) { DrivePresser(w, Vec3{0, 0, top}); EXPECT_TRUE(w.Step().AllOk()); }
    std::vector<Vec3> settled;
    DownloadParticles(w, &settled);
    const float rest_min_z = MinZUnderPresser(settled, presser_xy);

    const float indentation = 0.025f;
    const float bottom = rest_min_z + 0.5f * kContactDMin + kPresserRadius - indentation;
    for (uint32_t s = 0; s < 200u; ++s) {
        const float t = static_cast<float>(s) / 199.0f;
        DrivePresser(w, Vec3{0, 0, top + (bottom - top) * t});
        EXPECT_TRUE(w.Step().AllOk());
    }
    std::vector<nk::NkRow> rows;
    std::vector<float> lambda;
    float maximum_impulse = 0.0f;
    for (uint32_t s = 0; s < 150u; ++s) {
        DrivePresser(w, Vec3{0, 0, bottom});
        EXPECT_TRUE(w.Step().AllOk());
        maximum_impulse = std::max(maximum_impulse,
            ReadPresserReaction(w, rows, lambda).maximum_normal_impulse);
    }
    EXPECT_GT(maximum_impulse, 0.0f) << "the prescribed motion applied no contact load";
    std::vector<Vec3> pressed;
    DownloadParticles(w, &pressed);
    const float pressed_min_z = MinZUnderPresser(pressed, presser_xy);

    // RECOVER: lift the presser away over ~200 steps, let the cloth relax ~250.
    for (uint32_t s = 0; s < 200u; ++s) {
        const float t = static_cast<float>(s) / 199.0f;
        DrivePresser(w, Vec3{0, 0, bottom + (top - bottom) * t});
        EXPECT_TRUE(w.Step().AllOk());
    }
    for (uint32_t s = 0; s < 250u; ++s) { DrivePresser(w, Vec3{0, 0, top}); EXPECT_TRUE(w.Step().AllOk()); }
    std::vector<Vec3> recovered;
    DownloadParticles(w, &recovered);
    const float recovered_min_z = MinZUnderPresser(recovered, presser_xy);

    std::fprintf(stderr,
                 "[cloth-presser] rest_min_z=%.4f pressed_min_z=%.4f "
                 "recovered_min_z=%.4f (dip=%.4f recovery=%.4f) impulse=%.8g\n",
                 rest_min_z, pressed_min_z, recovered_min_z,
                 rest_min_z - pressed_min_z, recovered_min_z - pressed_min_z,
                 maximum_impulse);

    ASSERT_TRUE(std::isfinite(rest_min_z) && std::isfinite(pressed_min_z) &&
                std::isfinite(recovered_min_z));
    // (a) the cloth dipped measurably under the presser.
    EXPECT_LT(pressed_min_z, rest_min_z - 0.01f)
        << "the presser did not push the cloth down (no dip)";
    // (c) the cloth sprang back toward rest after the presser lifted.
    EXPECT_GT(recovered_min_z, pressed_min_z + 0.01f)
        << "the cloth did not recover after the presser lifted";
}

// A dynamic presser transfers contact impulse to the cloth and rests above a control.
TEST(ClothPresserTwoWay, DynamicPresserIsResistedByCloth) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const Vec3 center{0.0f, 0.0f, 0.0f};
    // A probe wider than a mesh cell is supported by the drape's tension arc.
    const float inv_mass = 1.0f / 0.35f;
    // A gap-fitting presser (radius < half the ridge gap): WITHOUT the cloth it falls
    // clean through the gap to the ground; WITH the cloth it is caught by the drape.
    const float resist_radius = 0.038f;

    // Settle the presser away from the cloth, then drop it onto the drape.
    // The control moves the cloth below reach and measures the unopposed fall.
    auto drop_presser = [&](bool cloth_present, float* out_lambda,
                            uint32_t* out_particle_rows) -> float {
        nk::Model m = BuildScene(Vec3{2.0f, 2.0f, resist_radius}, inv_mass,
                                 resist_radius);
        if (!cloth_present)
            for (Vec3& p : m.particles.initial_pos) p.z -= 10.0f;  // cloth out of reach.
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return 1e9f;
        // Settle the cloth while the presser falls to rest far away (x=2).
        for (uint32_t s = 0; s < kSettleSteps; ++s) EXPECT_TRUE(w.Step().AllOk());
        const float drape_top = [&] {
            std::vector<Vec3> s; DownloadParticles(w, &s);
            return MinZUnderPresser(s, center);
        }();
        const float release_z = (cloth_present ? drape_top : kRidgeTopZ) +
                                resist_radius + 0.02f;
        // Teleport the (still dynamic) presser above the center, at rest, and drop it.
        Transform tf = Transform::Identity();
        tf.position = Vec3{center.x, center.y, release_z};
        const Vec3 zero = Vec3::Zero();
        nk::Data& d = w.GetData();
        d.UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
        d.UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
        d.UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);

        std::vector<nk::NkRow> urows;
        std::vector<float> lambda;
        float max_lambda = 0.0f;
        uint32_t particle_rows = 0u;
        for (uint32_t s = 0; s < 300u; ++s) {
            EXPECT_TRUE(w.Step().AllOk());
            const PresserReaction reaction = ReadPresserReaction(w, urows, lambda);
            max_lambda = std::max(max_lambda, reaction.maximum_normal_impulse);
            particle_rows += reaction.particle_rows;
        }
        Transform got = Transform::Identity();
        d.DownloadField(nk::FieldId::BodyPose, &got, sizeof(Transform), kPoseOff);
        if (cloth_present) {
            std::vector<Vec3> ppos;
            DownloadParticles(w, &ppos);
            float min_under = 1e9f;
            for (const Vec3& p : ppos) {
                const float dx = p.x - center.x, dy = p.y - center.y;
                if (dx * dx + dy * dy < resist_radius * resist_radius)
                    min_under = std::min(min_under, p.z);
            }
            std::fprintf(stderr,
                         "[cloth-presser] final presser z=%.4f min particle z under "
                         "presser=%.4f particles=%zu\n",
                         got.position.z, min_under, ppos.size());
        }
        if (out_lambda) *out_lambda = max_lambda;
        if (out_particle_rows) *out_particle_rows = particle_rows;
        return got.position.z;
    };

    float body_particle_lambda = 0.0f;
    uint32_t particle_rows_seen = 0u;
    const float presser_z_with_cloth =
        drop_presser(true, &body_particle_lambda, &particle_rows_seen);
    const float presser_z_no_cloth = drop_presser(false, nullptr, nullptr);

    std::fprintf(stderr,
                 "[cloth-presser] presser_z_with_cloth=%.4f presser_z_no_cloth=%.4f "
                 "body_particle_lambda=%.6f particle_rows_seen=%u\n",
                 presser_z_with_cloth, presser_z_no_cloth, body_particle_lambda,
                 particle_rows_seen);

    // The reaction direction: a body<->particle normal row carried a real push-out.
    EXPECT_GT(particle_rows_seen, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_GT(body_particle_lambda, 0.0f)
        << "the cloth produced no body-side reaction impulse on the presser";
    // Cloth support holds the presser above the control resting on the ground.
    EXPECT_GT(presser_z_with_cloth, presser_z_no_cloth + 0.02f)
        << "the cloth did not resist the presser (it fell as far as the no-cloth "
           "control)";
}
