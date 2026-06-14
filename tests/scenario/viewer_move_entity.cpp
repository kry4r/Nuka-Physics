// ---------------------------------------------------------------------------
// viewer_move_entity -- M11 VIEW-4 + VIEW-6 HOST gates (runnable in build-cuda128,
// NO Vulkan). The interactive viewer's drag-to-move and StepOnce paths are pure
// command_queue + nk::Data + the frame-loop systems, so their CORRECTNESS is
// verifiable headlessly here (the windowed interaction itself is owner-verified).
//
// VIEW-4 (drag -> LIVE nk Data): build a body-only nk::World (one movable free
// body) + a RenderWorld whose single instance binds that body (PoseSource::Body
// row 0), then push a MoveEntity through Simulation's CommandQueue and drive ONE
// FramePublish. Assert:
//   (a) DownloadField(BodyPose) == the requested transform (the general path
//       command_queue -> ApplyMoveEntity -> Data::UploadField landed), and
//   (b) BodyLinearVelocity / BodyAngularVelocity were ZEROED (OD-6 place-here).
// The handler resolves entity -> Data row via RenderInstance::pose_source (the
// SAME map the publisher reads) -- no SceneIR / .nks touch (R13).
//
// VIEW-6 (StepOnce same-frame): a queued ViewerControl::StepOnce is honored in the
// SAME FramePublish it is drained in (no one-frame lag). Assert exactly ONE step
// ran (a free body picks up -g*dt of vz), and a no-command FramePublish(do_step=
// false) runs ZERO steps (vz unchanged).
//
// HOST-ONLY / zero-CUDA-token (the systems/Data path is behind the phi v2 vtable).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "render/render_world.hpp"
#include "runtime/app/command_queue.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/systems.hpp"

#include <cmath>
#include <cstdint>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace app = nuka::runtime::app;
namespace render = nuka::render;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kDt = 1.0f / 240.0f;
constexpr float kGravityZ = -9.81f;

struct Ctx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Ctx GetCtx() {
    static Ctx c = [] {
        Ctx r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

nk::Pipeline::SolverConfig Config() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = kGravityZ;
    return cfg;
}

// A model with ONE movable free rigid body (no articulation, no particles, no
// contact). Enough to exercise BodyPose / BodyLinearVelocity Data writes + a step.
nk::Model BuildBodyModel() {
    nk::Model model;
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.bodies_per_env = 1;

    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, 1.0f};
    bi.inv_mass = 1.0f / 2.0f;  // 2 kg free body
    const float ii = 1.0f / 0.2f;
    bi.inv_inertia = Vec3{ii, ii, ii};
    model.body_init.push_back(bi);
    return model;
}

// A RenderWorld with a single instance bound to body row 0 (PoseSource::Body), so
// ApplyMoveEntity resolves entity -> the cup-like body row exactly as the cook's
// SceneMap would.
render::RenderWorld BuildBodyRenderWorld(nuka::scene::EntityId e) {
    render::RenderWorld rw;
    render::RenderInstance inst;
    inst.entity = e;
    inst.pose_source.kind = render::PoseSource::Kind::Body;
    inst.pose_source.row = 0u;
    inst.world_xform = Transform::Identity();
    rw.instances.push_back(inst);
    return rw;
}

bool TfNear(const Transform& a, const Transform& b, float eps) {
    return std::fabs(a.position.x - b.position.x) < eps &&
           std::fabs(a.position.y - b.position.y) < eps &&
           std::fabs(a.position.z - b.position.z) < eps &&
           std::fabs(a.rotation.w - b.rotation.w) < eps &&
           std::fabs(a.rotation.x - b.rotation.x) < eps &&
           std::fabs(a.rotation.y - b.rotation.y) < eps &&
           std::fabs(a.rotation.z - b.rotation.z) < eps;
}

}  // namespace

// VIEW-4: MoveEntity through the command queue -> LIVE BodyPose; velocity zeroed.
TEST(ViewerMoveEntity, DragWritesLiveBodyPoseAndZeroesVelocity) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Ctx c = GetCtx();

    const nuka::scene::EntityId e{7u, 0u};
    nk::World world(BuildBodyModel(), 1u, c.dev, c.backend, Config());
    ASSERT_TRUE(world.Ready());

    app::HostDownloadPublisher publisher;
    app::Simulation sim(world, publisher, BuildBodyRenderWorld(e));

    // Seed a non-zero velocity so the OD-6 zeroing is observable.
    const Vec3 seed_v{0.5f, -0.3f, 0.9f};
    world.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &seed_v, sizeof(Vec3), 0);
    world.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &seed_v, sizeof(Vec3), 0);

    // The requested drag target (a teleport with a rotation).
    Transform target;
    target.position = Vec3{1.25f, -0.5f, 2.0f};
    target.rotation = nuka::math::Quat{0.92388f, 0.0f, 0.38268f, 0.0f};  // ~45deg about Y

    sim.Commands().Push(app::Command::MakeMoveEntity(e, target));

    // ONE FramePublish WITHOUT a physics step -> the MoveEntity applies, no drift.
    app::InputIntents intents;
    sim.FramePublish(&intents, /*do_step=*/false);

    EXPECT_EQ(intents.move_entity_count, 1u);
    ASSERT_EQ(intents.move_entities.size(), 1u);

    // (a) BodyPose == requested (the general path landed in Data).
    Transform got;
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyPose, &got, sizeof(Transform), 0));
    EXPECT_TRUE(TfNear(got, target, 1e-5f))
        << "BodyPose " << got.position.x << "," << got.position.y << "," << got.position.z
        << " != requested " << target.position.x << "," << target.position.y << ","
        << target.position.z;

    // (b) velocity zeroed (OD-6).
    Vec3 lin{9, 9, 9}, ang{9, 9, 9};
    world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &lin, sizeof(Vec3), 0);
    world.GetData().DownloadField(nk::FieldId::BodyAngularVelocity, &ang, sizeof(Vec3), 0);
    EXPECT_LT(lin.Length(), 1e-6f) << "linear velocity not zeroed on place";
    EXPECT_LT(ang.Length(), 1e-6f) << "angular velocity not zeroed on place";

    // R13 confirm (by construction): the move went through Data::UploadField only;
    // no SceneIR / nuka_scene_set_local / Save call exists in this path.
}

// VIEW-F3: an OFFSET visual node (cached_visual_local != identity) must round-trip.
// The drag command carries a VISUAL-frame transform (the viewer builds it from
// inst.world_xform == fk * cvl). ApplyMoveEntity must strip cvl before writing the
// PHYSICS-frame pose into BodyPose, so the publisher re-multiplying cvl reproduces
// the requested visual transform EXACTLY. (Pre-fix this double-applied the offset.)
TEST(ViewerMoveEntity, OffsetVisualNodeRoundTrips) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Ctx c = GetCtx();

    const nuka::scene::EntityId e{7u, 0u};
    nk::World world(BuildBodyModel(), 1u, c.dev, c.backend, Config());
    ASSERT_TRUE(world.Ready());

    // An instance whose visual node is OFFSET from the body frame: a 0.1m +x shift
    // plus a 90deg rotation about Z (a non-identity cached_visual_local).
    render::RenderWorld rw;
    render::RenderInstance inst;
    inst.entity = e;
    inst.pose_source.kind = render::PoseSource::Kind::Body;
    inst.pose_source.row = 0u;
    Transform cvl;
    cvl.position = Vec3{0.1f, 0.0f, 0.0f};
    cvl.rotation = nuka::math::Quat{0.70710678f, 0.0f, 0.0f, 0.70710678f};  // 90deg about Z
    inst.cached_visual_local = cvl;
    inst.world_xform = Transform::Identity();
    rw.instances.push_back(inst);

    app::HostDownloadPublisher publisher;
    app::Simulation sim(world, publisher, std::move(rw));

    // The drag target in the VISUAL frame (where the user dropped the visual mesh).
    Transform visual_target;
    visual_target.position = Vec3{1.25f, -0.5f, 2.0f};
    visual_target.rotation = nuka::math::Quat{0.92388f, 0.0f, 0.38268f, 0.0f};  // ~45deg about Y

    sim.Commands().Push(app::Command::MakeMoveEntity(e, visual_target));
    app::InputIntents intents;
    sim.FramePublish(&intents, /*do_step=*/false);
    EXPECT_EQ(intents.move_entity_count, 1u);

    // The PHYSICS pose written must be visual_target * Inverse(cvl).
    Transform expect_physics = visual_target * cvl.Inverse();
    Transform got_physics;
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyPose, &got_physics,
                                              sizeof(Transform), 0));
    EXPECT_TRUE(TfNear(got_physics, expect_physics, 1e-5f))
        << "BodyPose should hold the physics-frame pose (visual * cvl^-1), not the "
           "visual-frame transform";

    // And the publisher's re-compose (physics * cvl) must reproduce the requested
    // VISUAL transform -> no double-applied offset (the FramePublish above already
    // ran the publisher into the RenderWorld; read it back).
    const Transform reconstructed_visual = got_physics * cvl;
    EXPECT_TRUE(TfNear(reconstructed_visual, visual_target, 1e-5f))
        << "round-trip physics->visual must equal the dragged visual target";
    EXPECT_TRUE(TfNear(sim.GetRenderWorld().instances[0].world_xform, visual_target, 1e-5f))
        << "the published world_xform must equal the dragged visual target";
}

// VIEW-4: an UNKNOWN / unbound entity is a clean no-op (general handler).
TEST(ViewerMoveEntity, UnknownEntityIsNoOp) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Ctx c = GetCtx();
    const nuka::scene::EntityId bound{7u, 0u};
    nk::World world(BuildBodyModel(), 1u, c.dev, c.backend, Config());
    ASSERT_TRUE(world.Ready());

    const render::RenderWorld rw = BuildBodyRenderWorld(bound);
    Transform t; t.position = Vec3{3, 3, 3};
    // A different entity that no instance owns -> false (no write).
    EXPECT_FALSE(app::ApplyMoveEntity(world, rw, 0u,
                                      app::MoveEntity{nuka::scene::EntityId{99u, 0u}, t}));
}

// VIEW-6: a QUEUED StepOnce runs exactly ONE step in the SAME FramePublish.
TEST(ViewerMoveEntity, StepOnceAdvancesExactlyOneStepSameFrame) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Ctx c = GetCtx();
    nk::World world(BuildBodyModel(), 1u, c.dev, c.backend, Config());
    ASSERT_TRUE(world.Ready());
    app::HostDownloadPublisher publisher;
    app::Simulation sim(world, publisher, BuildBodyRenderWorld(nuka::scene::EntityId{7u, 0u}));

    auto vz = [&]() -> float {
        Vec3 v{0, 0, 0};
        world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &v, sizeof(Vec3), 0);
        return v.z;
    };

    // Baseline: free body at rest, vz == 0.
    EXPECT_NEAR(vz(), 0.0f, 1e-6f);

    // Queue StepOnce; FramePublish with do_step=false. The drained StepOnce must be
    // honored SAME-FRAME (VIEW-6 fix), so exactly ONE step runs: vz ~= g*dt.
    sim.Commands().Push(app::Command::MakeViewerControl(app::ViewerControl::Action::StepOnce));
    app::InputIntents in1;
    sim.FramePublish(&in1, /*do_step=*/false);
    EXPECT_TRUE(in1.step_once);
    const float after_one = vz();
    EXPECT_NEAR(after_one, kGravityZ * kDt, 1e-4f)
        << "queued StepOnce did not advance exactly one step same-frame";

    // No command, do_step=false -> ZERO steps: vz unchanged.
    app::InputIntents in2;
    sim.FramePublish(&in2, /*do_step=*/false);
    EXPECT_FALSE(in2.step_once);
    EXPECT_NEAR(vz(), after_one, 1e-6f) << "a no-command paused frame advanced the sim";
}
