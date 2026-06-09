// ---------------------------------------------------------------------------
// v0.8 P2.4a -- BatchedUnifiedWorld env-steps/sec PERF GATE (a hypothesis test).
//
// This is NOT a correctness gate (those 16 live in test_batched_unified_world.cpp and
// stay byte-exact; this TU links the SAME batched_unified_world.cpp, so any accidental
// physics change there shows up over there). It is a COMMITTED throughput harness +
// per-stage attribution that TESTS the roadmap §7 hypothesis:
//   (a) the per-env articulation SYNC STORM (pose_download + artic_download + crba_minv
//       + chain_jacobian) dominates the step -> the next increment (device-resident
//       N-env articulation) is the big win; AND
//   (b) the single-block row_solver is negligible -> it can be deferred to last.
//
// We sweep N in {1,8,32,64,256,1024}, run K=200 timed Step()s after a 10-step warm-up,
// and print env_steps_per_sec + ms_per_step at each N. If host orchestration is O(N)
// (the current path -- a per-env host loop), env_steps_per_sec is ~N-independent.
//
// At N=32 we dump the full PER-STAGE breakdown from world.Perf() (each tag's
// count*mean_us total + its % of the SUMMED tag time), plus the two roll-ups the
// verdict reads: ARTIC_SYNC_STORM (the 4 articulation host round-trips) and SOLVER
// (row_solver). A steady_clock Σ-check (Σtags vs measured wall) guards against a
// double-counted / nested timer corrupting the breakdown.
//
// The harness is GRASP-scene-based (the proven host-orchestrated grasp path is what
// carries the articulation sync storm). Asset-gated: SKIP if the cup hull is absent.
// Run from the repo root (the cup loads via a relative asset path).
//
// ADDITIVE: a NEW translation unit, a NEW test target. Modifies no production
// stepper/solver/golden and no existing test. The grasp scene builders below are
// REPLICATED from test_batched_unified_world.cpp's anon namespace (test-side
// scaffolding, byte-for-byte) so this TU does not depend on that test's internals.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"
#include "import/usd_importer.hpp"                          // LoadUsd (cup hull)
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"  // ArticulationDofCount
#include "runtime/articulation/articulation_state.hpp"     // BuildArticulationHostState
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"  // CoResident* / GraspConfig
#include "runtime/rigid/body_state.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"                                 // CookScene

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace coresident = nuka::runtime::coresident;
namespace articulation = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::coresident::CoResidentFingertip;
using nuka::runtime::coresident::GraspConfig;
using nuka::runtime::rigid::BodyState;

constexpr float kGraspGravityZ = -9.81f;
constexpr float kGraspDt = 1.0f / 240.0f;
constexpr float kCupMass = 0.2f;
constexpr float kCupInvMass = 1.0f / kCupMass;

const std::string kCupModelUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";
bool GraspCupAvailable() { return std::filesystem::exists(kCupModelUsda); }

// ---------------------------------------------------------------------------
// Grasp scene builders -- REPLICATED byte-for-byte from
// tests/coresident/test_batched_unified_world.cpp (test-side scaffolding). They build
// the SAME fixed-base 2-finger gripper + C7a cup hull the grasp gates use, so this
// perf harness exercises the identical host-orchestrated grasp Step() path.
// ---------------------------------------------------------------------------

struct GraspCupHull {
    std::vector<float> verts;
    Vec3 lo{}, hi{};
    uint32_t vcount = 0u;
};
GraspCupHull LoadGraspCupHull() {
    auto scene = nuka::import::LoadUsd(kCupModelUsda);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;
    }
    const auto blob = nuka::scene::CookScene(scene);
    GraspCupHull out;
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
        out.lo = Vec3{std::min(out.lo.x, v.x), std::min(out.lo.y, v.y),
                      std::min(out.lo.z, v.z)};
        out.hi = Vec3{std::max(out.hi.x, v.x), std::max(out.hi.y, v.y),
                      std::max(out.hi.z, v.z)};
    }
    return out;
}

struct GraspGripper {
    articulation::ArticulationHostState host;
    uint32_t finger_link[2];
    Vec3     fingertip_local[2];
    float    fingertip_radius = 0.008f;
};
GraspGripper BuildGraspGripper(const Vec3& base_pos, float cup_half_x,
                               float fingertip_radius) {
    constexpr uint32_t kInvalidLink = ~0u;
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u};
    topo.parent_links = {kInvalidLink, 0u, 0u};
    topo.joint_types = {articulation::ArticulationJointType::Fixed,
                        articulation::ArticulationJointType::Prismatic,
                        articulation::ArticulationJointType::Prismatic};
    topo.joint_axes = {Vec3::UnitX(), Vec3{-1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}};
    topo.parent_frames = {Transform::Identity(), Transform::Identity(),
                          Transform::Identity()};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity()};
    Transform base_lp = Transform::Identity();
    base_lp.position = base_pos;
    topo.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    topo.inertial_frames = {Transform::Identity(), Transform::Identity(),
                            Transform::Identity()};
    topo.initial_positions = {0.0f, 0.0f, 0.0f};
    topo.joint_dampings = {0.0f, 0.0f, 0.0f};
    topo.joint_armatures = {0.0f, 0.0f, 0.0f};
    topo.masses = {1.0f, 0.05f, 0.05f};
    topo.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                     Vec3{1e-4f, 1e-4f, 1e-4f}};

    nuka::scene::CookedBodyTable bodies;
    Transform base_pose = Transform::Identity();
    base_pose.position = base_pos;
    bodies.poses = {base_pose, Transform::Identity(), Transform::Identity()};
    bodies.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    bodies.inertial_frames = {Transform::Identity(), Transform::Identity(),
                              Transform::Identity()};
    bodies.masses = {1.0f, 0.05f, 0.05f};
    bodies.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                       Vec3{1e-4f, 1e-4f, 1e-4f}};
    bodies.is_static = {0u, 0u, 0u};

    GraspGripper g;
    g.host = articulation::BuildArticulationHostState({topo}, bodies);
    g.finger_link[0] = 1u;
    g.finger_link[1] = 2u;
    g.fingertip_radius = fingertip_radius;
    const float kPenetration = 0.0015f;  // 1.5 mm shallow pre-pose -> contact live @ step 0.
    const float reach = cup_half_x + fingertip_radius - kPenetration;
    g.fingertip_local[0] = Vec3{reach, 0.0f, 0.0f};
    g.fingertip_local[1] = Vec3{-reach, 0.0f, 0.0f};
    return g;
}

struct GraspSceneBundle {
    GraspGripper gripper;
    GraspConfig  config;
};
GraspSceneBundle BuildGraspSceneBundle(const GraspCupHull& hull, float grip_force,
                                       float mu) {
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    const Vec3 cup_local_center = (hull.hi + hull.lo) * 0.5f;
    const Vec3 cup_center{0.0f, 0.0f, 0.20f};
    const Vec3 base_pos{0.0f, 0.0f, 0.30f};

    GraspSceneBundle gs;
    gs.gripper = BuildGraspGripper(base_pos, half.x, 0.008f);
    const float drop = cup_center.z - base_pos.z;
    gs.gripper.fingertip_local[0].z += drop;
    gs.gripper.fingertip_local[1].z += drop;

    CoResidentFingertip ft0;
    ft0.link = gs.gripper.finger_link[0];
    ft0.broadphase_handle = gs.gripper.finger_link[0];
    ft0.local_offset = gs.gripper.fingertip_local[0];
    ft0.radius = gs.gripper.fingertip_radius;
    CoResidentFingertip ft1;
    ft1.link = gs.gripper.finger_link[1];
    ft1.broadphase_handle = gs.gripper.finger_link[1];
    ft1.local_offset = gs.gripper.fingertip_local[1];
    ft1.radius = gs.gripper.fingertip_radius;
    gs.config.fingertips = {ft0, ft1};

    gs.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        gs.config.cup.hull_verts[i * 3u + 0u] -= cup_local_center.x;
        gs.config.cup.hull_verts[i * 3u + 1u] -= cup_local_center.y;
        gs.config.cup.hull_verts[i * 3u + 2u] -= cup_local_center.z;
    }
    gs.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = kCupInvMass;
    const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = cup_center;
    cup.orientation = Quat::Identity();
    cup.linear_velocity = Vec3::Zero();
    cup.angular_velocity = Vec3::Zero();
    gs.config.cup_state = cup;

    const uint32_t link_count = gs.gripper.host.TotalLinkCount();
    gs.config.grip_torque.assign(link_count, 0.0f);
    gs.config.grip_torque[gs.gripper.finger_link[0]] = grip_force;
    gs.config.grip_torque[gs.gripper.finger_link[1]] = grip_force;
    gs.config.drive_force_limits.assign(link_count, 0.0f);
    gs.config.friction_mu = mu;
    gs.config.condim = 3u;
    return gs;
}

coresident::BatchedSceneTemplate MakeGraspTemplate(const GraspSceneBundle& gs) {
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {gs.config.cup_state};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = gs.gripper.host;
    tmpl.fingertips = gs.config.fingertips;
    tmpl.cup = gs.config.cup;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque = gs.config.grip_torque;
    tmpl.drive_force_limits = gs.config.drive_force_limits;
    tmpl.friction_mu = gs.config.friction_mu;
    tmpl.condim = gs.config.condim;
    return tmpl;
}

// ---------------------------------------------------------------------------
// Run the timed harness for one N: warm up kWarm steps (reset perf after), then run
// K timed Step()s on a steady wall clock. Returns env_steps_per_sec.
// ---------------------------------------------------------------------------
constexpr uint32_t kWarm = 10u;
constexpr uint32_t kK = 200u;  // the N=32 breakdown's step count (good sample stats).

// The host orchestration is O(N) -- env_steps_per_sec is ~N-independent (the hypothesis),
// so per-step wall time grows ~linearly with N. A fixed K=200 would make N=1024 take
// ~20 min. We therefore use an ADAPTIVE K per N: target ~kTargetEnvSteps env-steps total
// (clamped to [kKMin, kK]) so each N's wall time is bounded (~tens of seconds) while still
// timing enough steps for a stable env-steps/sec. env_steps_per_sec is throughput -- K only
// sets how many samples we average, not the measured rate. N=32 keeps the full kK for the
// breakdown's statistics (kTargetEnvSteps/32 = 200 = kK, so N<=32 already runs full K).
constexpr uint32_t kTargetEnvSteps = 6400u;
constexpr uint32_t kKMin = 20u;

uint32_t StepsForN(uint32_t n) {
    const uint32_t k = (n == 0u) ? kK : (kTargetEnvSteps / n);
    return std::clamp(k, kKMin, kK);
}

double RunHarness(const nuka::phi::DeviceContext& context,
                  const coresident::BatchedSceneTemplate& tmpl, uint32_t n,
                  uint32_t k, double* ms_per_step_out) {
    coresident::BatchedUnifiedWorld world(context, tmpl, n, kGraspGravityZ, kGraspDt);
    for (uint32_t s = 0u; s < kWarm; ++s) world.Step();
    world.Perf().Reset();  // discard warm-up samples.

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0u; s < k; ++s) world.Step();
    const auto t1 = std::chrono::steady_clock::now();

    const double elapsed_sec =
        std::chrono::duration<double>(t1 - t0).count();
    const double env_steps_per_sec =
        static_cast<double>(n) * k / elapsed_sec;
    *ms_per_step_out = 1.0e3 * elapsed_sec / k;
    return env_steps_per_sec;
}

// The 9 canonical stage tags (the order they are printed in the breakdown).
const char* const kStageTags[] = {
    "pose_download", "artic_download", "crba_minv",   "chain_jacobian", "aba_integrate",
    "narrowphase",   "row_assembly",   "row_solver",  "scatter_integrate",
};

// Per-tag total wall time (us) = count * mean_us (PerfRecorder exposes no sum; mean is
// exactly sum/count, so this is the exact total). Zero if the tag never recorded.
double TagTotalUs(const nuka::core::perf::PerfRecorder& perf, const char* tag) {
    const auto st = perf.Stats(tag);
    return static_cast<double>(st.count) * st.mean_us;
}

}  // namespace

// ---------------------------------------------------------------------------
// THE PERF GATE: env-steps/sec sweep + N=32 stage breakdown + the verdict roll-ups.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldPerf, EnvStepsPerSecondAndStageBreakdown) {
    if (!GraspCupAvailable())
        GTEST_SKIP() << "newton-assets cup not present -- perf gate skipped (it drives "
                        "the grasp scene; the cup hull asset is fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const GraspCupHull hull = LoadGraspCupHull();
    ASSERT_GT(hull.vcount, 0u) << "cup hull cooked empty -- perf harness has no contact";

    // The grasp scene template (grip 8 N, mu 0.8 -- the grasp gates' nominal grip). All
    // envs are constructed from the SAME replicated template, so every env establishes a
    // finger<->cup contact -> the breakdown reflects the FULL-LOAD articulation path (no
    // no-contact envs skewing the storm share down).
    const GraspSceneBundle gs = BuildGraspSceneBundle(hull, /*grip_force=*/8.0f,
                                                      /*mu=*/0.8f);
    const coresident::BatchedSceneTemplate tmpl = MakeGraspTemplate(gs);

    // ----- (1) the env-steps/sec sweep -----------------------------------------------
    const uint32_t kNs[] = {1u, 8u, 32u, 64u, 256u, 1024u};
    double measured_n32_eps = 0.0;
    for (uint32_t n : kNs) {
        const uint32_t k = StepsForN(n);
        double ms_per_step = 0.0;
        const double eps = RunHarness(context, tmpl, n, k, &ms_per_step);
        std::printf("[PERF] N=%u K=%u env_steps_per_sec=%.1f ms_per_step=%.3f\n", n, k,
                    eps, ms_per_step);
        std::fflush(stdout);
        if (n == 32u) measured_n32_eps = eps;
    }

    // ----- (2) the N=32 PER-STAGE breakdown (a fresh, isolated N=32 run) --------------
    // Re-run N=32 standalone so the recorder holds ONLY this run's samples (the sweep's
    // N=32 world was destroyed; build a clean one and time it the same way).
    coresident::BatchedUnifiedWorld w32(context, tmpl, 32u, kGraspGravityZ, kGraspDt);
    for (uint32_t s = 0u; s < kWarm; ++s) w32.Step();
    w32.Perf().Reset();
    const auto wall0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0u; s < kK; ++s) w32.Step();
    const auto wall1 = std::chrono::steady_clock::now();
    const double wall_total_ms =
        std::chrono::duration<double, std::milli>(wall1 - wall0).count();

    const auto& perf = w32.Perf();
    double sum_tags_ms = 0.0;
    for (const char* tag : kStageTags) sum_tags_ms += TagTotalUs(perf, tag) * 1.0e-3;

    // Denominator for both per-tag % and the roll-ups = Σ(the 9 tag totals), so the
    // shares sum to 100% by construction (the verdict is then robust to any untimed gap).
    const double denom_ms = sum_tags_ms > 0.0 ? sum_tags_ms : 1.0;
    for (const char* tag : kStageTags) {
        const double tag_ms = TagTotalUs(perf, tag) * 1.0e-3;
        std::printf("[PERF-BREAKDOWN N=32] %s=%.3f (%.1f%%)\n", tag, tag_ms,
                    100.0 * tag_ms / denom_ms);
    }

    // The two verdict roll-ups.
    const double storm_ms = (TagTotalUs(perf, "pose_download") +
                             TagTotalUs(perf, "artic_download") +
                             TagTotalUs(perf, "crba_minv") +
                             TagTotalUs(perf, "chain_jacobian")) *
                            1.0e-3;
    const double solver_ms = TagTotalUs(perf, "row_solver") * 1.0e-3;
    const double storm_pct = 100.0 * storm_ms / denom_ms;
    const double solver_pct = 100.0 * solver_ms / denom_ms;
    std::printf("[PERF-BREAKDOWN N=32] ARTIC_SYNC_STORM=%.3f (%.1f%%)\n", storm_ms,
                storm_pct);
    std::printf("[PERF-BREAKDOWN N=32] SOLVER=%.3f (%.1f%%)\n", solver_ms, solver_pct);

    // Σ-check (the advisor's nested-timer guard): Σtags must be <= wall and close. If
    // Σtags > wall, a timer was double-counted (nested) and the breakdown is corrupt.
    std::printf("[PERF-BREAKDOWN N=32] SUM_TAGS=%.3f WALL_TOTAL=%.3f (sum/wall=%.1f%%)\n",
                sum_tags_ms, wall_total_ms,
                wall_total_ms > 0.0 ? 100.0 * sum_tags_ms / wall_total_ms : 0.0);
    std::fflush(stdout);
    EXPECT_LE(sum_tags_ms, wall_total_ms * 1.05)
        << "Σ(stage tags) exceeds the measured wall time -> a timer is nested / "
           "double-counted; the breakdown percentages cannot be trusted";

    // The verdict (printed plainly for the report).
    std::printf("[PERF-VERDICT N=32] ARTIC_SYNC_STORM=%.1f%% vs SOLVER(row_solver)=%.1f%% "
                "-> %s dominates\n",
                storm_pct, solver_pct,
                storm_pct >= solver_pct ? "ARTIC_SYNC_STORM" : "SOLVER");
    std::fflush(stdout);

    // ----- (3) ONE loose regression guard -------------------------------------------
    // Perf is noisy run-to-run on a shared box (the absolute throughput varies with host
    // load), so the floor is GENEROUS: 0.3x the measured N=32 env-steps/sec observed when
    // this gate was baked. The PRINTED numbers are the deliverable; this assert only
    // catches a GROSS regression (a 3x+ slowdown, e.g. an accidental extra per-env device
    // round-trip in the host orchestration loop), not noise.
    //
    // BASELINE: ~183 env-steps/sec measured at bake time on the build-cuda128 box (N=32,
    // K=200), consistent with the roadmap's "~180 env-steps/sec independent of N" finding
    // -- the whole point of this gate is that the current path is O(N) host orchestration.
    // Floor = 0.3 x baseline = 54 env-steps/sec. Update the constant if the baseline moves
    // (e.g. after the device-resident N-env articulation increment lifts throughput).
    constexpr double kN32BaselineEps = 180.0;  // see comment above; floor is 0.3x this.
    EXPECT_GT(measured_n32_eps, 0.3 * kN32BaselineEps)
        << "N=32 throughput (" << measured_n32_eps
        << " env-steps/sec) fell below 0.3x the baseline (" << kN32BaselineEps
        << ") -- a gross perf regression, not noise";
}
