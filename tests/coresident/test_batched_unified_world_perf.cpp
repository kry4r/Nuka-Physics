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

#include "constraint/row.hpp"                                // Row / RowJacobian6 / kInvalidBodyIndex
#include "constraint/row_buffers.hpp"                        // RowBuffers (synthetic coloring input)
#include "core/perf/perf_recorder.hpp"
#include "import/usd_importer.hpp"                          // LoadUsd (cup hull)
#include "solver/gpu/row_scheduler.hpp"                      // BuildRowColorPartitions / ValidateNoSharedBodiesPerColor
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

// A1: the per-step RL cost is SetActions + Step + ExportObsState. The timed region BRACKETS
// all three so env_steps_per_sec is the FULL RL per-step throughput (GATE-5). We build one
// env-major action vector once (the template grip torque on every finger DOF -> the SAME drive
// as the default, so the dynamics are unchanged from the pre-A1 harness; SetActions exercises
// the upload path each step) and reuse an ObsStateBatch (sized once by the first export).
double RunHarness(const nuka::phi::DeviceContext& context,
                  const coresident::BatchedSceneTemplate& tmpl, uint32_t n,
                  uint32_t k, double* ms_per_step_out) {
    coresident::BatchedUnifiedWorld world(context, tmpl, n, kGraspGravityZ, kGraspDt);
    // A fixed per-env per-DOF action == the scene grip torque (8 N on each finger DOF), so the
    // timed rollout's physics matches the constant-grip gates while still paying SetActions.
    const uint32_t d = world.ActionDim();
    std::vector<float> actions(static_cast<size_t>(n) * d, 8.0f);
    coresident::ObsStateBatch obs;
    for (uint32_t s = 0u; s < kWarm; ++s) {
        world.SetActions(actions.data(), actions.size());
        world.Step();
        world.ExportObsState(obs);
    }
    world.Perf().Reset();  // discard warm-up samples.

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0u; s < k; ++s) {
        world.SetActions(actions.data(), actions.size());  // per-step action injection.
        world.Step();                                       // advance (applies the actions).
        world.ExportObsState(obs);                          // the per-step RL obs readout.
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double elapsed_sec =
        std::chrono::duration<double>(t1 - t0).count();
    const double env_steps_per_sec =
        static_cast<double>(n) * k / elapsed_sec;
    *ms_per_step_out = 1.0e3 * elapsed_sec / k;
    return env_steps_per_sec;
}

// The canonical stage tags (the order they are printed in the breakdown). A1 adds
// `obs_export` (the batched RL readout) so the breakdown shows its per-step cost + scaling.
const char* const kStageTags[] = {
    "pose_download", "artic_download", "crba_minv",   "chain_jacobian", "aba_integrate",
    "narrowphase",   "row_assembly",   "row_solver",  "scatter_integrate", "obs_export",
};

// Per-tag total wall time (us) = count * mean_us (PerfRecorder exposes no sum; mean is
// exactly sum/count, so this is the exact total). Zero if the tag never recorded.
double TagTotalUs(const nuka::core::perf::PerfRecorder& perf, const char* tag) {
    const auto st = perf.Stats(tag);
    return static_cast<double>(st.count) * st.mean_us;
}

// ---------------------------------------------------------------------------
// (A) Per-stage breakdown at an arbitrary N. Builds a CLEAN world, warms up, resets the
// recorder, runs `k` timed Step()s, then prints the SAME PERF-BREAKDOWN block the N=32
// path prints (each tag's total ms + its % of Σtags, the two roll-ups, and the Σ-check).
// Returns the PER-STEP row_solver tag ms (the tag total / k). ★ PER-STEP is essential for
// the (D) verdict: the printed tag TOTALS are summed over k Step()s, and k differs per N
// (200/20/10), so a TOTAL-vs-TOTAL comparison against the per-CALL coloring time would be
// off by exactly the factor k. There is exactly one UnifiedSolve (hence one
// BuildRowColorPartitions) per Step, so per-step row_solver is the right denominator for
// the per-call coloring time. The in-function breakdown %s are within-run ratios (k
// cancels) -> unaffected by this division.
// ---------------------------------------------------------------------------
double RunStageBreakdown(const nuka::phi::DeviceContext& context,
                         const coresident::BatchedSceneTemplate& tmpl, uint32_t n,
                         uint32_t k, double* obs_ms_per_step_out = nullptr) {
    coresident::BatchedUnifiedWorld world(context, tmpl, n, kGraspGravityZ, kGraspDt);
    // A1: drive the FULL RL per-step cost (SetActions + Step + ExportObsState), so the
    // breakdown attributes the obs_export tag alongside the dynamics stages.
    const uint32_t d = world.ActionDim();
    std::vector<float> actions(static_cast<size_t>(n) * d, 8.0f);
    coresident::ObsStateBatch obs;
    for (uint32_t s = 0u; s < kWarm; ++s) {
        world.SetActions(actions.data(), actions.size());
        world.Step();
        world.ExportObsState(obs);
    }
    world.Perf().Reset();
    const auto wall0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0u; s < k; ++s) {
        world.SetActions(actions.data(), actions.size());
        world.Step();
        world.ExportObsState(obs);
    }
    const auto wall1 = std::chrono::steady_clock::now();
    const double wall_total_ms =
        std::chrono::duration<double, std::milli>(wall1 - wall0).count();

    const auto& perf = world.Perf();
    double sum_tags_ms = 0.0;
    for (const char* tag : kStageTags) sum_tags_ms += TagTotalUs(perf, tag) * 1.0e-3;
    const double denom_ms = sum_tags_ms > 0.0 ? sum_tags_ms : 1.0;
    for (const char* tag : kStageTags) {
        const double tag_ms = TagTotalUs(perf, tag) * 1.0e-3;
        std::printf("[PERF-BREAKDOWN N=%u K=%u] %s=%.3f (%.1f%%)\n", n, k, tag, tag_ms,
                    100.0 * tag_ms / denom_ms);
    }
    const double storm_ms = (TagTotalUs(perf, "pose_download") +
                             TagTotalUs(perf, "artic_download") +
                             TagTotalUs(perf, "crba_minv") +
                             TagTotalUs(perf, "chain_jacobian")) *
                            1.0e-3;
    const double solver_ms = TagTotalUs(perf, "row_solver") * 1.0e-3;
    std::printf("[PERF-BREAKDOWN N=%u K=%u] ARTIC_SYNC_STORM=%.3f (%.1f%%)\n", n, k,
                storm_ms, 100.0 * storm_ms / denom_ms);
    std::printf("[PERF-BREAKDOWN N=%u K=%u] SOLVER=%.3f (%.1f%%)\n", n, k, solver_ms,
                100.0 * solver_ms / denom_ms);
    std::printf("[PERF-BREAKDOWN N=%u K=%u] SUM_TAGS=%.3f WALL_TOTAL=%.3f (sum/wall=%.1f%%)\n",
                n, k, sum_tags_ms, wall_total_ms,
                wall_total_ms > 0.0 ? 100.0 * sum_tags_ms / wall_total_ms : 0.0);
    std::fflush(stdout);
    EXPECT_LE(sum_tags_ms, wall_total_ms * 1.05)
        << "Σ(stage tags) exceeds the measured wall time at N=" << n
        << " -> a timer is nested / double-counted; the breakdown cannot be trusted";
    if (obs_ms_per_step_out != nullptr)
        *obs_ms_per_step_out =
            TagTotalUs(perf, "obs_export") * 1.0e-3 / static_cast<double>(k > 0u ? k : 1u);
    return solver_ms / static_cast<double>(k > 0u ? k : 1u);  // PER-STEP (see header).
}

// ---------------------------------------------------------------------------
// (B) FAITHFUL synthetic row set for the coloring micro-benchmark: N disjoint K-cliques.
// Empirically K=10 (condim=3 -> 5 rows/contact-point x 2 fingertip contacts; confirmed by
// a temporary [KPROBE] in batched_unified_world.cpp, since reverted: total_rows = N*10 for
// every N in the sweep). The real grasp row assembly (batched_unified_world.cpp:846-849)
// rewrites BOTH body_indices of every finger row of env e to {body_a = total_body_count+e,
// body_b = cup BodyIndex(e,...)}, so all K rows in env e share BOTH bodies (a K-clique that
// all pairwise-conflict via RowsConflict), while cross-env rows share NO index (never
// conflict). RowsConflict (row_scheduler.cu:14) reads ONLY BodiesForRow -> body_indices
// [body_list_offset] and [+1]; the jacobian/material/anchor payload is irrelevant to
// coloring. We build via RowBuffers::AddRow, which sets body_count=2 and body_list_offset
// =2*idx for us (row_buffers.cpp:20-40) -- exactly what BodiesForRow reads -- so the CSR is
// correct by construction. The exact index VALUES are immaterial (only the disjoint-clique
// STRUCTURE drives cost + ColorCount); we use body_a = n_envs + e, body_b = e (disjoint
// across envs, equal within an env). The decisive invariant: ColorCount() == K for EVERY N
// (one color per clique member, reused across all envs). If it is 1 or N*K, the structure
// is wrong.
constexpr uint32_t kSyntheticK = 10u;  // confirmed empirically (see comment above).
nuka::constraint::RowBuffers BuildSyntheticGraspRows(uint32_t n_envs, uint32_t k) {
    nuka::constraint::RowBuffers rows;
    rows.rows.reserve(static_cast<size_t>(n_envs) * k);
    rows.body_indices.reserve(2u * static_cast<size_t>(n_envs) * k);
    rows.jacobian_data.reserve(2u * static_cast<size_t>(n_envs) * k);
    rows.materials.reserve(static_cast<size_t>(n_envs) * k);
    rows.anchors.reserve(static_cast<size_t>(n_envs) * k);
    const nuka::constraint::RowJacobian6 zero_j{};
    for (uint32_t e = 0u; e < n_envs; ++e) {
        const uint32_t finger_key = n_envs + e;  // synthetic finger-side key (disjoint/env).
        const uint32_t cup_key = e;              // synthetic cup body index (disjoint/env).
        for (uint32_t f = 0u; f < k; ++f) {
            rows.AddRow(nuka::constraint::Row{}, {finger_key, cup_key}, zero_j, zero_j);
        }
    }
    return rows;
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
    double measured_n1024_eps = 0.0;  // A1 GATE-5: the FULL RL per-step throughput at N=1024.
    for (uint32_t n : kNs) {
        const uint32_t k = StepsForN(n);
        double ms_per_step = 0.0;
        // RunHarness now brackets SetActions + Step + ExportObsState (the full RL per-step cost).
        const double eps = RunHarness(context, tmpl, n, k, &ms_per_step);
        std::printf("[PERF] N=%u K=%u env_steps_per_sec=%.1f ms_per_step=%.3f "
                    "(INCLUDES SetActions+ExportObsState)\n", n, k, eps, ms_per_step);
        std::fflush(stdout);
        if (n == 32u) measured_n32_eps = eps;
        if (n == 1024u) measured_n1024_eps = eps;
    }

    // ----- (2) the PER-STAGE breakdown at N=32, 256, 1024 (fresh, isolated runs) --------
    // Each call builds a CLEAN world so the recorder holds ONLY that N's samples (the
    // sweep's worlds were destroyed). The big-N runs use a small K (the step orchestration
    // is O(N), so a fixed K=200 would make N=1024 take ~10 min) -- K is printed in every
    // line. This is the (A) deliverable: does row_solver (and which other tags) grow
    // super-linearly in N? Returns each N's PER-STEP row_solver ms for the (D) verdict (the
    // printed BREAKDOWN lines are tag TOTALS over k steps; these returns are total/k -- one
    // UnifiedSolve/BuildRowColorPartitions per Step, so per-step is the coloring denominator).
    double obs_ms_n32 = 0.0, obs_ms_n256 = 0.0, obs_ms_n1024 = 0.0;
    const double solver_ms_n32 = RunStageBreakdown(context, tmpl, 32u, kK, &obs_ms_n32);  // K=200.
    const double solver_ms_n256 = RunStageBreakdown(context, tmpl, 256u, 20u, &obs_ms_n256);
    const double solver_ms_n1024 = RunStageBreakdown(context, tmpl, 1024u, 10u, &obs_ms_n1024);

    // ----- (2a) A1 GATE-5: the ExportObsState per-step ms scaling (N=32/256/1024) ----------
    // It must be ~linear in N and small (like the other downloads) -- NOT re-introducing an
    // O(N) host-sync per-env storm. The whole obs export is a bulk device download (q + qdot
    // via direct CopyToHost over the env-major buffers -- NOT a full DownloadArticulationState)
    // + host packing + the host-resident fingertip copy.
    std::printf("[A1 OBS-EXPORT per-step] N=32=%.4f_ms N=256=%.4f_ms N=1024=%.4f_ms "
                "(ratio 1024/32=%.1fx for 32x envs -- ~linear if ~32x)\n",
                obs_ms_n32, obs_ms_n256, obs_ms_n1024,
                obs_ms_n32 > 0.0 ? obs_ms_n1024 / obs_ms_n32 : 0.0);
    std::fflush(stdout);

    // ----- (2b) THE DECISIVE MEASUREMENT: isolate the coloring cost ---------------------
    // Directly host-wall-clock-time nuka::solver::gpu::BuildRowColorPartitions (the named
    // O(rows^2) super-linearity suspect) on a FAITHFUL synthetic N-disjoint-K-clique row
    // set at N in {32,256,1024}. BuildRowColorPartitions + ValidateNoSharedBodiesPerColor
    // are BOTH host O(rows^2) and BOTH run inside the `row_solver` tag (row_solver.cu:1128
    // -1129, behind UnifiedSolve); everything after them in SolveRowsWithSides (the scratch
    // uploads at 1138-1230, the device_view assembly, the single-block SolveRowsSweepKernel
    // at 1291) is O(rows) host work + device work. So `row_solver` tag ~= (Build + Validate)
    // + O(rows) residual. We time each separately, several reps (more at small N where each
    // call is microseconds), and report mean ms + ColorCount() + row count.
    struct ColoringPoint {
        uint32_t n, rows, colors, reps;
        double build_ms, validate_ms;
    };
    std::vector<ColoringPoint> coloring;
    const uint32_t kColorNs[] = {32u, 256u, 1024u};
    const uint32_t kColorReps[] = {500u, 50u, 10u};  // inverse-scaled with N (see comment).
    for (size_t i = 0u; i < 3u; ++i) {
        const uint32_t n = kColorNs[i];
        const uint32_t reps = kColorReps[i];
        const nuka::constraint::RowBuffers rows = BuildSyntheticGraspRows(n, kSyntheticK);
        // One untimed call to capture ColorCount() + validity (and warm any allocation).
        const auto warm_part = nuka::solver::gpu::BuildRowColorPartitions(rows);
        const bool valid =
            nuka::solver::gpu::ValidateNoSharedBodiesPerColor(rows, warm_part);
        EXPECT_TRUE(valid) << "synthetic partition self-conflicts at N=" << n;
        EXPECT_EQ(warm_part.ColorCount(), kSyntheticK)
            << "synthetic coloring at N=" << n << " gave ColorCount="
            << warm_part.ColorCount() << " (expected K=" << kSyntheticK
            << "); the N-disjoint-K-clique structure is WRONG";

        const auto b0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0u; r < reps; ++r) {
            const auto p = nuka::solver::gpu::BuildRowColorPartitions(rows);
            (void)p.ColorCount();  // keep the result live (defeat dead-code elision).
        }
        const auto b1 = std::chrono::steady_clock::now();
        const double build_ms =
            std::chrono::duration<double, std::milli>(b1 - b0).count() / reps;

        const auto v0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0u; r < reps; ++r) {
            const bool ok =
                nuka::solver::gpu::ValidateNoSharedBodiesPerColor(rows, warm_part);
            (void)ok;
        }
        const auto v1 = std::chrono::steady_clock::now();
        const double validate_ms =
            std::chrono::duration<double, std::milli>(v1 - v0).count() / reps;

        coloring.push_back({n, rows.RowCount(), warm_part.ColorCount(), reps, build_ms,
                            validate_ms});
        std::printf("[COLORING N=%u] rows=%u ColorCount=%u reps=%u BuildRowColorPartitions="
                    "%.4f_ms ValidateNoSharedBodiesPerColor=%.4f_ms (Build+Validate=%.4f_ms)\n",
                    n, rows.RowCount(), warm_part.ColorCount(), reps, build_ms,
                    validate_ms, build_ms + validate_ms);
        std::fflush(stdout);
    }

    // ----- (2c) THE VERDICT (D): coloring as a fraction of the row_solver tag -----------
    auto coloring_at = [&](uint32_t n) -> const ColoringPoint& {
        for (const auto& c : coloring)
            if (c.n == n) return c;
        return coloring.front();
    };
    const ColoringPoint& c32 = coloring_at(32u);
    const ColoringPoint& c1024 = coloring_at(1024u);
    const double col32_ms = c32.build_ms + c32.validate_ms;
    const double col1024_ms = c1024.build_ms + c1024.validate_ms;

    // Sanity (B last bullet): the synthetic coloring @ N=32 must be CONSISTENT with the real
    // N=32 per-step row_solver -- it is the dominant share of it (already ~75% at N=32),
    // never EXCEEDING it (coloring is INSIDE the tag). If coloring > tag, K or the clique
    // structure is wrong (the synthetic is over-counting conflicts).
    std::printf("[COLORING-SANITY N=32] Build+Validate=%.4f_ms row_solver_per_step=%.4f_ms "
                "(coloring/per_step=%.1f%%)\n",
                col32_ms, solver_ms_n32,
                solver_ms_n32 > 0.0 ? 100.0 * col32_ms / solver_ms_n32 : 0.0);
    EXPECT_LE(col32_ms, solver_ms_n32 * 1.10)
        << "synthetic coloring (" << col32_ms << " ms) exceeds the real per-step row_solver ("
        << solver_ms_n32 << " ms) -- coloring is INSIDE that tag, so the synthetic must be "
           "over-counting conflicts (K or the clique structure is wrong)";

    // The O(rows) residual = per-step row_solver - coloring (the scratch uploads + the
    // single-block kernel + the host round-trip). It must be SMALL and ~linear in N; the
    // quadratic coloring swamping it is the COLORING-BOUND signature.
    const double resid32 = solver_ms_n32 - col32_ms;
    const double resid256 = solver_ms_n256 - (coloring_at(256u).build_ms +
                                              coloring_at(256u).validate_ms);
    const double resid1024 = solver_ms_n1024 - col1024_ms;

    const double col1024_frac =
        solver_ms_n1024 > 0.0 ? 100.0 * col1024_ms / solver_ms_n1024 : 0.0;
    const bool coloring_bound = col1024_frac >= 50.0;
    std::printf("[PERF-VERDICT N=1024] coloring(Build+Validate)=%.4f_ms "
                "row_solver_per_step=%.4f_ms -> coloring/per_step=%.1f%% -> %s\n",
                col1024_ms, solver_ms_n1024, col1024_frac,
                coloring_bound ? "COLORING-BOUND" : "KERNEL/TRANSFER-BOUND");
    std::printf("[PERF-VERDICT] row_solver PER-STEP growth: N=32=%.3f_ms N=256=%.3f_ms "
                "N=1024=%.3f_ms (%.0fx for 32x envs) ; coloring growth: N=32=%.4f_ms "
                "N=256=%.4f_ms N=1024=%.4f_ms ; O(rows) residual: N=32=%.3f_ms "
                "N=256=%.3f_ms N=1024=%.3f_ms\n",
                solver_ms_n32, solver_ms_n256, solver_ms_n1024,
                solver_ms_n32 > 0.0 ? solver_ms_n1024 / solver_ms_n32 : 0.0,
                col32_ms, coloring_at(256u).build_ms + coloring_at(256u).validate_ms,
                col1024_ms, resid32, resid256, resid1024);
    std::fflush(stdout);

    // ----- (3) ONE loose regression guard -------------------------------------------
    // Perf is noisy run-to-run on a shared box (the absolute throughput varies with host
    // load), so the floor is GENEROUS: 0.3x the measured N=32 env-steps/sec observed when
    // this gate was baked. The PRINTED numbers are the deliverable; this assert only
    // catches a GROSS regression (a 3x+ slowdown, e.g. an accidental extra per-env device
    // round-trip in the host orchestration loop), not noise.
    //
    // BASELINE: ~4836 env-steps/sec measured at P2.4c bake time on the build-cuda128 box (N=32,
    // K=200), up ~9.2x from the P2.4b-era ~525 (which was narrowphase-bound at 90.9%). P2.4c
    // moved the grasp narrowphase (sphere fingertip x cup ConvexHull) host->GPU: ONE batched
    // grasp_sphere_hull_kernel wrapping the HD-clean cvx::SphereHull over all (env x fingertip)
    // slots, replacing the per-env host BuildContactManifolds loop. narrowphase collapsed
    // 90.9%->~10.6% at N=32; the step is now ROW_SOLVER-bound (~79.7% -- the single-block
    // batched UnifiedSolve + its host round-trip), which is the P2.4d/P2.4e target (device-
    // resident rows + a multi-block solve kernel). The floor stays GENEROUS at 0.3x (catches a
    // 3x+ regression, not noise); set conservatively below the measured 4836 so a noisy box
    // does not flake. Update if the baseline moves again (e.g. after P2.4d/P2.4e lift throughput).
    constexpr double kN32BaselineEps = 4500.0;  // see comment above; floor is 0.3x this.
    EXPECT_GT(measured_n32_eps, 0.3 * kN32BaselineEps)
        << "N=32 throughput (" << measured_n32_eps
        << " env-steps/sec) fell below 0.3x the baseline (" << kN32BaselineEps
        << ") -- a gross perf regression, not noise";

    // ----- (4) A1 GATE-5: the obs export must NOT re-introduce O(N) host-sync cost ----------
    // The per-step measurement above BRACKETS SetActions + Step + ExportObsState, so
    // measured_n1024_eps is the FULL RL per-step throughput at N=1024. The bar: stay within
    // ~0.85x of the coloring-fix baseline (~10,183 env-steps/sec measured at HEAD 0ae4388 on
    // this box, BEFORE A1 -- Step only). If the obs export were a per-env DownloadGripper loop
    // it would crater this; the ONE consolidated download keeps it ~flat. The floor is 0.85x of
    // a conservative baseline so a noisy box does not flake while still catching a real O(N)
    // regression in the export. Update the baseline if the box's clean throughput moves.
    constexpr double kN1024BaselineEps = 10000.0;  // ~coloring-fix baseline (Step-only, HEAD).
    std::printf("[A1 GATE-5 N=1024] env_steps_per_sec(WITH obs export)=%.1f  bar=0.85x_baseline="
                "%.1f (baseline=%.1f)\n",
                measured_n1024_eps, 0.85 * kN1024BaselineEps, kN1024BaselineEps);
    std::fflush(stdout);
    EXPECT_GT(measured_n1024_eps, 0.85 * kN1024BaselineEps)
        << "N=1024 throughput WITH the obs export (" << measured_n1024_eps
        << " env-steps/sec) fell below 0.85x the coloring-fix baseline (" << kN1024BaselineEps
        << ") -- the obs export re-introduced O(N) host-sync cost; it must be batched, not a "
           "per-env DownloadGripper loop";
}
