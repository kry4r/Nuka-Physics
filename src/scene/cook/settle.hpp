#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::cook — Settle (M7 T2): the GENERIC deterministic settle + the
// rest-placement cook primitives the later M7 tasks (T3 nks-parse, T4 cook)
// compose.
//
// Settle steps a COOKED scene (an already-built nk::World) deterministically
// for `steps` device steps while PD-holding the named dof patterns at their
// seeded targets (so an articulation stays at its IC while a movable cup comes
// to rest on its support), then SNAPSHOTS the settled state (the Reset-restore
// source) and downloads it into an InitialStateComponent (articulation q + root
// base pose) plus the settled movable-body poses. This is the deterministic
// rest cook used for cup-on-surface rest, the determinism gate, and a confirming
// hold — NOT a biped CoP balancer (controller ruling R4: the H1 balanced stance
// IC is baked into the .nks initial_state by T3, not synthesized here).
//
// DETERMINISM (the M7 gate): the same (world, spec) settled twice produces a
// byte-identical InitialStateComponent + body poses. Only the deterministic
// device step path is used; no host RNG; downloads happen in a fixed field
// order. The Hold pattern resolution + drive seeding are pure functions of the
// SceneMap + SceneIR (stable record/link order).
//
// nk_engine lint scope: CUDA-type-free C++. Settle takes an already-built
// nk::World (the caller does CookToModel -> nk::World; see the pattern in
// tests/scenario/coupled_grasp_soft.cpp / tests/perf/nk_union_n1.cpp) — settle
// writes NO kernels and constructs no device; it only drives the host World API.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "math/transform.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/entity.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"

namespace nuka::scene::cook {

// One PD hold over a set of articulation dofs named by a SceneMap NAME PATTERN
// (matched against the entity's derived tree path, e.g. "h1/.*" holds every H1
// link). During settle, the matched links' PD drive targets are pinned to their
// CURRENT (seeded) q so the articulation holds that pose while the rest of the
// scene settles. (The articulation drive arrays are per-LINK; a Hold therefore
// resolves to LINK indices and pins drive_target[link] = q[link].)
struct SettleSpec {
    int   steps = 0;
    float dt    = 0.0f;   // informational (the World's SolverConfig owns the step
                          // dt); recorded here so T3/T4 can author both together.

    struct Hold {
        std::string dof_pattern;            // SceneMap name pattern (path glob)
        enum class Mode { PD } mode = Mode::PD;

        // PD hold gains applied to the matched links during settle. Defaults are
        // a firm generic hold (the factory's non-leg kKpHold/kKdHold class); a
        // caller (T3) may override per pattern.
        float kp = 50.0f;
        float kd = 4.0f;
    };
    std::vector<Hold> holds;
};

// The settle product: the settled articulation IC (q per LINK + root base pose)
// and the settled movable-body world poses keyed by the body's EntityId (so a
// writeback can target the right body record / component). Plain data — the gate
// compares these byte-for-byte across two runs.
struct SettleResult {
    InitialStateComponent initial_state;   // qpos = per-link q; root = base pose
    std::vector<std::pair<EntityId, math::Transform>> body_poses;  // settled
    bool ok = false;                        // false on a device/download failure
};

// Run the deterministic settle on an ALREADY-BUILT world. Resolves each Hold's
// dof_pattern -> link indices via the SceneMap (entity-of-link) + the SceneIR
// derived paths, pins those links' PD drive targets to their seeded q, steps the
// world `spec.steps` times on the byte-deterministic device path, dispatches
// SnapshotState (so the settled state is the Reset-restore source), then
// downloads the settled q + base pose + movable-body poses (env 0) in a fixed
// order. Returns the SettleResult (ok == false if the world is not Ready or a
// device op/download fails). Does NOT mutate the SceneIR (see ApplySettle*).
SettleResult Settle(nk::World& world, const SceneIR& scene, const SceneMap& map,
                    const SettleSpec& spec);

// Writeback (1): set the settled IC as an InitialStateComponent on
// `articulation_entity` in `registry`, and update each settled body's
// TransformComponent to its settled WORLD pose. Operates on a mutable Registry
// (the ECS write target — the SceneIR facade is const-only). T4 reads the
// InitialStateComponent during cook.
void ApplySettleToRegistry(Registry& registry, EntityId articulation_entity,
                           const SettleResult& result);

// Writeback (2): convenience for the SceneIR-records path — move each settled
// movable body's record local_transform to its settled pose (so the cooked /
// saved SceneIR carries the rested cup pose). The body's settled WORLD pose is
// applied as the record local_transform when the body is a ROOT body (parent ==
// kInvalidBody); a child body's settled pose is converted into its parent frame.
// (The articulation IC is NOT a record field — use ApplySettleToRegistry for the
// InitialStateComponent.)
void ApplySettleToSceneIR(SceneIR& scene, const SettleResult& result,
                          const SceneMap& map);

// --- pattern matching (exposed for tests/T3) --------------------------------
// Match a derived tree path against a SettleSpec dof_pattern. Supports a literal
// match, a trailing/embedded "*" or ".*" wildcard (any run), and "?" (one char).
// Pure + deterministic; no <regex>. ("h1/.*" or "h1/*" both match "h1/<link>".)
bool PathMatchesPattern(const std::string& path, const std::string& pattern);

}  // namespace nuka::scene::cook
