#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::cook — SettleSpec (M7): the PLAIN-DATA deterministic-settle
// directive, split out of settle.hpp (M7 T3) so the LIGHTWEIGHT scene IR /
// .nks format can persist it WITHOUT dragging in nk::World (settle.hpp pulls
// the device World; this header is std-only). settle.hpp includes this and adds
// the Settle() runner; the .nks Save/Load (nks.cpp) + the cook (T4) consume the
// spec straight from the SceneIR.
//
// HOST-ONLY, NO CUDA, NO nk/phi deps.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>

namespace nuka::scene::cook {

// One PD hold over a set of articulation dofs named by a SceneMap NAME PATTERN
// (matched against the entity's derived tree path, e.g. "h1/.*" holds every H1
// link). During settle, the matched links' PD drive targets are pinned to their
// CURRENT (seeded) q so the articulation holds that pose while the rest of the
// scene settles. (The articulation drive arrays are per-LINK; a Hold therefore
// resolves to LINK indices and pins drive_target[link] = q[link].)

// Default settle hold gains. Empirically tuned for a legged-robot mass/inertia
// scale (the factory's non-leg hold class); scenes with very different mass
// (manipulation objects, heavy industrial robots) should author explicit gains.
inline constexpr float kDefaultSettleKp = 50.0f;
inline constexpr float kDefaultSettleKd = 4.0f;

struct SettleSpec {
    int   steps = 0;
    float dt    = 0.0f;   // informational (the World's SolverConfig owns the step
                          // dt); recorded here so callers can author both together.

    struct Hold {
        std::string dof_pattern;            // SceneMap name pattern (path glob)
        enum class Mode { PD } mode = Mode::PD;

        // PD hold gains applied to the matched links during settle; a caller may
        // override per pattern (see kDefaultSettleKp/Kd for the scale caveat).
        float kp = kDefaultSettleKp;
        float kd = kDefaultSettleKd;
    };
    std::vector<Hold> holds;
};

}  // namespace nuka::scene::cook
