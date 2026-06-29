#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::SceneController -- a per-step control hook the Simulation
// calls before advancing physics. The controller reads world state and writes
// drive/force fields through the general nk::Data seam (UploadField); it owns its
// own control decimation. NOT owned by the Simulation (it must outlive it). This
// is the GENERAL seam: a policy/scripted/keyframe controller all implement it, so
// world.Step() / the solver carry NO controller-specific branch.
//
// HOST-ONLY (no CUDA tokens).
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::nk {
class World;
}  // namespace nuka::nk

namespace nuka::runtime::app {

class SceneController {
public:
    virtual ~SceneController() = default;

    // Called once per physics step (BEFORE it) for the active env; the controller
    // owns its decimation and writes outputs via world's Data seam.
    virtual void OnStep(nk::World& world, uint32_t env_index) = 0;

    // Optional command sink: a command-driven controller (e.g. a locomotion policy
    // that consumes a velocity command) overrides these; the default ignores commands.
    // A general producer (interactive teleop, a script) writes a command vector through
    // this base interface without knowing the controller type -- world.Step() stays free
    // of controller-specific branches.
    virtual bool AcceptsCommand() const { return false; }
    virtual void SetCommandVector(const float* cmd, uint32_t count) {
        (void)cmd;
        (void)count;
    }
};

}  // namespace nuka::runtime::app
