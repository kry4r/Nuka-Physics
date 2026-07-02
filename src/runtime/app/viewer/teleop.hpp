#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer -- interactive keyboard teleop mapping.
//
// A general live-input -> drive-command producer, the interactive sibling of the
// Drive sliders and the --policy controller. Per-DOF nudges write the SAME
// drive_targets / drive_dirty seam the Drive panel uses; WASD/QE produce a
// [vx,vy,wyaw] command for a command-consuming SceneController. Decoupled from ImGui
// (the viewer reads key state into TeleopKeys) so the mapping is unit-testable with
// synthetic intents. Host-only, no deps beyond ViewerUiState.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/imgui_layer.hpp"

#include <cstddef>
#include <cstdint>

namespace nuka::runtime::app::viewer {

// The per-frame keyboard state teleop maps. The dof_* fields are pressed-this-frame
// (one-shot edges); the move/yaw fields are held-down-now (continuous). The viewer
// fills these from ImGui; a test fills them synthetically.
struct TeleopKeys {
    bool dof_minus = false;   // '[' or ','  : nudge the selected DOF target down
    bool dof_plus  = false;   // ']' or '.'  : nudge it up
    bool dof_prev  = false;   // '-'         : select the previous DOF
    bool dof_next  = false;   // '='         : select the next DOF
    bool fwd = false, back = false;          // W / S : +vx / -vx
    bool strafe_l = false, strafe_r = false; // A / D : +vy / -vy
    bool yaw_l = false, yaw_r = false;       // Q / E : +wyaw / -wyaw
};

// One teleop tick's locomotion command, handed to a command-consuming controller.
// The producer sends it EVERY frame; a gated tick is all-zero (stops the robot).
struct TeleopResult {
    bool  command_active = false;
    float command[3]     = {0.0f, 0.0f, 0.0f};  // vx, vy, wyaw
};

// One teleop tick. Gated (`enabled` false) => ui untouched + the ZERO command (still
// sent every frame, so a gate/release stops the robot); enabled => base + held keys.
inline TeleopResult ApplyTeleop(bool enabled, const TeleopKeys& keys,
                                ViewerUiState& ui, uint32_t dof_count) {
    TeleopResult out;
    if (!enabled) return out;

    // Per-DOF: select (clamped into range) + nudge the active DOF's drive target.
    if (dof_count > 0u) {
        if (keys.dof_next) ui.teleop_dof += 1;
        if (keys.dof_prev) ui.teleop_dof -= 1;
        if (ui.teleop_dof < 0) ui.teleop_dof = 0;
        if (ui.teleop_dof >= static_cast<int>(dof_count))
            ui.teleop_dof = static_cast<int>(dof_count) - 1;
        if (ui.drive_targets.size() == dof_count && ui.drive_dirty.size() == dof_count) {
            float delta = 0.0f;
            if (keys.dof_plus)  delta += ui.teleop_step;
            if (keys.dof_minus) delta -= ui.teleop_step;
            if (delta != 0.0f) {
                const size_t d = static_cast<size_t>(ui.teleop_dof);
                ui.drive_targets[d] += delta;
                ui.drive_dirty[d] = 1u;
            }
        }
    }

    // Locomotion command: the persistent baseline plus the held keys (all keys
    // released -> back to the baseline, which is zero unless a CLI/script set it).
    float vx = ui.teleop_cmd_base[0], vy = ui.teleop_cmd_base[1], wz = ui.teleop_cmd_base[2];
    if (keys.fwd)      vx += ui.teleop_lin;
    if (keys.back)     vx -= ui.teleop_lin;
    if (keys.strafe_l) vy += ui.teleop_lin;
    if (keys.strafe_r) vy -= ui.teleop_lin;
    if (keys.yaw_l)    wz += ui.teleop_yaw;
    if (keys.yaw_r)    wz -= ui.teleop_yaw;
    out.command[0] = vx;
    out.command[1] = vy;
    out.command[2] = wz;
    out.command_active = (vx != 0.0f || vy != 0.0f || wz != 0.0f);
    ui.teleop_cmd[0] = vx;
    ui.teleop_cmd[1] = vy;
    ui.teleop_cmd[2] = wz;
    return out;
}

}  // namespace nuka::runtime::app::viewer
