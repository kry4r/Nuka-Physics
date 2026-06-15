#pragma once
// ---------------------------------------------------------------------------
// h1_grasp_driver.hpp -- M10 T1: the proven H1 union grasp driver, lifted
// VERBATIM out of tests/scenario/h1_grasp_lift.cpp into a reusable header so the
// headline demo video (examples/demo/h1_grasp_video.cpp) drives the EXACT same
// approach -> close -> lift -> friction-only-hold choreography the permanent
// grasp gate asserts. NOT a re-tune: the PD law, the clamp, the phase windows,
// the kill_grip BITE switch, and the TableEnabled<-0 lift trigger are byte-for-
// byte the h1_grasp_lift path (D1-frozen). The gate stays the authority; this is
// a code-share so the demo cannot drift from the gated physics.
//
// HOST-ONLY: pure C++ over the nk::World field IO (DownloadField/UploadField) +
// the cooked CookedUnionScene drive tables. No CUDA tokens, no render deps.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstdint>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_scene_template.hpp"

namespace nuka::demo {

namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace coresident = nuka::runtime::coresident;

// ---- choreography windows (h1_grasp_lift.cpp:86-89, verbatim) --------------
// hold -> rest -> close, with the table removed at the lift so the cup is then
// held by the fingers' friction ALONE.
inline constexpr uint32_t kWindowEnd = 10u;    // hold-phase end.
inline constexpr uint32_t kRestEnd   = 100u;   // rest end == close start.
inline constexpr uint32_t kLiftAt    = 180u;   // table off -> friction-only hold.

// ---- per-step closed-loop torque from a drive table (h1_grasp_lift.cpp:109) --
// u = kp*(target - q) - kd*qdot, clamped to +/-tlim; per-LINK (ApplyTorqueDrive
// layout). kill_grip zeroes the wrap-driven grip columns (the BITE switch).
inline void TableTorque(const std::vector<coresident::H1UnionDriveEntry>& tab,
                        const std::vector<float>& q_link,
                        const std::vector<float>& qdot_link, bool kill_grip,
                        std::vector<float>* torque) {
    for (const auto& d : tab) {
        if (d.link >= torque->size()) continue;
        if (kill_grip && d.grip != 0u) {
            (*torque)[d.link] = 0.0f;
            continue;
        }
        float u = d.kp * (d.target - q_link[d.link]) - d.kd * qdot_link[d.link];
        if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
        (*torque)[d.link] = u;
    }
}

// Seed the per-link torque buffer with the template grip torque (the legacy
// coresident world's action_torque_host_ init the grasp gate mirrors).
inline std::vector<float> InitTorque(const cook::CookedUnionScene& cooked,
                                     uint32_t links) {
    std::vector<float> t(links, 0.0f);
    const auto& g = cooked.tmpl.grip_torque;
    for (uint32_t l = 0; l < links && l < g.size(); ++l) t[l] = g[l];
    return t;
}

// Drive ONE step of the approach -> close -> lift choreography on the union nk
// world (h1_grasp_lift.cpp DriveStep, verbatim minus the per-step contact
// report the gate reads). `step` selects the drive table by phase; `kill_grip`
// forces the grip columns to 0 (the BITE); the table is removed once at kLiftAt.
// `torque`/`q`/`qdot` are caller-owned scratch buffers (length == links). The
// caller calls w.Step() AFTER this returns? NO -- this CALLS w.Step() itself so
// the demo and the gate share the identical upload->step ordering.
inline void DriveStep(nk::World& w, const cook::CookedUnionScene& cooked,
                      uint32_t links, uint32_t step, bool kill_grip,
                      std::vector<float>* torque, std::vector<float>* q,
                      std::vector<float>* qdot) {
    if (step == kLiftAt) {
        const uint32_t off = 0u;
        w.GetData().UploadField(nk::FieldId::TableEnabled, &off, sizeof(uint32_t));
    }
    const auto& tab = (step < kWindowEnd) ? cooked.drive_hold
                      : (step < kRestEnd) ? cooked.drive_rest
                                          : cooked.drive_close;
    w.GetData().DownloadField(nk::FieldId::Q, q->data(), links * sizeof(float));
    w.GetData().DownloadField(nk::FieldId::Qdot, qdot->data(),
                              links * sizeof(float));
    TableTorque(tab, *q, *qdot, kill_grip, torque);
    w.GetData().UploadField(nk::FieldId::DriveTarget, torque->data(),
                            torque->size() * sizeof(float));
    w.Step();
}

}  // namespace nuka::demo
