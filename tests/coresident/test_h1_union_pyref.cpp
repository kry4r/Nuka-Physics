// ---------------------------------------------------------------------------
// v0.8 G2 -- the C++ REFERENCE RECORDER for the python byte-parity gate.
// ---------------------------------------------------------------------------
// Drives the H1 whole-body UNION world at N=4 THROUGH THE C ABI
// (nuka_union_world_*, i.e. through libnuka -- the EXACT code path the python
// binding calls, so the python replay has no cross-compile seam) with a
// closed-loop lift choreography computed from the C-ABI obs export + the
// factory's reference drive tables:
//   WINDOW (0..9)   : hold the curl at the settled q (table carries the cup),
//   REST   (10..99) : wrap backed off -0.25 rad (honest table rest),
//   CLOSE  (100..179): the H1.1 close PD sweeps in + squeezes into the hold,
//   LIFT   (180..299): table removed -> friction-only hold.
// Per-env variations (cross-env discrimination for the python replay):
//   env 0: the full choreography (holds the cup after the toggle),
//   env 1: table removed @180 AND grip killed @180 -> the cup falls,
//   env 2: table removed @0 -> the REST back-off drops the cup early,
//   env 3: table NEVER removed -> the cup stays at table height.
// Ankle balance: the factory tables' POSTURE stand-in (not the test-side CoP
// law -- a python caller can reproduce this PD from the obs export alone).
//
// RECORDS to .g2_logs/g2_pyref_n4.bin (little-endian):
//   header : 8 x u32 [magic 'G2UF' 0x47325546, version=1, n, steps, dof, nf,
//            n_events, reserved] + 2 x f32 [dt, gravity_z]
//   events : n_events x { u32 step, u32 env (0xFFFFFFFF = all), u32 enabled }
//   per step: f32 actions[n*dof], then the obs blocks in the python harness's
//            FLOAT_FIELDS order (q, qdot, base_pose, base_vel, cup_pose,
//            cup_vel, fingertip_world_pos, finger_normal_impulse, foot_impulse,
//            table_impulse) then UINT_FIELDS (foot_rows, table_rows,
//            finger_contacts) -- every value read through the SAME C-ABI fill
//            python uses (byte-for-byte the python comparison target).
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"
#include "nuka/nuka_union.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x47325546u;  // 'G2UF'
constexpr uint32_t kEnvs = 4u;
constexpr uint32_t kSteps = 300u;
constexpr uint32_t kWindowEnd = 10u;   // hold phase end.
constexpr uint32_t kRestEnd = 100u;    // rest phase end == close start.
constexpr uint32_t kLiftAt = 180u;     // table off (envs 0,1) + env1 grip kill.

struct Event {
    uint32_t step, env, enabled;
};

const char* kOutPath = ".g2_logs/g2_pyref_n4.bin";

struct DriveTable {
    std::vector<uint32_t> dof;
    std::vector<float> target, kp, kd, tlim;
    std::vector<uint8_t> grip;
};

DriveTable FetchTable(nuka_union_world_handle w, uint32_t which, uint32_t len) {
    DriveTable t;
    t.dof.resize(len);
    t.target.resize(len);
    t.kp.resize(len);
    t.kd.resize(len);
    t.tlim.resize(len);
    t.grip.resize(len);
    EXPECT_EQ(NUKA_RESULT_OK,
              nuka_union_world_drive_table(w, which, t.dof.data(), t.target.data(),
                                           t.kp.data(), t.kd.data(), t.tlim.data(),
                                           t.grip.data(), len));
    return t;
}

template <typename T>
void WriteVec(std::ofstream& f, const std::vector<T>& v) {
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
}

}  // namespace

TEST(H1UnionPyRef, RecordReferenceTrajectory) {
    if (!std::filesystem::exists(
            ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml") ||
        !std::filesystem::exists(
            ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda")) {
        GTEST_SKIP() << "h1_with_hand / cup not present";
    }

    nuka_device_desc_t ddesc{};
    ddesc.gpu_index = 0u;
    ddesc.backend_selection_layer_enabled = 1u;
    nuka_device_handle dev = nullptr;
    ASSERT_EQ(NUKA_RESULT_OK, nuka_device_create(&ddesc, &dev));

    // The SAME sentinel construction the python binding's defaults use.
    nuka_union_world_desc_t desc{};
    desc.env_count = kEnvs;
    nuka_union_world_handle w = nullptr;
    ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_create(dev, &desc, &w));

    nuka_union_world_dims_t dims{};
    ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_dims(w, &dims));
    nuka_union_scene_info_t info{};
    ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_scene_info(w, &info));
    ASSERT_EQ(dims.action_dim, 51u);
    ASSERT_EQ(dims.base_dof, 6u);
    ASSERT_EQ(dims.num_fingertips, 30u);
    ASSERT_EQ(dims.num_feet, 4u);
    ASSERT_EQ(dims.num_grip_dofs, 12u);
    ASSERT_EQ(info.place_found, 1u);
    const uint32_t n = dims.env_count;
    const uint32_t ad = dims.action_dim;
    const uint32_t nf = dims.num_fingertips;
    std::printf("[PYREF] union world up: n=%u dof=%u nf=%u feet=%u table_z=%.4f "
                "m_total=%.2f ankle_tau=(%.3f,%.3f)\n",
                n, ad, nf, dims.num_feet, info.table_height, info.total_mass,
                info.ankle_settle_tau[0], info.ankle_settle_tau[1]);

    const DriveTable hold = FetchTable(w, 0u, dims.drive_table_len);
    const DriveTable rest = FetchTable(w, 1u, dims.drive_table_len);
    const DriveTable close = FetchTable(w, 2u, dims.drive_table_len);
    std::vector<uint32_t> grip_dofs(dims.num_grip_dofs);
    ASSERT_EQ(NUKA_RESULT_OK,
              nuka_union_world_grip_dofs(w, grip_dofs.data(), grip_dofs.size()));

    // The per-env table-toggle event schedule (applied IN ORDER at each step;
    // the python replay applies the identical records).
    const std::vector<Event> events = {
        {0u, 2u, 0u},        // env 2: never-rested (table off pre-step-0).
        {kLiftAt, 0u, 0u},   // env 0: the lift (friction-only hold follows).
        {kLiftAt, 1u, 0u},   // env 1: lift + grip kill -> the cup falls.
    };

    // Obs buffers (the SAME C-ABI fills python reads).
    std::vector<float> q(n * ad), qdot(n * ad), bp(n * 7u), bv(n * 6u);
    std::vector<float> ft(n * nf * 3u), fimp(n * nf), la(n * ad);
    std::vector<float> cp(n * 7u), cv(n * 6u), foot_imp(n), table_imp(n);
    std::vector<uint32_t> foot_rows(n), table_rows(n), finger_contacts(n);

    auto export_all = [&]() {
        ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_export_obs(
                                      w, q.data(), q.size(), qdot.data(), qdot.size(),
                                      bp.data(), bp.size(), bv.data(), bv.size(),
                                      ft.data(), ft.size(), fimp.data(), fimp.size(),
                                      la.data(), la.size()));
        ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_read_cup_contacts(
                                      w, cp.data(), cp.size(), cv.data(), cv.size(),
                                      foot_rows.data(), n, foot_imp.data(), n,
                                      table_rows.data(), n, table_imp.data(), n,
                                      finger_contacts.data(), n));
    };

    std::filesystem::create_directories(".g2_logs");
    std::ofstream f(kOutPath, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(f.is_open()) << "cannot open " << kOutPath;
    const uint32_t header[8] = {kMagic, 1u, n, kSteps, ad, nf,
                                static_cast<uint32_t>(events.size()), 0u};
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(&info.dt), sizeof(float));
    f.write(reinterpret_cast<const char*>(&info.gravity_z), sizeof(float));
    for (const Event& e : events) {
        const uint32_t rec[3] = {e.step, e.env, e.enabled};
        f.write(reinterpret_cast<const char*>(rec), sizeof(rec));
    }

    std::vector<float> actions(n * ad, 0.0f);
    float cup_z_at_toggle[4] = {0, 0, 0, 0};
    for (uint32_t s = 0u; s < kSteps; ++s) {
        // (1) events for this step (recorded order).
        for (const Event& e : events) {
            if (e.step != s) continue;
            ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_set_table_enabled(
                                          w, static_cast<int32_t>(e.env), e.enabled));
        }
        if (s == 0u) {
            uint32_t te = 9u;
            ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_table_enabled(w, 2u, &te));
            EXPECT_EQ(te, 0u);
            ASSERT_EQ(NUKA_RESULT_OK, nuka_union_world_table_enabled(w, 3u, &te));
            EXPECT_EQ(te, 1u);
        }
        if (s == kLiftAt) {
            export_all();
            for (uint32_t e = 0u; e < n; ++e) cup_z_at_toggle[e] = cp[e * 7u + 2u];
        }

        // (2) the closed-loop choreography PD from THIS step's pre-step state.
        export_all();
        const DriveTable& tab =
            (s < kWindowEnd) ? hold : (s < kRestEnd) ? rest : close;
        std::fill(actions.begin(), actions.end(), 0.0f);
        for (uint32_t e = 0u; e < n; ++e) {
            const size_t off = static_cast<size_t>(e) * ad;
            for (size_t i = 0u; i < tab.dof.size(); ++i) {
                const uint32_t d = tab.dof[i];
                if (d >= ad) continue;
                if (tab.grip[i] != 0u && e == 1u && s >= kLiftAt)
                    continue;  // env 1 grip kill: close columns stay 0.
                float u = tab.kp[i] * (tab.target[i] - q[off + d]) -
                          tab.kd[i] * qdot[off + d];
                if (tab.tlim[i] > 0.0f)
                    u = std::max(-tab.tlim[i], std::min(tab.tlim[i], u));
                actions[off + d] = u;
            }
        }

        // (3) record the action bytes, drive, record the post-step obs.
        WriteVec(f, actions);
        ASSERT_EQ(NUKA_RESULT_OK,
                  nuka_union_world_step_with_actions(w, actions.data(), actions.size()));
        export_all();
        WriteVec(f, q);
        WriteVec(f, qdot);
        WriteVec(f, bp);
        WriteVec(f, bv);
        WriteVec(f, cp);
        WriteVec(f, cv);
        WriteVec(f, ft);
        WriteVec(f, fimp);
        WriteVec(f, foot_imp);
        WriteVec(f, table_imp);
        WriteVec(f, foot_rows);
        WriteVec(f, table_rows);
        WriteVec(f, finger_contacts);

        // last_action must echo the injected bytes exactly (the C-ABI contract
        // the python gate also asserts).
        ASSERT_EQ(0, std::memcmp(la.data(), actions.data(),
                                 actions.size() * sizeof(float)))
            << "last_action echo broke at step " << s;

        if (s % 50u == 0u || s + 1u == kSteps) {
            std::printf("[PYREF] s=%3u cup_z={%.3f %.3f %.3f %.3f} table_rows={%u %u %u %u} "
                        "foot_rows={%u %u %u %u} contacts={%u %u %u %u} base_z0=%.3f\n",
                        s, cp[2], cp[9], cp[16], cp[23], table_rows[0], table_rows[1],
                        table_rows[2], table_rows[3], foot_rows[0], foot_rows[1],
                        foot_rows[2], foot_rows[3], finger_contacts[0],
                        finger_contacts[1], finger_contacts[2], finger_contacts[3],
                        bp[2]);
        }
    }
    f.close();

    // ----- recorder-side physics sanity (the recording must be a MEANINGFUL
    // reference: held vs dropped cups discriminate the python replay) ----------
    export_all();
    const float z0 = cp[0 * 7u + 2u], z1 = cp[1 * 7u + 2u], z2 = cp[2 * 7u + 2u];
    std::printf("[PYREF] final: cup_z={%.3f %.3f %.3f %.3f} toggle_z0=%.3f "
                "table_z=%.3f foot_rows0=%u\n",
                z0, z1, z2, cp[3 * 7u + 2u], cup_z_at_toggle[0], info.table_height,
                foot_rows[0]);
    // env 0 holds through the lift (allow the measured ~6cm wrap sink + wobble).
    EXPECT_GT(z0, cup_z_at_toggle[0] - 0.35f)
        << "env0 dropped the cup after the toggle -- the choreography hold failed";
    // env 1 (grip killed at the toggle) falls well below the table plane.
    EXPECT_LT(z1, cup_z_at_toggle[1] - 0.4f) << "env1 cup did NOT fall (grip kill)";
    // env 2 (never-rested, wrap opened in REST) fell early.
    EXPECT_LT(z2, info.table_height - 0.3f) << "env2 cup did NOT fall (no table)";
    // env 3 keeps its table -> the cup stays at/above the table plane.
    EXPECT_GT(cp[3 * 7u + 2u], info.table_height - 0.05f) << "env3 lost its table rest";
    // the robot still stands in env 0 (feet loaded).
    EXPECT_GT(foot_rows[0], 0u) << "env0 robot lost foot contact";

    const auto bytes = std::filesystem::file_size(kOutPath);
    std::printf("[PYREF] wrote %s (%zu bytes, %u steps, %u events)\n", kOutPath,
                static_cast<size_t>(bytes), kSteps,
                static_cast<uint32_t>(events.size()));

    nuka_union_world_destroy(w);
    nuka_device_destroy(dev);
}
