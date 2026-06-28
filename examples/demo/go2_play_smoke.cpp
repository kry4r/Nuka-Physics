// ---------------------------------------------------------------------------
// go2_play_smoke -- headless closed-loop smoke for the Go2 terrain scene.
//
// Cooks go2_terrain30.nks through the editor's ONE load path (LoadEditorScene),
// attaches the trained Go2PolicyController, and drives the dogs closed-loop for N
// control steps. Asserts every dog's base z stays in a sane standing band over the
// retained terrain (no collapse / explosion / NaN) and prints the z trajectory.
// Pure host C++: the only device touch is nk::World step + Data D2H/H2D.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/inference/go2_policy_controller.hpp"
#include "scene/terrain/heightfield_sample.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace nphi = nuka::phi;
namespace viewer = nuka::runtime::app::viewer;
namespace inf = nuka::runtime::inference;

namespace {

constexpr uint32_t kBlc = 13u;        // root + 12 leg joints per dog
constexpr uint32_t kJoints = 12u;
constexpr uint32_t kDecimation = 4u;  // physics steps per policy eval (Config default)

// URDF-order nominal stance the scene cooks each leg to; identity-permutation check
// reads Q in buffer-slot order and compares to this (thigh 0.8/1.0 = front/rear).
constexpr float kDefaultAngles[kJoints] = {0.1f,  0.8f, -1.5f, -0.1f, 0.8f, -1.5f,
                                           0.1f,  1.0f, -1.5f, -0.1f, 1.0f, -1.5f};

// Per-joint URDF effort limits (hip/thigh 23.7, calf 35.55), URDF order.
constexpr float kForceLimit[kJoints] = {23.7f,  23.7f, 35.55f, 23.7f, 23.7f, 35.55f,
                                        23.7f,  23.7f, 35.55f, 23.7f, 23.7f, 35.55f};

// Policy-free PD hold: apply the Go2 gains once and pin every leg slot at the
// nominal stance, isolating the contact/actuator physics from the policy.
class PdHold : public nuka::runtime::app::SceneController {
public:
    PdHold(uint32_t links_per_env, uint32_t num_dogs)
        : L_(links_per_env), dogs_(num_dogs > 0u ? num_dogs : 1u),
          links_per_dog_(L_ / (num_dogs > 0u ? num_dogs : 1u)) {}
    void OnStep(nuka::nk::World& world, uint32_t env) override {
        if (done_) return;
        done_ = true;
        using nuka::nk::FieldId;
        nuka::nk::Data& d = world.GetData();
        const uint64_t base = static_cast<uint64_t>(env) * L_;
        std::vector<float> kp(L_, 0.0f), kd(L_, 0.0f), fl(L_, 0.0f), tgt(L_, 0.0f);
        d.DownloadField(FieldId::DriveTarget, tgt.data(), L_ * 4u, base * 4u);
        for (uint32_t k = 0; k < dogs_; ++k) {
            const uint32_t root = k * links_per_dog_;
            for (uint32_t u = 0; u < kJoints; ++u) {
                const uint32_t s = root + 1u + u;
                kp[s] = 20.0f; kd[s] = 0.5f; fl[s] = kForceLimit[u];
                tgt[s] = kDefaultAngles[u];
            }
        }
        d.UploadField(FieldId::DriveStiffness, kp.data(), L_ * 4u, base * 4u);
        d.UploadField(FieldId::DriveDamping, kd.data(), L_ * 4u, base * 4u);
        d.UploadField(FieldId::DriveForceLimit, fl.data(), L_ * 4u, base * 4u);
        d.UploadField(FieldId::DriveTarget, tgt.data(), L_ * 4u, base * 4u);
    }
private:
    uint32_t L_, dogs_, links_per_dog_;
    bool done_ = false;
};

// World gravity [0,0,-1] rotated into the body frame (Isaac quat_rotate_inverse,
// w-first quat); out[2] near -1 == upright. Mirrors the controller's helper.
inline void ProjectedGravity(float w, float x, float y, float z, float* out) {
    const float vz = -1.0f;
    const float a = w * w - (x * x + y * y + z * z);
    const float cx = y * vz, cy = -x * vz;
    const float qv = z * vz;
    out[0] = -2.0f * w * cx + 2.0f * x * qv;
    out[1] = -2.0f * w * cy + 2.0f * y * qv;
    out[2] = a * vz + 2.0f * z * qv;
}

struct ZStats {
    float zmin = 1e30f, zmax = -1e30f, zmean = 0.0f;
    float rel_min = 1e30f, rel_max = -1e30f;  // base z minus local terrain surface
    uint32_t nan = 0u, in_band = 0u, total = 0u;
};

// Download env-0 BasePose (num_dogs x [pos,quat]) and summarize base z vs the local
// terrain surface, counting dogs inside the sane crouch band [surf+0.08, surf+0.8].
ZStats ReadBaseZ(nuka::nk::World& world, uint32_t num_dogs,
                 const nuka::terrain::HeightField& terrain) {
    std::vector<float> bp(static_cast<size_t>(num_dogs) * 7u, 0.0f);
    world.GetData().DownloadField(nuka::nk::FieldId::BasePose, bp.data(),
                                  static_cast<uint64_t>(num_dogs) * 7u * sizeof(float), 0);
    ZStats s;
    s.total = num_dogs;
    double sum = 0.0;
    for (uint32_t k = 0; k < num_dogs; ++k) {
        const float bx = bp[k * 7u + 0u], by = bp[k * 7u + 1u], bz = bp[k * 7u + 2u];
        if (!std::isfinite(bz)) { ++s.nan; continue; }
        const float surf = nuka::terrain::SampleHeightFieldZ(terrain, bx, by);
        const float rel = bz - surf;
        s.zmin = std::min(s.zmin, bz);
        s.zmax = std::max(s.zmax, bz);
        s.rel_min = std::min(s.rel_min, rel);
        s.rel_max = std::max(s.rel_max, rel);
        sum += bz;
        if (rel >= 0.08f && rel <= 0.8f) ++s.in_band;
    }
    const uint32_t finite = num_dogs - s.nan;
    s.zmean = finite > 0u ? static_cast<float>(sum / finite) : 0.0f;
    return s;
}

void PrintRow(const char* tag, int ctrl, const ZStats& s) {
    std::printf("  %-8s ctrl=%-4d base_z[min/mean/max]=%.3f/%.3f/%.3f  "
                "rel_surf[min/max]=%.3f/%.3f  in_band=%u/%u  nan=%u\n",
                tag, ctrl, s.zmin, s.zmean, s.zmax, s.rel_min, s.rel_max,
                s.in_band, s.total, s.nan);
    std::fflush(stdout);
}

// Host-oracle vs GPU-resident parity: run ONE control step with each controller on
// the SAME initial world state and compare the DriveTarget they write. Neither steps
// physics, so the obs source fields are unchanged between the two evaluations -- the
// delta isolates obs-assembly + MLP + drive-write GPU-vs-host. Gate < 1e-4.
int RunGpuParity(viewer::EditorScene& es, nphi::Backend* backend,
                 const std::string& policy, float vx, float vy, float vw) {
    const uint32_t L = es.caps.links_per_env;
    const uint32_t dogs = es.caps.articulations_per_env;
    inf::Go2PolicyController host_ctrl, gpu_ctrl;
    if (!host_ctrl.Init(policy, L, dogs, es.terrain) ||
        !gpu_ctrl.Init(policy, L, dogs, es.terrain)) {
        std::fprintf(stderr, "[parity] controller init failed\n");
        return 6;
    }
    host_ctrl.SetUseGpu(false);
    if (!gpu_ctrl.UsingGpu()) {
        std::fprintf(stderr, "[parity] GPU path unavailable\n");
        return 6;
    }
    host_ctrl.SetCommand(vx, vy, vw);
    gpu_ctrl.SetCommand(vx, vy, vw);

    std::vector<float> a(L, 0.0f), b(L, 0.0f);
    host_ctrl.OnStep(*es.world, 0);
    nphi::BackendSynchronize(backend);
    es.world->GetData().DownloadField(nuka::nk::FieldId::DriveTarget, a.data(), L * 4u, 0);
    gpu_ctrl.OnStep(*es.world, 0);
    nphi::BackendSynchronize(backend);
    es.world->GetData().DownloadField(nuka::nk::FieldId::DriveTarget, b.data(), L * 4u, 0);

    double maxd = 0.0;
    uint32_t worst = 0;
    for (uint32_t i = 0; i < L; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > maxd) { maxd = d; worst = i; }
    }
    std::printf("[parity] host-vs-GPU DriveTarget: dogs=%u links/env=%u  "
                "max|delta|=%.3e @slot %u (host=%.6f gpu=%.6f)\n",
                dogs, L, maxd, worst, a[worst], b[worst]);
    const bool pass = maxd < 1.0e-4;
    std::printf("[parity] RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Per-control-step policy cost on this device: host OnStep (the D2H + host MLP + H2D
// round-trip) vs GPU OnStep (kernels enqueued on the backend stream). decimation=1 so
// every call computes. The host loop self-synchronizes (DownloadField); the GPU loop
// is enqueue + one final sync (the cost the host frame thread actually pays).
int RunPolicyBench(viewer::EditorScene& es, nphi::Backend* backend,
                   const std::string& policy) {
    using Clock = std::chrono::steady_clock;
    const uint32_t L = es.caps.links_per_env;
    const uint32_t dogs = es.caps.articulations_per_env;
    inf::Go2PolicyController::Config cfg;
    cfg.decimation = 1u;
    inf::Go2PolicyController host_ctrl, gpu_ctrl;
    if (!host_ctrl.Init(policy, L, dogs, es.terrain, cfg) ||
        !gpu_ctrl.Init(policy, L, dogs, es.terrain, cfg)) {
        std::fprintf(stderr, "[bench] controller init failed\n");
        return 6;
    }
    host_ctrl.SetUseGpu(false);
    if (!gpu_ctrl.UsingGpu()) {
        std::fprintf(stderr, "[bench] GPU path unavailable\n");
        return 6;
    }
    const int N = 100;
    host_ctrl.OnStep(*es.world, 0);
    gpu_ctrl.OnStep(*es.world, 0);
    nphi::BackendSynchronize(backend);

    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) host_ctrl.OnStep(*es.world, 0);
    nphi::BackendSynchronize(backend);
    const double host_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / N;

    auto g0 = Clock::now();
    for (int i = 0; i < N; ++i) gpu_ctrl.OnStep(*es.world, 0);
    nphi::BackendSynchronize(backend);
    const double gpu_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - g0).count() / N;

    std::printf("[bench] per-control-step policy cost (dogs=%u, decimation=1):\n", dogs);
    std::printf("[bench]   HOST round-trip : %.3f ms/step\n", host_ms);
    std::printf("[bench]   GPU  enqueue    : %.3f ms/step\n", gpu_ms);
    std::printf("[bench]   speedup         : %.1fx\n",
                gpu_ms > 0.0 ? host_ms / gpu_ms : 0.0);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string scene = "examples/scenes/go2_terrain30.nks";
    std::string policy = "out/go2_terrain_hs/go2_policy.bin";
    float dt = 0.005f, cmd_vx = 1.0f, cmd_vy = 0.0f, cmd_w = 0.0f;
    int control_steps = 400;
    bool hold = false;  // policy-free PD hold diagnostic
    bool validate_gpu = false;  // host-oracle vs GPU-resident DriveTarget parity
    bool bench_policy = false;   // time host round-trip vs GPU enqueue per control step
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) scene = argv[++i];
        else if (a == "--policy" && i + 1 < argc) policy = argv[++i];
        else if (a == "--dt" && i + 1 < argc) dt = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--control-steps" && i + 1 < argc) control_steps = std::atoi(argv[++i]);
        else if (a == "--hold") hold = true;
        else if (a == "--validate-gpu") validate_gpu = true;
        else if (a == "--bench-policy") bench_policy = true;
        else if (a == "--command" && i + 3 < argc) {
            cmd_vx = static_cast<float>(std::atof(argv[++i]));
            cmd_vy = static_cast<float>(std::atof(argv[++i]));
            cmd_w = static_cast<float>(std::atof(argv[++i]));
        }
    }

    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "go2_play_smoke: no physics device\n"); return 3; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "go2_play_smoke: no physics backend\n"); return 3; }

    std::unique_ptr<viewer::EditorScene> es =
        viewer::LoadEditorScene(scene, dev, backend, dt);
    if (!es || !es->sim) {
        std::fprintf(stderr, "go2_play_smoke: load failed: %s\n", scene.c_str());
        return 4;
    }
    const uint32_t num_dogs = es->caps.articulations_per_env;
    std::printf("[smoke] %s  dt=%.4f  dogs=%u  links/env=%u  dof/env=%u  envs=%u\n",
                scene.c_str(), dt, num_dogs, es->caps.links_per_env,
                es->caps.dofs_per_env, es->caps.env_count);
    std::printf("[smoke] caps: bodies/env=%u  max_contacts/env=%u  max_rows/env=%u\n",
                es->caps.bodies_per_env, es->caps.max_contacts_per_env,
                es->caps.max_rows_per_env);
    if (!nuka::terrain::IsValid(es->terrain)) {
        std::fprintf(stderr, "go2_play_smoke: terrain not retained (height-scan impossible)\n");
        return 5;
    }

    if (validate_gpu) return RunGpuParity(*es, backend, policy, cmd_vx, cmd_vy, cmd_w);
    if (bench_policy) return RunPolicyBench(*es, backend, policy);

    inf::Go2PolicyController ctrl;
    PdHold hold_ctrl(es->caps.links_per_env, es->caps.articulations_per_env);
    if (hold) {
        es->sim->SetController(&hold_ctrl);
        std::printf("[smoke] PD-HOLD diagnostic (policy disabled)\n");
    } else {
        if (!ctrl.Init(policy, es->caps.links_per_env, es->caps.articulations_per_env,
                       es->terrain)) {
            std::fprintf(stderr, "go2_play_smoke: controller Init failed (policy=%s)\n",
                         policy.c_str());
            return 6;
        }
        ctrl.SetCommand(cmd_vx, cmd_vy, cmd_w);
        es->sim->SetController(&ctrl);
    }

    // Identity-permutation check: Q is seeded from the cooked nominal stance, so dog
    // 0's leg slots must read the URDF-order default vector when the mapping is identity.
    {
        std::vector<float> q(es->caps.links_per_env, 0.0f);
        es->world->GetData().DownloadField(
            nuka::nk::FieldId::Q, q.data(),
            static_cast<uint64_t>(es->caps.links_per_env) * sizeof(float), 0);
        float maxdev = 0.0f;
        std::printf("[smoke] dog0 leg-slot Q: ");
        for (uint32_t u = 0; u < kJoints; ++u) {
            const float qv = q[1u + u];
            std::printf("%.2f ", qv);
            maxdev = std::max(maxdev, std::fabs(qv - kDefaultAngles[u]));
        }
        std::printf(" max|Q-urdf_default|=%.3f -> joint map %s\n", maxdev,
                    maxdev < 0.05f ? "IDENTITY (ok)" : "NON-IDENTITY (suspect!)");
    }

    std::printf("[smoke] command=[%.2f,%.2f,%.2f]  control_steps=%d  decimation=%u\n",
                cmd_vx, cmd_vy, cmd_w, control_steps, kDecimation);

    ZStats init = ReadBaseZ(*es->world, num_dogs, es->terrain);
    PrintRow("INIT", 0, init);
    // Capture init base x (all dogs face +x, command +x body -> forward = +x).
    std::vector<float> bp0(static_cast<size_t>(num_dogs) * 7u, 0.0f);
    es->world->GetData().DownloadField(nuka::nk::FieldId::BasePose, bp0.data(),
                                       static_cast<uint64_t>(num_dogs) * 7u * 4u, 0);

    ZStats worst;  // running min/max across the whole rollout
    worst.total = num_dogs;
    bool any_step_unhealthy = false;
    for (int c = 0; c < control_steps; ++c) {
        for (uint32_t d = 0; d < kDecimation; ++d) {
            if (!es->sim->FramePublish(nullptr, /*do_step=*/true)) any_step_unhealthy = true;
        }
        if (c + 1 == control_steps) {  // settled contact + foot-penetration diagnostic
            const uint32_t L = es->caps.links_per_env;
            std::vector<float> pose(static_cast<size_t>(L) * 7u, 0.0f);
            es->world->GetData().DownloadField(nuka::nk::FieldId::LinkPose, pose.data(),
                                               static_cast<uint64_t>(L) * 7u * 4u, 0);
            uint32_t cc = 0u, rc = 0u, uc = 0u;
            es->world->GetData().DownloadField(nuka::nk::FieldId::ContactCount, &cc, 4u, 0);
            es->world->GetData().DownloadField(nuka::nk::FieldId::RowCount, &rc, 4u, 0);
            es->world->GetData().DownloadField(nuka::nk::FieldId::UcontactCount, &uc, 4u, 0);
            std::printf("[smoke] SETTLED counts env0: contact=%u row=%u ucontact=%u\n",
                        cc, rc, uc);
            std::printf("[smoke] dog0 calf-link z vs surface (settled):\n");
            for (uint32_t leg = 0; leg < 4u; ++leg) {
                const uint32_t calf = 3u + leg * 3u;  // hip,thigh,CALF per leg
                const float* p = &pose[static_cast<size_t>(calf) * 7u];
                const float surf = nuka::terrain::SampleHeightFieldZ(es->terrain, p[0], p[1]);
                std::printf("    calf%u link_z=%.3f surf=%.3f link-surf=%.3f\n", leg, p[2],
                            surf, p[2] - surf);
            }
        }
        ZStats s = ReadBaseZ(*es->world, num_dogs, es->terrain);
        worst.zmin = std::min(worst.zmin, s.zmin);
        worst.zmax = std::max(worst.zmax, s.zmax);
        worst.rel_min = std::min(worst.rel_min, s.rel_min);
        worst.rel_max = std::max(worst.rel_max, s.rel_max);
        worst.nan = std::max(worst.nan, s.nan);
        if (c + 1 == 100 || c + 1 == control_steps) PrintRow("RUN", c + 1, s);
    }

    // NUKA_WALL_TIMING: clean steady-state wall-clock of the pure physics step
    // (no controller/publish/event-sync) to cross-check the per-op breakdown.
    if (const char* wt = std::getenv("NUKA_WALL_TIMING")) {
        (void)wt;
        const int N = 200;
        nphi::BackendSynchronize(backend);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) es->world->Step();
        nphi::BackendSynchronize(backend);
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count() / N;
        std::printf("[smoke] WALL pure-step: %.3f ms/step  (%.1f fps)  contacts standing\n",
                    ms, ms > 0.0 ? 1000.0 / ms : 0.0);
    }

    {
        std::vector<float> q(es->caps.links_per_env, 0.0f);
        es->world->GetData().DownloadField(
            nuka::nk::FieldId::Q, q.data(),
            static_cast<uint64_t>(es->caps.links_per_env) * sizeof(float), 0);
        std::printf("[smoke] dog0 END leg-slot Q: ");
        for (uint32_t u = 0; u < kJoints; ++u) std::printf("%.2f ", q[1u + u]);
        std::printf("\n");
    }

    ZStats fin = ReadBaseZ(*es->world, num_dogs, es->terrain);
    std::vector<float> bp1(static_cast<size_t>(num_dogs) * 7u, 0.0f);
    es->world->GetData().DownloadField(nuka::nk::FieldId::BasePose, bp1.data(),
                                       static_cast<uint64_t>(num_dogs) * 7u * 4u, 0);
    double dxsum = 0.0; uint32_t dxn = 0u;
    for (uint32_t k = 0; k < num_dogs; ++k) {
        const float dx = bp1[k * 7u] - bp0[k * 7u];
        if (std::isfinite(dx)) { dxsum += dx; ++dxn; }
    }
    const float mean_dx = dxn > 0u ? static_cast<float>(dxsum / dxn) : 0.0f;
    // Base tilt from upright (acos of -projected_gravity_z) -- catches a tipped dog
    // a height band alone misses; a low crouch is upright, a fallen dog is not.
    float tilt_max = 0.0f; double tilt_sum = 0.0; uint32_t upright = 0u;
    for (uint32_t k = 0; k < num_dogs; ++k) {
        const float* q = &bp1[k * 7u + 3u];  // quat wxyz
        float pg[3];
        ProjectedGravity(q[0], q[1], q[2], q[3], pg);
        const float c = pg[2] < -1.0f ? -1.0f : (pg[2] > 1.0f ? 1.0f : pg[2]);
        const float tilt = std::acos(-c) * 57.2958f;  // deg from upright
        tilt_max = std::max(tilt_max, tilt);
        tilt_sum += tilt;
        if (tilt < 35.0f) ++upright;
    }
    const float tilt_mean = num_dogs > 0u ? static_cast<float>(tilt_sum / num_dogs) : 0.0f;
    std::printf("[smoke] tilt(deg) mean=%.1f max=%.1f  upright(<35deg)=%u/%u\n",
                tilt_mean, tilt_max, upright, num_dogs);
    std::printf("[smoke] ROLLUP min_z=%.3f max_z=%.3f  rel_surf[min/max]=%.3f/%.3f  "
                "mean_forward_dx=%.3f m  max_nan=%u  step_unhealthy=%s\n",
                worst.zmin, worst.zmax, worst.rel_min, worst.rel_max, mean_dx, worst.nan,
                any_step_unhealthy ? "yes" : "no");

    // PASS reproduces the reference reality: no NaN, never exploded, never deeply
    // collapsed / fell through, dogs settle to a stable crouch band, and translate
    // forward under the command (closed-loop walking, not a static heap).
    const bool no_nan = worst.nan == 0u;
    const bool no_explode = worst.rel_max < 3.0f && worst.zmax < 50.0f;
    const bool no_fallthrough = worst.rel_min > -0.25f;
    const bool stable_crouch = fin.in_band * 2u >= num_dogs;  // >= half in crouch band
    const bool walks_forward = mean_dx > 0.15f;
    const bool pass = no_nan && no_explode && no_fallthrough && stable_crouch &&
                      walks_forward && !any_step_unhealthy;
    std::printf("[smoke] VERDICT no_nan=%d no_explode=%d no_fallthrough=%d "
                "crouch_band=%u/%u(>=half:%d) walks_forward(dx=%.2f)=%d -> %s\n",
                no_nan, no_explode, no_fallthrough, fin.in_band, num_dogs, stable_crouch,
                mean_dx, walks_forward, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
