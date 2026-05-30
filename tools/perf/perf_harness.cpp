// ---------------------------------------------------------------------------
// perf_harness.cpp -- parameterized perf CLI for the 4096-env Go2 batched step.
//
// v0.3 p01 Wave-2 #4. This is the binary `tools/perf/run_perf_baseline.py`
// invokes per env count. It drives the *real, built* batched articulated
// stepper (gpu::BatchedArticulatedWorld) over the owner golden go2_stand.usda
// scene and emits the engine's PerfRecorder JSON (schema_version + per_tag) to
// STDOUT -- nothing else goes to stdout, so the Python tool can json.loads() it
// verbatim. All progress / diagnostics go to STDERR.
//
//   perf_harness --scene <usda> --envs <N> --steps <M>
//                [--determinism d1] [--warmup <W>] [--perf-json -]
//
// The physics setup (scene-derived hold drives + base-relative foot shapes)
// mirrors CookGo2() in tests/runtime/test_batched_articulated_world_perf_detail
// .cpp -- the proven-stable golden drive config -- so the measured numbers are
// the genuine production step path, not a synthetic micro-benchmark.
//
// Detail (fine-grained) instrumentation is ENABLED for the measured window so
// the per-sub-stage hotspot breakdown (factor_inertia_m_inv, ...) is recorded
// alongside the canonical-tag stages. NOTE: detail-on forces a stream sync per
// scope, which slightly inflates step_total vs. the production (detail-off)
// path; this is fine for a *relative* baseline because the post-optimization
// capture is also detail-on (apples-to-apples). See docs/architecture/
// perf-budget.md.
//
// This file is a measurement harness only. It never edits the physics path and
// never modifies the protected golden scene.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"
#include "import/usd_importer.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace gpu = nuka::runtime::gpu;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

// JSON-escape a device name (GPU names are plain ASCII, but be safe).
std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

struct HoldDrives {
    std::vector<float> targets;
    std::vector<float> stiffness;
    std::vector<float> damping;
    std::vector<float> force_limits;
};

struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    HoldDrives drives;
};

// Mirrors CookGo2() in the perf-detail gtest: scene-derived hold drives and
// base-relative foot shapes -- the proven-stable golden drive config.
CookedGo2 CookGo2(const std::string& scene_path) {
    const auto scene = nuka::import::LoadUsd(scene_path);
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedGo2 result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);

    const auto& shapes = world.template_view.shape_table;
    const uint32_t link_count = result.host.TotalLinkCount();
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == kInvalidLink) {
            continue;
        }
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position
                                : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }

    HoldDrives& d = result.drives;
    d.targets = result.host.q;
    d.stiffness.assign(link_count, 0.0f);
    d.damping.assign(link_count, 0.0f);
    d.force_limits.assign(link_count, 0.0f);
    const auto& actuators = world.template_view.actuator_table;
    const auto& joints = world.template_view.joint_table;
    for (uint32_t actuator = 0u;
         actuator < world.template_view.actuator_count; ++actuator) {
        if (actuator >= actuators.joint_ids.size() ||
            actuator >= actuators.types.size() ||
            actuators.types[actuator] != nuka::scene::ActuatorType::Position) {
            continue;
        }
        const auto joint = actuators.joint_ids[actuator];
        if (joint >= joints.child_bodies.size()) {
            continue;
        }
        const auto child_body = joints.child_bodies[joint];
        uint32_t link = kInvalidLink;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (result.host.link_body[l] == child_body) {
                link = l;
                break;
            }
        }
        if (link == kInvalidLink ||
            result.host.joint_type[link] ==
                articulation::ArticulationJointType::Fixed) {
            continue;
        }
        const float gain = actuator < actuators.gains.size()
                               ? std::fmax(actuators.gains[actuator], 0.0f)
                               : 0.0f;
        const float force_limit =
            actuator < actuators.force_limits.size()
                ? std::fmax(actuators.force_limits[actuator], 0.0f)
                : 0.0f;
        if (gain > 0.0f) {
            d.stiffness[link] = gain;
            d.damping[link] = 2.0f * std::sqrt(gain);
        }
        if (force_limit > 0.0f) {
            d.force_limits[link] = force_limit;
        }
    }
    return result;
}

struct DeviceDrives {
    nuka::phi::Buffer targets;
    nuka::phi::Buffer stiffness;
    nuka::phi::Buffer damping;
    nuka::phi::Buffer force_limits;
};

DeviceDrives UploadDrives(const HoldDrives& base, uint32_t env_count) {
    const size_t per = base.targets.size();
    std::vector<float> targets(per * env_count);
    std::vector<float> stiffness(per * env_count);
    std::vector<float> damping(per * env_count);
    std::vector<float> limits(per * env_count);
    for (uint32_t e = 0u; e < env_count; ++e) {
        for (size_t i = 0u; i < per; ++i) {
            targets[e * per + i] = base.targets[i];
            stiffness[e * per + i] = base.stiffness[i];
            damping[e * per + i] = base.damping[i];
            limits[e * per + i] = base.force_limits[i];
        }
    }
    DeviceDrives d;
    auto up = [](const std::vector<float>& v) {
        nuka::phi::Buffer b(v.size() * sizeof(float),
                            nuka::phi::MemoryKind::Device);
        b.CopyFromHost(v.data(), v.size() * sizeof(float));
        return b;
    };
    d.targets = up(targets);
    d.stiffness = up(stiffness);
    d.damping = up(damping);
    d.force_limits = up(limits);
    return d;
}

[[noreturn]] void Usage(const char* prog, int code) {
    std::fprintf(stderr,
        "usage: %s --scene <usda> --envs <N> --steps <M>\n"
        "          [--determinism d1] [--warmup <W>] [--perf-json -]\n"
        "\n"
        "Drives the real batched articulated Go2 stepper and prints the\n"
        "engine PerfRecorder JSON (schema_version + per_tag, plus gpu/n_envs/\n"
        "steps) to STDOUT. All diagnostics go to STDERR.\n",
        prog);
    std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
    std::string scene = "examples/scenes/go2_stand.usda";
    long envs = 4096;
    long steps = 5000;
    long warmup = -1;  // <0 => auto: max(100, steps/20)
    std::string determinism = "d1";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", name);
                Usage(argv[0], 2);
            }
            return argv[++i];
        };
        if (a == "--scene") {
            scene = need("--scene");
        } else if (a == "--envs") {
            envs = std::strtol(need("--envs").c_str(), nullptr, 10);
        } else if (a == "--steps") {
            steps = std::strtol(need("--steps").c_str(), nullptr, 10);
        } else if (a == "--warmup") {
            warmup = std::strtol(need("--warmup").c_str(), nullptr, 10);
        } else if (a == "--determinism") {
            determinism = need("--determinism");
        } else if (a == "--perf-json") {
            // Only stdout ("-") is supported; the value is accepted and
            // ignored so the documented Python contract works verbatim.
            (void)need("--perf-json");
        } else if (a == "-h" || a == "--help") {
            Usage(argv[0], 0);
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            Usage(argv[0], 2);
        }
    }

    if (envs <= 0 || steps <= 0) {
        std::fprintf(stderr, "error: --envs and --steps must be positive\n");
        return 2;
    }
    if (determinism != "d1") {
        // The batched step path runs the D1 (default) determinism tier; there is
        // no D2 code path here. Reject d2 rather than silently mislabel the
        // recorded "determinism" while running d1 physics (a footgun for the
        // regression comparison in #6).
        std::fprintf(stderr,
            "error: --determinism must be d1 (the batched step path has no "
            "separate d2 tier); got '%s'\n",
            determinism.c_str());
        return 2;
    }
    const uint32_t env_count = static_cast<uint32_t>(envs);
    const uint32_t step_count = static_cast<uint32_t>(steps);
    const uint32_t warmup_count = warmup >= 0
        ? static_cast<uint32_t>(warmup)
        : std::max<uint32_t>(100u, step_count / 20u);

    if (!std::filesystem::exists(scene)) {
        std::fprintf(stderr, "error: scene not found: %s\n", scene.c_str());
        return 1;
    }

    try {
        // GPU identity via cudaGetDeviceProperties (NOT nvidia-smi, which is
        // broken on this box). Default device context is device 0.
        const nuka::phi::DeviceInfo gpu_info = nuka::phi::GetDeviceInfo(0);
        std::fprintf(stderr,
            "[perf_harness] gpu='%s' scene='%s' envs=%u steps=%u warmup=%u "
            "determinism=%s\n",
            gpu_info.name, scene.c_str(), env_count, step_count, warmup_count,
            determinism.c_str());

        const auto context = nuka::phi::MakeDefaultDeviceContext();

        auto cooked = CookGo2(scene);
        auto& base = cooked.host;
        if (base.ArticulationCount() != 1u) {
            std::fprintf(stderr,
                "error: expected exactly 1 articulation in scene, got %u\n",
                base.ArticulationCount());
            return 1;
        }
        const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
        const uint32_t max_dof = dof;

        // Constants match the canonical timing test / perf-detail harness so
        // the measured step is the proven-stable golden config.
        const float kGravityZ = -9.81f;
        const float kDt = 1.0f / 240.0f;
        const float kGround = 0.31f;
        const float kBaumgarteMaxVel = 3.0f;

        auto batched =
            articulation::ReplicateArticulationHostState(base, env_count);
        DeviceDrives dd = UploadDrives(cooked.drives, env_count);

        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        kGround);
        gpu::BatchedArticulatedStepParams params;
        params.drive_targets = static_cast<const float*>(dd.targets.Data());
        params.drive_stiffness = static_cast<const float*>(dd.stiffness.Data());
        params.drive_damping = static_cast<const float*>(dd.damping.Data());
        params.drive_force_limits =
            static_cast<const float*>(dd.force_limits.Data());
        params.gravity_z = kGravityZ;
        params.dt = kDt;
        params.baumgarte_max_velocity = kBaumgarteMaxVel;

        // Warm-up (NOT measured, detail OFF): first-touch allocation, I-cache,
        // GPU clock ramp.
        std::fprintf(stderr, "[perf_harness] warmup %u steps...\n",
                     warmup_count);
        for (uint32_t s = 0u; s < warmup_count; ++s) {
            bw.Step(params);
        }
        context.stream.Synchronize();

        // Measured window: detail ON, fresh recorder. Both the canonical-tag
        // stages and the fine-grained sub-stage hotspots are captured.
        bw.Perf().Reset();
        bw.Perf().SetDetailEnabled(true);
        std::fprintf(stderr, "[perf_harness] measuring %u steps (detail on)...\n",
                     step_count);
        for (uint32_t s = 0u; s < step_count; ++s) {
            bw.Step(params);
        }
        context.stream.Synchronize();

        // Sanity: step_total must have been recorded exactly once per step.
        const auto step_stats = bw.Perf().Stats("step_total");
        if (step_stats.count != step_count) {
            std::fprintf(stderr,
                "warning: step_total count=%llu != steps=%u (recorder "
                "anomaly)\n",
                static_cast<unsigned long long>(step_stats.count), step_count);
        }
        std::fprintf(stderr,
            "[perf_harness] done: step_total p50=%.3f us mean=%.3f us "
            "(per env-step p50=%.4f us)\n",
            step_stats.p50_us, step_stats.mean_us,
            step_stats.p50_us / static_cast<double>(env_count));

        // ---- STDOUT: pure JSON, byte-compatible with PerfRecorder.DumpJson()
        // plus run metadata (gpu / n_envs / steps / determinism). The Python
        // tool only requires "per_tag"; the extra top-level keys are harmless
        // and let the baseline record the device-reported GPU name. ----
        const std::string per_tag_json = bw.Perf().DumpJson();
        // per_tag_json is a full object {"schema_version":1,"per_tag":{...}}.
        // Re-emit it with the metadata keys spliced in before "schema_version".
        std::string out;
        out.reserve(per_tag_json.size() + 256);
        out += "{\n";
        out += "  \"gpu\": \"";
        out += JsonEscape(gpu_info.name);
        out += "\",\n";
        out += "  \"n_envs\": ";
        out += std::to_string(env_count);
        out += ",\n";
        out += "  \"steps\": ";
        out += std::to_string(step_count);
        out += ",\n";
        out += "  \"warmup\": ";
        out += std::to_string(warmup_count);
        out += ",\n";
        out += "  \"determinism\": \"";
        out += JsonEscape(determinism);
        out += "\",\n";
        // Strip the leading "{\n" from DumpJson() and append its body.
        const auto brace = per_tag_json.find('{');
        std::string body = per_tag_json.substr(brace + 1);
        // body begins with "\n  \"schema_version\"..." -- trim a leading
        // newline so indentation stays clean.
        if (!body.empty() && body.front() == '\n') {
            body.erase(body.begin());
        }
        out += body;

        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
