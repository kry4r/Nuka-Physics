#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::inference::Go2PolicyController -- a SceneController that drives K
// co-resident Go2 closed-loop under a trained locomotion policy. Each control step
// (decimation) it builds the 235-dim height-scan obs per dog (48 proprio + 187 scan
// of the retained terrain grid), runs the MLP policy, and writes PD drive targets
// through the general Data UploadField seam.
//
// The Go2 obs/action contract -- joint slot order, scales, default stance, scan grid,
// PD gains -- is CONFIG DATA on the Config struct, NOT a branch in world.Step() or
// the solver: a different quadruped/policy is a different Config + weights file.
//
// HOST-ONLY (only Data D2H/H2D copies behind the phi vtable; no CUDA tokens).
// ---------------------------------------------------------------------------

#include "runtime/app/scene_controller.hpp"
#include "runtime/inference/gpu_policy.hpp"
#include "runtime/inference/mlp_policy.hpp"
#include "scene/terrain/heightfield.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::inference {

class Go2PolicyController : public app::SceneController {
public:
    // The Go2 locomotion contract DATA (legged_gym FL/FR/RL/RR x hip/thigh/calf).
    // slot_of_urdf[u] = buffer leg-slot of URDF joint u (identity for go2.nks).
    struct Config {
        std::array<uint32_t, 12> slot_of_urdf{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::array<float, 12> default_angles{0.1f,  0.8f, -1.5f, -0.1f, 0.8f, -1.5f,
                                             0.1f,  1.0f, -1.5f, -0.1f, 1.0f, -1.5f};
        std::array<float, 12> force_limit{23.7f,  23.7f, 35.55f, 23.7f, 23.7f, 35.55f,
                                          23.7f,  23.7f, 35.55f, 23.7f, 23.7f, 35.55f};
        float lin_vel_scale = 2.0f, ang_vel_scale = 0.25f;
        float dof_pos_scale = 1.0f, dof_vel_scale = 0.05f;
        std::array<float, 3> cmd_scale{2.0f, 2.0f, 0.25f};
        float action_scale = 0.25f, obs_clip = 100.0f, action_clip = 1.0f;
        float scan_x_min = -0.8f, scan_x_max = 0.8f;
        float scan_y_min = -0.5f, scan_y_max = 0.5f;
        float scan_res = 0.1f, scan_scale = 5.0f, scan_clip = 1.0f;
        float scan_z_offset = 0.5f;  // measured-height reference: base_z - offset - surf
        std::array<float, 3> command{1.0f, 0.0f, 0.0f};  // [vx, vy, wyaw] (forward walk)
        uint32_t decimation = 4u;
        float kp = 20.0f, kd = 0.5f;
    };

    // Load the policy + bind the cooked env layout (links_per_env, K articulations) +
    // the retained terrain. False on load failure or obs-width mismatch (48+scan).
    bool Init(const std::string& policy_path, uint32_t links_per_env,
              uint32_t articulations_per_env,
              const ::nuka::terrain::HeightField& terrain, const Config& cfg);
    // Convenience overload using the default Go2 contract.
    bool Init(const std::string& policy_path, uint32_t links_per_env,
              uint32_t articulations_per_env,
              const ::nuka::terrain::HeightField& terrain) {
        return Init(policy_path, links_per_env, articulations_per_env, terrain, Config{});
    }
    bool Ready() const { return ready_; }
    void SetCommand(float vx, float vy, float wyaw) {
        cfg_.command = {vx, vy, wyaw};
        gpu_.SetCommand(vx, vy, wyaw);
    }

    // Production drives the GPU-resident path (zero host round-trip); the host MLP
    // stays the numeric ORACLE. SetUseGpu(false) forces the host path (validation
    // A/B + the no-CUDA fallback). UsingGpu() reports the resolved choice.
    void SetUseGpu(bool on) { use_gpu_ = on && gpu_.Ready(); }
    bool UsingGpu() const { return use_gpu_; }

    void OnStep(nk::World& world, uint32_t env_index) override;

private:
    void ApplyGains(nk::World& world, uint32_t env_index);
    void BuildObs(uint32_t dog, float* obs) const;

    MlpPolicy policy_;
    Go2GpuPolicy gpu_;
    Config cfg_;
    ::nuka::terrain::HeightField terrain_;
    bool ready_ = false, gains_applied_ = false, use_gpu_ = false;
    uint32_t links_per_env_ = 0, num_dogs_ = 0, links_per_dog_ = 0;
    uint32_t scan_nx_ = 0, scan_ny_ = 0, scan_n_ = 0;
    uint64_t step_ = 0;

    // host scratch: one env's downloaded fields + per-dog obs/action state.
    mutable std::vector<float> pose_, vel_, q_, qd_, bp_, tgt_;
    mutable std::vector<float> obs_, mu_;
    std::vector<float> last_action_;  // num_dogs*12, persists across control steps
};

}  // namespace nuka::runtime::inference
