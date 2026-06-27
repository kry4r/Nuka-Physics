#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::inference -- device-resident Go2 policy (CUDA-FREE interface).
//
// GpuMlp is a GENERAL batched MLP forward on the device (RunningMeanStd normalize
// -> [Linear+ELU] x N -> linear head), weights uploaded once from a host MlpPolicy
// and the SAME arithmetic (double-accumulated dot, fp32 ELU) so it reproduces the
// host oracle to ~1e-5. Go2GpuPolicy runs ONE control step entirely on the world
// backend's main stream: obs-assembly kernels read the device fields directly, the
// MLP runs, drive targets are written back -- NO host round-trip (no D2H/H2D).
//
// The go2 specifics (which field maps to which obs slot, the action scale, the
// default stance, the scan grid) are DATA on Go2GpuContract, never baked into a
// kernel body -- a different quadruped/policy is a different contract + weights.
//
// PURE C++ -- zero CUDA tokens. The kernels + all CUDA live in gpu_policy.cu.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka {
namespace terrain {
struct HeightField;
}  // namespace terrain
}  // namespace nuka

namespace nuka::runtime::inference {

class MlpPolicy;

// The Go2 obs/action contract as CUDA-free DATA (the scalars on
// Go2PolicyController::Config). No go2 constant lives in a kernel body; the
// kernels read these. scan_z_offset is the legged_gym measured-height reference
// (base_z - offset - surface), a named datum rather than an inline literal.
struct Go2GpuContract {
    float    default_angles[12]{};
    uint32_t slot_of_urdf[12]{};
    float    force_limit[12]{};
    float    lin_vel_scale = 2.0f, ang_vel_scale = 0.25f;
    float    dof_pos_scale = 1.0f, dof_vel_scale = 0.05f;
    float    cmd_scale[3]{2.0f, 2.0f, 0.25f};
    float    command[3]{1.0f, 0.0f, 0.0f};
    float    action_scale = 0.25f, obs_clip = 100.0f, action_clip = 1.0f;
    float    scan_x_min = -0.8f, scan_y_min = -0.5f;
    float    scan_res = 0.1f, scan_scale = 5.0f, scan_clip = 1.0f, scan_z_offset = 0.5f;
    float    kp = 20.0f, kd = 0.5f;
    uint32_t njoints = 12u, proprio_dim = 48u, scan_nx = 0u, scan_ny = 0u;
};

// General batched MLP on the device. Weights are uploaded once from a host
// MlpPolicy; Forward runs an arbitrary batch through the same math as the host.
class GpuMlp {
public:
    GpuMlp();
    ~GpuMlp();
    GpuMlp(const GpuMlp&) = delete;
    GpuMlp& operator=(const GpuMlp&) = delete;

    bool     InitFromHost(const MlpPolicy& host);
    bool     Ready() const;
    uint32_t ObsDim() const;
    uint32_t ActDim() const;

    // Host-batch forward (uploads obs[batch*ObsDim], runs, downloads out[batch*ActDim],
    // synchronizes). The GPU-vs-host validation seam. False if not Ready.
    bool Forward(const float* obs, float* out, uint32_t batch);

    struct Impl;  // device buffers + the on-stream forward (defined in gpu_policy.cu)

private:
    Impl* impl_ = nullptr;
    friend class Go2GpuPolicy;  // reuses the on-stream device forward
};

// Device-resident Go2 policy core. One control step = obs kernels -> MLP -> drive
// write, all enqueued on the world backend's main stream (ordered after the prior
// physics, before the next) -- zero host round-trip.
class Go2GpuPolicy {
public:
    Go2GpuPolicy();
    ~Go2GpuPolicy();
    Go2GpuPolicy(const Go2GpuPolicy&) = delete;
    Go2GpuPolicy& operator=(const Go2GpuPolicy&) = delete;

    // Upload weights + contract + terrain; size the per-dog obs/action/last-action
    // device buffers. scan_nx/scan_ny on the contract give the height-scan grid;
    // proprio + scan must equal the policy obs width or Init fails.
    bool Init(const MlpPolicy& host, const Go2GpuContract& contract,
              const ::nuka::terrain::HeightField& terrain, uint32_t links_per_env,
              uint32_t num_dogs, uint32_t links_per_dog);
    bool Ready() const;
    void SetCommand(float vx, float vy, float wyaw);

    // Once: write PD gains + per-joint force limits on every dog's leg slots
    // (device kernel; no host copy). `backend` is the opaque phi::Backend*; the
    // three pointers are the env-0 base device pointers of the drive fields.
    void ApplyGains(void* backend, float* drive_stiffness, float* drive_damping,
                    float* drive_force_limit, uint32_t env_index);

    // One control step. Pointers are the env-0 base device pointers of the fields
    // (World::FieldPtr); env_index selects the env. Enqueues on the backend main
    // stream and returns immediately (no sync).
    void Step(void* backend, const float* link_pose, const float* link_velocity,
              const float* q, const float* qdot, const float* base_pose,
              float* drive_target, uint32_t env_index);

private:
    struct GImpl;
    GImpl* g_ = nullptr;
};

}  // namespace nuka::runtime::inference
