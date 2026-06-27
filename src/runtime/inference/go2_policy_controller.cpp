// ---------------------------------------------------------------------------
// Go2PolicyController -- obs build + policy + drive write (see the header).
// ---------------------------------------------------------------------------

#include "runtime/inference/go2_policy_controller.hpp"

#include <cmath>
#include <cstdio>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/terrain/heightfield_sample.hpp"

namespace nuka::runtime::inference {

namespace {
constexpr uint32_t kBlc = 13u;        // base_link_count: root + 12 leg joints
constexpr uint32_t kJoints = 12u;
constexpr uint32_t kProprio = 48u;

inline float Clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// World gravity [0,0,-1] rotated into the body frame (R(q)^T @ [0,0,-1]); q is
// w-first. Mirrors Isaac/legged_gym quat_rotate_inverse (minus on the cross term).
inline void ProjectedGravity(float w, float x, float y, float z, float* out) {
    const float vz = -1.0f;
    const float a = w * w - (x * x + y * y + z * z);
    const float cx = y * vz, cy = -x * vz, cz = 0.0f;  // cross(qvec, [0,0,-1])
    const float qv = z * vz;                            // qvec . v
    out[0] = -2.0f * w * cx + 2.0f * x * qv;
    out[1] = -2.0f * w * cy + 2.0f * y * qv;
    out[2] = a * vz - 2.0f * w * cz + 2.0f * z * qv;
}
}  // namespace

bool Go2PolicyController::Init(const std::string& policy_path, uint32_t links_per_env,
                              uint32_t articulations_per_env,
                              const ::nuka::terrain::HeightField& terrain,
                              const Config& cfg) {
    cfg_ = cfg;
    terrain_ = terrain;
    ready_ = false;
    gains_applied_ = false;
    step_ = 0;
    if (!policy_.Load(policy_path)) return false;

    links_per_env_ = links_per_env;
    num_dogs_ = articulations_per_env > 0u ? articulations_per_env : 1u;
    links_per_dog_ = num_dogs_ > 0u ? links_per_env / num_dogs_ : links_per_env;
    if (links_per_dog_ < kBlc) {
        std::fprintf(stderr, "[go2_policy] links_per_dog %u < %u\n", links_per_dog_, kBlc);
        return false;
    }
    scan_nx_ = static_cast<uint32_t>(
                   std::lround((cfg_.scan_x_max - cfg_.scan_x_min) / cfg_.scan_res)) + 1u;
    scan_ny_ = static_cast<uint32_t>(
                   std::lround((cfg_.scan_y_max - cfg_.scan_y_min) / cfg_.scan_res)) + 1u;
    scan_n_ = scan_nx_ * scan_ny_;
    if (kProprio + scan_n_ != policy_.ObsDim()) {
        std::fprintf(stderr, "[go2_policy] obs width %u != policy %u\n",
                     kProprio + scan_n_, policy_.ObsDim());
        return false;
    }

    pose_.resize(static_cast<size_t>(links_per_env_) * 7u);
    vel_.resize(static_cast<size_t>(links_per_env_) * 6u);
    q_.resize(links_per_env_);
    qd_.resize(links_per_env_);
    tgt_.resize(links_per_env_);
    bp_.resize(static_cast<size_t>(num_dogs_) * 7u);
    obs_.resize(policy_.ObsDim());
    mu_.resize(policy_.ActDim());
    last_action_.assign(static_cast<size_t>(num_dogs_) * kJoints, 0.0f);

    // Stand up the GPU-resident policy from the SAME weights + contract; production
    // drives it, the host MLP above remains the validation oracle. A GPU init
    // failure degrades to the host path (use_gpu_ stays false).
    Go2GpuContract gc;
    for (uint32_t u = 0; u < kJoints; ++u) {
        gc.default_angles[u] = cfg_.default_angles[u];
        gc.slot_of_urdf[u] = cfg_.slot_of_urdf[u];
        gc.force_limit[u] = cfg_.force_limit[u];
    }
    gc.lin_vel_scale = cfg_.lin_vel_scale;
    gc.ang_vel_scale = cfg_.ang_vel_scale;
    gc.dof_pos_scale = cfg_.dof_pos_scale;
    gc.dof_vel_scale = cfg_.dof_vel_scale;
    gc.cmd_scale[0] = cfg_.cmd_scale[0];
    gc.cmd_scale[1] = cfg_.cmd_scale[1];
    gc.cmd_scale[2] = cfg_.cmd_scale[2];
    gc.command[0] = cfg_.command[0];
    gc.command[1] = cfg_.command[1];
    gc.command[2] = cfg_.command[2];
    gc.action_scale = cfg_.action_scale;
    gc.obs_clip = cfg_.obs_clip;
    gc.action_clip = cfg_.action_clip;
    gc.scan_x_min = cfg_.scan_x_min;
    gc.scan_y_min = cfg_.scan_y_min;
    gc.scan_res = cfg_.scan_res;
    gc.scan_scale = cfg_.scan_scale;
    gc.scan_clip = cfg_.scan_clip;
    gc.scan_z_offset = cfg_.scan_z_offset;
    gc.kp = cfg_.kp;
    gc.kd = cfg_.kd;
    gc.njoints = kJoints;
    gc.proprio_dim = kProprio;
    gc.scan_nx = scan_nx_;
    gc.scan_ny = scan_ny_;
    use_gpu_ = gpu_.Init(policy_, gc, terrain_, links_per_env_, num_dogs_,
                         links_per_dog_);
    if (!use_gpu_) {
        std::fprintf(stderr, "[go2_policy] GPU policy init failed; host fallback\n");
    }
    ready_ = true;
    return true;
}

void Go2PolicyController::ApplyGains(nk::World& world, uint32_t env_index) {
    // Write the trained PD gains + per-joint force limits onto every dog's 12 leg
    // slots (the policy's contract); non-leg slots are untouched. Done once.
    nk::Data& data = world.GetData();
    const uint64_t link_base = static_cast<uint64_t>(env_index) * links_per_env_;
    std::vector<float> kp(links_per_env_, 0.0f), kd(links_per_env_, 0.0f),
        fl(links_per_env_, 0.0f);
    data.DownloadField(nk::FieldId::DriveStiffness, kp.data(),
                       static_cast<uint64_t>(links_per_env_) * 4u, link_base * 4u);
    data.DownloadField(nk::FieldId::DriveDamping, kd.data(),
                       static_cast<uint64_t>(links_per_env_) * 4u, link_base * 4u);
    data.DownloadField(nk::FieldId::DriveForceLimit, fl.data(),
                       static_cast<uint64_t>(links_per_env_) * 4u, link_base * 4u);
    for (uint32_t k = 0; k < num_dogs_; ++k) {
        const uint32_t root = k * links_per_dog_;
        for (uint32_t u = 0; u < kJoints; ++u) {
            const uint32_t s = root + 1u + cfg_.slot_of_urdf[u];
            kp[s] = cfg_.kp;
            kd[s] = cfg_.kd;
            fl[s] = cfg_.force_limit[u];
        }
    }
    data.UploadField(nk::FieldId::DriveStiffness, kp.data(),
                     static_cast<uint64_t>(links_per_env_) * 4u, link_base * 4u);
    data.UploadField(nk::FieldId::DriveDamping, kd.data(),
                     static_cast<uint64_t>(links_per_env_) * 4u, link_base * 4u);
    data.UploadField(nk::FieldId::DriveForceLimit, fl.data(),
                     static_cast<uint64_t>(links_per_env_) * 4u, link_base * 4u);
}

void Go2PolicyController::BuildObs(uint32_t dog, float* obs) const {
    const uint32_t root = dog * links_per_dog_;
    const float* v = &vel_[static_cast<size_t>(root) * 6u];   // [ang.xyz, lin.xyz] body
    const float* p = &pose_[static_cast<size_t>(root) * 7u];  // [pos.xyz, quat.wxyz]
    float pg[3];
    ProjectedGravity(p[3], p[4], p[5], p[6], pg);

    uint32_t o = 0;
    obs[o++] = v[3] * cfg_.lin_vel_scale;
    obs[o++] = v[4] * cfg_.lin_vel_scale;
    obs[o++] = v[5] * cfg_.lin_vel_scale;
    obs[o++] = v[0] * cfg_.ang_vel_scale;
    obs[o++] = v[1] * cfg_.ang_vel_scale;
    obs[o++] = v[2] * cfg_.ang_vel_scale;
    obs[o++] = pg[0];
    obs[o++] = pg[1];
    obs[o++] = pg[2];
    obs[o++] = cfg_.command[0] * cfg_.cmd_scale[0];
    obs[o++] = cfg_.command[1] * cfg_.cmd_scale[1];
    obs[o++] = cfg_.command[2] * cfg_.cmd_scale[2];
    for (uint32_t u = 0; u < kJoints; ++u) {
        const float qv = q_[root + 1u + cfg_.slot_of_urdf[u]];
        obs[o++] = (qv - cfg_.default_angles[u]) * cfg_.dof_pos_scale;
    }
    for (uint32_t u = 0; u < kJoints; ++u) {
        obs[o++] = qd_[root + 1u + cfg_.slot_of_urdf[u]] * cfg_.dof_vel_scale;
    }
    const float* la = &last_action_[static_cast<size_t>(dog) * kJoints];
    for (uint32_t u = 0; u < kJoints; ++u) obs[o++] = la[u];
    for (uint32_t i = 0; i < kProprio; ++i) obs[i] = Clampf(obs[i], -cfg_.obs_clip, cfg_.obs_clip);

    // Height scan (187) from the AUTHORITATIVE base pose (xy/z + yaw).
    const float* b = &bp_[static_cast<size_t>(dog) * 7u];
    const float bx = b[0], by = b[1], bz = b[2];
    const float qw = b[3], qx = b[4], qy = b[5], qz = b[6];
    const float yaw = std::atan2(2.0f * (qw * qz + qx * qy),
                                 1.0f - 2.0f * (qy * qy + qz * qz));
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    for (uint32_t ix = 0; ix < scan_nx_; ++ix) {
        const float px = cfg_.scan_x_min + static_cast<float>(ix) * cfg_.scan_res;
        for (uint32_t iy = 0; iy < scan_ny_; ++iy) {
            const float py = cfg_.scan_y_min + static_cast<float>(iy) * cfg_.scan_res;
            const float wx = bx + (px * cy - py * sy);
            const float wy = by + (px * sy + py * cy);
            const float surf = ::nuka::terrain::SampleHeightFieldZ(terrain_, wx, wy);
            const float val =
                Clampf(bz - cfg_.scan_z_offset - surf, -cfg_.scan_clip, cfg_.scan_clip) *
                cfg_.scan_scale;
            obs[kProprio + ix * scan_ny_ + iy] = val;
        }
    }
}

void Go2PolicyController::OnStep(nk::World& world, uint32_t env_index) {
    if (!ready_) return;

    // GPU-resident path: obs-assembly -> MLP -> drive-write run entirely on the
    // world backend's main stream (it reads the device fields directly), so a
    // control step issues NO device<->host copy. void* keeps this TU CUDA-free.
    if (use_gpu_) {
        void* backend = static_cast<void*>(world.Backend());
        if (!gains_applied_) {
            gpu_.ApplyGains(backend,
                            world.FieldPtr<float>(nk::FieldId::DriveStiffness),
                            world.FieldPtr<float>(nk::FieldId::DriveDamping),
                            world.FieldPtr<float>(nk::FieldId::DriveForceLimit),
                            env_index);
            gains_applied_ = true;
        }
        if ((step_++ % cfg_.decimation) != 0u) return;
        gpu_.Step(backend, world.FieldPtr<float>(nk::FieldId::LinkPose),
                  world.FieldPtr<float>(nk::FieldId::LinkVelocity),
                  world.FieldPtr<float>(nk::FieldId::Q),
                  world.FieldPtr<float>(nk::FieldId::Qdot),
                  world.FieldPtr<float>(nk::FieldId::BasePose),
                  world.FieldPtr<float>(nk::FieldId::DriveTarget), env_index);
        return;
    }

    if (!gains_applied_) {
        ApplyGains(world, env_index);
        gains_applied_ = true;
    }
    // Decimation: recompute the policy every N steps; the PD holds the target between.
    if ((step_++ % cfg_.decimation) != 0u) return;

    nk::Data& data = world.GetData();
    const uint32_t L = links_per_env_;
    const uint64_t link_base = static_cast<uint64_t>(env_index) * L;
    const uint64_t art_base = static_cast<uint64_t>(env_index) * num_dogs_;
    data.DownloadField(nk::FieldId::LinkPose, pose_.data(),
                       static_cast<uint64_t>(L) * 7u * 4u, link_base * 7u * 4u);
    data.DownloadField(nk::FieldId::LinkVelocity, vel_.data(),
                       static_cast<uint64_t>(L) * 6u * 4u, link_base * 6u * 4u);
    data.DownloadField(nk::FieldId::Q, q_.data(), static_cast<uint64_t>(L) * 4u,
                       link_base * 4u);
    data.DownloadField(nk::FieldId::Qdot, qd_.data(), static_cast<uint64_t>(L) * 4u,
                       link_base * 4u);
    data.DownloadField(nk::FieldId::BasePose, bp_.data(),
                       static_cast<uint64_t>(num_dogs_) * 7u * 4u, art_base * 7u * 4u);
    data.DownloadField(nk::FieldId::DriveTarget, tgt_.data(),
                       static_cast<uint64_t>(L) * 4u, link_base * 4u);

    for (uint32_t k = 0; k < num_dogs_; ++k) {
        BuildObs(k, obs_.data());
        policy_.Forward(obs_.data(), mu_.data());
        float* la = &last_action_[static_cast<size_t>(k) * kJoints];
        const uint32_t root = k * links_per_dog_;
        for (uint32_t u = 0; u < kJoints; ++u) {
            const float a = Clampf(mu_[u], -cfg_.action_clip, cfg_.action_clip);
            la[u] = a;
            tgt_[root + 1u + cfg_.slot_of_urdf[u]] =
                cfg_.default_angles[u] + cfg_.action_scale * a;
        }
    }
    data.UploadField(nk::FieldId::DriveTarget, tgt_.data(),
                     static_cast<uint64_t>(L) * 4u, link_base * 4u);
}

}  // namespace nuka::runtime::inference
