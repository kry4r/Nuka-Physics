// ContractPolicyController -- contract observations, host MLP and drive writes (see the header).

#include "runtime/inference/contract_policy_controller.hpp"

#include <algorithm>
#include <cstring>

namespace nuka::runtime::inference {

namespace {
// The training envs clamp actions here and nowhere tighter.
constexpr float kActionClip = 100.0f;

struct TermName {
    const char* name;
    ContractPolicyController::Term term;
};
constexpr TermName kTerms[] = {
    {"base_lin_vel", ContractPolicyController::Term::BaseLinVel},
    {"base_ang_vel", ContractPolicyController::Term::BaseAngVel},
    {"projected_gravity", ContractPolicyController::Term::ProjectedGravity},
    {"velocity_commands", ContractPolicyController::Term::VelocityCommands},
    {"joint_pos_rel", ContractPolicyController::Term::JointPosRel},
    {"joint_vel_rel", ContractPolicyController::Term::JointVelRel},
    {"last_action", ContractPolicyController::Term::LastAction},
};

const char* NameOf(ContractPolicyController::Term term) {
    for (const TermName& t : kTerms)
        if (t.term == term) return t.name;
    return "";
}

bool PerJoint(ContractPolicyController::Term term) {
    return term == ContractPolicyController::Term::JointPosRel || term == ContractPolicyController::Term::JointVelRel ||
           term == ContractPolicyController::Term::LastAction;
}

// v in the frame of the unit quaternion q (w, x, y, z), as Isaac Lab's quat_rotate_inverse.
void RotateInverse(const float* q, const float* v, float* out) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float a = 2.0f * w * w - 1.0f;
    const float dot = 2.0f * (x * v[0] + y * v[1] + z * v[2]);
    const float c[3] = {y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]};
    out[0] = v[0] * a - c[0] * 2.0f * w + x * dot;
    out[1] = v[1] * a - c[1] * 2.0f * w + y * dot;
    out[2] = v[2] * a - c[2] * 2.0f * w + z * dot;
}

void Download(nuka_world_handle world, nuka_state_field_t field, std::vector<float>* out, size_t per_link,
              size_t links, size_t first) {
    nuka_world_download_field(world, field, out->data(), links * per_link * 4u, first * per_link * 4u);
}
}  // namespace

bool ContractPolicyController::ParseTerm(const std::string& name, Term* out) {
    for (const TermName& t : kTerms) {
        if (name == t.name) {
            *out = t.term;
            return true;
        }
    }
    return false;
}

bool ContractPolicyController::Init(const std::string& policy_path, uint32_t links_per_env,
                                    const std::vector<Robot>& robots, const Contract& contract, std::string* error) {
    if (!policy_.Load(policy_path)) {
        *error = "cannot load the policy '" + policy_path + "'";
        return false;
    }
    const uint32_t joints = static_cast<uint32_t>(contract.defaults.size());
    if (policy_.ActDim() != joints) {
        *error = "the policy has " + std::to_string(policy_.ActDim()) + " actions; its contract drives " +
                 std::to_string(joints) + " joints";
        return false;
    }
    uint32_t frame = 0u;
    for (const TermSpec& t : contract.terms) {
        const uint32_t dims = PerJoint(t.term) ? joints : 3u;
        if (t.dims != dims) {
            *error = std::string("the contract's ") + NameOf(t.term) + " has " + std::to_string(t.dims) +
                     " values; the runner builds " + std::to_string(dims);
            return false;
        }
        frame += dims;
    }
    if (frame * contract.history != policy_.ObsDim()) {
        *error = "the policy observes " + std::to_string(policy_.ObsDim()) + " values; its contract builds " +
                 std::to_string(frame * contract.history);
        return false;
    }
    contract_ = contract;
    robots_ = robots;
    links_per_env_ = links_per_env;
    frame_ = frame;
    roots_.clear();
    links_.clear();
    defaults_.clear();
    for (const Robot& r : robots_) {
        roots_.push_back(r.root);
        links_.insert(links_.end(), r.joints.begin(), r.joints.end());
        defaults_.insert(defaults_.end(), contract_.defaults.begin(), contract_.defaults.end());
    }
    const size_t n = robots_.size();
    history_.assign(n * contract_.history * frame_, 0.0f);
    last_action_.assign(n * joints, 0.0f);
    pose_.resize(size_t{links_per_env_} * 7u);
    vel_.resize(size_t{links_per_env_} * 6u);
    q_.resize(links_per_env_);
    qd_.resize(links_per_env_);
    tgt_.resize(links_per_env_);
    frame_buf_.resize(frame_);
    obs_.resize(policy_.ObsDim());
    mu_.resize(joints);
    step_ = 0u;
    gains_applied_ = false;
    fresh_ = true;
    return true;
}

void ContractPolicyController::SetCommandVector(const float* cmd, uint32_t count) {
    for (uint32_t k = 0; k < 3u; ++k) command_[k] = k < count ? cmd[k] : 0.0f;
}

void ContractPolicyController::ApplyGains(nuka_world_handle world, uint32_t env_index) {
    // The contract's gains on its joints; the other links keep theirs.
    const size_t first = size_t{env_index} * links_per_env_;
    std::vector<float> kp(links_per_env_), kd(links_per_env_), fl(links_per_env_);
    Download(world, NUKA_FIELD_DRIVE_STIFFNESS, &kp, 1u, links_per_env_, first);
    Download(world, NUKA_FIELD_DRIVE_DAMPING, &kd, 1u, links_per_env_, first);
    Download(world, NUKA_FIELD_DRIVE_FORCE_LIMIT, &fl, 1u, links_per_env_, first);
    for (const Robot& r : robots_) {
        for (size_t u = 0; u < r.joints.size(); ++u) {
            kp[r.joints[u]] = contract_.stiffness[u];
            kd[r.joints[u]] = contract_.damping[u];
            fl[r.joints[u]] = contract_.effort[u];
        }
    }
    const size_t bytes = size_t{links_per_env_} * 4u;
    nuka_world_upload_field(world, NUKA_FIELD_DRIVE_STIFFNESS, kp.data(), bytes, first * 4u);
    nuka_world_upload_field(world, NUKA_FIELD_DRIVE_DAMPING, kd.data(), bytes, first * 4u);
    nuka_world_upload_field(world, NUKA_FIELD_DRIVE_FORCE_LIMIT, fl.data(), bytes, first * 4u);
}

void ContractPolicyController::BuildFrame(uint32_t r, float* frame) const {
    const Robot& robot = robots_[r];
    const float* p = &pose_[size_t{robot.root} * 7u];  // [pos.xyz, quat.wxyz]
    const float* v = &vel_[size_t{robot.root} * 6u];   // [ang.xyz, lin.xyz], body frame
    const size_t joints = robot.joints.size();
    const float* last = &last_action_[r * joints];
    uint32_t o = 0u;
    for (const TermSpec& t : contract_.terms) {
        switch (t.term) {
        case Term::BaseLinVel:
            for (int k = 0; k < 3; ++k) frame[o + k] = v[3 + k] * t.scale;
            break;
        case Term::BaseAngVel:
            for (int k = 0; k < 3; ++k) frame[o + k] = v[k] * t.scale;
            break;
        case Term::ProjectedGravity: {
            const float down[3] = {0.0f, 0.0f, -1.0f};
            RotateInverse(p + 3, down, frame + o);
            for (int k = 0; k < 3; ++k) frame[o + k] *= t.scale;
            break;
        }
        case Term::VelocityCommands:
            for (int k = 0; k < 3; ++k) frame[o + k] = command_[k] * t.scale;
            break;
        case Term::JointPosRel:
            for (size_t u = 0; u < joints; ++u) frame[o + u] = (q_[robot.joints[u]] - contract_.defaults[u]) * t.scale;
            break;
        case Term::JointVelRel:
            for (size_t u = 0; u < joints; ++u) frame[o + u] = qd_[robot.joints[u]] * t.scale;
            break;
        case Term::LastAction:
            for (size_t u = 0; u < joints; ++u) frame[o + u] = last[u] * t.scale;
            break;
        }
        o += t.dims;
    }
}

void ContractPolicyController::OnStep(nuka_world_handle world, uint32_t env_index) {
    if (!gains_applied_) {
        ApplyGains(world, env_index);
        gains_applied_ = true;
    }
    if ((step_++ % contract_.decimation) != 0u) return;

    const size_t first = size_t{env_index} * links_per_env_;
    Download(world, NUKA_FIELD_ARTICULATION_LINK_POSE, &pose_, 7u, links_per_env_, first);
    Download(world, NUKA_FIELD_LINK_VELOCITY, &vel_, 6u, links_per_env_, first);
    Download(world, NUKA_FIELD_JOINT_POSITION, &q_, 1u, links_per_env_, first);
    Download(world, NUKA_FIELD_JOINT_VELOCITY, &qd_, 1u, links_per_env_, first);
    Download(world, NUKA_FIELD_DRIVE_TARGET, &tgt_, 1u, links_per_env_, first);

    const uint32_t history = contract_.history;
    const size_t joints = contract_.defaults.size();
    for (uint32_t r = 0; r < robots_.size(); ++r) {
        BuildFrame(r, frame_buf_.data());
        float* h = &history_[size_t{r} * history * frame_];
        // The first decision sees its frame as the whole history, as a training reset does.
        if (fresh_) {
            for (uint32_t s = 0; s < history; ++s) std::copy(frame_buf_.begin(), frame_buf_.end(), h + s * frame_);
        } else {
            std::memmove(h, h + frame_, size_t{history - 1u} * frame_ * sizeof(float));
            std::copy(frame_buf_.begin(), frame_buf_.end(), h + size_t{history - 1u} * frame_);
        }
        uint32_t o = 0u, start = 0u;
        for (const TermSpec& t : contract_.terms) {
            for (uint32_t s = 0; s < history; ++s) {
                std::copy_n(h + size_t{s} * frame_ + start, t.dims, &obs_[o]);
                o += t.dims;
            }
            start += t.dims;
        }
        policy_.Forward(obs_.data(), mu_.data());
        float* last = &last_action_[r * joints];
        const Robot& robot = robots_[r];
        for (size_t u = 0; u < joints; ++u) {
            last[u] = std::clamp(mu_[u], -kActionClip, kActionClip);
            tgt_[robot.joints[u]] = contract_.defaults[u] + contract_.action_scale * last[u];
        }
    }
    fresh_ = false;
    nuka_world_upload_field(world, NUKA_FIELD_DRIVE_TARGET, tgt_.data(), size_t{links_per_env_} * 4u, first * 4u);
}

}  // namespace nuka::runtime::inference
