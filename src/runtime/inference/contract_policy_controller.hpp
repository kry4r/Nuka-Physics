#pragma once
// A PolicyController for any actor whose training task described it in a contract: the joints it
// drives with their gains and default stance, and the observation terms it reads, each over the
// last `history` decisions, term by term and oldest first. Runs the host MLP.

#include "runtime/inference/mlp_policy.hpp"
#include "runtime/inference/policy_controller.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::inference {

class ContractPolicyController : public PolicyController {
public:
    enum class Term {
        BaseLinVel, BaseAngVel, ProjectedGravity, VelocityCommands, JointPosRel, JointVelRel, LastAction
    };
    struct TermSpec {
        Term term = Term::BaseAngVel;
        uint32_t dims = 0u;
        float scale = 1.0f;
    };
    // Per joint: stiffness N m/rad, damping N m s/rad and effort limit N m.
    struct Contract {
        std::vector<float> defaults, stiffness, damping, effort;
        float action_scale = 0.25f;
        uint32_t decimation = 4u;
        uint32_t history = 1u;
        std::vector<TermSpec> terms;
    };
    // A robot's root link slot and the link slot of each contract joint on it.
    struct Robot {
        uint32_t root = 0u;
        std::vector<uint32_t> joints;
    };

    // The term a contract names; false for one this runner does not build.
    static bool ParseTerm(const std::string& name, Term* out);

    // False, with `error` saying why, when the policy does not load or does not fit the contract.
    bool Init(const std::string& policy_path, uint32_t links_per_env, const std::vector<Robot>& robots,
              const Contract& contract, std::string* error);

    const std::vector<uint32_t>& RootLinks() const override { return roots_; }
    const std::vector<uint32_t>& ActionLinks() const override { return links_; }
    const std::vector<float>& ActionDefaults() const override { return defaults_; }
    uint32_t ObsDim() const override { return policy_.ObsDim(); }
    uint32_t ActDim() const override { return policy_.ActDim(); }
    uint32_t Decimation() const override { return contract_.decimation; }
    float ActionScale() const override { return contract_.action_scale; }
    bool UsingGpu() const override { return false; }
    std::array<float, 3> Command() const override { return command_; }
    void SetCommandVector(const float* cmd, uint32_t count) override;
    void OnStep(nuka_world_handle world, uint32_t env_index) override;

private:
    void ApplyGains(nuka_world_handle world, uint32_t env_index);
    // One decision's terms for robot `r`, in contract order.
    void BuildFrame(uint32_t r, float* frame) const;

    MlpPolicy policy_;
    Contract contract_;
    std::vector<Robot> robots_;
    std::vector<uint32_t> roots_, links_;
    std::vector<float> defaults_;
    std::array<float, 3> command_{0.0f, 0.0f, 0.0f};
    uint32_t links_per_env_ = 0u, frame_ = 0u;  // frame_: the values of one decision's terms
    uint64_t step_ = 0u;
    bool gains_applied_ = false, fresh_ = true;
    std::vector<float> history_;      // per robot: `history` frames, oldest first
    std::vector<float> last_action_;  // per robot: the previous decision's actions
    std::vector<float> pose_, vel_, q_, qd_, tgt_, frame_buf_, obs_, mu_;
};

}  // namespace nuka::runtime::inference
