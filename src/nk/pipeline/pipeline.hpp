#pragma once
// ---------------------------------------------------------------------------
// nk::Pipeline — the build-time OpCall list (plan §3.2).
//
// A physics step is a FIXED linear sequence of ops (not an arbitrary DAG).
// Pipeline::Build(Model) emits std::vector<phi::OpCall> in the §3.2 canonical
// order, INCLUDING an op only when the Model has that system (no particles =>
// no Particle* ops; no articulation DOFs => no ABA/CRBA ops; etc.). Pipeline
// OWNS the per-op Params POD storage so the `params` pointers in the OpCalls
// stay valid for the Pipeline's lifetime (the World plan-capture contract).
//
// PURE C++ — zero CUDA tokens.
// ---------------------------------------------------------------------------

#include <vector>

#include "phi/backend.hpp"     // OpCall
#include "phi/op_schema.hpp"   // NkOp + <Op>Params

namespace nuka::nk {

class Model;

class Pipeline {
public:
    Pipeline() = default;

    // Build the per-step op list from the Model's systems. dt / gravity / iters
    // come from the SolverConfig (carried on the Model in later milestones; M3a
    // takes them as explicit args so the test can drive a tiny scene).
    struct SolverConfig {
        float    dt = 1.0f / 240.0f;
        float    gravity[3] = {0.0f, 0.0f, -9.81f};
        uint16_t vel_iters = 32;
        uint16_t pos_iters = 0;
        float    contact_margin = 0.0f;
        uint32_t max_pairs = 0;
        // M3b articulation-pipeline knobs (both production paths use 1/1; the
        // friction/baumgarte values default from the Model, see Build()).
        uint32_t defer_velocity_damping = 1;  // PD drive emits Kp torque only
        uint32_t fold_drive_damping = 1;      // CRBA folds dt*C -> (M+dt*C)^-1
    };

    // `device` (optional) enables the ggml-style capability query (§3.1
    // supports_op): an op the backend has no implementation for is NOT emitted
    // (M3b: the M5 broadphase + NarrowphaseSdf and M6 particle ops). This keeps
    // Step() all-Ok and the CUDA-graph plan capturable while later milestones
    // light the ops up — the op order itself stays the §3.2 fixed order.
    void Build(const Model& model, const SolverConfig& cfg,
               phi::Device* device = nullptr);

    const std::vector<phi::OpCall>& Calls() const { return calls_; }
    size_t Size() const { return calls_.size(); }

private:
    // Append an op with its params; the params live in `param_storage_` (a node-
    // stable deque-like vector of variant blocks) so addresses never move after
    // Build returns. We keep one typed vector per param POD; pointers into a
    // reserved vector are stable because we reserve to the worst-case count
    // (each op appears at most once per step).
    std::vector<phi::OpCall> calls_;

    // Stable param storage: one slot per op kind. Reserved so push_back never
    // reallocates (each op is emitted at most once). Plain value members keep
    // the addresses stable for the Pipeline's lifetime.
    phi::ApplyDrivesParams            p_apply_drives_{};
    phi::AbaForwardParams             p_aba_{};
    phi::IntegrateVelocityParams      p_int_vel_{};
    phi::FkWorldPosesParams           p_fk_{};
    phi::IntegratePositionParams      p_int_pos_{};
    phi::CrbaComputeMParams           p_crba_m_{};
    phi::CrbaFactorMParams            p_crba_factor_{};
    phi::ApplyImplicitDampingParams   p_apply_damping_{};  // L1-b standalone damping
    phi::SyncLinkBodyPoseParams       p_sync_body_pose_{};  // general contact B2
    phi::BuildAabbsParams             p_aabbs_{};
    phi::LbvhBuildParams              p_lbvh_build_{};
    phi::LbvhQueryPairsParams         p_lbvh_query_{};
    phi::ParticleGridBuildParams      p_grid_{};
    phi::NarrowphasePrimitivesParams  p_np_prim_{};
    phi::NarrowphaseHeightfieldParams p_np_hf_{};    // general contact H3 heightfield
    phi::NarrowphaseSdfParams         p_np_sdf_{};
    phi::ContactTangentBasisParams    p_tangent_{};
    phi::AssembleRowsParams           p_assemble_{};
    phi::SolveRowsBlockIslandParams   p_solve_{};
    phi::ParticlePredictParams        p_part_predict_{};
    phi::XpbdProjectParams            p_xpbd_{};
    phi::PbfDensityLambdaParams       p_pbf_density_{};
    phi::PbfApplyDeltaParams          p_pbf_apply_{};
    phi::ParticleFinalizeParams       p_part_finalize_{};
    phi::ParticleParticleContactParams p_pp_contact_{};
    phi::ReadoutContactWrenchParams   p_readout_{};
    // L1-c: p_union_obs_ (the union-only per-env contact-obs readout params)
    // was DELETED with the UnionCsr path / ReadoutUnionContactObs op.
};

} // namespace nuka::nk
