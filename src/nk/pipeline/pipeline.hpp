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
#include "nk/pipeline/coupling_provider.hpp"  // CouplingProvider / RowCouplingProvider

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
        // Per-world contact PGS sweep count (the general SolveRowsBlockIsland loop).
        // This IS the runtime promotion of the legacy articulated path's named
        // default (runtime::articulation::kContactSolverIterations == 48); kept at
        // 32 here so existing cooked worlds stay byte-identical.
        uint16_t vel_iters = 32;
        // Split-impulse position-correction sweeps (the general PairDriven path).
        // 0 keeps the velocity-only solve; the production default runs a small pass
        // that geometrically expels accumulated penetration without energy
        // injection (a SEPARATE pseudo velocity integrated into position only).
        uint16_t pos_iters = 4;
        float    pos_beta = 0.3f;      // penetration-closing fraction per step.
        float    pos_slop = 0.001f;    // allowed residual penetration (m).
        float    contact_margin = 0.0f;
        uint32_t max_pairs = 0;
        // M3b articulation-pipeline knobs (both production paths use 1/1; the
        // friction/baumgarte values default from the Model, see Build()).
        uint32_t defer_velocity_damping = 1;  // PD drive emits Kp torque only
        uint32_t fold_drive_damping = 1;      // CRBA folds dt*C -> (M+dt*C)^-1
        uint32_t substeps = 1u;
    };

    // Demand mask for pure-readout ops: a readout writes an output field no other
    // op consumes, so it is emitted only when a consumer requested that field.
    enum ReadoutDemand : uint32_t {
        kReadoutContactWrench = 1u << 0,  // ContactForce + LinkContactWrench
    };

    // All emitted physics ops are required; unsupported demands leave no runnable calls.
    phi::Status Build(const Model& model, const SolverConfig& cfg,
               phi::Device* device = nullptr, uint32_t readout_demand = 0u);
    static uint32_t SubstepCount(const Model& model, const SolverConfig& cfg);

    const std::vector<phi::OpCall>& Calls() const { return calls_; }
    size_t Size() const { return calls_.size(); }
    const std::vector<phi::NkOp>& MissingOps() const { return missing_ops_; }

private:
    phi::Status BuildInterval(const Model& model, const SolverConfig& cfg,
                              phi::Device* device, uint32_t readout_demand);
    // The single op-emission helper (capability query + push). The builder and
    // the coupling providers both append through it so the emit semantics match.
    void AddOp(phi::NkOp op, const void* params, phi::Device* device);
    friend struct CouplingBuildCtx;

    // Parameter storage is sized before emission so captured calls retain stable addresses.
    std::vector<phi::OpCall> calls_;
    std::vector<phi::NkOp> missing_ops_;
    std::vector<phi::XpbdProjectParams> p_xpbd_iterations_;
    std::vector<phi::SolveRowsBlockIslandParams> p_solve_iterations_;
    std::vector<phi::AccumulateStepParams> p_accumulate_step_;
    phi::FkLinkVelocitiesParams p_fk_velocity_{};

    // The build-time coupling providers (row path + MLS-MPM grid-transfer path).
    // Owned by the Pipeline (their lifetime parallels the Params PODs); consulted
    // only at Build, never at step time. The grid provider emits its umbrella op
    // ONLY for a cooked MPM medium (build-time gated -> non-MPM op list unchanged).
    RowCouplingProvider row_coupling_provider_;
    MpmCouplingProvider mpm_coupling_provider_;

    // Reused parameter blocks are immutable after the pipeline is built.
    phi::ApplyDrivesParams            p_apply_drives_{};
    phi::ApplyOscDrivesParams         p_apply_osc_{};
    phi::AbaForwardParams             p_aba_{};
    phi::IntegrateVelocityParams      p_int_vel_{};
    phi::SnapshotStepVelocityParams   p_step_velocity_{};
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
    phi::NarrowphaseBodyParticleParams p_np_body_particle_{};  // body<->particle
    phi::NarrowphaseSdfParams         p_np_sdf_{};
    phi::ContactTangentBasisParams    p_tangent_{};
    phi::AssembleRowsParams           p_assemble_{};
    phi::BuildSolveIslandsParams      p_islands_{};  // dynamic CC solve schedule
    phi::ContactWarmStartParams       p_warm_start_prepare_{};
    phi::ContactWarmStartParams       p_warm_start_commit_{};
    phi::SolveRowsBlockIslandParams   p_solve_{};
    phi::AeroDragParams               p_aero_drag_{};
    phi::ParticlePredictParams        p_part_predict_{};
    phi::ParticleProjectionVelocityParams p_part_projection_velocity_{};
    phi::ParticleContactDeltaParams   p_part_contact_delta_{};
    phi::XpbdProjectParams            p_xpbd_{};
    phi::PbfDensityLambdaParams       p_pbf_density_{};
    phi::PbfApplyDeltaParams          p_pbf_apply_{};
    phi::ParticleFinalizeParams       p_part_finalize_{};
    phi::ParticleParticleContactParams p_pp_contact_{};
    phi::MpmParams                    p_mpm_{};
    phi::ReadoutContactWrenchParams   p_readout_{};
    // L1-c: p_union_obs_ (the union-only per-env contact-obs readout params)
    // was DELETED with the UnionCsr path / ReadoutUnionContactObs op.
};

} // namespace nuka::nk
