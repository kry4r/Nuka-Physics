#pragma once
// nk::CouplingProvider — build-time medium↔rigid coupling abstraction; providers
// shape the fixed Pipeline OpCall list once at Build (host-only, zero CUDA tokens).

#include "phi/backend.hpp"     // OpCall
#include "phi/op_schema.hpp"   // NkOp + <Op>Params

namespace nuka::nk {

class Model;
class Pipeline;

// The build-time context a provider appends ops into: the Model, device, resolved
// launch scalars, and pointers to the Pipeline-owned Params PODs (never stack).
struct CouplingBuildCtx {
    const Model*  model = nullptr;
    phi::Device*  device = nullptr;

    // Build-time scalars resolved once in Pipeline::Build (the emission reads
    // these instead of recomputing them, so the derivation stays in ONE place).
    uint32_t family = 0u;
    uint32_t env_count = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t particles_per_env = 0u;
    uint32_t max_contacts_per_env = 0u;
    uint32_t rigid_cap = 0u;
    uint32_t particle_mode = 0u;
    uint32_t coupled_internal = 0u;
    uint32_t particle_count = 0u;
    uint32_t n_soft = 0u;
    float    dt = 0.0f;
    float    contact_margin = 0.0f;
    uint32_t pos_pass = 0u;          // split-impulse pos pass active (finalize)

    // Pipeline-owned Params storage the emissions fill (lifetime == Pipeline).
    phi::NarrowphaseBodyParticleParams* p_np_body_particle = nullptr;
    phi::ParticleFinalizeParams*        p_part_finalize = nullptr;
    phi::ParticleParticleContactParams* p_pp_contact = nullptr;

    // Append one op; defined in pipeline.cpp where Pipeline is complete so the
    // capability query + push match the builder's local `add` lambda exactly.
    void Emit(phi::NkOp op, const void* params) const;

    Pipeline* pipeline = nullptr;
};

// A provider contributes a medium's coupling ops (deposit pre-contact state,
// exchange two-way momentum vs rigid/artic, return the coupled velocity).
struct CouplingProvider {
    virtual ~CouplingProvider() = default;

    // Deposit pre-contact state where the shared solve sees it.
    virtual void PreCouple(const CouplingBuildCtx&) const = 0;

    // Exchange two-way momentum vs rigid/artic into the shared body sink.
    virtual void Couple(const CouplingBuildCtx&) const = 0;

    // Return the coupled velocity without clobbering the two-way impulse.
    virtual void PostCouple(const CouplingBuildCtx&) const = 0;
};

// The row provider: today's body↔particle coupling re-expressed as one provider,
// emitting the EXACT ops the builder emitted inline, in order, with the same PODs.
struct RowCouplingProvider final : CouplingProvider {
    void PreCouple(const CouplingBuildCtx&) const override;
    void Couple(const CouplingBuildCtx&) const override;
    void PostCouple(const CouplingBuildCtx&) const override;
};

} // namespace nuka::nk
