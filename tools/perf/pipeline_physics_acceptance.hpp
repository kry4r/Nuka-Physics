#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "nk/pipeline/world.hpp"
#include "nk/readout/energy_ledger.hpp"
#include "nk/readout/physics_diagnostics.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "scene/format/json.hpp"

namespace nuka::perf {

class PipelinePhysicsAcceptance {
    using Json = scene::json::Value;
    using E = nk::EnergyColumn;
    using S = nk::PhysicsStageColumn;
    using C = nk::ContactAuditCount;
    static constexpr size_t kResidualClasses = 4u;
    static constexpr std::array<const char*, nk::kContactAuditCountSize> kCountNames = {
        "contacts", "rows", "invalid_rows", "linear_closure_violations", "cone_violations",
        "negative_normal_impulses", "unmeasured_gaps", "gap_violations"};
    static constexpr std::array<const char*, nk::kContactAuditMetricCount> kMetricNames = {
        "relative_linear_closure", "cone_impulse_excess_Ns", "negative_normal_impulse_Ns",
        "frozen_gap_penetration_m", "normal_displacement_rate_m_s", "tangent_displacement_rate_m_s",
        "gap_penetration_m"};
    static constexpr std::array<const char*, nk::kEnergyColumnCount> kEnergyNames = {
        "begin_kinetic", "begin_gravity", "begin_elastic", "free_kinetic", "solved_kinetic",
        "before_dat_kinetic", "before_dat_gravity", "before_dat_elastic", "end_kinetic",
        "end_gravity", "end_elastic", "drive_work", "aero_work", "external_work",
        "kinematic_elastic_work", "kinematic_contact_work", "friction_loss", "normal_loss",
        "limit_loss", "passive_loss", "rayleigh_loss", "position_potential", "dat_kinetic_loss",
        "dat_potential", "residual", "throughput", "dt", "row_rate_work",
        "row_physical_impulse_work", "unclassified_row_work", "valid"};

    struct Maximum {
        double value = 0.0;
        uint32_t step = 0u, env = 0u, substep = 0u, row = 0u;
        uint64_t samples = 0u;
        void Observe(double sample, uint32_t at_step, uint32_t at_env, uint32_t at_substep, uint32_t at_row = 0u) {
            ++samples;
            if (samples != 1u && sample <= value) return;
            value = sample; step = at_step; env = at_env; substep = at_substep; row = at_row;
        }
        Json Report(bool row_index = false) const {
            Json result = Json::Object();
            result.Set("value", samples != 0u ? Json::Float(value) : Json::Null());
            result.Set("samples", Json::Int(samples));
            result.Set("step", samples != 0u ? Json::Int(step) : Json::Null());
            result.Set("env", samples != 0u ? Json::Int(env) : Json::Null());
            result.Set("substep", samples != 0u ? Json::Int(substep) : Json::Null());
            if (row_index) result.Set("row", samples != 0u ? Json::Int(row) : Json::Null());
            return result;
        }
    };

public:
    explicit PipelinePhysicsAcceptance(nk::World& world) {
        for (nk::FieldId id : {nk::FieldId::EnergyLedger, nk::FieldId::EnergyLedgerStatus,
                              nk::FieldId::PhysicsStageMetrics, nk::FieldId::ContactAuditCounts,
                              nk::FieldId::ContactAuditMetrics, nk::FieldId::SolverResidual})
            if (world.FieldPtr(id) == nullptr) throw std::runtime_error("physics readout demand failed");
        const auto& caps = world.GetModel().capacities;
        envs_ = caps.env_count;
        substeps_ = caps.integration_substeps;
        const size_t records = size_t{envs_} * substeps_;
        energy_.resize(records * nk::kEnergyColumnCount);
        coverage_.resize(records);
        stages_.resize(records * nk::kEnergyStageCount * nk::kPhysicsStageColumnCount);
        counts_.resize(records * nk::kContactAuditCountSize);
        metrics_.resize(records * nk::kContactAuditMetricCount);
        residual_.resize(size_t{envs_} * kResidualClasses);
        dat_.resize(envs_);
        for (const auto& call : world.GetPipeline().Calls()) {
            if (call.op != phi::NkOp::BlockDescentSolve) continue;
            const auto& params = *static_cast<const phi::BlockDescentSolveParams*>(call.params);
            velocity_tolerance_ = params.velocity_tolerance;
            residual_measured_ = params.measure_contact_residual != 0u;
        }
        acceptance_velocity_tolerance_ = velocity_tolerance_ > 0.0f ? velocity_tolerance_ : phi::kTileVelocityTolerance;
    }

    void Observe(nk::World& world, uint32_t step) {
        Read(world, nk::FieldId::EnergyLedger, energy_);
        Read(world, nk::FieldId::EnergyLedgerStatus, coverage_);
        Read(world, nk::FieldId::PhysicsStageMetrics, stages_);
        Read(world, nk::FieldId::ContactAuditCounts, counts_);
        Read(world, nk::FieldId::ContactAuditMetrics, metrics_);
        Read(world, nk::FieldId::SolverResidual, residual_);
        Read(world, nk::FieldId::DatTruncationCount, dat_);
        for (uint32_t env = 0u; env < envs_; ++env) {
            dat_visits_ += dat_[env];
            for (uint32_t substep = 0u; substep < substeps_; ++substep) {
                ++records_;
                const size_t record = size_t{env} * substeps_ + substep;
                const uint32_t status = coverage_[record];
                coverage_union_ |= status;
                for (uint32_t bit = 0u; bit < coverage_counts_.size(); ++bit)
                    if ((status & (1u << bit)) != 0u) ++coverage_counts_[bit];
                nonfinite_ |= (status & nk::energy_status::kNonfinite) != 0u;
                physics_failure_ |= (status & nk::energy_status::kPhysicsFailure) != 0u;
                const float* energy = energy_.data() + record * nk::kEnergyColumnCount;
                if (energy[static_cast<uint32_t>(E::Valid)] != 1.0f) ++unwritten_energy_;
                if (status == 0u && energy[static_cast<uint32_t>(E::Valid)] == 1.0f) ++covered_energy_;
                for (uint32_t c = 0u; c < nk::kEnergyColumnCount; ++c) {
                    if (std::isfinite(energy[c])) { energy_sums_[c] += energy[c]; ++energy_samples_[c]; }
                    else if (status == 0u || (c != static_cast<uint32_t>(E::Residual) &&
                                              c != static_cast<uint32_t>(E::Throughput))) nonfinite_ = true;
                }
                for (uint32_t c = 0u; c < nk::kContactAuditCountSize; ++c) {
                    const uint32_t count = counts_[record * nk::kContactAuditCountSize + c];
                    count_totals_[c] += count;
                    if (count != 0u && c != static_cast<uint32_t>(C::Contacts) &&
                        c != static_cast<uint32_t>(C::Rows) && c != static_cast<uint32_t>(C::UnmeasuredGaps) &&
                        first_contact_failure_.IsNull()) {
                        first_contact_failure_ = Location(step, env, substep);
                        first_contact_failure_.Set("count", Json::Str(kCountNames[c]));
                        first_contact_failure_.Set("value", Json::Int(count));
                    }
                }
                for (uint32_t c = 0u; c < nk::kContactAuditMetricCount; ++c) {
                    const uint64_t packed = metrics_[record * nk::kContactAuditMetricCount + c];
                    if (packed == 0u) continue;
                    const uint32_t bits = static_cast<uint32_t>(packed >> 32u);
                    float value = 0.0f;
                    std::memcpy(&value, &bits, sizeof(value));
                    if (!std::isfinite(value)) { nonfinite_ = true; continue; }
                    contact_max_[c].Observe(value, step, env, substep, ~static_cast<uint32_t>(packed));
                    if ((c == static_cast<uint32_t>(nk::ContactAuditMetric::NormalDisplacementRate) ||
                         c == static_cast<uint32_t>(nk::ContactAuditMetric::TangentDisplacementRate)) &&
                        value > acceptance_velocity_tolerance_) ++work_rate_violations_;
                }
                const float* solved = stages_.data() +
                    (record * nk::kEnergyStageCount + static_cast<uint32_t>(nk::EnergyStage::Solved)) * nk::kPhysicsStageColumnCount;
                for (uint32_t stage = 0u; stage < nk::kEnergyStageCount; ++stage) {
                    const float* values = stages_.data() + (record * nk::kEnergyStageCount + stage) * nk::kPhysicsStageColumnCount;
                    for (uint32_t c = 0u; c < nk::kPhysicsStageColumnCount; ++c)
                        nonfinite_ |= !std::isfinite(values[c]);
                    if (!(values[static_cast<uint32_t>(S::Dt)] > 0.0f)) ++unwritten_stages_;
                    for (S c : {S::NonfiniteParticles, S::NonfiniteBodies, S::NonfiniteLinks})
                        nonfinite_ |= values[static_cast<uint32_t>(c)] != 0.0f;
                }
                const float error = solved[static_cast<uint32_t>(S::MaxVbdMomentumVelocityError)];
                if (std::isfinite(error)) vbd_error_.Observe(error, step, env, substep);
                const float gain = solved[static_cast<uint32_t>(S::MaxPassiveRowEnergyGain)];
                if (std::isfinite(gain)) passive_gain_.Observe(gain, step, env, substep);
            }
            if (!residual_measured_) continue;
            for (uint32_t c = 0u; c < kResidualClasses; ++c) {
                const float value = residual_[size_t{env} * kResidualClasses + c];
                if (!std::isfinite(value)) { nonfinite_ = true; continue; }
                residual_max_[c].Observe(value, step, env, substeps_ - 1u);
                if (value > acceptance_velocity_tolerance_) ++residual_violations_;
            }
        }
    }

    bool HasKnownFailure() const {
        return nonfinite_ || physics_failure_ || residual_violations_ != 0u || work_rate_violations_ != 0u ||
               !first_contact_failure_.IsNull();
    }

    void ObserveRows(nk::World& world, const std::vector<nk::NkRow>& rows,
                     const std::vector<float>& impulses, uint32_t step) {
        for (uint32_t env = 0u; env < envs_; ++env) {
            const size_t record = size_t{env} * substeps_ + substeps_ - 1u;
            const uint64_t packed = metrics_[record * nk::kContactAuditMetricCount +
                static_cast<uint32_t>(nk::ContactAuditMetric::RelativeLinearClosure)];
            const uint32_t bits = static_cast<uint32_t>(packed >> 32u);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            if (!(value > closure_witness_value_)) continue;
            const uint32_t row = ~static_cast<uint32_t>(packed);
            if (row >= rows.size()) throw std::runtime_error("contact audit row is outside the row buffer");
            closure_witness_ = Location(step, env, substeps_ - 1u);
            closure_witness_.Set("row", Json::Int(row));
            closure_witness_.Set("relative_linear_closure", Json::Float(value));
            closure_witness_.Set("impulse_Ns", Json::Float(impulses[row]));
            closure_witness_.Set("side_a", SideReport(world, rows[row].a));
            closure_witness_.Set("side_b", SideReport(world, rows[row].b));
            closure_witness_value_ = value;
        }
    }

    Json Report(bool replay_equal) const {
        Json result = Json::Object(), checks = Json::Object();
        const bool contacts = count_totals_[static_cast<uint32_t>(C::Contacts)] != 0u;
        checks.Set("finite_state_and_execution", Decision(nonfinite_ || physics_failure_, records_ != 0u && unwritten_stages_ == 0u));
        const uint64_t bad_impulses = count_totals_[static_cast<uint32_t>(C::InvalidRows)] +
            count_totals_[static_cast<uint32_t>(C::LinearClosureViolations)] +
            count_totals_[static_cast<uint32_t>(C::ConeViolations)] +
            count_totals_[static_cast<uint32_t>(C::NegativeNormalImpulses)];
        checks.Set("contact_impulse_laws", Decision(bad_impulses != 0u, contacts));
        checks.Set("passive_contact_work_rate", Decision(work_rate_violations_ != 0u, contacts));
        checks.Set("committed_contact_gaps", Decision(count_totals_[static_cast<uint32_t>(C::GapViolations)] != 0u,
            contacts && count_totals_[static_cast<uint32_t>(C::UnmeasuredGaps)] == 0u));
        checks.Set("solver_residual_budget", Decision(residual_violations_ != 0u,
            residual_measured_ && records_ != 0u && substeps_ == 1u));
        checks.Set("full_system_energy", Decision(nonfinite_ || physics_failure_, false));
        checks.Set("analytic_balance", Decision(false, false));
        checks.Set("matched_external_reference", Decision(false, false));
        checks.Set("timestep_and_grid_convergence", Decision(false, false));
        checks.Set("independent_ccd", Decision(false, false));
        checks.Set("timed_replay_binding", Decision(!replay_equal, records_ != 0u));
        bool failed = false, complete = true;
        for (const auto& check : checks.Items()) {
            const auto& status = check.second.At("status").AsString();
            failed |= status == "failed";
            complete &= status == "passed";
        }
        result.Set("status", Json::Str(failed ? "failed" : complete ? "passed" : "unmeasured"));
        result.Set("passes_full_physics_acceptance", Json::Bool(complete));
        result.Set("checks", std::move(checks));
        result.Set("scope", Json::Str("untimed quality replay, every step/environment/physical interval including warmup"));
        result.Set("interval_records", Json::Int(records_));
        result.Set("unwritten_stage_records", Json::Int(unwritten_stages_));
        result.Set("timed_replay_bit_equal", Json::Bool(replay_equal));
        Json counts = Json::Object(), maxima = Json::Object();
        for (uint32_t c = 0u; c < nk::kContactAuditCountSize; ++c) counts.Set(kCountNames[c], Json::Int(count_totals_[c]));
        for (uint32_t c = 0u; c < nk::kContactAuditMetricCount; ++c) maxima.Set(kMetricNames[c], contact_max_[c].Report(true));
        result.Set("contact_count_visits", std::move(counts));
        result.Set("contact_maxima", std::move(maxima));
        result.Set("first_contact_failure", first_contact_failure_);
        result.Set("last_interval_linear_closure_witness", closure_witness_);
        result.Set("contact_work_rate_violations", Json::Int(work_rate_violations_));
        result.Set("contact_thresholds", Json::Str("production audit: linear closure 1e-6 relative, cone factor 1.001, nonnegative normal impulse, configured position slop"));
        Json energy = Json::Object(), terms = Json::Object(), coverage = Json::Array();
        for (uint64_t value : coverage_counts_) coverage.PushBack(Json::Int(value));
        for (uint32_t c = 0u; c < nk::kEnergyColumnCount; ++c) {
            if (c == static_cast<uint32_t>(E::Dt) || c == static_cast<uint32_t>(E::Valid)) continue;
            Json term = Json::Object();
            term.Set("sum_J", energy_samples_[c] != 0u ? Json::Float(energy_sums_[c]) : Json::Null());
            term.Set("finite_records", Json::Int(energy_samples_[c]));
            terms.Set(kEnergyNames[c], std::move(term));
        }
        energy.Set("coverage_status_union", Json::Int(coverage_union_));
        energy.Set("coverage_bit_record_counts", std::move(coverage));
        energy.Set("fully_covered_records", Json::Int(covered_energy_));
        energy.Set("unwritten_records", Json::Int(unwritten_energy_));
        energy.Set("coverage_complete", Json::Bool(records_ != 0u && covered_energy_ == records_ && unwritten_energy_ == 0u));
        energy.Set("balance_criterion_status", Json::Str("unmeasured"));
        energy.Set("coverage_bit_order", Json::Str("nonfinite,unclassified_rows,other_particle_material,grid_material,external_body_load,floating_position_work,rigid_angular_position_work,armature_energy,physics_failure"));
        energy.Set("terms", std::move(terms));
        energy.Set("scope", Json::Str("partial measured terms; incomplete coverage cannot establish full energy balance"));
        result.Set("energy", std::move(energy));
        Json residual = Json::Object();
        const char* names[] = {"contact_normal", "contact_tangent", "joint", "primal_coupling"};
        for (uint32_t c = 0u; c < kResidualClasses; ++c) residual.Set(names[c], residual_max_[c].Report());
        residual.Set("units", Json::Str("m/s"));
        residual.Set("scope", Json::Str("last solve per policy step/environment; earlier substep residuals are not retained"));
        residual.Set("configured_velocity_tolerance_m_s", Json::Float(velocity_tolerance_));
        residual.Set("acceptance_velocity_tolerance_m_s", Json::Float(acceptance_velocity_tolerance_));
        residual.Set("acceptance_budget_source", Json::Str(velocity_tolerance_ > 0.0f ?
            "configured production tolerance" : "production tile tolerance; fixed iteration solve unchanged"));
        residual.Set("budget_violations", Json::Int(residual_violations_));
        result.Set("solver_residual", std::move(residual));
        result.Set("vbd_solved_momentum_velocity_error_m_s", vbd_error_.Report());
        result.Set("passive_row_positive_work_J", passive_gain_.Report());
        result.Set("dat_truncation_count_visits", Json::Int(dat_visits_));
        result.Set("dat_count_scope", Json::Str("last physical interval per policy step/environment; does not cover earlier substeps"));
        return result;
    }

private:
    static Json Vector(math::Vec3 value) {
        Json result = Json::Array();
        result.PushBack(Json::Float(value.x)); result.PushBack(Json::Float(value.y));
        result.PushBack(Json::Float(value.z));
        return result;
    }
    static Json SideReport(nk::World& world, const nk::NkRowSide& side) {
        Json result = Json::Object();
        result.Set("kind", Json::Int(side.kind)); result.Set("index", Json::Int(side.index));
        result.Set("jlin", Vector(side.jlin)); result.Set("jang", Vector(side.jang));
        if (side.kind != nk::kNkSidePointEndpoint) return result;
        const auto& caps = world.GetModel().capacities;
        nk::PointEndpointRange range;
        if (side.index >= uint64_t{caps.env_count} * caps.point_endpoints_per_env ||
            !world.GetData().DownloadField(nk::FieldId::PointEndpointRanges, &range, sizeof(range),
                                           uint64_t{side.index} * sizeof(range)))
            throw std::runtime_error("contact endpoint range download failed");
        if (uint64_t{range.first} + range.count > uint64_t{caps.env_count} * caps.point_endpoint_terms_per_env)
            throw std::runtime_error("contact endpoint terms are outside the term buffer");
        std::vector<nk::PointEndpointTerm> terms(range.count);
        if (!world.GetData().DownloadField(nk::FieldId::PointEndpointTerms, terms.data(),
                terms.size() * sizeof(nk::PointEndpointTerm), uint64_t{range.first} * sizeof(nk::PointEndpointTerm)))
            throw std::runtime_error("contact endpoint terms download failed");
        Json values = Json::Array();
        for (const auto& term : terms) {
            Json value = Json::Object(), columns = Json::Array();
            value.Set("kind", Json::Int(term.kind)); value.Set("index", Json::Int(term.index));
            for (const auto& column : term.column) columns.PushBack(Vector(column));
            value.Set("columns", std::move(columns));
            values.PushBack(std::move(value));
        }
        result.Set("terms", std::move(values));
        return result;
    }
    template <class T> static void Read(nk::World& world, nk::FieldId id, std::vector<T>& values) {
        if (!world.GetData().DownloadField(id, values.data(), values.size() * sizeof(T)))
            throw std::runtime_error("physics readout download failed");
    }
    static Json Decision(bool failed, bool complete) {
        Json result = Json::Object();
        result.Set("status", Json::Str(failed ? "failed" : complete ? "passed" : "unmeasured"));
        result.Set("coverage_complete", Json::Bool(complete));
        return result;
    }
    static Json Location(uint32_t step, uint32_t env, uint32_t substep) {
        Json result = Json::Object();
        result.Set("step", Json::Int(step)); result.Set("env", Json::Int(env));
        result.Set("substep", Json::Int(substep));
        return result;
    }
    uint32_t envs_ = 0u, substeps_ = 0u, coverage_union_ = 0u;
    float velocity_tolerance_ = 0.0f;
    float acceptance_velocity_tolerance_ = phi::kTileVelocityTolerance;
    bool residual_measured_ = false, nonfinite_ = false, physics_failure_ = false;
    uint64_t records_ = 0u, covered_energy_ = 0u, unwritten_energy_ = 0u, residual_violations_ = 0u, dat_visits_ = 0u;
    uint64_t unwritten_stages_ = 0u, work_rate_violations_ = 0u;
    std::array<uint64_t, 9> coverage_counts_{};
    std::array<uint64_t, nk::kContactAuditCountSize> count_totals_{};
    std::array<double, nk::kEnergyColumnCount> energy_sums_{};
    std::array<uint64_t, nk::kEnergyColumnCount> energy_samples_{};
    std::array<Maximum, nk::kContactAuditMetricCount> contact_max_{};
    std::array<Maximum, kResidualClasses> residual_max_{};
    Maximum vbd_error_, passive_gain_;
    Json first_contact_failure_;
    Json closure_witness_;
    float closure_witness_value_ = 0.0f;
    std::vector<float> energy_, stages_, residual_;
    std::vector<uint32_t> coverage_, counts_, dat_;
    std::vector<uint64_t> metrics_;
};

}  // namespace nuka::perf
