#pragma once

#include <cstdint>

namespace nuka::nk {

enum class PhysicsStageColumn : uint32_t {
    DynamicMass, LinearMomentumX, LinearMomentumY, LinearMomentumZ,
    AngularMomentumX, AngularMomentumY, AngularMomentumZ,
    VbdMomentumDefectX, VbdMomentumDefectY, VbdMomentumDefectZ,
    VbdAngularDefectX, VbdAngularDefectY, VbdAngularDefectZ,
    MaxVbdMomentumVelocityError, MaxVbdForce,
    NonfiniteParticles, NonfiniteBodies, NonfiniteLinks,
    MinVbdEffectiveDt, MaxVbdEffectiveDt, VbdDynamicParticles, Dt,
    ActiveRows, EnergyInjectingPassiveRows, MaxPassiveRowEnergyGain,
    MaxAbsoluteVertexResidualWork,
    VbdDiscreteMomentumX, VbdDiscreteMomentumY, VbdDiscreteMomentumZ,
    VbdGravityImpulseX, VbdGravityImpulseY, VbdGravityImpulseZ,
    VbdElasticBoundaryImpulseX, VbdElasticBoundaryImpulseY, VbdElasticBoundaryImpulseZ,
    VbdRowImpulseX, VbdRowImpulseY, VbdRowImpulseZ, VbdBdfParticles,
    SolverContactNormalResidual, SolverContactTangentResidual, SolverJointResidual,
    SolverPrimalResidual, SolverResidualMeasured, Count
};

inline constexpr uint32_t kPhysicsStageColumnCount =
    static_cast<uint32_t>(PhysicsStageColumn::Count);

enum class ContactAuditCount : uint32_t {
    Contacts, Rows, InvalidRows, LinearClosureViolations,
    ConeViolations, NegativeNormalImpulses, UnmeasuredGaps, GapViolations, Count
};

enum class ContactAuditMetric : uint32_t {
    RelativeLinearClosure, ConeImpulseExcess, NegativeNormalImpulse,
    FrozenGapPenetration, NormalDisplacementRate, TangentDisplacementRate, GapPenetration, Count
};

inline constexpr uint32_t kContactAuditCountSize = static_cast<uint32_t>(ContactAuditCount::Count);
inline constexpr uint32_t kContactAuditMetricCount = static_cast<uint32_t>(ContactAuditMetric::Count);

enum class VbdSolveAuditColumn : uint32_t {
    ForceX, ForceY, ForceZ,
    NewtonCorrectionX, NewtonCorrectionY, NewtonCorrectionZ,
    LastDirectionX, LastDirectionY, LastDirectionZ,
    LastScale, LastEnergyChange, LastHalvings,
    AcceptedSteps, RejectedSteps, ZeroSlopeSteps, RoundedSteps, TotalSteps,
    LastPrimalForceX, LastPrimalForceY, LastPrimalForceZ, Count
};

// Dense block momentum balance columns, as nuka_momentum_residual_t.
enum class MomentumResidualColumn : uint32_t {
    Velocity, Dof, Force, Scale, Relative, Row, RowTerm, Sweep, Count
};

inline constexpr uint32_t kMomentumResidualColumnCount = static_cast<uint32_t>(MomentumResidualColumn::Count);

inline constexpr uint32_t kVbdSolveAuditColumnCount =
    static_cast<uint32_t>(VbdSolveAuditColumn::Count);

}  // namespace nuka::nk
