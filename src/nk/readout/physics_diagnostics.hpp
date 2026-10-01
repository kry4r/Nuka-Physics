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
    VbdRowImpulseX, VbdRowImpulseY, VbdRowImpulseZ, VbdBdfParticles, Count
};

inline constexpr uint32_t kPhysicsStageColumnCount =
    static_cast<uint32_t>(PhysicsStageColumn::Count);

enum class ContactAuditCount : uint32_t {
    Contacts, Rows, InvalidRows, LinearClosureViolations,
    ConeViolations, NegativeNormalImpulses, UnmeasuredGaps, GapViolations, Count
};

enum class ContactAuditMetric : uint32_t {
    RelativeLinearClosure, ConeImpulseExcess, NegativeNormalImpulse,
    FrozenGapPenetration, PositiveNormalWork, PositiveTangentWork, GapPenetration, Count
};

inline constexpr uint32_t kContactAuditCountSize = static_cast<uint32_t>(ContactAuditCount::Count);
inline constexpr uint32_t kContactAuditMetricCount = static_cast<uint32_t>(ContactAuditMetric::Count);

}  // namespace nuka::nk
