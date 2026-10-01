#pragma once

#include <cstdint>

namespace nuka::nk {

enum class EnergyStage : uint32_t { Begin, Free, Solved, Projected, End, Count };

enum class EnergyColumn : uint32_t {
    BeginKinetic, BeginGravity, BeginElastic, FreeKinetic, SolvedKinetic,
    BeforeDatKinetic, BeforeDatGravity, BeforeDatElastic,
    EndKinetic, EndGravity, EndElastic,
    DriveWork, AeroWork, ExternalWork, KinematicElasticWork, KinematicContactWork,
    FrictionLoss, NormalLoss, LimitLoss, MimicLoss, PassiveLoss, RayleighLoss,
    PositionPotential, DatKineticLoss, DatPotential, Residual, Throughput, Dt,
    RowRateWork, RowPhysicalImpulseWork, UnclassifiedRowWork, Valid, Count
};

inline constexpr uint32_t kEnergyColumnCount = static_cast<uint32_t>(EnergyColumn::Count);
inline constexpr uint32_t kEnergyStageCount = static_cast<uint32_t>(EnergyStage::Count);

namespace energy_status {
inline constexpr uint32_t kNonfinite = 1u << 0;
inline constexpr uint32_t kUnclassifiedRows = 1u << 1;
inline constexpr uint32_t kOtherParticleMaterial = 1u << 2;
inline constexpr uint32_t kGridMaterial = 1u << 3;
inline constexpr uint32_t kExternalBodyLoad = 1u << 4;
inline constexpr uint32_t kFloatingPositionWork = 1u << 5;
inline constexpr uint32_t kRigidAngularPositionWork = 1u << 6;
inline constexpr uint32_t kArmatureEnergy = 1u << 7;
inline constexpr uint32_t kPhysicsFailure = 1u << 8;
}

}  // namespace nuka::nk
