#pragma once

#include <array>

#include "nk/model/generated/field_ids.hpp"

namespace nuka::nk {

inline constexpr std::array<FieldId, 6> kIntervalParticleFields = {
    FieldId::IntervalParticlePos, FieldId::IntervalParticleVel,
    FieldId::IntervalParticleF, FieldId::IntervalParticleC,
    FieldId::IntervalParticlePlasticF, FieldId::IntervalParticlePlastic};
inline constexpr std::array<FieldId, 3> kIntervalBodyFields = {
    FieldId::IntervalBodyPose, FieldId::IntervalBodyLinearVelocity,
    FieldId::IntervalBodyAngularVelocity};
inline constexpr std::array<FieldId, 5> kIntervalLinkFields = {
    FieldId::IntervalQ, FieldId::IntervalQdot, FieldId::IntervalJointRotation,
    FieldId::IntervalJointAngularVelocity, FieldId::IntervalLinkVelocity};

inline constexpr bool IsIntervalStateField(FieldId id) {
    if (id == FieldId::IntervalStateDt || id == FieldId::IntervalBasePose) return true;
    for (FieldId field : kIntervalParticleFields) if (field == id) return true;
    for (FieldId field : kIntervalBodyFields) if (field == id) return true;
    for (FieldId field : kIntervalLinkFields) if (field == id) return true;
    return false;
}

}  // namespace nuka::nk
