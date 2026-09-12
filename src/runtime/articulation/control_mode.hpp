#pragma once

#include "phi/articulation_contract.hpp"

namespace nuka::runtime::articulation {

using ControlMode = phi::ArticulationControlMode;

inline constexpr bool IsControlModeImplemented(uint8_t mode) {
    return mode <= static_cast<uint8_t>(ControlMode::Actuator);
}

}  // namespace nuka::runtime::articulation
