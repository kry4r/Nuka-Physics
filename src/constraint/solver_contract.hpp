#pragma once

#include <cstdint>

namespace nuka::constraint {

enum class LawSource : uint32_t { None, MuJoCo };
enum class FrictionCone : uint32_t { Pyramidal, Elliptic };
enum class Integrator : uint32_t { Euler, Rk4, Implicit, ImplicitFast };

struct SolverOptions {
    FrictionCone cone = FrictionCone::Pyramidal;
    Integrator integrator = Integrator::Euler;
    float impratio = 1.0f;
    bool refsafe = true;
    // A MuJoCo source integrates articulation damping with its integrator after the constraint solve.
    LawSource law_source = LawSource::None;
};

struct SolverReference {
    float solref[2] = {0.02f, 1.0f};
    float solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
};

}  // namespace nuka::constraint
