#pragma once

#include "sensor/state_types.hpp"

#if defined(__CUDACC__)
#define NUKA_WRENCH_HD __host__ __device__
#else
#define NUKA_WRENCH_HD
#endif

namespace nuka::sensor {
namespace wrench_detail {

struct Vector {
    double x, y, z;
    NUKA_WRENCH_HD Vector operator+(Vector b) const { return {x + b.x, y + b.y, z + b.z}; }
    NUKA_WRENCH_HD Vector operator-(Vector b) const { return {x - b.x, y - b.y, z - b.z}; }
    NUKA_WRENCH_HD Vector operator*(double s) const { return {x * s, y * s, z * s}; }
    NUKA_WRENCH_HD Vector Cross(Vector b) const { return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x}; }
    NUKA_WRENCH_HD math::Vec3 Float() const { return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}; }
};

NUKA_WRENCH_HD inline Vector From(math::Vec3 v) { return {v.x, v.y, v.z}; }

NUKA_WRENCH_HD inline Vector Rotate(math::Quat q, Vector v, bool inverse = false) {
    const double sign = inverse ? -1.0 : 1.0;
    const Vector axis{sign * q.x, sign * q.y, sign * q.z};
    const auto twice = axis.Cross(v) * 2.0;
    return v + twice * q.w + axis.Cross(twice);
}

NUKA_WRENCH_HD inline Vector SpinMomentum(const MotionFrame& frame, const float* inertia,
                                          double mass, Vector center) {
    const auto w = Rotate(frame.pose.rotation, From(frame.angular_velocity), true);
    const Vector origin_momentum{
        inertia[0] * w.x + inertia[1] * w.y + inertia[2] * w.z,
        inertia[6] * w.x + inertia[7] * w.y + inertia[8] * w.z,
        inertia[12] * w.x + inertia[13] * w.y + inertia[14] * w.z};
    return Rotate(frame.pose.rotation, origin_momentum - center.Cross(w.Cross(center)) * mass);
}

}  // namespace wrench_detail

// The live spatial inertia uses angular/linear ordering in the link frame.
// Midpoint lever arms approximate distributed loads while excluding position-correction pseudo velocities.
NUKA_WRENCH_HD inline WrenchImpulse InertialLoadImpulse(const MotionFrame& before,
    const MotionFrame& after, const float* inertia, math::Vec3 gravity, double interval) {
    using namespace wrench_detail;
    const double mass = inertia[21];
    const Vector center = mass > 0.0 ? Vector{inertia[16] / mass, inertia[5] / mass, inertia[9] / mass} : Vector{};
    const auto first_arm = Rotate(before.pose.rotation, center);
    const auto last_arm = Rotate(after.pose.rotation, center);
    const auto first_velocity = From(before.linear_velocity) + From(before.angular_velocity).Cross(first_arm);
    const auto last_velocity = From(after.linear_velocity) + From(after.angular_velocity).Cross(last_arm);
    const auto impulse = (last_velocity - first_velocity - From(gravity) * interval) * mass;
    const auto spin_change = SpinMomentum(after, inertia, mass, center) - SpinMomentum(before, inertia, mass, center);
    const auto moment = spin_change + ((first_arm + last_arm) * 0.5).Cross(impulse);
    return {impulse.Float(), moment.Float()};
}

}  // namespace nuka::sensor

#undef NUKA_WRENCH_HD
