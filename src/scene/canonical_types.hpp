#pragma once
// ---------------------------------------------------------------------------
// nuka::scene – Common enums and type aliases for scene representation
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::scene {

using BodyId   = uint32_t;
using JointId  = uint32_t;
using ShapeId  = uint32_t;
using SensorId = uint32_t;
using MaterialId = uint32_t;
using CameraId = uint32_t;
using LightId = uint32_t;
using ActuatorId = uint32_t;

constexpr BodyId  kInvalidBody  = ~uint32_t(0);
constexpr JointId kInvalidJoint = ~uint32_t(0);
constexpr MaterialId kInvalidMaterial = ~uint32_t(0);

enum class JointType : uint8_t {
    Revolute,
    Prismatic,
    Fixed,
    Spherical,
    Free
};

enum class ShapeType : uint8_t {
    Sphere,
    Capsule,
    Box,
    Plane,
    ConvexHull,
    TriMesh,
    HeightField
};

enum class SensorType : uint8_t {
    Imu,
    Lidar,
    Camera,
    ForceTorque,
    Contact,
    FramePose
};

enum class LightType : uint8_t {
    Point,
    Directional,
    Spot,
    Area
};

enum class ActuatorType : uint8_t {
    Motor,
    Position,
    Velocity,
    Force
};

} // namespace nuka::scene
