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
constexpr ShapeId kInvalidShape = ~uint32_t(0);   // v0.8 C1b: <pair> geom sentinel
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

// Convex-decomposition intent authored on a (mesh) collision shape. Consumed
// by the cooker (V-HACD). Mirrors nuka::import::cooker::DecomposeMode but lives
// here so the scene IR / cooked blob carry it without depending on the cooker
// library's headers.
enum class DecomposeMode : uint8_t {
    Auto,   // cooker decides (already-convex => 1 piece; concave => V-HACD)
    Force,  // always run V-HACD
    Skip,   // treat the source mesh as a single convex piece (its own hull)
};

} // namespace nuka::scene
