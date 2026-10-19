#pragma once
// ---------------------------------------------------------------------------
// nuka::import - Mass properties of collision shapes, summed into a body
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

namespace nuka::import {

// Volume, centroid and unit-density inertia about the centroid, in a shape's own frame.
struct MassProperties {
    double volume = 0.0;
    math::Vec3 centroid = math::Vec3::Zero();
    double inertia[3][3] = {};
};

// Closed forms for the primitives; a TriMesh integrates its closed surface. Planes carry no volume.
MassProperties ShapeMassProperties(const scene::CollisionShapeRecord& shape);

// A body's shapes summed about its origin.
struct BodyMass {
    double mass = 0.0;
    double first[3] = {0.0, 0.0, 0.0};
    double inertia[3][3] = {};
};

// A non-negative `mass` spreads evenly over the shape's volume; otherwise `density` holds.
void AddGeomMass(const scene::CollisionShapeRecord& shape, float density, float mass, BodyMass& body);

// The summed shapes become the body's mass, centre of mass and principal inertia.
void SetBodyMass(const BodyMass& sum, scene::RigidBodyRecord& body);

} // namespace nuka::import
