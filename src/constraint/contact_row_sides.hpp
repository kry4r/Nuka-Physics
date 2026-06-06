#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::ContactRowSides -- per-emitted-row CollidableRef side tags.
// ---------------------------------------------------------------------------
// Split out of row_builder.hpp (v0.8 C5a) so the GPU row solver (a C++17 CUDA
// TU) can consume this small POD struct WITHOUT pulling in row_builder.hpp's
// <span> dependency (std::span is C++20-only and the CUDA standard here is C++17).
// row_builder.hpp re-includes this header, so its API is unchanged.
//
// The emitter (EmitCompliantContactRows) fills ONE entry per appended row, in the
// SAME order as the rows: a row at appended-index i has its sides at
// out_sides[base + i] where base is the RowBuffers row count captured at emitter
// entry. C5's unified solver reads it to dispatch each contact side class-blind
// WITHOUT mutating any production-read struct (Row / RowAnchor / RowBodyPair stay
// byte-identical -> goldens do not move). Production never reads this.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"  // CollidableRef
#include "math/vec3.hpp"              // math::Vec3 (contact_point)

namespace nuka::constraint {

struct ContactRowSides {
    CollidableRef a;  // side-A collidable (type/react/handle from the manifold)
    CollidableRef b;  // side-B collidable
    // v0.8 C5c co-residence: the WORLD contact point of this row (the manifold
    // point.position). The unified solver fills the rigid side's angular contact
    // Jacobian r x n IN-KERNEL from it (r = contact_point - body.position[COM]),
    // since the compliant emitter builds a linear-only row (mirrors the legacy
    // maximal solver computing r x n from body state). Defaults to origin; a
    // hand-built ContactRowSides with a RIGID side whose COM is NOT the origin MUST
    // set this (else the in-kernel r x n is spurious). Articulation / static /
    // particle sides ignore j.angular, so the value is irrelevant for them.
    math::Vec3 contact_point = math::Vec3{0.0f, 0.0f, 0.0f};
};

} // namespace nuka::constraint
