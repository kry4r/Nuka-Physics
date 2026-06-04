#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::row_builder -- contact/joint/drive to CSR rows
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"
#include "constraint/row_buffers.hpp"
#include "math/transform.hpp"

#include <span>
#include <vector>

namespace nuka::constraint {

// ---------------------------------------------------------------------------
// v0.8 C4b -- compliant-normal + linearized-pyramid-friction row emission.
// ---------------------------------------------------------------------------
// SCOPE DEVIATION (controller + advisor RULED): the C4b spec text says "Replace
// AppendContactGroup". It is DELIBERATELY NOT replaced here. The compliance +
// 4-edge pyramid changes solver behavior and MOVES the contact goldens; per
// Appendix C + C4b's OWN re-baseline risk note (detailed-tasks §C4b "Pyramid
// (4 edges) ... contributes to the C5 golden re-baseline"), that re-baseline is
// scheduled at C5, NOT C4b. So this is a NEW, validated-NOT-wired emitter that
// matches the whole spine's "validated-not-wired until C5" discipline (C2
// broadphase, C3 dispatch -- all unwired, goldens unmoved). AppendContactGroup,
// BOTH BuildContactRows overloads, and the production call site
// (runtime/world_stepper.cpp:665 BuildContactRows(manifolds,&rows)) are left
// byte-untouched.
//   NAMED DOWNSTREAM CONSUMER = C5: it flips world_stepper:665 to call this
//   emitter and re-baselines the contact goldens.
// ---------------------------------------------------------------------------

// The per-contact-row INPUT seam that C5 wires (C4b takes them as parameters; a
// test supplies known values). Per the C4a/MuJoCo sign + invweight convention:
//   - vel       : the SIGNED constraint-space NORMAL relative velocity Jqvel
//                 (enters aref signed). C5 supplies it from the integrated
//                 state at solve time.
//   - invweight : the reaction inverse-weight body_invweight0[a] +
//                 body_invweight0[b] from the reaction providers (C4c). C5
//                 sources it per pair from the resolved providers.
//   - dt        : the substep dt (drives the REFSAFE timeconst clamp in C4a).
//   - condim    : the friction contact dimensionality. condim is NOT carried on
//                 ContactManifold; C5 supplies it per pair (or uniform).
//                 condim=3 -> 2*(condim-1)=4 pyramid edges (the v0.8 case);
//                 condim=1 -> frictionless (0 friction rows); condim 4/6
//                 (torsional/rolling) is the Q8 v1.0 seam (deferred).
// All four are UNIFORM across the manifold span in this struct's simplest form;
// the design intentionally keeps them per-call so C5 can specialize later.
struct ContactRowComplianceInputs {
    float vel = 0.0f;          // signed normal relative velocity (Jqvel)
    float invweight = 1.0f;    // reaction inv-weight sum (>0); from C4c providers
    float dt = 0.0f;           // substep dt (REFSAFE clamp)
    uint32_t condim = 3u;      // friction dimensionality (3 -> 4 pyramid edges)
    bool refsafe = true;       // REFSAFE timeconst clamp (MuJoCo default on)
};

// Per-emitted-row side tags so C5's reaction providers can dispatch each side of
// a contact row class-blind WITHOUT mutating any production-read struct (Row /
// RowAnchor / RowBodyPair stay byte-identical -> goldens do not move). The
// emitter fills ONE entry per row it appends, in the SAME order as the rows; a
// row at appended-index i has its sides at out_sides[base + i] where base is the
// RowBuffers row count captured at emitter entry. Production never reads this;
// C5 does.
struct ContactRowSides {
    CollidableRef a;  // side-A collidable (type/react/handle from the manifold)
    CollidableRef b;  // side-B collidable
};

// THE C4b DELIVERABLE. For each manifold, per contact point:
//   * ONE compliant NORMAL row (kMaximalContactRowClassId, Unilateral|GradActive,
//     lower=0, upper=FLT_MAX) with compliance_alpha=R and rhs=aref from the C4a
//     ComputeCompliantRow (pos = -penetration, signed: penetrating is negative).
//   * 2*(condim-1) PURE-TANGENT pyramid friction rows (condim=3 -> 4 spokes:
//     +t0, -t0, +t1, -t1; fixed order) -- the linearized isotropic friction
//     polygon. mu = manifold.friction is carried in RowMaterial.friction (the
//     cone bound is applied by C5), NOT baked into the Jacobian; friction rows
//     carry compliance_alpha=0, rhs=0 (no positional bias), mirroring
//     AppendContactGroup. Tangent basis via the deterministic ChooseTangent.
// Appends to *out_rows and (one-per-appended-row, same order) to *out_sides.
void EmitCompliantContactRows(std::span<const ContactManifold> manifolds,
                              const ContactRowComplianceInputs& inputs,
                              RowBuffers* out_rows,
                              std::vector<ContactRowSides>* out_sides);

uint32_t BuildContactRows(uint32_t contact_points, RowBuffers* out_rows);
void BuildContactRows(const std::vector<ContactManifold>& manifolds,
                      RowBuffers* out_rows);
void BuildContactRows(std::span<const ContactManifold> manifolds,
                      RowBuffers* out_rows);

void AppendRevoluteRows(RowBuffers* out_rows,
                        uint32_t body_a,
                        uint32_t body_b,
                        math::Vec3 axis,
                        math::Transform parent_frame,
                        math::Transform child_frame,
                        float lower_limit,
                        float upper_limit,
                        float current_angle = 0.0f);

void AppendFixedRows(RowBuffers* out_rows,
                     uint32_t body_a,
                     uint32_t body_b,
                     math::Transform parent_frame,
                     math::Transform child_frame);

void AppendDriveRow(RowBuffers* out_rows,
                    uint32_t body_a,
                    uint32_t body_b,
                    math::Vec3 axis,
                    float target_velocity,
                    float max_force);

} // namespace nuka::constraint
