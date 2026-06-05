#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::Row -- CSR universal constraint row
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::constraint {

enum class RowKind : uint8_t {
    Contact,
    Joint,
    Drive,
    FeatherstoneContact
};

struct Row {
    uint32_t row_class_id = 0u;
    uint32_t body_count = 0u;
    uint32_t body_list_offset = 0u;
    uint32_t jacobian_offset = 0u;
    float rhs = 0.0f;
    float lambda = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
    float compliance_alpha = 0.0f;
    float damping_beta = 0.0f;
    uint16_t flags = 0u;
    uint16_t adjoint_kernel_id = 0u;
    uint8_t gradient_mode = 0u;
    uint8_t recompute_mode = 0u;
    uint8_t event_flag_field = 0u;
    uint8_t contact_softness = 0u;
};

namespace row_flags {
constexpr uint16_t Equality = 1u << 0;
constexpr uint16_t Unilateral = 1u << 1;
constexpr uint16_t Friction = 1u << 2;
constexpr uint16_t Coupled = 1u << 3;
constexpr uint16_t GradActive = 1u << 4;
// v0.8 C5a: discriminates COMPLIANT contact rows (emitted by
// EmitCompliantContactRows) from the legacy class-id-0 contact rows (both use
// kMaximalContactRowClassId, so class id cannot discriminate). The unified
// solver's compliant branch gates on this flag: compliant rows add
// Row.compliance_alpha (the dual regularizer R) to the effective-mass
// denominator, route reaction through the ReactionProvider per side, use
// unilateral coupled-pyramid friction bounds, and skip Baumgarte. Legacy rows
// never carry this flag, so the legacy solve path is byte-identical.
constexpr uint16_t Compliant = 1u << 5;
} // namespace row_flags

inline constexpr uint32_t kMaximalContactRowClassId = 0u;
inline constexpr uint32_t kMaximalJointRowClassId = 1u;
inline constexpr uint32_t kMaximalDriveRowClassId = 2u;
inline constexpr uint32_t kFeatherstoneContactRowClassId = 3u;
// v0.7 p08-B: SDF (Newton-summed-field) contact rows. These MIRROR the ids in
// the generated registry (src/codegen/generated/row_class_registry.hpp, the
// source of truth produced by regen.py); the constants are hand-mirrored here
// for the same reason ids 0-3 are -- so nuka_constraint need not include the
// generated header. A codegen-roundtrip test guards that the ids stay in sync.
inline constexpr uint32_t kRigidSdfContactRowClassId = 4u;
inline constexpr uint32_t kFeatherstoneSdfContactRowClassId = 5u;
// v0.7 p09-A: XPBD soft-body distance constraint row class. Hand-mirrors the id
// in the generated registry (= 6), same rationale as ids 0-5 above; the
// codegen-roundtrip test guards that this stays in sync with the registry.
inline constexpr uint32_t kXpbdDistanceRowClassId = 6u;
// v0.7 p11 (K3): cross-system particle-particle contact row class. Hand-mirrors
// the id in the generated registry (= 10), same rationale as ids 0-6 above; the
// codegen-roundtrip test guards that this stays in sync with the registry.
inline constexpr uint32_t kParticleParticleContactRowClassId = 10u;
inline constexpr uint32_t kInvalidBodyIndex = ~0u;
inline constexpr float kRowHugeLimit = 1.0e6f;

struct RowJacobian6 {
    math::Vec3 linear;
    math::Vec3 angular;
};

struct RowMaterial {
    RowKind kind = RowKind::Contact;
    uint32_t group_id = 0u;
    uint32_t normal_row_count = 0u;
    uint32_t first_friction_row = 0u;
    uint32_t friction_row_count = 0u;
    float friction = 0.0f;
    float restitution = 0.0f;
    float position_error = 0.0f;
};

struct RowAnchor {
    math::Vec3 local_a = math::Vec3::Zero();
    math::Vec3 local_b = math::Vec3::Zero();
};

} // namespace nuka::constraint
