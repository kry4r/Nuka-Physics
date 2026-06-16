#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M3b articulation op shared types.
//
// VERBATIM local copies of the articulation device-state types
// (runtime/articulation/articulation_state.hpp + articulation_contacts.hpp).
// They were duplicated here (NOT #included) because the ops layer includes the
// PHI v2 phi/buffer.hpp (opaque `struct Buffer`), which used to collide with the
// legacy `class phi::Buffer` that articulation_state.hpp pulled in via the
// (now-deleted) legacy buffer header. That collision is now GONE -- the legacy
// RAII buffer was removed in the M11 BUF sweep and articulation_state.hpp is on the
// v2 phi/buffer.hpp -- so these PODs can fold to the single runtime definition.
// NAMED DEBT (M11 OD-18, fast-follow): the
// collapse is deferred to keep the buffer sweep's blast radius off the byte-D1
// ops kernel layer; the layouts are byte-identical (static_asserts below guard
// the contract) so no behavioral risk while the duplication remains.
//
// LAYOUT CONTRACT (the byte-exact port hinges on it):
//   * every struct is the exact legacy layout (float[6]/float[36] PODs);
//   * ArticulationJointType is uint8_t-backed and the Model joint_type field is
//     staged as u8, so the table aliases as ArticulationJointType* directly;
//   * MakeArticulationDeviceState builds the legacy kernel-parameter struct
//     from the generated ModelView/DataView — the ONLY thing the port changes
//     is where the pointers come from (the plan §3.2 contract).
// ---------------------------------------------------------------------------

#include <cstdint>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"
#include "phi/op_schema.hpp"

namespace nuka::phi::nkops {

// --- runtime/articulation/articulation_state.hpp (verbatim) ----------------
enum class ArticulationJointType : uint8_t {
    Revolute = 0,
    Prismatic = 1,
    Fixed = 2,
    FloatingBase = 3,
};

struct LinkSpatialInertia {
    float I[36] = {};
};

struct LinkSpatialVel {
    float v[6] = {};
};

struct LinkSpatialAccel {
    float a[6] = {};
};

struct LinkSpatialTransform {
    float X[36] = {};
};

struct LinkArticulatedInertia {
    float Ia[36] = {};
};

struct LinkBiasForce {
    float p[6] = {};
};

struct LinkMotionSubspace {
    float s[6] = {};
};

struct ArticulationDeviceState {
    LinkSpatialInertia* link_inertia = nullptr;
    LinkSpatialVel* link_velocity = nullptr;
    LinkSpatialAccel* link_acceleration = nullptr;
    LinkSpatialTransform* link_xup = nullptr;
    LinkSpatialVel* link_velocity_bias = nullptr;
    LinkArticulatedInertia* link_articulated_I = nullptr;
    LinkBiasForce* link_bias_force = nullptr;
    LinkBiasForce* link_u_spatial = nullptr;
    LinkMotionSubspace* joint_motion_subspace = nullptr;
    math::Transform* link_pose = nullptr;
    math::Transform* link_local_pose = nullptr;
    math::Transform* link_inertial_frame = nullptr;
    math::Transform* base_pose = nullptr;
    float* q = nullptr;
    float* qdot = nullptr;
    float* qddot = nullptr;
    float* tau = nullptr;
    float* joint_damping = nullptr;
    float* joint_armature = nullptr;
    float* joint_diagonal = nullptr;
    float* joint_force = nullptr;
    math::Vec3* joint_axis = nullptr;
    math::Vec3* parent_offset = nullptr;
    ArticulationJointType* joint_type = nullptr;
    uint32_t* parent_link = nullptr;
    uint32_t* link_body = nullptr;
    uint32_t* link_to_articulation = nullptr;
    uint32_t* articulation_link_count = nullptr;
    uint32_t* articulation_link_offset = nullptr;
    uint32_t total_link_count = 0;
    uint32_t articulation_count = 0;
};

// --- runtime/articulation/articulation_contacts.hpp (verbatim) -------------
constexpr uint32_t kMaxFootContactsPerEnv = 4u;

// Task 1/2: the per-ARTICULATION contact-slot stride for a MULTI-DOG world is a
// RUNTIME value (the foot/dog-dog/solver share it) so K co-resident dogs each get
// room for their 4 feet PLUS body-vs-terrain PLUS dog-dog rows in ONE block. At
// K<=1 the runtime stride is EXACTLY kMaxFootContactsPerEnv (4) -> byte-identical.
// kMaxContactsPerArtic is the COMPILE-TIME ceiling sizing the FUSED solver's
// per-slot register arrays (lambda_n/t1/t2). Raising the array sizes does not
// change any float op for a stride <= the old 4 (the loop is bounded by the
// runtime stride; extra slots are never visited at K==1), so the K==1 results
// stay byte-identical -- the same discipline as the kMaxArticulationDof bump.
constexpr uint32_t kMaxContactsPerArtic = 12u;   // 4 feet + body-terrain + dog-dog
// The multi-dog (K>1) per-articulation stride. A Go2 has 4 feet; up to ~6 body
// links can touch terrain when it collapses; plus a few dog-dog rows -> 12 covers
// the worst realistic case with margin. <= kMaxContactsPerArtic (the array cap).
constexpr uint32_t kMultiDogContactsPerArtic = 12u;

struct FootShape {
    uint32_t calf_local_link = 0u;
    math::Vec3 local_offset{};
    float radius = 0.0f;
};

constexpr float kInertiaDiagonalEpsilon = 1.0e-6f;
constexpr float kEffectiveMassDenomEpsilon = 1.0e-9f;
constexpr uint32_t kContactSolverIterations = 48u;
constexpr float kBaumgarteBeta = 0.2f;
constexpr float kPenetrationSlop = 1.0e-4f;

enum class ContactRowType : uint32_t {
    kRowInactive = 0u,
    kRowNormal = 1u,
    kRowFriction1 = 2u,
    kRowFriction2 = 3u,
    kRowJointLimit = 4u,
};

struct ArticulatedContactRow {
    ContactRowType row_type = ContactRowType::kRowInactive;
    uint32_t articulation = ~0u;
    uint32_t contact_link = ~0u;
    float effective_mass = 0.0f;
    float effective_mass_t1 = 0.0f;
    float effective_mass_t2 = 0.0f;
    float depth = 0.0f;
    math::Vec3 normal{};
    math::Vec3 tangent1{};
    math::Vec3 tangent2{};
};

constexpr uint32_t kMaxArticulationDof = 64u;

// The rows DataView field is f32 elem:16 per row_slot (64B) — the exact
// ArticulatedContactRow footprint, so the field aliases as the struct.
static_assert(sizeof(ArticulatedContactRow) == 16 * sizeof(float),
              "rows field elem:16 must match ArticulatedContactRow");
static_assert(sizeof(FootShape) == 5 * sizeof(float),
              "foot_shape field packs 5 scalars per slot");
static_assert(sizeof(math::Transform) == 7 * sizeof(float),
              "transform dtype lanes==7");

// --- runtime/articulation/articulation_device_helpers.cuh (verbatim bodies) -
__forceinline__ __device__ uint32_t JointDofCountDevice(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
        case ArticulationJointType::FloatingBase:
            return 6u;  // T8a: free-floating root contributes 6 DOF.
    }
    return 0u;
}

__forceinline__ __device__ uint32_t LocalDofIndexDevice(const ArticulationDeviceState& state,
                                                        uint32_t offset,
                                                        uint32_t link) {
    uint32_t index = 0u;
    for (uint32_t k = offset; k < link; ++k) {
        index += JointDofCountDevice(state.joint_type[k]);
    }
    return index;
}

// --- ModelView/DataView -> legacy kernel-parameter struct ------------------
// The pointer-source rebinding the plan mandates: every member name matches the
// legacy field 1:1 BY DESIGN (M3a codegen), so this is a pure re-aim.
inline ArticulationDeviceState MakeArticulationDeviceState(
    const ModelView& model, const DataView& data,
    uint32_t total_link_count, uint32_t articulation_count) {
    ArticulationDeviceState s;
    s.link_inertia = reinterpret_cast<LinkSpatialInertia*>(model.link_inertia);
    s.link_velocity = reinterpret_cast<LinkSpatialVel*>(data.link_velocity);
    s.link_acceleration = reinterpret_cast<LinkSpatialAccel*>(data.link_acceleration);
    s.link_xup = reinterpret_cast<LinkSpatialTransform*>(data.link_xup);
    s.link_velocity_bias = reinterpret_cast<LinkSpatialVel*>(data.link_velocity_bias);
    s.link_articulated_I = reinterpret_cast<LinkArticulatedInertia*>(data.link_articulated_I);
    s.link_bias_force = reinterpret_cast<LinkBiasForce*>(data.link_bias_force);
    s.link_u_spatial = reinterpret_cast<LinkBiasForce*>(data.link_u_spatial);
    s.joint_motion_subspace =
        reinterpret_cast<LinkMotionSubspace*>(data.joint_motion_subspace);
    s.link_pose = data.link_pose;
    s.link_local_pose = model.link_local_pose;
    s.link_inertial_frame = model.link_inertial_frame;
    s.base_pose = data.base_pose;
    s.q = data.q;
    s.qdot = data.qdot;
    s.qddot = data.qddot;
    s.tau = data.tau;
    s.joint_damping = model.joint_damping;
    s.joint_armature = model.joint_armature;
    s.joint_diagonal = data.joint_diagonal;
    s.joint_force = data.joint_force;
    s.joint_axis = model.joint_axis;
    s.parent_offset = model.parent_offset;
    s.joint_type = reinterpret_cast<ArticulationJointType*>(model.joint_type);
    s.parent_link = model.parent_link;
    s.link_body = model.link_body;
    s.link_to_articulation = model.link_to_articulation;
    s.articulation_link_count = model.articulation_link_count;
    s.articulation_link_offset = model.articulation_link_offset;
    s.total_link_count = total_link_count;
    s.articulation_count = articulation_count;
    return s;
}

} // namespace nuka::phi::nkops
