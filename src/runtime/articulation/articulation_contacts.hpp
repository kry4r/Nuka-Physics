#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- world-pose FK + foot-vs-ground contacts
// ---------------------------------------------------------------------------
//
// p01-F T2. Two GPU passes that together produce the per-step, per-env contact
// data (which Go2 feet touch a static ground plane, where, how deep) that a
// later row-assembly task turns into constraint rows.
//
//  (A) UpdateWorldLinkPoses -- forward kinematics. The cooked link_pose in
//      ArticulationDeviceState is cook-time STATIC (rest pose); the Featherstone
//      ABA pass writes link_xup (spatial parent->child transforms) but never
//      world poses. This pass recomputes an up-to-date world math::Transform per
//      link from the current generalized coordinates q, walking links in local
//      order (the cooker guarantees parent-local-index < child, so one forward
//      pass suffices). The relative (parent->child) transform mirrors
//      featherstone_aba.cu's JointTransform in SE(3) form:
//        relative.position = link_local_pose.position + parent_offset
//                            (+ joint_axis * q for prismatic)
//        relative.rotation = link_local_pose.rotation * jointRotation(axis, q)
//      with world[child] = world[parent] o relative, root parent = identity.
//      Output is written to a SEPARATE buffer (not state.link_pose, which is
//      treated as the static rest pose elsewhere).
//
//  (B) DetectFootGroundContacts -- deterministic sphere-vs-plane. The Go2 feet
//      are spheres on the 4 calf links; the scene has no ground collider, so a
//      static half-space (z = ground_height, normal +Z) is supplied here. One
//      thread per env walks its (base-relative) feet in fixed order and compacts
//      penetrating feet into the env's own fixed-stride output slots. No atomics,
//      each env owns its slots -> D1-deterministic.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

namespace nuka::runtime::articulation {

// Max foot contacts emitted per environment (Go2 has 4 feet).
constexpr uint32_t kMaxFootContactsPerEnv = 4u;

// Per-foot static description, expressed in BASE (single-replica) link indices.
// The detection kernel turns calf_local_link into a per-replica global link via
//   global_link = env * base_link_count + calf_local_link
// so the same table drives every replicated environment.
struct FootShape {
    uint32_t calf_local_link = 0u;   // base-relative global link index of the calf
    math::Vec3 local_offset{};       // sphere center in the calf link frame
    float radius = 0.0f;             // sphere radius
};

// (A) Recompute world poses for every link from the current q. `out_world_pose`
// must be a device buffer of length state.total_link_count. One block per
// articulation, single forward pass over the links.
void UpdateWorldLinkPoses(const phi::DeviceContext& context,
                          ArticulationDeviceState state,
                          math::Transform* out_world_pose);

// (B) Detect foot-vs-ground contacts. One thread per environment; each env
// inspects its `foot_count` feet (device buffer of FootShape, length foot_count,
// shared across all envs) against the half-space z = ground_height (normal +Z).
//
// Output layout (all device buffers, fixed stride kMaxFootContactsPerEnv per
// env, env-major):
//   out_contact_link   [env_count * kMaxFootContactsPerEnv] uint32_t -- global
//                       link index (the calf) of each emitted contact.
//   out_contact_point  [env_count * kMaxFootContactsPerEnv] Vec3 -- world contact
//                       point (foot center projected onto the ground surface).
//   out_contact_normal [env_count * kMaxFootContactsPerEnv] Vec3 -- world normal,
//                       always (0,0,1).
//   out_contact_depth  [env_count * kMaxFootContactsPerEnv] float -- penetration
//                       depth d = (ground_height + radius) - foot_world_z, > 0.
//   out_contact_count  [env_count] uint32_t -- number of emitted contacts in
//                       this env's slot block (0..kMaxFootContactsPerEnv).
// Unused trailing slots are cleared: link = ~0u (kInvalidLink, which downstream
// chain-Jacobian consumers skip), point/normal = 0, depth = 0. `foot_count` must
// be <= kMaxFootContactsPerEnv.
void DetectFootGroundContacts(const phi::DeviceContext& context,
                              const math::Transform* world_pose,
                              const FootShape* feet,
                              uint32_t foot_count,
                              uint32_t env_count,
                              uint32_t base_link_count,
                              float ground_height,
                              uint32_t* out_contact_link,
                              math::Vec3* out_contact_point,
                              math::Vec3* out_contact_normal,
                              float* out_contact_depth,
                              uint32_t* out_contact_count);

} // namespace nuka::runtime::articulation
