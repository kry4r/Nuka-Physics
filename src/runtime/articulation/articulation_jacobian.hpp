#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- Featherstone generalized Jacobian builder
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

namespace nuka::runtime::articulation {

void ComputeLinkPointJacobians(const phi::DeviceContext& context,
                               ArticulationDeviceState state,
                               const uint32_t* contact_link_indices,
                               const math::Vec3* contact_point_world,
                               uint32_t contact_count,
                               float* out_jacobian_scalars);

// Full-chain contact Jacobian. For each contact, walks the chain of ancestor
// joints from the contact's link up to the articulation root and writes the
// joint-space (generalized) Jacobian entry for every ancestor DOF into a
// `dof_stride`-wide slice of `out_chain_jacobian`. Projecting a contact force
// F onto the chain gives the generalized force tau_j = J_j . F per joint.
//
// Conventions:
//  * `contact_link_indices`  : global link index of the contact's link.
//  * `contact_point_world`   : world contact position (one per contact).
//  * `contact_normal_world`  : world contact normal (one per contact);
//                              normalized on read, falls back to +Z if zero.
//  * `dof_stride`            : base-inclusive max DOF count per articulation,
//                              i.e. the column count per contact. For the
//                              current Go2 (fixed root, 12 revolute) this is
//                              12; with a future 6-DOF floating base it is 18.
//  * `out_chain_jacobian`    : [contact_count * dof_stride] floats. Column
//                              dof_index(j) of contact c lives at
//                              out_chain_jacobian[c * dof_stride + dof_index(j)].
//
// DOF indexing is a base-inclusive prefix sum of per-joint DOF counts along the
// articulation's link order (Revolute=Prismatic=1, Fixed=0, future
// FloatingBase=6). The caller must zero `out_chain_jacobian` before the call;
// the kernel only writes the columns of the contact's ancestor chain.
void ComputeContactChainJacobians(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  const uint32_t* contact_link_indices,
                                  const math::Vec3* contact_point_world,
                                  const math::Vec3* contact_normal_world,
                                  uint32_t contact_count,
                                  uint32_t dof_stride,
                                  float* out_chain_jacobian);

} // namespace nuka::runtime::articulation
