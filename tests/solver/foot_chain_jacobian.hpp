#pragma once
// ---------------------------------------------------------------------------
// Shared test helper -- the 18-wide foot CHAIN JACOBIAN for a floating-base go2,
// projected on a contact normal (v0.8 C5c, the advisor's named-debt fold).
// ---------------------------------------------------------------------------
// Three v0.8 unified-spine tests (test_foot_ground_subsume.cpp, test_foot_ground_
// mjx_parity.cpp, and the co-residence GATE test_foot_box_coresidence.cpp) all
// build a per-foot 18-wide chain Jacobian via the production
// ComputeContactChainJacobians (the T8b 6-base-column walk). They each carried a
// COPY of this helper; this header is the single shared definition so the FK-
// refresh-correct version is used everywhere.
//
// FK-REFRESH CORRECTNESS (the reason the subsume copy was latently wrong).
// ComputeContactChainJacobians reads state.link_pose (the WORLD link poses), but
// UploadArticulationState leaves link_pose at the STATIC COOKED rest pose (q=0,
// base at cook origin). The production pipeline refreshes it from the current q
// each step (the legacy batched articulated step stage 4: UpdateWorldLinkPoses ->
// copy into state.link_pose) BEFORE the chain-J. We mirror that here: the caller passes
// the FK world poses (already computed via ForwardKinematics) which we write into a
// COPY of the host state's link_pose before upload. WITHOUT this refresh the leg
// joint anchors collapse to the base origin and the leg columns degenerate to
// -lever_x (a stale-FK artifact, not a kernel bug). The subsume copy OMITTED this
// refresh; for its +Z-normal coplanar-feet scenario the omission is invisible (the
// base columns drop z and the leg columns are unused by its base-recoil
// assertions), but it is latently wrong -- this shared helper fixes it for all
// three callers. (test_foot_ground_subsume.cpp's own (B) base-column structural
// assertion is z-invariant, so adopting the refreshed helper leaves it green.)
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::test {

// 18-wide foot chain Jacobian projected on `contact_normal`, for ONE foot. REUSES
// the production ComputeContactChainJacobians (the T8b 6-base-column walk). `host`
// is taken BY VALUE so we can refresh its link_pose without mutating the caller's
// state; `fk_world_poses` are the current-q FK world poses (one per link).
inline std::vector<float> ComputeFootChainJ18(
    const nuka::phi::DeviceContext& context,
    nuka::runtime::articulation::ArticulationHostState host,  // by value (refreshed).
    const std::vector<nuka::math::Transform>& fk_world_poses,
    uint32_t contact_link,
    const nuka::math::Vec3& contact_point,
    const nuka::math::Vec3& contact_normal,
    uint32_t dof_stride) {
    namespace articulation = nuka::runtime::articulation;
    using nuka::math::Vec3;
    if (fk_world_poses.size() == host.link_pose.size()) {
        host.link_pose = fk_world_poses;  // refresh world poses from current q.
    }
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer link_buf(sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer point_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer normal_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer jbuf(static_cast<size_t>(dof_stride) * sizeof(float),
                           nuka::phi::MemoryKind::Device);
    link_buf.CopyFromHost(&contact_link, sizeof(uint32_t));
    point_buf.CopyFromHost(&contact_point, sizeof(Vec3));
    normal_buf.CopyFromHost(&contact_normal, sizeof(Vec3));
    std::vector<float> zero(dof_stride, 0.0f);
    jbuf.CopyFromHost(zero.data(), zero.size() * sizeof(float));
    articulation::ComputeContactChainJacobians(
        context, device.View(),
        static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()),
        1u, dof_stride, static_cast<float*>(jbuf.Data()));
    context.stream.Synchronize();
    std::vector<float> j(dof_stride);
    jbuf.CopyToHost(j.data(), j.size() * sizeof(float));
    return j;
}

}  // namespace nuka::test
