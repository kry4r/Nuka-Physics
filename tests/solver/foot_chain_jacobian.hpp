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
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>
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
    cudaStream_t context, int context_dev,
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
    auto device = articulation::UploadArticulationState(context, context_dev, host);
    // phi v2 device buffers from the device-level DEFAULT (stream-0) type. Upload/
    // download run on stream 0 like the legacy CopyFromHost/ToHost; the chain-J
    // kernel runs on context.stream over the base pointers (unchanged).
    nuka::phi::BufferType* bt =
        nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice());
    nuka::phi::Buffer* link_buf = nuka::phi::BufferAlloc(bt, sizeof(uint32_t));
    nuka::phi::Buffer* point_buf = nuka::phi::BufferAlloc(bt, sizeof(Vec3));
    nuka::phi::Buffer* normal_buf = nuka::phi::BufferAlloc(bt, sizeof(Vec3));
    nuka::phi::Buffer* jbuf =
        nuka::phi::BufferAlloc(bt, static_cast<size_t>(dof_stride) * sizeof(float));
    nuka::phi::BufferUpload(link_buf, &contact_link, 0, sizeof(uint32_t));
    nuka::phi::BufferUpload(point_buf, &contact_point, 0, sizeof(Vec3));
    nuka::phi::BufferUpload(normal_buf, &contact_normal, 0, sizeof(Vec3));
    std::vector<float> zero(dof_stride, 0.0f);
    nuka::phi::BufferUpload(jbuf, zero.data(), 0, zero.size() * sizeof(float));
    articulation::ComputeContactChainJacobians(
        context, context_dev, device.View(),
        static_cast<const uint32_t*>(nuka::phi::BufferBase(link_buf)),
        static_cast<const Vec3*>(nuka::phi::BufferBase(point_buf)),
        static_cast<const Vec3*>(nuka::phi::BufferBase(normal_buf)),
        1u, dof_stride, static_cast<float*>(nuka::phi::BufferBase(jbuf)));
    cudaStreamSynchronize(context);
    std::vector<float> j(dof_stride);
    nuka::phi::BufferDownload(jbuf, j.data(), 0, j.size() * sizeof(float));
    nuka::phi::BufferFree(link_buf);
    nuka::phi::BufferFree(point_buf);
    nuka::phi::BufferFree(normal_buf);
    nuka::phi::BufferFree(jbuf);
    return j;
}

}  // namespace nuka::test
