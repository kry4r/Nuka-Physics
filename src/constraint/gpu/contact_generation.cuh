#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::gpu::contact_generation -- CUDA contact manifolds
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"
#include "constraint/contact_manifold.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/gpu/device_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::constraint::gpu {

struct CudaContactReport {
    uint32_t pair_count = 0;
    uint32_t contact_manifold_count = 0;
    uint32_t contact_point_count = 0;
};

class CudaContactResult {
public:
    CudaContactResult() = default;
    CudaContactResult(uint32_t pair_slot_count,
                      phi::Buffer manifolds,
                      phi::Buffer report);

    CudaContactResult(const CudaContactResult&) = delete;
    CudaContactResult& operator=(const CudaContactResult&) = delete;
    CudaContactResult(CudaContactResult&&) noexcept = default;
    CudaContactResult& operator=(CudaContactResult&&) noexcept = default;

    CudaContactReport DownloadReport() const;
    std::vector<constraint::ContactManifold> DownloadManifolds() const;

    uint32_t PairSlotCount() const { return pair_slot_count_; }
    const constraint::ContactManifold* DeviceManifolds() const;
    const CudaContactReport* DeviceReport() const;

private:
    uint32_t pair_slot_count_ = 0;
    phi::Buffer manifolds_;
    phi::Buffer report_;
};

CudaContactResult GenerateCudaContacts(
    const phi::DeviceContext& context,
    const runtime::gpu::DeviceWorld& device_world,
    const collision::gpu::CudaBroadphaseResult& broadphase);
CudaContactResult GenerateCudaContacts(
    const runtime::gpu::DeviceWorld& device_world,
    const collision::gpu::CudaBroadphaseResult& broadphase);

} // namespace nuka::constraint::gpu
