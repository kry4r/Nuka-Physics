#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::CudaVulkanInteropPublisher -- the ZERO-COPY physics->render
// pose bridge (M11 §3.E INT-6, OD-2=(A)).
//
// The M11 sibling of HostDownloadPublisher: instead of downloading the FK poses to
// host, composing world_xform = fk * cached_visual_local on the CPU, and pushing
// each transform as a per-draw push constant (one D2H copy per field per frame),
// this publisher SCATTERS the composed object->world matrices DIRECTLY into a
// Vulkan-imported, device-local per-instance transform SSBO with a CUDA kernel --
// NO device->host copy at all (the firm render<->physics shared-memory directive:
// Vulkan renders the REAL physics device state). The renderer's instanced vertex
// shader (mesh_instanced.vert) then reads each instance's matrix from the SSBO by
// gl_InstanceIndex.
//
// It implements the SAME abstract PosePublisher interface, so it drops in behind
// the existing seam with no change to TransformSyncSystem / Simulation (the M8
// no-rework swap point). Callers depend on PosePublisher&.
//
// HOST-ONLY / ZERO CUDA TOKEN (this dir is the src/runtime/app/** lint red-line,
// m11-recon §R11): the device scatter + cudaImportExternalMemory live entirely
// behind the CUDA-FREE phi::InteropScatterI seam (phi/interop_scatter.hpp). This
// header/TU names NO cuda_runtime, NO kernel-launch chevrons, NO phi/backend_cuda
// include. The fd
// transport is a plain `int` (phi::ExternalMemoryDesc); Vulkan handles never cross
// in (the renderer exports the fd, this layer only forwards it).
//
// ★ NOT LOCALLY VERIFIABLE (headless box: real CUDA GPU + lavapipe CPU Vulkan ICD;
// CUDA-GPU memory cannot be shared with a CPU Vulkan device). TryEnable() detects
// the unsupported case and returns false; the caller (viewer_main / the INT-9
// smoke) then falls back to HostDownloadPublisher. Real zero-copy is OWNER-VERIFIED
// on a display+NVIDIA machine.
// ---------------------------------------------------------------------------

#include "phi/interop_scatter.hpp"  // CUDA-FREE seam (InteropScatterI + descriptors)
#include "render/render_world.hpp"
#include "runtime/app/pose_publisher.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::nk {
class World;
}  // namespace nuka::nk

namespace nuka::runtime::app {

// ---------------------------------------------------------------------------
// CudaVulkanInteropPublisher -- device-scatter PosePublisher behind the seam.
//
// Setup (once, before the loop): construct, then TryEnable(world, render_world,
// ssbo_mem, instance_capacity) where ssbo_mem is the renderer's exported transform
// SSBO (phi::ExternalMemoryDesc{fd,size,dedicated}) obtained from the renderer's
// opt-in exportable-SSBO path. TryEnable:
//   * Creates a CUDA scatter (phi::CreateCudaVkScatter()) -- nullptr when no CUDA
//     device -> returns false.
//   * Imports the SSBO (cudaImportExternalMemory) -- false when the fd is invalid
//     / lavapipe never exported one -> returns false.
//   * Builds the per-instance scatter bindings (PoseSource + cached_visual_local)
//     from render_world ONCE (the binding is build-time constant).
// On ANY failure TryEnable returns false and Enabled() stays false; the caller
// degrades to HostDownloadPublisher.
//
// Per frame (Publish): when enabled, reads the FK device field pointers from the
// World's live Data (host-side, no CUDA) and calls scatter_->Scatter(...) -- the
// composed mat4s land in the SSBO with no host copy. When NOT enabled, Publish is
// a no-op (the caller should be using HostDownloadPublisher instead; this guards
// against a misuse).
// ---------------------------------------------------------------------------
class CudaVulkanInteropPublisher : public PosePublisher {
public:
    CudaVulkanInteropPublisher();
    ~CudaVulkanInteropPublisher() override;

    // Attempt to stand up the zero-copy path. Returns true iff a live CUDA scatter
    // imported the exported SSBO successfully (the device path is active). False on
    // ANY failure (no CUDA device, SSBO not exportable on this ICD, import failed)
    // -> the caller falls back to HostDownloadPublisher. `instance_capacity` is the
    // number of mat4 rows the SSBO was allocated for (>= render_world.instances).
    bool TryEnable(const nk::World& world,
                   const render::RenderWorld& render_world,
                   const phi::ExternalMemoryDesc& ssbo_mem,
                   uint32_t instance_capacity);

    // Optional: import the Vulkan<->CUDA sync semaphores (INT-7). When not called,
    // Scatter falls back to a stream synchronize for SSBO coherence. Returns false
    // if neither semaphore imported (then keep the synchronize path).
    bool ImportSemaphores(const phi::ExternalSemaphoreDesc& sem);

    bool Enabled() const { return enabled_; }

    // PosePublisher: scatter the selected env's composed transforms into the SSBO.
    // No-op when not enabled. The env_index modulo contract matches the base class
    // / HostDownloadPublisher exactly (env_count==1 today -> a no-op modulo).
    void Publish(const nk::World& world, uint32_t env_index,
                 render::RenderWorld& render_world) override;

private:
    // Lower each render instance's PoseSource + cached_visual_local into the flat
    // scatter binding rows (kind/row + 7-float transform). Called once by TryEnable.
    void BuildRows(const render::RenderWorld& render_world);

    std::unique_ptr<phi::InteropScatterI> scatter_;
    std::vector<phi::InstanceScatterRow>  rows_;       // build-time constant bindings
    uint32_t                              instance_count_ = 0;
    bool                                  enabled_ = false;
};

}  // namespace nuka::runtime::app
