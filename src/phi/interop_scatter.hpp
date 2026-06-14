#pragma once
// ---------------------------------------------------------------------------
// nuka::phi::InteropScatterI -- the backend-agnostic CUDA<->Vulkan interop
// scatter interface (M11 §3.E INT-1, OD-1=(A)).
//
// This is the CUDA-FREE seam between the zero-CUDA app layer
// (runtime/app/cuda_vulkan_interop.{hpp,cpp}, the CudaVulkanInteropPublisher) and
// the CUDA backend impl (phi/backend_cuda/interop/cuda_vk_scatter.{hpp,cu}). The
// app layer includes ONLY this header (and never cuda_runtime / backend_cuda), so
// the src/runtime/app/** + src/render/** zero-CUDA-token lint red-line stays green
// (m11-recon §R11). The CUDA scatter kernel + cudaImportExternalMemory +
// external-semaphore plumbing live entirely behind this vtable, in backend_cuda.
//
// CONTRACT (the device-scatter zero-copy path, OD-2=(A)):
//   1. The Vulkan renderer allocates a per-instance transform SSBO from EXPORTABLE
//      external memory (VK_KHR_external_memory_fd, OPAQUE_FD) and exports it as a
//      plain OS file descriptor (`int fd`) + its byte size + the matching binary
//      semaphore fds (signal-when-Vulkan-done / signal-when-CUDA-done). Those land
//      in the POD descriptors below -- NO Vulkan or CUDA TOKEN crosses this seam,
//      only `int fd` + sizes (the OD-3 external-memory-fd transport; DLPack only
//      DESCRIBES bytes, it does not move them).
//   2. ImportMemory() / ImportSemaphores() hand those fds to the CUDA backend,
//      which cudaImportExternalMemory()'s the SSBO into a device pointer and
//      cudaImportExternalSemaphore()'s the sync primitives (INT-2/INT-7).
//   3. Scatter() launches the device kernel (via launch.cuh, the only <<<>>>
//      wrapper) that writes, for every instance row, the column-major 4x4
//      object->world matrix BIT-IDENTICAL to HostDownloadPublisher's host
//      composition (R14): world_xform = downloaded_pose(fk) * cached_visual_local,
//      reading the SAME FK device field buffers DownloadField reads. No D2H copy.
//
// FALLBACK (the only path runnable on this lavapipe/CPU-Vulkan box): when the
// Vulkan device lacks VK_KHR_external_memory_fd, OR vkGetMemoryFdKHR is
// unavailable, OR cudaImportExternalMemory fails, the renderer reports the SSBO as
// NOT exportable and the publisher NEVER constructs a live scatter -> the
// CudaVulkanInteropPublisher gracefully degrades to HostDownloadPublisher
// (m11-recon §INT-8). CreateCudaVkScatter() likewise returns nullptr when no CUDA
// device is available (the phi "unavailable -> null" registry contract), so the
// app layer compiles + links with NO CUDA dependency and a NULL fallback.
//
// PURE C++ -- zero CUDA / Vulkan tokens. Compiles under g++.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <memory>

namespace nuka::phi {

// ---------------------------------------------------------------------------
// ExternalMemoryDesc -- an exported Vulkan memory handle, transport-neutral.
//
// `fd` is a plain OS file descriptor produced by vkGetMemoryFdKHR (OPAQUE_FD).
// `size_bytes` is the full allocation size (NOT the used range -- POSIX FD import
// requires the whole VkDeviceMemory size). `dedicated` mirrors whether the Vulkan
// allocation used VkMemoryDedicatedAllocateInfo (NVIDIA recommends/requires it for
// imported images/buffers; CUDA's cudaExternalMemoryHandleDesc carries the same
// flag). NO Vulkan / CUDA type appears here -- a plain int + sizes only.
// ---------------------------------------------------------------------------
struct ExternalMemoryDesc {
    int      fd         = -1;     // OPAQUE_FD from vkGetMemoryFdKHR (owned by caller until imported)
    uint64_t size_bytes = 0;      // full VkDeviceMemory size
    bool     dedicated  = false;  // VkMemoryDedicatedAllocateInfo was used
};

// ---------------------------------------------------------------------------
// ExternalSemaphoreDesc -- an exported Vulkan binary semaphore handle (OPAQUE_FD).
//
// Two are used (INT-7): `wait_fd` is the semaphore CUDA WAITS on (Vulkan signals
// it once the SSBO is no longer in use by the previous frame's draw), `signal_fd`
// is the semaphore CUDA SIGNALS after the scatter completes (the draw WAITS on it
// before reading the SSBO). Either may be -1 when the consumer drives sync another
// way; the backend then falls back to a stream synchronize.
// ---------------------------------------------------------------------------
struct ExternalSemaphoreDesc {
    int wait_fd   = -1;  // CUDA waits (Vulkan-signalled: SSBO free for this frame)
    int signal_fd = -1;  // CUDA signals (draw waits: scatter complete)
};

// ---------------------------------------------------------------------------
// InstanceScatterRow -- one render instance's pose-source binding, the device
// scatter's per-row input. Mirrors render::PoseSource + the instance's
// cached_visual_local, lowered to a POD (no render/scene include here, keeping the
// seam dependency-light). The app-layer publisher fills these ONCE (the binding is
// build-time constant) and re-uses them every frame.
//
//   kind: 0=Static (skip, keep bind pose), 1=Link, 2=Body, 3=Base.
//   row : index into the selected env's LinkPose / BodyPose rows (ignored for Base).
//   cached_visual_local: the 7-float physics-frame->visual-frame transform
//                        (px,py,pz, qw,qx,qy,qz) -- the SAME bytes
//                        math::Transform carries, passed flat so this header
//                        needs no math include.
// ---------------------------------------------------------------------------
struct InstanceScatterRow {
    uint32_t kind = 0;   // PoseSource::Kind
    uint32_t row  = 0;   // link/body row within the env
    float    cached_visual_local[7] = {0, 0, 0, 1, 0, 0, 0};  // pos(3)+quat(w,x,y,z)
};

// ---------------------------------------------------------------------------
// ScatterFkSource -- the device FK field buffers + env layout the scatter reads.
//
// These are the SAME device pointers Data::Ptr(FieldId) returns (the buffers
// DownloadField copies FROM), so the scattered transform is bit-identical to the
// host-downloaded one (R14). The pointers are plain `void*` device addresses --
// no CUDA type. The kernel indexes link_pose[env*links_per_env + row] etc.,
// matching HostDownloadPublisher's env-major offset math exactly.
//
// Each *_stride_floats is the per-row float count of a math::Transform (7:
// pos3+quat4) so the kernel strides identically to the host. A null field pointer
// means "that pose kind is unavailable this frame" (the kernel keeps the row's
// bind pose, exactly like HostDownloadPublisher's have_link_/have_body_ guard).
// ---------------------------------------------------------------------------
struct ScatterFkSource {
    const void* link_pose = nullptr;  // device LinkPose base (math::Transform rows)
    const void* body_pose = nullptr;  // device BodyPose base
    const void* base_pose = nullptr;  // device BasePose base
    uint32_t    env_index       = 0;
    uint32_t    links_per_env   = 0;
    uint32_t    bodies_per_env  = 0;
};

// ---------------------------------------------------------------------------
// InteropScatterI -- the abstract device-scatter backend.
//
// Lifetime: created ONCE (CreateCudaVkScatter) when CUDA is available; the
// renderer's exported SSBO + semaphores are imported once (ImportMemory /
// ImportSemaphores) and re-used; Scatter() runs per frame; everything is released
// by the destructor. All methods are no-throw / return-bool so the publisher can
// degrade on any failure.
// ---------------------------------------------------------------------------
class InteropScatterI {
public:
    virtual ~InteropScatterI() = default;

    // Import the exported Vulkan SSBO (cudaImportExternalMemory) and map it to a
    // device pointer of `instance_capacity` mat4 rows (16 floats each). Returns
    // false on any failure (the publisher then degrades). Re-callable (re-import
    // after a swapchain/SSBO recreate); a previous import is released first.
    virtual bool ImportMemory(const ExternalMemoryDesc& mem,
                              uint32_t instance_capacity) = 0;

    // Import the Vulkan<->CUDA sync semaphores (cudaImportExternalSemaphore).
    // Optional: when not called (or fds are -1) Scatter() falls back to a stream
    // synchronize for ordering. Returns false on failure.
    virtual bool ImportSemaphores(const ExternalSemaphoreDesc& sem) = 0;

    // Launch the device scatter: for each of the first `instance_count` rows, write
    // the column-major object->world mat4 into the imported SSBO at row*16 floats,
    // composing fk(rows[i]) * cached_visual_local bit-identically to the host
    // publisher (R14). `rows` is a HOST array of `instance_count` bindings (the
    // backend uploads it to a persistent device buffer; it is build-time constant
    // so this is allocation-free after the first call). Returns false on failure.
    //
    // When semaphores were imported, the kernel is fenced wait(wait_fd)->scatter->
    // signal(signal_fd) on the backend stream; otherwise it synchronizes the stream
    // after the launch so the SSBO is coherent before the draw.
    virtual bool Scatter(const ScatterFkSource& fk,
                         const InstanceScatterRow* rows,
                         uint32_t instance_count) = 0;
};

// ---------------------------------------------------------------------------
// Factory (OD-1=(A)): a heap-owned CUDA interop scatter when the CUDA interop TU
// is linked AND a device initializes, else nullptr. DEFINED in the CUDA backend TU
// (phi/backend_cuda/interop/cuda_vk_scatter.cu); a WEAK fallback (no CUDA backend
// linked) returns nullptr so the app/render libs link standalone -- mirrors the RT
// facet's CreateCudaRtBackend / RtBackendAvailable split exactly.
// ---------------------------------------------------------------------------
std::unique_ptr<InteropScatterI> CreateCudaVkScatter();

// Cheap probe: true iff CreateCudaVkScatter() would return a usable backend.
bool CudaVkScatterAvailable();

}  // namespace nuka::phi
