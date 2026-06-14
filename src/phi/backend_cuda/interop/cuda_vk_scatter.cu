// ---------------------------------------------------------------------------
// nuka::phi CUDA<->Vulkan interop scatter -- the CONCRETE CUDA backend (M11 §3.E
// INT-2 scatter kernel + cudaImportExternalMemory + INT-7 external semaphore).
//
// This TU is the ONLY place CUDA interop tokens (cudaImportExternalMemory,
// cudaExternalSemaphore*, the <<<>>> scatter launch) live. It implements the
// CUDA-FREE phi::InteropScatterI (phi/interop_scatter.hpp) and provides the STRONG
// CreateCudaVkScatter() / CudaVkScatterAvailable() that win over the app-layer weak
// fallback (runtime/app/interop_scatter_fallback.cpp) when nuka_phi2_interop is
// linked (the windowed viewer exe). It is OUTSIDE the engine zero-CUDA red-line
// (which covers only src/nk + src/scene + src/render + src/runtime/app); the
// app/render layers never include this file.
//
// ★ NOT LOCALLY VERIFIABLE (m11-recon §INT-2/INT-7 local_verifiable:false): this
// is a headless box -- the compute GPU is real CUDA but the only Vulkan ICD is
// lavapipe (CPU). CUDA-GPU memory cannot be shared with a CPU Vulkan device, so the
// import path CANNOT run here. The gate is build-link only; the real zero-copy run
// is OWNER-VERIFIED on a display+NVIDIA machine. Every import call is defensively
// checked and surfaced as a bool so the publisher degrades to HostDownloadPublisher
// on this box (the import simply fails / is never attempted because lavapipe does
// not export an SSBO fd).
//
// R14 (BYTE-EXACT): the scatter kernel composes world_xform = fk * cvl and lowers
// it to a column-major mat4 with arithmetic BIT-IDENTICAL to the host path
// (HostDownloadPublisher's `fk * cached_visual_local` via math::Transform::operator*
// + the renderer's TransformToMatrix). math::Transform::operator* / math::Quat are
// host-only `inline` (NOT __host__ __device__ -- confirmed in math/transform.hpp,
// math/quat.hpp), so device code CANNOT call them directly; per the repo's
// established instance_transform.cuh idiom we provide HD helpers performing the
// EXACT SAME fp32 add/mul/cross sequence (NO reorder, NO FMA-introducing
// refactor). --fmad=false on this TU (CMake) forbids FMA contraction so the device
// arithmetic matches the host's -ffp-contract default add/mul exactly.
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/interop/cuda_vk_scatter.hpp"

#include "phi/backend.hpp"   // InitBestDevice (availability probe)
#include "phi/backend_cuda/cuda_internal.cuh"  // CudaBackend, CudaBackendMainStream (INT-F2)
#include "phi/backend_cuda/launch.cuh"  // LaunchCuda (the sole <<<>>> wrapper)

#include "math/transform.hpp"
#include "math/quat.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <cstring>
#include <memory>

namespace nuka::phi {

namespace {

// ===========================================================================
// R14 device math -- BIT-IDENTICAL to the host composition.
//
// math::Transform::operator* / math::Quat are host-only (their ctors/factories are
// constexpr __host__, NOT __host__ __device__ -- confirmed in math/transform.hpp,
// math/quat.hpp), so device code CANNOT construct them. Following the repo's
// instance_transform.cuh discipline, we operate on plain float scalars + the
// __host__ __device__ math::Vec3 ONLY, performing the EXACT SAME fp32 add/mul/cross
// sequence as the host (NO reorder, NO FMA-introducing refactor). --fmad=false on
// this TU forbids device FMA contraction so the arithmetic matches the host's.
// A quaternion is carried as 4 plain floats (w,x,y,z), never as a math::Quat.
// ===========================================================================

// Rotate a vector by a unit quaternion (w,x,y,z): q * v * q^-1. EXACTLY
// math::Quat::Rotate's form (math/quat.hpp:60 `t = 2*cross(qv,v); v + w*t +
// cross(qv,t)`), via the HD math::Vec3 ops.
__device__ inline math::Vec3 DevQuatRotate(float qw, float qx, float qy, float qz,
                                           const math::Vec3& v) {
    const math::Vec3 qv{qx, qy, qz};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + qw * t + qv.Cross(t);
}

// Hamilton product a*b on plain floats. EXACTLY math::Quat::operator* (math/quat.hpp:36).
__device__ inline void DevQuatMul(float aw, float ax, float ay, float az,
                                  float bw, float bx, float by, float bz,
                                  float* ow, float* ox, float* oy, float* oz) {
    *ow = aw * bw - ax * bx - ay * by - az * bz;
    *ox = aw * bx + ax * bw + ay * bz - az * by;
    *oy = aw * by - ax * bz + ay * bw + az * bx;
    *oz = aw * bz + ax * by - ay * bx + az * bw;
}

// Unit-quaternion normalize in place. EXACTLY math::Quat::Normalized (math/quat.hpp:53):
// n=sqrt(w^2+x^2+y^2+z^2); n<1e-12 -> Identity {1,0,0,0}; else divide. sqrtf to
// match the host's std::sqrt<float>.
__device__ inline void DevQuatNormalize(float* w, float* x, float* y, float* z) {
    const float n = sqrtf((*w) * (*w) + (*x) * (*x) + (*y) * (*y) + (*z) * (*z));
    if (n < 1e-12f) { *w = 1.0f; *x = 0.0f; *y = 0.0f; *z = 0.0f; return; }
    *w /= n; *x /= n; *y /= n; *z /= n;
}

// Lower a rigid (rotation quat + translation) to a column-major 4x4. EXACTLY the
// renderer's TransformToMatrix (vulkan_raster_renderer.cpp:113 /
// vulkan_present_renderer.cpp:99): same quaternion->rotation expansion + translation
// in column 3, m[col*4+row].
__device__ inline void DevTransformToMatrix(float qw, float qx, float qy, float qz,
                                            float px, float py, float pz, float* m) {
    const float x = qx, y = qy, z = qz, w = qw;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    // Baseline (col-major identity), then fill the rigid 3x4.
    m[3] = m[7] = m[11] = 0.0f;
    m[15] = 1.0f;
    // column 0
    m[0] = 1.0f - 2.0f * (yy + zz);
    m[1] = 2.0f * (xy + wz);
    m[2] = 2.0f * (xz - wy);
    // column 1
    m[4] = 2.0f * (xy - wz);
    m[5] = 1.0f - 2.0f * (xx + zz);
    m[6] = 2.0f * (yz + wx);
    // column 2
    m[8] = 2.0f * (xz + wy);
    m[9] = 2.0f * (yz - wx);
    m[10] = 1.0f - 2.0f * (xx + yy);
    // column 3 (translation)
    m[12] = px;
    m[13] = py;
    m[14] = pz;
}

// ===========================================================================
// The scatter kernel: one thread per instance row (one writer per row -> no float
// atomics). Resolves the row's FK pose from the SAME env-major device buffers
// DownloadField reads, composes with cached_visual_local, lowers to a mat4. For an
// unresolved / Static / out-of-range row, leaves the existing SSBO mat4 untouched
// (the renderer pre-seeded it with the instance bind pose -- mirrors
// HostDownloadPublisher keeping the bind-pose world_xform).
// ===========================================================================
__global__ void ScatterTransformsKernel(ScatterFkSource fk,
                                        const InstanceScatterRow* rows,
                                        uint32_t instance_count,
                                        float* out_mat4) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= instance_count) return;

    const InstanceScatterRow row = rows[i];

    // cached_visual_local as plain floats (pos3 + quat w,x,y,z) -- the SAME bytes
    // the host math::Transform holds (publisher laid them out as pos.xyz, rot.wxyz).
    const math::Vec3 cvl_pos{row.cached_visual_local[0], row.cached_visual_local[1],
                             row.cached_visual_local[2]};
    const float cvl_qw = row.cached_visual_local[3];
    const float cvl_qx = row.cached_visual_local[4];
    const float cvl_qy = row.cached_visual_local[5];
    const float cvl_qz = row.cached_visual_local[6];

    // Resolve the FK pose (env-major: env*per_env + row), matching the host
    // staging-buffer offsets in HostDownloadPublisher exactly. A null field ptr =>
    // that kind is unavailable this frame -> keep the bind pose. (We READ the POD
    // math::Transform fields; we never CONSTRUCT a math::Transform on device.)
    const math::Transform* fk_pose = nullptr;
    switch (row.kind) {
        case 1u: {  // Link
            if (fk.link_pose != nullptr && row.row < fk.links_per_env) {
                const auto* base = static_cast<const math::Transform*>(fk.link_pose);
                fk_pose = base + (fk.env_index * fk.links_per_env + row.row);
            }
            break;
        }
        case 2u: {  // Body
            if (fk.body_pose != nullptr && row.row < fk.bodies_per_env) {
                const auto* base = static_cast<const math::Transform*>(fk.body_pose);
                fk_pose = base + (fk.env_index * fk.bodies_per_env + row.row);
            }
            break;
        }
        case 3u: {  // Base
            if (fk.base_pose != nullptr) {
                const auto* base = static_cast<const math::Transform*>(fk.base_pose);
                fk_pose = base + fk.env_index;
            }
            break;
        }
        default:  // 0 = Static / unknown -> keep bind pose.
            break;
    }
    if (fk_pose == nullptr) return;  // keep the pre-seeded bind-pose mat4.

    // fk fields (POD reads -- no construction).
    const math::Vec3 fk_pos = fk_pose->position;
    const float fk_qw = fk_pose->rotation.w;
    const float fk_qx = fk_pose->rotation.x;
    const float fk_qy = fk_pose->rotation.y;
    const float fk_qz = fk_pose->rotation.z;

    // world = fk * cvl  (math::Transform::operator*, R14):
    //   position = fk.rotation.Rotate(cvl.position) + fk.position
    //   rotation = (fk.rotation * cvl.rotation).Normalized()
    const math::Vec3 world_pos =
        DevQuatRotate(fk_qw, fk_qx, fk_qy, fk_qz, cvl_pos) + fk_pos;
    float wqw, wqx, wqy, wqz;
    DevQuatMul(fk_qw, fk_qx, fk_qy, fk_qz, cvl_qw, cvl_qx, cvl_qy, cvl_qz,
               &wqw, &wqx, &wqy, &wqz);
    DevQuatNormalize(&wqw, &wqx, &wqy, &wqz);

    DevTransformToMatrix(wqw, wqx, wqy, wqz, world_pos.x, world_pos.y, world_pos.z,
                         out_mat4 + static_cast<size_t>(i) * 16u);
}

// ===========================================================================
// CudaVkScatter -- the concrete InteropScatterI.
// ===========================================================================
class CudaVkScatter final : public InteropScatterI {
public:
    CudaVkScatter() {
        // Dedicated non-blocking stream for the scatter (self-contained; does not
        // touch the World's backend internals). cudaStreamCreate on a healthy
        // device never fails in practice; if it does, stream_ stays 0 (default).
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
            stream_ = nullptr;
        }
        // INT-F2: an event used to order the scatter AFTER the World's FK write.
        // Recorded on the World's main stream each Scatter, waited on the scatter
        // stream before the launch. Timing-disabled (we never query elapsed time)
        // for the lowest record/wait overhead. If creation fails, fk_done_ stays 0
        // and Scatter degrades to running the scatter ON the main stream (still
        // correctly ordered -- same-stream RAW).
        if (cudaEventCreateWithFlags(&fk_done_, cudaEventDisableTiming) != cudaSuccess) {
            fk_done_ = nullptr;
        }
    }

    ~CudaVkScatter() override {
        ReleaseMemory();
        ReleaseSemaphores();
        if (d_rows_ != nullptr) cudaFree(d_rows_);
        if (fk_done_ != nullptr) cudaEventDestroy(fk_done_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    bool ImportMemory(const ExternalMemoryDesc& mem, uint32_t instance_capacity) override {
        ReleaseMemory();
        if (mem.fd < 0 || mem.size_bytes == 0) return false;

        cudaExternalMemoryHandleDesc desc{};
        desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = mem.fd;       // CUDA dup()s the fd internally
        desc.size = mem.size_bytes;
        if (mem.dedicated) desc.flags |= cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&ext_mem_, &desc) != cudaSuccess) {
            ext_mem_ = nullptr;
            return false;
        }

        cudaExternalMemoryBufferDesc buf{};
        buf.offset = 0;
        buf.size = mem.size_bytes;
        buf.flags = 0;
        if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr_, ext_mem_, &buf) != cudaSuccess) {
            mapped_ptr_ = nullptr;
            ReleaseMemory();
            return false;
        }
        instance_capacity_ = instance_capacity;
        return true;
    }

    bool ImportSemaphores(const ExternalSemaphoreDesc& sem) override {
        ReleaseSemaphores();
        bool any = false;
        if (sem.wait_fd >= 0) {
            cudaExternalSemaphoreHandleDesc d{};
            d.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
            d.handle.fd = sem.wait_fd;
            if (cudaImportExternalSemaphore(&wait_sem_, &d) == cudaSuccess) any = true;
            else wait_sem_ = nullptr;
        }
        if (sem.signal_fd >= 0) {
            cudaExternalSemaphoreHandleDesc d{};
            d.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
            d.handle.fd = sem.signal_fd;
            if (cudaImportExternalSemaphore(&signal_sem_, &d) == cudaSuccess) any = true;
            else signal_sem_ = nullptr;
        }
        return any;
    }

    bool Scatter(const ScatterFkSource& fk, const InstanceScatterRow* rows,
                 uint32_t instance_count) override {
        if (mapped_ptr_ == nullptr || rows == nullptr || instance_count == 0) {
            return false;
        }
        if (instance_count > instance_capacity_) return false;

        // INT-F2 (cross-stream read-after-write ordering): the scatter READS the
        // live FK buffers (link/body/base pose) the World wrote on its backend's
        // MAIN stream. Pick the stream the scatter runs on so that read is correctly
        // ordered after that write:
        //   * scatter stream + the World main stream both exist -> record an event on
        //     the World main stream NOW (captures "FK write done") and make the
        //     scatter stream wait on it before the launch. The scatter then overlaps
        //     other main-stream work yet never reads torn/stale transforms.
        //   * otherwise -> run the scatter ON the World main stream itself (same-stream
        //     RAW is implicitly ordered). If there is no main stream either (defensive:
        //     null backend), fall back to the scatter's own stream with no ordering.
        cudaStream_t main_stream = nullptr;
        if (fk.world_backend != nullptr) {
            main_stream = CudaBackendMainStream(
                reinterpret_cast<CudaBackend*>(fk.world_backend));
        }
        cudaStream_t launch_stream = stream_;
        if (stream_ != nullptr && main_stream != nullptr && fk_done_ != nullptr) {
            // Order the scatter stream after the FK write on the main stream.
            if (cudaEventRecord(fk_done_, main_stream) == cudaSuccess) {
                cudaStreamWaitEvent(stream_, fk_done_, 0);
            }
        } else if (main_stream != nullptr) {
            // No usable scatter stream / event -> run directly on the main stream so
            // the read trivially follows the FK write on the same stream.
            launch_stream = main_stream;
        }

        // Upload the (build-time constant) per-instance bindings to a persistent
        // device buffer; re-alloc only when the count grows (allocation-free in
        // steady state -- the binding never changes).
        if (instance_count > d_rows_capacity_) {
            if (d_rows_ != nullptr) { cudaFree(d_rows_); d_rows_ = nullptr; }
            if (cudaMalloc(&d_rows_, instance_count * sizeof(InstanceScatterRow)) !=
                cudaSuccess) {
                d_rows_ = nullptr;
                d_rows_capacity_ = 0;
                return false;
            }
            d_rows_capacity_ = instance_count;
        }
        if (cudaMemcpyAsync(d_rows_, rows,
                            instance_count * sizeof(InstanceScatterRow),
                            cudaMemcpyHostToDevice, launch_stream) != cudaSuccess) {
            return false;
        }

        // INT-7: fence the scatter behind the Vulkan-signalled "SSBO free" semaphore
        // when imported (binary, value ignored).
        if (wait_sem_ != nullptr) {
            cudaExternalSemaphoreWaitParams wp{};
            cudaWaitExternalSemaphoresAsync(&wait_sem_, &wp, 1, launch_stream);
        }

        LaunchScatterTransforms(fk, d_rows_, instance_count,
                                static_cast<float*>(mapped_ptr_), launch_stream);

        if (signal_sem_ != nullptr) {
            cudaExternalSemaphoreSignalParams sp{};
            cudaSignalExternalSemaphoresAsync(&signal_sem_, &sp, 1, launch_stream);
        } else {
            // No CUDA->Vulkan semaphore: make the SSBO coherent before the draw.
            cudaStreamSynchronize(launch_stream);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    }

private:
    void ReleaseMemory() {
        if (ext_mem_ != nullptr) { cudaDestroyExternalMemory(ext_mem_); ext_mem_ = nullptr; }
        mapped_ptr_ = nullptr;
        instance_capacity_ = 0;
    }
    void ReleaseSemaphores() {
        if (wait_sem_ != nullptr) { cudaDestroyExternalSemaphore(wait_sem_); wait_sem_ = nullptr; }
        if (signal_sem_ != nullptr) { cudaDestroyExternalSemaphore(signal_sem_); signal_sem_ = nullptr; }
    }

    cudaStream_t           stream_            = nullptr;
    cudaEvent_t            fk_done_           = nullptr;   // INT-F2: orders scatter after FK write
    cudaExternalMemory_t   ext_mem_           = nullptr;
    void*                  mapped_ptr_        = nullptr;   // device ptr into the imported SSBO
    uint32_t               instance_capacity_ = 0;
    cudaExternalSemaphore_t wait_sem_         = nullptr;
    cudaExternalSemaphore_t signal_sem_       = nullptr;
    InstanceScatterRow*    d_rows_            = nullptr;   // persistent device copy of bindings
    uint32_t               d_rows_capacity_   = 0;
};

}  // namespace

// The launcher: through launch.cuh so the only <<<>>> in this TU is the wrapper's.
void LaunchScatterTransforms(const ScatterFkSource& fk,
                             const InstanceScatterRow* d_rows,
                             uint32_t instance_count,
                             float* out_mat4,
                             cudaStream_t stream) {
    const uint32_t kBlock = 128u;
    const uint32_t grid = (instance_count + kBlock - 1u) / kBlock;
    LaunchCuda(ScatterTransformsKernel, dim3(grid), dim3(kBlock), 0u, stream,
               fk, d_rows, instance_count, out_mat4);
}

// STRONG definitions (win over runtime/app/interop_scatter_fallback.cpp's weak
// symbols when nuka_phi2_interop is linked). Returns nullptr when no CUDA device
// initializes (the phi "unavailable -> null" contract -> publisher degrades).
std::unique_ptr<InteropScatterI> CreateCudaVkScatter() {
    if (phi::InitBestDevice() == nullptr) return nullptr;
    return std::make_unique<CudaVkScatter>();
}

bool CudaVkScatterAvailable() { return phi::InitBestDevice() != nullptr; }

}  // namespace nuka::phi
