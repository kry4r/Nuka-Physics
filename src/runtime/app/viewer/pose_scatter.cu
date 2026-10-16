#include "runtime/app/viewer/pose_scatter.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <cuda_runtime.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstring>
#include <stdexcept>
#include <string>

namespace nuka::runtime::app::viewer {
namespace {

void CheckCuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

// The host composes fk * local with math::Transform and lowers it with the renderer's
// TransformToMatrix; these repeat its fp32 operation order (built with --fmad=false).
__device__ inline math::Vec3 QuatRotate(float qw, float qx, float qy, float qz, const math::Vec3& v) {
    const math::Vec3 qv{qx, qy, qz};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + qw * t + qv.Cross(t);
}

__device__ inline void WriteMatrix(float qw, float qx, float qy, float qz, float px, float py, float pz,
                                   float* m) {
    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;
    m[0] = 1.0f - 2.0f * (yy + zz);
    m[1] = 2.0f * (xy + wz);
    m[2] = 2.0f * (xz - wy);
    m[3] = 0.0f;
    m[4] = 2.0f * (xy - wz);
    m[5] = 1.0f - 2.0f * (xx + zz);
    m[6] = 2.0f * (yz + wx);
    m[7] = 0.0f;
    m[8] = 2.0f * (xz + wy);
    m[9] = 2.0f * (yz - wx);
    m[10] = 1.0f - 2.0f * (xx + yy);
    m[11] = 0.0f;
    m[12] = px;
    m[13] = py;
    m[14] = pz;
    m[15] = 1.0f;
}

__global__ void ScatterKernel(const ScatterRow* rows, uint32_t count, const math::Transform* links,
                              uint32_t link_count, const math::Transform* bodies, uint32_t body_count,
                              float* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const ScatterRow r = rows[i];
    float* m = out + static_cast<size_t>(i) * 16u;
    const math::Vec3 local_pos{r.local[0], r.local[1], r.local[2]};
    const math::Transform* fk = nullptr;
    if (r.kind == 1u && links != nullptr && r.row < link_count) fk = links + r.row;
    if (r.kind == 2u && bodies != nullptr && r.row < body_count) fk = bodies + r.row;
    if (fk == nullptr) {
        WriteMatrix(r.local[3], r.local[4], r.local[5], r.local[6], local_pos.x, local_pos.y, local_pos.z, m);
        return;
    }
    const math::Vec3 fk_pos = fk->position;
    const float aw = fk->rotation.w, ax = fk->rotation.x, ay = fk->rotation.y, az = fk->rotation.z;
    const float bw = r.local[3], bx = r.local[4], by = r.local[5], bz = r.local[6];
    const math::Vec3 pos = QuatRotate(aw, ax, ay, az, local_pos) + fk_pos;
    float w = aw * bw - ax * bx - ay * by - az * bz;
    float x = aw * bx + ax * bw + ay * bz - az * by;
    float y = aw * by - ax * bz + ay * bw + az * bx;
    float z = aw * bz + ax * by - ay * bx + az * bw;
    const float n = sqrtf(w * w + x * x + y * y + z * z);
    if (n < 1e-12f) {
        w = 1.0f;
        x = y = z = 0.0f;
    } else {
        w /= n;
        x /= n;
        y /= n;
        z /= n;
    }
    WriteMatrix(w, x, y, z, pos.x, pos.y, pos.z, m);
}

const math::Transform* EnvRows(nuka_world_handle world, nuka_state_field_t field, uint32_t env, uint32_t per_env) {
    nuka_buffer_view_t view{};
    if (per_env == 0u || nuka_world_get_buffer_view(world, field, &view) != NUKA_RESULT_OK) return nullptr;
    return static_cast<const math::Transform*>(view.device_ptr) + static_cast<size_t>(env) * per_env;
}

cudaStream_t WorldStream(nuka_world_handle world) {
    void* stream = nullptr;
    nuka_world_get_stream(world, &stream);
    return static_cast<cudaStream_t>(stream);
}

// Win32 handles stay the importer's to close; a file descriptor CUDA imported is CUDA's.
struct CloseHandles {
    PoseSlotImport& slots;
    ~CloseHandles() {
#ifdef _WIN32
        if (slots.memory_handle != nullptr) CloseHandle(slots.memory_handle);
        if (slots.semaphore_handle != nullptr) CloseHandle(slots.semaphore_handle);
#else
        if (slots.memory_fd >= 0) close(slots.memory_fd);
        if (slots.semaphore_fd >= 0) close(slots.semaphore_fd);
#endif
        slots.memory_handle = slots.semaphore_handle = nullptr;
        slots.memory_fd = slots.semaphore_fd = -1;
    }
};

}  // namespace

void BindWorldDevice(nuka_world_handle world) {
    int device = 0;
    CheckCuda(cudaStreamGetDevice(WorldStream(world), &device), "cudaStreamGetDevice");
    CheckCuda(cudaSetDevice(device), "cudaSetDevice");
}

struct PoseScatter::Impl {
    int device = 0;
    cudaExternalMemory_t memory = nullptr;
    void* mapped = nullptr;
    cudaExternalSemaphore_t timeline = nullptr;
    uint64_t slot_floats = 0u;
    ScatterRow* rows = nullptr;
    uint32_t row_count = 0u;
    uint32_t row_capacity = 0u;

    ~Impl() {
        if (rows != nullptr) cudaFree(rows);
        if (timeline != nullptr) cudaDestroyExternalSemaphore(timeline);
        if (mapped != nullptr) cudaFree(mapped);
        if (memory != nullptr) cudaDestroyExternalMemory(memory);
    }
};

PoseScatter::PoseScatter(nuka_world_handle world, PoseSlotImport& slots) : impl_(std::make_unique<Impl>()) {
    const CloseHandles close_handles{slots};
    CheckCuda(cudaStreamGetDevice(WorldStream(world), &impl_->device), "cudaStreamGetDevice");
    CheckCuda(cudaSetDevice(impl_->device), "cudaSetDevice");
    cudaDeviceProp prop{};
    CheckCuda(cudaGetDeviceProperties(&prop, impl_->device), "cudaGetDeviceProperties");
    if (std::memcmp(prop.uuid.bytes, slots.device_uuid, sizeof(slots.device_uuid)) != 0)
        throw std::runtime_error("the Vulkan device is not the world's CUDA device");

    cudaExternalMemoryHandleDesc memory{};
#ifdef _WIN32
    memory.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memory.handle.win32.handle = slots.memory_handle;
#else
    memory.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memory.handle.fd = slots.memory_fd;
#endif
    memory.size = slots.memory_bytes;
    memory.flags = cudaExternalMemoryDedicated;
    CheckCuda(cudaImportExternalMemory(&impl_->memory, &memory), "cudaImportExternalMemory");
    slots.memory_fd = -1;
    cudaExternalMemoryBufferDesc buffer{};
    buffer.size = slots.memory_bytes;
    CheckCuda(cudaExternalMemoryGetMappedBuffer(&impl_->mapped, impl_->memory, &buffer),
              "cudaExternalMemoryGetMappedBuffer");
    impl_->slot_floats = slots.slot_bytes / sizeof(float);

    cudaExternalSemaphoreHandleDesc semaphore{};
#ifdef _WIN32
    semaphore.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    semaphore.handle.win32.handle = slots.semaphore_handle;
#else
    semaphore.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    semaphore.handle.fd = slots.semaphore_fd;
#endif
    CheckCuda(cudaImportExternalSemaphore(&impl_->timeline, &semaphore), "cudaImportExternalSemaphore");
    slots.semaphore_fd = -1;
}

PoseScatter::~PoseScatter() = default;

void PoseScatter::SetRows(nuka_world_handle world, const std::vector<ScatterRow>& rows) {
    const auto count = static_cast<uint32_t>(rows.size());
    if (count > impl_->row_capacity) {
        if (impl_->rows != nullptr) CheckCuda(cudaFree(impl_->rows), "cudaFree");
        impl_->rows = nullptr;
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl_->rows), count * sizeof(ScatterRow)), "cudaMalloc");
        impl_->row_capacity = count;
    }
    impl_->row_count = count;
    if (count > 0u)
        CheckCuda(cudaMemcpyAsync(impl_->rows, rows.data(), count * sizeof(ScatterRow), cudaMemcpyHostToDevice,
                                  WorldStream(world)),
                  "cudaMemcpyAsync(rows)");
}

void PoseScatter::Scatter(nuka_world_handle world, uint32_t env, uint32_t links_per_env, uint32_t bodies_per_env,
                          uint32_t slot, uint64_t value) {
    const cudaStream_t stream = WorldStream(world);
    float* out = static_cast<float*>(impl_->mapped) + slot * impl_->slot_floats;
    if (impl_->row_count > 0u) {
        const uint32_t block = 128u;
        ScatterKernel<<<(impl_->row_count + block - 1u) / block, block, 0, stream>>>(
            impl_->rows, impl_->row_count,
            EnvRows(world, NUKA_FIELD_ARTICULATION_LINK_POSE, env, links_per_env), links_per_env,
            EnvRows(world, NUKA_FIELD_RIGID_BODY_TRANSFORM, env, bodies_per_env), bodies_per_env, out);
        CheckCuda(cudaGetLastError(), "pose scatter launch");
    }
    cudaExternalSemaphoreSignalParams signal{};
    signal.params.fence.value = value;
    CheckCuda(cudaSignalExternalSemaphoresAsync(&impl_->timeline, &signal, 1u, stream),
              "cudaSignalExternalSemaphoresAsync");
}

}  // namespace nuka::runtime::app::viewer
