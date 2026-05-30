// ---------------------------------------------------------------------------
// nuka::collision::gpu::broadphase implementation
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"

#include "math/cuda_vec_ops.cuh"
#include "phi/buffer_transfer.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision::gpu {

namespace {

// Small-vector / quaternion primitives now come from the shared device math
// library (math/cuda_vec_ops.cuh). Bodies are bit-identical to the former local
// copies; `Rotate` -> RotateShort and `Mul` -> QuatMul (used by the file-local
// TransformPoint / Compose below). Buffer helpers come from
// phi/buffer_transfer.hpp.
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::MakeQuat;
using mg::MakeVec3;
using mg::QuatMul;
using mg::RotateShort;
using mg::Scale;
using mg::Sub;

__device__ math::Vec3 TransformPoint(math::Transform transform, math::Vec3 point) {
    return Add(RotateShort(transform.rotation, point), transform.position);
}

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = QuatMul(a.rotation, b.rotation);
    return result;
}

__device__ void Expand(collision::AABB& aabb, math::Vec3 point) {
    aabb.min.x = fminf(aabb.min.x, point.x);
    aabb.min.y = fminf(aabb.min.y, point.y);
    aabb.min.z = fminf(aabb.min.z, point.z);
    aabb.max.x = fmaxf(aabb.max.x, point.x);
    aabb.max.y = fmaxf(aabb.max.y, point.y);
    aabb.max.z = fmaxf(aabb.max.z, point.z);
}

__device__ collision::AABB SphereAabb(math::Vec3 center, float radius) {
    collision::AABB aabb;
    const math::Vec3 extents = MakeVec3(radius, radius, radius);
    aabb.min = Sub(center, extents);
    aabb.max = Add(center, extents);
    return aabb;
}

__device__ collision::AABB PlaneAabb(float plane_y) {
    collision::AABB aabb;
    aabb.min = MakeVec3(-1.0e6f, plane_y - 0.01f, -1.0e6f);
    aabb.max = MakeVec3(1.0e6f, plane_y + 0.01f, 1.0e6f);
    return aabb;
}

__device__ collision::AABB BoxAabb(math::Transform transform, math::Vec3 half_extents) {
    collision::AABB aabb;
    aabb.min = MakeVec3(FLT_MAX, FLT_MAX, FLT_MAX);
    aabb.max = MakeVec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int corner_index = 0; corner_index < 8; ++corner_index) {
        const math::Vec3 corner = MakeVec3(
            (corner_index & 1) ? half_extents.x : -half_extents.x,
            (corner_index & 2) ? half_extents.y : -half_extents.y,
            (corner_index & 4) ? half_extents.z : -half_extents.z);
        Expand(aabb, TransformPoint(transform, corner));
    }
    return aabb;
}

__device__ bool Overlaps(collision::AABB a, collision::AABB b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

__device__ uint32_t PairSlot(uint32_t shape_count, uint32_t i, uint32_t j) {
    return (i * (2u * shape_count - i - 1u)) / 2u + (j - i - 1u);
}

__global__ void GenerateAabbsKernel(uint32_t shape_count,
                                    const math::Transform* poses,
                                    const scene::BodyId* shape_body_ids,
                                    const scene::ShapeType* shape_types,
                                    const math::Transform* shape_local_transforms,
                                    const math::Vec3* shape_half_extents,
                                    const float* shape_radii,
                                    collision::AABB* aabbs) {
    const uint32_t shape_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (shape_index >= shape_count) {
        return;
    }

    const scene::BodyId body_id = shape_body_ids[shape_index];
    const math::Transform world_transform =
        Compose(poses[body_id], shape_local_transforms[shape_index]);
    const scene::ShapeType type = shape_types[shape_index];

    if (type == scene::ShapeType::Sphere) {
        aabbs[shape_index] = SphereAabb(world_transform.position, shape_radii[shape_index]);
        return;
    }
    if (type == scene::ShapeType::Plane) {
        aabbs[shape_index] = PlaneAabb(world_transform.position.y);
        return;
    }

    aabbs[shape_index] = BoxAabb(world_transform, shape_half_extents[shape_index]);
}

__global__ void GeneratePairSlotsKernel(uint32_t shape_count,
                                        const collision::AABB* aabbs,
                                        collision::CollisionPair* pairs,
                                        uint8_t* pair_active_flags,
                                        uint32_t* pair_count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shape_count) {
        return;
    }

    for (uint32_t j = i + 1u; j < shape_count; ++j) {
        const uint32_t slot = PairSlot(shape_count, i, j);
        pairs[slot] = {i, j};
        const bool overlaps = Overlaps(aabbs[i], aabbs[j]);
        pair_active_flags[slot] = overlaps ? 1u : 0u;
        if (overlaps) {
            atomicAdd(pair_count, 1u);
        }
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// DownloadVector(buf, count) now comes from the shared host buffer-transfer
// header (phi/buffer_transfer.hpp); the former local copy was byte-identical.
using ::nuka::phi::DownloadVector;

} // namespace

CudaBroadphaseResult::CudaBroadphaseResult(uint32_t shape_count,
                                           uint32_t pair_slot_count,
                                           phi::Buffer aabbs,
                                           phi::Buffer pairs,
                                           phi::Buffer pair_active_flags,
                                           phi::Buffer pair_count)
    : shape_count_(shape_count)
    , pair_slot_count_(pair_slot_count)
    , aabbs_(std::move(aabbs))
    , pairs_(std::move(pairs))
    , pair_active_flags_(std::move(pair_active_flags))
    , pair_count_(std::move(pair_count)) {}

uint32_t CudaBroadphaseResult::PairCount() const {
    if (pair_count_.Size() == 0u) {
        return 0u;
    }

    uint32_t pair_count = 0;
    pair_count_.CopyToHost(&pair_count, sizeof(pair_count));
    return pair_count;
}

const collision::AABB* CudaBroadphaseResult::DeviceAabbs() const {
    return static_cast<const collision::AABB*>(aabbs_.Data());
}

const collision::CollisionPair* CudaBroadphaseResult::DevicePairs() const {
    return static_cast<const collision::CollisionPair*>(pairs_.Data());
}

const uint8_t* CudaBroadphaseResult::DevicePairActiveFlags() const {
    return static_cast<const uint8_t*>(pair_active_flags_.Data());
}

const uint32_t* CudaBroadphaseResult::DevicePairCount() const {
    return static_cast<const uint32_t*>(pair_count_.Data());
}

std::vector<collision::AABB> CudaBroadphaseResult::DownloadAabbs() const {
    return DownloadVector<collision::AABB>(aabbs_, shape_count_);
}

std::vector<collision::CollisionPair> CudaBroadphaseResult::DownloadPairs() const {
    const auto pairs = DownloadVector<collision::CollisionPair>(pairs_, pair_slot_count_);
    const auto flags = DownloadVector<uint8_t>(pair_active_flags_, pair_slot_count_);

    std::vector<collision::CollisionPair> active_pairs;
    active_pairs.reserve(PairCount());
    for (uint32_t slot = 0; slot < pair_slot_count_; ++slot) {
        if (flags[slot] != 0u) {
            active_pairs.push_back(pairs[slot]);
        }
    }
    return active_pairs;
}

CudaBroadphaseResult BuildCudaBroadphase(const phi::DeviceContext& context,
                                         const runtime::gpu::DeviceWorld& device_world) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error(
            "BuildCudaBroadphase requires UploadDeviceState before generating CUDA AABBs");
    }

    const uint32_t shape_count = device_world.ShapeCount();
    const uint32_t pair_slot_count = shape_count > 1u
        ? (shape_count * (shape_count - 1u)) / 2u
        : 0u;

    phi::Buffer aabbs(shape_count * sizeof(collision::AABB), phi::MemoryKind::Device);
    phi::Buffer pairs(pair_slot_count * sizeof(collision::CollisionPair),
                      phi::MemoryKind::Device);
    phi::Buffer active_flags(pair_slot_count * sizeof(uint8_t), phi::MemoryKind::Device);
    phi::Buffer pair_count(sizeof(uint32_t), phi::MemoryKind::Device);
    CheckCuda(cudaMemsetAsync(pair_count.Data(), 0, sizeof(uint32_t), stream),
              "cudaMemsetAsync pair count");

    if (shape_count == 0u) {
        context.stream.Synchronize();
        return CudaBroadphaseResult(0, 0,
                                    std::move(aabbs),
                                    std::move(pairs),
                                    std::move(active_flags),
                                    std::move(pair_count));
    }

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t shape_blocks = (shape_count + kBlockSize - 1u) / kBlockSize;
    GenerateAabbsKernel<<<shape_blocks, kBlockSize, 0, stream>>>(
        shape_count,
        device_world.DevicePoses(),
        device_world.DeviceShapeBodyIds(),
        device_world.DeviceShapeTypes(),
        device_world.DeviceShapeLocalTransforms(),
        device_world.DeviceShapeHalfExtents(),
        device_world.DeviceShapeRadii(),
        static_cast<collision::AABB*>(aabbs.Data()));
    CheckCuda(cudaGetLastError(), "GenerateAabbsKernel launch");

    if (pair_slot_count > 0u) {
        GeneratePairSlotsKernel<<<shape_blocks, kBlockSize, 0, stream>>>(
            shape_count,
            static_cast<const collision::AABB*>(aabbs.Data()),
            static_cast<collision::CollisionPair*>(pairs.Data()),
            static_cast<uint8_t*>(active_flags.Data()),
            static_cast<uint32_t*>(pair_count.Data()));
        CheckCuda(cudaGetLastError(), "GeneratePairSlotsKernel launch");
    }

    context.stream.Synchronize();
    return CudaBroadphaseResult(shape_count,
                                pair_slot_count,
                                std::move(aabbs),
                                std::move(pairs),
                                std::move(active_flags),
                                std::move(pair_count));
}

CudaBroadphaseResult BuildCudaBroadphase(const runtime::gpu::DeviceWorld& device_world) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildCudaBroadphase(context, device_world);
}

} // namespace nuka::collision::gpu
