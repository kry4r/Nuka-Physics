// ---------------------------------------------------------------------------
// nuka::constraint::gpu::contact_generation implementation
// ---------------------------------------------------------------------------

#include "constraint/gpu/contact_generation.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::constraint::gpu {

namespace {

__device__ math::Vec3 MakeVec3(float x, float y, float z) {
    math::Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ math::Vec3 UnitX() {
    return MakeVec3(1.0f, 0.0f, 0.0f);
}

__device__ math::Vec3 UnitY() {
    return MakeVec3(0.0f, 1.0f, 0.0f);
}

__device__ math::Vec3 UnitZ() {
    return MakeVec3(0.0f, 0.0f, 1.0f);
}

__device__ math::Quat MakeQuat(float w, float x, float y, float z) {
    math::Quat q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

__device__ math::Vec3 Add(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ math::Vec3 Sub(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ math::Vec3 Scale(math::Vec3 v, float s) {
    return MakeVec3(v.x * s, v.y * s, v.z * s);
}

__device__ float Dot(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float Length(math::Vec3 v) {
    return sqrtf(Dot(v, v));
}

__device__ math::Vec3 Normalize(math::Vec3 v) {
    const float length = Length(v);
    if (length < 1.0e-6f) {
        return UnitY();
    }
    return Scale(v, 1.0f / length);
}

__device__ math::Vec3 Cross(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ math::Vec3 Rotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const math::Vec3 t = Scale(Cross(qv, v), 2.0f);
    return Add(Add(v, Scale(t, q.w)), Cross(qv, t));
}

__device__ math::Vec3 TransformPoint(math::Transform transform, math::Vec3 point) {
    return Add(Rotate(transform.rotation, point), transform.position);
}

__device__ math::Quat Mul(math::Quat a, math::Quat b) {
    return MakeQuat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = Mul(a.rotation, b.rotation);
    return result;
}

__device__ math::Vec3 AabbCenter(collision::AABB aabb) {
    return Scale(Add(aabb.min, aabb.max), 0.5f);
}

__device__ bool IsDynamic(scene::BodyId body_id, const float* inv_masses) {
    return inv_masses[body_id] > 0.0f;
}

__device__ void AddSinglePointManifold(uint32_t body_a,
                                       uint32_t body_b,
                                       math::Vec3 position,
                                       math::Vec3 normal,
                                       float penetration,
                                       constraint::ContactManifold* manifold) {
    if (penetration <= 0.0f) {
        return;
    }

    manifold->body_a = body_a;
    manifold->body_b = body_b;
    manifold->point_count = 1u;
    manifold->friction = 0.5f;
    manifold->restitution = 0.0f;
    manifold->points[0].position = position;
    manifold->points[0].normal = Normalize(normal);
    manifold->points[0].penetration = penetration;
    manifold->points[0].normal_impulse = 0.0f;
    manifold->points[0].friction_impulse_1 = 0.0f;
    manifold->points[0].friction_impulse_2 = 0.0f;
}

__device__ void GeneratePlaneContact(scene::BodyId dynamic_body,
                                     scene::BodyId plane_body,
                                     scene::ShapeType dynamic_type,
                                     math::Transform dynamic_transform,
                                     math::Transform plane_transform,
                                     collision::AABB dynamic_aabb,
                                     float dynamic_radius,
                                     constraint::ContactManifold* manifold) {
    const float plane_y = plane_transform.position.y;
    if (dynamic_type == scene::ShapeType::Sphere) {
        const float bottom = dynamic_transform.position.y - dynamic_radius;
        AddSinglePointManifold(
            dynamic_body,
            plane_body,
            MakeVec3(dynamic_transform.position.x, plane_y, dynamic_transform.position.z),
            UnitY(),
            plane_y - bottom,
            manifold);
        return;
    }

    const float bottom = dynamic_aabb.min.y;
    const math::Vec3 center = AabbCenter(dynamic_aabb);
    AddSinglePointManifold(
        dynamic_body,
        plane_body,
        MakeVec3(center.x, plane_y, center.z),
        UnitY(),
        plane_y - bottom,
        manifold);
}

__device__ void GenerateSphereSphereContact(scene::BodyId body_a,
                                            scene::BodyId body_b,
                                            math::Transform transform_a,
                                            math::Transform transform_b,
                                            float radius_a,
                                            float radius_b,
                                            constraint::ContactManifold* manifold) {
    const math::Vec3 delta = Sub(transform_a.position, transform_b.position);
    const float distance = Length(delta);
    const float penetration = radius_a + radius_b - distance;
    if (penetration <= 0.0f) {
        return;
    }

    const math::Vec3 normal = distance > 1.0e-6f
        ? Scale(delta, 1.0f / distance)
        : UnitY();
    const math::Vec3 position = Add(transform_b.position, Scale(normal, radius_b));
    AddSinglePointManifold(body_a, body_b, position, normal, penetration, manifold);
}

__device__ void GenerateBoxBoxContact(scene::BodyId body_a,
                                      scene::BodyId body_b,
                                      collision::AABB a,
                                      collision::AABB b,
                                      constraint::ContactManifold* manifold) {
    const float overlap_x = fminf(a.max.x, b.max.x) - fmaxf(a.min.x, b.min.x);
    const float overlap_y = fminf(a.max.y, b.max.y) - fmaxf(a.min.y, b.min.y);
    const float overlap_z = fminf(a.max.z, b.max.z) - fmaxf(a.min.z, b.min.z);
    if (overlap_x <= 0.0f || overlap_y <= 0.0f || overlap_z <= 0.0f) {
        return;
    }

    math::Vec3 normal = UnitX();
    float penetration = overlap_x;
    if (overlap_y < penetration) {
        normal = UnitY();
        penetration = overlap_y;
    }
    if (overlap_z < penetration) {
        normal = UnitZ();
        penetration = overlap_z;
    }

    if (Dot(Sub(AabbCenter(a), AabbCenter(b)), normal) < 0.0f) {
        normal = Scale(normal, -1.0f);
    }

    const math::Vec3 position = MakeVec3(
        (fmaxf(a.min.x, b.min.x) + fminf(a.max.x, b.max.x)) * 0.5f,
        (fmaxf(a.min.y, b.min.y) + fminf(a.max.y, b.max.y)) * 0.5f,
        (fmaxf(a.min.z, b.min.z) + fminf(a.max.z, b.max.z)) * 0.5f);
    AddSinglePointManifold(body_a, body_b, position, normal, penetration, manifold);
}

__device__ bool IsPlane(scene::ShapeType type) {
    return type == scene::ShapeType::Plane;
}

__device__ bool IsSphere(scene::ShapeType type) {
    return type == scene::ShapeType::Sphere;
}

__device__ constraint::ContactManifold EmptyManifold() {
    constraint::ContactManifold manifold;
    manifold.body_a = ~0u;
    manifold.body_b = ~0u;
    manifold.point_count = 0u;
    manifold.friction = 0.5f;
    manifold.restitution = 0.0f;
    return manifold;
}

__global__ void InitializeContactReportKernel(const uint32_t* pair_count,
                                              CudaContactReport* report) {
    report->pair_count = *pair_count;
    report->contact_manifold_count = 0u;
    report->contact_point_count = 0u;
}

__global__ void GenerateContactsKernel(
    uint32_t pair_slot_count,
    const collision::CollisionPair* pairs,
    const uint8_t* pair_active_flags,
    const collision::AABB* aabbs,
    const math::Transform* poses,
    const float* inv_masses,
    const scene::BodyId* shape_body_ids,
    const scene::ShapeType* shape_types,
    const math::Transform* shape_local_transforms,
    const float* shape_radii,
    constraint::ContactManifold* manifolds) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pair_slot_count) {
        return;
    }

    manifolds[slot] = EmptyManifold();

    if (pair_active_flags[slot] == 0u) {
        return;
    }

    const collision::CollisionPair pair = pairs[slot];
    const scene::BodyId first_body = shape_body_ids[pair.body_a];
    const scene::BodyId second_body = shape_body_ids[pair.body_b];
    if (first_body == second_body) {
        return;
    }

    const bool first_dynamic = IsDynamic(first_body, inv_masses);
    const bool second_dynamic = IsDynamic(second_body, inv_masses);
    if (!first_dynamic && !second_dynamic) {
        return;
    }

    uint32_t dynamic_shape = pair.body_a;
    uint32_t other_shape = pair.body_b;
    if (!first_dynamic && second_dynamic) {
        dynamic_shape = pair.body_b;
        other_shape = pair.body_a;
    }

    const scene::BodyId dynamic_body = shape_body_ids[dynamic_shape];
    const scene::BodyId other_body = shape_body_ids[other_shape];
    const scene::ShapeType dynamic_type = shape_types[dynamic_shape];
    const scene::ShapeType other_type = shape_types[other_shape];
    const math::Transform dynamic_transform =
        Compose(poses[dynamic_body], shape_local_transforms[dynamic_shape]);
    const math::Transform other_transform =
        Compose(poses[other_body], shape_local_transforms[other_shape]);

    if (IsPlane(other_type)) {
        GeneratePlaneContact(dynamic_body,
                             other_body,
                             dynamic_type,
                             dynamic_transform,
                             other_transform,
                             aabbs[dynamic_shape],
                             shape_radii[dynamic_shape],
                             &manifolds[slot]);
        return;
    }

    if (IsPlane(dynamic_type)) {
        GeneratePlaneContact(other_body,
                             dynamic_body,
                             other_type,
                             other_transform,
                             dynamic_transform,
                             aabbs[other_shape],
                             shape_radii[other_shape],
                             &manifolds[slot]);
        return;
    }

    if (IsSphere(dynamic_type) && IsSphere(other_type)) {
        GenerateSphereSphereContact(dynamic_body,
                                    other_body,
                                    dynamic_transform,
                                    other_transform,
                                    shape_radii[dynamic_shape],
                                    shape_radii[other_shape],
                                    &manifolds[slot]);
        return;
    }

    GenerateBoxBoxContact(dynamic_body,
                          other_body,
                          aabbs[dynamic_shape],
                          aabbs[other_shape],
                          &manifolds[slot]);
}

__global__ void CountGeneratedContactKernel(uint32_t pair_slot_count,
                                            const constraint::ContactManifold* manifolds,
                                            CudaContactReport* report) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pair_slot_count || manifolds[slot].point_count == 0u) {
        return;
    }

    atomicAdd(&report->contact_manifold_count, 1u);
    atomicAdd(&report->contact_point_count, manifolds[slot].point_count);
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

template <typename T>
std::vector<T> DownloadVector(const phi::Buffer& buffer, uint32_t count) {
    std::vector<T> values(count);
    if (!values.empty()) {
        buffer.CopyToHost(values.data(), values.size() * sizeof(T));
    }
    return values;
}

} // namespace

CudaContactResult::CudaContactResult(uint32_t pair_slot_count,
                                     phi::Buffer manifolds,
                                     phi::Buffer report)
    : pair_slot_count_(pair_slot_count)
    , manifolds_(std::move(manifolds))
    , report_(std::move(report)) {}

std::vector<constraint::ContactManifold> CudaContactResult::DownloadManifolds() const {
    const auto all_manifolds =
        DownloadVector<constraint::ContactManifold>(manifolds_, pair_slot_count_);

    std::vector<constraint::ContactManifold> contacts;
    for (const auto& manifold : all_manifolds) {
        if (manifold.point_count > 0u) {
            contacts.push_back(manifold);
        }
    }
    return contacts;
}

CudaContactReport CudaContactResult::DownloadReport() const {
    CudaContactReport report;
    if (report_.Size() > 0u) {
        report_.CopyToHost(&report, sizeof(report));
    }
    return report;
}

CudaContactResult GenerateCudaContacts(
    const runtime::gpu::DeviceWorld& device_world,
    const collision::gpu::CudaBroadphaseResult& broadphase) {
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error(
            "GenerateCudaContacts requires UploadDeviceState before generating CUDA contacts");
    }

    phi::Buffer manifolds(
        broadphase.PairSlotCount() * sizeof(constraint::ContactManifold),
        phi::MemoryKind::Device);
    phi::Buffer report(sizeof(CudaContactReport), phi::MemoryKind::Device);

    InitializeContactReportKernel<<<1, 1>>>(
        broadphase.DevicePairCount(),
        static_cast<CudaContactReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "InitializeContactReportKernel launch");

    if (broadphase.PairSlotCount() == 0u) {
        return CudaContactResult(0u, std::move(manifolds), std::move(report));
    }

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t blocks = (broadphase.PairSlotCount() + kBlockSize - 1u) / kBlockSize;
    GenerateContactsKernel<<<blocks, kBlockSize>>>(
        broadphase.PairSlotCount(),
        broadphase.DevicePairs(),
        broadphase.DevicePairActiveFlags(),
        broadphase.DeviceAabbs(),
        device_world.DevicePoses(),
        device_world.DeviceInvMasses(),
        device_world.DeviceShapeBodyIds(),
        device_world.DeviceShapeTypes(),
        device_world.DeviceShapeLocalTransforms(),
        device_world.DeviceShapeRadii(),
        static_cast<constraint::ContactManifold*>(manifolds.Data()));
    CheckCuda(cudaGetLastError(), "GenerateContactsKernel launch");
    CountGeneratedContactKernel<<<blocks, kBlockSize>>>(
        broadphase.PairSlotCount(),
        static_cast<constraint::ContactManifold*>(manifolds.Data()),
        static_cast<CudaContactReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "CountGeneratedContactKernel launch");

    return CudaContactResult(broadphase.PairSlotCount(),
                             std::move(manifolds),
                             std::move(report));
}

} // namespace nuka::constraint::gpu
