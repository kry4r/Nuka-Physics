// ---------------------------------------------------------------------------
// Batched-sensor device scatter gate: device world transforms + world-AABBs match
// the host fk*cvl composition byte-for-byte.
//
// ScatterEnvInstances composes, per (env x instance), world = fk(pose) * cvl from
// env-major device LinkPose/BodyPose/BasePose buffers and writes a DevInstance[E*M]
// table + an env-major world-AABB[E*M] array (the batched LBVH build's leaf bounds).
// This drives it over >=3 envs covering all 3 mount kinds (Link/Body/Base) + a
// Static row and asserts the outputs are BYTE-IDENTICAL (memcmp) to a host
// reference built with math::Transform::operator* + the SAME shared MakeDevInstance
// / InstanceWorldAabb. Compiled -ffp-contract=off to pair with the kernel's
// --fmad=false so host==device is bit-for-bit.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend_cuda/rt/sensor_scatter.hpp"
#include "phi/backend_cuda/rt/two_level_render_kernels.cuh"
#include "phi/interop_scatter.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

namespace {

using nuka::collision::AABB;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::phi::InstanceScatterRow;
using nuka::phi::ScatterFkSource;
using nuka::rt::DevInstance;
using nuka::rt::SensorBlasRef;

#define NK_CUDA_OK(call)                                          \
    do {                                                          \
        const cudaError_t e_ = (call);                           \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_);    \
    } while (0)

// A small distinct rigid transform (axis-angle so the quaternion is non-trivial).
Transform MakePose(float px, float py, float pz, float ax, float ay, float az,
                   float angle) {
    return Transform{Vec3{px, py, pz}, Quat::FromAxisAngle(Vec3{ax, ay, az}, angle)};
}

// cached_visual_local row from a Transform (pos3 + quat w,x,y,z), the flat 7-float
// the InstanceScatterRow carries.
void SetCvl(InstanceScatterRow& row, const Transform& cvl) {
    row.cached_visual_local[0] = cvl.position.x;
    row.cached_visual_local[1] = cvl.position.y;
    row.cached_visual_local[2] = cvl.position.z;
    row.cached_visual_local[3] = cvl.rotation.w;
    row.cached_visual_local[4] = cvl.rotation.x;
    row.cached_visual_local[5] = cvl.rotation.y;
    row.cached_visual_local[6] = cvl.rotation.z;
}

constexpr uint32_t kEnvCount = 4u;
constexpr uint32_t kLinksPerEnv = 5u;
constexpr uint32_t kBodiesPerEnv = 3u;

}  // namespace

// One scatter over E envs writes DevInstances + world-AABBs byte-identical to the
// host fk*cvl reference, across Link/Body/Base/Static rows.
TEST(SensorScatter, DeviceMatchesHostFkCvl) {
    // --- Env-major device pose buffers (math::Transform rows). ----------------
    std::vector<Transform> link_pose(static_cast<size_t>(kEnvCount) * kLinksPerEnv);
    std::vector<Transform> body_pose(static_cast<size_t>(kEnvCount) * kBodiesPerEnv);
    std::vector<Transform> base_pose(kEnvCount);
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        const float s = 1.0f + 0.37f * static_cast<float>(e);
        for (uint32_t l = 0; l < kLinksPerEnv; ++l) {
            link_pose[e * kLinksPerEnv + l] =
                MakePose(0.5f * l + s, -0.3f * l, 0.2f * e + 0.1f * l, 0.2f, 0.7f,
                         0.3f, 0.4f + 0.13f * l + 0.05f * e);
        }
        for (uint32_t b = 0; b < kBodiesPerEnv; ++b) {
            body_pose[e * kBodiesPerEnv + b] =
                MakePose(-0.4f * b - s, 0.6f * b, 0.15f * e, 0.9f, 0.1f, 0.4f,
                         -0.3f + 0.21f * b + 0.07f * e);
        }
        base_pose[e] = MakePose(2.0f + e, 0.5f * e, -0.7f, 0.0f, 0.0f, 1.0f,
                                0.25f + 0.11f * e);
    }

    Transform* d_link = nullptr;
    Transform* d_body = nullptr;
    Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_link, link_pose.size() * sizeof(Transform)));
    NK_CUDA_OK(cudaMalloc(&d_body, body_pose.size() * sizeof(Transform)));
    NK_CUDA_OK(cudaMalloc(&d_base, base_pose.size() * sizeof(Transform)));
    NK_CUDA_OK(cudaMemcpy(d_link, link_pose.data(),
                          link_pose.size() * sizeof(Transform), cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_body, body_pose.data(),
                          body_pose.size() * sizeof(Transform), cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_base, base_pose.data(),
                          base_pose.size() * sizeof(Transform), cudaMemcpyHostToDevice));

    // --- Static per-instance tables (env-invariant): rows + blas/material id. --
    // Kinds: 1=Link, 2=Body, 3=Base, 0=Static. cvl is a distinct local offset.
    std::vector<InstanceScatterRow> rows;
    std::vector<uint32_t> blas_id;
    std::vector<uint32_t> material_id;
    auto add = [&](uint32_t kind, uint32_t row, uint32_t bid, uint32_t mid,
                   const Transform& cvl) {
        InstanceScatterRow r;
        r.kind = kind;
        r.row = row;
        SetCvl(r, cvl);
        rows.push_back(r);
        blas_id.push_back(bid);
        material_id.push_back(mid);
    };
    add(1u, 0u, 0u, 7u, MakePose(0.05f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.1f));   // Link 0
    add(1u, 4u, 1u, 3u, MakePose(0.0f, 0.12f, -0.03f, 0.0f, 1.0f, 0.0f, 0.2f)); // Link 4
    add(2u, 0u, 0u, 5u, MakePose(-0.07f, 0.0f, 0.09f, 0.3f, 0.3f, 0.9f, 0.3f)); // Body 0
    add(2u, 2u, 2u, 9u, MakePose(0.0f, -0.04f, 0.0f, 1.0f, 1.0f, 0.0f, 0.4f));  // Body 2
    add(3u, 0u, 1u, 1u, MakePose(0.11f, 0.02f, 0.0f, 0.0f, 0.0f, 1.0f, 0.5f));  // Base
    add(0u, 0u, 0u, 2u, MakePose(1.3f, -0.5f, 0.2f, 0.5f, 0.5f, 0.7f, 0.6f));   // Static
    const uint32_t kInst = static_cast<uint32_t>(rows.size());

    // --- BLAS refs (one per unique mesh), indexed by blas_id. Distinct bounds +
    // sentinel pointer fields so the DevInstance memcmp covers the pointer copy.
    std::vector<SensorBlasRef> blas_refs(3);
    for (uint32_t m = 0; m < blas_refs.size(); ++m) {
        SensorBlasRef& r = blas_refs[m];
        r.blas_leaf_count = m + 1u;  // 1,2,3 (all > 0 -> exercise the box arith)
        r.blas_nodes = reinterpret_cast<const nuka::collision::gpu::LbvhNode*>(
            static_cast<uintptr_t>(0x1000u + m));
        r.blas.prims = reinterpret_cast<const nuka::rt::DevPrim*>(
            static_cast<uintptr_t>(0x2000u + m));
        r.blas.prim_count = m + 1u;
        const float h = 0.2f + 0.1f * m;
        r.local_bound.min = Vec3{-h - 0.05f * m, -0.15f, -0.1f * m - 0.02f};
        r.local_bound.max = Vec3{h, 0.25f + 0.05f * m, 0.1f * m + 0.03f};
    }

    InstanceScatterRow* d_rows = nullptr;
    uint32_t* d_blas_id = nullptr;
    uint32_t* d_material_id = nullptr;
    SensorBlasRef* d_blas_refs = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_rows, kInst * sizeof(InstanceScatterRow)));
    NK_CUDA_OK(cudaMalloc(&d_blas_id, kInst * sizeof(uint32_t)));
    NK_CUDA_OK(cudaMalloc(&d_material_id, kInst * sizeof(uint32_t)));
    NK_CUDA_OK(cudaMalloc(&d_blas_refs, blas_refs.size() * sizeof(SensorBlasRef)));
    NK_CUDA_OK(cudaMemcpy(d_rows, rows.data(), kInst * sizeof(InstanceScatterRow),
                          cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_blas_id, blas_id.data(), kInst * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_material_id, material_id.data(), kInst * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_blas_refs, blas_refs.data(),
                          blas_refs.size() * sizeof(SensorBlasRef),
                          cudaMemcpyHostToDevice));

    // --- Output buffers (zeroed so any padding is a deterministic 0). ----------
    const size_t total = static_cast<size_t>(kEnvCount) * kInst;
    DevInstance* d_inst = nullptr;
    AABB* d_aabb = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_inst, total * sizeof(DevInstance)));
    NK_CUDA_OK(cudaMalloc(&d_aabb, total * sizeof(AABB)));
    NK_CUDA_OK(cudaMemset(d_inst, 0, total * sizeof(DevInstance)));
    NK_CUDA_OK(cudaMemset(d_aabb, 0, total * sizeof(AABB)));

    ScatterFkSource fk;
    fk.link_pose = d_link;
    fk.body_pose = d_body;
    fk.base_pose = d_base;
    fk.env_index = 0u;  // ignored by the batched path (env from the global index)
    fk.links_per_env = kLinksPerEnv;
    fk.bodies_per_env = kBodiesPerEnv;

    nuka::rt::ScatterEnvInstances(/*stream=*/nullptr, fk, d_rows, d_blas_id,
                                  d_material_id, d_blas_refs, kEnvCount, kInst, d_inst,
                                  d_aabb);
    NK_CUDA_OK(cudaDeviceSynchronize());

    std::vector<DevInstance> got_inst(total);
    std::vector<AABB> got_aabb(total);
    NK_CUDA_OK(cudaMemcpy(got_inst.data(), d_inst, total * sizeof(DevInstance),
                          cudaMemcpyDeviceToHost));
    NK_CUDA_OK(cudaMemcpy(got_aabb.data(), d_aabb, total * sizeof(AABB),
                          cudaMemcpyDeviceToHost));

    // --- Host reference: the SAME fk*cvl compose + shared MakeDevInstance /
    // InstanceWorldAabb (zeroed buffers so padding matches the device's). --------
    std::vector<DevInstance> ref_inst(total);
    std::vector<AABB> ref_aabb(total);
    std::memset(ref_inst.data(), 0, total * sizeof(DevInstance));
    std::memset(ref_aabb.data(), 0, total * sizeof(AABB));
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        for (uint32_t local = 0; local < kInst; ++local) {
            const InstanceScatterRow& row = rows[local];
            const SensorBlasRef& ref = blas_refs[blas_id[local]];
            Transform cvl;
            cvl.position = Vec3{row.cached_visual_local[0], row.cached_visual_local[1],
                                row.cached_visual_local[2]};
            cvl.rotation = Quat{row.cached_visual_local[3], row.cached_visual_local[4],
                                row.cached_visual_local[5], row.cached_visual_local[6]};
            Transform world = cvl;  // Static / unresolved -> fk = identity
            if (row.kind == 1u && row.row < kLinksPerEnv) {
                world = link_pose[e * kLinksPerEnv + row.row] * cvl;
            } else if (row.kind == 2u && row.row < kBodiesPerEnv) {
                world = body_pose[e * kBodiesPerEnv + row.row] * cvl;
            } else if (row.kind == 3u) {
                world = base_pose[e] * cvl;
            }
            const size_t i = static_cast<size_t>(e) * kInst + local;
            ref_inst[i] = nuka::rt::MakeDevInstance(world, ref.blas_nodes,
                                                    ref.blas_leaf_count, ref.blas,
                                                    static_cast<uint32_t>(i),
                                                    material_id[local]);
            ref_aabb[i] = nuka::rt::InstanceWorldAabb(ref.blas_leaf_count,
                                                      ref.local_bound, world);
        }
    }

    EXPECT_EQ(std::memcmp(got_inst.data(), ref_inst.data(),
                          total * sizeof(DevInstance)),
              0)
        << "device DevInstance table != host fk*cvl reference (byte-diff)";
    EXPECT_EQ(std::memcmp(got_aabb.data(), ref_aabb.data(), total * sizeof(AABB)), 0)
        << "device world-AABB table != host reference (byte-diff)";

    cudaFree(d_link);
    cudaFree(d_body);
    cudaFree(d_base);
    cudaFree(d_rows);
    cudaFree(d_blas_id);
    cudaFree(d_material_id);
    cudaFree(d_blas_refs);
    cudaFree(d_inst);
    cudaFree(d_aabb);
}
