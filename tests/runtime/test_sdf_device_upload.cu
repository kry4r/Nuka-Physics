// ---------------------------------------------------------------------------
// p08-A: GPU upload of cooked sparse SDFs -- device sampler smoke test.
// ---------------------------------------------------------------------------
// Cooks known meshes' narrow-band SDFs on the host (p07 CookSparseSdf), packs
// them into a multi-SDF scene::CookedSdfTable (so the per-SDF key_offsets/
// key_counts slicing is exercised), uploads via SdfDeviceWorld, then launches a
// __global__ that calls the SHIPPING sampler (p07 sparse_sdf_sample) through the
// uploaded device SparseSdfDevice views. The GPU phi/grad are compared against
// the host oracle (MakeHostView over the same cooked SparseSdfData) at the same
// query points:
//   * inside-band  : phi < 0, finite, matches host
//   * outside-band : phi > 0, finite, matches host
//   * out-of-band  : phi == kOutsideBand (exact constant, no arithmetic)
//   * gradient      : matches host near-equal
//
// Two equality regimes, intentionally distinct:
//   GPU-vs-host : same float code but g++ vs nvcc may differ by an ULP (FMA
//                 contraction) -> assert near-equal (~1e-5 rel/abs). The
//                 out-of-band sentinel is a bare constant -> assert EXACT there.
//   D1 two-run  : device-vs-device, same binary -> assert BYTE-identical (memcmp
//                 of the raw downloaded result blob).
// ---------------------------------------------------------------------------

#include "runtime/sdf/sdf_device_world.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "scene/cooked_blob.hpp"

#include "tests/import/sdf_test_meshes.hpp"
#include "tests/import/sdf_host_sampler.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfData;
using nuka::import::cooker::SparseSdfParams;
using nuka::math::Vec3;
using nuka::runtime::sdf::SparseSdfDevice;
using nuka::runtime::sdf::SdfDeviceWorld;
using nuka::runtime::sdf::UploadSdfDeviceWorld;

// One GPU sample result, packed so we can both float-compare it to the host and
// memcmp it for the D1 byte-exact check.
struct SampleResult {
    float phi;
    float gx;
    float gy;
    float gz;
};

// Sample each query point against the SDF selected by piece_sdf_indices[piece]:
// q.piece picks the convex piece -> its SDF index -> the device view -> sampler.
struct Query {
    Vec3 p;
    uint32_t piece;
};

__global__ void SampleSdfKernel(const SparseSdfDevice* views,
                                const uint32_t* piece_sdf_indices,
                                const Query* queries,
                                SampleResult* out,
                                uint32_t query_count) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_count) {
        return;
    }
    const Query q = queries[tid];
    const uint32_t sdf_index = piece_sdf_indices[q.piece];
    SampleResult r;
    if (sdf_index == nuka::scene::kNoSdf) {
        r.phi = SparseSdfDevice::kOutsideBand;
        r.gx = r.gy = r.gz = 0.0f;
        out[tid] = r;
        return;
    }
    Vec3 grad;
    const float phi =
        nuka::runtime::sdf::sparse_sdf_sample(views[sdf_index], q.p, grad);
    r.phi = phi;
    r.gx = grad.x;
    r.gy = grad.y;
    r.gz = grad.z;
    out[tid] = r;
}

// Replicates scene::cooker::AppendSdf: append a cooked SparseSdfData into a flat
// multi-SDF CookedSdfTable, returning its sdf index. (Inlined so the test does
// not need the scene cooker's internal linkage.)
uint32_t AppendSdf(nuka::scene::CookedSdfTable& table, const SparseSdfData& sdf) {
    const uint32_t index = table.Count();
    table.origins.push_back({sdf.origin[0], sdf.origin[1], sdf.origin[2]});
    table.voxel_sizes.push_back(sdf.voxel_size);
    table.dims_x.push_back(sdf.dims[0]);
    table.dims_y.push_back(sdf.dims[1]);
    table.dims_z.push_back(sdf.dims[2]);
    table.key_offsets.push_back(static_cast<uint32_t>(table.cell_keys.size()));
    table.key_counts.push_back(sdf.CellCount());
    table.cell_keys.insert(table.cell_keys.end(), sdf.cell_keys.begin(),
                           sdf.cell_keys.end());
    table.cell_values.insert(table.cell_values.end(), sdf.cell_values.begin(),
                             sdf.cell_values.end());
    for (uint32_t n = 0; n < sdf.CellCount(); ++n) {
        table.cell_gradients.push_back({sdf.cell_gradients[3 * n + 0],
                                        sdf.cell_gradients[3 * n + 1],
                                        sdf.cell_gradients[3 * n + 2]});
    }
    return index;
}

SparseSdfParams TestParams(float voxel) {
    SparseSdfParams p;
    p.voxel_size = voxel;
    p.band_voxels = 2u;
    return p;
}

SparseSdfData CookMesh(const nuka::test::TestMesh& mesh, float voxel) {
    return CookSparseSdf(mesh.vertices.data(),
                         static_cast<uint32_t>(mesh.vertices.size() / 3),
                         mesh.indices.data(),
                         static_cast<uint32_t>(mesh.indices.size() / 3),
                         TestParams(voxel));
}

// Launch the device sampler over `queries` against `world`; return one
// SampleResult per query (downloaded to host).
std::vector<SampleResult> RunDeviceSamples(const SdfDeviceWorld& world,
                                           const std::vector<Query>& queries) {
    const uint32_t n = static_cast<uint32_t>(queries.size());
    nuka::phi::Buffer d_queries = nuka::phi::UploadVector(queries);
    nuka::phi::Buffer d_out(n * sizeof(SampleResult), nuka::phi::MemoryKind::Device);

    const uint32_t block = 64u;
    const uint32_t grid = (n + block - 1u) / block;
    SampleSdfKernel<<<grid, block>>>(
        world.DeviceViews(),
        world.DevicePieceSdfIndices(),
        static_cast<const Query*>(d_queries.Data()),
        static_cast<SampleResult*>(d_out.Data()),
        n);
    const cudaError_t launch_err = cudaGetLastError();
    EXPECT_EQ(launch_err, cudaSuccess) << cudaGetErrorString(launch_err);
    const cudaError_t sync_err = cudaDeviceSynchronize();
    EXPECT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

    return nuka::phi::DownloadVector<SampleResult>(d_out, n);
}

// Build a 2-SDF table: SDF 0 = unit sphere, SDF 1 = unit cube. Three pieces map
// to {sphere, cube, kNoSdf} so the piece->sdf mapping (incl. the no-sdf piece)
// is exercised on the device.
struct TwoSdfFixture {
    nuka::scene::CookedSdfTable table;
    SparseSdfData sphere;
    SparseSdfData cube;
    nuka::test::HostSdfView sphere_view;
    nuka::test::HostSdfView cube_view;
    static constexpr float kVoxel = 0.06f;
};

TwoSdfFixture MakeTwoSdfFixture() {
    TwoSdfFixture f;
    const auto sphere_mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 48, 96);
    const auto cube_mesh = nuka::test::UnitCubeMesh();  // [-0.5, 0.5]^3
    f.sphere = CookMesh(sphere_mesh, TwoSdfFixture::kVoxel);
    f.cube = CookMesh(cube_mesh, 0.04f);
    const uint32_t sphere_idx = AppendSdf(f.table, f.sphere);
    const uint32_t cube_idx = AppendSdf(f.table, f.cube);
    // pieces: 0->sphere, 1->cube, 2->no sdf
    f.table.piece_sdf_indices = {sphere_idx, cube_idx, nuka::scene::kNoSdf};
    f.sphere_view = nuka::test::MakeHostView(f.sphere);
    f.cube_view = nuka::test::MakeHostView(f.cube);
    return f;
}

TEST(SdfDeviceUpload, GpuSamplesMatchHostSampler) {
    auto f = MakeTwoSdfFixture();
    const float v = TwoSdfFixture::kVoxel;

    // Build query points + record the host oracle result for each.
    std::vector<Query> queries;
    std::vector<SampleResult> host_expected;
    std::vector<const char*> labels;

    auto add = [&](const nuka::test::HostSdfView& view, uint32_t piece, Vec3 p,
                   const char* label) {
        Vec3 g;
        const float phi = nuka::test::SampleHost(view, p.x, p.y, p.z, g);
        queries.push_back({p, piece});
        host_expected.push_back({phi, g.x, g.y, g.z});
        labels.push_back(label);
    };

    // --- SDF 0 (sphere, piece 0) ---
    add(f.sphere_view, 0, {1.0f - v, 0.0f, 0.0f}, "sphere inside +X");   // phi<0
    add(f.sphere_view, 0, {1.0f + v, 0.0f, 0.0f}, "sphere outside +X");  // phi>0
    add(f.sphere_view, 0, {0.0f, 1.0f - v, 0.0f}, "sphere inside +Y");
    add(f.sphere_view, 0, {0.0f, 0.0f, 1.0f + v}, "sphere outside +Z");
    add(f.sphere_view, 0, {5.0f, 5.0f, 5.0f}, "sphere far OOB");         // kOutsideBand

    // --- SDF 1 (cube, piece 1) ---
    add(f.cube_view, 1, {0.5f - 0.04f, 0.0f, 0.0f}, "cube inside +X");
    add(f.cube_view, 1, {0.5f + 0.04f, 0.0f, 0.0f}, "cube outside +X");
    add(f.cube_view, 1, {0.0f, 0.0f, 0.5f - 0.04f}, "cube inside +Z");
    add(f.cube_view, 1, {3.0f, 3.0f, 3.0f}, "cube far OOB");

    // --- piece 2 -> no sdf -> always kOutsideBand ---
    queries.push_back({{0.0f, 0.0f, 0.0f}, 2});
    host_expected.push_back({SparseSdfDevice::kOutsideBand, 0.0f, 0.0f, 0.0f});
    labels.push_back("no-sdf piece");

    auto world = UploadSdfDeviceWorld(f.table);
    ASSERT_EQ(world.SdfCount(), 2u);
    ASSERT_EQ(world.PieceCount(), 3u);

    const auto gpu = RunDeviceSamples(world, queries);
    ASSERT_EQ(gpu.size(), host_expected.size());

    double max_phi_err = 0.0;
    double max_grad_err = 0.0;
    bool saw_inside = false, saw_outside = false, saw_oob = false;

    for (size_t i = 0; i < gpu.size(); ++i) {
        const auto& h = host_expected[i];
        const auto& g = gpu[i];
        const bool host_oob = (h.phi >= 1.0e29f);
        if (host_oob) {
            // Sentinel: bare constant, no arithmetic -> exact.
            EXPECT_EQ(g.phi, h.phi) << labels[i];
            saw_oob = true;
            continue;
        }
        if (h.phi < 0.0f) saw_inside = true;
        if (h.phi > 0.0f) saw_outside = true;

        const double pe = std::fabs(static_cast<double>(g.phi) - h.phi);
        const double rel = pe / (std::fabs(static_cast<double>(h.phi)) + 1e-6);
        EXPECT_LT(std::min(pe, rel), 1e-5) << labels[i] << " phi gpu=" << g.phi
                                           << " host=" << h.phi;
        max_phi_err = std::max(max_phi_err, pe);

        const double ge = std::sqrt(
            (static_cast<double>(g.gx) - h.gx) * (static_cast<double>(g.gx) - h.gx) +
            (static_cast<double>(g.gy) - h.gy) * (static_cast<double>(g.gy) - h.gy) +
            (static_cast<double>(g.gz) - h.gz) * (static_cast<double>(g.gz) - h.gz));
        EXPECT_LT(ge, 1e-5) << labels[i] << " gradient";
        max_grad_err = std::max(max_grad_err, ge);
    }

    EXPECT_TRUE(saw_inside) << "expected at least one inside-band (phi<0) query";
    EXPECT_TRUE(saw_outside) << "expected at least one outside-band (phi>0) query";
    EXPECT_TRUE(saw_oob) << "expected at least one out-of-band query";

    std::printf("[sdf-device] queries=%zu  max|phi_gpu-phi_host|=%g  max|grad_gpu-grad_host|=%g\n",
                gpu.size(), max_phi_err, max_grad_err);
}

TEST(SdfDeviceUpload, D1TwoRunByteIdentical) {
    auto f = MakeTwoSdfFixture();
    const float v = TwoSdfFixture::kVoxel;

    std::vector<Query> queries = {
        {{1.0f - v, 0.0f, 0.0f}, 0},
        {{1.0f + v, 0.0f, 0.0f}, 0},
        {{0.0f, 1.0f - v, 0.0f}, 0},
        {{0.5f - 0.04f, 0.0f, 0.0f}, 1},
        {{0.5f + 0.04f, 0.0f, 0.0f}, 1},
        {{0.0f, 0.0f, 0.5f - 0.04f}, 1},
        {{5.0f, 5.0f, 5.0f}, 0},
        {{0.0f, 0.0f, 0.0f}, 2},
    };

    auto world_a = UploadSdfDeviceWorld(f.table);
    const auto run_a = RunDeviceSamples(world_a, queries);

    auto world_b = UploadSdfDeviceWorld(f.table);
    const auto run_b = RunDeviceSamples(world_b, queries);

    ASSERT_EQ(run_a.size(), run_b.size());
    const int cmp = std::memcmp(run_a.data(), run_b.data(),
                                run_a.size() * sizeof(SampleResult));
    EXPECT_EQ(cmp, 0) << "two-run device sample blobs differ (D1 violation)";
}

}  // namespace
