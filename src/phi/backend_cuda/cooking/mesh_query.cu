#include "import/cooker/mesh_query_backend.hpp"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

#include "collision/mesh_surface.hpp"

namespace nuka::import::cooker {
namespace {

void Check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Mesh cook query: ") + cudaGetErrorString(status));
}

template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if (data_) cudaFree(data_); }

    void Reserve(size_t count) {
        if (count <= capacity_) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::length_error("Mesh cook device array size overflow");
        T* next = nullptr;
        Check(cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(T)));
        if (data_) {
            const auto status = cudaFree(data_);
            if (status != cudaSuccess) { cudaFree(next); Check(status); }
        }
        data_ = next;
        capacity_ = count;
    }

    void Upload(const T* input, size_t count) {
        Reserve(count);
        if (count) Check(cudaMemcpy(data_, input, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* Data() const { return data_; }

private:
    T* data_ = nullptr;
    size_t capacity_ = 0;
};

__global__ void QueryMeshPoints(collision::MeshSurfaceView source,
    collision::MeshSurfaceInfo info, const math::Vec3* points, uint32_t count,
    collision::MeshSurfacePoint* results) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) results[i] = collision::QueryMeshSurface(source, info, points[i]);
}

class CudaMeshQueryBackend final : public MeshQueryBackend {
public:
    CudaMeshQueryBackend(collision::MeshSurfaceView source, collision::MeshSurfaceInfo info)
        : info_(info) {
        Check(cudaGetDevice(&device_));
        vertices_.Upload(source.vertices + static_cast<size_t>(info.vertex_offset) * 3u,
                         static_cast<size_t>(info.vertex_count) * 3u);
        triangles_.Upload(source.triangles + static_cast<size_t>(info.triangle_offset) * 3u,
                          static_cast<size_t>(info.triangle_count) * 3u);
        nodes_.Upload(source.nodes + info.node_offset, info.node_count);
        info_.vertex_offset = info_.triangle_offset = info_.node_offset = 0u;
    }

    const char* Name() const override { return "cuda"; }

    void Query(const std::vector<math::Vec3>& points,
               std::vector<collision::MeshSurfacePoint>& results) override {
        int current = 0;
        Check(cudaGetDevice(&current));
        if (current != device_) throw std::runtime_error("Mesh cook device changed during query");
        if (points.size() > std::numeric_limits<uint32_t>::max() - 127u)
            throw std::length_error("Mesh cook query count overflow");
        results.resize(points.size());
        if (points.empty()) return;
        queries_.Upload(points.data(), points.size());
        results_.Reserve(points.size());
        const collision::MeshSurfaceView source{vertices_.Data(), triangles_.Data(), nodes_.Data(),
            {info_.vertex_count, info_.triangle_count, info_.node_count}};
        const auto count = static_cast<uint32_t>(points.size());
        QueryMeshPoints<<<(count + 127u) / 128u, 128u>>>(source, info_, queries_.Data(),
                                                       count, results_.Data());
        Check(cudaGetLastError());
        Check(cudaMemcpy(results.data(), results_.Data(), points.size() * sizeof(results[0]),
                         cudaMemcpyDeviceToHost));
    }

private:
    int device_ = 0;
    collision::MeshSurfaceInfo info_;
    DeviceArray<float> vertices_;
    DeviceArray<uint32_t> triangles_;
    DeviceArray<collision::MeshBvhNode> nodes_;
    DeviceArray<math::Vec3> queries_;
    DeviceArray<collision::MeshSurfacePoint> results_;
};

}  // namespace

bool CudaMeshQueryAvailable() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        return false;
    }
    return true;
}

std::unique_ptr<MeshQueryBackend> CreateCudaMeshQueryBackend(
    collision::MeshSurfaceView source, collision::MeshSurfaceInfo info) {
    if (!CudaMeshQueryAvailable()) return {};
    return std::make_unique<CudaMeshQueryBackend>(source, info);
}

}  // namespace nuka::import::cooker
