#include "import/cooker/mesh_query_backend.hpp"

#include "collision/mesh_surface.hpp"

namespace nuka::import::cooker {

#if defined(NUKA_COOKER_HAS_CUDA)
bool CudaMeshQueryAvailable();
std::unique_ptr<MeshQueryBackend> CreateCudaMeshQueryBackend(
    collision::MeshSurfaceView source, collision::MeshSurfaceInfo info);
#endif

namespace {

class CpuMeshQueryBackend final : public MeshQueryBackend {
public:
    CpuMeshQueryBackend(collision::MeshSurfaceView source, collision::MeshSurfaceInfo info)
        : source_(source), info_(info) {}

    const char* Name() const override { return "cpu"; }

    void Query(const std::vector<math::Vec3>& points,
               std::vector<collision::MeshSurfacePoint>& results) override {
        results.resize(points.size());
        for (size_t i = 0; i < points.size(); ++i)
            results[i] = collision::QueryMeshSurface(source_, info_, points[i]);
    }

private:
    collision::MeshSurfaceView source_;
    collision::MeshSurfaceInfo info_;
};

}  // namespace

const char* MeshQueryBackendName(bool allow_device) {
#if defined(NUKA_COOKER_HAS_CUDA)
    if (allow_device && CudaMeshQueryAvailable()) return "cuda";
#else
    (void)allow_device;
#endif
    return "cpu";
}

std::unique_ptr<MeshQueryBackend> CreateMeshQueryBackend(
    collision::MeshSurfaceView source, collision::MeshSurfaceInfo info, bool allow_device) {
#if defined(NUKA_COOKER_HAS_CUDA)
    if (allow_device) {
        auto device = CreateCudaMeshQueryBackend(source, info);
        if (device) return device;
    }
#else
    (void)allow_device;
#endif
    return std::make_unique<CpuMeshQueryBackend>(source, info);
}

}  // namespace nuka::import::cooker
