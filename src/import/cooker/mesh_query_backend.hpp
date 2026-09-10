#pragma once

#include <memory>
#include <vector>

#include "collision/mesh_surface_types.hpp"

namespace nuka::import::cooker {

class MeshQueryBackend {
public:
    virtual ~MeshQueryBackend() = default;
    virtual const char* Name() const = 0;
    virtual void Query(const std::vector<math::Vec3>& points,
                       std::vector<collision::MeshSurfacePoint>& results) = 0;
};

const char* MeshQueryBackendName(bool allow_device = true);

// Source storage remains valid for the lifetime of the returned query object.
std::unique_ptr<MeshQueryBackend> CreateMeshQueryBackend(
    collision::MeshSurfaceView source, collision::MeshSurfaceInfo info,
    bool allow_device = true);

}  // namespace nuka::import::cooker
