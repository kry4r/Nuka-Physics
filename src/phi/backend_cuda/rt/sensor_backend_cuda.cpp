// ---------------------------------------------------------------------------
// nuka::render::CudaSensorBackend -- the CONCRETE CUDA implementation of the
// backend-agnostic render::SensorBackendI (the batched device-resident sensor
// render facet, sibling of the RtBackendI single-camera facet).
//
// This is the ONLY place outside the rt device TUs that reaches the batched sensor
// entry points (rt::BuildBatchedSensorScene / rt::SetSensorMounts /
// rt::RenderSensorsMounted + the device-ptr accessors). It WRAPS them -- it does
// NOT reimplement the batched render; this TU adds NO arithmetic. Like
// rt_backend_cuda.cpp it is HOST C++ (no kernel launch here): every device token
// (streams, the device camera buffer, the mount upload) lives behind those host
// entry points, in nuka_phi2_rt.
//
// Lives in src/phi/backend_cuda/rt/ (the CUDA backend layer); the redline render
// layer (src/render) only sees the interface (render/sensor_backend.hpp). Compiled
// into nuka_phi2_rt so linking that lib provides the strong CreateCudaSensorBackend
// definition (the RHI/render libs name only the interface).
// ---------------------------------------------------------------------------

#include "render/sensor_backend.hpp"

#include "phi/backend.hpp"  // ActiveBackend / InitBestDevice
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "scene/scene_ir.hpp"  // scene::SensorDesc / SensorType (split cam vs lidar)

#include <memory>
#include <vector>

namespace nuka::render {

// The opaque sensor scene handle owns the persistent batched scene device state
// (BLAS once + the env-shared binding + per-env TLAS scratch + the device AOV +
// range tensors + the camera/lidar mount tables) plus the last shapes. ONE scene/
// TLAS serves both cameras and lidars; the two mount tables are independent.
struct SensorSceneHandle {
    rt::BatchedSensorSceneDevice scene;
    SensorAovShape shape;
    SensorRangeShape range_shape;
    bool has_cameras = false;
    bool has_lidars = false;
};

namespace {

// The CUDA sensor backend: holds the selected phi backend (its device + main
// stream) and wraps the batched sensor render, which runs on that device/stream.
class CudaSensorBackend final : public SensorBackendI {
public:
    explicit CudaSensorBackend(phi::Backend* backend) : backend_(backend) {}

    SensorSceneHandle* BuildSensorScene(const SensorSceneDesc& desc) override {
        rt::BatchedSensorSceneDesc cuda_desc;
        cuda_desc.scene = desc.scene;
        cuda_desc.rows = desc.rows;
        cuda_desc.blas_id = desc.blas_id;
        cuda_desc.material_id = desc.material_id;

        // Split the type-tagged mount list: cameras and lidars ride the SAME scene/
        // TLAS but their own mount tables (one range trace, one AOV trace).
        std::vector<scene::SensorDesc> cams, lidars;
        for (const scene::SensorDesc& s : desc.sensors) {
            if (s.type == scene::SensorType::Lidar ||
                s.type == scene::SensorType::RangeScan) {
                lidars.push_back(s);
            } else {
                cams.push_back(s);
            }
        }

        auto handle = std::make_unique<SensorSceneHandle>();
        handle->scene = rt::BuildBatchedSensorScene(cuda_desc, backend_);
        if (!cams.empty()) {
            rt::SetSensorMounts(handle->scene, cams);
            handle->has_cameras = true;
        }
        if (!lidars.empty()) {
            rt::SetLidarMounts(handle->scene, lidars);
            handle->has_lidars = true;
        }
        return handle.release();
    }

    void RenderSensors(SensorSceneHandle* handle,
                       const phi::ScatterFkSource& fk,
                       uint32_t env_count,
                       uint32_t width,
                       uint32_t height) override {
        if (!handle->has_cameras) {
            return;  // a lidar-only handle has no AOV tensor to fill.
        }
        rt::RenderSensorsMounted(handle->scene, fk, env_count, width, height, backend_);
        handle->shape.env_count = env_count;
        handle->shape.sensors_per_env = rt::SensorsPerEnv(handle->scene);
        handle->shape.height = height;
        handle->shape.width = width;
    }

    void RenderLidars(SensorSceneHandle* handle, const phi::ScatterFkSource& fk,
                      uint32_t env_count) override {
        if (!handle->has_lidars) {
            return;  // a camera-only handle has no range tensor to fill.
        }
        rt::RenderLidarsMounted(handle->scene, fk, env_count, backend_);
        handle->range_shape.env_count = env_count;
        handle->range_shape.sensors_per_env = rt::LidarsPerEnv(handle->scene);
        handle->range_shape.az_count = rt::LidarAzCount(handle->scene);
        handle->range_shape.el_count = rt::LidarElCount(handle->scene);
    }

    const float* SensorRangeDevice(const SensorSceneHandle* handle) const override {
        return rt::SensorRangeDevice(handle->scene);
    }

    SensorRangeShape RangeShape(const SensorSceneHandle* handle) const override {
        return handle->range_shape;
    }

    const float* SensorColorDevice(const SensorSceneHandle* handle) const override {
        return rt::SensorColorDevice(handle->scene);
    }
    const float* SensorDepthDevice(const SensorSceneHandle* handle) const override {
        return rt::SensorDepthDevice(handle->scene);
    }
    const float* SensorNormalDevice(const SensorSceneHandle* handle) const override {
        return rt::SensorNormalDevice(handle->scene);
    }
    const float* SensorAlbedoDevice(const SensorSceneHandle* handle) const override {
        return rt::SensorAlbedoDevice(handle->scene);
    }
    const uint32_t* SensorPrimDevice(const SensorSceneHandle* handle) const override {
        return rt::SensorPrimDevice(handle->scene);
    }

    SensorAovShape AovShape(const SensorSceneHandle* handle) const override {
        return handle->shape;
    }

    void SetRenderDr(SensorSceneHandle* handle, const rt::RenderDrConfig& cfg,
                     uint32_t env_count) override {
        rt::SetRenderDr(handle->scene, cfg, env_count, backend_);
    }

    void SetSensorFidelity(SensorSceneHandle* handle,
                           const rt::SensorFidelityConfig& cfg) override {
        rt::SetSensorFidelity(handle->scene, cfg);
    }

    void FreeSensorScene(SensorSceneHandle* handle) override { delete handle; }

private:
    phi::Backend* backend_ = nullptr;
};

}  // namespace

// Bind to the global active architecture so render shares the device + stream
// physics uses. Null when no device is available (phi "unavailable -> null").
std::unique_ptr<SensorBackendI> CreateCudaSensorBackend() {
    phi::Backend* backend = phi::ActiveBackend();
    if (backend == nullptr) {
        return nullptr;
    }
    return std::make_unique<CudaSensorBackend>(backend);
}

bool SensorBackendAvailable() { return phi::InitBestDevice() != nullptr; }

}  // namespace nuka::render
