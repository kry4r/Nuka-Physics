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

#include <memory>

namespace nuka::render {

// The opaque sensor scene handle owns the persistent batched scene device state
// (BLAS once + the env-shared binding + per-env TLAS scratch + the device AOV
// tensor + the mount table + the device camera buffer) plus the last shape.
struct SensorSceneHandle {
    rt::BatchedSensorSceneDevice scene;
    SensorAovShape shape;
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

        auto handle = std::make_unique<SensorSceneHandle>();
        handle->scene = rt::BuildBatchedSensorScene(cuda_desc, backend_);
        rt::SetSensorMounts(handle->scene, desc.sensors);
        return handle.release();
    }

    void RenderSensors(SensorSceneHandle* handle,
                       const phi::ScatterFkSource& fk,
                       uint32_t env_count,
                       uint32_t width,
                       uint32_t height) override {
        rt::RenderSensorsMounted(handle->scene, fk, env_count, width, height, backend_);
        handle->shape.env_count = env_count;
        handle->shape.sensors_per_env = rt::SensorsPerEnv(handle->scene);
        handle->shape.height = height;
        handle->shape.width = width;
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
