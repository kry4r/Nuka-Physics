// ---------------------------------------------------------------------------
// nuka::rhi::offline::SensorRenderer -- implementation. Orchestrates the batched
// device-resident sensor render: a SensorSceneDesc + per-step device poses driven
// on the selected PHI backend through render::SensorBackendI into the persistent
// device AOV tensor. Sibling of OfflineRenderer; this TU adds no arithmetic.
//
// HOST-ONLY / CUDA-FREE: drives the render through render::SensorBackendI; the only
// device work happens inside that backend (the CUDA RT TU).
// ---------------------------------------------------------------------------

#include "rhi/offline/sensor_renderer.hpp"

namespace nuka::rhi::offline {

// The sensor backend is vended through PHI: CreateCudaSensorBackend() binds to the
// global active architecture (the device + stream physics uses) -- the only impl today.
SensorRenderer::SensorRenderer()
    : backend_(render::CreateCudaSensorBackend()) {}

SensorRenderer::~SensorRenderer() {
    if (backend_ != nullptr && handle_ != nullptr) {
        backend_->FreeSensorScene(handle_);
    }
}

bool SensorRenderer::Build(const render::SensorSceneDesc& desc,
                           const RenderProfile& profile) {
    if (backend_ == nullptr || profile.output != RenderOutput::Sensor) {
        return false;
    }
    if (handle_ != nullptr) {
        backend_->FreeSensorScene(handle_);
        handle_ = nullptr;
    }
    handle_ = backend_->BuildSensorScene(desc);
    if (handle_ == nullptr) {
        return false;
    }
    width_ = profile.image_width;
    height_ = profile.image_height;
    return true;
}

void SensorRenderer::Render(const phi::ScatterFkSource& fk, uint32_t env_count) {
    if (backend_ == nullptr || handle_ == nullptr) {
        return;
    }
    backend_->RenderSensors(handle_, fk, env_count, width_, height_);
}

const float* SensorRenderer::ColorDevice() const {
    return handle_ != nullptr ? backend_->SensorColorDevice(handle_) : nullptr;
}
const float* SensorRenderer::DepthDevice() const {
    return handle_ != nullptr ? backend_->SensorDepthDevice(handle_) : nullptr;
}
const float* SensorRenderer::NormalDevice() const {
    return handle_ != nullptr ? backend_->SensorNormalDevice(handle_) : nullptr;
}
const float* SensorRenderer::AlbedoDevice() const {
    return handle_ != nullptr ? backend_->SensorAlbedoDevice(handle_) : nullptr;
}
const uint32_t* SensorRenderer::PrimDevice() const {
    return handle_ != nullptr ? backend_->SensorPrimDevice(handle_) : nullptr;
}

render::SensorAovShape SensorRenderer::AovShape() const {
    return handle_ != nullptr ? backend_->AovShape(handle_) : render::SensorAovShape{};
}

}  // namespace nuka::rhi::offline
