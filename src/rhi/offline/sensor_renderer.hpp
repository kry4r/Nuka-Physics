#pragma once
// ---------------------------------------------------------------------------
// nuka::rhi::offline::SensorRenderer -- the host orchestrator of the BATCHED
// device-resident sensor render, sibling to OfflineRenderer. It is a CLIENT of
// PHI: it binds to the global active architecture (the device + stream physics
// uses) and drives the batched sensor render through render::SensorBackendI,
// exactly as OfflineRenderer drives the beauty trace through render::RtBackendI.
//
// A sibling class (NOT a RenderGraph SensorBatch pass) because the batched sensor
// render is a SINGLE fused op (scatter cameras -> scatter instances -> batched
// build -> ONE trace) producing a PERSISTENT device tensor that lives across
// steps, not a host framebuffer the per-frame PassContext/Execute(frame) graph is
// built for. RenderProfile::Sensor selects this path; OfflineRenderer owns the
// beauty path. ONE general path: the same tracer at a cheap sensor profile, this
// renderer adds zero numeric change over the SensorBackendI entry.
//
// HOST-ONLY / CUDA-FREE: names only render:: / phi:: / scene:: opaque + CUDA-free
// types -- the same zero-CUDA red-line src/render holds. The sensor backend is
// obtained through PHI (CreateCudaSensorBackend); device work lives behind it.
// ---------------------------------------------------------------------------

#include "render/sensor_backend.hpp"   // render::SensorBackendI / SensorSceneHandle / SensorSceneDesc
#include "rhi/offline/render_profile.hpp"

#include <memory>

namespace nuka::rhi::offline {

// The offline batched-sensor facade. One instance per render context; holds the
// sensor backend it vends + the built scene. Build once, render per step, read the
// device AOV tensor.
class SensorRenderer {
public:
    // Bind to the global active architecture and obtain the sensor backend through
    // PHI; usable() is false when no device is available (phi "unavailable -> null").
    SensorRenderer();
    ~SensorRenderer();

    SensorRenderer(const SensorRenderer&) = delete;
    SensorRenderer& operator=(const SensorRenderer&) = delete;

    // True iff a live sensor backend was obtained (a render will produce a tensor).
    bool usable() const { return backend_ != nullptr; }

    // Build the persistent batched sensor scene from `desc` (the env-shared cooked
    // scene + per-instance binding + the SensorDesc mount table) under a Sensor
    // profile; the profile carries the per-sensor image resolution. Returns false
    // when not usable or the profile is not a Sensor profile. Call once.
    bool Build(const render::SensorSceneDesc& desc, const RenderProfile& profile);

    // ONE step: scatter the per-env cameras from the mount table + `fk` (the device
    // LinkPose/BodyPose/BasePose source derived from an nk::World), then ONE batched
    // build/refit + ONE flat trace into the persistent device AOV tensor (no host
    // download). The image resolution + camera count come from the built profile +
    // `env_count`. A no-op when not usable / not built.
    void Render(const phi::ScatterFkSource& fk, uint32_t env_count);

    // Device pointers into the persistent AOV tensor after Render: color/normal/
    // albedo [N*H*W*3], depth/prim [N*H*W]. Null until built + the first Render.
    const float*    ColorDevice() const;
    const float*    DepthDevice() const;
    const float*    NormalDevice() const;
    const float*    AlbedoDevice() const;
    const uint32_t* PrimDevice() const;

    // The (sensors, height, width) shape of the tensor the accessors point into.
    render::SensorAovShape AovShape() const;

private:
    std::unique_ptr<render::SensorBackendI> backend_;
    render::SensorSceneHandle*              handle_ = nullptr;
    uint32_t                                width_  = 0u;
    uint32_t                                height_ = 0u;
};

}  // namespace nuka::rhi::offline
