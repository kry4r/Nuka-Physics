#pragma once

#include "render/render_world.hpp"

#include <cstdint>

namespace nuka::runtime::app::viewer {

inline constexpr uint32_t kMaxDebugOverlayInstances = 8192u;

struct DebugDrawReport {
    uint32_t colliders = 0u;
    uint32_t contacts = 0u;
    uint32_t unsupported_shapes = 0u;
    uint32_t invalid_colliders = 0u;
    uint32_t invalid_contacts = 0u;
    uint64_t omitted_instances = 0u;
};

bool IsValidDebugPose(const math::Transform& pose);

// Builds only the viewer's debug channel from already-read host values.
class DebugDrawBatch {
public:
    void Reset() { report_ = {}; }
    const DebugDrawReport& Report() const { return report_; }
    uint32_t Remaining() const { return kMaxDebugOverlayInstances - report_.colliders; }
    void RejectCollider() { ++report_.invalid_colliders; }
    void AppendCollider(render::RenderWorld& world, uint32_t kind, const float* params,
                        const math::Transform& pose, uint32_t material);
    // Contact points draw over the viewport rather than as instances; this only counts them.
    void CountContact(const math::Vec3& point);

private:
    bool Reserve();
    DebugDrawReport report_;
};

}  // namespace nuka::runtime::app::viewer
