#pragma once
// ---------------------------------------------------------------------------
// nuka::app::DebugDrawList -- Testable debug draw command buffer
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/transform.hpp"
#include "collision/aabb.hpp"

#include <cstdint>
#include <vector>

namespace nuka::app {

enum class DrawCommandType : uint8_t {
    Line,
    Sphere,
    Box,
    AABB,
    Frame,
    ContactPoint
};

struct DrawCommand {
    DrawCommandType type;
    math::Vec3 position = {};
    math::Vec3 end      = {};   // for lines
    math::Vec3 size     = {};   // for boxes
    float      radius   = 0.0f;
    uint32_t   color    = 0xFFFFFFFF;
};

class DebugDrawList {
public:
    void AddLine(math::Vec3 from, math::Vec3 to, uint32_t color = 0xFFFFFFFF);
    void AddSphere(math::Vec3 center, float radius, uint32_t color = 0xFFFFFFFF);
    void AddBox(math::Vec3 center, math::Vec3 half_extents, uint32_t color = 0xFFFFFFFF);
    void AddAABB(const collision::AABB& aabb, uint32_t color = 0xFFFFFFFF);
    void AddFrame(math::Transform transform, float size = 0.1f);
    void AddContactPoint(math::Vec3 position, math::Vec3 normal, float depth);

    const std::vector<DrawCommand>& Commands() const;
    size_t CommandCount() const;
    void Clear();

private:
    std::vector<DrawCommand> commands_;
};

/// Return the number of contact overlay draw commands for a given contact count.
uint32_t BuildContactOverlayCommandCount(uint32_t contacts);

} // namespace nuka::app
