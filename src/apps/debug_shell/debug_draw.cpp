// ---------------------------------------------------------------------------
// nuka::app::DebugDrawList -- Implementation
// ---------------------------------------------------------------------------

#include "apps/debug_shell/debug_draw.hpp"

namespace nuka::app {

void DebugDrawList::AddLine(math::Vec3 from, math::Vec3 to, uint32_t color) {
    DrawCommand cmd{};
    cmd.type     = DrawCommandType::Line;
    cmd.position = from;
    cmd.end      = to;
    cmd.color    = color;
    commands_.push_back(cmd);
}

void DebugDrawList::AddSphere(math::Vec3 center, float radius, uint32_t color) {
    DrawCommand cmd{};
    cmd.type     = DrawCommandType::Sphere;
    cmd.position = center;
    cmd.radius   = radius;
    cmd.color    = color;
    commands_.push_back(cmd);
}

void DebugDrawList::AddBox(math::Vec3 center, math::Vec3 half_extents, uint32_t color) {
    DrawCommand cmd{};
    cmd.type     = DrawCommandType::Box;
    cmd.position = center;
    cmd.size     = half_extents;
    cmd.color    = color;
    commands_.push_back(cmd);
}

void DebugDrawList::AddAABB(const collision::AABB& aabb, uint32_t color) {
    DrawCommand cmd{};
    cmd.type     = DrawCommandType::AABB;
    cmd.position = aabb.min;
    cmd.end      = aabb.max;
    cmd.color    = color;
    commands_.push_back(cmd);
}

void DebugDrawList::AddFrame(math::Transform transform, float size) {
    // Three lines for the X, Y, Z axes
    const math::Vec3 origin = transform.position;
    const math::Vec3 x_end  = origin + transform.TransformDirection(math::Vec3::UnitX()) * size;
    const math::Vec3 y_end  = origin + transform.TransformDirection(math::Vec3::UnitY()) * size;
    const math::Vec3 z_end  = origin + transform.TransformDirection(math::Vec3::UnitZ()) * size;

    AddLine(origin, x_end, 0xFF0000FF); // red   = X
    AddLine(origin, y_end, 0x00FF00FF); // green = Y
    AddLine(origin, z_end, 0x0000FFFF); // blue  = Z
}

void DebugDrawList::AddContactPoint(math::Vec3 position, math::Vec3 normal, float depth) {
    DrawCommand cmd{};
    cmd.type     = DrawCommandType::ContactPoint;
    cmd.position = position;
    cmd.end      = normal;
    cmd.radius   = depth;
    cmd.color    = 0xFF00FFFF; // magenta
    commands_.push_back(cmd);
}

const std::vector<DrawCommand>& DebugDrawList::Commands() const {
    return commands_;
}

size_t DebugDrawList::CommandCount() const {
    return commands_.size();
}

void DebugDrawList::Clear() {
    commands_.clear();
}

uint32_t BuildContactOverlayCommandCount(uint32_t contacts) {
    // One draw command per contact point
    return contacts;
}

} // namespace nuka::app
