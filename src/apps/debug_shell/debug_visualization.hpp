#pragma once
// ---------------------------------------------------------------------------
// nuka::app::BuildDebugVisualization -- compiled scene debug overlay builder
// ---------------------------------------------------------------------------

#include "apps/debug_shell/debug_draw.hpp"
#include "constraint/contact_manifold.hpp"
#include "constraint/row_buffers.hpp"
#include "render/render_scene.hpp"
#include "runtime/physics_world.hpp"
#include "scene/scene_graph.hpp"

#include <span>

namespace nuka::app {

struct DebugVisualizationOptions {
    bool draw_collision_shapes = true;
    bool draw_shape_aabbs = true;
    bool draw_joint_axes = true;
    bool draw_centers_of_mass = true;
    bool draw_contact_points = true;
    bool draw_constraint_errors = true;

    float joint_axis_length = 0.25f;
    float center_of_mass_radius = 0.025f;
    float constraint_error_scale = 1.0f;
};

struct DebugVisualizationInput {
    const render::RenderScene* render_scene = nullptr;
    const scene::SceneGraph* scene_graph = nullptr;
    const runtime::PhysicsWorld* physics_world = nullptr;
    std::span<const constraint::ContactManifold> contact_manifolds = {};
    const constraint::RowBuffers* constraint_rows = nullptr;
};

DebugDrawList BuildDebugVisualization(const DebugVisualizationInput& input,
                                      const DebugVisualizationOptions& options = {});

} // namespace nuka::app
