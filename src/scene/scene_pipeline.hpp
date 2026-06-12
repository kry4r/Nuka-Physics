#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::BuildCompiledScene - unified scene conversion pipeline
// ---------------------------------------------------------------------------

#include "render/render_scene.hpp"
#include "runtime/physics_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/pose_graph.hpp"
#include "scene/scene_ir.hpp"

namespace nuka::scene {

struct CompiledScene {
    PoseGraph graph;
    runtime::PhysicsWorld physics;
    render::RenderScene render;
};

PoseGraph BuildPoseGraph(const SceneIR& scene);
CompiledScene BuildCompiledScene(const SceneIR& scene);

void ApplyRuntimeStateToCompiledScene(const runtime::BuiltWorldState& instance,
                                      CompiledScene& compiled);

} // namespace nuka::scene
