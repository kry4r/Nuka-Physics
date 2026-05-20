#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::BuildCompiledScene - unified scene conversion pipeline
// ---------------------------------------------------------------------------

#include "render/render_scene.hpp"
#include "runtime/physics_world.hpp"
#include "runtime/world_instance.hpp"
#include "scene/scene_graph.hpp"
#include "scene/scene_ir.hpp"

namespace nuka::scene {

struct CompiledScene {
    SceneGraph graph;
    runtime::PhysicsWorld physics;
    render::RenderScene render;
};

SceneGraph BuildSceneGraph(const SceneIR& scene);
CompiledScene BuildCompiledScene(const SceneIR& scene);

void ApplyRuntimeStateToCompiledScene(const runtime::WorldInstance& instance,
                                      CompiledScene& compiled);

} // namespace nuka::scene
