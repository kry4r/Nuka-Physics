// ---------------------------------------------------------------------------
// nuka::scene::BuildCompiledScene implementation
// ---------------------------------------------------------------------------

#include "scene/scene_pipeline.hpp"

#include "scene/cooker.hpp"

#include <utility>

namespace nuka::scene {

namespace {

math::Transform ResolveWorldTransform(const SceneIR& scene, BodyId body_id) {
    const auto& body = scene.GetBody(body_id);
    if (body.parent_id == kInvalidBody) {
        return body.local_transform;
    }
    return ResolveWorldTransform(scene, body.parent_id) * body.local_transform;
}

} // namespace

SceneGraph BuildSceneGraph(const SceneIR& scene) {
    SceneGraph graph;
    for (const auto& body : scene.Bodies()) {
        SceneGraphNode node;
        node.body_id = body.id;
        node.parent = body.parent_id;
        node.name = body.name;
        node.local_transform = body.local_transform;
        node.world_transform = ResolveWorldTransform(scene, body.id);
        graph.AddNode(std::move(node));
    }
    return graph;
}

CompiledScene BuildCompiledScene(const SceneIR& scene) {
    CompiledScene compiled;
    compiled.graph = BuildSceneGraph(scene);

    const auto cooked = CookScene(scene);
    compiled.physics = runtime::BuildPhysicsWorld(cooked);
    compiled.render = render::BuildRenderScene(scene);
    return compiled;
}

} // namespace nuka::scene
