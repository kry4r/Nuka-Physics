// ---------------------------------------------------------------------------
// nuka::scene::BuildCompiledScene implementation
// ---------------------------------------------------------------------------

#include "scene/scene_pipeline.hpp"

#include "scene/cooker.hpp"

#include <algorithm>
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

PoseGraph BuildPoseGraph(const SceneIR& scene) {
    PoseGraph graph;
    for (const auto& body : scene.Bodies()) {
        PoseGraphNode node;
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
    compiled.graph = BuildPoseGraph(scene);

    const auto cooked = CookScene(scene);
    compiled.physics = runtime::BuildPhysicsWorld(cooked);
    compiled.render = render::BuildRenderScene(scene);
    return compiled;
}

void ApplyRuntimeStateToCompiledScene(const runtime::BuiltWorldState& instance,
                                      CompiledScene& compiled) {
    const size_t body_count = std::min<size_t>({
        instance.body_count,
        instance.poses.size(),
        compiled.graph.NodeCount()
    });

    for (size_t i = 0; i < body_count; ++i) {
        auto& node = compiled.graph.GetNodeMut(static_cast<BodyId>(i));
        node.world_transform = instance.poses[i];
    }

    for (auto& mesh : compiled.render.mesh_instances) {
        if (mesh.body_id >= body_count) {
            continue;
        }

        mesh.world_transform = instance.poses[mesh.body_id] * mesh.local_transform;
    }

    for (auto& proxy : compiled.render.debug_proxies) {
        if (proxy.body_id >= body_count) {
            continue;
        }

        proxy.world_transform = instance.poses[proxy.body_id] * proxy.local_transform;
    }

    for (auto& camera : compiled.render.cameras) {
        if (camera.attached_body == kInvalidBody ||
            camera.attached_body >= body_count) {
            camera.world_transform = camera.local_transform;
            continue;
        }
        camera.world_transform = instance.poses[camera.attached_body] *
            camera.local_transform;
    }

    for (auto& light : compiled.render.lights) {
        if (light.attached_body == kInvalidBody ||
            light.attached_body >= body_count) {
            light.world_transform = light.local_transform;
            continue;
        }
        light.world_transform = instance.poses[light.attached_body] *
            light.local_transform;
    }
}

} // namespace nuka::scene
