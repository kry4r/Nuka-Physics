#pragma once
// The editor's view of a physics world through the C ABI: the per-environment layout,
// the scene binding the renderer reads, and the pose reads and writes of the physics thread.

#include "nuka/nuka.h"
#include "render/render_world.hpp"
#include "runtime/app/viewer/pose_scatter.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/terrain/heightfield.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::app::viewer {

struct WorldLayout {
    uint32_t env_count = 1u;
    uint32_t links_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t articulations_per_env = 0u;
    uint32_t dofs_per_env = 0u;
    uint32_t contacts_per_env = 0u;
    uint32_t contact_slots_per_env = 0u;
    std::vector<nuka_link_binding_t> links;  // per-environment link order
};

// Throws std::runtime_error when a query fails; `what` names the call.
void CheckResult(nuka_result_t result, const char* what);

WorldLayout QueryLayout(nuka_world_handle world);

// The cooked terrain grid; invalid when the world has none.
terrain::HeightField QueryTerrain(nuka_world_handle world);

// Body records map to their cooked rows (record order) and link bodies to their links,
// the rows BuildRenderWorld resolves instance poses from.
scene::SceneMap BindScene(const scene::SceneIR& scene, const WorldLayout& layout);

// A free body or a floating articulation root: the pose sources a live move reaches.
bool IsMovableSource(const render::PoseSource& source, const WorldLayout& layout);

// A live teleport of one movable instance, as a physics-frame pose of its source row.
struct PoseWrite {
    render::PoseSource source;
    math::Transform pose;
};

// Resolves a visual-frame world pose for a movable entity; false for any other entity.
bool ResolvePoseWrite(const render::RenderWorld& render_world, const WorldLayout& layout,
                      scene::EntityId entity, const math::Transform& world_xform, PoseWrite* out);

// Teleports the row in one environment and zeroes its velocity.
bool ApplyPoseWrite(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                    const PoseWrite& write);

// Uniform PD gains on every non-root link of every environment, so drive targets act
// without a policy.
void ApplyHoldGains(nuka_world_handle world, const WorldLayout& layout, float kp, float kd);

// One environment's link and body poses; a field without rows comes back empty.
void DownloadEnvPoses(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                      std::vector<math::Transform>* links, std::vector<math::Transform>* bodies);

// The zero-copy rows of the render instances, in instance order.
std::vector<ScatterRow> BuildScatterRows(const render::RenderWorld& render_world);

// The UI thread's binding of one world: the layout and the render world it draws.
class EditorSim {
public:
    EditorSim(WorldLayout layout, render::RenderWorld render_world);

    // Re-poses instances, cameras and lights from one environment's poses.
    void ApplyPoses(const std::vector<math::Transform>& links, const std::vector<math::Transform>& bodies);

    // Edits queue live teleports; the frame hands them to the physics thread.
    void QueuePoseWrite(const PoseWrite& write) { pose_writes_.push_back(write); }
    std::vector<PoseWrite> TakePoseWrites();
    // Counts in-place moves of static instances, whose world pose the zero-copy rows carry.
    void MarkStaticMoved() { ++static_revision_; }
    uint64_t StaticRevision() const { return static_revision_; }

    render::RenderWorld& GetRenderWorld() { return render_world_; }
    const render::RenderWorld& GetRenderWorld() const { return render_world_; }
    const WorldLayout& Layout() const { return layout_; }

private:
    WorldLayout layout_;
    render::RenderWorld render_world_;
    std::vector<PoseWrite> pose_writes_;
    uint64_t static_revision_ = 0u;
};

}  // namespace nuka::runtime::app::viewer
