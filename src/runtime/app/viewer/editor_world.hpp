#pragma once
// The editor's view of a physics world through the C ABI: the per-environment layout,
// the scene binding the renderer reads, and the pose reads and writes of the physics thread.

#include "nuka/nuka.h"
#include "render/render_world.hpp"
#include "runtime/app/viewer/pose_scatter.hpp"
#include "runtime/soft/particle_surface.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/terrain/heightfield.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// One triangulated particle surface; its triangles index one environment's particles.
struct ParticleSurfaceBinding {
    nuka_particle_surface_t info{};
    std::vector<uint32_t> triangles;
};

struct WorldLayout {
    uint32_t env_count = 1u;
    uint32_t links_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t articulations_per_env = 0u;
    uint32_t dofs_per_env = 0u;
    uint32_t contacts_per_env = 0u;
    uint32_t contact_slots_per_env = 0u;
    uint32_t particles_per_env = 0u;
    std::vector<nuka_link_binding_t> links;  // per-environment link order
    std::vector<ParticleSurfaceBinding> surfaces;
};

// Throws std::runtime_error when a query fails; `what` names the call.
void CheckResult(nuka_result_t result, const char* what);

// A failed call's result message, followed by the text of the exception behind it when it threw.
std::string ResultText(nuka_result_t result);

// One element of a field, `row` elements in.
template <class T>
T DownloadRow(nuka_world_handle world, nuka_state_field_t field, size_t row) {
    T value{};
    CheckResult(nuka_world_download_field(world, field, &value, sizeof(T), row * sizeof(T)), "nuka_world_download_field");
    return value;
}

WorldLayout QueryLayout(nuka_world_handle world);

// The cooked terrain grid; invalid when the world has none.
terrain::HeightField QueryTerrain(nuka_world_handle world);

// Body records map to their cooked rows (record order) and link bodies to their links,
// the rows BuildRenderWorld resolves instance poses from.
scene::SceneMap BindScene(const scene::SceneIR& scene, const WorldLayout& layout);

// A free body or a floating articulation root: the pose sources a live move reaches.
bool IsMovableSource(const render::PoseSource& source, const WorldLayout& layout);

// The link or body pose a source names; false for static sources and rows the poses lack.
bool ResolvePose(const render::PoseSource& source, const std::vector<math::Transform>& links,
                 const std::vector<math::Transform>& bodies, math::Transform* out);

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

// One environment's particle positions; empty without particles.
void DownloadEnvParticles(nuka_world_handle world, const WorldLayout& layout, uint32_t env_index,
                          std::vector<math::Vec3>* out);

// A solver telemetry element in the world: where it is and the entity drawing it.
struct LocatedElement {
    math::Vec3 position = math::Vec3::Zero();
    scene::EntityId entity = scene::kInvalidEntity;
    uint32_t env = 0u;
    const char* kind = "";  // the label format taking index: "particle %u", "body %u" or "articulation %u"
    uint32_t index = 0u;
    int64_t row = -1;  // the contact row it sides, or -1
};

// Resolves a record's worst element, or a contact row's first located side; false when neither is.
bool LocateTelemetry(nuka_world_handle world, const WorldLayout& layout, const scene::SceneIR& scene,
                     const nuka_telemetry_record_t& record, LocatedElement* out);

// The zero-copy rows of the render instances, in instance order.
std::vector<ScatterRow> BuildScatterRows(const render::RenderWorld& render_world);

// The UI thread's binding of one world: the layout and the render world it draws, with one
// deforming instance per particle surface owned by its medium's entity.
class EditorSim {
public:
    // One copy row of `env_instances`: the environment it draws and the instance it copies.
    struct EnvCopy {
        uint32_t env;
        uint32_t instance;
    };
    // A velocity arrow from `at` along the planar `velocity`, in the command or the measured colour; the
    // measured one stacks on the command's.
    struct Arrow {
        math::Vec3 at;
        math::Vec3 velocity;
        bool command;
    };

    EditorSim(WorldLayout layout, render::RenderWorld render_world, const scene::SceneIR& scene);

    // Re-poses instances, cameras and lights from one environment's poses.
    void ApplyPoses(const std::vector<math::Transform>& links, const std::vector<math::Transform>& bodies);
    // Poses the ghost copies of the physics-driven instances; empty poses clear them.
    void ApplyGhost(const std::vector<math::Transform>& links, const std::vector<math::Transform>& bodies);
    // Clears the other environments' copies; AddEnv appends one environment's moving instances at `offset`.
    void ClearEnvs();
    void AddEnv(uint32_t env, const std::vector<math::Transform>& links, const std::vector<math::Transform>& bodies,
                const math::Vec3& offset);
    const std::vector<EnvCopy>& EnvCopies() const { return env_copies_; }
    // Rebuilds the particle surfaces from one environment's particles; per-particle residuals
    // colour them as a heatmap, an empty `heat` draws their materials.
    void ApplyParticles(const std::vector<math::Vec3>& positions, const std::vector<float>& heat, float alert);
    // Draws a terrain surface under the scene; an invalid grid removes it.
    void SetTerrain(const terrain::HeightField& terrain);
    // Replaces the velocity arrows.
    void SetArrows(const std::vector<Arrow>& arrows);
    // The highest point of the link-driven instances posed at one environment's `links`; -inf when none is.
    float LinkTop(const std::vector<math::Transform>& links) const;

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
    struct Surface {
        uint32_t instance;
        uint32_t mesh;
        uint32_t material;
        runtime::soft::SurfaceTopology topology;
    };

    WorldLayout layout_;
    render::RenderWorld render_world_;
    std::vector<Surface> surfaces_;
    uint32_t heat_material_ = render::kNoId;
    uint32_t ghost_material_ = render::kNoId;
    uint32_t terrain_material_ = render::kNoId;
    uint32_t arrow_materials_[2]{render::kNoId, render::kNoId};  // measured, command
    uint32_t terrain_mesh_ = render::kNoId;
    bool terrain_shown_ = false;  // the terrain leads extra_instances
    std::vector<EnvCopy> env_copies_;
    std::vector<PoseWrite> pose_writes_;
    uint64_t static_revision_ = 0u;
};

}  // namespace nuka::runtime::app::viewer
