#pragma once
// Collider generation from a body's visual meshes, merged in the body frame: one convex hull,
// a V-HACD decomposition, an SDF-contact mesh or the raw triangle mesh.

#include "scene/format/json.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

enum class CollisionMethod : uint8_t { Hull, Decompose, Sdf, Mesh };

bool ParseCollisionMethod(const std::string& name, CollisionMethod* out);
const char* CollisionMethodName(CollisionMethod method);

struct CollisionOptions {
    CollisionMethod method = CollisionMethod::Hull;
    uint32_t max_pieces = 16u;    // decomposition only
    uint32_t max_triangles = 0u;  // triangle mesh only: merges vertices on a grid to keep at most this many, 0 all
    bool replace = true;          // drop the body's colliders first
    bool only_missing = false;    // leave bodies that have a collider
    bool subtree = false;         // a body's descendants too; a group always takes all under it
};

// Reads the options an action or edit op names; missing keys keep the defaults.
CollisionOptions CollisionOptionsFromJson(const scene::json::Value& v);
scene::json::Value CollisionOptionsToJson(const CollisionOptions& options);

// What generation does to one body.
struct CollisionPreview {
    std::string path;
    uint32_t triangles = 0u;  // of the merged visual mesh
    uint32_t pieces = 0u;
    uint32_t vertices = 0u;
    bool had_collider = false;
    std::string skipped;  // why the body is left alone, empty when it is not
};

// The bodies the nodes at `paths` name: a body (with its descendants for `subtree`), the body a
// shape or joint sits in, or every body under a group.
std::vector<scene::BodyId> CollisionBodies(const scene::SceneIR& scene, const std::vector<std::string>& paths,
                                           bool subtree);
std::vector<CollisionPreview> PreviewCollision(const scene::SceneIR& scene, const std::vector<std::string>& paths,
                                               const CollisionOptions& options);
scene::json::Value PreviewToJson(const std::vector<CollisionPreview>& preview);
// Writes the colliders into the records; returns how many bodies got one.
size_t GenerateCollision(scene::SceneIR& scene, const std::vector<std::string>& paths, const CollisionOptions& options);
// Each colliding triangle mesh on a moving body becomes the convex hull of its own vertices, as MuJoCo
// collides mesh geoms; one with authored decomposition, SDF contact or reduction stays. Returns how many.
size_t HullMeshColliders(scene::SceneIR& scene);
// The colliding triangle meshes on moving bodies that HullMeshColliders would turn into hulls.
size_t MovingMeshColliders(const scene::SceneIR& scene);

}  // namespace nuka::runtime::app::viewer
