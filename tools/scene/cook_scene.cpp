// ---------------------------------------------------------------------------
// cook_scene - import a robot description (MJCF/URDF/USD) and Save it as a
// .nks/.nka scene pair (M8.5 T5: the visual-mesh cook).
// ---------------------------------------------------------------------------
// Usage: cook_scene <input.(xml|urdf|usd[a])> <output.nks>
//
// The .nks Save (scene/format/nks.cpp) routes each non-colliding (visual-only)
// geom's triangle geometry into a .nka MESH chunk and records a MESH AssetRef on
// the visual_mesh node, so a later Load -> BuildRenderWorld renders the robot as
// real triangle meshes instead of placeholder boxes. Collision meshes still go
// to CMSH; primitive geoms tessellate. HOST-ONLY (no CUDA), so it builds in the
// default config.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

std::string LowerExt(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

nuka::scene::SceneIR Import(const std::string& path) {
    const std::string ext = LowerExt(path);
    if (ext == "xml") return nuka::import::LoadMjcf(path);
    if (ext == "urdf") return nuka::import::LoadUrdf(path);
    if (ext == "usd" || ext == "usda" || ext == "usdc") return nuka::import::LoadUsd(path);
    throw std::runtime_error("cook_scene: unknown extension for input: " + path);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input.(xml|urdf|usd[a])> <output.nks>\n", argv[0]);
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];

    try {
        nuka::scene::SceneIR scene = Import(in_path);

        // Count visual-mesh shapes (the ones the cook routes to MESH chunks).
        size_t visual_with_tris = 0, total_tris = 0;
        for (nuka::scene::ShapeId i = 0; i < scene.ShapeCount(); ++i) {
            const auto& s = scene.GetShape(i);
            const bool is_visual = (s.contype == 0 && s.conaffinity == 0);
            if (is_visual && !s.mesh_vertices.empty() && !s.mesh_indices.empty()) {
                ++visual_with_tris;
                total_tris += s.mesh_indices.size() / 3;
            }
        }

        nuka::scene::nks::Save(scene, out_path);

        std::printf("cooked %s -> %s\n", in_path.c_str(), out_path.c_str());
        std::printf("  bodies=%u shapes=%u visual_mesh_shapes=%zu visual_triangles=%zu\n",
                    static_cast<unsigned>(scene.RigidBodyCount()),
                    static_cast<unsigned>(scene.ShapeCount()), visual_with_tris,
                    total_tris);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cook_scene FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
