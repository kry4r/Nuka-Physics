// ---------------------------------------------------------------------------
// go2_terrain_field - author a Go2-on-composite-rolling-heightfield .nks the
// editor loads AND simulates closed-loop.
//
// Unlike field_scene (a VISUAL-only surface), this emits a first-class COLLISION
// TerrainRecord: CookToModel bakes it into the static heightfield collidable AND
// returns the grid for the locomotion controller's height-scan obs. The SAME
// engine HeightField (GenerateHeightField over a parametric composite config --
// the 20-dog field: tiled ring-steps + hashed bumps) drives the collision, the
// tessellated visual surface, and the obs sampler, so render/physics/obs agree by
// construction. K FLOATING Go2 (root mass>0, not static) are Composed onto the
// local surface, each pre-posed to the policy nominal stance so the trained policy
// starts in-distribution. HOST-ONLY (no CUDA).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"
#include "scene/terrain/heightfield_sample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
namespace scene = nuka::scene;
namespace terrain = nuka::terrain;

struct Args {
    std::string robot_path;
    std::string out_path;
    uint32_t rows = 5u;
    uint32_t cols = 6u;
    float spacing = 2.5f;   // dog-to-dog pitch (m), matching the 20-dog dump.
    float cell = 0.1f;      // terrain grid spacing (m).
    float margin = 8.0f;    // field rim around the dog grid (m) -- room to walk.
};

// The policy nominal stance (rad) by joint name; dogs are authored at this pose so
// (q - default) ~ 0 at spawn (the trained policy starts in-distribution).
float NominalAngle(const std::string& joint_name) {
    struct Entry { const char* name; float angle; };
    static const Entry kStance[] = {
        {"FL_hip", 0.1f}, {"FL_thigh", 0.8f}, {"FL_calf", -1.5f},
        {"FR_hip", -0.1f}, {"FR_thigh", 0.8f}, {"FR_calf", -1.5f},
        {"RL_hip", 0.1f}, {"RL_thigh", 1.0f}, {"RL_calf", -1.5f},
        {"RR_hip", -0.1f}, {"RR_thigh", 1.0f}, {"RR_calf", -1.5f},
    };
    for (const Entry& e : kStance) {
        // Substring match resolves both a composed prefix ("dog3/FL_hip_joint") and
        // the URDF "_joint" suffix; the 12 leg keys are mutually non-substring.
        if (joint_name.find(e.name) != std::string::npos) return e.angle;
    }
    return 0.0f;
}

// The composite "rolling" field: tiled feature cells (ring-steps + hashed bumps) at
// gentle ~0.15m amplitudes -- the parametric field the terrain policy trained on.
terrain::TerrainGenConfig CompositeConfig(double xmin, double ymin, uint32_t nrow,
                                          uint32_t ncol, float cell) {
    terrain::TerrainGenConfig cfg;
    cfg.nrow = nrow;
    cfg.ncol = ncol;
    cfg.cell_x = cell;
    cfg.cell_y = cell;
    cfg.origin = Vec3{static_cast<float>(xmin), static_cast<float>(ymin), 0.0f};
    cfg.base_z = 0.0f;
    cfg.ring_rise = 0.15f;
    cfg.ring_width = 0.30f;
    cfg.ring_platform = 1.0f;
    cfg.ring_count = 8u;
    cfg.bump_height = 0.15f;
    cfg.bump_cell = 0.45f;
    cfg.feature_cell = 5.0f;
    cfg.feature_margin = 0.6f;
    return cfg;
}

// Mirror a TerrainGenConfig into a TerrainRecord (CookToModel rebuilds the same
// HeightField from it via GenerateHeightField -- the cook bakes the IDENTICAL grid).
scene::TerrainRecord RecordFromConfig(const terrain::TerrainGenConfig& cfg) {
    scene::TerrainRecord tr;
    tr.name = "terrain";
    tr.nrow = cfg.nrow;
    tr.ncol = cfg.ncol;
    tr.cell = cfg.cell_x;
    tr.origin = cfg.origin;
    tr.base_z = cfg.base_z;
    tr.ring_rise = cfg.ring_rise;
    tr.ring_width = cfg.ring_width;
    tr.ring_platform = cfg.ring_platform;
    tr.ring_count = cfg.ring_count;
    tr.bump_height = cfg.bump_height;
    tr.bump_cell = cfg.bump_cell;
    tr.feature_cell = cfg.feature_cell;
    tr.feature_margin = cfg.feature_margin;
    return tr;
}

// Tessellate the HeightField grid into a triangle surface (two tris per cell, smooth
// per-vertex normals from a central-difference gradient), in WORLD coordinates.
void TessellateField(const terrain::HeightField& hf, std::vector<float>* pos,
                     std::vector<uint32_t>* idx, std::vector<float>* nrm) {
    const uint32_t nr = hf.nrow, nc = hf.ncol;
    auto z_at = [&](uint32_t r, uint32_t c) { return terrain::WorldZAt(hf, r, c); };
    for (uint32_t r = 0; r < nr; ++r) {
        for (uint32_t c = 0; c < nc; ++c) {
            const float x = hf.origin.x + static_cast<float>(c) * hf.cell_x;
            const float y = hf.origin.y + static_cast<float>(r) * hf.cell_y;
            pos->push_back(x);
            pos->push_back(y);
            pos->push_back(z_at(r, c));
            const uint32_t cl = c > 0 ? c - 1u : c, cr = c + 1u < nc ? c + 1u : c;
            const uint32_t rd = r > 0 ? r - 1u : r, ru = r + 1u < nr ? r + 1u : r;
            const float dzdx = (z_at(r, cr) - z_at(r, cl)) /
                               (static_cast<float>(cr - cl) * hf.cell_x + 1e-9f);
            const float dzdy = (z_at(ru, c) - z_at(rd, c)) /
                               (static_cast<float>(ru - rd) * hf.cell_y + 1e-9f);
            Vec3 n = Vec3{-dzdx, -dzdy, 1.0f};
            const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            n = Vec3{n.x / len, n.y / len, n.z / len};
            nrm->push_back(n.x);
            nrm->push_back(n.y);
            nrm->push_back(n.z);
        }
    }
    for (uint32_t r = 0; r + 1 < nr; ++r) {
        for (uint32_t c = 0; c + 1 < nc; ++c) {
            const uint32_t a = r * nc + c, b = r * nc + c + 1u;
            const uint32_t d = (r + 1u) * nc + c, e = (r + 1u) * nc + c + 1u;
            idx->push_back(a); idx->push_back(b); idx->push_back(e);
            idx->push_back(a); idx->push_back(e); idx->push_back(d);
        }
    }
}

Args ParseArgs(int argc, char** argv) {
    Args a;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next_u = [&](uint32_t def) {
            return (i + 1 < argc) ? static_cast<uint32_t>(std::atoi(argv[++i])) : def;
        };
        auto next_f = [&](float def) {
            return (i + 1 < argc) ? static_cast<float>(std::atof(argv[++i])) : def;
        };
        if (s == "--rows") a.rows = std::max(1u, next_u(a.rows));
        else if (s == "--cols") a.cols = std::max(1u, next_u(a.cols));
        else if (s == "--spacing") a.spacing = next_f(a.spacing);
        else if (s == "--cell") a.cell = std::max(0.02f, next_f(a.cell));
        else if (s == "--margin") a.margin = next_f(a.margin);
        else if (positional == 0) { a.robot_path = s; ++positional; }
        else if (positional == 1) { a.out_path = s; ++positional; }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);
    if (args.robot_path.empty() || args.out_path.empty()) {
        std::fprintf(stderr,
                     "usage: %s <robot.nks> <out.nks> [--rows R] [--cols C] "
                     "[--spacing S] [--cell C] [--margin M]\n",
                     argv[0]);
        return 2;
    }

    try {
        // -- field extent from the dog grid + rim ------------------------------
        const double half_x = 0.5 * (args.cols - 1) * args.spacing;
        const double half_y = 0.5 * (args.rows - 1) * args.spacing;
        const double xmin = -half_x - args.margin, xmax = half_x + args.margin;
        const double ymin = -half_y - args.margin, ymax = half_y + args.margin;
        const auto ncol = std::max(2u, static_cast<uint32_t>(
                                           std::lround((xmax - xmin) / args.cell)) + 1u);
        const auto nrow = std::max(2u, static_cast<uint32_t>(
                                           std::lround((ymax - ymin) / args.cell)) + 1u);

        // The ONE HeightField the collision/render/obs all derive from.
        const terrain::TerrainGenConfig cfg =
            CompositeConfig(xmin, ymin, nrow, ncol, args.cell);
        terrain::HeightField hf;
        if (!terrain::GenerateHeightField(cfg, hf)) {
            std::fprintf(stderr, "go2_terrain_field FAILED: degenerate terrain\n");
            return 1;
        }

        std::vector<float> pos, nrm;
        std::vector<uint32_t> idx;
        TessellateField(hf, &pos, &idx, &nrm);

        scene::SceneIR out;
        scene::MaterialRecord mat;
        mat.name = "terrain";
        mat.base_color = Vec3{0.34f, 0.40f, 0.26f};
        mat.roughness = 0.92f;
        mat.metallic = 0.0f;
        mat.friction_mu = 1.0f;
        const scene::MaterialId mid = out.AddMaterial(mat);

        // The COLLISION terrain (cook bakes the static heightfield collidable).
        out.AddTerrain(RecordFromConfig(cfg));

        // A static body carrying the visual-mesh surface (so the field renders).
        scene::RigidBodyRecord body;
        body.name = "terrain";
        body.is_static = true;
        const scene::BodyId bid = out.AddRigidBody(body);
        scene::CollisionShapeRecord surf;
        surf.body_id = bid;
        surf.material_id = mid;
        surf.name = "surface";
        surf.type = scene::ShapeType::TriMesh;
        surf.contype = 0;       // visual-only -> VisualMeshComponent -> MESH chunk.
        surf.conaffinity = 0;
        surf.decompose_mode = scene::DecomposeMode::Skip;
        surf.mesh_vertices = pos;
        surf.mesh_indices = idx;
        surf.mesh_normals = nrm;
        out.AddCollisionShape(surf);

        // -- the robot, pre-posed to the policy nominal stance -----------------
        scene::SceneIR robot = scene::nks::Load(args.robot_path);
        for (size_t j = 0; j < robot.JointCount(); ++j) {
            scene::JointRecord& jr =
                robot.GetJointMut(static_cast<scene::JointId>(j));
            if (jr.type == scene::JointType::Revolute) {
                jr.initial_position = NominalAngle(jr.name);
            }
        }

        // -- compose K floating dogs, each on the local surface ----------------
        uint32_t placed = 0u;
        for (uint32_t r = 0; r < args.rows; ++r) {
            for (uint32_t c = 0; c < args.cols; ++c) {
                const float x =
                    static_cast<float>(-half_x + static_cast<double>(c) * args.spacing);
                const float y =
                    static_cast<float>(-half_y + static_cast<double>(r) * args.spacing);
                const float z = terrain::SampleHeightFieldZ(hf, x, y);
                const Transform place{Vec3{x, y, z}, Quat::Identity()};
                out = scene::Compose(out, robot, place,
                                     "dog" + std::to_string(placed) + "/");
                ++placed;
            }
        }

        scene::nks::Save(out, args.out_path);
        std::printf("go2_terrain_field: composite field + %u dogs -> %s\n", placed,
                    args.out_path.c_str());
        std::printf("  grid=%ux%u cells, extent=[%.2f,%.2f]x[%.2f,%.2f], surface tris=%zu\n",
                    hf.nrow, hf.ncol, xmin, xmax, ymin, ymax, idx.size() / 3);
        std::printf("  scene bodies=%zu shapes=%zu terrain_records=%zu\n",
                    out.RigidBodyCount(), out.ShapeCount(), out.TerrainCount());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "go2_terrain_field FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
