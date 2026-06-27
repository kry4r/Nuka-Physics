// ---------------------------------------------------------------------------
// field_scene - compose a robot .nks scene over a procedural field terrain and
// Save the result as a .nks/.nka pair the viewer loads via --scene.
// ---------------------------------------------------------------------------
// Usage: field_scene <robot.nks> <out.nks> [--rows R] [--cols C]
//                    [--spacing S] [--amp A] [--cell C] [--margin M]
//
// Builds ONE HeightField (the engine's elevation grid), tessellates it into a
// visual-mesh surface (a static body, contype/conaffinity == 0 -> a .nka MESH
// chunk BuildRenderWorld decodes), then Composes the robot scene R*C times, each
// re-rooted onto the local surface height via nuka::scene::Compose. The SAME
// HeightField sampler places the feet and the mesh vertices, so a dog can never
// drift from the rendered surface. General authoring: one robot in, N on terrain.
// HOST-ONLY (no CUDA); mirrors cook_scene's link set.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"
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
    uint32_t rows = 2u;
    uint32_t cols = 3u;
    float spacing = 2.6f;   // dog-to-dog pitch (m).
    float amp = 0.16f;      // peak-to-peak field elevation (m).
    float cell = 0.18f;     // terrain grid spacing (m).
    float margin = 2.2f;    // flat-ish field rim around the dog grid (m).
};

// A gentle multi-frequency rolling field in roughly [-1, 1]; long wavelengths so
// the slope under one dog's footprint stays a few centimetres.
double FieldRaw(double x, double y) {
    double z = 0.55 * std::sin(0.55 * x + 0.4) * std::cos(0.48 * y - 0.2);
    z += 0.30 * std::sin(0.31 * (x + y) + 1.1);
    z += 0.15 * std::cos(0.83 * x - 0.6) * std::sin(0.27 * y + 0.9);
    return z;
}

// Fill a HeightField over [xmin,xmax]x[ymin,ymax] from FieldRaw, normalized to
// [0,1] with scale_z == amp (so world z in [0, amp]).
terrain::HeightField BuildField(double xmin, double xmax, double ymin,
                                double ymax, float cell, float amp) {
    const auto ncol = std::max(2u, static_cast<uint32_t>(
                                       std::lround((xmax - xmin) / cell)) + 1u);
    const auto nrow = std::max(2u, static_cast<uint32_t>(
                                       std::lround((ymax - ymin) / cell)) + 1u);
    terrain::HeightField hf;
    hf.nrow = nrow;
    hf.ncol = ncol;
    hf.cell_x = static_cast<float>((xmax - xmin) / (ncol - 1u));
    hf.cell_y = static_cast<float>((ymax - ymin) / (nrow - 1u));
    hf.origin = Vec3{static_cast<float>(xmin), static_cast<float>(ymin), 0.0f};
    hf.base_z = 0.0f;
    hf.scale_z = amp;
    hf.values.resize(static_cast<size_t>(nrow) * ncol);
    double lo = 1e30, hi = -1e30;
    std::vector<double> raw(hf.values.size());
    for (uint32_t r = 0; r < nrow; ++r) {
        for (uint32_t c = 0; c < ncol; ++c) {
            const double x = xmin + static_cast<double>(c) * hf.cell_x;
            const double y = ymin + static_cast<double>(r) * hf.cell_y;
            const double v = FieldRaw(x, y);
            raw[static_cast<size_t>(r) * ncol + c] = v;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    const double span = (hi - lo) > 1e-9 ? (hi - lo) : 1.0;
    for (size_t i = 0; i < raw.size(); ++i)
        hf.values[i] = static_cast<float>((raw[i] - lo) / span);
    return hf;
}

// Tessellate the HeightField grid into a triangle surface with smooth per-vertex
// normals (central-difference gradient), in WORLD coordinates (terrain body at
// identity). Two triangles per cell, row-major.
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
        else if (s == "--amp") a.amp = next_f(a.amp);
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
                     "[--spacing S] [--amp A] [--cell C] [--margin M]\n",
                     argv[0]);
        return 2;
    }

    try {
        // -- field extent from the dog grid + rim ------------------------------
        const double half_x = 0.5 * (args.cols - 1) * args.spacing;
        const double half_y = 0.5 * (args.rows - 1) * args.spacing;
        const double xmin = -half_x - args.margin, xmax = half_x + args.margin;
        const double ymin = -half_y - args.margin, ymax = half_y + args.margin;
        const terrain::HeightField hf =
            BuildField(xmin, xmax, ymin, ymax, args.cell, args.amp);

        std::vector<float> pos, nrm;
        std::vector<uint32_t> idx;
        TessellateField(hf, &pos, &idx, &nrm);

        // -- terrain scene: a static body carrying the visual-mesh surface -----
        scene::SceneIR out;
        scene::MaterialRecord mat;
        mat.name = "terrain";
        mat.base_color = Vec3{0.34f, 0.40f, 0.26f};
        mat.roughness = 0.92f;
        mat.metallic = 0.0f;
        mat.friction_mu = 1.0f;
        const scene::MaterialId mid = out.AddMaterial(mat);

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

        // -- compose the robot R*C times, each on the local surface ------------
        const scene::SceneIR robot = scene::nks::Load(args.robot_path);
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
        std::printf("field_scene: %s + %ux%u %s -> %s\n", "terrain", args.rows,
                    args.cols, args.robot_path.c_str(), args.out_path.c_str());
        std::printf("  field grid=%ux%u cells, extent=[%.2f,%.2f]x[%.2f,%.2f] amp=%.2fm\n",
                    hf.nrow, hf.ncol, xmin, xmax, ymin, ymax, args.amp);
        std::printf("  surface tris=%zu, dogs=%u, scene bodies=%zu shapes=%zu\n",
                    idx.size() / 3, placed, out.RigidBodyCount(), out.ShapeCount());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "field_scene FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
