// ---------------------------------------------------------------------------
// go2_terrain_demo.cpp -- Go2-on-stairs demo, PHASE 3: the MULTI-DOG TERRAIN
// RENDER (an Isaac-Lab-cover-style video).
//
// Renders N Go2 quadrupeds, EACH on its own procedural-terrain tile in a grid,
// with a clean bright studio look, soft per-foot contact shadows that track the
// LOCAL terrain height under each foot, atmospheric white haze fading the distant
// dogs/terrain, and a ROTATING (orbit) hero camera that circles the scene to
// SHOWCASE THE GAIT from multiple angles -- offscreen on lavapipe-CPU, to a 1080p
// mp4 (PPM frames -> ffmpeg).
//
// This is the multi-robot + terrain EXTENSION of go2_walk_video.cpp. It reuses:
//   * the FROZEN go2.nks/.nka beauty meshes (Load -> CookToModel -> BuildRenderWorld),
//   * the SAME shape->link rebind + premium dark-body/metal/foot material override,
//   * the contact_points[] per-foot shadow-decal feature, and
//   * the offscreen PBR raster path (studio ground + 3-point rig + the new fog).
// It ADDS:
//   * a TERRAIN MESH TESSELLATOR that samples the SHARED physics heightfield
//     nuka::terrain::SampleTerrainHeight() (anti-drift: the rendered ground can
//     NEVER disagree with the surface the feet rest on -- ONE header, ONE h(x,y)),
//   * N-robot composition (each dog's dumped link poses are offset to its grid
//     tile origin at PUBLISH time; physics was per-env isolated),
//   * the orbit camera, the haze, and per-foot shadows placed at the local terrain
//     height under each foot.
//
// ============================ TRAJECTORY FORMAT =============================
// The real terrain rollout (5 trained-policy dogs on terrain) is NOT ready yet.
// This tool consumes a NEW N-robot format, 'GO2T' v1, and the REAL rollout dump
// plugs in at the SAME seam (--bin <path>). Until then, --placeholder synthesises
// an N-robot GO2T in-memory by replicating the EXISTING single-robot flat-walk
// GO2W trajectory (out/go2_walk_trajectory.bin) to N robots, assigning each a
// terrain tile (mix of Stairs/Pit/Boxes spanning difficulties 0.08-0.15 m) and a
// grid offset. (The flat-walk dog won't perfectly track its terrain -- that's
// fine; this tests the RENDER infra.)
//
// GO2T v1 BINARY LAYOUT (little-endian; flat, no padding):
//   HEADER (int32 unless noted):
//     [0]  magic            = 0x47324F54  ('G2OT' on disk; read LE uint32 == this)
//     [1]  version          = 1
//     [2]  num_robots       = R
//     [3]  num_steps        = N   (recorded physics sub-steps; same N for all robots)
//     [4]  dof_count        = 13  (per-robot DRIVE_TARGET slot count incl. root)
//     [5]  link_count       = 13  (per-robot ARTICULATION_LINK_POSE link count)
//     [6]  pose_floats      = 7   (per link: px,py,pz,qw,qx,qy,qz)
//     [7]  decimation       = 4
//     float [8] dt          = 0.005
//     float [9] ctrl_dt     = 0.020
//   Then R PER-ROBOT TILE records, each:
//     float grid_offset_x, grid_offset_y   (tile origin in world xy)
//     int32 terrain_type                   (0 Flat,1 PyramidStairs,2 InvertedPyramid,3 RandomBoxes)
//     float difficulty                     (vertical feature scale, ~0.08-0.15)
//   Then, contiguous, N records, each holding R robots back-to-back:
//     per robot: drive_target[dof_count] f32 , link_pose[link_count*pose_floats] f32
//   (the per-robot record is byte-identical to a GO2W record, so a real dump can
//    just concatenate R single-robot rollouts step-interleaved.)
//
//   The per-robot TILE TerrainParams (step_height, step_width, platform_width,
//   grid_width, grid_height_max, ground_height) are NOT in the .bin -- they are a
//   FIXED studio-terrain preset (kTilePreset below) shared by the renderer and (in
//   the real rollout) the physics cook. difficulty scales the vertical magnitude
//   via nuka::terrain::ScaleTerrainDifficulty, exactly as the physics kernel does.
//
// =============================== REAL-ROLLOUT SEAM ==========================
// To plug in the real 5-trained-policy terrain rollout: extend
// go2_dump_walk_trajectory.py to write a GO2T .bin (num_robots header + per-robot
// tile records + step-interleaved per-robot GO2W records) from R per-env policy
// rollouts on the procedural terrain, then run this tool with --bin <that file>
// (NO --placeholder). The renderer is agnostic to how the poses were produced.
//
// Built behind NK_BUILD_VULKAN_VALIDATION (build-viewer / lavapipe).
//
// USAGE:
//   go2_terrain_demo [--placeholder] [--bin PATH] [--robots R] [--frames N]
//                    [--stride S] [--out-dir DIR] [--width W] [--height H]
//                    [--orbit-turns T] [--fog D] [--probe]
//     --placeholder  : synthesise the N-robot GO2T from the single-robot GO2W
//                      (out/go2_walk_trajectory.bin). The DEFAULT when no --bin.
//     --bin PATH     : load a real GO2T .bin (the trained-terrain rollout seam).
//     --robots R     : placeholder robot count (default 5).
//     --orbit-turns T: camera azimuth sweep in full turns over the clip (default 0.6).
//     --fog D        : haze density per metre (default 0.022; 0 => no haze).
//     --probe        : build + render 1 mid-clip frame to <out-dir>/probe_frame.ppm.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "sensor/terrain/terrain_field.hpp"

namespace {

namespace render = nuka::render;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/go2.nks";
constexpr const char* kGo2WBin = "out/go2_walk_trajectory.bin";
constexpr uint32_t kGo2WMagic = 0x474F3257u;  // 'GO2W' (single-robot source)
constexpr uint32_t kGo2TMagic = 0x47324F54u;  // 'G2OT' (N-robot terrain format)
constexpr float kPi = 3.14159265358979323846f;

// ---- THE FIXED STUDIO-TERRAIN PRESET --------------------------------------
// Shared by the render tessellator and (in the real rollout) the physics cook --
// the SINGLE place the horizontal geometry is defined. difficulty scales only the
// vertical magnitude (step_height / grid_height_max) via ScaleTerrainDifficulty,
// EXACTLY as the foot-contact kernel does, so the render mesh tracks the physics
// surface. ground_height = 0 puts every tile's reference floor at the studio z=0.
// Horizontal step 0.30 m reads as crisp stairs; the tessellator spacing (<=0.05 m,
// below) is fine enough that the step edges stay sharp.
terrain::TerrainParams TilePreset() {
    terrain::TerrainParams p{};
    p.ground_height   = 0.0f;
    p.step_height     = 1.0f;    // base rise per ring -> scaled by difficulty (-> ~0.08-0.15 m)
    p.step_width      = 0.30f;   // horizontal width of each step ring (crisp stairs)
    p.platform_width  = 0.50f;   // central top platform / pit-floor edge length
    p.grid_width      = 0.45f;   // RandomBoxes cell edge
    p.grid_height_max = 1.0f;    // base max box rise -> scaled by difficulty
    return p;
}

// The render extent of ONE tile (half-width in x and y, world metres). Large
// enough to show several stair rings (8 rings * 0.30 m = 2.4 m radius of feature)
// plus a margin of flat floor around the dog's walk.
constexpr float kTileHalf = 1.60f;     // tile spans [-kTileHalf, +kTileHalf] in x,y (local)
// EDGE-TO-EDGE tiling (the Isaac-Lab cover look): the pitch == the tile full width
// (2*half) so neighbouring tiles TOUCH with no gap, reading as one continuous
// terrain field. A hair of overlap (0.999) hides any seam from float error.
constexpr float kTilePitch = 2.0f * kTileHalf * 0.999f;  // edge-to-edge tile spacing

// ---- CLI ------------------------------------------------------------------
struct Args {
    bool        placeholder = true;   // default: synthesise from the single-robot GO2W
    std::string bin;                  // a real GO2T .bin (overrides placeholder when set)
    uint32_t    robots = 0u;          // 0 => derive from rows*cols (the grid); else override
    uint32_t    rows = 3u;            // grid rows  (default 3x4 = 12-dog Isaac-Lab field)
    uint32_t    cols = 4u;            // grid cols
    uint32_t    frames = 0u;          // 0 => all available (after stride)
    uint32_t    stride = 2u;          // render every stride-th dumped sub-step
    std::string out_dir = "/tmp/go2_terrain_frames";
    uint32_t    width = 1920u;
    uint32_t    height = 1080u;
    float       orbit_turns = 0.6f;   // azimuth sweep in full turns over the clip
    float       fog = 0.004f;         // haze density per metre (0 => off); subtle far-field
    bool        probe = false;
    bool        no_shadows = false;   // disable the directional shadow map (fallback)
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next_u = [&](uint32_t def) -> uint32_t {
            return (i + 1 < argc) ? static_cast<uint32_t>(std::atoi(argv[++i])) : def;
        };
        auto next_f = [&](float def) -> float {
            return (i + 1 < argc) ? static_cast<float>(std::atof(argv[++i])) : def;
        };
        if (s == "--placeholder") a.placeholder = true;
        else if (s == "--bin" && i + 1 < argc) { a.bin = argv[++i]; a.placeholder = false; }
        else if (s == "--robots") a.robots = std::max(1u, next_u(a.robots));
        else if (s == "--rows") a.rows = std::max(1u, next_u(a.rows));
        else if (s == "--cols") a.cols = std::max(1u, next_u(a.cols));
        else if (s == "--frames") a.frames = next_u(a.frames);
        else if (s == "--stride") a.stride = std::max(1u, next_u(a.stride));
        else if (s == "--width") a.width = next_u(a.width);
        else if (s == "--height") a.height = next_u(a.height);
        else if (s == "--orbit-turns") a.orbit_turns = next_f(a.orbit_turns);
        else if (s == "--fog") a.fog = next_f(a.fog);
        else if (s == "--probe") a.probe = true;
        else if (s == "--no-shadows") a.no_shadows = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
    }
    // Default robot count = a full rows*cols grid (the Isaac-Lab field) unless the
    // caller explicitly set --robots.
    if (a.robots == 0u) a.robots = a.rows * a.cols;
    return a;
}

// ---- one robot's terrain tile ---------------------------------------------
struct TileSpec {
    float    offset_x = 0.0f, offset_y = 0.0f;   // tile centre in world xy
    uint32_t terrain_type = 0u;                  // 0/1/2/3
    float    difficulty   = 0.0f;                // vertical feature scale
};

// ---- the N-robot trajectory (GO2T) ----------------------------------------
struct NTrajectory {
    uint32_t num_robots = 0u;
    uint32_t num_steps = 0u;
    uint32_t dof_count = 0u;
    uint32_t link_count = 0u;
    uint32_t pose_floats = 0u;
    float dt = 0.0f, ctrl_dt = 0.0f;
    std::vector<TileSpec> tiles;                 // [num_robots]
    // Per-robot START-PHASE offset (sub-steps) into the looped trajectory so the
    // replicated placeholder dogs do NOT march in lockstep. 0 for a real GO2T.
    std::vector<uint32_t> phase;                 // [num_robots]
    // pose[step] is a flat (num_robots * link_count * pose_floats) array.
    std::vector<std::vector<float>> pose;

    // World transform of robot r's link l at sub-step `step` (quat w-first), in the
    // robot's OWN (per-env) frame -- the tile offset is applied by the caller.
    Transform LinkPose(uint32_t step, uint32_t r, uint32_t l) const {
        const size_t per_robot = static_cast<size_t>(link_count) * pose_floats;
        const float* p = &pose[step][static_cast<size_t>(r) * per_robot +
                                     static_cast<size_t>(l) * pose_floats];
        return Transform{Vec3{p[0], p[1], p[2]}, Quat{p[3], p[4], p[5], p[6]}};
    }

    // Robot r's link l at clip step `step`, with the per-robot START-PHASE applied
    // (looped) so the replicated dogs are desynced. phase is 0 for a real GO2T.
    Transform PhasedPose(uint32_t step, uint32_t r, uint32_t l) const {
        uint32_t s = step;
        if (!phase.empty() && num_steps > 0u) s = (step + phase[r]) % num_steps;
        return LinkPose(s, r, l);
    }
};

// ---- load a single-robot GO2W trajectory (the placeholder source) ----------
struct SingleTrajectory {
    uint32_t num_steps = 0u, dof_count = 0u, link_count = 0u, pose_floats = 0u;
    float dt = 0.0f, ctrl_dt = 0.0f;
    std::vector<std::vector<float>> pose;  // pose[step] = link_count*pose_floats f32
};

bool LoadGo2W(const std::string& path, SingleTrajectory* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[go2_terrain] cannot open %s\n", path.c_str()); return false; }
    int32_t hdr_i[8]; float hdr_f[2];
    f.read(reinterpret_cast<char*>(hdr_i), sizeof(hdr_i));
    f.read(reinterpret_cast<char*>(hdr_f), sizeof(hdr_f));
    if (!f || static_cast<uint32_t>(hdr_i[0]) != kGo2WMagic) {
        std::fprintf(stderr, "[go2_terrain] bad/short GO2W header in %s\n", path.c_str());
        return false;
    }
    out->num_steps = static_cast<uint32_t>(hdr_i[2]);
    out->dof_count = static_cast<uint32_t>(hdr_i[3]);
    out->link_count = static_cast<uint32_t>(hdr_i[4]);
    out->pose_floats = static_cast<uint32_t>(hdr_i[5]);
    out->dt = hdr_f[0]; out->ctrl_dt = hdr_f[1];
    const size_t pose_n = static_cast<size_t>(out->link_count) * out->pose_floats;
    const size_t rec_floats = static_cast<size_t>(out->dof_count) + pose_n;
    out->pose.resize(out->num_steps);
    std::vector<float> rec(rec_floats);
    for (uint32_t s = 0; s < out->num_steps; ++s) {
        f.read(reinterpret_cast<char*>(rec.data()),
               static_cast<std::streamsize>(rec_floats * sizeof(float)));
        if (!f) { std::fprintf(stderr, "[go2_terrain] short GO2W record %u\n", s); return false; }
        out->pose[s].assign(rec.begin() + out->dof_count, rec.end());
    }
    return true;
}

// ---- load a real N-robot GO2T trajectory (the trained-rollout seam) ---------
bool LoadGo2T(const std::string& path, NTrajectory* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[go2_terrain] cannot open %s\n", path.c_str()); return false; }
    int32_t hdr_i[8]; float hdr_f[2];
    f.read(reinterpret_cast<char*>(hdr_i), sizeof(hdr_i));
    f.read(reinterpret_cast<char*>(hdr_f), sizeof(hdr_f));
    if (!f || static_cast<uint32_t>(hdr_i[0]) != kGo2TMagic) {
        std::fprintf(stderr, "[go2_terrain] bad/short GO2T header in %s\n", path.c_str());
        return false;
    }
    out->num_robots = static_cast<uint32_t>(hdr_i[2]);
    out->num_steps  = static_cast<uint32_t>(hdr_i[3]);
    out->dof_count  = static_cast<uint32_t>(hdr_i[4]);
    out->link_count = static_cast<uint32_t>(hdr_i[5]);
    out->pose_floats = static_cast<uint32_t>(hdr_i[6]);
    out->dt = hdr_f[0]; out->ctrl_dt = hdr_f[1];
    out->tiles.resize(out->num_robots);
    for (uint32_t r = 0; r < out->num_robots; ++r) {
        float fx, fy, diff; int32_t type;
        f.read(reinterpret_cast<char*>(&fx), 4);
        f.read(reinterpret_cast<char*>(&fy), 4);
        f.read(reinterpret_cast<char*>(&type), 4);
        f.read(reinterpret_cast<char*>(&diff), 4);
        if (!f) { std::fprintf(stderr, "[go2_terrain] short tile record %u\n", r); return false; }
        out->tiles[r] = TileSpec{fx, fy, static_cast<uint32_t>(type), diff};
    }
    const size_t per_robot = static_cast<size_t>(out->dof_count) +
        static_cast<size_t>(out->link_count) * out->pose_floats;
    const size_t pose_per_robot = static_cast<size_t>(out->link_count) * out->pose_floats;
    out->pose.resize(out->num_steps);
    std::vector<float> rec(per_robot);
    for (uint32_t s = 0; s < out->num_steps; ++s) {
        out->pose[s].resize(static_cast<size_t>(out->num_robots) * pose_per_robot);
        for (uint32_t r = 0; r < out->num_robots; ++r) {
            f.read(reinterpret_cast<char*>(rec.data()),
                   static_cast<std::streamsize>(per_robot * sizeof(float)));
            if (!f) { std::fprintf(stderr, "[go2_terrain] short GO2T record %u/%u\n", s, r);
                      return false; }
            std::copy(rec.begin() + out->dof_count, rec.end(),
                      out->pose[s].begin() + static_cast<size_t>(r) * pose_per_robot);
        }
    }
    return true;
}

// ---- a REGULAR rows x cols grid of tiles (the Isaac-Lab cover layout) -------
// Centres the grid on the origin and tiles EDGE-TO-EDGE (kTilePitch == tile full
// width) so the sub-terrains read as one continuous field. Terrain TYPE follows a
// regular repeating checkerboard over {PyramidStairs, InvertedPyramid, RandomBoxes,
// Flat} so neighbours differ (the cover's patchwork), and difficulty climbs with
// the row so the back of the field is the hardest (a visible curriculum spread,
// owner's 0.08-0.15 m band). The first `robots` cells get a dog; extra grid cells
// past `robots` are not produced (robots defaults to rows*cols => a full grid).
std::vector<TileSpec> MakeGrid(uint32_t robots, uint32_t rows, uint32_t cols) {
    std::vector<TileSpec> tiles(robots);
    if (cols == 0u) cols = 1u;
    if (rows == 0u) rows = 1u;
    const float x0 = -0.5f * static_cast<float>(cols - 1u) * kTilePitch;
    const float y0 = -0.5f * static_cast<float>(rows - 1u) * kTilePitch;
    // 2x2 checkerboard of the 4 terrain types over (row parity, col parity) so any
    // 2x2 block shows all four kinds (exactly the cover's mixed field).
    const uint32_t type_grid[2][2] = {
        {terrain::kTerrainPyramidStairs, terrain::kTerrainRandomBoxes},
        {terrain::kTerrainInvertedPyramid, terrain::kTerrainFlat}};
    for (uint32_t r = 0; r < robots; ++r) {
        const uint32_t c = r % cols, rw = r / cols;
        tiles[r].offset_x = x0 + static_cast<float>(c) * kTilePitch;
        tiles[r].offset_y = y0 + static_cast<float>(rw) * kTilePitch;
        tiles[r].terrain_type = type_grid[rw % 2u][c % 2u];
        // Difficulty rises with the row index (front easy -> back hard), 0.08-0.15.
        const float t = (rows > 1u) ? static_cast<float>(rw) / static_cast<float>(rows - 1u)
                                    : 0.0f;
        tiles[r].difficulty = 0.08f + 0.07f * t;
    }
    return tiles;
}

// ---- PLACEHOLDER: replicate the single-robot GO2W to an N-robot GO2T --------
// Each robot replays the SAME dumped flat walk (offset to its tile at publish
// time). This exercises the full render infra (terrain mesh + N-instance + orbit
// cam + haze + per-foot shadows) without the real terrain rollout.
NTrajectory MakePlaceholder(const SingleTrajectory& src, uint32_t robots,
                            uint32_t rows, uint32_t cols) {
    NTrajectory t;
    t.num_robots = robots;
    t.num_steps  = src.num_steps;
    t.dof_count  = src.dof_count;
    t.link_count = src.link_count;
    t.pose_floats = src.pose_floats;
    t.dt = src.dt; t.ctrl_dt = src.ctrl_dt;
    t.tiles = MakeGrid(robots, rows, cols);
    // Per-robot start-phase: a spread of offsets across the loop so the pack is
    // desynced. A deterministic pseudo-random-ish stride (golden-ratio step) gives
    // a well-distributed set of phases without two dogs sharing one.
    t.phase.resize(robots);
    for (uint32_t r = 0; r < robots; ++r) {
        const uint32_t off = (src.num_steps > 0u)
            ? static_cast<uint32_t>((static_cast<uint64_t>(r) * 6151u) % src.num_steps)
            : 0u;
        t.phase[r] = off;
    }
    const size_t pose_per_robot = static_cast<size_t>(src.link_count) * src.pose_floats;
    t.pose.resize(src.num_steps);
    for (uint32_t s = 0; s < src.num_steps; ++s) {
        t.pose[s].resize(static_cast<size_t>(robots) * pose_per_robot);
        for (uint32_t r = 0; r < robots; ++r) {
            std::copy(src.pose[s].begin(), src.pose[s].end(),
                      t.pose[s].begin() + static_cast<size_t>(r) * pose_per_robot);
        }
    }
    return t;
}

// ---------------------------------------------------------------------------
// TerrainTileMesh -- TESSELLATE one tile's surface by sampling the SHARED
// physics heightfield (anti-drift). For each grid vertex at LOCAL (x,y) in
// [-half, +half], z = SampleTerrainHeight(type, x, y, ScaleTerrainDifficulty(
// preset, difficulty)). Per-quad geometric normals (flat per triangle -> crisp
// stair risers). The mesh is in the TILE-LOCAL frame; the caller translates it to
// the tile centre. resolution = vertices per axis; spacing = 2*half/(res-1) must
// be <= ~0.05 m so a 0.30 m horizontal step reads as a sharp edge (res ~ 65 for
// half=1.6 -> spacing 0.050 m). The same SampleTerrainHeight the feet rest on ->
// the rendered surface can never drift from the physics surface.
// ---------------------------------------------------------------------------
render::MeshGeometry TessellateTile(uint32_t type, const terrain::TerrainParams& base,
                                    float difficulty, float half, uint32_t res) {
    render::MeshGeometry m;
    const terrain::TerrainParams p = terrain::ScaleTerrainDifficulty(base, difficulty);
    const uint32_t n = std::max(2u, res);
    const float span = 2.0f * half;
    const float step = span / static_cast<float>(n - 1u);

    // Flat shading: emit 4 independent vertices per quad (2 triangles) so each
    // riser/tread has its own normal and stair edges stay crisp (no smoothing).
    auto h = [&](float x, float y) { return terrain::SampleTerrainHeight(type, x, y, p); };
    auto push_v = [&](float x, float y, float z, const Vec3& nrm) {
        m.positions.push_back(x); m.positions.push_back(y); m.positions.push_back(z);
        m.normals.push_back(nrm.x); m.normals.push_back(nrm.y); m.normals.push_back(nrm.z);
    };
    for (uint32_t j = 0; j + 1 < n; ++j) {
        for (uint32_t i = 0; i + 1 < n; ++i) {
            const float x0 = -half + static_cast<float>(i) * step;
            const float x1 = -half + static_cast<float>(i + 1u) * step;
            const float y0 = -half + static_cast<float>(j) * step;
            const float y1 = -half + static_cast<float>(j + 1u) * step;
            const Vec3 a{x0, y0, h(x0, y0)};
            const Vec3 b{x1, y0, h(x1, y0)};
            const Vec3 c{x1, y1, h(x1, y1)};
            const Vec3 dd{x0, y1, h(x0, y1)};
            // Two triangles a,b,c and a,c,d; flat geometric normals (CCW -> +up).
            const uint32_t base_idx = static_cast<uint32_t>(m.positions.size() / 3u);
            Vec3 n1 = (b - a).Cross(c - a); if (n1.Length() > 1e-12f) n1 = n1.Normalized(); else n1 = {0,0,1};
            Vec3 n2 = (c - a).Cross(dd - a); if (n2.Length() > 1e-12f) n2 = n2.Normalized(); else n2 = {0,0,1};
            if (n1.z < 0.0f) n1 = -n1;
            if (n2.z < 0.0f) n2 = -n2;
            push_v(a.x, a.y, a.z, n1); push_v(b.x, b.y, b.z, n1); push_v(c.x, c.y, c.z, n1);
            push_v(a.x, a.y, a.z, n2); push_v(c.x, c.y, c.z, n2); push_v(dd.x, dd.y, dd.z, n2);
            for (uint32_t k = 0; k < 6u; ++k) m.indices.push_back(base_idx + k);
        }
    }
    return m;
}

bool WritePpm(const render::VulkanOffscreenReport& rep, const std::string& path) {
    if (rep.pixels.empty() || rep.width == 0u || rep.height == 0u) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    char hdr[64];
    const int hn = std::snprintf(hdr, sizeof(hdr), "P6\n%u %u\n255\n", rep.width, rep.height);
    bool ok = (hn > 0) &&
              (std::fwrite(hdr, 1, static_cast<size_t>(hn), f) == static_cast<size_t>(hn));
    std::vector<unsigned char> rgb(static_cast<size_t>(rep.width) * 3u);
    for (uint32_t y = 0; ok && y < rep.height; ++y) {
        for (uint32_t x = 0; x < rep.width; ++x) {
            const auto& p = rep.pixels[static_cast<size_t>(y) * rep.width + x];
            rgb[x * 3 + 0] = p.r; rgb[x * 3 + 1] = p.g; rgb[x * 3 + 2] = p.b;
        }
        if (std::fwrite(rgb.data(), 1, rgb.size(), f) != rgb.size()) ok = false;
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

// ---- a PBR material maker (mirrors go2_walk_video) -------------------------
nuka::scene::RenderMaterial Mk(float r, float g, float b, float metallic, float rough) {
    nuka::scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough;
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    if (!std::filesystem::exists(kNksPath)) {
        std::fprintf(stderr, "[go2_terrain] missing scene %s\n", kNksPath);
        return 2;
    }

    // ---- 1. ASSEMBLE THE N-ROBOT TRAJECTORY (real GO2T or placeholder) ------
    NTrajectory traj;
    if (!args.placeholder && !args.bin.empty()) {
        // REAL-ROLLOUT SEAM: a trained-terrain GO2T .bin.
        if (!LoadGo2T(args.bin, &traj)) return 3;
        std::printf("[go2_terrain] REAL GO2T: robots=%u steps=%u links=%u (from %s)\n",
                    traj.num_robots, traj.num_steps, traj.link_count, args.bin.c_str());
    } else {
        // PLACEHOLDER: replicate the single-robot flat walk to N tiled robots.
        const std::string src_path = !args.bin.empty() ? args.bin : kGo2WBin;
        if (!std::filesystem::exists(src_path)) {
            std::fprintf(stderr, "[go2_terrain] missing single-robot GO2W %s "
                         "(run go2_dump_walk_trajectory.py first)\n", src_path.c_str());
            return 2;
        }
        SingleTrajectory src;
        if (!LoadGo2W(src_path, &src)) return 3;
        traj = MakePlaceholder(src, args.robots, args.rows, args.cols);
        std::printf("[go2_terrain] PLACEHOLDER: replicated %u-step single walk -> "
                    "%u tiled robots (src %s)\n", src.num_steps, traj.num_robots,
                    src_path.c_str());
    }
    std::printf("[go2_terrain] trajectory: %u robots x %u sub-steps, links=%u "
                "pose_floats=%u dt=%.4f (sim %.2fs)\n", traj.num_robots, traj.num_steps,
                traj.link_count, traj.pose_floats, traj.dt, traj.num_steps * traj.dt);
    for (uint32_t r = 0; r < traj.num_robots; ++r) {
        const TileSpec& t = traj.tiles[r];
        const char* names[4] = {"Flat", "PyramidStairs", "InvertedPyramid", "RandomBoxes"};
        std::printf("[go2_terrain]   tile[%u] @(%.2f,%.2f) type=%s diff=%.3f\n", r,
                    t.offset_x, t.offset_y,
                    names[t.terrain_type < 4u ? t.terrain_type : 0u], t.difficulty);
    }

    // ---- 2. FROZEN go2 beauty scene -> ONE RenderWorld TEMPLATE -------------
    // We cook the go2 ONCE, build ONE RenderWorld (33 visual instances), recover
    // the shape->link binding, then REPLICATE the instance list N times (one block
    // per robot) sharing the SAME interned meshes. Per-robot publish offsets each
    // robot's poses to its tile origin.
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
    render::RenderWorld tmpl = render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
    if (tmpl.instances.empty()) {
        std::fprintf(stderr, "[go2_terrain] FATAL: BuildRenderWorld produced 0 instances\n");
        return 4;
    }
    const uint32_t per_robot_inst = tmpl.InstanceCount();

    // Recover each template instance's trajectory link + geom-local (the go2 cook
    // collapses visual geoms onto body 0; the real owning link is the SceneIR
    // shape's body_id, identity-mapped to the trajectory link). Mirrors go2_walk_video.
    std::vector<uint32_t>  tmpl_link(tmpl.instances.size(), 0u);
    std::vector<Transform> tmpl_vlocal(tmpl.instances.size(), Transform::Identity());
    uint32_t n_rebound = 0u, n_oob = 0u;
    for (size_t i = 0; i < tmpl.instances.size(); ++i) {
        const nuka::scene::CookedRef* ref = cooked.scene_map.RefOf(tmpl.instances[i].entity);
        if (ref && ref->shape_row != nuka::scene::SceneMap::kNoRow &&
            ref->shape_row < scene.Shapes().size()) {
            const auto& sh = scene.GetShape(ref->shape_row);
            const uint32_t link = static_cast<uint32_t>(sh.body_id);
            if (link < traj.link_count) {
                tmpl_link[i] = link; tmpl_vlocal[i] = sh.local_transform; ++n_rebound;
            } else { ++n_oob; }
        }
    }
    if (n_oob > 0u) {
        std::fprintf(stderr, "[go2_terrain] FATAL: %u instances reference link >= %u\n",
                     n_oob, traj.link_count);
        return 5;
    }
    std::printf("[go2_terrain] template: %u instances (rebound_to_link=%u), "
                "%u meshes\n", per_robot_inst, n_rebound, tmpl.meshes.Count());

    // ---- 3. PREMIUM MATERIAL OVERRIDE (the go2_walk_video studio look) ------
    render::RenderWorld rw;
    rw.meshes = std::move(tmpl.meshes);   // shared interned go2 meshes
    const uint32_t kMatShell = 0u, kMatMetal = 1u, kMatFoot = 2u, kMatAccent = 3u;
    rw.materials.push_back(Mk(0.016f, 0.017f, 0.020f, 0.02f, 0.62f));  // 0 shell charcoal
    rw.materials.push_back(Mk(0.040f, 0.043f, 0.052f, 0.90f, 0.32f));  // 1 dark gunmetal
    rw.materials.push_back(Mk(0.010f, 0.010f, 0.012f, 0.02f, 0.85f));  // 2 matte black rubber
    rw.materials.push_back(Mk(0.10f,  0.115f, 0.15f,  0.70f, 0.28f));  // 3 cool steel accent
    // 4 terrain material: light-grey CONCRETE matching the floor disc so the field
    // reads as one continuous concrete surface (the Isaac-Lab cover). Matte; the sun
    // + shadow map (not albedo) carve the geometry, so a single mid-light grey is
    // right -- the N.L on risers + cast shadows do the shaping.
    const uint32_t kMatTerrain = static_cast<uint32_t>(rw.materials.size());
    rw.materials.push_back(Mk(0.380f, 0.392f, 0.410f, 0.02f, 0.86f));
    rw.default_material_id = kMatShell;

    auto link_role_material = [&](uint32_t link, const Transform& vlocal) -> uint32_t {
        if (link == 0u) return kMatShell;
        const uint32_t k = ((link - 1u) % 3u);
        if (k == 0u) return kMatMetal;
        if (k == 1u) return kMatShell;
        return (vlocal.position.z < -0.15f) ? kMatFoot : kMatMetal;
    };

    // ---- 4. REPLICATE the template instances to N robots + add terrain tiles ----
    // Per-robot instance metadata: which trajectory link drives it + its geom-local
    // + which robot it belongs to (so publish offsets it to the right tile).
    struct InstMeta { uint32_t robot; uint32_t link; Transform vlocal; };
    std::vector<InstMeta> inst_meta;
    inst_meta.reserve(static_cast<size_t>(traj.num_robots) * per_robot_inst);
    // Find the single accent piece (the last base-link instance) per the walk tool.
    int accent_local = -1;
    for (size_t i = 0; i < tmpl.instances.size(); ++i)
        if (tmpl_link[i] == 0u) accent_local = static_cast<int>(i);

    for (uint32_t r = 0; r < traj.num_robots; ++r) {
        for (size_t i = 0; i < tmpl.instances.size(); ++i) {
            render::RenderInstance inst = tmpl.instances[i];  // copies mesh_id, entity
            inst.pose_source = render::PoseSource{};          // we drive world_xform directly
            inst.render_material_id =
                (static_cast<int>(i) == accent_local) ? kMatAccent
                                                      : link_role_material(tmpl_link[i], tmpl_vlocal[i]);
            rw.instances.push_back(inst);
            inst_meta.push_back(InstMeta{r, tmpl_link[i], tmpl_vlocal[i]});
        }
    }
    const uint32_t robot_inst_end = rw.InstanceCount();  // terrain meshes appended after

    // ---- terrain tile meshes: tessellate ONE per robot (shared header h(x,y)) ---
    // Resolution: spacing <= 0.05 m -> res = ceil(2*half / 0.05) + 1.
    const terrain::TerrainParams preset = TilePreset();
    const uint32_t tile_res =
        static_cast<uint32_t>(std::ceil(2.0f * kTileHalf / 0.05f)) + 1u;  // ~65
    std::printf("[go2_terrain] terrain tessellation: %ux%u verts/tile "
                "(spacing %.3f m, half=%.2f m)\n", tile_res, tile_res,
                2.0f * kTileHalf / static_cast<float>(tile_res - 1u), kTileHalf);
    std::vector<uint32_t> tile_inst_index(traj.num_robots, render::kNoId);
    for (uint32_t r = 0; r < traj.num_robots; ++r) {
        const TileSpec& t = traj.tiles[r];
        char key[96];
        std::snprintf(key, sizeof(key), "terrain:%u:%.4f:%.2f:%u",
                      t.terrain_type, t.difficulty, kTileHalf, tile_res);
        const uint32_t mesh_id = rw.meshes.InternPrimitive(key, [&]() {
            return TessellateTile(t.terrain_type, preset, t.difficulty, kTileHalf, tile_res);
        });
        render::RenderInstance ti;
        ti.mesh_id = mesh_id;
        ti.render_material_id = kMatTerrain;
        Transform xf = Transform::Identity();
        xf.position = {t.offset_x, t.offset_y, 0.0f};  // tile centre at studio z=0
        ti.world_xform = xf;
        ti.pose_source = render::PoseSource{};
        tile_inst_index[r] = rw.InstanceCount();
        rw.instances.push_back(ti);
    }
    std::printf("[go2_terrain] RenderWorld: %u robot instances + %u terrain tiles = %u total\n",
                robot_inst_end, traj.num_robots, rw.InstanceCount());

    // ---- 5. PER-FRAME PUBLISH: offset each robot's poses to its tile ----------
    // robot r's link pose (per-env frame) -> world = tile_off + grounding o link_pose
    // o geom. Three corrections make the PLACEHOLDER flat walk read as a dog walking
    // ON its terrain tile (so the look is cover-quality even before the real rollout):
    //
    //   (a) RE-CENTER the forward drift. The dumped flat walk strides ~3.9 m in +x
    //       over the clip, which would carry the dog off a 1.6 m-half tile. We
    //       subtract the base link's xy drift from its step-0 position so the dog
    //       WALKS IN PLACE near its tile centre (gait fully preserved -- only the
    //       net translation is removed). The real terrain rollout sets recenter to
    //       false (its xy already belongs on the tile).
    //   (b) GROUND each robot onto the LOCAL terrain surface. The flat walk's feet
    //       rest near z=0; we lift the whole robot by the terrain height under its
    //       (recentred) base xy so the feet land ON the stairs/pit/boxes instead of
    //       floating over them. ONE shared h(x,y) -> render mesh + grounding agree.
    //   (c) The tile (offset_x, offset_y) places the per-env walk on its grid tile.
    //
    // The grounding/recenter is a RENDER-SIDE placeholder dressing; the real rollout
    // (--bin, recenter=false, ground_lift=0) replays its own already-on-terrain z.
    const bool kRecenter = args.placeholder;   // real GO2T keeps its own xy/z
    const bool kGroundLift = args.placeholder;
    // Per-robot step-0 base xy (link 0) for the re-centring subtraction.
    std::vector<Vec3> base0(traj.num_robots, Vec3{0.0f, 0.0f, 0.0f});
    for (uint32_t r = 0; r < traj.num_robots; ++r)
        base0[r] = traj.PhasedPose(0u, r, 0u).position;

    auto publish = [&](uint32_t step) {
        for (uint32_t r = 0; r < traj.num_robots; ++r) {
            const TileSpec& t = traj.tiles[r];
            const terrain::TerrainParams tp =
                terrain::ScaleTerrainDifficulty(preset, t.difficulty);
            // (a) re-centre: remove the net xy drift from step 0 (gait preserved).
            const Vec3 cur_base = traj.PhasedPose(step, r, 0u).position;
            Vec3 recenter{0.0f, 0.0f, 0.0f};
            if (kRecenter) { recenter.x = -(cur_base.x - base0[r].x);
                             recenter.y = -(cur_base.y - base0[r].y); }
            // (b) ground lift: terrain height under the recentred base xy (local frame).
            float z_lift = 0.0f;
            if (kGroundLift) {
                const float bx = cur_base.x + recenter.x;  // tile-local base xy
                const float by = cur_base.y + recenter.y;
                z_lift = terrain::SampleTerrainHeight(t.terrain_type, bx, by, tp);
            }
            const Vec3 robot_off{t.offset_x + recenter.x,
                                 t.offset_y + recenter.y, z_lift};
            for (uint32_t i = 0; i < per_robot_inst; ++i) {
                const size_t gi = static_cast<size_t>(r) * per_robot_inst + i;
                const Transform fk = traj.PhasedPose(step, r, inst_meta[gi].link);
                Transform world = fk * inst_meta[gi].vlocal;
                world.position += robot_off;
                rw.instances[gi].world_xform = world;
            }
        }
    };

    // ---- 6. PER-FOOT CONTACT SHADOWS on the LOCAL terrain surface -------------
    // For each robot's 4 feet, find the foot's lowest world vertex; place a shadow
    // decal at its xy and fade strength by the foot's height ABOVE the LOCAL terrain
    // surface (sampled from the SAME h(x,y), in the tile-local frame). Planted ->
    // full; lifted -> faint. This is the non-flat-terrain version of go2_walk_video's
    // per-foot fade. Foot instances = the kMatFoot material within each robot block.
    std::vector<std::vector<size_t>> robot_feet(traj.num_robots);  // global instance idx
    for (size_t gi = 0; gi < robot_inst_end; ++gi) {
        if (rw.instances[gi].render_material_id == kMatFoot)
            robot_feet[inst_meta[gi].robot].push_back(gi);
    }
    auto set_contact_shadows = [&](render::RasterOptions& o) {
        o.contact_points.clear();
        for (uint32_t r = 0; r < traj.num_robots; ++r) {
            const TileSpec& t = traj.tiles[r];
            const terrain::TerrainParams tp =
                terrain::ScaleTerrainDifficulty(preset, t.difficulty);
            for (size_t fi : robot_feet[r]) {
                const auto& geo = rw.meshes.Geometry(rw.instances[fi].mesh_id);
                const Transform& wx = rw.instances[fi].world_xform;
                float min_z = 1e9f, fx = 0.0f, fy = 0.0f;
                for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
                    const Vec3 wp = wx.TransformPoint(
                        Vec3{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]});
                    if (wp.z < min_z) { min_z = wp.z; fx = wp.x; fy = wp.y; }
                }
                // Local terrain height under the foot (subtract the tile offset to
                // sample in the tile-local frame the heightfield is defined in).
                const float lx = fx - t.offset_x, ly = fy - t.offset_y;
                const float surf_z = terrain::SampleTerrainHeight(t.terrain_type, lx, ly, tp);
                const float above = min_z - surf_z;        // foot height over local surface
                const float z_full = 0.02f, z_zero = 0.12f;
                float frac = (z_zero - above) / (z_zero - z_full);
                frac = frac < 0.0f ? 0.0f : (frac > 1.0f ? 1.0f : frac);
                render::RasterOptions::ContactPoint cp;
                cp.x = fx; cp.y = fy;
                cp.radius   = 0.10f + 0.04f * frac;
                cp.strength = 0.70f * (0.12f + 0.88f * frac);
                o.contact_points.push_back(cp);
            }
        }
    };

    // ---- 7. RENDERER + studio PBR options + haze -----------------------------
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[go2_terrain] no Vulkan device: %s\n", e.what());
        return 6;
    }
    std::printf("[go2_terrain] renderer ICD: %s\n", renderer->DeviceName().c_str());

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = true;
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 42.0f;
    // Concrete floor matching the terrain tiles (one continuous grey field, like the
    // Isaac-Lab cover). The flat clear is replaced by a cool vertical SKY gradient.
    opts.background = {178, 188, 202, 255};  // fallback flat clear (sky_gradient on)
    opts.ground_color[0] = 0.37f; opts.ground_color[1] = 0.382f; opts.ground_color[2] = 0.400f;
    opts.contact_shadow_strength = 0.0f;  // explicit per-foot points below

    // ---- CINEMATIC SUN + DIRECTIONAL SHADOW MAP (the Isaac-Lab look) ----------
    // A single strong low-angle sun carves the stairs (vertical risers darken) and
    // casts crisp shadows into the grooves + from the dogs via the shadow map.
    opts.use_sun_light = true;
    // Low-ish sun from the front-left so shadows rake ACROSS the field toward camera.
    // Intensity tuned against the ACES tonemap so a 0.5 concrete albedo stays a
    // readable MID-grey (not blown to white): a modest key + LOW ambient gives the
    // wide tread-vs-riser contrast that carves the stairs.
    opts.sun_direction[0] = 0.56f; opts.sun_direction[1] = -0.46f; opts.sun_direction[2] = 0.33f;
    opts.sun_color[0] = 2.05f; opts.sun_color[1] = 1.97f; opts.sun_color[2] = 1.78f;
    opts.sun_ambient_sky[0] = 0.105f; opts.sun_ambient_sky[1] = 0.122f; opts.sun_ambient_sky[2] = 0.158f;
    opts.sun_ambient_ground[0] = 0.066f; opts.sun_ambient_ground[1] = 0.066f; opts.sun_ambient_ground[2] = 0.072f;
    opts.shadow_strength = args.no_shadows ? 0.0f : 0.96f;  // crisp near-black grooves
    opts.shadow_map_size = 2560u;
    opts.shadow_bias = 0.0018f;

    // ---- COOL VERTICAL SKY GRADIENT ------------------------------------------
    opts.sky_gradient = true;
    opts.sky_top[0] = 0.55f; opts.sky_top[1] = 0.62f; opts.sky_top[2] = 0.72f;     // cooler zenith
    opts.sky_bottom[0] = 0.86f; opts.sky_bottom[1] = 0.89f; opts.sky_bottom[2] = 0.93f; // bright horizon

    // Subtle FAR-FIELD haze only (foreground dogs stay crisp): low density + the
    // sky-horizon colour so distant tiles melt into the sky, not a white wall.
    opts.fog_density = args.fog;
    opts.fog_color[0] = 0.84f; opts.fog_color[1] = 0.88f; opts.fog_color[2] = 0.93f;
    if (const char* fv = std::getenv("NK_CAM_FOV"))
        opts.camera_fov_degrees = static_cast<float>(std::atof(fv));

    // ---- grid AABB (xy) for the orbit framing -------------------------------
    float gx_min = 1e9f, gx_max = -1e9f, gy_min = 1e9f, gy_max = -1e9f;
    for (const TileSpec& t : traj.tiles) {
        gx_min = std::min(gx_min, t.offset_x - kTileHalf);
        gx_max = std::max(gx_max, t.offset_x + kTileHalf);
        gy_min = std::min(gy_min, t.offset_y - kTileHalf);
        gy_max = std::max(gy_max, t.offset_y + kTileHalf);
    }
    const Vec3 grid_ctr{0.5f * (gx_min + gx_max), 0.5f * (gy_min + gy_max), 0.0f};
    const float grid_radius = 0.5f * std::sqrt((gx_max - gx_min) * (gx_max - gx_min) +
                                               (gy_max - gy_min) * (gy_max - gy_min));

    auto smoothstep = [](float e0, float e1, float x) {
        float t = (x - e0) / (e1 - e0);
        t = std::max(0.0f, std::min(1.0f, t));
        return t * t * (3.0f - 2.0f * t);
    };
    auto lerp = [](float a, float b, float u) { return a + (b - a) * u; };

    // ---- ROTATING ORBIT CAMERA (required: circle the grid to showcase gait) ---
    // The target is the grid centre (a touch above the floor). The eye orbits in
    // azimuth phi over the clip (orbit_turns full turns), at a gentle elevation
    // that eases up then settles, on a radius that keeps the WHOLE grid AABB in
    // frame (radius from grid_radius + the fov half-angle). Smooth eased azimuth so
    // the sweep never snaps -- the dogs are shown from many angles, gait readable.
    auto aim_camera = [&](uint32_t step) {
        const float t = (traj.num_steps > 1u)
            ? static_cast<float>(step) / static_cast<float>(traj.num_steps - 1u) : 0.0f;
        const float e = smoothstep(0.0f, 1.0f, t);
        // Azimuth sweep: start on the front-left 3/4, circle by orbit_turns turns.
        const float phi = -0.55f * kPi + 2.0f * kPi * args.orbit_turns * e;
        // Elevation: a low-ish hero crane (15-22 deg) -- enough to read the stepped
        // terrain in 3/4 but never a top-down plan view of the grid.
        const float elev_deg = lerp(15.0f, 22.0f, e) - 3.0f * std::sin(e * kPi);
        const float elev = elev_deg * kPi / 180.0f;
        // Radius: fill the frame. Fit the grid's half-diagonal into the vertical fov
        // with a modest margin (the eye tilts down, so a tighter fit still frames
        // the whole grid). std::cos(elev) un-projects the slant range to ground reach.
        const float fov_y = opts.camera_fov_degrees * kPi / 180.0f;
        const float fit_r = (grid_radius * 0.72f) / std::tan(fov_y * 0.5f);
        const float radius = std::max(fit_r, grid_radius * 0.95f);
        const float ch = std::cos(elev), sh = std::sin(elev);
        const float dx = std::sin(phi), dy = -std::cos(phi);
        opts.camera_eye = {grid_ctr.x + radius * ch * dx,
                           grid_ctr.y + radius * ch * dy,
                           grid_ctr.z + radius * sh + 0.30f};
        opts.camera_target = {grid_ctr.x, grid_ctr.y, grid_ctr.z + 0.30f};
        // Far clip generous enough that the haze (not the clip) fades the horizon.
        opts.camera_far = radius * 6.0f + grid_radius * 8.0f + 50.0f;
    };

    // ---- 8. PROBE: render 1 mid-clip frame, exit ----------------------------
    if (args.probe) {
        const uint32_t mid = traj.num_steps / 2u;
        publish(mid);
        set_contact_shadows(opts);
        aim_camera(mid);
        render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
        std::filesystem::create_directories(args.out_dir);
        const std::string p = args.out_dir + "/probe_frame.ppm";
        const bool ok = WritePpm(rep, p);
        std::printf("[go2_terrain] PROBE: instances=%u meshes=%u non_bg=%zu (%ux%u) "
                    "eye=(%.2f,%.2f,%.2f) -> %s (%s)\n", rw.InstanceCount(),
                    rw.meshes.Count(), rep.non_background_pixel_count, rep.width, rep.height,
                    opts.camera_eye.x, opts.camera_eye.y, opts.camera_eye.z,
                    p.c_str(), ok ? "OK" : "WRITE-FAIL");
        return ok ? 0 : 7;
    }

    // ---- 9. REPLAY + RENDER LOOP --------------------------------------------
    std::filesystem::create_directories(args.out_dir);
    uint32_t written = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    for (uint32_t s = 0; s < traj.num_steps; s += args.stride) {
        if (args.frames != 0u && written >= args.frames) break;
        publish(s);
        set_contact_shadows(opts);
        aim_camera(s);
        render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        const std::string path = args.out_dir + "/" + name;
        if (!WritePpm(rep, path)) {
            std::fprintf(stderr, "[go2_terrain] write fail @ sub-step %u\n", s);
            return 7;
        }
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        ++written;
        if ((written % 30u) == 1u)
            std::printf("[go2_terrain] frame %u (sub-step %u/%u) non_bg=%zu eye=(%.2f,%.2f,%.2f)\n",
                        written - 1u, s, traj.num_steps, rep.non_background_pixel_count,
                        opts.camera_eye.x, opts.camera_eye.y, opts.camera_eye.z);
    }
    std::printf("[go2_terrain] DONE: wrote %u frames to %s (first non_bg=%zu, last=%zu)\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg);
    return 0;
}
