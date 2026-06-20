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
#include <chrono>
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
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"
#include "scene/terrain/heightfield_sample.hpp"

namespace {

namespace render = nuka::render;
namespace rt = nuka::rt;
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
// Shared by the render tessellator and the PHYSICS cook -- the SINGLE place the
// horizontal geometry is defined. THESE VALUES MUST MATCH the training-time terrain
// cook geometry (examples/training/go2_ppo_cfg.yaml config.env_config.terrain:
// step_height 0.15, step_width 0.3, platform_width 1.0, grid_width 0.45,
// grid_height_max 0.15, ground 0) so the rendered surface is byte-identical to the
// one the dog's feet actually rested on during the trained rollout. difficulty (the
// RAW curriculum value in [0,1], carried in the GO2T tile record) scales only the
// vertical magnitude (step_height / grid_height_max) via ScaleTerrainDifficulty,
// EXACTLY as the foot-contact kernel does -> realized step rise = 0.15 m *
// difficulty (0.5 -> 0.075 m, 0.9 -> 0.135 m, owner's 0.08-0.15 band). ground_height
// = 0 puts every tile's reference floor at the studio z=0. The tessellator spacing
// (<=0.05 m, below) keeps the 0.30 m step edges sharp.
//   NB the placeholder (--placeholder, no real rollout) replicates a FLAT walk and
// uses MakeGrid difficulties in the same [0.08,0.15]-realized band; the dressing
// recenters+ground-lifts it onto these tiles. The real GO2T (--bin) carries its own
// per-dog (type, difficulty) + on-terrain z (recenter/lift OFF).
// The render terrain feature parameters (matching the physics cook knobs). These
// are PARAMETERS, not a type: a tile's integer code selects an amplitude sign /
// which feature is expressed, all routed through ONE parametric generator.
struct TilePreset {
    float step_height = 0.15f;    // MAX ring rise (* difficulty).
    float step_width = 0.30f;     // ring width.
    float platform_width = 1.00f; // central platform edge.
    float grid_width = 0.45f;     // bump footprint.
    float grid_height_max = 0.15f;// MAX bump rise (* difficulty).
};

// Tile integer codes carried in the trajectory (0=flat,1=stairs,2=pit,3=boxes).
// They are mapped to a parametric TerrainGenConfig -- the engine has no terrain
// types, only this grid generator (MuJoCo hfield + Newton create_heightfield model).
constexpr uint32_t kTileFlat = 0u, kTileStairs = 1u, kTilePit = 2u, kTileBoxes = 3u;
constexpr uint32_t kTileComposite = 4u;

// Build a parametric HeightField for a tile, centred at the world origin over
// [-half,+half], at `res` vertices/axis. difficulty scales the vertical magnitude.
terrain::HeightField TileHeightField(uint32_t code, const TilePreset& base,
                                     float difficulty, float half, uint32_t res) {
    const uint32_t n = std::max(2u, res);
    const float cell = 2.0f * half / static_cast<float>(n - 1u);
    terrain::TerrainGenConfig cfg;
    cfg.nrow = n;
    cfg.ncol = n;
    cfg.cell_x = cell;
    cfg.cell_y = cell;
    cfg.origin = Vec3{-half, -half, 0.0f};
    cfg.base_z = 0.0f;
    if (code == kTileStairs || code == kTilePit) {
        const float sign = (code == kTilePit) ? -1.0f : 1.0f;
        cfg.ring_rise = sign * base.step_height * difficulty;
        cfg.ring_width = base.step_width;
        cfg.ring_platform = base.platform_width;
        cfg.ring_count = 8u;
    } else if (code == kTileBoxes) {
        cfg.bump_height = base.grid_height_max * difficulty;
        cfg.bump_cell = base.grid_width;
    }  // kTileFlat: every amplitude 0.
    terrain::HeightField hf;
    terrain::GenerateHeightField(cfg, hf);
    return hf;
}

// Build the WORLD composite HeightField over [xmin,xmax]x[ymin,ymax] at `spacing`.
// The tiled multi-feature field (feature_cell > 0) -- the 20-dog complex scene as
// pure parameters, no per-type tiles.
terrain::HeightField CompositeHeightField(const TilePreset& base, float difficulty,
                                          float xmin, float xmax, float ymin,
                                          float ymax, float spacing) {
    const float sx = std::max(1e-3f, spacing);
    const uint32_t nx =
        std::max(2u, static_cast<uint32_t>(std::ceil((xmax - xmin) / sx)) + 1u);
    const uint32_t ny =
        std::max(2u, static_cast<uint32_t>(std::ceil((ymax - ymin) / sx)) + 1u);
    terrain::TerrainGenConfig cfg;
    cfg.nrow = ny;
    cfg.ncol = nx;
    cfg.cell_x = (xmax - xmin) / static_cast<float>(nx - 1u);
    cfg.cell_y = (ymax - ymin) / static_cast<float>(ny - 1u);
    cfg.origin = Vec3{xmin, ymin, 0.0f};
    cfg.base_z = 0.0f;
    cfg.ring_rise = base.step_height * difficulty;
    cfg.ring_width = base.step_width;
    cfg.ring_platform = base.platform_width;
    cfg.ring_count = 8u;
    cfg.bump_height = base.grid_height_max * difficulty;
    cfg.bump_cell = base.grid_width;
    cfg.feature_cell = 5.0f;
    cfg.feature_margin = 0.6f;
    terrain::HeightField hf;
    terrain::GenerateHeightField(cfg, hf);
    return hf;
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
    bool        gpu = false;          // CUDA ray-traced backend instead of lavapipe
    bool        beauty = true;        // GPU beauty path (MSAA+soft-shadow+AO+GI); --flat-gpu off
    uint32_t    samples = 16u;        // beauty MSAA samples per pixel
    bool        no_shadows = false;   // disable the directional shadow map (fallback)
    uint32_t    shadow_map = 2560u;   // directional shadow-map resolution. The single
                                      // biggest per-frame cost on lavapipe-CPU; drop to
                                      // 1280/1536 to roughly halve render time for a
                                      // first-cut clip (edges soften but the look holds).
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
        else if (s == "--shadow-map") a.shadow_map = std::max(256u, next_u(a.shadow_map));
        else if (s == "--probe") a.probe = true;
        else if (s == "--gpu") a.gpu = true;
        else if (s == "--flat-gpu") { a.gpu = true; a.beauty = false; }
        else if (s == "--samples") a.samples = std::max(1u, next_u(a.samples));
        else if (s == "--no-shadows") a.no_shadows = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
    }
    // Default robot count = a full rows*cols grid (the Isaac-Lab field) unless the
    // caller explicitly set --robots.
    if (a.robots == 0u) a.robots = a.rows * a.cols;
    if (const char* g = std::getenv("NK_GPU_RENDER"))
        a.gpu = a.gpu || (g[0] == '1');
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
        {kTileStairs, kTileBoxes},
        {kTilePit, kTileFlat}};
    for (uint32_t r = 0; r < robots; ++r) {
        const uint32_t c = r % cols, rw = r / cols;
        tiles[r].offset_x = x0 + static_cast<float>(c) * kTilePitch;
        tiles[r].offset_y = y0 + static_cast<float>(rw) * kTilePitch;
        tiles[r].terrain_type = type_grid[rw % 2u][c % 2u];
        // Difficulty is the RAW curriculum value in [0,1] (same convention as the
        // physics cook + the real GO2T tile record): realized rise = step_height
        // (0.15) * difficulty. Front row easy -> back row hard over [0.5, 0.9] so
        // the placeholder field reads as 0.075-0.135 m steps (owner's 0.08-0.15 band).
        const float t = (rows > 1u) ? static_cast<float>(rw) / static_cast<float>(rows - 1u)
                                    : 0.0f;
        tiles[r].difficulty = 0.50f + 0.40f * t;
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
render::MeshGeometry TessellateTile(uint32_t type, const TilePreset& base,
                                    float difficulty, float half, uint32_t res) {
    render::MeshGeometry m;
    const uint32_t n = std::max(2u, res);
    const float span = 2.0f * half;
    const float step = span / static_cast<float>(n - 1u);
    // Sample the ONE cooked-style grid (the same parametric generator + bilinear
    // sampler the cook/obs use), so the rendered surface cannot drift from physics.
    const terrain::HeightField hf = TileHeightField(type, base, difficulty, half, n);

    // Flat shading: emit 4 independent vertices per quad (2 triangles) so each
    // riser/tread has its own normal and stair edges stay crisp (no smoothing).
    auto h = [&](float x, float y) { return terrain::SampleHeightFieldZ(hf, x, y); };
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

// ---------------------------------------------------------------------------
// TessellateComposite -- tessellate the kComposite field as ONE BIG WORLD-SPACE
// mesh spanning [xmin,xmax] x [ymin,ymax]. Unlike TessellateTile (a tile-local
// patch translated to a per-robot offset), this samples SampleTerrainHeight(
// kComposite, WORLD_x, WORLD_y, p) so the whole field is ONE continuous, seamless
// surface in world coordinates -- every dog sits on its TRUE physics (x,y) on this
// single mesh (no per-dog tile, no float, no clip). Vertices are absolute world
// positions; the caller adds this instance at world origin (identity xform). Flat
// per-triangle normals keep stair risers crisp.
// ---------------------------------------------------------------------------
render::MeshGeometry TessellateComposite(const TilePreset& base,
                                         float difficulty, float xmin, float xmax,
                                         float ymin, float ymax, float spacing) {
    render::MeshGeometry m;
    const terrain::HeightField hf =
        CompositeHeightField(base, difficulty, xmin, xmax, ymin, ymax, spacing);
    const float sx = std::max(1e-3f, spacing);
    const uint32_t nx = std::max(2u, static_cast<uint32_t>(std::ceil((xmax - xmin) / sx)) + 1u);
    const uint32_t ny = std::max(2u, static_cast<uint32_t>(std::ceil((ymax - ymin) / sx)) + 1u);
    const float dx = (xmax - xmin) / static_cast<float>(nx - 1u);
    const float dy = (ymax - ymin) / static_cast<float>(ny - 1u);
    auto h = [&](float x, float y) { return terrain::SampleHeightFieldZ(hf, x, y); };
    auto push_v = [&](float x, float y, float z, const Vec3& nrm) {
        m.positions.push_back(x); m.positions.push_back(y); m.positions.push_back(z);
        m.normals.push_back(nrm.x); m.normals.push_back(nrm.y); m.normals.push_back(nrm.z);
    };
    for (uint32_t j = 0; j + 1 < ny; ++j) {
        for (uint32_t i = 0; i + 1 < nx; ++i) {
            const float x0 = xmin + static_cast<float>(i) * dx;
            const float x1 = xmin + static_cast<float>(i + 1u) * dx;
            const float y0 = ymin + static_cast<float>(j) * dy;
            const float y1 = ymin + static_cast<float>(j + 1u) * dy;
            const Vec3 a{x0, y0, h(x0, y0)};
            const Vec3 b{x1, y0, h(x1, y0)};
            const Vec3 c{x1, y1, h(x1, y1)};
            const Vec3 dd{x0, y1, h(x0, y1)};
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

// ---- CUDA ray-traced backend (the GPU render path) ------------------------
// A backend swap on the SAME RenderWorld + RasterOptions: re-express the world
// as the path-tracer scene-desc and shade it on the GPU (one analytic light).
class GpuRenderer {
public:
    explicit GpuRenderer(const render::RenderWorld& rw) {
        backend_ = render::CreateCudaRtBackend();
        if (!backend_) return;
        scene_ = render::RenderWorldToTwoLevelScene(rw);
        handle_ = backend_->BuildScene(scene_);
    }
    ~GpuRenderer() {
        if (backend_ && handle_) backend_->FreeScene(handle_);
    }
    bool ok() const { return backend_ != nullptr && handle_ != nullptr; }

    // Enable the stochastic beauty path (MSAA + soft shadows + AO + 1-bounce GI +
    // procedural sky/fog) with the given sub-pixel sample count; off => the flat
    // deterministic sensor trace.
    void SetBeauty(bool on, uint32_t samples) { beauty_ = on; samples_ = samples; }

    // Refresh transforms/materials/light from the live world (BLAS reused), trace
    // on the GPU, tonemap the linear framebuffer into the RGBA8 report.
    render::VulkanOffscreenReport Render(const render::RenderWorld& rw,
                                         const render::RasterOptions& opts) {
        scene_ = render::RenderWorldToTwoLevelScene(rw);
        ApplyLighting(opts);
        const rt::PinholeCamera cam = CameraFromOptions(opts);
        rt::Framebuffer fb;
        if (beauty_) {
            fb = backend_->TraceBeautyToHost(handle_, scene_, cam, BeautyFromOptions(opts));
        } else {
            fb = backend_->TraceToHost(handle_, scene_, cam);
        }
        return render::FramebufferToReport(fb, opts.background);
    }

private:
    void ApplyLighting(const render::RasterOptions& opts) {
        // The sun direction in RasterOptions points TOWARD the sun; the rt::Light
        // direction is the travel direction (away from the sun) -> negate.
        rt::Light& l = scene_.light;
        l.directional = true;
        Vec3 to_sun{opts.sun_direction[0], opts.sun_direction[1], opts.sun_direction[2]};
        if (to_sun.Length() > 1e-6f) to_sun = to_sun.Normalized();
        l.direction = -to_sun;
        l.color = {opts.sun_color[0], opts.sun_color[1], opts.sun_color[2]};
        l.intensity = 1.0f;
        // Flat ambient = the average of the sky + ground hemispheric fill so unlit
        // faces lift off pure black, matching the raster ambient floor.
        scene_.ambient.color = {
            0.5f * (opts.sun_ambient_sky[0] + opts.sun_ambient_ground[0]),
            0.5f * (opts.sun_ambient_sky[1] + opts.sun_ambient_ground[1]),
            0.5f * (opts.sun_ambient_sky[2] + opts.sun_ambient_ground[2])};
    }

    rt::PinholeCamera CameraFromOptions(const render::RasterOptions& opts) {
        const float vfov = opts.camera_fov_degrees * (kPi / 180.0f);
        return rt::BuildPinhole(opts.camera_eye, opts.camera_target, opts.camera_up,
                                vfov, opts.width, opts.height);
    }

    // Map the SAME RasterOptions atmosphere (sky gradient + fog + sun) the lavapipe
    // path consumes onto the beauty path's stochastic controls -> consistent look.
    rt::BeautyOptions BeautyFromOptions(const render::RasterOptions& opts) {
        rt::BeautyOptions b;
        b.samples = samples_;
        b.shadow_rays = (opts.shadow_strength > 0.0f) ? 6u : 1u;
        b.sun_angular_radius = 0.045f;
        b.gi_bounces = 1u;
        b.ao_samples = 4u;
        b.ao_radius = 0.7f;
        b.seed = 0x9e3779b9u;
        b.sky_top = {opts.sky_top[0], opts.sky_top[1], opts.sky_top[2]};
        b.sky_bottom = {opts.sky_bottom[0], opts.sky_bottom[1], opts.sky_bottom[2]};
        b.sky_ground = {opts.ground_color[0], opts.ground_color[1], opts.ground_color[2]};
        b.fog_color = {opts.fog_color[0], opts.fog_color[1], opts.fog_color[2]};
        b.fog_density = opts.fog_density;
        // Sky-dome ambient kept a modest FILL so the cool sky lifts the unlit body
        // off pure black WITHOUT washing the scene to white -- the sun key (~2.0)
        // stays dominant, ACES keeps the lit treads mid-grey.
        b.sky_intensity = 0.30f;
        return b;
    }

    std::unique_ptr<render::RtBackendI> backend_;
    render::RtSceneHandle* handle_ = nullptr;
    rt::TwoLevelScene scene_;
    bool beauty_ = false;
    uint32_t samples_ = 16u;
};

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
        const char* names[5] = {"Flat", "PyramidStairs", "InvertedPyramid",
                                 "RandomBoxes", "Composite"};
        std::printf("[go2_terrain]   tile[%u] @(%.2f,%.2f) type=%s diff=%.3f\n", r,
                    t.offset_x, t.offset_y,
                    names[t.terrain_type < 5u ? t.terrain_type : 0u], t.difficulty);
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

    // Drive each visual instance from the link BuildRenderWorld already resolved:
    // pose_source.row is the cook link/body row (identity-mapped to the trajectory
    // link), cached_visual_local is the geom offset in that link frame. Every visual
    // sub-mesh -- including the leg geoms whose entity carries no direct CookedRef --
    // binds, so all 13 links of every dog render their real mesh.
    std::vector<uint32_t>  tmpl_link(tmpl.instances.size(), 0u);
    std::vector<Transform> tmpl_vlocal(tmpl.instances.size(), Transform::Identity());
    uint32_t n_rebound = 0u, n_oob = 0u, n_static = 0u;
    for (size_t i = 0; i < tmpl.instances.size(); ++i) {
        const render::PoseSource& ps = tmpl.instances[i].pose_source;
        if (ps.kind == render::PoseSource::Kind::Static || ps.row == render::kNoId) {
            ++n_static; continue;
        }
        if (ps.row < traj.link_count) {
            tmpl_link[i]   = ps.row;
            tmpl_vlocal[i] = tmpl.instances[i].cached_visual_local;
            ++n_rebound;
        } else { ++n_oob; }
    }
    if (n_oob > 0u) {
        std::fprintf(stderr, "[go2_terrain] FATAL: %u instances reference link >= %u\n",
                     n_oob, traj.link_count);
        return 5;
    }
    if (n_static > 0u) {
        std::fprintf(stderr, "[go2_terrain] WARN: %u instances have no link pose "
                     "(left at link 0)\n", n_static);
    }
    std::printf("[go2_terrain] template: %u instances (rebound_to_link=%u), "
                "%u meshes\n", per_robot_inst, n_rebound, tmpl.meshes.Count());

    // ---- 3. PREMIUM MATERIAL OVERRIDE (the go2_walk_video studio look) ------
    render::RenderWorld rw;
    rw.meshes = std::move(tmpl.meshes);   // shared interned go2 meshes
    const uint32_t kMatShell = 0u, kMatMetal = 1u, kMatFoot = 2u, kMatAccent = 3u;
    // Body materials lifted off near-black so the trunk reads as a SOLID lit
    // volume: the sky-ambient + AO + GI of the beauty path (and the sun + shadow
    // of the raster path) need a non-zero albedo to carve form -- a 0.016 shell
    // stays a flat black smear no matter the lighting. A plausible dark-charcoal
    // PBR with moderate roughness catches a sun highlight without going cartoonish.
    rw.materials.push_back(Mk(0.072f, 0.076f, 0.085f, 0.05f, 0.55f));  // 0 shell charcoal
    rw.materials.push_back(Mk(0.090f, 0.096f, 0.110f, 0.85f, 0.34f));  // 1 dark gunmetal
    rw.materials.push_back(Mk(0.045f, 0.045f, 0.050f, 0.04f, 0.72f));  // 2 dark rubber
    rw.materials.push_back(Mk(0.16f,  0.180f, 0.225f, 0.65f, 0.30f));  // 3 cool steel accent
    // Warm sandy concrete matching the floor disc: one continuous matte surface the
    // sun + shadow map carve, and dark dogs pop against it.
    const uint32_t kMatTerrain = static_cast<uint32_t>(rw.materials.size());
    rw.materials.push_back(Mk(0.460f, 0.400f, 0.300f, 0.02f, 0.86f));
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

    const TilePreset preset;

    // ---- COMPOSITE vs per-tile detection --------------------------------------
    // The real terrain rollout (--bin) writes a composite code + offset (0,0) for
    // every robot: ONE continuous world-frame field on which every dog sits at its
    // TRUE physics (x,y). We tessellate that as a SINGLE big mesh over the dogs'
    // world AABB. The placeholder keeps the legacy per-robot tile.
    bool composite_mode = (traj.num_robots > 0u);
    for (uint32_t r = 0; r < traj.num_robots; ++r)
        if (traj.tiles[r].terrain_type != kTileComposite)
            composite_mode = false;

    // World AABB of all dog base (link-0) positions over the WHOLE trajectory -- the
    // extent the composite MESH must cover (every dog, every step, sits on it).
    float dog_x_min = 1e9f, dog_x_max = -1e9f, dog_y_min = 1e9f, dog_y_max = -1e9f;
    // SEPARATE, TIGHTER AABB of the SPAWN positions (step 0) -- the climb happens at
    // spawn, so the orbit frames the spawn cluster so the climbing dogs read LARGE
    // (owner: the first cut's dogs were too small). A dog that later wanders off is
    // fine -- the money shot is the ascent.
    float spawn_x_min = 1e9f, spawn_x_max = -1e9f, spawn_y_min = 1e9f, spawn_y_max = -1e9f;
    for (uint32_t s = 0; s < traj.num_steps; ++s) {
        for (uint32_t r = 0; r < traj.num_robots; ++r) {
            const Vec3 bp = traj.LinkPose(s, r, 0u).position;  // world (real GO2T)
            dog_x_min = std::min(dog_x_min, bp.x); dog_x_max = std::max(dog_x_max, bp.x);
            dog_y_min = std::min(dog_y_min, bp.y); dog_y_max = std::max(dog_y_max, bp.y);
            if (s == 0u) {
                spawn_x_min = std::min(spawn_x_min, bp.x); spawn_x_max = std::max(spawn_x_max, bp.x);
                spawn_y_min = std::min(spawn_y_min, bp.y); spawn_y_max = std::max(spawn_y_max, bp.y);
            }
        }
    }
    if (!(dog_x_max >= dog_x_min)) { dog_x_min = -2.0f; dog_x_max = 2.0f;
                                     dog_y_min = -2.0f; dog_y_max = 2.0f; }
    if (!(spawn_x_max >= spawn_x_min)) { spawn_x_min = dog_x_min; spawn_x_max = dog_x_max;
                                         spawn_y_min = dog_y_min; spawn_y_max = dog_y_max; }

    std::vector<uint32_t> tile_inst_index(traj.num_robots, render::kNoId);
    if (composite_mode) {
        // ONE big composite mesh spanning the dog AABB + a modest flat margin so the
        // field reads as an environment without a wide dead-flat border around the pack.
        const float margin = 1.8f;
        const float xmin = dog_x_min - margin, xmax = dog_x_max + margin;
        const float ymin = dog_y_min - margin, ymax = dog_y_max + margin;
        const float spacing = 0.05f;  // crisp 0.30 m stair edges
        const float diff = traj.tiles[0].difficulty;  // all equal (env difficulty)
        std::printf("[go2_terrain] COMPOSITE field: ONE mesh over world "
                    "[%.1f,%.1f]x[%.1f,%.1f] (spacing %.3f m, diff %.2f)\n",
                    xmin, xmax, ymin, ymax, spacing, diff);
        char key[96];
        std::snprintf(key, sizeof(key), "composite:%.4f:%.1f:%.1f:%.1f:%.1f",
                      diff, xmin, xmax, ymin, ymax);
        const uint32_t mesh_id = rw.meshes.InternPrimitive(key, [&]() {
            return TessellateComposite(preset, diff, xmin, xmax, ymin, ymax, spacing);
        });
        render::RenderInstance ti;
        ti.mesh_id = mesh_id;
        ti.render_material_id = kMatTerrain;
        ti.world_xform = Transform::Identity();  // vertices are absolute world coords
        ti.pose_source = render::PoseSource{};
        rw.instances.push_back(ti);
    } else {
        // ---- per-robot tile meshes (placeholder): tessellate ONE per robot -------
        const uint32_t tile_res =
            static_cast<uint32_t>(std::ceil(2.0f * kTileHalf / 0.05f)) + 1u;  // ~65
        std::printf("[go2_terrain] terrain tessellation: %ux%u verts/tile "
                    "(spacing %.3f m, half=%.2f m)\n", tile_res, tile_res,
                    2.0f * kTileHalf / static_cast<float>(tile_res - 1u), kTileHalf);
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
    }
    std::printf("[go2_terrain] RenderWorld: %u robot instances + terrain = %u total "
                "(%s)\n", robot_inst_end, rw.InstanceCount(),
                composite_mode ? "COMPOSITE one-field" : "per-tile");

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
                const uint32_t tile_res =
                    static_cast<uint32_t>(std::ceil(2.0f * kTileHalf / 0.05f)) + 1u;
                const terrain::HeightField hf = TileHeightField(
                    t.terrain_type, preset, t.difficulty, kTileHalf, tile_res);
                z_lift = terrain::SampleHeightFieldZ(hf, bx, by);
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
            const uint32_t tile_res =
                static_cast<uint32_t>(std::ceil(2.0f * kTileHalf / 0.05f)) + 1u;
            const terrain::HeightField hf = TileHeightField(
                t.terrain_type, preset, t.difficulty, kTileHalf, tile_res);
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
                const float surf_z = terrain::SampleHeightFieldZ(hf, lx, ly);
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
    // ONE general path, two backends on the SAME RenderWorld: the default lavapipe
    // raster renderer, or (--gpu / NK_GPU_RENDER=1) the CUDA ray-traced backend.
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    std::unique_ptr<GpuRenderer> gpu;
    if (args.gpu) {
        gpu = std::make_unique<GpuRenderer>(rw);
        if (!gpu->ok()) {
            std::fprintf(stderr, "[go2_terrain] --gpu: no CUDA RT backend\n");
            return 6;
        }
        gpu->SetBeauty(args.beauty, args.samples);
        std::printf("[go2_terrain] renderer: CUDA two-level ray tracer (GPU, %s%s)\n",
                    args.beauty ? "BEAUTY MSAA/AO/GI" : "flat",
                    args.beauty ? "" : " sensor");
    } else {
        try {
            renderer = std::make_unique<render::VulkanRasterRenderer>();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[go2_terrain] no Vulkan device: %s\n", e.what());
            return 6;
        }
        std::printf("[go2_terrain] renderer ICD: %s\n", renderer->DeviceName().c_str());
    }
    auto render_frame = [&](const render::RasterOptions& o) {
        return gpu ? gpu->Render(rw, o) : renderer->Render(rw, o);
    };

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = true;
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 42.0f;
    // Concrete floor matching the terrain tiles (one continuous grey field, like the
    // Isaac-Lab cover). The flat clear is replaced by a cool vertical SKY gradient.
    opts.background = {150, 178, 212, 255};  // fallback flat clear (sky_gradient on)
    opts.ground_color[0] = 0.460f; opts.ground_color[1] = 0.400f; opts.ground_color[2] = 0.300f;
    opts.contact_shadow_strength = 0.0f;  // explicit per-foot points below

    // ---- CINEMATIC SUN + DIRECTIONAL SHADOW MAP (the Isaac-Lab look) ----------
    // A single strong low-angle sun carves the stairs (vertical risers darken) and
    // casts crisp shadows into the grooves + from the dogs via the shadow map.
    opts.use_sun_light = true;
    // Low-ish sun from the front-left so shadows rake ACROSS the field toward camera.
    // Intensity tuned against the ACES tonemap so a 0.5 concrete albedo stays a
    // readable MID-grey (not blown to white): a modest key + LOW ambient gives the
    // wide tread-vs-riser contrast that carves the stairs.
    opts.sun_direction[0] = 0.52f; opts.sun_direction[1] = -0.44f; opts.sun_direction[2] = 0.30f;
    opts.sun_color[0] = 2.30f; opts.sun_color[1] = 2.08f; opts.sun_color[2] = 1.70f;
    opts.sun_ambient_sky[0] = 0.082f; opts.sun_ambient_sky[1] = 0.094f; opts.sun_ambient_sky[2] = 0.120f;
    opts.sun_ambient_ground[0] = 0.052f; opts.sun_ambient_ground[1] = 0.049f; opts.sun_ambient_ground[2] = 0.043f;
    opts.shadow_strength = args.no_shadows ? 0.0f : 0.97f;  // crisp near-black grooves
    opts.shadow_map_size = args.shadow_map;  // default 2560; --shadow-map lowers for speed
    opts.shadow_bias = 0.0018f;

    // ---- COOL VERTICAL SKY GRADIENT ------------------------------------------
    opts.sky_gradient = true;
    opts.sky_top[0] = 0.26f; opts.sky_top[1] = 0.45f; opts.sky_top[2] = 0.74f;     // deep clear-blue zenith
    opts.sky_bottom[0] = 0.62f; opts.sky_bottom[1] = 0.74f; opts.sky_bottom[2] = 0.88f; // soft blue horizon

    // Subtle FAR-FIELD haze only (foreground dogs stay crisp): low density + the
    // sky-horizon colour so distant tiles melt into the sky, not a white wall.
    opts.fog_density = args.fog;
    opts.fog_color[0] = 0.66f; opts.fog_color[1] = 0.76f; opts.fog_color[2] = 0.88f;
    if (const char* fv = std::getenv("NK_CAM_FOV"))
        opts.camera_fov_degrees = static_cast<float>(std::atof(fv));

    // ---- framing for the orbit ----------------------------------------------
    // COMPOSITE (the real rollout): the dogs SPREAD as they walk, so a single static
    // AABB over the whole clip frames too wide -> dogs read tiny (owner round-1+2
    // complaint). Instead we TRACK the pack: per render-step the camera targets the
    // dog CENTROID and fits a ROBUST radius (the median dog-distance-from-centroid,
    // scaled) so the BULK of the pack fills the frame and a single dog that wanders
    // off does NOT pull the camera back. The centroid path is lightly smoothed (a
    // moving average) so the pan glides. PLACEHOLDER: the legacy static tile grid.
    std::vector<Vec3> step_ctr(traj.num_steps, Vec3{0, 0, 0});
    std::vector<float> step_rad(traj.num_steps, 2.5f);
    float gx_min = 1e9f, gx_max = -1e9f, gy_min = 1e9f, gy_max = -1e9f;
    if (composite_mode) {
        // Raw per-step centroid + robust radius (median distance from centroid).
        std::vector<Vec3> raw_ctr(traj.num_steps, Vec3{0, 0, 0});
        std::vector<float> raw_rad(traj.num_steps, 2.5f);
        for (uint32_t s = 0; s < traj.num_steps; ++s) {
            Vec3 c{0, 0, 0};
            for (uint32_t r = 0; r < traj.num_robots; ++r)
                c = c + traj.LinkPose(s, r, 0u).position;
            c = c * (1.0f / static_cast<float>(std::max(1u, traj.num_robots)));
            c.z = 0.0f;
            raw_ctr[s] = c;
            std::vector<float> dist(traj.num_robots, 0.0f);
            for (uint32_t r = 0; r < traj.num_robots; ++r) {
                const Vec3 bp = traj.LinkPose(s, r, 0u).position;
                const float dx = bp.x - c.x, dy = bp.y - c.y;
                dist[r] = std::sqrt(dx * dx + dy * dy);
            }
            std::sort(dist.begin(), dist.end());
            // 60th-percentile distance: frames the dense core of the pack large; the
            // outermost wanderers drift to the frame edge rather than shrink everyone.
            const float pct = dist.empty() ? 2.5f
                : dist[std::min(dist.size() - 1, (dist.size() * 3u) / 5u)];
            float cap = 8.0f;
            if (const char* rc = std::getenv("NK_RAD_CAP")) cap = static_cast<float>(std::atof(rc));
            raw_rad[s] = std::min(cap, std::max(2.5f, pct));
        }
        // Smooth the centroid + radius with a centred moving average (window ~ 0.5 s
        // of render-steps) so the camera glide is buttery, not jittery.
        const int win = std::max(1, static_cast<int>(0.5f / std::max(1e-3f, traj.ctrl_dt)) / 2);
        for (uint32_t s = 0; s < traj.num_steps; ++s) {
            Vec3 acc{0, 0, 0};
            float racc = 0.0f;
            int cnt = 0;
            for (int k = -win; k <= win; ++k) {
                const int j = static_cast<int>(s) + k;
                if (j < 0 || j >= static_cast<int>(traj.num_steps)) continue;
                acc = acc + raw_ctr[static_cast<uint32_t>(j)];
                racc += raw_rad[static_cast<uint32_t>(j)];
                ++cnt;
            }
            if (cnt > 0) { step_ctr[s] = acc * (1.0f / cnt); step_rad[s] = racc / cnt; }
            else { step_ctr[s] = raw_ctr[s]; step_rad[s] = raw_rad[s]; }
        }
    } else {
        for (const TileSpec& t : traj.tiles) {
            gx_min = std::min(gx_min, t.offset_x - kTileHalf);
            gx_max = std::max(gx_max, t.offset_x + kTileHalf);
            gy_min = std::min(gy_min, t.offset_y - kTileHalf);
            gy_max = std::max(gy_max, t.offset_y + kTileHalf);
        }
    }
    const Vec3 grid_ctr{0.5f * (gx_min + gx_max), 0.5f * (gy_min + gy_max), 0.0f};
    // A floor on the radius so a near-collinear cluster still frames sensibly.
    const float grid_radius = std::max(2.5f,
        0.5f * std::sqrt((gx_max - gx_min) * (gx_max - gx_min) +
                         (gy_max - gy_min) * (gy_max - gy_min)));

    auto smoothstep = [](float e0, float e1, float x) {
        float t = (x - e0) / (e1 - e0);
        t = std::max(0.0f, std::min(1.0f, t));
        return t * t * (3.0f - 2.0f * t);
    };
    auto lerp = [](float a, float b, float u) { return a + (b - a) * u; };

    // One keyframed curve drives elevation, azimuth (turns past a front-left 3/4
    // start), fov-fit and look-at height; the eye is derived around the live pack
    // centroid each step so framing tracks the pack through a shot sequence.
    struct CamKey { float t, elev_deg, azim_turns, fit, target_z; };
    static const CamKey kCamPath[] = {
        {0.00f, 52.0f, 0.00f, 1.55f, 0.35f},  // establishing crane (high wide)
        {0.16f, 30.0f, 0.05f, 1.05f, 0.32f},  // crane descends
        {0.21f, 13.0f, 0.08f, 0.52f, 0.30f},  // drop to low tracking dolly
        {0.42f, 11.0f, 0.11f, 0.48f, 0.34f},  // dolly travels with the pack
        {0.47f,  8.0f, 0.17f, 0.43f, 0.46f},  // hero low-angle crest (look up)
        {0.62f,  9.0f, 0.31f, 0.45f, 0.46f},  // hero arc across the climb
        {0.67f, 17.0f, 0.42f, 0.62f, 0.33f},  // rise into the orbit
        {0.85f, 21.0f, 0.92f, 0.66f, 0.32f},  // orbit reveal (~half turn)
        {0.90f, 26.0f, 1.00f, 0.86f, 0.32f},  // begin pull-back
        {1.00f, 40.0f, 1.14f, 1.30f, 0.30f},  // crane-up finale (wide)
    };
    const int num_cam_keys = static_cast<int>(sizeof(kCamPath) / sizeof(kCamPath[0]));
    // --orbit-turns scales total azimuth travel (0.6 = the authored path).
    const float azim_scale = (args.orbit_turns > 0.0f) ? args.orbit_turns / 0.6f : 1.0f;
    // NK_FIT_FACTOR globally scales the fov-fit (smaller = eye IN = bigger dogs).
    float fit_mul = 1.0f;
    if (const char* ff = std::getenv("NK_FIT_FACTOR"))
        fit_mul = static_cast<float>(std::atof(ff));
    auto eval_cam = [&](float t, float& elev_deg, float& azim_turns, float& fit, float& tz) {
        int i = 0;
        while (i + 1 < num_cam_keys && t > kCamPath[i + 1].t) ++i;
        const CamKey& a = kCamPath[i];
        const CamKey& b = kCamPath[std::min(i + 1, num_cam_keys - 1)];
        const float u = (b.t > a.t) ? smoothstep(a.t, b.t, t) : 0.0f;
        elev_deg   = lerp(a.elev_deg, b.elev_deg, u);
        azim_turns = lerp(a.azim_turns, b.azim_turns, u);
        fit        = lerp(a.fit, b.fit, u);
        tz         = lerp(a.target_z, b.target_z, u);
    };
    auto aim_camera = [&](uint32_t step) {
        const float t = (traj.num_steps > 1u)
            ? static_cast<float>(step) / static_cast<float>(traj.num_steps - 1u) : 0.0f;
        float elev_deg, azim_turns, fit_k, tz;
        eval_cam(t, elev_deg, azim_turns, fit_k, tz);
        const float phi = -0.55f * kPi + 2.0f * kPi * azim_turns * azim_scale;
        const float elev = elev_deg * kPi / 180.0f;
        // Centre + pack radius: live-tracked in composite mode, static otherwise.
        const Vec3 ctr = composite_mode ? step_ctr[step] : grid_ctr;
        const float rad = composite_mode ? step_rad[step] : grid_radius;
        // Fit the pack radius into the vertical fov; a hero floor keeps a tight
        // pack from collapsing the eye onto the centroid.
        const float fov_y = opts.camera_fov_degrees * kPi / 180.0f;
        const float fit_r = (rad * fit_k * fit_mul) / std::tan(fov_y * 0.5f);
        float radius = std::max(fit_r, 4.5f);
        if (const char* hc = std::getenv("NK_HERO_CAP"))
            radius = std::min(radius, static_cast<float>(std::atof(hc)));
        const float ch = std::cos(elev), sh = std::sin(elev);
        const float dx = std::sin(phi), dy = -std::cos(phi);
        opts.camera_eye = {ctr.x + radius * ch * dx,
                           ctr.y + radius * ch * dy,
                           ctr.z + radius * sh + 0.30f};
        opts.camera_target = {ctr.x, ctr.y, ctr.z + tz};
        opts.camera_far = radius * 6.0f + rad * 8.0f + 50.0f;
    };

    // ---- 8. PROBE: render 1 mid-clip frame, exit ----------------------------
    if (args.probe) {
        const uint32_t mid = traj.num_steps / 2u;
        publish(mid);
        set_contact_shadows(opts);
        aim_camera(mid);
        const auto t0 = std::chrono::steady_clock::now();
        render::VulkanOffscreenReport rep = render_frame(opts);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::filesystem::create_directories(args.out_dir);
        const std::string p = args.out_dir + "/probe_frame.ppm";
        const bool ok = WritePpm(rep, p);
        std::printf("[go2_terrain] PROBE[%s]: instances=%u meshes=%u non_bg=%zu (%ux%u) "
                    "render=%.1f ms eye=(%.2f,%.2f,%.2f) -> %s (%s)\n",
                    gpu ? "GPU" : "CPU", rw.InstanceCount(),
                    rw.meshes.Count(), rep.non_background_pixel_count, rep.width, rep.height,
                    ms, opts.camera_eye.x, opts.camera_eye.y, opts.camera_eye.z,
                    p.c_str(), ok ? "OK" : "WRITE-FAIL");
        return ok ? 0 : 7;
    }

    // ---- 9. REPLAY + RENDER LOOP --------------------------------------------
    std::filesystem::create_directories(args.out_dir);
    uint32_t written = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    // Render-only wall-time accumulator: isolates the tracer cost from the CPU
    // scene re-assembly + PPM I/O the loop ms/frame also includes.
    double render_ms_total = 0.0;
    const auto clip_t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < traj.num_steps; s += args.stride) {
        if (args.frames != 0u && written >= args.frames) break;
        publish(s);
        set_contact_shadows(opts);
        aim_camera(s);
        const auto r_t0 = std::chrono::steady_clock::now();
        render::VulkanOffscreenReport rep = render_frame(opts);
        const auto r_t1 = std::chrono::steady_clock::now();
        render_ms_total += std::chrono::duration<double, std::milli>(r_t1 - r_t0).count();
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
    const auto clip_t1 = std::chrono::steady_clock::now();
    const double clip_s = std::chrono::duration<double>(clip_t1 - clip_t0).count();
    std::printf("[go2_terrain] DONE[%s]: wrote %u frames to %s in %.2f s (%.1f ms/frame, "
                "render-only %.1f ms/frame, first non_bg=%zu, last=%zu)\n",
                gpu ? "GPU" : "CPU", written, args.out_dir.c_str(), clip_s,
                written ? 1000.0 * clip_s / written : 0.0,
                written ? render_ms_total / written : 0.0, first_nonbg, last_nonbg);
    return 0;
}
