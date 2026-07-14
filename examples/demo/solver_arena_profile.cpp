// End-to-end solver layout probe: .nks -> cook -> upload -> arena -> op list.

#include "nk/data/arena.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

using nuka::nk::Arena;
using nuka::nk::FieldId;

std::vector<std::string> LoadFieldNames() {
    std::ifstream input(std::string(NUKA_SOURCE_DIR) + "/src/nk/model/fields.yaml");
    std::vector<std::string> names;
    std::string line;
    constexpr const char* key = "- {name:";
    while (std::getline(input, line)) {
        const size_t begin = line.find(key);
        if (begin == std::string::npos) continue;
        const size_t value = line.find_first_not_of(' ', begin + std::char_traits<char>::length(key));
        const size_t end = line.find(',', value);
        if (value != std::string::npos && end != std::string::npos)
            names.push_back(line.substr(value, end - value));
    }
    return names;
}

const char* OpName(nuka::phi::NkOp op) {
    using nuka::phi::NkOp;
    switch (op) {
        case NkOp::ApplyDrives: return "ApplyDrives";
        case NkOp::AbaForward: return "AbaForward";
        case NkOp::IntegrateVelocity: return "IntegrateVelocity";
        case NkOp::FkWorldPoses: return "FkWorldPoses";
        case NkOp::IntegratePosition: return "IntegratePosition";
        case NkOp::CrbaComputeM: return "CrbaComputeM";
        case NkOp::CrbaFactorM: return "CrbaFactorM";
        case NkOp::ApplyImplicitDamping: return "ApplyImplicitDamping";
        case NkOp::BuildAabbs: return "BuildAabbs";
        case NkOp::LbvhBuild: return "LbvhBuild";
        case NkOp::LbvhQueryPairs: return "LbvhQueryPairs";
        case NkOp::ParticleGridBuild: return "ParticleGridBuild";
        case NkOp::NarrowphasePrimitives: return "NarrowphasePrimitives";
        case NkOp::NarrowphaseSdf: return "NarrowphaseSdf";
        case NkOp::ContactTangentBasis: return "ContactTangentBasis";
        case NkOp::AssembleRows: return "AssembleRows";
        case NkOp::SolveRowsBlockIsland: return "SolveRowsBlockIsland";
        case NkOp::ParticleAeroDrag: return "ParticleAeroDrag";
        case NkOp::ParticlePredict: return "ParticlePredict";
        case NkOp::XpbdProject: return "XpbdProject";
        case NkOp::PbfDensityLambda: return "PbfDensityLambda";
        case NkOp::PbfApplyDelta: return "PbfApplyDelta";
        case NkOp::ParticleFinalize: return "ParticleFinalize";
        case NkOp::MpmStep: return "MpmStep";
        case NkOp::ParticleParticleContact: return "ParticleParticleContact";
        case NkOp::SyncLinkBodyPose: return "SyncLinkBodyPose";
        case NkOp::NarrowphaseBodyParticle: return "NarrowphaseBodyParticle";
        case NkOp::NarrowphaseHeightfield: return "NarrowphaseHeightfield";
        case NkOp::BuildSolveIslands: return "BuildSolveIslands";
        default: return "other";
    }
}

struct Row {
    std::string name;
    uint8_t arena = 0;
    uint64_t bytes = 0;
};

bool DumpField(nuka::nk::World& world, FieldId field,
               const std::string& path) {
    uint64_t bytes = 0u;
    for (const Arena::Segment& segment : world.GetData().Segments()) {
        if (segment.field == field) {
            bytes = segment.bytes;
            break;
        }
    }
    if (bytes == 0u) return false;
    std::vector<uint8_t> data(bytes);
    if (!world.GetData().DownloadField(field, data.data(), bytes)) return false;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
    return output.good();
}

bool PrintRowStats(nuka::nk::World& world) {
    const auto& cap = world.GetModel().capacities;
    const uint64_t total_rows64 =
        static_cast<uint64_t>(cap.max_rows_per_env) * cap.env_count;
    if (total_rows64 == 0u || total_rows64 > 0xFFFFFFFFull) return false;
    const uint32_t total_rows = static_cast<uint32_t>(total_rows64);
    std::vector<nuka::nk::NkRow> rows(total_rows);
    if (!world.GetData().DownloadField(FieldId::Urows, rows.data(),
                                       rows.size() * sizeof(rows[0]))) {
        return false;
    }
    uint32_t active = 0u, normals = 0u, friction = 0u;
    uint32_t artic_sides = 0u, rigid_sides = 0u, particle_sides = 0u;
    for (const nuka::nk::NkRow& row : rows) {
        if (!(row.flags & nuka::nk::nk_row_flags::kActive)) continue;
        ++active;
        if (row.flags & nuka::nk::nk_row_flags::kFriction) ++friction;
        else ++normals;
        for (const nuka::nk::NkRowSide* side : {&row.a, &row.b}) {
            if (side->kind == nuka::nk::kNkSideArtic) ++artic_sides;
            else if (side->kind == nuka::nk::kNkSideRigid) ++rigid_sides;
            else if (side->kind == nuka::nk::kNkSideParticle) ++particle_sides;
        }
    }
    uint32_t island_count = 0u;
    if (!world.GetData().DownloadField(FieldId::IslandCount, &island_count,
                                       sizeof(island_count))) {
        return false;
    }
    if (island_count > total_rows) return false;
    std::vector<uint32_t> quads(static_cast<size_t>(island_count) * 4u);
    if (!quads.empty() &&
        !world.GetData().DownloadField(FieldId::IslandQuads, quads.data(),
                                       quads.size() * sizeof(quads[0]))) {
        return false;
    }
    std::vector<uint32_t> island_sizes;
    island_sizes.reserve(island_count);
    uint64_t island_rows = 0u;
    uint32_t min_rows = island_count > 0u ? ~0u : 0u;
    uint32_t max_rows = 0u;
    for (uint32_t i = 0u; i < island_count; ++i) {
        const uint32_t count = quads[static_cast<size_t>(i) * 4u + 1u];
        island_sizes.push_back(count);
        island_rows += count;
        min_rows = std::min(min_rows, count);
        max_rows = std::max(max_rows, count);
    }
    std::sort(island_sizes.begin(), island_sizes.end(), std::greater<uint32_t>());
    std::printf("row_stats active=%u normals=%u friction=%u capacity=%u "
                "islands=%u island_rows=%llu min=%u max=%u mean=%.2f "
                "sides(artic=%u rigid=%u particle=%u)\n",
                active, normals, friction, total_rows, island_count,
                static_cast<unsigned long long>(island_rows), min_rows, max_rows,
                island_count > 0u
                    ? static_cast<double>(island_rows) / island_count : 0.0,
                artic_sides, rigid_sides, particle_sides);
    std::printf("row_stats largest_islands");
    for (size_t i = 0; i < std::min<size_t>(16u, island_sizes.size()); ++i)
        std::printf(" %u", island_sizes[i]);
    std::printf("\n");
    return island_rows == active;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: solver_arena_profile SCENE [--envs N] [--top N] "
                     "[--steps N] [--vel-iters N] [--pos-iters N] "
                     "[--dump-prefix PATH] [--row-stats]\n");
        return 2;
    }
    std::string scene_path = argv[1];
    std::string dump_prefix;
    uint32_t envs = 1u;
    uint32_t top = 30u;
    uint32_t steps = 0u;
    uint32_t vel_iters = 32u;
    uint32_t pos_iters = 4u;
    bool row_stats = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) envs = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--top" && i + 1 < argc) top = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--steps" && i + 1 < argc)
            steps = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--vel-iters" && i + 1 < argc)
            vel_iters = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--pos-iters" && i + 1 < argc)
            pos_iters = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--dump-prefix" && i + 1 < argc) dump_prefix = argv[++i];
        else if (arg == "--row-stats") row_stats = true;
    }
    if (envs == 0u || vel_iters > 0xFFFFu || pos_iters > 0xFFFFu) return 2;

    nuka::phi::Device* device = nuka::phi::InitBestDevice();
    nuka::phi::Backend* backend = device != nullptr
        ? nuka::phi::DeviceInitBackend(device, nullptr) : nullptr;
    if (device == nullptr || backend == nullptr) {
        std::fprintf(stderr, "solver_arena_profile: no device/backend\n");
        return 3;
    }

    nuka::scene::cook::CookToModelOptions options;
    options.bake_link_sdf = true;
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(scene_path);
    auto cooked = nuka::scene::cook::CookSceneToModel(scene, envs, options);
    nuka::nk::Pipeline::SolverConfig config;
    config.dt = 1.0f / 240.0f;
    config.vel_iters = static_cast<uint16_t>(vel_iters);
    config.pos_iters = static_cast<uint16_t>(pos_iters);
    nuka::nk::World world(std::move(cooked.model), envs, device, backend, config);
    if (!world.Ready()) return 4;

    const auto& cap = world.GetModel().capacities;
    std::printf("scene=%s envs=%u vel_iters=%u pos_iters=%u particles/env=%u "
                "mpm_particles/env=%u bodies/env=%u "
                "dofs/env=%u contact_slots/env=%u rows/env=%u mpm_nodes/env=%u\n",
                scene_path.c_str(), envs, vel_iters, pos_iters, cap.particles_per_env,
                world.GetModel().particles.n_mpm_particles, cap.bodies_per_env,
                cap.dofs_per_env, cap.max_contacts_per_env, cap.max_rows_per_env,
                cap.mpm_grid_nodes_per_env);
    std::printf("scratch_bytes particle_grid=%llu mpm_sort=%llu island_sort=%llu\n",
                static_cast<unsigned long long>(cap.grid_sort_scratch_bytes),
                static_cast<unsigned long long>(cap.mpm_grid_sort_scratch_bytes),
                static_cast<unsigned long long>(cap.island_cub_temp_bytes));

    uint64_t arena_totals[3] = {};
    const auto segments = Arena::ComputeSegments(cap, arena_totals);
    uint64_t model_total = 0u;
    (void)world.GetModel().ComputeModelSegments(&model_total);
    std::printf("arena_bytes persistent=%llu scratch=%llu tape=%llu model=%llu total=%llu\n",
                static_cast<unsigned long long>(arena_totals[0]),
                static_cast<unsigned long long>(arena_totals[1]),
                static_cast<unsigned long long>(arena_totals[2]),
                static_cast<unsigned long long>(model_total),
                static_cast<unsigned long long>(arena_totals[0] + arena_totals[1] +
                                                arena_totals[2] + model_total));

    const std::vector<std::string> names = LoadFieldNames();
    std::vector<Row> rows;
    rows.reserve(segments.size());
    for (const Arena::Segment& segment : segments) {
        const size_t id = static_cast<size_t>(segment.field);
        rows.push_back(Row{id < names.size() ? names[id] : std::to_string(id),
                           segment.arena, segment.bytes});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.bytes > b.bytes;
    });
    std::printf("top_data_fields name arena bytes bytes_per_env\n");
    for (size_t i = 0; i < std::min<size_t>(top, rows.size()); ++i) {
        std::printf("  %-30s %u %12llu %12.1f\n", rows[i].name.c_str(), rows[i].arena,
                    static_cast<unsigned long long>(rows[i].bytes),
                    static_cast<double>(rows[i].bytes) / envs);
    }

    std::printf("ops");
    for (const nuka::phi::OpCall& call : world.GetPipeline().Calls())
        std::printf(" %s", OpName(call.op));
    std::printf("\n");

    for (uint32_t step = 0u; step < steps; ++step) {
        const nuka::nk::StepResult result = world.Step();
        if (!result.AllOk()) {
            std::fprintf(stderr, "solver_arena_profile: step %u failed\n", step);
            return 5;
        }
    }
    if (row_stats && !PrintRowStats(world)) {
        std::fprintf(stderr, "solver_arena_profile: row stats failed/inconsistent\n");
        return 6;
    }
    if (!dump_prefix.empty()) {
        const bool pos = DumpField(world, FieldId::ParticlePos,
                                   dump_prefix + ".particle_pos.bin");
        const bool vel = DumpField(world, FieldId::ParticleVel,
                                   dump_prefix + ".particle_vel.bin");
        const bool link = DumpField(world, FieldId::LinkPose,
                                    dump_prefix + ".link_pose.bin");
        if (!pos || !vel || !link) {
            std::fprintf(stderr, "solver_arena_profile: state dump failed\n");
            return 7;
        }
        std::printf("state_dump prefix=%s steps=%u\n", dump_prefix.c_str(), steps);
    }
    return 0;
}
