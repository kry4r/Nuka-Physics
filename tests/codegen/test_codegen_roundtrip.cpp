// ---------------------------------------------------------------------------
// Row codegen round-trip tests
// ---------------------------------------------------------------------------

#include "codegen/generated/row_class_registry.hpp"

// v0.8 C2a: the GENERATED collidable-type registry (emitted by regen.py's second,
// parallel collidable pass). Assert the static metadata table is correct here --
// this TU already add_dependencies(nuka_codegen) and has src/ on its include path.
#include "codegen/generated/collidable_registry.hpp"
#include "constraint/collidable_kinds.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::filesystem::path SourceRoot() {
    return std::filesystem::path(NUKA_SOURCE_DIR);
}

std::string ReadFirstLine(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    return line;
}

int RunCommand(const std::string& command) {
    return std::system(command.c_str());
}

} // namespace

TEST(CodegenRoundtrip, RegenExitsZeroForCurrentIrSet) {
    const auto temp = std::filesystem::temp_directory_path() / "nuka_codegen_roundtrip_out";
    std::filesystem::remove_all(temp);
    std::filesystem::create_directories(temp);

    const std::string command = std::string("\"") + NUKA_PYTHON_EXECUTABLE +
        "\" \"" + (SourceRoot() / "tools/codegen/regen.py").string() +
        "\" --output-dir \"" + temp.string() + "\"";

    EXPECT_EQ(RunCommand(command), 0);

    std::filesystem::remove_all(temp);
}

namespace {
std::string ReadAll(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}
}  // namespace

// The four committed nk field headers must BYTE-MATCH a fresh regen of fields.yaml
// (the stale-header guard the exit-0 test above does not catch: edit yaml, forget
// to regen + commit, or hand-edit a header). Regens into a temp --fields-output-dir
// and byte-diffs each output against the committed src/nk/model/generated/*.hpp.
TEST(CodegenRoundtrip, CommittedNkFieldHeadersMatchFreshRegen) {
    const auto temp = std::filesystem::temp_directory_path() / "nuka_nk_fields_out";
    std::filesystem::remove_all(temp);
    std::filesystem::create_directories(temp);

    const std::string command = std::string("\"") + NUKA_PYTHON_EXECUTABLE +
        "\" \"" + (SourceRoot() / "tools/codegen/regen.py").string() +
        "\" --fields-output-dir \"" + temp.string() + "\"";
    ASSERT_EQ(RunCommand(command), 0);

    const auto committed = SourceRoot() / "src/nk/model/generated";
    const std::array<const char*, 4> headers = {
        "field_ids.hpp", "views.hpp", "arena_layout.hpp", "dlpack_table.hpp"};
    for (const char* h : headers) {
        const auto fresh = temp / h;
        const auto on_disk = committed / h;
        ASSERT_TRUE(std::filesystem::exists(fresh)) << fresh;
        ASSERT_TRUE(std::filesystem::exists(on_disk)) << on_disk;
        EXPECT_EQ(ReadAll(on_disk), ReadAll(fresh))
            << "committed " << h << " is stale vs a fresh regen of fields.yaml "
            << "(run: python tools/codegen/regen.py and commit the 4 headers)";
    }
    std::filesystem::remove_all(temp);
}

TEST(CodegenRoundtrip, GeneratedFilesCarryDoNotEditHeader) {
    const auto generated = SourceRoot() / "src/codegen/generated";
    const std::array<const char*, 9> files = {
        "maximal_contact_forward.cu",
        "maximal_joint_forward.cu",
        "maximal_drive_forward.cu",
        "featherstone_contact_forward.cu",
        "rigid_sdf_contact_forward.cu",
        "featherstone_sdf_contact_forward.cu",
        "coupling_contact_forward.cu",
        "row_dispatch.cu",
        "row_class_registry.hpp",
    };

    for (const char* file : files) {
        const auto path = generated / file;
        EXPECT_TRUE(std::filesystem::exists(path)) << path;
        EXPECT_EQ(ReadFirstLine(path), "// GENERATED — DO NOT EDIT") << path;
    }
}

TEST(CodegenRoundtrip, GeneratedRegistryLinksAndReportsBaseRows) {
    using nuka::solver::generated::GradientMode;
    using nuka::solver::generated::IsKnownRowClass;
    using nuka::solver::generated::RowClassGradientMode;
    using nuka::solver::generated::RowClassHasAdjoint;
    using nuka::solver::generated::RowClassMaxRowsPerBlock;
    using nuka::solver::generated::RowClassName;
    using nuka::solver::generated::kFeatherstoneContactRowId;
    using nuka::solver::generated::kFeatherstoneSDFContactRowId;
    using nuka::solver::generated::kMaximalContactRowId;
    using nuka::solver::generated::kMaximalDriveRowId;
    using nuka::solver::generated::kMaximalJointRowId;
    using nuka::solver::generated::kRigidSDFContactRowId;
    using nuka::solver::generated::kCouplingContactRowId;
    using nuka::solver::generated::kRowClassCount;

    // Ids 0-3 are the rigid and articulated rows, 4/5 the SDF contact rows and
    // 6 the cross-system coupling row.
    EXPECT_EQ(kRowClassCount, 7u);
    EXPECT_TRUE(IsKnownRowClass(kMaximalContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kMaximalJointRowId));
    EXPECT_TRUE(IsKnownRowClass(kMaximalDriveRowId));
    EXPECT_TRUE(IsKnownRowClass(kFeatherstoneContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kRigidSDFContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kFeatherstoneSDFContactRowId));
    EXPECT_EQ(kRigidSDFContactRowId, 4u);
    EXPECT_EQ(kFeatherstoneSDFContactRowId, 5u);
    EXPECT_STREQ(RowClassName(kMaximalContactRowId), "MaximalContactRow");
    EXPECT_STREQ(RowClassName(kRigidSDFContactRowId), "RigidSDFContactRow");
    EXPECT_STREQ(RowClassName(kFeatherstoneSDFContactRowId), "FeatherstoneSDFContactRow");
    EXPECT_EQ(RowClassMaxRowsPerBlock(kMaximalContactRowId), 6u);
    EXPECT_EQ(RowClassMaxRowsPerBlock(kMaximalDriveRowId), 1u);
    EXPECT_EQ(RowClassMaxRowsPerBlock(kRigidSDFContactRowId), 6u);
    // SDF-contact rows (id4/id5) are FORWARD-ONLY at the row-class level. p08-C built
    // the reverse mode as a SYSTEM-level IFT d/dM,d/dJ reference + envelope DEPTH
    // adjoint (normal/point are flat-valley degenerate); the per-row adjoint was
    // deliberately NOT added, and engine-grounding + backward WIRING are DEFERRED to
    // p11. RowClassHasAdjoint()==false is the AUTHORITATIVE "no dispatchable adjoint"
    // signal -- any backward consumer MUST gate on it (false => stop-gradient / hard
    // error, never silent zero). gradient_mode records the TARGET channel. This block
    // is the contract tripwire: when p11 wires the adjoint, update it consciously.
    EXPECT_FALSE(RowClassHasAdjoint(kRigidSDFContactRowId));
    EXPECT_FALSE(RowClassHasAdjoint(kFeatherstoneSDFContactRowId));
    EXPECT_EQ(RowClassGradientMode(kRigidSDFContactRowId),
              static_cast<uint8_t>(GradientMode::ift_at_convergence));
    EXPECT_EQ(RowClassGradientMode(kFeatherstoneSDFContactRowId),
              static_cast<uint8_t>(GradientMode::ift_at_convergence));

    // The generic CROSS-SYSTEM coupling-row class (id 6, architecture
    // 0.7) -- a solver-resolved compliant non-penetration + friction row whose two
    // CollidableRef sides may be any two CollidableTypes. FORWARD-ONLY this
    // milestone (the id4/id5 SDF precedent): RowClassHasAdjoint(6)==false (the
    // authoritative "no adjoint" signal) and gradient mode `none`. The framework
    // that EMITS these rows is src/runtime/coupling/coupling_row_framework.hpp; the
    // solve is the unified PGS contact path, not this generated stub.
    EXPECT_EQ(kCouplingContactRowId, 6u);
    EXPECT_TRUE(IsKnownRowClass(kCouplingContactRowId));
    EXPECT_STREQ(RowClassName(kCouplingContactRowId), "CouplingContactRow");
    EXPECT_EQ(RowClassMaxRowsPerBlock(kCouplingContactRowId), 6u);
    EXPECT_FALSE(RowClassHasAdjoint(kCouplingContactRowId));
    EXPECT_EQ(RowClassGradientMode(kCouplingContactRowId),
              static_cast<uint8_t>(GradientMode::none));

    EXPECT_FALSE(IsKnownRowClass(99u));
}

// v0.8 C2a: the GENERATED collidable-type registry. Proves regen.py's second,
// parallel pass emitted the static metadata table correctly: 4 contiguous types,
// enum values 0..3, and each type's accel/proxy/react kind selectors match its IR.
TEST(CodegenRoundtrip, CollidableTypeRegistryIsCorrect) {
    using nuka::constraint::AccelStructureKind;
    using nuka::constraint::ReactionProviderKind;
    using nuka::constraint::ShapeProxyKind;
    using nuka::constraint::generated::CollidableType;
    using nuka::constraint::generated::GetCollidableTypeInfo;
    using nuka::constraint::generated::kCollidableTypeCount;

    // Type ids are append-only and index the generated registry.
    EXPECT_EQ(kCollidableTypeCount, 7u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::RigidBody), 0u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::ArticulationLink), 1u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::Particle), 2u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::StaticWorld), 3u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::GridNode), 4u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::StaticBoundary), 5u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::ParticleSurface), 6u);
    const auto& surface = GetCollidableTypeInfo(CollidableType::ParticleSurface);
    EXPECT_EQ(surface.react, ReactionProviderKind::PointEndpoint);

    // RigidBody (id 0): rigid LBVH, shape-backed, rigid inverse-mass reaction.
    const auto& rigid = GetCollidableTypeInfo(CollidableType::RigidBody);
    EXPECT_EQ(rigid.type, CollidableType::RigidBody);
    EXPECT_EQ(rigid.accel, AccelStructureKind::LbvhRigid);
    EXPECT_EQ(rigid.proxy, ShapeProxyKind::ShapeBacked);
    EXPECT_EQ(rigid.react, ReactionProviderKind::RigidInvMass);

    // ArticulationLink (id 1): rigid LBVH, shape-backed, chain-Jacobian reaction.
    const auto& link = GetCollidableTypeInfo(CollidableType::ArticulationLink);
    EXPECT_EQ(link.accel, AccelStructureKind::LbvhRigid);
    EXPECT_EQ(link.proxy, ShapeProxyKind::ShapeBacked);
    EXPECT_EQ(link.react, ReactionProviderKind::ArticulationChainJ);

    // Particle (id 2): uniform-grid, point-sphere proxy, particle inverse-mass.
    const auto& particle = GetCollidableTypeInfo(CollidableType::Particle);
    EXPECT_EQ(particle.accel, AccelStructureKind::UniformGridParticle);
    EXPECT_EQ(particle.proxy, ShapeProxyKind::PointSphere);
    EXPECT_EQ(particle.react, ReactionProviderKind::ParticleInvMass);

    // StaticWorld (id 3): not broadphased (None), shape-backed, null reaction.
    const auto& world = GetCollidableTypeInfo(CollidableType::StaticWorld);
    EXPECT_EQ(world.accel, AccelStructureKind::None);
    EXPECT_EQ(world.proxy, ShapeProxyKind::ShapeBacked);
    EXPECT_EQ(world.react, ReactionProviderKind::StaticNull);
}

TEST(CodegenRoundtrip, SchemaRejectionReportsMalformedIrWithPath) {
    const auto temp = std::filesystem::temp_directory_path() / "nuka_codegen_bad_ir";
    std::filesystem::remove_all(temp);
    std::filesystem::create_directories(temp);

    const auto bad_ir = temp / "bad.yaml";
    {
        std::ofstream file(bad_ir);
        file << "row_class_name: BadRow\n";
    }

    const auto out = temp / "generated";
    const std::string command = std::string("\"") + NUKA_PYTHON_EXECUTABLE +
        "\" \"" + (SourceRoot() / "tools/codegen/regen.py").string() +
        "\" --classes-dir \"" + temp.string() +
        "\" --output-dir \"" + out.string() + "\" > \"" +
        (temp / "stdout.txt").string() + "\" 2> \"" +
        (temp / "stderr.txt").string() + "\"";

    EXPECT_NE(RunCommand(command), 0);

    std::ifstream stderr_file(temp / "stderr.txt");
    std::stringstream buffer;
    buffer << stderr_file.rdbuf();
    const std::string stderr_text = buffer.str();
    EXPECT_NE(stderr_text.find("bad.yaml"), std::string::npos);
    EXPECT_NE(stderr_text.find("missing required field"), std::string::npos);

    std::filesystem::remove_all(temp);
}
