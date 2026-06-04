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

TEST(CodegenRoundtrip, GeneratedFilesCarryDoNotEditHeader) {
    const auto generated = SourceRoot() / "src/codegen/generated";
    const std::array<const char*, 16> files = {
        "maximal_contact_forward.cu",
        "maximal_joint_forward.cu",
        "maximal_drive_forward.cu",
        "featherstone_contact_forward.cu",
        "xpbd_distance_forward.cu",
        "xpbd_distance_adjoint.cuh",
        // v0.7 p09-B: bend (id 7) + volume (id 8) forward + adjoint pairs.
        "xpbd_bend_forward.cu",
        "xpbd_bend_adjoint.cuh",
        "xpbd_volume_forward.cu",
        "xpbd_volume_adjoint.cuh",
        // v0.7 p09-C: shape-match (id 9) forward + adjoint pair.
        "xpbd_shape_match_forward.cu",
        "xpbd_shape_match_adjoint.cuh",
        // v0.7 p11 (K3): particle-particle contact (id 10) forward + adjoint pair.
        "particle_particle_contact_forward.cu",
        "particle_particle_contact_adjoint.cuh",
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
    using nuka::solver::generated::kRowClassCount;
    using nuka::solver::generated::kXPBDBendRowId;
    using nuka::solver::generated::kParticleParticleContactRowId;
    using nuka::solver::generated::kXPBDDistanceRowId;
    using nuka::solver::generated::kXPBDShapeMatchRowId;
    using nuka::solver::generated::kXPBDVolumeRowId;

    // v0.7 p08-B added the two SDF contact row classes (ids 4/5); p09-A added the
    // XPBD soft-body distance row (id 6); p09-B added the XPBD bend (id 7) +
    // volume (id 8) rows; p09-C added the XPBD shape-match (id 9) row; p11 adds the
    // cross-system particle-particle contact row (id 10) -> 11 total.
    EXPECT_EQ(kRowClassCount, 11u);
    EXPECT_TRUE(IsKnownRowClass(kMaximalContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kMaximalJointRowId));
    EXPECT_TRUE(IsKnownRowClass(kMaximalDriveRowId));
    EXPECT_TRUE(IsKnownRowClass(kFeatherstoneContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kRigidSDFContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kFeatherstoneSDFContactRowId));
    EXPECT_TRUE(IsKnownRowClass(kXPBDDistanceRowId));
    EXPECT_EQ(kRigidSDFContactRowId, 4u);
    EXPECT_EQ(kFeatherstoneSDFContactRowId, 5u);
    EXPECT_EQ(kXPBDDistanceRowId, 6u);
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

    // v0.7 p09-A: the XPBD soft-body distance row (id 6) ships a GENUINE
    // dispatchable per-row adjoint -- this is exit-crit 6 and what distinguishes
    // it from the FORWARD-ONLY SDF rows above. This block is the contract
    // tripwire: RowClassHasAdjoint(6) MUST stay true and the gradient mode MUST
    // stay dense_adjoint. If the XPBD adjoint is ever removed/changed, this fails
    // consciously rather than silently degrading to a stop-gradient.
    EXPECT_STREQ(RowClassName(kXPBDDistanceRowId), "XPBDDistanceRow");
    EXPECT_EQ(RowClassMaxRowsPerBlock(kXPBDDistanceRowId), 6u);
    EXPECT_TRUE(RowClassHasAdjoint(kXPBDDistanceRowId));
    EXPECT_EQ(RowClassGradientMode(kXPBDDistanceRowId),
              static_cast<uint8_t>(GradientMode::dense_adjoint));

    // v0.7 p09-B: the XPBD BEND (id 7) + VOLUME (id 8) soft rows. Both ship a
    // GENUINE dispatchable per-row adjoint -- exit-crit 6, same as the distance
    // row. This block is the contract tripwire: RowClassHasAdjoint(7)/(8) MUST
    // stay true and the gradient mode MUST stay dense_adjoint. If either XPBD
    // adjoint is ever removed/changed, this fails consciously rather than
    // silently degrading to a stop-gradient.
    EXPECT_TRUE(IsKnownRowClass(kXPBDBendRowId));
    EXPECT_TRUE(IsKnownRowClass(kXPBDVolumeRowId));
    EXPECT_EQ(kXPBDBendRowId, 7u);
    EXPECT_EQ(kXPBDVolumeRowId, 8u);
    EXPECT_STREQ(RowClassName(kXPBDBendRowId), "XPBDBendRow");
    EXPECT_STREQ(RowClassName(kXPBDVolumeRowId), "XPBDVolumeRow");
    EXPECT_TRUE(RowClassHasAdjoint(kXPBDBendRowId));
    EXPECT_TRUE(RowClassHasAdjoint(kXPBDVolumeRowId));
    EXPECT_EQ(RowClassGradientMode(kXPBDBendRowId),
              static_cast<uint8_t>(GradientMode::dense_adjoint));
    EXPECT_EQ(RowClassGradientMode(kXPBDVolumeRowId),
              static_cast<uint8_t>(GradientMode::dense_adjoint));

    // v0.7 p09-C: the XPBD SHAPE-MATCH (id 9) soft row -- the meshless cluster
    // regularizer (Mueller et al. 2005), the 4th and hardest XPBD row class. It
    // ships a GENUINE dispatchable per-row adjoint -- exit-crit 6 -- but, unlike
    // distance/bend/volume, NOT the scalar XPBD multiplier law: shape matching has
    // no scalar constraint C and no Lagrange multiplier, so its dispatchable
    // adjoint is the per-particle GOAL PROJECTION
    // (position_new = position + stiffness*(goal - position)), multilinear in
    // {position, goal, stiffness} and validated by test_adjoint_fd_xpbd_shape_match.
    // The polar-decomposition rotation R that builds the goal (and its derivative
    // dR/dA) is DOWNSTREAM geometry, host-FD validated in the cluster sim test
    // (test_xpbd_shape_match), the analog of the volume row's grad-C check. This
    // block is the contract tripwire: RowClassHasAdjoint(9) MUST stay true and the
    // gradient mode MUST stay dense_adjoint. If the adjoint is ever removed/changed,
    // this fails consciously rather than silently degrading to a stop-gradient.
    EXPECT_TRUE(IsKnownRowClass(kXPBDShapeMatchRowId));
    EXPECT_EQ(kXPBDShapeMatchRowId, 9u);
    EXPECT_STREQ(RowClassName(kXPBDShapeMatchRowId), "XPBDShapeMatchRow");
    EXPECT_TRUE(RowClassHasAdjoint(kXPBDShapeMatchRowId));
    EXPECT_EQ(RowClassGradientMode(kXPBDShapeMatchRowId),
              static_cast<uint8_t>(GradientMode::dense_adjoint));

    // v0.7 p11 (K3): the cross-system particle-particle CONTACT row (id 10) -- the
    // v0.7 infrastructure deliverable for cross-system coupling. It ships a GENUINE
    // dispatchable per-row adjoint -- the SAME multilinear XPBD multiplier law as
    // the distance/volume rows (dense_adjoint), NOT the forward-only posture of the
    // id4/id5 SDF rows. This block is the contract tripwire: RowClassHasAdjoint(10)
    // MUST stay true and the gradient mode MUST stay dense_adjoint. If the adjoint
    // is ever removed/changed, this fails consciously rather than silently
    // degrading to a stop-gradient.
    EXPECT_TRUE(IsKnownRowClass(kParticleParticleContactRowId));
    EXPECT_EQ(kParticleParticleContactRowId, 10u);
    EXPECT_STREQ(RowClassName(kParticleParticleContactRowId),
                 "ParticleParticleContactRow");
    EXPECT_EQ(RowClassMaxRowsPerBlock(kParticleParticleContactRowId), 6u);
    EXPECT_TRUE(RowClassHasAdjoint(kParticleParticleContactRowId));
    EXPECT_EQ(RowClassGradientMode(kParticleParticleContactRowId),
              static_cast<uint8_t>(GradientMode::dense_adjoint));

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

    // v0.8 has 4 collidable types (v0.9 adds 4 more); enum values are the
    // contiguous range 0..3 (the table is indexed by these, so this is a contract
    // tripwire -- if a type is added/reordered, update this consciously).
    EXPECT_EQ(kCollidableTypeCount, 4u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::RigidBody), 0u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::ArticulationLink), 1u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::Particle), 2u);
    EXPECT_EQ(static_cast<uint8_t>(CollidableType::StaticWorld), 3u);

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
