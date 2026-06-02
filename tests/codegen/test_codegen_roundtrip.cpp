// ---------------------------------------------------------------------------
// Row codegen round-trip tests
// ---------------------------------------------------------------------------

#include "codegen/generated/row_class_registry.hpp"

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
    const std::array<const char*, 6> files = {
        "maximal_contact_forward.cu",
        "maximal_joint_forward.cu",
        "maximal_drive_forward.cu",
        "featherstone_contact_forward.cu",
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
    using nuka::solver::generated::IsKnownRowClass;
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

    // v0.7 p08-B added the two SDF contact row classes (ids 4/5) -> 6 total.
    EXPECT_EQ(kRowClassCount, 6u);
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
    // p08-B SDF rows are FORWARD-ONLY (the IFT adjoint is p08-C).
    EXPECT_FALSE(RowClassHasAdjoint(kRigidSDFContactRowId));
    EXPECT_FALSE(RowClassHasAdjoint(kFeatherstoneSDFContactRowId));
    EXPECT_FALSE(IsKnownRowClass(99u));
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
