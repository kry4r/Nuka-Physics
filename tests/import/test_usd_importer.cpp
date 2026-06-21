// ---------------------------------------------------------------------------
// Tests for nuka::import::LoadUsd
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "import/usd_stage_adapter.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path TempUsdPath(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

void CopyFileTo(const std::filesystem::path& source, const std::filesystem::path& target) {
    std::filesystem::remove(target);
    std::filesystem::copy_file(source, target);
}

std::string ExceptionTextForLoadUsd(const std::string& path) {
    try {
        (void)nuka::import::LoadUsd(path);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

} // namespace

TEST(UsdImporter, LoadsUsdPhysicsRigidBodyAndJoint) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.ShapeCount(), 2u);
}

TEST(UsdImporter, ParsesRigidBodyNamesAndMass) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    ASSERT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.GetBody(0).name, "base");
    EXPECT_EQ(scene.GetBody(1).name, "link1");
    EXPECT_TRUE(scene.GetBody(0).is_static);
    EXPECT_FLOAT_EQ(scene.GetBody(1).mass, 0.5f);
    EXPECT_FLOAT_EQ(scene.GetBody(1).inertia.x, 0.05f);
    EXPECT_FLOAT_EQ(scene.GetBody(1).inertia.z, 0.01f);
}

TEST(UsdImporter, ParsesUsdPhysicsJoint) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    ASSERT_EQ(scene.JointCount(), 1u);
    const auto& joint = scene.GetJoint(0);
    EXPECT_EQ(joint.name, "joint0");
    EXPECT_EQ(joint.type, nuka::scene::JointType::Revolute);
    EXPECT_EQ(joint.parent_body, 0u);
    EXPECT_EQ(joint.child_body, 1u);
    EXPECT_FLOAT_EQ(joint.axis.z, 1.0f);
}

TEST(UsdImporter, CookedBlobContainsUsdImportedTables) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");
    const auto blob = nuka::scene::CookScene(scene);

    EXPECT_EQ(blob.body_count, 2u);
    EXPECT_EQ(blob.joint_count, 1u);
    EXPECT_EQ(blob.shape_count, 2u);
    EXPECT_EQ(blob.material_count, 1u);
    EXPECT_EQ(blob.camera_count, 1u);
    EXPECT_EQ(blob.light_count, 1u);
    EXPECT_EQ(blob.actuator_count, 1u);
    EXPECT_EQ(blob.sensor_count, 1u);
}

TEST(UsdImporter, ParsesRenderControlAndSensorRecords) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    EXPECT_EQ(scene.MaterialCount(), 1u);
    EXPECT_EQ(scene.CameraCount(), 1u);
    EXPECT_EQ(scene.LightCount(), 1u);
    EXPECT_EQ(scene.ActuatorCount(), 1u);
    EXPECT_EQ(scene.SensorCount(), 1u);

    EXPECT_EQ(scene.GetMaterial(0).name, "mat_robot");
    EXPECT_FLOAT_EQ(scene.GetMaterial(0).base_color.x, 0.8f);
    EXPECT_EQ(scene.GetCamera(0).name, "track");
    EXPECT_EQ(scene.GetLight(0).type, nuka::scene::LightType::Directional);
    EXPECT_EQ(scene.GetActuator(0).type, nuka::scene::ActuatorType::Velocity);
    EXPECT_EQ(scene.GetSensor(0).type, nuka::scene::SensorType::Imu);
}

TEST(UsdImporter, ParsesLidarAndDepthSensorDescs) {
    using nuka::scene::SensorType;
    using nuka::scene::MountFrame;
    const auto scene = nuka::import::LoadUsd("tests/data/sensors_suite.usda");

    ASSERT_EQ(scene.SensorCount(), 2u);
    EXPECT_EQ(scene.CameraCount(), 1u);
    EXPECT_EQ(scene.GetCamera(0).name, "onboard");

    // lidar NukaSensor -> Lidar fan + range, mounted on the base body.
    const auto& lidar = scene.GetSensor(0);
    EXPECT_EQ(lidar.type, SensorType::Lidar);
    EXPECT_EQ(lidar.mount, MountFrame::Body);
    EXPECT_EQ(lidar.mount_index, 0u);
    EXPECT_EQ(lidar.lidar.az_count, 180u);
    EXPECT_EQ(lidar.lidar.el_count, 16u);
    EXPECT_NEAR(lidar.lidar.az_max, 1.570795f, 1e-4f);
    EXPECT_NEAR(lidar.lidar.az_min, -1.570795f, 1e-4f);
    EXPECT_FLOAT_EQ(lidar.lidar.min_range, 0.1f);
    EXPECT_FLOAT_EQ(lidar.lidar.max_range, 20.0f);

    // depth NukaSensor -> Depth image with width/height + vFOV.
    const auto& depth = scene.GetSensor(1);
    EXPECT_EQ(depth.type, SensorType::Depth);
    EXPECT_EQ(depth.cam.width, 160u);
    EXPECT_EQ(depth.cam.height, 120u);
    EXPECT_NEAR(depth.cam.vfov_degrees, 60.0f, 0.1f);
}

TEST(UsdImporter, DetectsUsdStageFormatsCaseInsensitively) {
    using nuka::import::DetectUsdStageFormat;
    using nuka::import::UsdStageFormat;

    EXPECT_EQ(DetectUsdStageFormat("robot.USDA"), UsdStageFormat::UsdaText);
    EXPECT_EQ(DetectUsdStageFormat("robot.usd"), UsdStageFormat::Usd);
    EXPECT_EQ(DetectUsdStageFormat("robot.USDC"), UsdStageFormat::UsdcCrate);
    EXPECT_EQ(DetectUsdStageFormat("robot.USDZ"), UsdStageFormat::UsdzPackage);
    EXPECT_EQ(DetectUsdStageFormat("robot.obj"), UsdStageFormat::Unknown);
}

TEST(UsdImporter, LoadsTextUsdExtensionThroughAdapter) {
    const auto usd_path = TempUsdPath("nuka_minimal_scene_text.usd");
    CopyFileTo("tests/data/minimal_scene.usda", usd_path);

    const auto scene = nuka::import::LoadUsd(usd_path.string());

    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.ShapeCount(), 2u);

    std::filesystem::remove(usd_path);
}

TEST(UsdImporter, NonCrateBinaryUsdReportsOpenUsdSdkAdapterRequirement) {
    // A binary .usd that is NOT a USDC crate (no PXR-USDC magic, not text) is not
    // natively readable and must report the OpenUSD SDK adapter requirement.
    const auto usd_path = TempUsdPath("nuka_binary_scene.usd");
    {
        std::ofstream out(usd_path, std::ios::binary);
        const unsigned char bytes[] = {0x00, 0x01, 0x02, 0x03, 0x7f, 0x80, 0xfe, 0xff};
        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }

    const std::string message = ExceptionTextForLoadUsd(usd_path.string());

    EXPECT_NE(message.find("OpenUSD SDK adapter"), std::string::npos);
    EXPECT_NE(message.find(".usd"), std::string::npos);

    std::filesystem::remove(usd_path);
}

TEST(UsdImporter, UsdcReadsViaCrateReaderAndUsdzRequiresOpenUsdAdapter) {
    // .usdc routes to the self-written crate reader (a malformed crate yields a
    // "USDC:" diagnostic); .usdz is not crate-readable -> OpenUSD adapter required.
    const auto usdc_path = TempUsdPath("nuka_scene.usdc");
    const auto usdz_path = TempUsdPath("nuka_scene.usdz");
    {
        std::ofstream usdc(usdc_path, std::ios::binary);
        usdc << "PXR-USDC";
        usdc.put('\0');
        std::ofstream usdz(usdz_path, std::ios::binary);
        usdz << "PK";
    }

    const std::string usdc_message = ExceptionTextForLoadUsd(usdc_path.string());
    const std::string usdz_message = ExceptionTextForLoadUsd(usdz_path.string());

    EXPECT_NE(usdc_message.find("USDC"), std::string::npos);
    EXPECT_EQ(usdc_message.find("OpenUSD SDK adapter"), std::string::npos);
    EXPECT_NE(usdz_message.find("OpenUSD SDK adapter"), std::string::npos);
    EXPECT_NE(usdz_message.find("USDZ"), std::string::npos);

    std::filesystem::remove(usdc_path);
    std::filesystem::remove(usdz_path);
}

TEST(UsdImporter, RejectsUnsupportedUsdExtensionBeforeParsing) {
    const auto txt_path = TempUsdPath("nuka_scene.txt");
    {
        std::ofstream out(txt_path);
        out << "#usda 1.0\n";
    }

    const std::string message = ExceptionTextForLoadUsd(txt_path.string());

    EXPECT_NE(message.find("unsupported USD extension"), std::string::npos);

    std::filesystem::remove(txt_path);
}

TEST(UsdImporter, ThrowsOnMissingFile) {
    EXPECT_THROW(nuka::import::LoadUsd("tests/data/missing_scene.usda"), std::runtime_error);
}
