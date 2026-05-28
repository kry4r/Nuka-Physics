#include "nuka/nuka.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

} // namespace

TEST(CppWrapper, RaiiCreateStepReadDestroy) {
    auto device = nuka::Device::Create(0u, nullptr);
    ASSERT_TRUE(device.has_value()) << device.error().message();

    const auto scene = SourcePath("examples/scenes/go2_stand.usda");
    auto world = nuka::World::CreateFromScene(*device, scene.string(), 1u, 1.0f / 240.0f);
    ASSERT_TRUE(world.has_value()) << world.error().message();

    auto step_result = world->StepN(32u);
    ASSERT_TRUE(step_result.has_value()) << step_result.error().message();

    auto view = world->GetBufferView(NUKA_FIELD_JOINT_POSITION);
    ASSERT_TRUE(view.has_value()) << view.error().message();
    ASSERT_NE(view->device_ptr, nullptr);
    ASSERT_GT(view->element_count, 12u);

    std::vector<float> values(view->element_count);
    ASSERT_EQ(cudaMemcpy(values.data(),
                         view->device_ptr,
                         values.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
}
