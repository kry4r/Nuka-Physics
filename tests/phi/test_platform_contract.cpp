// ---------------------------------------------------------------------------
// Platform contract tests for the production GPU simulation path.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "phi/capabilities.hpp"
#include "phi/device.hpp"
#include "phi/platform_contract.hpp"

#include <cstring>

TEST(PlatformContract, RequiresCudaForProductionPhysics) {
    const nuka::phi::PlatformContract contract{};
    EXPECT_EQ(contract.physics_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_EQ(contract.reference_backend, nuka::phi::PhysicsBackend::CpuReference);
    EXPECT_TRUE(contract.requires_gpu_simulation);
    EXPECT_TRUE(contract.cpu_reference_only);
    EXPECT_TRUE(contract.backend_selection_layer_enabled);
    EXPECT_EQ(contract.default_backend_policy, nuka::phi::BackendSelectionPolicy::PreferCuda);
}

TEST(PlatformContract, DefaultSelectionPrefersCudaOnThisWorkstation) {
    const auto selection = nuka::phi::ResolvePhysicsBackend({});
    EXPECT_EQ(selection.selected_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_TRUE(selection.production_backend);
    EXPECT_FALSE(selection.reference_backend);
    EXPECT_GE(selection.cuda_device_id, 0);
}

TEST(PlatformContract, CpuReferenceSelectionIsMarkedValidationOnly) {
    nuka::phi::BackendSelectionRequest request;
    request.policy = nuka::phi::BackendSelectionPolicy::ForceCpuReference;

    const auto selection = nuka::phi::ResolvePhysicsBackend(request);
    EXPECT_EQ(selection.selected_backend, nuka::phi::PhysicsBackend::CpuReference);
    EXPECT_FALSE(selection.production_backend);
    EXPECT_TRUE(selection.reference_backend);
}

TEST(PlatformContract, ReportsActiveCudaDeviceForPhysics) {
    const nuka::phi::PlatformContract contract{};
    EXPECT_GE(nuka::phi::GetDeviceCount(), 1);
    const auto device = nuka::phi::GetDeviceInfo(contract.cuda_device_id);
    EXPECT_GT(std::strlen(device.name), 0u);
    EXPECT_GT(device.total_memory, 0u);
}

TEST(PlatformContract, ExposesGpuExecutionCapabilities) {
    const auto caps = nuka::phi::QueryCapabilities();
    EXPECT_GE(caps.warp_size, 32);
    EXPECT_GE(caps.max_threads_per_block, 128);
    EXPECT_GT(caps.multiprocessor_count, 0);
}
