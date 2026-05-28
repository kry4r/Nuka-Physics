// ---------------------------------------------------------------------------
// PHI DeviceContext per-handle tests.
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(DeviceContext, TwoContextsIndependentConfig) {
    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

    auto contract_a = nuka::phi::PlatformContract{};
    auto contract_b = nuka::phi::PlatformContract{};
    contract_a.backend_selection_layer_enabled = false;
    contract_b.backend_selection_layer_enabled = true;

    auto context_a = nuka::phi::MakeDeviceContext(0, stream_a, contract_a);
    auto context_b = nuka::phi::MakeDeviceContext(0, stream_b, contract_b);

    EXPECT_NE(context_a.contract.backend_selection_layer_enabled,
              context_b.contract.backend_selection_layer_enabled);
    EXPECT_NE(context_a.stream.Native(), context_b.stream.Native());
    EXPECT_EQ(context_a.device_id, 0);
    EXPECT_EQ(context_b.device_id, 0);

    EXPECT_EQ(cudaStreamDestroy(stream_a), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_b), cudaSuccess);
}
