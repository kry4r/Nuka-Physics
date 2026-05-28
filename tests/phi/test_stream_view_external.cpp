// ---------------------------------------------------------------------------
// PHI StreamView external lifecycle tests.
// ---------------------------------------------------------------------------

#include "phi/stream_view.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

__global__ void WriteSentinelKernel(int* value) {
    *value = 42;
}

} // namespace

TEST(PhiStreamView, WrapsExternalStreamWithoutTakingOwnership) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    int* device_value = nullptr;
    ASSERT_EQ(cudaMalloc(&device_value, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_value, 0, sizeof(int), stream), cudaSuccess);

    {
        const nuka::phi::StreamView view{stream};
        ASSERT_FALSE(view.IsNull());
        WriteSentinelKernel<<<1, 1, 0, view.Native()>>>(device_value);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_NO_THROW(view.Synchronize());
    }

    int host_value = 0;
    EXPECT_EQ(cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(host_value, 42);
    EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);

    EXPECT_EQ(cudaFree(device_value), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
