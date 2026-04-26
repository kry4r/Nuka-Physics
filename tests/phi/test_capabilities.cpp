// ---------------------------------------------------------------------------
// PHI tests – capabilities, device, buffer, stream
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "phi/capabilities.hpp"
#include "phi/device.hpp"
#include "phi/buffer.hpp"
#include "phi/stream.hpp"

#include <cstring>
#include <vector>

// ---- Capabilities --------------------------------------------------------

TEST(PhiCapabilities, ReportsBasicRuntimeProperties) {
    const auto caps = nuka::phi::QueryCapabilities();
    EXPECT_GT(caps.warp_size, 0);
    EXPECT_GE(caps.max_threads_per_block, 32);
}

// ---- Device --------------------------------------------------------------

TEST(PhiDevice, DeviceCountIsPositive) {
    EXPECT_GT(nuka::phi::GetDeviceCount(), 0);
}

TEST(PhiDevice, DeviceInfoNameNotEmpty) {
    auto info = nuka::phi::GetDeviceInfo(0);
    EXPECT_GT(std::strlen(info.name), 0u);
    EXPECT_GT(info.total_memory, 0u);
    EXPECT_EQ(info.device_id, 0);
}

// ---- Buffer --------------------------------------------------------------

TEST(PhiBuffer, AllocateAndHostRoundTrip) {
    constexpr size_t N = 64;
    std::vector<float> src(N, 3.14f);
    std::vector<float> dst(N, 0.0f);

    nuka::phi::Buffer buf(N * sizeof(float), nuka::phi::MemoryKind::Device);
    EXPECT_EQ(buf.Size(), N * sizeof(float));
    EXPECT_NE(buf.Data(), nullptr);

    buf.CopyFromHost(src.data(), N * sizeof(float));
    buf.CopyToHost(dst.data(), N * sizeof(float));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(dst[i], 3.14f);
    }
}

TEST(PhiBuffer, MoveSemantics) {
    nuka::phi::Buffer a(128, nuka::phi::MemoryKind::Device);
    EXPECT_NE(a.Data(), nullptr);

    nuka::phi::Buffer b(std::move(a));
    EXPECT_NE(b.Data(), nullptr);
    EXPECT_EQ(a.Data(), nullptr);
    EXPECT_EQ(a.Size(), 0u);
}

// ---- Stream --------------------------------------------------------------

TEST(PhiStream, CreateAndSync) {
    nuka::phi::Stream stream;
    EXPECT_NE(stream.NativeHandle(), nullptr);
    EXPECT_NO_THROW(stream.Synchronize());
}

TEST(PhiStream, MoveSemantics) {
    nuka::phi::Stream a;
    void* handle = a.NativeHandle();
    EXPECT_NE(handle, nullptr);

    nuka::phi::Stream b(std::move(a));
    EXPECT_EQ(b.NativeHandle(), handle);
    EXPECT_EQ(a.NativeHandle(), nullptr);
}
