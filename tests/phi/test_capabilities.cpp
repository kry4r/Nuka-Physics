// ---------------------------------------------------------------------------
// PHI tests – capabilities, device, buffer, stream
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "phi/capabilities.hpp"
#include "phi/device.hpp"
#include "phi/backend.hpp"  // InitBestDevice / DeviceBufferType (v2)
#include "phi/buffer.hpp"   // Buffer* / BufferAlloc / BufferUpload / BufferDownload (v2)

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

// ---- Buffer (PHI v2) -----------------------------------------------------
// BUF-11/BUF-13: the legacy RAII phi::Buffer was deleted in the M11 BUF
// sweep; this is the v2 buffer-capabilities smoke that preserves the original
// coverage intent (allocate a device buffer + host round-trip). The opaque-handle
// v2 surface is BufferAlloc(DeviceBufferType(InitBestDevice()), bytes) ->
// BufferUpload/BufferDownload -> BufferFree. (The fuller v2 buffer + op-dispatch
// gate lives in tests/scenario/phi2_smoke.cpp.)

TEST(PhiBuffer, V2AllocateAndHostRoundTrip) {
    using namespace nuka::phi;
    constexpr size_t N = 64;
    std::vector<float> src(N, 3.14f);
    std::vector<float> dst(N, 0.0f);

    Device* dev = InitBestDevice();
    ASSERT_NE(dev, nullptr);
    BufferType* bt = DeviceBufferType(dev);
    ASSERT_NE(bt, nullptr);

    Buffer* buf = BufferAlloc(bt, N * sizeof(float));
    ASSERT_NE(buf, nullptr);
    EXPECT_NE(BufferBase(buf), nullptr);

    BufferUpload(buf, src.data(), /*off=*/0, N * sizeof(float));
    BufferDownload(buf, dst.data(), /*off=*/0, N * sizeof(float));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(dst[i], 3.14f);
    }
    BufferFree(buf);
}

TEST(PhiBuffer, V2ZeroByteAllocIsSafe) {
    using namespace nuka::phi;
    Device* dev = InitBestDevice();
    ASSERT_NE(dev, nullptr);
    BufferType* bt = DeviceBufferType(dev);
    ASSERT_NE(bt, nullptr);
    // A zero-byte allocation must not crash and must be Free-able (mirrors the
    // UploadVectorV2 empty-vector path several production consumers rely on).
    Buffer* buf = BufferAlloc(bt, 0u);
    EXPECT_NO_THROW({ if (buf != nullptr) BufferFree(buf); });
}
