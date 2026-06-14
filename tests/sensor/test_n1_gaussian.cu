// ---------------------------------------------------------------------------
// v0.5 p04 N1 -- Gaussian sensor noise: Philox KAT + D1 two-run + statistics
// ---------------------------------------------------------------------------
//
// Asserts the headline determinism gate and statistical correctness of the
// counter-based Gaussian noise primitive:
//   1. PHILOX KAT: the host Philox4x32-10 reproduces the canonical Random123
//      known-answer vectors (locks the round constants -- a wrong constant would
//      still be deterministic and could pass a loose stat check, so we pin it).
//   2. D1 two-run BIT-EXACT: same (seed, count, seq) -> memcmp-identical output
//      across two separate kernel launches.
//   3. STATISTICAL sanity: over 1e5 elements, sample mean ~= mean and sample
//      stddev ~= stddev (tight tol so a degenerate/constant generator fails).
//   4. INDEPENDENCE: different seed -> different noise; same seed different seq
//      -> different noise (independence across steps); element index actually
//      enters the counter (adjacent elements differ).
// ---------------------------------------------------------------------------

#include "sensor/noise/n1_gaussian.hpp"
#include "sensor/noise/philox.cuh"
#include "phi/backend.hpp"  // InitBestDevice / DeviceBufferType
#include "phi/buffer.hpp"   // Buffer* / BufferAlloc / BufferUpload / BufferDownload
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

namespace noise = nuka::sensor::noise;

// BUF-11/BUF-13: file-local RAII over the phi-v2 opaque Buffer*, mirroring the
// legacy phi::Buffer surface (sized-ctor + Data/CopyFromHost/CopyToHost) so the
// RunGaussian body stays byte-identical after the BUF sweep. Stream-0
// DeviceBufferType -> NULL-stream transfers byte+ordering identical to the legacy
// synchronous memcpy; the noise kernel still runs on ctx.stream UNCHANGED.
class OwnedDeviceBuffer {
public:
    explicit OwnedDeviceBuffer(size_t bytes) {
        buf_ = nuka::phi::BufferAlloc(
            nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), bytes);
    }
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    void CopyFromHost(const void* src, size_t bytes) {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferUpload(buf_, src, 0, bytes);
    }
    void CopyToHost(void* dst, size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    nuka::phi::Buffer* buf_ = nullptr;
};

std::vector<float> RunGaussian(cudaStream_t ctx, int ctx_dev,
                               uint32_t count, float mean, float stddev,
                               uint64_t seed, uint64_t seq) {
    std::vector<float> host(count, 0.0f);  // add-to-zero so output == sample
    OwnedDeviceBuffer d(static_cast<size_t>(count) * sizeof(float));
    d.CopyFromHost(host.data(), host.size() * sizeof(float));
    noise::LaunchGaussianNoise(ctx, ctx_dev, static_cast<float*>(d.Data()), count, mean,
                               stddev, seed, seq);
    cudaStreamSynchronize(ctx);
    d.CopyToHost(host.data(), host.size() * sizeof(float));
    return host;
}

}  // namespace

// 1. Philox4x32-10 known-answer vectors (Random123 kat_vectors reference).
TEST(N1GaussianNoise, PhiloxKnownAnswerVectors) {
    using noise::Philox4x32_10;
    using noise::Philox4x32Counter;
    using noise::Philox4x32Key;

    // ctr = {0,0,0,0}, key = {0,0}
    {
        Philox4x32Counter c{{0u, 0u, 0u, 0u}};
        Philox4x32Key k{{0u, 0u}};
        Philox4x32Counter o = Philox4x32_10(c, k);
        EXPECT_EQ(o.v[0], 0x6627e8d5u);
        EXPECT_EQ(o.v[1], 0xe169c58du);
        EXPECT_EQ(o.v[2], 0xbc57ac4cu);
        EXPECT_EQ(o.v[3], 0x9b00dbd8u);
    }
    // ctr = all 0xffffffff, key = all 0xffffffff
    {
        Philox4x32Counter c{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
        Philox4x32Key k{{0xffffffffu, 0xffffffffu}};
        Philox4x32Counter o = Philox4x32_10(c, k);
        EXPECT_EQ(o.v[0], 0x408f276du);
        EXPECT_EQ(o.v[1], 0x41c83b0eu);
        EXPECT_EQ(o.v[2], 0xa20bc7c6u);
        EXPECT_EQ(o.v[3], 0x6d5451fdu);
    }
    // ctr = {pi, e digits...}, key = {pi digits...}  (Random123 third vector)
    {
        Philox4x32Counter c{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}};
        Philox4x32Key k{{0xa4093822u, 0x299f31d0u}};
        Philox4x32Counter o = Philox4x32_10(c, k);
        EXPECT_EQ(o.v[0], 0xd16cfe09u);
        EXPECT_EQ(o.v[1], 0x94fdccebu);
        EXPECT_EQ(o.v[2], 0x5001e420u);
        EXPECT_EQ(o.v[3], 0x24126ea1u);
    }
}

// 2. D1: two separate launches with the same args are memcmp-identical.
TEST(N1GaussianNoise, DeterminismTwoRunBitExact) {
    const cudaStream_t ctx = nullptr;  // BUF-14: stream 0
    const int ctx_dev = 0;
    const uint32_t count = 100000u;
    const float mean = 0.5f;
    const float stddev = 0.02f;
    const uint64_t seed = 0x1234abcd5678ef90ull;
    const uint64_t seq = 7u;

    const std::vector<float> a = RunGaussian(ctx, ctx_dev, count, mean, stddev, seed, seq);
    const std::vector<float> b = RunGaussian(ctx, ctx_dev, count, mean, stddev, seed, seq);

    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0)
        << "Gaussian noise is NOT two-run byte-exact (D1 violation)";
}

// 3. Statistical sanity: mean ~= mean, stddev ~= stddev over a large buffer.
TEST(N1GaussianNoise, StatisticalMeanAndStddev) {
    const cudaStream_t ctx = nullptr;  // BUF-14: stream 0
    const int ctx_dev = 0;
    const uint32_t count = 100000u;
    const float mean = 0.5f;
    const float stddev = 0.02f;
    const uint64_t seed = 0xdeadbeefcafef00dull;

    const std::vector<float> s = RunGaussian(ctx, ctx_dev, count, mean, stddev, seed, 0u);

    double sum = 0.0;
    for (float v : s) sum += static_cast<double>(v);
    const double sample_mean = sum / count;
    double var = 0.0;
    for (float v : s) {
        const double d = static_cast<double>(v) - sample_mean;
        var += d * d;
    }
    const double sample_stddev = std::sqrt(var / count);

    // Standard error of the mean ~ stddev/sqrt(N) ~ 6.3e-5; allow a generous
    // multiple. Tight enough that a constant/zero generator (mean=0,stddev=0)
    // fails both checks.
    EXPECT_NEAR(sample_mean, static_cast<double>(mean), 5.0 * stddev / std::sqrt(static_cast<double>(count)));
    // sample stddev within ~3% of the requested stddev.
    EXPECT_GT(sample_stddev, 0.97 * stddev);
    EXPECT_LT(sample_stddev, 1.03 * stddev);

    ::testing::Test::RecordProperty("sample_mean", std::to_string(sample_mean));
    ::testing::Test::RecordProperty("sample_stddev", std::to_string(sample_stddev));
    std::printf("[N1Gaussian] N=%u requested(mean=%.4f stddev=%.4f) measured(mean=%.6f stddev=%.6f)\n",
                count, mean, stddev, sample_mean, sample_stddev);
}

// 4. Independence: seed, seq, and element index all change the noise.
TEST(N1GaussianNoise, IndependenceSeedSeqElement) {
    const cudaStream_t ctx = nullptr;  // BUF-14: stream 0
    const int ctx_dev = 0;
    const uint32_t count = 4096u;
    const float mean = 0.0f;
    const float stddev = 1.0f;

    const std::vector<float> base = RunGaussian(ctx, ctx_dev, count, mean, stddev, 1u, 0u);
    const std::vector<float> diff_seed =
        RunGaussian(ctx, ctx_dev, count, mean, stddev, 2u, 0u);
    const std::vector<float> diff_seq =
        RunGaussian(ctx, ctx_dev, count, mean, stddev, 1u, 1u);

    // Different seed -> different noise (buffers must differ).
    EXPECT_NE(std::memcmp(base.data(), diff_seed.data(),
                          base.size() * sizeof(float)),
              0)
        << "different seed produced identical noise";
    // Same seed, different seq -> different noise (independence across steps).
    EXPECT_NE(
        std::memcmp(base.data(), diff_seq.data(), base.size() * sizeof(float)),
        0)
        << "different seq produced identical noise";

    // Element index enters the counter: adjacent elements differ (would be
    // constant within a launch if idx did NOT enter the counter).
    bool all_same = true;
    for (uint32_t i = 1u; i < count; ++i) {
        if (base[i] != base[0]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "all elements identical -> idx not in counter";
}
