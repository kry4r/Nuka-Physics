// ---------------------------------------------------------------------------
// v0.5 p04 N1 -- Poisson sensor noise: D1 two-run + Poisson mean/variance
// ---------------------------------------------------------------------------
//
// The kernel adds a raw Poisson(lambda) count to the buffer. Added to a ZERO
// buffer each element is Poisson(lambda), whose mean AND variance both equal
// lambda -- the variance equality is the signature that proves it is Poisson and
// not Gaussian/constant. Asserts:
//   1. D1 two-run BIT-EXACT (memcmp identical across two launches).
//   2. STATISTICAL: sample mean ~= lambda AND sample variance ~= lambda.
//   3. INDEPENDENCE: different seed / different seq -> different noise.
//   4. Integer-valued: each sample is a non-negative integer count.
// ---------------------------------------------------------------------------

#include "sensor/noise/n1_poisson.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

namespace noise = nuka::sensor::noise;

std::vector<float> RunPoisson(const nuka::phi::DeviceContext& ctx,
                              uint32_t count, float lambda, uint64_t seed,
                              uint64_t seq) {
    std::vector<float> host(count, 0.0f);  // add-to-zero so output == count
    nuka::phi::Buffer d(static_cast<size_t>(count) * sizeof(float));
    d.CopyFromHost(host.data(), host.size() * sizeof(float));
    noise::LaunchPoissonNoise(ctx, static_cast<float*>(d.Data()), count, lambda,
                              seed, seq);
    ctx.stream.Synchronize();
    d.CopyToHost(host.data(), host.size() * sizeof(float));
    return host;
}

}  // namespace

// 1. D1: two separate launches with the same args are memcmp-identical.
TEST(N1PoissonNoise, DeterminismTwoRunBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t count = 100000u;
    const float lambda = 4.0f;
    const uint64_t seed = 0x0badf00d12345678ull;
    const uint64_t seq = 3u;

    const std::vector<float> a = RunPoisson(ctx, count, lambda, seed, seq);
    const std::vector<float> b = RunPoisson(ctx, count, lambda, seed, seq);

    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0)
        << "Poisson noise is NOT two-run byte-exact (D1 violation)";
}

// 2. Poisson mean == variance == lambda (the distinguishing signature).
TEST(N1PoissonNoise, StatisticalMeanAndVariance) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t count = 100000u;
    const float lambda = 4.0f;
    const uint64_t seed = 0xfeedface99887766ull;

    const std::vector<float> s = RunPoisson(ctx, count, lambda, seed, 0u);

    double sum = 0.0;
    for (float v : s) sum += static_cast<double>(v);
    const double sample_mean = sum / count;
    double var = 0.0;
    for (float v : s) {
        const double d = static_cast<double>(v) - sample_mean;
        var += d * d;
    }
    const double sample_var = var / count;

    // Mean and variance both ~= lambda. SE(mean) ~ sqrt(lambda/N) ~ 6.3e-3; SE on
    // the variance is larger, so allow ~10% on the variance.
    EXPECT_NEAR(sample_mean, static_cast<double>(lambda),
                5.0 * std::sqrt(static_cast<double>(lambda) / count));
    EXPECT_GT(sample_var, 0.88 * lambda);
    EXPECT_LT(sample_var, 1.12 * lambda);

    std::printf("[N1Poisson] N=%u lambda=%.4f measured(mean=%.6f var=%.6f)\n",
                count, lambda, sample_mean, sample_var);
}

// 3. Independence across seed and seq.
TEST(N1PoissonNoise, IndependenceSeedSeq) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t count = 4096u;
    const float lambda = 5.0f;

    const std::vector<float> base = RunPoisson(ctx, count, lambda, 1u, 0u);
    const std::vector<float> diff_seed = RunPoisson(ctx, count, lambda, 2u, 0u);
    const std::vector<float> diff_seq = RunPoisson(ctx, count, lambda, 1u, 1u);

    EXPECT_NE(std::memcmp(base.data(), diff_seed.data(),
                          base.size() * sizeof(float)),
              0)
        << "different seed produced identical noise";
    EXPECT_NE(
        std::memcmp(base.data(), diff_seq.data(), base.size() * sizeof(float)),
        0)
        << "different seq produced identical noise";
}

// 4. Each sample is a non-negative integer count.
TEST(N1PoissonNoise, NonNegativeIntegerCounts) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t count = 8192u;
    const float lambda = 3.0f;

    const std::vector<float> s = RunPoisson(ctx, count, lambda, 42u, 0u);
    for (float v : s) {
        EXPECT_GE(v, 0.0f);
        EXPECT_EQ(v, std::floor(v)) << "Poisson sample is not integer-valued";
    }
}
