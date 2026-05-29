// ---------------------------------------------------------------------------
// Contract test for in-engine perf timing infrastructure (Task 3.1.1).
//
// 1. PerfRecorder math + DumpJson() JSON contract (host-only path).
// 2. ScopedCudaTimer / NUKA_CUDA_TIME capturing a real kernel launch.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"
#include "core/perf/timing.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace {

using nuka::core::perf::PerfRecorder;
using nuka::core::perf::TagStats;

// Extracts the JSON number following the first occurrence of "key" in `json`.
// Minimal targeted parse so the test does not pull in a JSON dependency.
double ExtractNumber(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const std::size_t key_pos = json.find(needle);
    EXPECT_NE(key_pos, std::string::npos) << "key not found: " << key;
    if (key_pos == std::string::npos) {
        return 0.0;
    }
    std::size_t pos = key_pos + needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) {
        ++pos;
    }
    return std::strtod(json.c_str() + pos, nullptr);
}

// Trivial busy kernel so the timed scope measures real GPU work.
__global__ void BusyKernel(int* out, int iters) {
    int acc = 0;
    for (int i = 0; i < iters; ++i) {
        acc += (i * 3 + 1) % 7;
    }
    if (out != nullptr) {
        *out = acc;
    }
}

} // namespace

TEST(PerfRecorderMath, AggregatesAndDumpsJsonContract) {
    PerfRecorder recorder;
    // Known sample set (microseconds): n=5.
    const double samples[] = {8.1, 11.0, 12.0, 40.2, 51.0};
    for (const double s : samples) {
        recorder.AddSample("kernelA", s);
    }
    // A second tag, recorded later, to verify insertion order in the JSON.
    recorder.AddSample("kernelB", 100.0);

    const TagStats a = recorder.Stats("kernelA");
    EXPECT_EQ(a.count, 5u);
    EXPECT_NEAR(a.mean_us, 24.46, 1e-9);   // 122.3 / 5
    EXPECT_NEAR(a.p50_us, 12.0, 1e-9);     // nearest-rank ceil(0.5*5)=3 -> idx2
    EXPECT_NEAR(a.p99_us, 51.0, 1e-9);     // nearest-rank ceil(0.99*5)=5 -> idx4
    EXPECT_NEAR(a.min_us, 8.1, 1e-9);
    EXPECT_NEAR(a.max_us, 51.0, 1e-9);

    const std::string json = recorder.DumpJson();

    // schema_version must be the integer 1 (no trailing ".0").
    EXPECT_NE(json.find("\"schema_version\": 1"), std::string::npos);

    // Insertion order: kernelA appears before kernelB.
    const std::size_t pos_a = json.find("\"kernelA\"");
    const std::size_t pos_b = json.find("\"kernelB\"");
    ASSERT_NE(pos_a, std::string::npos);
    ASSERT_NE(pos_b, std::string::npos);
    EXPECT_LT(pos_a, pos_b);

    // Parse the numeric fields of kernelA back out and compare.
    EXPECT_NEAR(ExtractNumber(json, "count"), 5.0, 1e-9);
    EXPECT_NEAR(ExtractNumber(json, "mean_us"), 24.46, 1e-6);
    EXPECT_NEAR(ExtractNumber(json, "p50_us"), 12.0, 1e-6);
    EXPECT_NEAR(ExtractNumber(json, "p99_us"), 51.0, 1e-6);
    EXPECT_NEAR(ExtractNumber(json, "min_us"), 8.1, 1e-6);
    EXPECT_NEAR(ExtractNumber(json, "max_us"), 51.0, 1e-6);

    // No trailing commas before closing braces.
    EXPECT_EQ(json.find(",}"), std::string::npos);
    EXPECT_EQ(json.find(", }\n}"), std::string::npos);
    EXPECT_EQ(json.find(",\n  }"), std::string::npos);
}

TEST(ScopedCudaTimer, CapturesPositiveFiniteSampleForKernel) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    int* device_out = nullptr;
    ASSERT_EQ(cudaMalloc(&device_out, sizeof(int)), cudaSuccess);

    PerfRecorder recorder;
    {
        NUKA_CUDA_TIME(recorder, "busy_kernel", stream);
        BusyKernel<<<64, 128, 0, stream>>>(device_out, 4096);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    } // timer destructor records stop, syncs, and feeds the sample here.

    const TagStats stats = recorder.Stats("busy_kernel");
    EXPECT_EQ(stats.count, 1u);
    EXPECT_GT(stats.mean_us, 0.0);
    EXPECT_TRUE(std::isfinite(stats.mean_us));
    EXPECT_GT(stats.max_us, 0.0);

    EXPECT_EQ(cudaFree(device_out), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
