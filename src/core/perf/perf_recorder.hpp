#pragma once
// ---------------------------------------------------------------------------
// nuka::core::perf - per-tag performance sample aggregator
//
// PerfRecorder is host-only pure C++ (NO CUDA include) so any translation unit
// can link it. CUDA-aware timing helpers live in core/perf/timing.hpp.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nuka::core::perf {

// Aggregated statistics for a single tag. All durations are microseconds.
struct TagStats {
    std::uint64_t count = 0;
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

// Aggregates timing samples grouped by tag.
//
// Per-tag it keeps the full sample history (for exact percentiles) plus a
// bounded sliding window of the most recent samples for "recent" stats. The
// JSON contract and PrintTable report full-history statistics.
class PerfRecorder {
public:
    // Default sliding-window capacity for recent-sample statistics.
    static constexpr std::size_t kDefaultWindow = 256;

    PerfRecorder() = default;
    explicit PerfRecorder(std::size_t recent_window);

    // Feeds a single sample (microseconds) to the recorder under `tag`.
    void AddSample(std::string_view tag, double us);

    // Clears all recorded tags and samples.
    void Reset();

    // Number of distinct tags recorded.
    std::size_t TagCount() const { return order_.size(); }

    // Full-history aggregated statistics for `tag` (nearest-rank percentiles).
    // Returns a zero-initialized TagStats if the tag was never recorded.
    TagStats Stats(std::string_view tag) const;

    // Statistics over the bounded sliding window of most-recent samples.
    TagStats RecentStats(std::string_view tag) const;

    // Prints a readable per-tag table (full-history stats) to `out`.
    void PrintTable(std::FILE* out = stdout) const;

    // Serializes full-history stats to the stable JSON contract (schema v1).
    // Tags appear in insertion order; numbers are emitted as JSON numbers.
    std::string DumpJson() const;

private:
    struct TagData {
        std::vector<double> samples;       // full history
        std::deque<double> recent;         // bounded sliding window
        double sum = 0.0;
        double min_us = 0.0;
        double max_us = 0.0;
    };

    static TagStats ComputeStats(const std::vector<double>& samples,
                                 double sum,
                                 double min_us,
                                 double max_us);

    std::size_t window_ = kDefaultWindow;
    std::unordered_map<std::string, TagData> tags_;
    std::vector<std::string> order_;       // insertion order of tags
};

} // namespace nuka::core::perf
