// ---------------------------------------------------------------------------
// nuka::core::perf - per-tag performance sample aggregator
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace nuka::core::perf {

namespace {

// Nearest-rank percentile over a sorted, non-empty sample vector.
// `pct` is in [0, 1]. Uses rank = ceil(pct * n) clamped to [1, n] (1-based).
double NearestRank(const std::vector<double>& sorted, double pct) {
    const std::size_t n = sorted.size();
    if (n == 0) {
        return 0.0;
    }
    double rank = std::ceil(pct * static_cast<double>(n));
    if (rank < 1.0) {
        rank = 1.0;
    }
    if (rank > static_cast<double>(n)) {
        rank = static_cast<double>(n);
    }
    const std::size_t index = static_cast<std::size_t>(rank) - 1;
    return sorted[index];
}

// Formats a microsecond value as a JSON float, guaranteeing a fractional part
// so whole numbers render as "12.0" (not "12") to match the typed contract.
std::string FormatFloat(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.10g", value);
    std::string text(buffer);
    // If it printed as a plain integer (no '.', exponent, nan or inf), make it
    // an explicit float by appending ".0".
    if (text.find_first_of(".eEni") == std::string::npos) {
        text += ".0";
    }
    return text;
}

} // namespace

PerfRecorder::PerfRecorder(std::size_t recent_window)
    : window_(recent_window == 0 ? std::size_t{1} : recent_window) {}

void PerfRecorder::AddSample(std::string_view tag, double us) {
    const std::string key(tag);
    auto it = tags_.find(key);
    if (it == tags_.end()) {
        it = tags_.emplace(key, TagData{}).first;
        order_.push_back(key);
        it->second.min_us = us;
        it->second.max_us = us;
    } else {
        it->second.min_us = std::min(it->second.min_us, us);
        it->second.max_us = std::max(it->second.max_us, us);
    }

    TagData& data = it->second;
    data.samples.push_back(us);
    data.sum += us;

    data.recent.push_back(us);
    if (data.recent.size() > window_) {
        data.recent.pop_front();
    }
}

void PerfRecorder::Reset() {
    tags_.clear();
    order_.clear();
}

TagStats PerfRecorder::ComputeStats(const std::vector<double>& samples,
                                    double sum,
                                    double min_us,
                                    double max_us) {
    TagStats stats;
    if (samples.empty()) {
        return stats;
    }
    stats.count = static_cast<std::uint64_t>(samples.size());
    stats.mean_us = sum / static_cast<double>(samples.size());
    stats.min_us = min_us;
    stats.max_us = max_us;

    std::vector<double> sorted(samples);
    std::sort(sorted.begin(), sorted.end());
    stats.p50_us = NearestRank(sorted, 0.50);
    stats.p99_us = NearestRank(sorted, 0.99);
    return stats;
}

TagStats PerfRecorder::Stats(std::string_view tag) const {
    const auto it = tags_.find(std::string(tag));
    if (it == tags_.end()) {
        return TagStats{};
    }
    const TagData& data = it->second;
    return ComputeStats(data.samples, data.sum, data.min_us, data.max_us);
}

TagStats PerfRecorder::RecentStats(std::string_view tag) const {
    const auto it = tags_.find(std::string(tag));
    if (it == tags_.end()) {
        return TagStats{};
    }
    const TagData& data = it->second;
    if (data.recent.empty()) {
        return TagStats{};
    }

    std::vector<double> window(data.recent.begin(), data.recent.end());
    double sum = 0.0;
    double min_us = window.front();
    double max_us = window.front();
    for (const double v : window) {
        sum += v;
        min_us = std::min(min_us, v);
        max_us = std::max(max_us, v);
    }
    return ComputeStats(window, sum, min_us, max_us);
}

void PerfRecorder::PrintTable(std::FILE* out) const {
    if (out == nullptr) {
        return;
    }
    std::fprintf(out,
                 "%-32s %10s %12s %12s %12s %12s %12s\n",
                 "tag", "count", "mean_us", "p50_us", "p99_us", "min_us", "max_us");
    for (const std::string& tag : order_) {
        const TagStats s = Stats(tag);
        std::fprintf(out,
                     "%-32s %10llu %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                     tag.c_str(),
                     static_cast<unsigned long long>(s.count),
                     s.mean_us,
                     s.p50_us,
                     s.p99_us,
                     s.min_us,
                     s.max_us);
    }
}

std::string PerfRecorder::DumpJson() const {
    std::string out;
    out.reserve(order_.size() * 128 + 64);
    out += "{\n";
    out += "  \"schema_version\": 1,\n";
    out += "  \"per_tag\": {";

    char buffer[64];
    bool first = true;
    for (const std::string& tag : order_) {
        const TagStats s = Stats(tag);
        out += first ? "\n" : ",\n";
        first = false;

        out += "    \"";
        out += tag;
        out += "\": { ";

        std::snprintf(buffer, sizeof(buffer), "%llu",
                      static_cast<unsigned long long>(s.count));
        out += "\"count\": ";
        out += buffer;

        out += ", \"mean_us\": ";
        out += FormatFloat(s.mean_us);

        out += ", \"p50_us\": ";
        out += FormatFloat(s.p50_us);

        out += ", \"p99_us\": ";
        out += FormatFloat(s.p99_us);

        out += ", \"min_us\": ";
        out += FormatFloat(s.min_us);

        out += ", \"max_us\": ";
        out += FormatFloat(s.max_us);

        out += " }";
    }

    if (!first) {
        out += "\n  }\n";
    } else {
        out += "}\n";
    }
    out += "}\n";
    return out;
}

} // namespace nuka::core::perf
