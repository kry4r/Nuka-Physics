#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#if defined(__linux__)
#include <time.h>
#endif

namespace nuka::perf {

// Elapsed measurements use an unadjusted host clock where the platform exposes one.
struct MeasurementClock {
    using rep = int64_t;
    using period = std::nano;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<MeasurementClock>;
    static constexpr bool is_steady = true;

    static const char* Name() {
#if defined(__linux__)
        return "CLOCK_MONOTONIC_RAW";
#else
        return "std::chrono::steady_clock";
#endif
    }

    static time_point now() {
#if defined(__linux__)
        timespec value{};
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0)
            throw std::runtime_error("Cannot read the unadjusted monotonic clock");
        return time_point{std::chrono::seconds(value.tv_sec) + std::chrono::nanoseconds(value.tv_nsec)};
#else
        return time_point{std::chrono::duration_cast<duration>(std::chrono::steady_clock::now().time_since_epoch())};
#endif
    }
};

}  // namespace nuka::perf
