#pragma once
// ---------------------------------------------------------------------------
// nuka::core::perf - scoped CUDA event timer
//
// CUDA-aware (includes <cuda_runtime.h>); confined to translation units that
// already depend on CUDA. The aggregator it feeds (PerfRecorder) is pure C++
// and lives in core/perf/perf_recorder.hpp. Header-only by design so any CUDA
// TU can drop NUKA_CUDA_TIME(...) around a kernel launch without extra linkage.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"

#include <cuda_runtime.h>

namespace nuka::core::perf {

// RAII timer that brackets a scope with a pair of CUDA events on `stream`.
//
// Construction records a start event; destruction records a stop event,
// synchronizes it, computes the elapsed time, and feeds it (in microseconds)
// to `recorder` under `tag`. The destructor never throws: on any CUDA failure
// it silently drops the sample. Events are always cleaned up.
class ScopedCudaTimer {
public:
    ScopedCudaTimer(PerfRecorder& recorder, const char* tag, cudaStream_t stream)
        : recorder_(recorder), tag_(tag), stream_(stream) {
        if (cudaEventCreate(&start_) != cudaSuccess) {
            start_ = nullptr;
        }
        if (cudaEventCreate(&stop_) != cudaSuccess) {
            stop_ = nullptr;
        }
        if (start_ != nullptr) {
            if (cudaEventRecord(start_, stream_) != cudaSuccess) {
                armed_ = false;
            }
        } else {
            armed_ = false;
        }
    }

    // Detail-gated timer: when `is_detail` is true the scope only arms if the
    // recorder has fine-grained ("detail") instrumentation enabled. With detail
    // OFF this constructs NO CUDA events and issues NO cudaEventRecord /
    // cudaEventSynchronize -- it is a true no-op, so the production step path pays
    // nothing for these deep per-sub-stage timers when profiling is disabled.
    ScopedCudaTimer(PerfRecorder& recorder, const char* tag, cudaStream_t stream,
                    bool is_detail)
        : recorder_(recorder), tag_(tag), stream_(stream) {
        if (is_detail && !recorder.DetailEnabled()) {
            armed_ = false;  // disabled: no events created, no work issued.
            return;
        }
        if (cudaEventCreate(&start_) != cudaSuccess) {
            start_ = nullptr;
        }
        if (cudaEventCreate(&stop_) != cudaSuccess) {
            stop_ = nullptr;
        }
        if (start_ != nullptr) {
            if (cudaEventRecord(start_, stream_) != cudaSuccess) {
                armed_ = false;
            }
        } else {
            armed_ = false;
        }
    }

    ~ScopedCudaTimer() {
        if (armed_ && stop_ != nullptr) {
            if (cudaEventRecord(stop_, stream_) == cudaSuccess &&
                cudaEventSynchronize(stop_) == cudaSuccess) {
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) == cudaSuccess) {
                    recorder_.AddSample(tag_,
                                        static_cast<double>(elapsed_ms) * 1000.0);
                }
            }
        }
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    ScopedCudaTimer(const ScopedCudaTimer&) = delete;
    ScopedCudaTimer& operator=(const ScopedCudaTimer&) = delete;
    ScopedCudaTimer(ScopedCudaTimer&&) = delete;
    ScopedCudaTimer& operator=(ScopedCudaTimer&&) = delete;

private:
    PerfRecorder& recorder_;
    const char* tag_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    bool armed_ = true;
};

// Two-level concatenation so __LINE__ expands before pasting.
#define NUKA_PERF_CONCAT_INNER(a, b) a##b
#define NUKA_PERF_CONCAT(a, b) NUKA_PERF_CONCAT_INNER(a, b)
#define NUKA_UNIQUE_NAME(prefix) NUKA_PERF_CONCAT(prefix, __LINE__)

// Creates a uniquely-named ScopedCudaTimer for the enclosing scope.
#define NUKA_CUDA_TIME(recorder, tag, stream)                                 \
    ::nuka::core::perf::ScopedCudaTimer NUKA_UNIQUE_NAME(nuka_cuda_timer_)(    \
        (recorder), (tag), (stream))

// Fine-grained ("detail") variant: the scope is a true no-op (no CUDA events,
// no syncs) unless `recorder.DetailEnabled()` is set. Use this for the deep
// per-sub-stage breakdown so the production step path pays nothing when
// profiling is disabled.
#define NUKA_CUDA_TIME_DETAIL(recorder, tag, stream)                          \
    ::nuka::core::perf::ScopedCudaTimer NUKA_UNIQUE_NAME(nuka_cuda_timer_)(    \
        (recorder), (tag), (stream), true)

} // namespace nuka::core::perf
