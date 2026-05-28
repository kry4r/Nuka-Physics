#pragma once
// ---------------------------------------------------------------------------
// nuka::core::diagnostics - invariant trace sinks
// ---------------------------------------------------------------------------

#include "core/diagnostics/invariants.hpp"

#include <cstdio>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace nuka::core::diagnostics {

class TraceSink {
public:
    virtual ~TraceSink() = default;
    virtual void OnSample(const InvariantSample& sample) = 0;
    virtual void Flush() = 0;
};

class NullTraceSink final : public TraceSink {
public:
    void OnSample(const InvariantSample& sample) override;
    void Flush() override;
};

class CsvTraceSink final : public TraceSink {
public:
    explicit CsvTraceSink(std::string_view path);
    ~CsvTraceSink() override;

    CsvTraceSink(const CsvTraceSink&) = delete;
    CsvTraceSink& operator=(const CsvTraceSink&) = delete;

    void OnSample(const InvariantSample& sample) override;
    void Flush() override;

private:
    std::FILE* file_ = nullptr;
};

class InMemoryRingTraceSink final : public TraceSink {
public:
    explicit InMemoryRingTraceSink(uint32_t capacity = 256);

    void OnSample(const InvariantSample& sample) override;
    void Flush() override;

    std::vector<InvariantSample> Samples() const;

private:
    uint32_t capacity_ = 0;
    std::deque<InvariantSample> samples_;
};

} // namespace nuka::core::diagnostics
