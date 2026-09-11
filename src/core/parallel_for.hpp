#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace nuka::core {
namespace detail {

class HostParallelExecutor {
public:
    HostParallelExecutor() {
        const auto count = std::max(1u, std::thread::hardware_concurrency());
        threads_.reserve(count - 1u);
        try {
            for (size_t i = 1u; i < count; ++i) threads_.emplace_back([this, i] { Worker(i); });
        } catch (...) {
            Stop();
            throw;
        }
    }
    ~HostParallelExecutor() { Stop(); }
    HostParallelExecutor(const HostParallelExecutor&) = delete;
    HostParallelExecutor& operator=(const HostParallelExecutor&) = delete;

    void Execute(size_t count, std::function<void(size_t)> task) {
        if (count == 0u) return;
        if (Active() == this) {
            for (size_t i = 0u; i < count; ++i) task(i);
            return;
        }
        std::unique_lock<std::mutex> execution(execution_mutex_);
        const size_t participants = std::min(count, threads_.size() + 1u);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count_ = count;
            participants_ = participants;
            completed_ = 0u;
            error_ = nullptr;
            task_ = std::move(task);
            ++generation_;
        }
        work_.notify_all();
        Range(0u, count, participants);
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return completed_ == participants - 1u; });
            error = error_;
            task_ = {};
        }
        if (error) std::rethrow_exception(error);
    }

private:
    static const HostParallelExecutor*& Active() {
        static thread_local const HostParallelExecutor* executor = nullptr;
        return executor;
    }
    void Range(size_t participant, size_t count, size_t participants) {
        const auto* previous = Active();
        Active() = this;
        const size_t width = count / participants;
        const size_t remainder = count % participants;
        const size_t begin = participant * width + std::min(participant, remainder);
        const size_t end = begin + width + (participant < remainder ? 1u : 0u);
        try {
            for (size_t i = begin; i < end; ++i) task_(i);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
        Active() = previous;
    }
    void Worker(size_t participant) {
        size_t generation = 0u;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            work_.wait(lock, [&] { return stopped_ || generation != generation_; });
            if (stopped_) return;
            generation = generation_;
            if (participant >= participants_) continue;
            const size_t count = count_, participants = participants_;
            lock.unlock();
            Range(participant, count, participants);
            lock.lock();
            ++completed_;
            if (completed_ == participants - 1u) done_.notify_one();
        }
    }
    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        work_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }

    std::mutex execution_mutex_, mutex_;
    std::condition_variable work_, done_;
    std::vector<std::thread> threads_;
    std::function<void(size_t)> task_;
    std::exception_ptr error_;
    size_t generation_ = 0u, count_ = 0u, participants_ = 1u, completed_ = 0u;
    bool stopped_ = false;
};

inline HostParallelExecutor& HostExecutor() {
    static HostParallelExecutor executor;
    return executor;
}

}  // namespace detail

// Each item owns its output; accumulation within an item keeps the caller's order.
// Nested calls use the current worker, and exceptions return to the submitting thread.
template <typename Function>
void ParallelFor(size_t count, Function&& function) {
    if (count != 0u) detail::HostExecutor().Execute(count, std::forward<Function>(function));
}

}  // namespace nuka::core
