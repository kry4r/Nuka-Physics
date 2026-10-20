#pragma once
// Whole-window video recording: presented frames stream into an ffmpeg child.

#include "runtime/app/viewer/editor_instance.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nuka::runtime::app::viewer {

// Encodes presented frames at a fixed rate. A frame stays in the video until the next one is due,
// so the video keeps wall-clock time whatever rate the editor draws at.
class FrameRecorder {
public:
    struct Settings {
        std::string file;                  // the video it writes
        std::string codec = "h264_nvenc";  // or libx264
        double fps = 30.0;
        int quality = 18;                  // constant quality, lower is better
    };
    struct Totals {
        std::string file;
        uint64_t frames = 0u;   // video frames written
        uint64_t dropped = 0u;  // presented frames lost to a full queue or a changed window size
        double seconds = 0.0;
        uint32_t width = 0u, height = 0u;
        std::string error;
    };

    FrameRecorder() = default;
    ~FrameRecorder();
    FrameRecorder(const FrameRecorder&) = delete;
    FrameRecorder& operator=(const FrameRecorder&) = delete;

    // Begins a recording; ffmpeg starts with the first frame, once the size is known.
    void Start(Settings settings);
    bool Recording() const { return recording_; }
    // With each presented frame: 4-byte pixels, top row first.
    void Push(const uint8_t* pixels, uint32_t width, uint32_t height, bool bgra);
    // Ends the recording once ffmpeg has encoded the queue and closed the file.
    Totals Stop();
    Totals Progress();

private:
    using Clock = std::chrono::steady_clock;
    struct Frame {
        std::vector<uint8_t> pixels;
        uint64_t repeat = 0u;
    };

    void Launch(uint32_t width, uint32_t height, bool bgra);
    void Write();

    Settings settings_;
    bool recording_ = false;
    std::unique_ptr<PipedProcess> ffmpeg_;
    std::thread writer_;
    std::mutex mu_;
    std::condition_variable wake_;
    std::deque<Frame> queue_;
    std::vector<std::vector<uint8_t>> spare_;
    bool closing_ = false;
    std::atomic<bool> failed_{false};  // ffmpeg did not start or stopped reading
    Clock::time_point start_;
    uint64_t due_ = 0u;  // video frames handed to the queue
    Totals totals_;
};

}  // namespace nuka::runtime::app::viewer
