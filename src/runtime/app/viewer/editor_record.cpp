#include "runtime/app/viewer/editor_record.hpp"

#include "runtime/app/viewer/editor_profile.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

// Presented frames waiting for ffmpeg to encode them.
constexpr size_t kQueueFrames = 8u;

std::string LogPath(const std::string& file) { return file + ".log"; }

// The last line ffmpeg logged, which names what went wrong.
std::string LastLine(const std::string& log) {
    std::ifstream in(Utf8Path(log), std::ios::binary);
    std::string line, last;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) last = line;
    }
    return last;
}

}  // namespace

FrameRecorder::~FrameRecorder() {
    if (recording_) Stop();
}

void FrameRecorder::Start(Settings settings) {
    settings_ = std::move(settings);
    totals_ = Totals{};
    totals_.file = settings_.file;
    totals_.timing = settings_.fixed_frames ? "frames" : "wall";
    queue_.clear();
    due_ = 0u;
    closing_ = false;
    failed_ = false;
    recording_ = true;
}

void FrameRecorder::Launch(uint32_t width, uint32_t height, bool bgra) {
    const std::string log = LogPath(settings_.file);
    std::error_code ec;
    std::filesystem::remove(Utf8Path(log), ec);
    std::ostringstream rate;
    rate << settings_.fps;
    const std::string quality = std::to_string(settings_.quality);
    std::vector<std::string> argv = {"ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                                     bgra ? "bgra" : "rgba", "-s",
                                     std::to_string(width) + "x" + std::to_string(height), "-framerate", rate.str(),
                                     "-i", "-", "-c:v", settings_.codec};
    if (settings_.codec == "h264_nvenc")
        argv.insert(argv.end(), {"-preset", "p5", "-rc", "vbr", "-cq", quality, "-b:v", "0"});
    else
        argv.insert(argv.end(), {"-preset", "veryfast", "-crf", quality});
    argv.insert(argv.end(), {"-pix_fmt", "yuv420p", "-movflags", "+faststart", settings_.file});
    std::string error;
    ffmpeg_ = std::make_unique<PipedProcess>(argv, log, &error);
    if (!ffmpeg_->Started()) {
        ffmpeg_.reset();
        totals_.error = error + "; recording runs ffmpeg from the PATH";
        failed_ = true;
        return;
    }
    totals_.width = width;
    totals_.height = height;
    start_ = Clock::now();
    writer_ = std::thread([this] { Write(); });
}

void FrameRecorder::Push(const uint8_t* pixels, uint32_t width, uint32_t height, bool bgra) {
    if (!recording_ || failed_) return;
    if (!ffmpeg_) {
        Launch(width, height, bgra);
        if (!ffmpeg_) return;
    }
    std::unique_lock<std::mutex> lock(mu_);
    if (width != totals_.width || height != totals_.height) {
        ++totals_.dropped;
        return;
    }
    if (settings_.fixed_frames) {
        wake_.wait(lock, [this] { return queue_.size() < kQueueFrames || failed_; });
        if (failed_) return;
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
    const uint64_t due = settings_.fixed_frames ? due_ + 1u :
                         static_cast<uint64_t>(std::floor(elapsed * settings_.fps)) + 1u;
    if (due <= due_) return;
    const uint64_t repeat = due - due_;
    due_ = due;
    // A full queue keeps its newest frame on for the time this one would have shown.
    if (queue_.size() >= kQueueFrames) {
        queue_.back().repeat += repeat;
        ++totals_.dropped;
        return;
    }
    std::vector<uint8_t> buffer;
    if (!spare_.empty()) {
        buffer = std::move(spare_.back());
        spare_.pop_back();
    }
    lock.unlock();
    const size_t bytes = static_cast<size_t>(width) * height * 4u;
    buffer.resize(bytes);
    std::memcpy(buffer.data(), pixels, bytes);
    lock.lock();
    queue_.push_back(Frame{std::move(buffer), repeat});
    lock.unlock();
    wake_.notify_one();
}

void FrameRecorder::Write() {
    for (;;) {
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(mu_);
            wake_.wait(lock, [this] { return closing_ || !queue_.empty(); });
            if (queue_.empty()) return;
            frame = std::move(queue_.front());
            queue_.pop_front();
            wake_.notify_all();
        }
        bool ok = true;
        for (uint64_t i = 0u; i < frame.repeat && ok; ++i)
            ok = ffmpeg_->Write(frame.pixels.data(), frame.pixels.size());
        std::lock_guard<std::mutex> lock(mu_);
        if (!ok) {
            failed_ = true;
            queue_.clear();
            wake_.notify_all();
            return;
        }
        totals_.frames += frame.repeat;
        spare_.push_back(std::move(frame.pixels));
    }
}

FrameRecorder::Totals FrameRecorder::Stop() {
    recording_ = false;
    if (ffmpeg_) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closing_ = true;
        }
        wake_.notify_one();
        writer_.join();
        const int code = ffmpeg_->Finish();
        ffmpeg_.reset();
        const std::string log = LogPath(settings_.file);
        if (code != 0) {
            const std::string why = LastLine(log);
            totals_.error = "ffmpeg exited with code " + std::to_string(code) + (why.empty() ? "" : ": " + why);
        } else {
            std::error_code ec;
            std::filesystem::remove(Utf8Path(log), ec);
        }
    } else if (totals_.error.empty()) {
        totals_.error = "no frame was drawn while recording";
    }
    spare_.clear();
    return Progress();
}

FrameRecorder::Totals FrameRecorder::Progress() {
    std::lock_guard<std::mutex> lock(mu_);
    Totals t = totals_;
    t.seconds = static_cast<double>(t.frames) / settings_.fps;
    return t;
}

}  // namespace nuka::runtime::app::viewer
