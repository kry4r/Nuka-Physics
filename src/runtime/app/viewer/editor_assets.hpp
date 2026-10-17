#pragma once
// The asset browser's index: importable files under a set of folders, and their pictures, drawn
// off the UI thread by a small software rasterizer.

#include "runtime/app/viewer/editor_import.hpp"
#include "scene/scene_ir.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {

constexpr int kThumbWidth = 320;
constexpr int kThumbHeight = 180;

struct AssetFile {
    std::string path;  // absolute and normal
    std::string name;  // the file name
    std::string root;  // the folder it was found under
    AssetKind kind = AssetKind::Unknown;
    uint64_t bytes = 0u;
    int64_t stamp = 0;  // the file's write time, for cache keys
};

// The importable files under `roots`, by root then path; folders starting with '.' are skipped.
std::vector<AssetFile> ScanAssets(const std::vector<std::string>& roots);
// One file as an asset found under `root`; its kind is Unknown when the editor cannot import it.
AssetFile AssetAt(const std::string& path, const std::string& root);

// Draws `scene` from an oblique view into a kThumbWidth x kThumbHeight PPM; false with nothing to draw.
bool RenderThumbnail(const scene::SceneIR& scene, const std::filesystem::path& out);

// Asset pictures kept in `dir` by path, size and write time; a missing one is drawn on a worker thread.
class ThumbnailCache {
public:
    explicit ThumbnailCache(std::filesystem::path dir);
    ~ThumbnailCache();
    ThumbnailCache(const ThumbnailCache&) = delete;
    ThumbnailCache& operator=(const ThumbnailCache&) = delete;

    // The picture's file, or empty while it is drawn or when the asset has nothing to draw.
    std::string Thumbnail(const AssetFile& asset);
    // Bumps each time a picture lands.
    uint64_t Version() const { return version_.load(); }

private:
    void Run();

    std::filesystem::path dir_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::pair<AssetFile, std::filesystem::path>> queue_;
    std::unordered_map<std::string, std::string> known_;  // cache key -> picture, empty when drawn blank or queued
    bool stop_ = false;
    std::atomic<uint64_t> version_{0u};
    std::thread worker_;
};

}  // namespace nuka::runtime::app::viewer
