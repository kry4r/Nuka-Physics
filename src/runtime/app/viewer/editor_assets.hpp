#pragma once
// The asset browser's index: importable files under a set of folders, and their pictures, drawn
// off the UI thread by a small software rasterizer.

#include "runtime/app/viewer/editor_import.hpp"
#include "rt/material.hpp"
#include "scene/scene_ir.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
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
// Fits `image` into a `width` x `height` PPM on the thumbnails' backdrop; false when it is empty.
bool RenderImage(const rt::Texture& image, int width, int height, const std::filesystem::path& out);
// A lit ball wearing `m`, its maps loaded from their files, as a `size` x `size` PPM.
bool RenderMaterialBall(const scene::RenderMaterial& m, int size, const std::filesystem::path& out);

// Pictures kept in `dir` by what they show; a missing one is drawn on a worker thread.
// An image asset's picture is `image_width` x `image_height`.
class ThumbnailCache {
public:
    explicit ThumbnailCache(std::filesystem::path dir, int image_width = kThumbWidth, int image_height = kThumbHeight);
    ~ThumbnailCache();
    ThumbnailCache(const ThumbnailCache&) = delete;
    ThumbnailCache& operator=(const ThumbnailCache&) = delete;

    // The picture's file, or empty while it is drawn or when the asset has nothing to draw.
    std::string Thumbnail(const AssetFile& asset);
    // A `size`-pixel ball wearing `m`; `key` changes whenever its fields or its maps' files do.
    std::string Ball(const scene::RenderMaterial& m, const std::string& key, int size);
    // Bumps each time a picture lands.
    uint64_t Version() const { return version_.load(); }

private:
    using Draw = std::function<bool(const std::filesystem::path&)>;
    // The picture `key` names, queued for `draw` when it is not on disk yet.
    std::string Picture(const std::string& key, Draw draw);
    void Run();

    std::filesystem::path dir_;
    int image_width_, image_height_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::pair<Draw, std::filesystem::path>> queue_;
    std::unordered_map<std::string, std::string> known_;  // cache key -> picture, empty when drawn blank or queued
    bool stop_ = false;
    std::atomic<uint64_t> version_{0u};
    std::thread worker_;
};

}  // namespace nuka::runtime::app::viewer
