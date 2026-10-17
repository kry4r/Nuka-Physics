#pragma once
// The editor's user folder (settings.toml, recent.json, logs/, autosave/, thumbnails/):
// %APPDATA%\Nuka\editor on Windows, $XDG_CONFIG_HOME/nuka/editor elsewhere, or --profile.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

struct RecentScene {
    std::string path;
    int64_t time = 0;  // seconds since the epoch of the last open or save
};

class EditorProfile {
public:
    // An empty `dir` takes the platform default; the folder and its subfolders are created.
    explicit EditorProfile(const std::string& dir);

    const std::filesystem::path& Dir() const { return dir_; }
    // The name the primary editor of this folder listens on, distinct per user and folder.
    std::string InstanceName() const;

    // A string in settings.toml, or `fallback` when the key is not set.
    std::string Setting(const std::string& table, const std::string& key, const std::string& fallback) const;
    double SettingNumber(const std::string& table, const std::string& key, double fallback) const;
    void SetSetting(const std::string& table, const std::string& key, const std::string& value);

    std::vector<RecentScene> Recent() const;
    // Moves `path` to the front of the list, which keeps the last kRecentCap scenes.
    void TouchRecent(const std::string& path);
    void ForgetRecent(const std::string& path);
    static constexpr size_t kRecentCap = 12u;
    // Where the viewport picture of a scene's last save is kept.
    std::filesystem::path Thumbnail(const std::string& scene_path) const;

private:
    std::filesystem::path dir_;
};

std::filesystem::path Utf8Path(const std::string& text);
std::string PathText(const std::filesystem::path& path);
// Absolute and lexically normal, the form recent.json and the primary editor compare.
std::string NormalPath(const std::string& path);
int64_t UnixNow();
// strftime of local time; "2026-10-17 14:05" by default.
std::string LocalTimeText(int64_t unix_seconds, const char* format = "%Y-%m-%d %H:%M");

}  // namespace nuka::runtime::app::viewer
