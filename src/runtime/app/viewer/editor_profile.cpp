#include "runtime/app/viewer/editor_profile.hpp"

#include <toml.hpp>

#include "scene/format/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <system_error>

namespace nuka::runtime::app::viewer {
namespace {

namespace fs = std::filesystem;
namespace json = nuka::scene::json;

std::string ReadText(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

// Writes a sibling first, so a crash never leaves a half-written file behind.
void WriteText(const fs::path& file, const std::string& text) {
    fs::path staged = file;
    staged += ".tmp";
    {
        std::ofstream out(staged, std::ios::binary | std::ios::trunc);
        out << text;
    }
    fs::rename(staged, file);
}

fs::path DefaultDir() {
#if defined(_WIN32)
    return fs::path(_wgetenv(L"APPDATA")) / "Nuka" / "editor";
#else
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    const fs::path base = xdg != nullptr && *xdg != '\0' ? fs::path(xdg) : fs::path(std::getenv("HOME")) / ".config";
    return base / "nuka" / "editor";
#endif
}

toml::table Settings(const fs::path& dir) {
    const fs::path file = dir / "settings.toml";
    return toml::parse(ReadText(file), PathText(file));
}

void WriteRecent(const fs::path& dir, const std::vector<RecentScene>& list) {
    json::Value out = json::Value::Array();
    for (const RecentScene& r : list) {
        json::Value entry = json::Value::Object();
        entry.Set("path", json::Value::Str(r.path));
        entry.Set("time", json::Value::Int(r.time));
        out.PushBack(std::move(entry));
    }
    WriteText(dir / "recent.json", out.Dump(2));
}

}  // namespace

fs::path Utf8Path(const std::string& text) { return fs::path(std::u8string(text.begin(), text.end())); }

std::string PathText(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

std::string NormalPath(const std::string& path) { return PathText(fs::absolute(Utf8Path(path)).lexically_normal()); }

int64_t UnixNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string LocalTimeText(int64_t unix_seconds, const char* format) {
    const std::time_t when = static_cast<std::time_t>(unix_seconds);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &when);
#else
    localtime_r(&when, &local);
#endif
    char text[32];
    std::strftime(text, sizeof(text), format, &local);
    return text;
}

EditorProfile::EditorProfile(const std::string& dir)
    : dir_(dir.empty() ? DefaultDir() : fs::absolute(Utf8Path(dir)).lexically_normal()) {
    for (const char* sub : {"logs", "autosave", "extensions", "thumbnails"}) fs::create_directories(dir_ / sub);
}

// The folder is hashed case-folded on Windows, where paths compare without case.
std::string EditorProfile::InstanceName() const {
#if defined(_WIN32)
    std::string user = PathText(fs::path(_wgetenv(L"USERNAME")));
#else
    std::string user = std::getenv("USER");
#endif
    std::string key = PathText(fs::weakly_canonical(dir_));
#if defined(_WIN32)
    for (char& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    uint64_t hash = 14695981039346656037ull;
    for (const char c : user + "\n" + key) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ull;
    }
    for (char& c : user)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
    char suffix[17];
    std::snprintf(suffix, sizeof(suffix), "%016llx", static_cast<unsigned long long>(hash));
    return "nuka-editor-" + user + "-" + suffix;
}

std::string EditorProfile::Setting(const std::string& table, const std::string& key, const std::string& fallback) const {
    return Settings(dir_)[table][key].value_or(fallback);
}

double EditorProfile::SettingNumber(const std::string& table, const std::string& key, double fallback) const {
    return Settings(dir_)[table][key].value_or(fallback);
}

bool EditorProfile::SettingBool(const std::string& table, const std::string& key, bool fallback) const {
    return Settings(dir_)[table][key].value_or(fallback);
}

template <class T>
void EditorProfile::Store(const std::string& table, const std::string& key, T value) {
    toml::table settings = Settings(dir_);
    if (settings[table].as_table() == nullptr) settings.insert_or_assign(table, toml::table{});
    settings[table].as_table()->insert_or_assign(key, std::move(value));
    std::ostringstream text;
    text << settings << "\n";
    WriteText(dir_ / "settings.toml", text.str());
}

void EditorProfile::SetSetting(const std::string& table, const std::string& key, const std::string& value) {
    Store(table, key, value);
}

void EditorProfile::SetSettingNumber(const std::string& table, const std::string& key, double value) {
    Store(table, key, value);
}

void EditorProfile::SetSettingBool(const std::string& table, const std::string& key, bool value) {
    Store(table, key, value);
}

std::vector<std::string> EditorProfile::SettingList(const std::string& table, const std::string& key) const {
    std::vector<std::string> values;
    const toml::table settings = Settings(dir_);
    if (const toml::array* list = settings[table][key].as_array())
        for (const toml::node& v : *list) values.push_back(v.value_or(std::string()));
    return values;
}

void EditorProfile::SetSettingList(const std::string& table, const std::string& key, const std::vector<std::string>& values) {
    toml::table settings = Settings(dir_);
    if (settings[table].as_table() == nullptr) settings.insert_or_assign(table, toml::table{});
    toml::array list;
    for (const std::string& v : values) list.push_back(v);
    settings[table].as_table()->insert_or_assign(key, std::move(list));
    std::ostringstream text;
    text << settings << "\n";
    WriteText(dir_ / "settings.toml", text.str());
}

std::vector<RecentScene> EditorProfile::Recent() const {
    std::vector<RecentScene> list;
    const std::string text = ReadText(dir_ / "recent.json");
    if (text.empty()) return list;
    const json::Value entries = json::Value::Parse(text);
    for (const json::Value& entry : entries.Elements())
        list.push_back(RecentScene{entry.At("path").AsString(), entry.At("time").AsInt()});
    return list;
}

void EditorProfile::TouchRecent(const std::string& path) {
    const std::string normal = NormalPath(path);
    std::vector<RecentScene> list = Recent();
    list.erase(std::remove_if(list.begin(), list.end(), [&](const RecentScene& r) { return r.path == normal; }), list.end());
    list.insert(list.begin(), RecentScene{normal, UnixNow()});
    if (list.size() > kRecentCap) list.resize(kRecentCap);
    WriteRecent(dir_, list);
}

void EditorProfile::ForgetRecent(const std::string& path) {
    const std::string normal = NormalPath(path);
    std::vector<RecentScene> list = Recent();
    list.erase(std::remove_if(list.begin(), list.end(), [&](const RecentScene& r) { return r.path == normal; }), list.end());
    WriteRecent(dir_, list);
}

fs::path EditorProfile::Thumbnail(const std::string& scene_path) const {
    uint64_t hash = 14695981039346656037ull;
    for (const char c : NormalPath(scene_path)) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ull;
    }
    char name[24];
    std::snprintf(name, sizeof(name), "%016llx.ppm", static_cast<unsigned long long>(hash));
    return dir_ / "thumbnails" / name;
}

}  // namespace nuka::runtime::app::viewer
