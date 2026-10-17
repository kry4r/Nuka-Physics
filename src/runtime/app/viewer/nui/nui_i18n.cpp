// UI string tables: layers of English -> translated text per locale, merged for the active one.

#include "runtime/app/viewer/nui/nui.hpp"

#include "render/viewer_resources.hpp"

#include <toml.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>

namespace nuka::nui {

namespace {

struct Hash {
    using is_transparent = void;
    size_t operator()(std::string_view text) const { return std::hash<std::string_view>{}(text); }
};
using Table = std::unordered_map<std::string, std::string, Hash, std::equal_to<>>;

struct Layer {
    std::string owner;
    std::string locale;
    Table strings;
};

struct Described {
    LocaleInfo info;
    std::string owner;
};

struct Strings {
    std::mutex mu;
    std::vector<Layer> layers;  // later layers win
    std::vector<Described> locales;
    std::string locale = "en";
    bool dirty = false;
    // Read by the UI thread without the lock; only ApplyStrings writes them.
    bool english = true;
    Table merged;
    std::unordered_set<std::string, Hash, std::equal_to<>> missing;
};

Strings& S() {
    static Strings strings;
    return strings;
}

bool HasLetters(std::string_view text) {
    return std::any_of(text.begin(), text.end(), [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); });
}

const std::string* Lookup(std::string_view text) {
    Strings& s = S();
    const auto it = s.merged.find(text);
    if (it != s.merged.end()) return &it->second;
    if (HasLetters(text) && s.missing.find(text) == s.missing.end()) s.missing.emplace(text);
    return nullptr;
}

void Describe(Strings& s, const LocaleInfo& info, const std::string& owner) {
    const auto it = std::find_if(s.locales.begin(), s.locales.end(), [&](const Described& d) { return d.info.id == info.id; });
    if (it == s.locales.end()) s.locales.push_back(Described{info, owner});
    else if (!info.name.empty()) it->info = info;
}

}  // namespace

const std::string& Tr(const std::string& text) {
    if (S().english) return text;
    const std::string* found = Lookup(text);
    return found != nullptr ? *found : text;
}

const char* Tr(const char* text) {
    if (S().english) return text;
    const std::string* found = Lookup(text);
    return found != nullptr ? found->c_str() : text;
}

std::string TrCopy(const std::string& text) {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    const auto it = s.merged.find(text);
    return it != s.merged.end() ? it->second : text;
}

std::vector<LocaleInfo> Locales() {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    std::vector<LocaleInfo> out{{"en", "English", "EN"}};
    for (const Described& d : s.locales) out.push_back(d.info);
    return out;
}

std::string Locale() {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.locale;
}

void SetLocale(const std::string& locale) {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    s.locale = locale;
    s.dirty = true;
}

void AddTranslations(const std::string& locale, const std::string& owner,
                     const std::vector<std::pair<std::string, std::string>>& strings) {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = std::find_if(s.layers.begin(), s.layers.end(), [&](const Layer& l) { return l.owner == owner && l.locale == locale; });
    if (it == s.layers.end()) it = s.layers.insert(s.layers.end(), Layer{owner, locale, {}});
    for (const auto& [key, text] : strings) it->strings.insert_or_assign(key, text);
    Describe(s, LocaleInfo{locale, std::string(), std::string()}, owner);
    s.dirty = true;
}

void RemoveTranslations(const std::string& owner) {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    std::erase_if(s.layers, [&](const Layer& l) { return l.owner == owner; });
    std::erase_if(s.locales, [&](const Described& d) {
        return d.owner == owner &&
               std::none_of(s.layers.begin(), s.layers.end(), [&](const Layer& l) { return l.locale == d.info.id; });
    });
    s.dirty = true;
}

void LoadTranslations(const std::filesystem::path& file, const std::string& owner) {
    std::ifstream input(file, std::ios::binary);
    std::stringstream text;
    text << input.rdbuf();
    const toml::table doc = toml::parse(text.str(), file.string());
    const std::string locale = doc["meta"]["locale"].value_or(std::string());
    std::vector<std::pair<std::string, std::string>> strings;
    if (const toml::table* table = doc["strings"].as_table())
        for (const auto& [key, value] : *table) strings.emplace_back(std::string(key.str()), value.value_or(std::string()));
    AddTranslations(locale, owner, strings);
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    Describe(s, LocaleInfo{locale, doc["meta"]["name"].value_or(std::string(locale)), doc["meta"]["tag"].value_or(std::string(locale))},
             owner);
}

void LoadTranslationDir(const std::filesystem::path& dir, const std::string& owner) {
    std::error_code error;
    if (!std::filesystem::is_directory(dir, error)) return;
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        if (entry.path().extension() == ".toml") files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    for (const auto& file : files) LoadTranslations(file, owner);
}

std::filesystem::path BuiltinTranslationDir() {
    const std::filesystem::path installed = render::ViewerModuleDirectory() / "assets" / "editor" / "i18n";
    std::error_code error;
    return std::filesystem::is_directory(installed, error) ? installed : std::filesystem::path(NUKA_EDITOR_I18N_DIR);
}

void ApplyStrings() {
    Strings& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.dirty) return;
    s.dirty = false;
    s.english = s.locale == "en";
    s.merged.clear();
    s.missing.clear();
    for (const Layer& layer : s.layers)
        if (layer.locale == s.locale)
            for (const auto& [key, text] : layer.strings) s.merged.insert_or_assign(key, text);
}

std::vector<std::string> MissingStrings() {
    std::vector<std::string> out(S().missing.begin(), S().missing.end());
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace nuka::nui
