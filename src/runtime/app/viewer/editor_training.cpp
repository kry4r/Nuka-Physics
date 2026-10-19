#include "runtime/app/viewer/editor_training.hpp"

#include "runtime/app/viewer/editor_instance.hpp"
#include "runtime/app/viewer/editor_profile.hpp"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iterator>

namespace nuka::runtime::app::viewer {

namespace json = scene::json;
namespace fs = std::filesystem;

namespace {

std::string ReadText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace

json::Value ReadJsonFile(const fs::path& path) {
    if (!fs::is_regular_file(path)) return json::Value::Null();
    return json::Value::Parse(ReadText(path));
}

json::Value ReadJsonLines(const fs::path& path, size_t skip) {
    json::Value rows = json::Value::Array();
    if (!fs::is_regular_file(path)) return rows;
    const std::string text = ReadText(path);
    size_t line = 0u;
    for (size_t begin = 0u, end; (end = text.find('\n', begin)) != std::string::npos; begin = end + 1u, ++line)
        if (line >= skip && end > begin) rows.PushBack(json::Value::Parse(text.substr(begin, end - begin)));
    return rows;
}

std::string TailText(const fs::path& path, size_t lines) {
    if (!fs::is_regular_file(path)) return std::string();
    const std::string text = ReadText(path);
    // A final newline ends the last line rather than starting another.
    size_t begin = text.size() - (!text.empty() && text.back() == '\n' ? 1u : 0u);
    for (size_t n = 0u; n < lines; ++n) {
        const size_t nl = begin == 0u ? std::string::npos : text.rfind('\n', begin - 1u);
        if (nl == std::string::npos) return text;
        begin = nl;
    }
    return text.substr(begin + 1u);
}

std::vector<fs::path> RunFolders(const fs::path& root) {
    std::vector<fs::path> runs;
    if (!fs::is_directory(root)) return runs;
    for (const fs::directory_entry& entry : fs::directory_iterator(root))
        if (entry.is_directory() && fs::is_regular_file(entry.path() / "job.json")) runs.push_back(entry.path());
    std::sort(runs.begin(), runs.end(), [](const fs::path& a, const fs::path& b) {
        return fs::last_write_time(a / "job.json") < fs::last_write_time(b / "job.json");
    });
    return runs;
}

json::Value RunStatus(const fs::path& dir, uint32_t pid) {
    const json::Value status = ReadJsonFile(dir / "status.json");
    if (status.IsObject()) pid = static_cast<uint32_t>(status.At("pid").AsInt());
    const bool alive = pid != 0u && ProcessAlive(pid);
    std::string state = status.IsObject() ? status.At("state").AsString() : "starting";
    if ((state == "starting" || state == "running") && !alive) state = "lost";
    json::Value out = json::Value::Object();
    out.Set("name", json::Value::Str(PathText(dir.filename())));
    out.Set("dir", json::Value::Str(PathText(dir)));
    out.Set("state", json::Value::Str(state));
    out.Set("alive", json::Value::Bool(alive));
    if (status.IsObject())
        for (const auto& [key, value] : status.Items())
            if (key != "name" && key != "state") out.Set(key, value);
    return out;
}

std::string RunPolicy(const json::Value& checkpoints, const std::string& kind) {
    const std::vector<json::Value>& rows = checkpoints.Elements();
    for (auto row = rows.rbegin(); row != rows.rend(); ++row)
        if (kind == "last" || row->At("kind").AsString() == kind) return row->At("policy").AsString();
    return std::string();
}

std::string RunName() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char name[32];
    std::strftime(name, sizeof(name), "run-%Y%m%d-%H%M%S", &local);
    return name;
}

}  // namespace nuka::runtime::app::viewer
