#include "runtime/app/viewer/editor_training.hpp"

#include "runtime/app/viewer/editor_instance.hpp"
#include "runtime/app/viewer/editor_profile.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iterator>
#include <optional>
#include <thread>
#include <utility>

namespace nuka::runtime::app::viewer {

namespace json = scene::json;
namespace fs = std::filesystem;

namespace {

std::string ReadText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A row's number at `key`, or none when the row lacks it or holds null.
std::optional<double> Number(const json::Value& row, const char* key) {
    const json::Value* v = row.Find(key);
    return v != nullptr && !v->IsNull() ? std::optional<double>(v->AsDouble()) : std::nullopt;
}

int64_t Whole(const json::Value& row, const char* key) { return static_cast<int64_t>(Number(row, key).value_or(0.0)); }

// The metrics.jsonl keys the Dashboard plots and the columns they fill.
constexpr std::pair<const char*, std::vector<double> TrainSeries::*> kSeriesColumns[] = {
    {"epoch", &TrainSeries::epoch},
    {"time", &TrainSeries::time},
    {"reward", &TrainSeries::reward},
    {"length", &TrainSeries::length},
    {"fps", &TrainSeries::fps},
    {"actor_loss", &TrainSeries::actor_loss},
    {"critic_loss", &TrainSeries::critic_loss},
    {"entropy", &TrainSeries::entropy},
    {"kl", &TrainSeries::kl},
    {"lr", &TrainSeries::lr}};

}  // namespace

json::Value ReadJsonFile(const fs::path& path) {
    // The runner swaps its reports in whole, and Windows refuses an open that lands mid-swap, so that one waits.
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (!fs::is_regular_file(path)) return json::Value::Null();
        std::ifstream in(path, std::ios::binary);
        if (in) return json::Value::Parse(std::string(std::istreambuf_iterator<char>(in), {}));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return json::Value::Null();
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

TrainRunRow RunRow(const json::Value& status) {
    TrainRunRow row;
    row.name = status.At("name").AsString();
    row.state = status.At("state").AsString();
    row.alive = status.At("alive").AsBool();
    const json::Value* port = status.Find("observe");
    row.serving = row.alive && port != nullptr && !port->IsNull();
    row.epoch = Whole(status, "epoch");
    row.max_epochs = Whole(status, "max_epochs");
    row.envs = Whole(status, "envs");
    row.frames = Whole(status, "frames");
    row.checkpoints = Whole(status, "checkpoints");
    row.reward = Number(status, "reward");
    row.best = Number(status, "best_reward");
    row.fps = Number(status, "fps");
    row.started = Number(status, "started").value_or(0.0);
    row.ended = Number(status, "ended").value_or(0.0);
    const json::Value* error = status.Find("error");
    if (error != nullptr && !error->IsNull()) row.error = error->AsString();
    return row;
}

void ReadSeries(const fs::path& metrics, TrainSeries* series) {
    const json::Value rows = ReadJsonLines(metrics, series->next);
    for (const json::Value& row : rows.Elements())
        for (const auto& [key, column] : kSeriesColumns) (series->*column).push_back(Number(row, key).value_or(NAN));
    series->next += rows.Elements().size();
}

std::vector<TrainSave> RunSaves(const json::Value& checkpoints) {
    std::vector<TrainSave> saves;
    for (const json::Value& row : checkpoints.Elements()) {
        // The best save overwrites one file; its latest row stands for it.
        std::erase_if(saves, [&](const TrainSave& s) { return s.policy == row.At("policy").AsString(); });
        TrainSave& save = saves.emplace_back();
        save.kind = row.At("kind").AsString();
        save.policy = row.At("policy").AsString();
        save.epoch = Whole(row, "epoch");
        save.frames = Whole(row, "frames");
        save.reward = Number(row, "reward");
        save.time = Number(row, "time").value_or(0.0);
    }
    return saves;
}

}  // namespace nuka::runtime::app::viewer
