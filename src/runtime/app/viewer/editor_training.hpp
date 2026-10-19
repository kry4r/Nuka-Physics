#pragma once
// Training runs as the editor reads them: the report files of a run folder, read while
// nuka.train.runner writes them.

#include "scene/format/json.hpp"
#include "runtime/app/viewer/train_view.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// The JSON in the file at `path`; null when there is none or it does not parse.
scene::json::Value ReadJsonFile(const std::filesystem::path& path);
// The rows of a JSON-lines file after the first `skip`; a last line without its newline is
// still being written and is left out.
scene::json::Value ReadJsonLines(const std::filesystem::path& path, size_t skip = 0u);
// The last `lines` lines of a text file.
std::string TailText(const std::filesystem::path& path, size_t lines);
// The run folders under `root`, oldest first.
std::vector<std::filesystem::path> RunFolders(const std::filesystem::path& root);
// A run's status.json with its `name`, `dir` and whether its process is `alive`; a run that
// says it trains but whose process is gone is "lost". `pid` stands in until the run reports one.
scene::json::Value RunStatus(const std::filesystem::path& dir, uint32_t pid = 0u);
// The policy of the newest checkpoint row of `kind` ("best", "final", "epoch"; "last" is any);
// empty when there is none.
std::string RunPolicy(const scene::json::Value& checkpoints, const std::string& kind);
// A name for a run that starts now: "run-YYYYMMDD-HHMMSS".
std::string RunName();
// The Training panel's row of a run's RunStatus.
TrainRunRow RunRow(const scene::json::Value& status);
// Appends the rows of a run's metrics.jsonl past `series->next` to its columns.
void ReadSeries(const std::filesystem::path& metrics, TrainSeries* series);
// The saves of a run's checkpoints.jsonl rows.
std::vector<TrainSave> RunSaves(const scene::json::Value& checkpoints);

}  // namespace nuka::runtime::app::viewer
