#pragma once
// Semantic labels as JSON and how nodes inherit them, and the raw streams a dataset export
// captures for nuka.editor.dataset to build the dataset from.

#include "scene/format/json.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// The label of the node at `path` or of its nearest labeled ancestor, whose path goes to `from`;
// "" when none has one.
std::string LabelOf(const std::map<std::string, std::string>& labels, const std::string& path,
                    std::string* from = nullptr);
// The distinct labels, sorted.
std::vector<std::string> LabelClasses(const std::map<std::string, std::string>& labels);
// Each instance's semantic class: 1 + its label's index in `classes`, or 0 without a label.
std::vector<uint32_t> InstanceClasses(const std::map<std::string, std::string>& labels,
                                      const std::vector<std::string>& classes,
                                      const std::vector<std::string>& instances);
// An instance plane's value as a semantic class; 0 where nothing was hit.
uint32_t PixelClass(uint32_t bits, const std::vector<uint32_t>& classes);

// {"labels": {path: label}}
scene::json::Value LabelsToJson(const std::map<std::string, std::string>& labels);
std::map<std::string, std::string> LabelsFromJson(const scene::json::Value& v);

// Appends the bytes of `values` to the file at `path`.
void AppendRaw(const std::filesystem::path& path, const std::vector<float>& values);

}  // namespace nuka::runtime::app::viewer
