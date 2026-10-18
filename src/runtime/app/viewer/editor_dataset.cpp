#include "runtime/app/viewer/editor_dataset.hpp"

#include <algorithm>
#include <fstream>
#include <set>

namespace nuka::runtime::app::viewer {

namespace json = scene::json;

namespace {

// An instance plane holds the instance in its top 12 bits; all bits set where nothing was hit.
constexpr uint32_t kMissBits = 0xFFFFFFFFu;
constexpr uint32_t kInstanceShift = 20u;

}  // namespace

std::string LabelOf(const std::map<std::string, std::string>& labels, const std::string& path, std::string* from) {
    for (std::string at = path;;) {
        if (const auto it = labels.find(at); it != labels.end()) {
            if (from != nullptr) *from = at;
            return it->second;
        }
        const size_t slash = at.rfind('/');
        if (slash == std::string::npos) break;
        at.resize(slash);
    }
    if (from != nullptr) from->clear();
    return std::string();
}

std::vector<std::string> LabelClasses(const std::map<std::string, std::string>& labels) {
    std::set<std::string> distinct;
    for (const auto& [path, label] : labels) distinct.insert(label);
    return std::vector<std::string>(distinct.begin(), distinct.end());
}

std::vector<uint32_t> InstanceClasses(const std::map<std::string, std::string>& labels,
                                      const std::vector<std::string>& classes,
                                      const std::vector<std::string>& instances) {
    std::vector<uint32_t> out;
    for (const std::string& path : instances) {
        const auto it = std::find(classes.begin(), classes.end(), LabelOf(labels, path));
        out.push_back(it == classes.end() ? 0u : static_cast<uint32_t>(it - classes.begin()) + 1u);
    }
    return out;
}

uint32_t PixelClass(uint32_t bits, const std::vector<uint32_t>& classes) {
    return bits == kMissBits ? 0u : classes[bits >> kInstanceShift];
}

json::Value LabelsToJson(const std::map<std::string, std::string>& labels) {
    json::Value map = json::Value::Object();
    for (const auto& [path, label] : labels) map.Set(path, json::Value::Str(label));
    json::Value v = json::Value::Object();
    v.Set("labels", std::move(map));
    return v;
}

std::map<std::string, std::string> LabelsFromJson(const json::Value& v) {
    std::map<std::string, std::string> labels;
    for (const auto& [path, label] : v.At("labels").Items()) labels[path] = label.AsString();
    return labels;
}

void AppendRaw(const std::filesystem::path& path, const std::vector<float>& values) {
    const auto bytes = static_cast<std::streamsize>(values.size() * sizeof(float));
    std::ofstream(path, std::ios::binary | std::ios::app).write(reinterpret_cast<const char*>(values.data()), bytes);
}

}  // namespace nuka::runtime::app::viewer
