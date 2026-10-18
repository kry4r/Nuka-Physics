#pragma once
// Semantic labels as the editor keeps them for a document, and the dataset export: the form that
// asks for one and how far the running one has come.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// The image planes an export can write, in this order.
inline constexpr const char* kDatasetChannels[] = {"rgb", "depth", "instance", "semantic", "normal", "albedo"};
inline constexpr size_t kDatasetChannelCount = sizeof(kDatasetChannels) / sizeof(kDatasetChannels[0]);

// Labels by node path; a node without one takes its nearest labeled ancestor's.
struct LabelView {
    std::map<std::string, std::string> labels;
    std::vector<std::string> classes;  // the labels in use, sorted; semantic class k + 1 is classes[k]
    uint64_t version = 0u;             // bumped whenever `labels` changes
};

// What the export window asks for; seed and randomization start from the document's settings.
struct DatasetForm {
    char out[512] = {};
    int frames = 100, every = 1, seed = 1;
    std::vector<std::string> cameras;  // unchecked camera names
    std::array<bool, kDatasetChannelCount> channels = {true, true, true, true, false, false};
    bool lidar = true, state = true;
    bool render = false, physics = false;
    bool reset = true;
};

struct DatasetView {
    bool open = false;  // the export window
    bool running = false;
    bool cancelled = false;  // the last export stopped early
    uint32_t frame = 0u, frames = 0u;  // frames written of those asked
    std::string out;
    std::string error;
    uint64_t version = 0u;  // bumped when an export starts or ends
};

}  // namespace nuka::runtime::app::viewer
