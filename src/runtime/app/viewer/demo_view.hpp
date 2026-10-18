#pragma once
// What the Demos panel shows: the recording, the episodes and a replay in progress.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

struct DemoView {
    struct Episode {
        uint32_t env = 0u;
        size_t steps = 0u;
        double seconds = 0.0;
    };
    bool recording = false;
    bool all = false;       // record every displayed environment, not only the shown one
    uint64_t samples = 0u;  // recorded since the start, one per environment and step
    std::vector<Episode> episodes;
    int selected = -1;
    int playing = -1;    // the replayed episode; -1 when none
    uint32_t left = 0u;  // the steps its replay still drives
    std::string error;
    char path_edit[512] = {};
};

}  // namespace nuka::runtime::app::viewer
