#pragma once
// Imports the renderer's pose slots and timeline semaphore, then writes each instance's
// world matrix from the live pose fields on the world's stream.

#include "nuka/nuka.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::runtime::app::viewer {

// One instance's transform source. Kind 0 writes `local` as the world pose; kinds 1
// and 2 compose the displayed environment's live link or body pose with `local`.
struct ScatterRow {
    uint32_t kind = 0u;
    uint32_t row = 0u;
    float local[7] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};  // position, quaternion w x y z
};

// The renderer's exported slots. PoseScatter's constructor takes ownership of the
// handles, whether or not the import succeeds.
struct PoseSlotImport {
    void* memory_handle = nullptr;
    int memory_fd = -1;
    uint64_t memory_bytes = 0u;
    void* semaphore_handle = nullptr;
    int semaphore_fd = -1;
    uint64_t slot_bytes = 0u;
    uint8_t device_uuid[16] = {};
};

// Makes the world's CUDA device current on the calling thread.
void BindWorldDevice(nuka_world_handle world);

class PoseScatter {
public:
    // Throws std::runtime_error when the world's CUDA device cannot import the slots.
    PoseScatter(nuka_world_handle world, PoseSlotImport& slots);
    ~PoseScatter();
    PoseScatter(const PoseScatter&) = delete;
    PoseScatter& operator=(const PoseScatter&) = delete;

    // The instance bindings later scatters read, queued on the world stream.
    void SetRows(nuka_world_handle world, const std::vector<ScatterRow>& rows);
    // Writes `slot` from environment `env`'s poses after the queued physics work, then
    // signals the timeline to `value`.
    void Scatter(nuka_world_handle world, uint32_t env, uint32_t links_per_env, uint32_t bodies_per_env,
                 uint32_t slot, uint64_t value);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::runtime::app::viewer
