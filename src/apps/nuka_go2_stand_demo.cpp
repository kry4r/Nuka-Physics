#include "nuka/nuka.h"
#include "oracle/golden_trajectory.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

const char* DefaultGo2Scene() {
    return "examples/scenes/go2_stand.usda";
}

const char* DefaultPpmOutput() {
    return "out/go2_stand_sanity.ppm";
}

int Fail(const char* operation, nuka_result_t result) {
    std::fprintf(stderr,
                 "nuka_go2_stand_demo: %s failed: %s (%d)\n",
                 operation,
                 nuka_result_message(result),
                 static_cast<int>(result));
    return 1;
}

int WritePpmSanityFrame(const char* output_path, const std::vector<float>& joint_positions) {
    constexpr int kWidth = 96;
    constexpr int kHeight = 64;
    std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "nuka_go2_stand_demo: failed to open %s\n", output_path);
        return 1;
    }

    out << "P6\n" << kWidth << " " << kHeight << "\n255\n";
    const float first_joint = joint_positions.empty() ? 0.0f : joint_positions.front();
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const bool body = x >= 32 && x < 64 && y >= 20 && y < 36;
            const bool leg = (x == 30 || x == 42 || x == 54 || x == 66) && y >= 34 && y < 56;
            const unsigned char accent =
                static_cast<unsigned char>(64 + (static_cast<int>(first_joint * 1000.0f) & 63));
            const unsigned char pixel[3] = {
                static_cast<unsigned char>(body ? 210 : leg ? 160 : 24),
                static_cast<unsigned char>(body ? accent : leg ? 180 : 28),
                static_cast<unsigned char>(body ? 70 : leg ? 90 : 36),
            };
            out.write(reinterpret_cast<const char*>(pixel), sizeof(pixel));
        }
    }
    return out ? 0 : 1;
}

std::vector<float> DownloadJointPositions(nuka_world_handle world) {
    nuka_buffer_view_t view{};
    const nuka_result_t result =
        nuka_world_get_buffer_view(world, NUKA_FIELD_JOINT_POSITION, &view);
    if (result != NUKA_RESULT_OK || view.device_ptr == nullptr) {
        throw std::runtime_error("nuka_world_get_buffer_view failed");
    }

    std::vector<float> joint_positions(view.element_count);
    if (!joint_positions.empty()) {
        const cudaError_t cuda_result =
            cudaMemcpy(joint_positions.data(),
                       view.device_ptr,
                       joint_positions.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (cuda_result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(cuda_result));
        }
    }
    return joint_positions;
}

void AppendTrajectoryRecord(std::vector<float>* payload,
                            const std::vector<float>& joint_positions) {
    if (payload == nullptr) {
        return;
    }
    payload->insert(payload->end(), joint_positions.begin(), joint_positions.end());
}

void WriteTrajectory(const char* path,
                     const char* scene_path,
                     uint32_t step_count,
                     uint32_t joint_count,
                     const std::vector<float>& payload) {
    nuka::tests::oracle::GoldenTrajectory trajectory;
    trajectory.kind = nuka::tests::oracle::GoldenKind::JointTrajectory;
    trajectory.sample_count = step_count;
    trajectory.qpos_count = joint_count;
    trajectory.qvel_count = 0u;
    trajectory.qacc_count = 0u;
    trajectory.model_name = scene_path;
    trajectory.payload = payload;
    nuka::tests::oracle::WriteGoldenTrajectory(path, trajectory);
}

} // namespace

int main(int argc, char** argv) {
    const char* scene_path = argc > 1 ? argv[1] : DefaultGo2Scene();
    const char* output_path = argc > 2 ? argv[2] : DefaultPpmOutput();
    const char* trajectory_path = argc > 3 ? argv[3] : nullptr;

    nuka_device_desc_t device_desc{};
    device_desc.gpu_index = 0u;
    device_desc.cuda_stream = nullptr;
    device_desc.backend_selection_layer_enabled = 1u;

    nuka_device_handle device = nullptr;
    nuka_result_t result = nuka_device_create(&device_desc, &device);
    if (result != NUKA_RESULT_OK) {
        return Fail("nuka_device_create", result);
    }

    nuka_world_desc_t world_desc{};
    world_desc.scene_path = scene_path;
    world_desc.env_count = 1u;
    world_desc.fixed_dt = 1.0f / 240.0f;

    nuka_world_handle world = nullptr;
    result = nuka_world_create_from_scene(device, &world_desc, &world);
    if (result != NUKA_RESULT_OK) {
        nuka_device_destroy(device);
        return Fail("nuka_world_create_from_scene", result);
    }

    constexpr uint32_t kStepCount = 1200u;
    std::vector<float> trajectory_payload;
    std::vector<float> joint_positions;
    const auto start = std::chrono::steady_clock::now();
    if (trajectory_path != nullptr) {
        for (uint32_t step = 0u; step < kStepCount; ++step) {
            result = nuka_world_step(world);
            if (result != NUKA_RESULT_OK) {
                break;
            }
            try {
                joint_positions = DownloadJointPositions(world);
            } catch (const std::exception& error) {
                nuka_world_destroy(world);
                nuka_device_destroy(device);
                std::fprintf(stderr,
                             "nuka_go2_stand_demo: trajectory readback failed: %s\n",
                             error.what());
                return 1;
            }
            AppendTrajectoryRecord(&trajectory_payload, joint_positions);
        }
    } else {
        result = nuka_world_step_n(world, kStepCount);
    }
    const auto end = std::chrono::steady_clock::now();
    if (result != NUKA_RESULT_OK) {
        nuka_world_destroy(world);
        nuka_device_destroy(device);
        return Fail("nuka_world_step_n", result);
    }

    try {
        joint_positions = DownloadJointPositions(world);
    } catch (const std::exception& error) {
        nuka_world_destroy(world);
        nuka_device_destroy(device);
        std::fprintf(stderr,
                     "nuka_go2_stand_demo: final readback failed: %s\n",
                     error.what());
        return 1;
    }

    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::printf("Go2 stand demo scene=%s steps=%u elapsed_ms=%.3f avg_us=%.3f joints=%zu\n",
                scene_path,
                kStepCount,
                static_cast<double>(elapsed_us) / 1000.0,
                static_cast<double>(elapsed_us) / static_cast<double>(kStepCount),
                joint_positions.size());

    if (WritePpmSanityFrame(output_path, joint_positions) != 0) {
        nuka_world_destroy(world);
        nuka_device_destroy(device);
        return 1;
    }
    std::printf("Go2 stand demo wrote %s\n", output_path);
    if (trajectory_path != nullptr) {
        try {
            WriteTrajectory(trajectory_path,
                            scene_path,
                            kStepCount,
                            static_cast<uint32_t>(joint_positions.size()),
                            trajectory_payload);
        } catch (const std::exception& error) {
            nuka_world_destroy(world);
            nuka_device_destroy(device);
            std::fprintf(stderr,
                         "nuka_go2_stand_demo: trajectory write failed: %s\n",
                         error.what());
            return 1;
        }
        std::printf("Go2 stand demo wrote trajectory %s\n", trajectory_path);
    }

    nuka_world_destroy(world);
    nuka_device_destroy(device);
    return 0;
}
