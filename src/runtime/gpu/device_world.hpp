#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::DeviceWorld -- CUDA-resident cooked runtime tables
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "runtime/world_template.hpp"
#include "scene/canonical_types.hpp"

#include <cstddef>
#include <cstdint>

namespace nuka::runtime::gpu {

struct DeviceWorldSummary {
    uint32_t body_count = 0;
    uint32_t shape_count = 0;
    uint32_t joint_count = 0;
    uint32_t actuator_count = 0;

    math::Vec3 first_body_position = math::Vec3::Zero();
    float second_body_inv_mass = 0.0f;
    scene::BodyId first_shape_body_id = 0;
    scene::BodyId first_joint_child_body = 0;
    float first_actuator_gain = 0.0f;
    float first_actuator_force_limit = 0.0f;
};

class DeviceWorld {
public:
    DeviceWorld() = default;

    DeviceWorld(uint32_t body_count,
                uint32_t shape_count,
                uint32_t joint_count,
                uint32_t actuator_count,
                phi::Buffer body_poses,
                phi::Buffer body_inv_masses,
                phi::Buffer body_inv_inertias,
                phi::Buffer shape_body_ids,
                phi::Buffer joint_child_bodies,
                phi::Buffer actuator_gains,
                phi::Buffer actuator_force_limits);

    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;
    DeviceWorld(DeviceWorld&&) noexcept = default;
    DeviceWorld& operator=(DeviceWorld&&) noexcept = default;

    uint32_t BodyCount() const { return body_count_; }
    uint32_t ShapeCount() const { return shape_count_; }
    uint32_t JointCount() const { return joint_count_; }
    uint32_t ActuatorCount() const { return actuator_count_; }

    std::size_t DeviceBytes() const;
    DeviceWorldSummary DownloadSummary() const;

private:
    uint32_t body_count_ = 0;
    uint32_t shape_count_ = 0;
    uint32_t joint_count_ = 0;
    uint32_t actuator_count_ = 0;

    phi::Buffer body_poses_;
    phi::Buffer body_inv_masses_;
    phi::Buffer body_inv_inertias_;
    phi::Buffer shape_body_ids_;
    phi::Buffer joint_child_bodies_;
    phi::Buffer actuator_gains_;
    phi::Buffer actuator_force_limits_;
};

DeviceWorld UploadDeviceWorld(const WorldTemplate& world_template);

} // namespace nuka::runtime::gpu
