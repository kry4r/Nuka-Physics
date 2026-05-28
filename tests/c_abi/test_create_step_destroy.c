#include "nuka/nuka.h"

#include <assert.h>

int main(void) {
    nuka_device_desc_t device_desc;
    device_desc.gpu_index = 0u;
    device_desc.cuda_stream = 0;
    device_desc.backend_selection_layer_enabled = 1u;

    nuka_device_handle device = 0;
    assert(nuka_device_create(&device_desc, &device) == NUKA_RESULT_OK);
    assert(device != 0);

    nuka_world_desc_t world_desc;
    world_desc.scene_path = "examples/scenes/go2_stand.usda";
    world_desc.env_count = 1u;
    world_desc.fixed_dt = 1.0f / 240.0f;

    nuka_world_handle world = 0;
    assert(nuka_world_create_from_scene(device, &world_desc, &world) == NUKA_RESULT_OK);
    assert(world != 0);

    assert(nuka_world_step_n(world, 4u) == NUKA_RESULT_OK);

    nuka_buffer_view_t view;
    assert(nuka_world_get_buffer_view(world, NUKA_FIELD_JOINT_POSITION, &view) ==
           NUKA_RESULT_OK);
    assert(view.device_ptr != 0);
    assert(view.element_count > 0u);
    assert(view.element_stride_bytes == sizeof(float));

    nuka_world_destroy(world);
    nuka_device_destroy(device);
    return 0;
}
