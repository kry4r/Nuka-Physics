#include <cstdio>
#include <nuka/nuka.h>

// Minimal C++ diagnostic to read ucontact fields directly from the solver
int main() {
    nuka_device_t device;
    if (nuka_device_create(&device, 0) != NUKA_STATUS_OK) {
        fprintf(stderr, "Device creation failed\n");
        return 1;
    }

    nuka_world_desc_t desc = {};
    desc.env_count = 1;
    desc.dt = 0.005f;
    desc.determinism = NUKA_DETERMINISM_STRONG;
    desc.control_mode = NUKA_CONTROL_MODE_OSC;
    desc.osc_task_link = 7;
    desc.contact_family = 1;
    desc.heightfield_terrain_type = 0;

    nuka_world_t world;
    const char* scene = ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks";
    if (nuka_world_create_from_scene(&world, device, scene, &desc) != NUKA_STATUS_OK) {
        fprintf(stderr, "World creation failed\n");
        nuka_device_destroy(device);
        return 1;
    }

    // Step to steady state
    for (int i = 0; i < 100; ++i) {
        nuka_world_step(world);
    }

    // Read ucontact fields (not exposed to Python)
    size_t count_size, point_size, normal_size;
    void* count_ptr = nuka_world_buffer_view(world, NUKA_FIELD_UCONTACT_COUNT, &count_size);
    void* point_ptr = nuka_world_buffer_view(world, NUKA_FIELD_UCONTACT_POINT, &point_size);
    void* normal_ptr = nuka_world_buffer_view(world, NUKA_FIELD_UCONTACT_NORMAL, &normal_size);

    if (!count_ptr || !point_ptr || !normal_ptr) {
        fprintf(stderr, "ucontact fields not accessible\n");
    } else {
        printf("ucontact_count size: %zu bytes\n", count_size);
        printf("ucontact_point size: %zu bytes\n", point_size);
        printf("ucontact_normal size: %zu bytes\n", normal_size);

        // Dump first few slots
        uint32_t* counts = (uint32_t*)count_ptr;
        float* points = (float*)point_ptr;
        for (int slot = 0; slot < 10; ++slot) {
            uint32_t n = counts[slot];
            if (n > 0) {
                printf("slot %d: count=%u\n", slot, n);
                for (uint32_t pt = 0; pt < n && pt < 4; ++pt) {
                    size_t base = (slot * 4 + pt) * 3;
                    printf("  pt%u: [%.4f, %.4f, %.4f]\n", pt,
                           points[base], points[base+1], points[base+2]);
                }
            }
        }
    }

    nuka_world_destroy(world);
    nuka_device_destroy(device);
    return 0;
}
