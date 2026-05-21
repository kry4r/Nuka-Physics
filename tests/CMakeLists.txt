# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(GoogleTest)

# --- Smoke test -----------------------------------------------------------
add_executable(nuka_smoke_test
    smoke/test_build_smoke.cpp
)

target_link_libraries(nuka_smoke_test
    PRIVATE
        nuka_core
        GTest::gtest_main
)

gtest_discover_tests(nuka_smoke_test)

# --- Math tests ------------------------------------------------------------
add_executable(nuka_math_test
    math/test_spatial_math.cpp
)

target_link_libraries(nuka_math_test
    PRIVATE
        nuka_core
        GTest::gtest_main
)

gtest_discover_tests(nuka_math_test)

# --- Scene tests -----------------------------------------------------------
add_executable(nuka_scene_test
    scene/test_scene_ir.cpp
    scene/test_scene_cooker.cpp
    scene/test_scene_pipeline.cpp
)

target_link_libraries(nuka_scene_test
    PRIVATE
        nuka_scene_pipeline
        nuka_import
        GTest::gtest_main
)

gtest_discover_tests(nuka_scene_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Runtime tests ---------------------------------------------------------
add_executable(nuka_runtime_test
    runtime/test_world_build.cpp
    runtime/test_rigid_integration.cpp
    runtime/test_world_stepper.cpp
)

target_link_libraries(nuka_runtime_test
    PRIVATE
        nuka_runtime
        GTest::gtest_main
)

gtest_discover_tests(nuka_runtime_test)

# --- Import tests ----------------------------------------------------------
add_executable(nuka_import_test
    import/test_mjcf_importer.cpp
    import/test_urdf_importer.cpp
    import/test_usd_importer.cpp
)

target_link_libraries(nuka_import_test
    PRIVATE
        nuka_import
        GTest::gtest_main
)

gtest_discover_tests(nuka_import_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Collision tests -------------------------------------------------------
add_executable(nuka_collision_test
    collision/test_raycast.cpp
    collision/test_broadphase.cpp
)

target_link_libraries(nuka_collision_test
    PRIVATE
        nuka_collision
        GTest::gtest_main
)

gtest_discover_tests(nuka_collision_test)

# --- Constraint tests ------------------------------------------------------
add_executable(nuka_constraint_test
    constraint/test_contact_manifold.cpp
    constraint/test_constraint_block.cpp
)

target_link_libraries(nuka_constraint_test
    PRIVATE
        nuka_constraint
        GTest::gtest_main
)

gtest_discover_tests(nuka_constraint_test)

# --- Solver tests ----------------------------------------------------------
add_executable(nuka_solver_test
    solver/test_contact_resting.cpp
    solver/test_contact_material.cpp
    solver/test_joint_limit.cpp
    solver/test_joint_projection.cpp
)

target_link_libraries(nuka_solver_test
    PRIVATE
        nuka_solver
        nuka_articulation
        GTest::gtest_main
)

gtest_discover_tests(nuka_solver_test)

# --- Articulation tests ----------------------------------------------------
add_executable(nuka_articulation_test
    runtime/test_revolute_joint.cpp
    runtime/test_joint_drive.cpp
)

target_link_libraries(nuka_articulation_test
    PRIVATE
        nuka_articulation
        GTest::gtest_main
)

gtest_discover_tests(nuka_articulation_test)

# --- Sensor tests ----------------------------------------------------------
add_executable(nuka_sensor_test
    sensor/test_imu_sensor.cpp
    sensor/test_lidar_sensor.cpp
)

target_link_libraries(nuka_sensor_test
    PRIVATE
        nuka_sensor
        GTest::gtest_main
)

gtest_discover_tests(nuka_sensor_test)

# --- Debug draw tests ------------------------------------------------------
add_executable(nuka_debug_draw_test
    apps/test_debug_draw_list.cpp
    apps/test_scene_demo.cpp
)

target_link_libraries(nuka_debug_draw_test
    PRIVATE
        nuka_debug_draw
        nuka_scene_pipeline
        nuka_import
        nuka_constraint
        GTest::gtest_main
)

gtest_discover_tests(nuka_debug_draw_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Batch context tests ---------------------------------------------------
add_executable(nuka_batch_test
    runtime/test_batch_context.cpp
)

target_link_libraries(nuka_batch_test
    PRIVATE
        nuka_runtime
        GTest::gtest_main
)

gtest_discover_tests(nuka_batch_test)

# --- Regression tests ------------------------------------------------------
add_executable(nuka_regression_test
    regression/test_free_fall.cpp
    regression/test_rest_stack.cpp
    regression/test_two_link_arm.cpp
)

target_link_libraries(nuka_regression_test
    PRIVATE
        nuka_solver
        nuka_articulation
        nuka_collision
        nuka_runtime
        GTest::gtest_main
)

gtest_discover_tests(nuka_regression_test)

# --- Performance tests -----------------------------------------------------
add_executable(nuka_perf_test
    perf/test_step_timing.cpp
    perf/test_render_demo_timing.cpp
    perf/test_vulkan_scene_demo_timing.cpp
)

target_link_libraries(nuka_perf_test
    PRIVATE
        nuka_runtime
        nuka_debug_draw
        nuka_scene_pipeline
        nuka_import
        nuka_solver
        GTest::gtest_main
)

gtest_discover_tests(nuka_perf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- PHI tests (requires CUDA) --------------------------------------------
if(CMAKE_CUDA_COMPILER)
    add_executable(nuka_phi_test
        phi/test_capabilities.cpp
        phi/test_platform_contract.cpp
    )

    target_link_libraries(nuka_phi_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_phi_test)

    add_executable(nuka_cuda_runtime_test
        runtime/test_cuda_batched_world.cpp
        runtime/test_cuda_device_world.cpp
        runtime/test_cuda_world_stepper.cpp
        runtime/test_cuda_contacts.cpp
        runtime/test_cuda_solver.cpp
    )

    target_link_libraries(nuka_cuda_runtime_test
        PRIVATE
            nuka_runtime_gpu
            nuka_runtime
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_runtime_test)

    add_executable(nuka_cuda_sensor_test
        sensor/test_cuda_sensors.cpp
    )

    target_link_libraries(nuka_cuda_sensor_test
        PRIVATE
            nuka_sensor_gpu
            nuka_sensor
            nuka_runtime_gpu
            nuka_runtime
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_sensor_test)

    add_executable(nuka_cuda_perf_test
        perf/test_cuda_batch_timing.cpp
        perf/test_cuda_batch_contact_timing.cpp
        perf/test_cuda_batch_joint_drive_timing.cpp
        perf/test_cuda_batch_sensor_timing.cpp
        perf/test_cuda_contact_timing.cpp
        perf/test_cuda_scene_demo_timing.cpp
        perf/test_cuda_sensor_timing.cpp
        perf/test_cuda_solver_timing.cpp
        perf/test_cuda_step_timing.cpp
    )

    target_link_libraries(nuka_cuda_perf_test
        PRIVATE
            nuka_debug_draw
            nuka_sensor_gpu
            nuka_sensor
            nuka_runtime_gpu
            nuka_runtime
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_perf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# --- Vulkan renderer tests -------------------------------------------------
add_executable(nuka_render_test
    render/test_vulkan_backend.cpp
)

target_link_libraries(nuka_render_test
    PRIVATE
        nuka_render
        GTest::gtest_main
)

gtest_discover_tests(nuka_render_test)
