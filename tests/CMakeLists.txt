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
)

target_link_libraries(nuka_scene_test
    PRIVATE
        nuka_scene
        GTest::gtest_main
)

gtest_discover_tests(nuka_scene_test)

# --- Runtime tests ---------------------------------------------------------
add_executable(nuka_runtime_test
    runtime/test_world_build.cpp
    runtime/test_rigid_integration.cpp
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
    solver/test_joint_limit.cpp
)

target_link_libraries(nuka_solver_test
    PRIVATE
        nuka_solver
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

# --- PHI tests (requires CUDA) --------------------------------------------
if(CMAKE_CUDA_COMPILER)
    add_executable(nuka_phi_test
        phi/test_capabilities.cpp
    )

    target_link_libraries(nuka_phi_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_phi_test)
endif()
