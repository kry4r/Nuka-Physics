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
