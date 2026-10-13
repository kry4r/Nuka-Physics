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

gtest_discover_tests(nuka_smoke_test
    PROPERTIES LABELS "fast_full")

# --- Math tests ------------------------------------------------------------
add_executable(nuka_math_test
    math/test_spatial_math.cpp
    math/test_hencky_j2.cpp
)

target_link_libraries(nuka_math_test
    PRIVATE
        nuka_core
        GTest::gtest_main
)

gtest_discover_tests(nuka_math_test
    PROPERTIES LABELS "fast_full")

if(TARGET nuka)
    add_executable(nuka_c_abi_create_step_destroy_test
        c_abi/test_create_step_destroy.c
    )

    target_include_directories(nuka_c_abi_create_step_destroy_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(nuka_c_abi_create_step_destroy_test
        PRIVATE
            nuka
    )

    add_test(NAME CAbi.CreateStepDestroy
             COMMAND nuka_c_abi_create_step_destroy_test)
    set_tests_properties(CAbi.CreateStepDestroy PROPERTIES LABELS "fast_full")
    set_tests_properties(CAbi.CreateStepDestroy
        PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    if(NUKA_HAS_EXPECTED_WRAPPER)
        add_executable(nuka_cpp_wrapper_raii_test
            c_abi/test_cpp_wrapper_raii.cpp
        )

        target_compile_definitions(nuka_cpp_wrapper_raii_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_cpp_wrapper_raii_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_cpp_wrapper_raii_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_cpp_wrapper_raii_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "fast_full")

        add_executable(nuka_multi_env_world_test
            c_abi/test_multi_env_world.cpp
        )

        target_compile_definitions(nuka_multi_env_world_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_multi_env_world_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_multi_env_world_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_multi_env_world_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")

        # --- Coupled world (robot + cloth/fluid particles) C ABI bridge ---------
        # Create + step + read a coupled world through nuka_world_create_coupled_
        # from_scene; prove two-way coupling fires through the C step path and the
        # terrain+particle budget reserve stays disjoint.
        add_executable(nuka_coupled_world_cabi_test
            c_abi/test_coupled_world.cpp
        )

        target_compile_definitions(nuka_coupled_world_cabi_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_coupled_world_cabi_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_coupled_world_cabi_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_coupled_world_cabi_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")

        # --- Device-resident batched camera sensor C ABI (attach/render/view) --
        # The in-the-loop obs surface: attach a camera to an N-env world, render
        # every env into ONE device AOV tensor, read it back zero-copy via
        # nuka_world_get_sensor_view (the bridge cooks the world's retained scene to
        # the per-env visual binding + drives the batched sensor backend).
        add_executable(nuka_camera_sensor_cabi_test
            c_abi/test_camera_sensor.cpp
        )

        target_compile_definitions(nuka_camera_sensor_cabi_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_camera_sensor_cabi_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_camera_sensor_cabi_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_camera_sensor_cabi_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")

        # --- Device-resident batched LIDAR sensor C ABI (attach/render/RANGE) ---
        # The lidar in-the-loop obs: attach a lidar to an N-env world, render every
        # env's (az,el) fan into ONE device RANGE tensor on the SAME RT TLAS the
        # cameras use, read it back zero-copy via nuka_world_get_sensor_view(RANGE).
        add_executable(nuka_lidar_sensor_cabi_test
            c_abi/test_lidar_sensor_cabi.cpp
        )

        target_compile_definitions(nuka_lidar_sensor_cabi_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_lidar_sensor_cabi_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_lidar_sensor_cabi_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_lidar_sensor_cabi_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")

        # --- Multiple cameras per env C ABI (S cameras, one per mount row) ------
        # Attach two cameras with different mount rows, render into ONE (E,S,H,W)
        # tensor, and assert each (env,sensor) tile equals the single-camera render
        # of that mount (the S>1 fan-out is the S==1 path per camera).
        add_executable(nuka_multi_camera_sensor_cabi_test
            c_abi/test_multi_camera_sensor.cpp
        )

        target_compile_definitions(nuka_multi_camera_sensor_cabi_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_multi_camera_sensor_cabi_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_multi_camera_sensor_cabi_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_multi_camera_sensor_cabi_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")

        # --- Per-env drive-target write + q/qd/base read C ABI (sim-val #33) --
        add_executable(nuka_drive_target_io_test
            c_abi/test_drive_target_io.cpp
        )

        target_compile_definitions(nuka_drive_target_io_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_drive_target_io_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_drive_target_io_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_drive_target_io_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")
    else()
        message(STATUS "Skipping nuka.hpp RAII test: nuka::expected wrapper failed to compile")
    endif()


    add_executable(nuka_sensor_noise_cabi_test
        c_abi/test_sensor_noise.cpp
    )

    target_compile_definitions(nuka_sensor_noise_cabi_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_sensor_noise_cabi_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(nuka_sensor_noise_cabi_test
        PRIVATE
            nuka
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sensor_noise_cabi_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    if(CMAKE_CUDA_COMPILER)
        add_executable(nuka_n2_domain_randomization_test
            sensor/test_n2_domain_randomization.cpp
        )

        target_compile_definitions(nuka_n2_domain_randomization_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_n2_domain_randomization_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
                ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(nuka_n2_domain_randomization_test
            PRIVATE
                nuka
                nuka_sensor_noise
                nuka_runtime
                nuka_contact_wrench
                nuka_core
                nuka_phi
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_n2_domain_randomization_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")
    endif()

    if(CMAKE_CUDA_COMPILER)


    endif()
endif()


# --- M2c scenario gate: .nks/.nka scene roundtrip fidelity -----------------
# Import -> Compose -> Save(.nks+.nka) -> Load -> assert structural + cook
# byte-identity, plus second-save byte-identity, override layer and SceneMap.
# Registered as a single ctest "SceneRoundtrip" (runs the whole executable).
add_executable(nuka_scene_roundtrip_test
    scenario/scene_roundtrip.cpp
)

target_link_libraries(nuka_scene_roundtrip_test
    PRIVATE
        nuka_scene
        nuka_import
        nuka_cooker
        GTest::gtest_main
)

add_test(NAME SceneRoundtrip
         COMMAND nuka_scene_roundtrip_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(SceneRoundtrip PROPERTIES LABELS "fast_full")

# Editor edit -> nks::Save -> reload gate: transform + material edits persist and
# collision mesh geometry survives (the LoadLightScene data-loss trap is avoided).
add_executable(nuka_editor_edit_roundtrip_test
    scenario/editor_edit_roundtrip.cpp
)
target_link_libraries(nuka_editor_edit_roundtrip_test
    PRIVATE
        nuka_scene
        nuka_import
        nuka_cooker
        GTest::gtest_main
)
add_test(NAME EditorEditRoundtrip
         COMMAND nuka_editor_edit_roundtrip_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(EditorEditRoundtrip PROPERTIES LABELS "fast_full")

# Editor undo/redo state round-trip gate: for each of the six edit kinds, apply ->
# undo restores the SceneIR (persisted .nks bytes) + the live RenderWorld exactly,
# redo re-reaches the edited state. Cooks a real EditorScene, so it compiles the
# editor's host-only edit seams in directly (editor_edits / editor_scene / editor_undo
# live only in the gated editor exe) and links the cook + nk + render + app stack.
add_executable(nuka_editor_undo_roundtrip_test
    scenario/editor_undo_roundtrip.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_edits.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_undo.cpp
)
add_dependencies(nuka_editor_undo_roundtrip_test nuka_codegen)
target_include_directories(nuka_editor_undo_roundtrip_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_editor_undo_roundtrip_test
    PRIVATE
        nuka_app
        nuka_scene_cook
        nuka_render
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_contact_wrench
        nuka_collision_gpu
        nuka_core
        nuka_phi
        GTest::gtest_main
)
gtest_discover_tests(nuka_editor_undo_roundtrip_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

# Editor primitive-spawn round-trip gate: for each of box/sphere/capsule/plane, spawn
# through SpawnPrimitive -> assert the authority + cook grew (body/shapes/material +
# render instance + movable row), Save+reload survives the records and the bytes, and
# undo/redo restore exactly. Compiles the host-only editor edit seams in directly.
add_executable(nuka_editor_spawn_roundtrip_test
    scenario/editor_spawn_roundtrip.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_edits.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_undo.cpp
)
add_dependencies(nuka_editor_spawn_roundtrip_test nuka_codegen)
target_include_directories(nuka_editor_spawn_roundtrip_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_editor_spawn_roundtrip_test
    PRIVATE
        nuka_app
        nuka_scene_cook
        nuka_render
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_contact_wrench
        nuka_collision_gpu
        nuka_core
        nuka_phi
        GTest::gtest_main
)
gtest_discover_tests(nuka_editor_spawn_roundtrip_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

# Editor delete-preservation gate: SceneIR::RemoveBodySubtree keeps every
# unrelated record (joints/actuators/sensors/pairs/ICs) with exact remaps (pure
# host), and the live spawn->delete->re-cook->Save round-trip on go2 preserves
# joints + scripts + excludes byte-identically. Also pins the /script node
# delete, the reparent-onto-a-shape guard, and the transactional re-cook.
add_executable(nuka_editor_delete_preserve_test
    scenario/editor_delete_preserve.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_edits.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
)
add_dependencies(nuka_editor_delete_preserve_test nuka_codegen)
target_include_directories(nuka_editor_delete_preserve_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_editor_delete_preserve_test
    PRIVATE
        nuka_app
        nuka_scene_cook
        nuka_render
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_contact_wrench
        nuka_collision_gpu
        nuka_core
        nuka_phi
        GTest::gtest_main
)
gtest_discover_tests(nuka_editor_delete_preserve_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

# Editor true-physics-Reset round-trip gate: cook a real EditorScene (articulation +
# the soft / cloth / fluid demo media), evolve it, fire ResetEditorScenePhysics, and
# assert the authored qpos / root / particle state returns with zeroed velocities and
# re-seeded drive targets. Compiles the host-only editor_scene seam in directly.
add_executable(nuka_editor_reset_roundtrip_test
    scenario/editor_reset_roundtrip.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
)
add_dependencies(nuka_editor_reset_roundtrip_test nuka_codegen)
target_include_directories(nuka_editor_reset_roundtrip_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_editor_reset_roundtrip_test
    PRIVATE
        nuka_app
        nuka_scene_cook
        nuka_render
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_contact_wrench
        nuka_collision_gpu
        nuka_core
        nuka_phi
        GTest::gtest_main
)
gtest_discover_tests(nuka_editor_reset_roundtrip_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

# Read-only debug-overlay data-path gate: cook a real EditorScene, evolve it, and
# assert the DebugOverlay emits one collider proxy per cooked primitive (at the live
# BodyPose, kind-coloured) + one marker per active contact point (count-matched to the
# ContactCount watermark). Compiles the host-only editor_scene + debug_overlay seams in.
add_executable(nuka_editor_debug_overlay_test
    scenario/editor_debug_overlay.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/debug_overlay.cpp
)
add_dependencies(nuka_editor_debug_overlay_test nuka_codegen)
target_include_directories(nuka_editor_debug_overlay_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_editor_debug_overlay_test
    PRIVATE
        nuka_app
        nuka_scene_cook
        nuka_render
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_contact_wrench
        nuka_collision_gpu
        nuka_core
        nuka_phi
        GTest::gtest_main
)
gtest_discover_tests(nuka_editor_debug_overlay_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

# Editor keyboard-teleop gate: the PURE mapping (synthetic key intents nudge the
# per-DOF drive seam by EXACTLY the step / select+clamp the DOF / WASD->command, and a
# gated tick is inert) plus the LIVE world (a nudge reaches nk::Data through the viewer
# upload and the PD-held joint swings; a command reaches a command-consuming controller
# through the general base interface). Compiles the host-only editor_scene seam in.
add_executable(nuka_editor_teleop_test
    scenario/editor_teleop.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
)
add_dependencies(nuka_editor_teleop_test nuka_codegen)
target_include_directories(nuka_editor_teleop_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_editor_teleop_test
    PRIVATE
        nuka_app
        nuka_scene_cook
        nuka_render
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_contact_wrench
        nuka_collision_gpu
        nuka_core
        nuka_phi
        GTest::gtest_main
)
gtest_discover_tests(nuka_editor_teleop_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

# Editor live-world script LIFECYCLE gate: cook a real EditorScene, embed the
# ScriptHost bound to it, register a /script node, and drive the SAME step + reset
# path the viewer does -- asserting on_ready (once) / on_step (each step, reaching the
# live world) / on_reset (after the reset entry) fire, and a throwing callback is
# caught + bounded-disabled while a sibling keeps running. Compiles the host-only
# script_host / editor_scene / editor_edits seams in directly + embeds CPython, so it
# is built ONLY where Python3 Development.Embed is present (skipped otherwise).
find_package(Python3 QUIET COMPONENTS Interpreter Development.Embed)
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; sys.stdout.write(sys.prefix)"
        OUTPUT_VARIABLE NUKA_TEST_PY_HOME)
    add_executable(nuka_editor_script_lifecycle_test
        scenario/editor_script_lifecycle.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/script_host.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_edits.cpp
    )
    add_dependencies(nuka_editor_script_lifecycle_test nuka_codegen)
    target_include_directories(nuka_editor_script_lifecycle_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(nuka_editor_script_lifecycle_test
        SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
    target_compile_definitions(nuka_editor_script_lifecycle_test
        PRIVATE NUKA_PY_HOME="${NUKA_TEST_PY_HOME}")
    target_link_libraries(nuka_editor_script_lifecycle_test
        PRIVATE
            Python3::Python
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_editor_script_lifecycle_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")
endif()

add_executable(nuka_nks_grasp_scene_roundtrip_test
    scenario/nks_grasp_scene_roundtrip.cpp
)
target_link_libraries(nuka_nks_grasp_scene_roundtrip_test
    PRIVATE
        nuka_scene
        nuka_import
        nuka_cooker
        GTest::gtest_main
)
add_test(NAME NksGraspSceneRoundtrip
         COMMAND nuka_nks_grasp_scene_roundtrip_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(NksGraspSceneRoundtrip PROPERTIES LABELS "fast_full")

add_executable(nuka_visual_mesh_cook_smoke_test
    scenario/visual_mesh_cook_smoke.cpp
)
target_link_libraries(nuka_visual_mesh_cook_smoke_test
    PRIVATE
        nuka_scene
        nuka_import
        nuka_render
        nuka_cooker
        GTest::gtest_main
)
add_test(NAME VisualMeshCook
         COMMAND nuka_visual_mesh_cook_smoke_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(VisualMeshCook PROPERTIES LABELS "full")


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

gtest_discover_tests(nuka_runtime_test
    PROPERTIES LABELS "fast_full")

# --- Import tests (the consolidated HOST import/scene/cooker category) -------
add_executable(nuka_import_test
    # scene unit suites (were nuka_scene_test)
    scene/test_scene_ir.cpp
    scene/test_scene_compose.cpp
    scene/test_scene_cooker.cpp
    scene/test_contact_metadata_cook.cpp
    scene/test_filter_precedence.cpp
    # importers (mjcf / urdf / usd / usdc / mesh)
    import/test_mjcf_importer.cpp
    import/test_mjcf_mass_pipeline.cpp
    import/test_mjcf_contact_filtering.cpp
    import/test_inertia_friction.cpp
    import/test_urdf_importer.cpp
    import/test_usd_importer.cpp
    import/test_usd_mesh_geometry.cpp
    # v0.8 C7a: the self-written USDC (crate) reader on the REAL newton cup.
    import/test_usdc_cup.cpp
    import/test_decompose_attribute.cpp
    import/test_mesh_file_loader.cpp
    # V-HACD convex decomposition (was nuka_vhacd_test); needs the source-root
    # include for tests/import/vhacd_test_meshes.hpp (added below).
    import/test_vhacd_simple_cube.cpp
    import/test_vhacd_robot_finger.cpp
    import/test_vhacd_determinism.cpp
    import/test_vhacd_cooker_integration.cpp
    # sparse narrow-band SDF cooker (was nuka_sparse_sdf_test); host/CPU sampler.
    import/test_sparse_sdf_known_shapes.cpp
    import/test_sparse_sdf_thin_shell.cpp
    import/test_sparse_sdf_memory.cpp
    import/test_sparse_sdf_determinism.cpp
)

target_include_directories(nuka_import_test
    PRIVATE
        # repo root: tests/import/vhacd_test_meshes.hpp + tests/import/sparse_sdf_*.hpp
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_import_test
    PRIVATE
        nuka_import
        nuka_scene
        nuka_cooker
        GTest::gtest_main
)

gtest_discover_tests(nuka_import_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "fast_full")


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

gtest_discover_tests(nuka_collision_test
    PROPERTIES LABELS "fast_full")

# --- Constraint tests ------------------------------------------------------
add_executable(nuka_constraint_test
    constraint/test_contact_manifold.cpp
    constraint/test_contact_identity.cpp
    constraint/test_csr_row_layout.cpp
)

target_link_libraries(nuka_constraint_test
    PRIVATE
        nuka_constraint
        GTest::gtest_main
)

gtest_discover_tests(nuka_constraint_test
    PROPERTIES LABELS "fast_full")

# --- v0.8 C4b: compliant-normal + pyramid-friction row emitter -------------
# Validates the NEW, VALIDATED-NOT-WIRED EmitCompliantContactRows (row_builder):
# row counts (4 normal + 16 pyramid for a condim=3 box), per-normal R/aref
# cross-check vs an independent ComputeCompliantRow (C4a), the pyramid spokes
# summing to the isotropic disk, the ContactRowSides side-output, 2-run host
# byte-identity, and INERTNESS of the legacy BuildContactRows/AppendContactGroup
# path. Plain host C++ (no CUDA): nuka_constraint is host-only.
add_executable(nuka_compliant_rows_test
    constraint/test_compliant_rows.cpp
)
target_link_libraries(nuka_compliant_rows_test
    PRIVATE
        nuka_constraint
        GTest::gtest_main
)
gtest_discover_tests(nuka_compliant_rows_test
    PROPERTIES LABELS "fast_full")


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

gtest_discover_tests(nuka_articulation_test
    PROPERTIES LABELS "fast_full")

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

gtest_discover_tests(nuka_sensor_test
    PROPERTIES LABELS "fast_full")

# --- Batch context tests ---------------------------------------------------
add_executable(nuka_batch_test
    runtime/test_batch_context.cpp
)

target_link_libraries(nuka_batch_test
    PRIVATE
        nuka_runtime
        GTest::gtest_main
)

gtest_discover_tests(nuka_batch_test
    PROPERTIES LABELS "fast_full")

# --- Regression tests ------------------------------------------------------
add_executable(nuka_oracle_test
    oracle/featherstone_oracle_harness.cpp
)

target_compile_definitions(nuka_oracle_test
    PRIVATE
        NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(nuka_oracle_test
    PRIVATE
        nuka_runtime
        nuka_import
        $<$<TARGET_EXISTS:nuka_contact_wrench>:nuka_contact_wrench>
        $<$<TARGET_EXISTS:nuka_phi>:nuka_phi>
        $<$<TARGET_EXISTS:nuka_scene_cook>:nuka_scene_cook>
        GTest::gtest_main
)

target_include_directories(nuka_oracle_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
)

gtest_discover_tests(nuka_oracle_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "full")

add_executable(nuka_regression_test
    regression/test_free_fall.cpp
    regression/test_go2_stand_5s.cpp
)

target_compile_definitions(nuka_regression_test
    PRIVATE
        NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(nuka_regression_test
    PRIVATE
        nuka_solver
        nuka_articulation
        nuka_collision
        nuka_runtime
        $<$<TARGET_EXISTS:nuka>:nuka>
        GTest::gtest_main
)

target_include_directories(nuka_regression_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/src/include
)

gtest_discover_tests(nuka_regression_test
    PROPERTIES LABELS "full")

# --- Performance tests -----------------------------------------------------
# --- PHI tests (requires CUDA) --------------------------------------------
if(CMAKE_CUDA_COMPILER)
    add_executable(nuka_codegen_test
        codegen/test_codegen_roundtrip.cpp
    )

    add_dependencies(nuka_codegen_test nuka_codegen)

    target_compile_definitions(nuka_codegen_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NUKA_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
    )

    target_link_libraries(nuka_codegen_test
        PRIVATE
            nuka_codegen_generated
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_codegen_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    add_executable(nuka_adjoint_fd_test
        codegen/test_adjoint_fd_maximal_drive.cpp
        codegen/test_adjoint_fd_maximal_joint.cpp
        codegen/test_adjoint_fd_maximal_contact.cpp
        codegen/test_adjoint_fd_featherstone_contact.cpp
    )

    add_dependencies(nuka_adjoint_fd_test nuka_codegen)

    target_link_libraries(nuka_adjoint_fd_test
        PRIVATE
            nuka_codegen_v3_validation
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_adjoint_fd_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    add_executable(nuka_phi_test
        phi/test_capabilities.cpp
        phi/test_platform_contract.cpp
    )

    target_link_libraries(nuka_phi_test
        PRIVATE
            # nuka_phi supplies Capabilities / Device / PlatformContract; the
            # BUF-14 sweep deleted the legacy device-context / owned-stream /
            # stream-view types (and their pure-type tests).
            nuka_phi
            # BUF-13: test_capabilities's buffer tests are the phi-v2 surface
            # (BufferAlloc/InitBestDevice); nuka_phi2 supplies them.
            nuka_phi2
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_phi_test
        PROPERTIES LABELS "fast_full")

    # --- PHI v2 (M1) scenario gate ----------------------------------------
    # Exercises the ggml-style multi-backend layer (nuka_phi2) end-to-end:
    # registry -> device -> backend, buffer roundtrip, dispatch + CUDA-graph
    # plan replay (D1 byte-identical), event smoke. The op TU is CUDA-language;
    # the driver is host-only C++ (no CUDA types — phi v2 public headers are
    # pure C++). Registered via add_test with a `phi2_smoke` name so
    # `ctest -R phi2_smoke` finds it (gtest_discover_tests names would not
    # contain that substring).
    add_executable(nuka_phi2_smoke_test
        scenario/phi2_smoke.cpp
        scenario/phi2_smoke_ops.cu
    )

    target_include_directories(nuka_phi2_smoke_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/scenario
    )

    target_link_libraries(nuka_phi2_smoke_test
        PRIVATE
            nuka_phi2
            # M3a: the nk core (Model/Data/World) + CookToModel for the
            # NkWorldSkeleton acceptance case. nuka_scene_cook transitively pulls
            # nuka_nk + nuka_scene + nuka_articulation.
            nuka_nk
            nuka_scene_cook
            GTest::gtest_main
    )

    add_test(NAME phi2_smoke COMMAND nuka_phi2_smoke_test)
    set_tests_properties(phi2_smoke PROPERTIES LABELS "fast_full")


    add_executable(nuka_render_frame_loop_smoke_test
        scenario/render_frame_loop_smoke.cpp
    )
    add_dependencies(nuka_render_frame_loop_smoke_test nuka_codegen)
    target_compile_definitions(nuka_render_frame_loop_smoke_test
        PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    if(NK_BUILD_VULKAN_VALIDATION)
        target_compile_definitions(nuka_render_frame_loop_smoke_test
            PRIVATE NK_BUILD_VULKAN_VALIDATION)
    endif()
    target_include_directories(nuka_render_frame_loop_smoke_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_render_frame_loop_smoke_test
        PRIVATE
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_render_frame_loop_smoke_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # =====================================================================
    # CATEGORY EXE 3/5: nuka_perf_test -- throughput / timing GATES.
    # Consolidates: go2_4096env step-time, cuda_perf (lbvh 50k / 1M particles),
    # perf_timing infra, rt_two_level_bench. L1-c DELETED nk_union_n1.cpp (the
    # <=5ms/step + grasp throughput UnionCsr gate) with the UnionCsr path, so this
    # exe no longer links the union bridge (nuka_test_validated_sources). The RT
    # bench source keeps its per-file -ffp-contract=off (host==device bit-exact).
    # =====================================================================
    add_executable(nuka_perf_test
        perf/test_go2_4096env_step_time.cpp
        perf/test_lbvh_50k_bodies.cpp
        perf/test_1m_particles_grid_perf.cpp
        perf/test_perf_timing.cu
        rt/test_two_level_bench.cpp
    )
    add_dependencies(nuka_perf_test nuka_codegen)
    set_source_files_properties(rt/test_two_level_bench.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_compile_definitions(nuka_perf_test
        PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_perf_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_perf_test
        PRIVATE
            # L1-c: nuka_test_validated_sources (the union_nk_model bridge) was
            # dropped here with nk_union_n1.cpp (the UnionCsr perf gate).
            nuka_scene_cook
            nuka_nk
            nuka_phi2
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_runtime_gpu
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_phi2_rt
            nuka_core
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_perf_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "perf_full")

    add_executable(nuka_scenario_test
        scenario/cook_settle_determinism.cpp
        scenario/cloth_on_box.cpp
        # Multi-articulation co-residence + GENERAL-PATH collision: K Go2
        # articulations co-resident in ONE env of ONE general nk::World. Proves K
        # articulations co-step finite + INDEPENDENT, AND collide via the GENERAL
        # contact path (overlapping dogs push apart; dogs rest on a general
        # heightfield incl. step edges -- no dog_dog hack). Composes go2 with itself.
        scenario/multi_dog_costep.cpp
        # SCALE: 20 Go2 articulations co-resident in ONE env on PyramidStairs
        # terrain (~22400 rows in ONE per-env island). Pins the PairDriven solver's
        # storage cost so a 20-dog single-env world constructs (no dynamic-shared
        # overflow) and steps finite with feet loaded and no foot-sink-through.
        scenario/multi_dog_terrain_scale.cpp
        # General contact pipeline Phase 1A: the PairDriven NARROWPHASE DETECTION
        # gate. A body-only PairDriven world with two overlapping collidables runs
        # broadphase -> cvx GJK/EPA narrowphase -> the unified contact buffer;
        # asserts contacts detected, <=4 pts, no NaN, retained (a,b) ids, tangent
        # basis (C4), and two-run byte-identity (the new path's own D1).
        scenario/multi_geom_cook.cpp
        scenario/pairdriven_narrowphase.cpp
        # General contact pipeline Phase 1B: the PairDriven SOLVER + INTEGRATION gate.
        # Free x free (two boxes collide + symmetric momentum exchange), free x static
        # (box rests on a static ground plane), and two-run byte-identity for the new
        # mixed-island solve path (S1-S6 fed into the existing coupling engine).
        scenario/pairdriven_solve.cpp
        # Cloth <-> presser two-way coupling: a 576-particle VBD cloth drapes over a
        # low static ridge, a rigid presser dips it (cloth Δz down + spring-back on
        # lift) and a dynamic presser is arrested by the cloth (body-side reaction
        # lambda>0 vs a no-cloth control) -- all via the body<->particle row solver.
        scenario/cloth_presser_two_way.cpp
        # Articulated LINK <-> particle two-way coupling on the same body<->particle
        # row a foot uses on the ground: a minimal one-revolute pendulum arm presses a
        # cloth patch (coupling row whose BODY side maps to an articulation link, and
        # the reaction reaches the single revolute qdot + tip pose vs a no-patch control).
        scenario/articulated_link_particle_coupling.cpp
        # Phase 1B dog cases: artic x rigid (free box on a dog's back = mixed island)
        # and artic x artic (two dogs collide via the GENERAL path; the ONE path).
        scenario/pairdriven_dog_solve.cpp
        # General contact pipeline Phase 2A: the GENERAL HEIGHTFIELD narrowphase
        # gate. A box on a FLAT grid rests; ★ a box on a real PyramidStairs STEP
        # EDGE settles finite + no 穿模 (the exact gate2 explosion fix); per-cell
        # TRIANGLE_PRISM contacts via the EXISTING cvx GJK/EPA; two-run identity.
        scenario/pairdriven_heightfield.cpp
        scenario/vproof_go2_ground.cpp
        scenario/vproof_go2_terrain.cpp
        scenario/vproof_h1_grasp.cpp
        # M11 VIEW-4 + VIEW-6 host gates (command_queue -> ApplyMoveEntity -> nk::Data
        # round-trip + StepOnce same-frame). Pure host/Data path (no Vulkan), so it
        # runs in build-cuda128 alongside the other scenario gates.
        scenario/viewer_move_entity.cpp
        # MLS-MPM transfer ROUND-TRIP IDENTITY gate: clear -> P2G (deterministic
        # gather, no atomics) -> G2P with C=0/F=I reproduces particle velocity; the
        # gather is byte-identical run-to-run; snapshot/restore round-trips F; an
        # out-of-AABB particle flags the env-status escape bit.
        scenario/mpm_transfer_roundtrip.cpp
        # MLS-MPM elastic jelly-ball gate: a fixed-corotated ball drops onto the static
        # floor plane (grid BC), squashes <= ~25%, recovers, no collapse/spike/escape,
        # >= 5 s, finite, two-run byte-identical (the deterministic gather).
        scenario/mpm_jelly_ball.cpp
        # Two-way rigid <-> MPM coupling gate: a free rigid box rests on an MPM bed;
        # the per-substep grid->body reaction balances gravity (m*g*dt), the free-fall
        # BITE drops the box, two runs byte-identical (the deterministic per-body gather).
        scenario/rigid_on_mpm_rest.cpp
        # Articulated-LINK <-> MPM coupling gate: a fixed-base prismatic-Z link rests
        # on an MPM bed; the link's grid reaction seeds qdot_flat via M^-1 J^T (support,
        # balance, BITE, two-run byte-identical, one-way bit cleared, 2-link reduction).
        scenario/artic_link_on_mpm_rest.cpp
        # MLS-MPM granular Drucker-Prager constitutive gate (model_kind == 4): a
        # released sand column piles at a dp_friction-tracking repose angle (fluid
        # control flattens, elastic control stands); a scripted capsule press leaves
        # a >= 3 s persistent crater; particle state byte-identical run-to-run.
        scenario/mpm_granular.cpp
        scenario/pairdriven_rigid_release.cpp
        scenario/osc_passive_damping.cpp
        scenario/free_rigid_dynamics.cpp
        scenario/multi_articulation_reset.cpp
    )
    add_dependencies(nuka_scenario_test nuka_codegen)
    target_compile_definitions(nuka_scenario_test
        PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_scenario_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_scenario_test
        PRIVATE
            # L1-c: nuka_test_validated_sources (the union_nk_model bridge) was
            # dropped here -- the UnionCsr grasp gates are deleted; the surviving
            # general-path V-PROOF cooks via nuka_scene_cook's CookToModel.
            nuka_scene_cook
            # M11 VIEW-4/6: the frame-loop spine (Simulation / systems / publisher /
            # ApplyMoveEntity) for viewer_move_entity. nuka_app pulls nuka_render for
            # the RenderWorld POD types (Vulkan is a required package -> link OK even
            # with NK_BUILD_VULKAN_VALIDATION OFF; RenderSystem is then inert).
            nuka_app
            nuka_nk
            nuka_phi2
            nuka_solver
            nuka_constraint
            # nuka_runtime_gpu carries the cook-time cloth topology builder
            # (BuildClothConstraints) the cloth_presser_two_way scene uses.
            nuka_runtime_gpu
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_scenario_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    add_executable(nuka_contact_identity_cuda_test
        scenario/pairdriven_narrowphase.cpp
    )
    add_dependencies(nuka_contact_identity_cuda_test nuka_codegen)
    target_include_directories(nuka_contact_identity_cuda_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_contact_identity_cuda_test
        PRIVATE
            nuka_nk
            nuka_phi2
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_contact_identity_cuda_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TEST_PREFIX "ContactIdentityCuda."
        PROPERTIES LABELS "fast_full")


    add_executable(nuka_scene_compose_h1_cup_table_test
        scene/test_scene_compose_h1_cup_table.cpp
    )
    add_dependencies(nuka_scene_compose_h1_cup_table_test nuka_codegen)
    target_include_directories(nuka_scene_compose_h1_cup_table_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_scene_compose_h1_cup_table_test
        PRIVATE
            nuka_nk
            # nuka_scene_cook transitively pulls nuka_nk + nuka_scene +
            # nuka_articulation (CookToModel / FindRestPlacement / CookScene + the
            # .nks Save/Load live in nuka_scene).
            nuka_scene_cook
            nuka_import
            nuka_phi2
            nuka_phi
            nuka_core
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_scene_compose_h1_cup_table_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")


    # --- .nks /script node roundtrip. Host/CPU only (no device step): Saves/Loads a
    # SceneIR with /script nodes and asserts the records + the projected
    # ScriptComponent / parent path round-trip, plus a script-free scene stays
    # byte-identical (the additive looser gate).
    add_executable(nuka_nks_script_roundtrip_test
        scenario/nks_script_roundtrip.cpp
    )
    target_include_directories(nuka_nks_script_roundtrip_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_nks_script_roundtrip_test
        PRIVATE
            nuka_scene
            nuka_import
            nuka_core
            GTest::gtest_main
    )
    add_dependencies(nuka_nks_script_roundtrip_test nuka_codegen)
    add_test(NAME NksScriptRoundtrip
             COMMAND nuka_nks_script_roundtrip_test
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(NksScriptRoundtrip PROPERTIES LABELS "fast_full")


    # Particle volume cooker: lattice count, uniform mass, placement and sphere carve.
    # Host-only, but the cooker lives in the CUDA-language nuka_runtime_gpu lib.
    add_executable(nuka_particle_volume_cooker_test
        import/test_particle_volume_cooker.cpp
    )

    target_include_directories(nuka_particle_volume_cooker_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_particle_volume_cooker_test
        PRIVATE
            nuka_runtime_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_particle_volume_cooker_test
        PROPERTIES LABELS "fast_full")


    # --- v0.8 C4a: MuJoCo _efc_row solref/solimp compliance port -----------
    # Tests src/constraint/solref_solimp.hpp (header-only HD). Two regimes:
    #   (1) function-vs-hand-evaluated-table (tolerance, independent oracle), and
    #   (2) HOST-vs-DEVICE BYTE-EXACT parity -- the gate that justifies OPEN-F's
    #       integer-power decision. The .cpp holds a __global__ so it compiles as
    #       CUDA. Built --fmad=false (device) + -ffp-contract=off (host) so the
    #       shared HD arithmetic is host==device bit-for-bit (no FMA contraction
    #       on either side, integer-power => no libm powf divergence). Header-only
    #       => no library link beyond cudart + gtest.
    add_executable(nuka_solref_solimp_test
        constraint/test_solref_solimp.cpp
    )
    set_source_files_properties(constraint/test_solref_solimp.cpp
        PROPERTIES LANGUAGE CUDA
                   COMPILE_OPTIONS "--fmad=false;-Xcompiler=-ffp-contract=off")
    target_include_directories(nuka_solref_solimp_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_solref_solimp_test
        PRIVATE
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_solref_solimp_test
        PROPERTIES LABELS "fast_full")

    add_executable(nuka_lbvh_vs_sap_test
        collision/test_lbvh_vs_sap_pair_set.cpp
    )

    target_link_libraries(nuka_lbvh_vs_sap_test
        PRIVATE
            nuka_collision_gpu
            nuka_runtime_gpu
            nuka_runtime
            nuka_phi
            nuka_phi2  # phi-v2 buffer (collision broadphase on the phi-v2 Buffer*)
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_lbvh_vs_sap_test
        PROPERTIES LABELS "fast_full")

    # --- shared batched LBVH build primitive: env-batching == independent builds.
    # One BuildLbvhBatchedNodes over E envs == E single-env builds, byte-exact per
    # env (the u64 cross-env stable_sort isolates envs bit-exactly); same for the
    # batched refit + the split lo/hi overload. Drives lbvh_batched.cuh directly,
    # so it is a .cu; links nuka_collision_gpu (the primitive's compiled home).
    add_executable(nuka_lbvh_batched_test
        collision/test_lbvh_batched.cu
    )
    target_include_directories(nuka_lbvh_batched_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_lbvh_batched_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_lbvh_batched_test
        PROPERTIES LABELS "fast_full")

    # Point-mass endpoint operators and the compliant contact fixed point against dense
    # references over merged mass degrees of freedom. Drives union_types.cuh, so a .cu.
    add_executable(nuka_point_endpoint_operator_test
        solver/test_point_endpoint_operator.cu
    )
    target_include_directories(nuka_point_endpoint_operator_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_point_endpoint_operator_test
        PRIVATE
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_point_endpoint_operator_test
        PROPERTIES LABELS "fast_full")

    add_executable(nuka_particle_grid_test
        collision/test_particle_grid_correctness.cpp
        collision/test_particle_grid_determinism.cpp
        collision/test_cross_system_query.cpp
    )

    target_link_libraries(nuka_particle_grid_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            nuka_phi2  # phi-v2 buffer (collision broadphase on the phi-v2 Buffer*)
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_particle_grid_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C2a: unified candidate-pair stream (the broadphase OUTPUT type) ---
    # Canonical stable_key (type-in-high-bits, collision-free), thrust::stable_sort
    # builder over a fixed host-provided pair set, two-run byte-exact + N>=32
    # cross-replica D1. Links nuka_collision_gpu (carries candidate_pair.cu) +
    # nuka_phi. Needs nuka_codegen: candidate_pair.hpp includes the GENERATED
    # collidable_registry.hpp (via constraint/collidable.hpp).
    add_executable(nuka_candidate_stream_test
        collision/test_candidate_stream.cpp
    )
    add_dependencies(nuka_candidate_stream_test nuka_codegen)
    target_include_directories(nuka_candidate_stream_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_candidate_stream_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_candidate_stream_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C2b: RIGID<->RIGID filtered candidate-pair stream validation -----
    # BuildRigidCandidatePairs (LBVH over per-shape AABBs -> bitmask + exclude +
    # parent-child auto-exclude filter -> explicit-<pair> force-include -> sorted
    # D1 stream) vs an independent host oracle driven by real CookScene policy.
    # Gates: unfiltered==brute-force overlap, filtered==oracle (per-case), >=32
    # body D1 two-run byte-exact, empty/no-overlap sanity. Links nuka_collision_gpu
    # (rigid_candidate_pairs.cu) + nuka_scene (CookScene) + nuka_phi. Needs
    # nuka_codegen (candidate_pair.hpp -> generated collidable_registry.hpp).
    add_executable(nuka_lbvh_filtered_pairs_test
        collision/test_lbvh_filtered_pairs.cpp
    )
    add_dependencies(nuka_lbvh_filtered_pairs_test nuka_codegen)
    target_include_directories(nuka_lbvh_filtered_pairs_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_lbvh_filtered_pairs_test
        PRIVATE
            nuka_collision_gpu
            nuka_scene
            nuka_phi
            nuka_phi2  # phi-v2 buffer (collision broadphase on the phi-v2 Buffer*)
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_lbvh_filtered_pairs_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 co-residence (commit 2/3): ARTICULATION-LINK <-> RIGID candidate
    # pairs through the generalized tagged broadphase core. Synthetic mixed AABB
    # sets + per-AABB tags (no cooked robot) exercise: cross-type emission (link
    # AABB overlapping a rigid AABB -> one CandidatePair with correct per-side
    # type + handle, canonical stable_key), same-type alongside, cross-type
    # bitmask + exclude drops (non-vacuous), the rigid wrapper byte-identical to
    # the tagged core (regression), two-run D1 byte-exactness, and the handle<2^28
    # guard. Consumes commit 1's LinkShapeAabbs (link_aabb.hpp). VALIDATED-NOT-
    # WIRED. Links nuka_collision_gpu (rigid_candidate_pairs.cu) + nuka_phi. Needs
    # nuka_codegen (candidate_pair.hpp -> generated collidable_registry.hpp); the
    # scene::CookedFilterPolicy struct comes from nuka_scene (header-only use).
    add_executable(nuka_link_rigid_candidate_pairs_test
        collision/test_link_rigid_candidate_pairs.cpp
    )
    add_dependencies(nuka_link_rigid_candidate_pairs_test nuka_codegen)
    target_include_directories(nuka_link_rigid_candidate_pairs_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_link_rigid_candidate_pairs_test
        PRIVATE
            nuka_collision_gpu
            nuka_scene
            nuka_phi
            nuka_phi2  # phi-v2 buffer (collision broadphase on the phi-v2 Buffer*)
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_link_rigid_candidate_pairs_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C3a: extended ContactManifold + narrowphase dispatch table -------
    # Validates SelectTier routing (primitive->Analytical, mesh/convex->Convex,
    # SDF-equipped->Sdf), the constexpr (typeA,typeB,tier)->handler dispatch table
    # (looked-up fn-ptr == expected tier stub AND call-through stamps the tier
    # marker), and the extended ContactManifold round-trip (AddPoint clamp@4, Clear
    # reset, CollidableRef a/b type+react+handle, trivially-copyable memcpy
    # byte-stability). Header-only (narrowphase_dispatch.hpp + contact_manifold.hpp
    # + collidable.hpp) -- needs nuka_codegen (collidable.hpp -> generated
    # collidable_registry.hpp) + the src include dir; nuka_phi for completeness.
    add_executable(nuka_narrowphase_dispatch_test
        collision/test_narrowphase_dispatch.cpp
    )
    add_dependencies(nuka_narrowphase_dispatch_test nuka_codegen)
    # M11 COL-1: the salvaged SDF-tier gate (folded in) cooks a unit-cube SDF via
    # nuka_cooker (CookSparseSdf) and includes tests/import/sdf_{host_sampler,
    # test_meshes}.hpp -> add the repo-root include + nuka_cooker. The SDF forward
    # path is header-inline (find_sdf_contact_newton, NUKA_SDF_HD) so NO
    # nuka_collision_gpu is needed.
    target_include_directories(nuka_narrowphase_dispatch_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_narrowphase_dispatch_test
        PRIVATE
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_narrowphase_dispatch_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C3b: analytical primitive narrowphase (multi-point/face-face) -----
    # Validates the REAL Analytical-tier handlers (analytical_manifold.hpp +
    # narrowphase_dispatch.hpp): box-on-plane 4 corners, box-box face-clip 4-pt
    # patch (order-insensitive SET vs hand oracle), sphere x {sphere,box,plane}
    # 1-pt analytic, capsule x plane (2 endpoints) / capsule x sphere (1-pt),
    # both (a,b) orderings (normal sign = separation dir for side A), the
    # determinism-trap gate (symmetric box-box -> stable 4 distinct corners) and
    # the two-run byte-identical (memcmp on zeroed buffers) D1 gate. Header-only
    # (same deps as the C3a dispatch test) -- needs nuka_codegen + the src include.
    add_executable(nuka_analytical_manifold_test
        collision/test_analytical_manifold.cpp
    )
    add_dependencies(nuka_analytical_manifold_test nuka_codegen)
    target_include_directories(nuka_analytical_manifold_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_analytical_manifold_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_analytical_manifold_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C3c: convex narrowphase (GJK/EPA/face-clip) -----------------------
    # Validates the REAL Convex-tier handler (convex_narrowphase.hpp +
    # narrowphase_dispatch.hpp): two penetrating tetrahedra / two boxes-as-hulls vs
    # analytic depth+normal (and a TIGHT depth/normal cross-check vs the C3b box-box
    # SAT), a separated pair -> no contact, convex x box / convex x sphere / convex
    # x plane (both orderings, the plane special path + sign), and the EPA D1 gate:
    # 2-run byte-identity (memset+memcmp) + N>=32 cross-replica on a SYMMETRIC
    # tie-prone box-box config (a heap-based EPA would diverge there). Header-only
    # (same deps as the C3a/C3b dispatch tests) -- needs nuka_codegen + src include.
    add_executable(nuka_gjk_epa_convex_test
        collision/test_gjk_epa_convex.cpp
    )
    add_dependencies(nuka_gjk_epa_convex_test nuka_codegen)
    target_include_directories(nuka_gjk_epa_convex_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    # nuka_import/nuka_scene/nuka_cooker: the C3-hardening regression
    # (SphereHullShallowPenetrationIsMonotone) loads the REAL cup hull -- the only
    # geometry that reproduces the EPA shallow-penetration dead band -- via
    # LoadUsd + CookScene (asset-gated; SKIPs when the fetch-per-env cup is absent).
    target_link_libraries(nuka_gjk_epa_convex_test
        PRIVATE
            nuka_phi
            nuka_phi2       # M5: the NarrowphaseSdf op (SDF precision oracle).
            nuka_import
            nuka_scene
            nuka_cooker
            CUDA::cudart    # cudaDeviceSynchronize in the SDF oracle harness.
            GTest::gtest_main
    )
    # WORKING_DIRECTORY = repo root so the relative cup asset path resolves.
    gtest_discover_tests(nuka_gjk_epa_convex_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")


    # --- v0.8 C5a: contact stream DRIVER (CandidatePair -> ContactManifold) ------
    # Validates the resolver-decoupled host driver (contact_stream_driver.hpp/.cpp,
    # BuildContactManifolds): resolve each side via a caller-supplied ShapeResolver
    # (the rigid-BodyState / articulation-link-pose decoupling seam), derive
    # has_sdf = a.has_sdf && b.has_sdf (both-sides contract), route through the
    # constexpr dispatch table, invoke the handler, merge the two sides' MuJoCo
    # contact params via scene::MergeContactParams, and append non-empty manifolds.
    # Cases: box-on-plane (rigid x static) geometry == direct ResolveNarrowphase
    # call + merged friction/solimp/solref + manifold_key/a/b; foot sphere-on-plane
    # (articulation-link x static, the C5c foot-ground shape); box-box (rigid x
    # rigid) multi-point == direct call; param merge cross-checked vs an
    # INDEPENDENT MergeContactParams call; has_sdf both-sides (one-sided -> real
    # analytical, both-sided + null seam -> Sdf null-guards -> skipped); skip when a
    # side fails to resolve; D1 two-run byte-exact (memcmp). Compiles the driver
    # .cpp directly (host-only -- the dispatch table is host fn-ptrs; the device
    # driver is C5/later). Header deps == the C3a dispatch test (narrowphase_
    # dispatch.hpp + contact_filter.hpp) -- nuka_codegen + the src include + nuka_phi.
    add_executable(nuka_contact_stream_driver_test
        collision/test_contact_stream_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_contact_stream_driver_test nuka_codegen)
    target_include_directories(nuka_contact_stream_driver_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_contact_stream_driver_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_contact_stream_driver_test
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C2c: particle<->rigid + particle<->particle candidate-pair builders
    # gated by the C1c SystemPairMatrix. Validates the builders' reframed candidate
    # SET against (1) a hand-reframe of the raw cross-system CSR (non-identity
    # shape->body map + 2-shapes->1-body dedup case), (2) a brute-force i<j radius
    # oracle for p<->p, (3) the matrix-toggle gate (non-vacuous, same scene),
    # (4) >=32-particle 2-run byte-exact D1, (5) 0-particle sanity. Links
    # nuka_collision_gpu (particle_candidate_pairs.cu). Needs nuka_codegen
    # (candidate_pair.hpp -> generated collidable_registry.hpp).
    add_executable(nuka_cross_system_matrix_test
        collision/test_cross_system_matrix.cpp
    )
    add_dependencies(nuka_cross_system_matrix_test nuka_codegen)
    target_include_directories(nuka_cross_system_matrix_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_cross_system_matrix_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            nuka_phi2  # phi-v2 buffer (collision broadphase on the phi-v2 Buffer*)
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_cross_system_matrix_test
        PROPERTIES LABELS "fast_full")

    # --- v0.7 p12: self-written CUDA ray tracer (camera -> LBVH -> closest-hit
    # depth, box leaves). Matches a CPU brute-force ray-vs-all-boxes oracle that
    # REUSES the same __host__ __device__ slab test + closest-hit update + camera
    # ray-gen as the GPU kernel; D1 two-run byte-exact (memcmp); degenerate-deep
    # tree (deeper than stack[64]) renders with no overflow. The oracle TU is
    # built with -ffp-contract=off to pair with the kernel's --fmad=false so the
    # shared slab arithmetic is host==device bit-for-bit (depth max-err == 0).
    add_executable(nuka_rt_traversal_test
        rt/test_bvh_ray_traversal.cpp
    )
    set_source_files_properties(rt/test_bvh_ray_traversal.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_traversal_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_traversal_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_traversal_test
        PROPERTIES LABELS "fast_full")

    add_executable(nuka_rt_scene_render_test
        rt/test_scene_render.cpp
    )
    set_source_files_properties(rt/test_scene_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_scene_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_scene_render_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_scene_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.7 G1-A: the self-written TWO-LEVEL (TLAS/BLAS) ray tracer
    # (rt::BuildTwoLevelScene + rt::RenderFrame). Gates:
    #   * Oracle 1 (host two-level brute-force, shares QuatInvRotate/intersect/
    #     PackPrimId): byte-exact host==device across ALL 6 AOVs for ANY transform
    #     incl. rotation (D1 + host/device-parity + traversal-independence).
    #   * Oracle 2 (the trusted INDEPENDENT flat p13 RenderScene with each
    #     instance baked to world verts): memcmp==0 depth/normal/color/albedo at
    #     IDENTITY instances; near-exact (stated tol) for rotated/translated.
    #   * Analytic anchors: p13's single-triangle t through the nest + a NEW
    #     rotated+translated instance with a hand-computed world t AND an exact
    #     prim_id == PackPrimId(known_instance, known_local) (bit-split + instance-
    #     high ordering gate).
    #   * Instance-transform tracking (move one instance, reuse the BLAS).
    #   * D1 two-run byte-exact across ALL 6 AOVs.
    # Built -ffp-contract=off to pair with the kernel's --fmad=false so Oracle 1's
    # shared fp64-internal math + fp32 QuatInvRotate is host==device bit-for-bit.
    add_executable(nuka_rt_two_level_test
        rt/test_two_level_render.cpp
    )
    set_source_files_properties(rt/test_two_level_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_two_level_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_two_level_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_two_level_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Batched-sensor device scatter gate: ScatterEnvInstances composes, per
    # (env x instance), world = fk*cvl from env-major device pose buffers and writes
    # a DevInstance[E*M] table + env-major world-AABBs (the batched LBVH leaf
    # bounds). Asserts both outputs are byte-identical to a host fk*cvl reference
    # over Link/Body/Base/Static rows + >=3 envs. A .cu (drives the device launcher +
    # device structs directly); --fmad=false + host -ffp-contract=off pair the
    # device compose with the host reference bit-for-bit.
    add_executable(nuka_rt_sensor_scatter_test
        rt/test_sensor_scatter.cu
    )
    target_compile_options(nuka_rt_sensor_scatter_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_sensor_scatter_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_sensor_scatter_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_sensor_scatter_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Batched sensor render gate: a per-env tile of the ONE batched (N,H,W,ch)
    # AOV tensor is byte-identical to a standalone single-env (E=1) render of that
    # env's scene + camera (same FP32 trace + same per-env TLAS + same camera ->
    # same bytes). Drives BuildBatchedSensorScene + RenderSensorsBatched directly
    # over envs 0/mid/last, comparing color/depth/normal/albedo/prim. A .cu (drives
    # the device structs + launchers); --fmad=false matches the render TU.
    add_executable(nuka_rt_batched_sensor_render_test
        rt/test_batched_sensor_render.cu
    )
    target_compile_options(nuka_rt_batched_sensor_render_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_batched_sensor_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_batched_sensor_render_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_batched_sensor_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- GPU lidar range trace gate: lidar rides the SAME batched TLAS/LBVH/
    # ClosestHit the camera path builds. Known-distance geometry along a known
    # (az,el) ray -> range == distance; miss -> max_range; clamp to [min,max];
    # per-env tiles byte-identical; the mounted path matches the device path. A .cu;
    # --fmad=false matches the render TU.
    add_executable(nuka_rt_batched_lidar_render_test
        rt/test_batched_lidar_render.cu
    )
    target_compile_options(nuka_rt_batched_lidar_render_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_batched_lidar_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_batched_lidar_render_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_batched_lidar_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Per-env render domain-randomization gate: the trace always reads
    # material/light/ambient BY ENV. DR off -> base replicas -> cross-env tiles
    # byte-identical; DR on -> env tiles differ (appearance variety) AND are
    # deterministic (same seed -> byte-identical). A .cu; --fmad=false matches the
    # render TU so the per-env tables shade bit-exactly.
    add_executable(nuka_rt_render_dr_test
        rt/test_render_dr.cu
    )
    target_compile_options(nuka_rt_render_dr_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_render_dr_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_render_dr_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_render_dr_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Opt-in shading-fidelity gate: SetSensorFidelity lifts the batched RGB to
    # the beauty look (MSAA + soft shadow + AO + GI + tonemap) in the SAME kernel,
    # gated on the config. DEFAULT == byte no-op; fidelity-on is deterministic
    # (Philox-seeded), visibly different + finite, AOVs unchanged. A .cu;
    # --fmad=false matches the render TU.
    add_executable(nuka_rt_sensor_fidelity_test
        rt/test_sensor_fidelity.cu
    )
    target_compile_options(nuka_rt_sensor_fidelity_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_sensor_fidelity_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_sensor_fidelity_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_sensor_fidelity_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Per-env camera-mount gate: ScatterEnvCameras builds each env's pinhole
    # camera on device from cam_world = fk(pose) * local_offset (USD/Isaac axes).
    # Asserts the device cameras are byte-identical to the host fk*local_offset
    # composition + shared BuildPinholeBasis over Link/Body/Base/Static + >=3 envs,
    # then drives the batched trace with the scatter-built cameras (non-empty render
    # + per-env tile == E=1 render). A .cu; --fmad=false + host -ffp-contract=off.
    add_executable(nuka_rt_sensor_camera_test
        rt/test_sensor_camera.cu
    )
    target_compile_options(nuka_rt_sensor_camera_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_sensor_camera_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_sensor_camera_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_sensor_camera_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Camera intrinsics + lens distortion gate. Default intrinsics (fx<=0) =>
    # ray-gen byte-identical to the plain pinhole; distortion bends off-axis rays
    # (center unchanged); off-center principal point + per-axis focal shift rays
    # (known-value check); far/near clip turns out-of-range hits into depth misses.
    # A .cu; --fmad=false + host -ffp-contract=off so host==device.
    add_executable(nuka_rt_sensor_intrinsics_test
        rt/test_sensor_intrinsics.cu
    )
    target_compile_options(nuka_rt_sensor_intrinsics_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_sensor_intrinsics_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_sensor_intrinsics_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_sensor_intrinsics_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Batched device-resident RGB-sensor FEASIBILITY bench. Measures the
    # per-env-TLAS render path (BuildTwoLevelScene + RenderFrameToAovs, the cheap
    # flat trace) across N envs x resolution and decomposes TLAS build/refit vs
    # trace vs N-launch host overhead. Standalone main (a bench, not a gate);
    # -ffp-contract=off to match the rt suite. NOT a discovered test. -------------
    add_executable(nuka_rt_batched_sensor_bench
        rt/test_batched_sensor_bench.cpp
    )
    set_source_files_properties(rt/test_batched_sensor_bench.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_batched_sensor_bench
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_batched_sensor_bench
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
    )

    # --- Batched sensor PERF deliverable: times the device-resident BATCHED path
    # (ONE batched LBVH build + ONE flat [E*H*W] trace into a (N,H,W,ch) tensor)
    # vs the naive per-env loop at N=256/1024/4096 x 64^2 x 1spp and prints the
    # speedup + fps -- the near-real-time verdict. A .cu (drives the batched render
    # entry); --fmad=false matches the render TU. Standalone main, NOT discovered.
    add_executable(nuka_rt_batched_sensor_perf
        rt/test_batched_sensor_perf.cu
    )
    target_compile_options(nuka_rt_batched_sensor_perf PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rt_batched_sensor_perf
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_batched_sensor_perf
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
    )

    # --- H1 cup demo: PBR-style hero scene smoke over the two-level RT path ----
    add_executable(nuka_h1_cup_pbr_render_test
        rt/test_h1_cup_pbr_render.cpp
    )
    set_source_files_properties(rt/test_h1_cup_pbr_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_h1_cup_pbr_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_h1_cup_pbr_render_test
        PRIVATE
            nuka_phi2_rt
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_h1_cup_pbr_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- M11 RT-5: the RenderWorld -> rt_adapter -> RtBackendI offscreen D1 gate.
    # Drives the §3.10 backend-agnostic RT interface (render::RtBackendI, impl in
    # nuka_phi2_rt) from a RenderWorld through the host CUDA-free adapter
    # (render::RenderWorldToTwoLevelScene, in nuka_render). Asserts: (a) two
    # dispatches 6-AOV byte-identical (D1), (b) adapter scene == hand-built oracle
    # 6-AOV byte-exact (faithful mapping), (b') BufferI device-output path ==
    # host-convenience download (OD-12 output contract), (c) non-trivial coverage.
    # Built -ffp-contract=off to stay consistent with the rt kernel --fmad=false.
    # In the CUDA region OUTSIDE NK_BUILD_VULKAN (this box runs CUDA RT locally).
    add_executable(nuka_rt_render_world_test
        rt/test_render_world_rt.cpp
    )
    set_source_files_properties(rt/test_render_world_rt.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_render_world_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_render_world_test
        PRIVATE
            nuka_phi2_rt
            nuka_render
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_render_world_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    add_executable(nuka_elastoplastic_compression_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/elastoplastic_compression_demo.cpp)
    add_dependencies(nuka_elastoplastic_compression_demo nuka_codegen)
    target_compile_definitions(nuka_elastoplastic_compression_demo PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_elastoplastic_compression_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_elastoplastic_compression_demo PRIVATE
        nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2
        nuka_runtime_gpu nuka_runtime nuka_import nuka_cooker
        nuka_contact_wrench nuka_collision_gpu nuka_core nuka_phi)

    add_executable(nuka_robot_elastoplastic_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/robot_elastoplastic_demo.cpp)
    add_dependencies(nuka_robot_elastoplastic_demo nuka_codegen)
    target_compile_definitions(nuka_robot_elastoplastic_demo PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_robot_elastoplastic_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_robot_elastoplastic_demo PRIVATE
        nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2
        nuka_runtime_gpu nuka_runtime nuka_import nuka_cooker
        nuka_contact_wrench nuka_collision_gpu nuka_core nuka_phi)

    add_executable(nuka_bunny_water_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/bunny_water_demo.cpp)
    add_dependencies(nuka_bunny_water_demo nuka_codegen)
    target_compile_definitions(nuka_bunny_water_demo PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_bunny_water_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_bunny_water_demo PRIVATE
        nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2
        nuka_runtime_gpu nuka_runtime nuka_import nuka_cooker
        nuka_contact_wrench nuka_collision_gpu nuka_core nuka_phi)

    add_executable(nuka_elastoplastic_bunny_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/elastoplastic_bunny_demo.cpp)
    add_dependencies(nuka_elastoplastic_bunny_demo nuka_codegen)
    target_compile_definitions(nuka_elastoplastic_bunny_demo PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_elastoplastic_bunny_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_elastoplastic_bunny_demo PRIVATE
        nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2
        nuka_runtime_gpu nuka_runtime nuka_import nuka_cooker
        nuka_contact_wrench nuka_collision_gpu nuka_core nuka_phi)

    # --- Offline RHI replay gate: the host render layer (nuka_rhi) over the
    # PHI-unified tracer must add ZERO numeric change vs driving RtBackendI
    # directly. Renders one RenderWorld both ways with a Beauty profile and
    # asserts the framebuffers are byte-identical across all AOVs (D1). Built
    # -ffp-contract=off to stay consistent with the rt kernel's --fmad=false.
    add_executable(nuka_rhi_offline_replay_test
        rhi/test_offline_replay.cpp
    )
    set_source_files_properties(rhi/test_offline_replay.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rhi_offline_replay_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rhi_offline_replay_test
        PRIVATE
            nuka_rhi
            nuka_phi2_rt
            nuka_render
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rhi_offline_replay_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Offline RHI sensor-batch gate: the SensorRenderer (the host render layer
    # over the PHI-unified batched sensor render) must add ZERO numeric change vs
    # driving the rt-layer batched entry (rt::RenderSensorsMounted) directly. Builds
    # one N-env replicated sensor scene both ways with a Sensor profile and asserts
    # the device AOV tensors are byte-identical across all channels (downloaded for
    # the memcmp only). A .cu (the reference drives the CUDA-token rt entry); the
    # render TU's --fmad=false + host -ffp-contract=off keep host==device.
    add_executable(nuka_rhi_sensor_batch_test
        rhi/test_sensor_batch.cu
    )
    target_compile_options(nuka_rhi_sensor_batch_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
    )
    target_include_directories(nuka_rhi_sensor_batch_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rhi_sensor_batch_test
        PRIVATE
            nuka_rhi
            nuka_phi2_rt
            nuka_render
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rhi_sensor_batch_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")


    add_executable(nuka_sdf_contact_adjoint_test
        collision/test_sdf_contact_adjoint_fd.cpp
    )

    target_include_directories(nuka_sdf_contact_adjoint_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_sdf_contact_adjoint_test
        PRIVATE
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sdf_contact_adjoint_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")


    add_executable(nuka_sensor_noise_test
        sensor/test_n1_gaussian.cu
        sensor/test_n1_poisson.cu
    )

    target_link_libraries(nuka_sensor_noise_test
        PRIVATE
            nuka_sensor_noise
            # BUF-13: the n1 noise tests' device buffers are phi-v2 (stream-0);
            # nuka_phi2 supplies BufferAlloc/InitBestDevice. nuka_phi stays for
            # the scoped device guard the noise launchers use (BUF-14).
            nuka_phi
            nuka_phi2
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sensor_noise_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")


    # --- v0.8 co-residence PEAK (commit 1/3): per-link/per-shape WORLD AABB
    # extraction from articulation FK (src/collision/link_aabb.hpp). The
    # artic-link<->rigid broadphase PREREQUISITE C2b excluded (no per-link AABB).
    # Cooks go2_float, sweeps >=3 distinct FK poses, asserts each foot sphere's
    # world AABB contains its surface samples + is exactly tight, a synthetic
    # rotated-box AABB == an independent 8-corner reference (and grows under
    # rotation), a capsule's caps are contained, D1 two-run byte-identity, AND a
    # __global__ kernel proves the HD core compiles + runs on-device == host
    # (so commit 2 can call it in a .cu). VALIDATED-NOT-WIRED. The .cu drives the
    # device-core proof; needs nuka_codegen for the generated registry CookScene/
    # BuildWorld pull in (same dep set as the foot-ground subsume exe).
    add_executable(nuka_link_aabb_test
        collision/test_link_aabb.cu
    )
    add_dependencies(nuka_link_aabb_test nuka_codegen)

    target_compile_definitions(nuka_link_aabb_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_link_aabb_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_link_aabb_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_scene_cook
            nuka_contact_wrench
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_link_aabb_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")


endif()

# --- Beauty texture pipeline (host: stb decode + texture library + adapter) --
# Gates the untextured no-op (an old material maps to identical rt::Material
# data) + the textured plumbing (decode, dedupe, UV ride-along, miss-color
# report). Pure host; no CUDA, no Vulkan.
add_executable(nuka_texture_pipeline_test
    render/test_texture_pipeline.cpp
)
target_include_directories(nuka_texture_pipeline_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(nuka_texture_pipeline_test
    PRIVATE
        nuka_render
        nuka_import
        nuka_stb
        GTest::gtest_main
)
gtest_discover_tests(nuka_texture_pipeline_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES LABELS "fast_full")


# --- Vulkan renderer tests -------------------------------------------------
if(NK_BUILD_VULKAN_VALIDATION)
    add_executable(nuka_render_test
        render/test_vulkan_backend.cpp
    )

    target_link_libraries(nuka_render_test
        PRIVATE
            nuka_render
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_render_test
        PROPERTIES LABELS "fast_full")

    add_executable(nuka_render_raster_smoke_test
        scenario/render_raster_smoke.cpp
    )

    target_include_directories(nuka_render_raster_smoke_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_render_raster_smoke_test
        PRIVATE
            nuka_render
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_render_raster_smoke_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # Compare live four-system poses and particle surfaces with rendered geometry.
    # Check environment selection, repeated stage updates and image determinism.
    add_executable(nuka_render_physics_parity_test
        scenario/render_physics_parity.cpp
    )
    add_dependencies(nuka_render_physics_parity_test nuka_codegen)
    target_compile_definitions(nuka_render_physics_parity_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_render_physics_parity_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_render_physics_parity_test
        PRIVATE
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_phi2_rt
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_render_physics_parity_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # L1-c: the nuka_h1_grasp_video demo exe (examples/demo/h1_grasp_video.cpp +
    # h1_grasp_driver.hpp) was DELETED with the UnionCsr path -- it rendered the
    # UnionCsr cup-grasp choreography via BuildNkUnionModel (union_nk_model.cpp,
    # gone). Cup-grasping moved to RL; the grasp video will be re-authored on the
    # general path once the RL grip is trained.

    # M10 Phase A -- THE SECOND HEADLINE DEMO EXE: go2_walk_video. A unified
    # offscreen C++ video tool that POSE-REPLAYS the proven, golden-validated Go2
    # WALK (out/go2_walk_trajectory.bin, +3.91 m forward, D1) to a PBR-beautiful
    # PPM sequence on lavapipe -- the M8.5 go2 visual MESHES (go2.nks/.nka), a
    # premium dark-body / metallic-joint material override, a base-tracking hero
    # follow-camera, and the studio ground at z=0. PURE KINEMATIC REPLAY: no
    # device, no nk::World, no policy, no torch in C++ -- it Load(.nks) ->
    # CookToModel (host-only, for the EntityId->link SceneMap) -> BuildRenderWorld
    # and drives each visual instance's world_xform from the dumped per-frame
    # link_pose. NOT a gtest -- a real main() exe. Behind NK_BUILD_VULKAN_VALIDATION
    # (real Vulkan device). nuka_scene_cook (CookToModel) transitively pulls the nk
    # core; the link set mirrors the H1 video exe so the cook symbols resolve.
    add_executable(nuka_go2_walk_video
        ${CMAKE_SOURCE_DIR}/examples/demo/go2_walk_video.cpp
    )
    add_dependencies(nuka_go2_walk_video nuka_codegen)
    target_compile_definitions(nuka_go2_walk_video
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_go2_walk_video
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_go2_walk_video
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    add_executable(nuka_go2_skill_video_cuda
        ${CMAKE_SOURCE_DIR}/examples/demo/go2_skill_video_cuda.cpp
    )
    add_dependencies(nuka_go2_skill_video_cuda nuka_codegen)
    target_compile_definitions(nuka_go2_skill_video_cuda PRIVATE
        NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_go2_skill_video_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_go2_skill_video_cuda PRIVATE
        nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2
        nuka_runtime_gpu nuka_runtime nuka_import nuka_cooker
        nuka_contact_wrench nuka_collision_gpu nuka_core nuka_phi)


    # MLS-MPM headline clip: a fixed-corotated elastic jelly ball (densely sampled
    # on a lattice, sim_method=mlsmpm via the config selector) dropped onto the
    # STATIC floor plane (the grid BC, no dynamic body). STEPS nk::World on CUDA and
    # renders the live deforming surface (skin bound to the nearest MPM particles) on
    # the CUDA RT backend (--gpu --beauty) or lavapipe raster. Replaces the soft-ball.
    add_executable(nuka_mpm_jelly_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/mpm_jelly_demo.cpp
    )
    add_dependencies(nuka_mpm_jelly_demo nuka_codegen)
    target_compile_definitions(nuka_mpm_jelly_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_mpm_jelly_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_mpm_jelly_demo
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_phi2_rt
            nuka_nk
            nuka_phi2
            nuka_runtime_gpu
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # Two-way rigid<->MPM headline clip: a free rigid box presses into a firm MLS-MPM
    # bed (sim_method=mlsmpm via the config selector); the grid->box reaction holds
    # the box up and the bed deforms under it. STEPS nk::World on CUDA and renders the
    # box cube (posed from body_pose) + the bed isosurface (the fluid marching-cubes
    # mesher over the MPM particle set) on the CUDA RT backend or lavapipe raster.
    add_executable(nuka_mpm_rigid_rest_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/mpm_rigid_rest_demo.cpp
    )
    add_dependencies(nuka_mpm_rigid_rest_demo nuka_codegen)
    target_compile_definitions(nuka_mpm_rigid_rest_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_mpm_rigid_rest_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_mpm_rigid_rest_demo
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_phi2_rt
            nuka_nk
            nuka_phi2
            nuka_runtime_gpu
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )


    # Go2 HANDSTAND (倒立) hero clip -- the same offscreen GO2W-replay renderer as
    # go2_walk_video (Load(go2.nks) -> CookToModel SceneMap -> BuildRenderWorld ->
    # drive each visual instance from the dumped per-sub-step link_pose), but with a
    # LOW slow-orbit + dolly-in hero camera framing the INVERTED dog from below.
    # Replays out/go2_handstand_trajectory.bin (the deterministic, gate-passing
    # handstand rollout dumped by go2_dump_handstand_trajectory.py).
    add_executable(nuka_go2_handstand_video
        ${CMAKE_SOURCE_DIR}/examples/demo/go2_handstand_video.cpp
    )
    add_dependencies(nuka_go2_handstand_video nuka_codegen)
    target_compile_definitions(nuka_go2_handstand_video
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_go2_handstand_video
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_go2_handstand_video
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # Go2-on-stairs demo PHASE 3 -- go2_terrain_demo: the MULTI-DOG TERRAIN RENDER
    # (Isaac-Lab-cover style). Renders N Go2 each on its own procedural-terrain tile
    # (tessellated by sampling the SHARED nuka::terrain::SampleTerrainHeight() so the
    # rendered ground matches the physics surface -- anti-drift), with a ROTATING
    # orbit camera, atmospheric white haze (the new fog SceneUbo slot), and per-foot
    # contact shadows placed at the LOCAL terrain height under each foot. Consumes an
    # N-robot GO2T trajectory (the real trained-terrain rollout seam, --bin) or
    # synthesises a PLACEHOLDER by replicating the single-robot GO2W flat walk
    # (out/go2_walk_trajectory.bin) to N tiled robots. PURE KINEMATIC REPLAY: no
    # device, no nk::World, no policy. NOT a gtest -- a real main() exe. Behind
    # NK_BUILD_VULKAN_VALIDATION. Same link set as the other video exes.
    add_executable(nuka_go2_terrain_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/go2_terrain_demo.cpp
    )
    add_dependencies(nuka_go2_terrain_demo nuka_codegen)
    target_compile_definitions(nuka_go2_terrain_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_go2_terrain_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_go2_terrain_demo
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_phi2_rt
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # M10 -- scene_still: render ONE offscreen still of any .nks (kitchen-scene
    # verification + the H1-kitchen-grasp layout/look STILL the owner approves
    # BEFORE the RL effort). Load(.nks) -> CookToModel (host-only SceneMap) ->
    # BuildRenderWorld -> rasterise one frame (grounded studio look). Same link
    # set as the video exes so the cook symbols resolve.
    add_executable(nuka_scene_still
        ${CMAKE_SOURCE_DIR}/examples/demo/scene_still.cpp
    )
    add_dependencies(nuka_scene_still nuka_codegen)
    target_compile_definitions(nuka_scene_still
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_scene_still
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_scene_still
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # scene_load_probe -- headless load-path check: compiles the real editor_scene.cpp
    # and runs viewer::LoadEditorScene over every discovered .nks with no window.
    add_executable(nuka_scene_load_probe
        ${CMAKE_SOURCE_DIR}/examples/demo/scene_load_probe.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
    )
    add_dependencies(nuka_scene_load_probe nuka_codegen)
    target_compile_definitions(nuka_scene_load_probe
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_scene_load_probe
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_scene_load_probe
        PRIVATE
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # go2_play_smoke -- headless closed-loop smoke: cook go2_terrain30.nks via the
    # editor load path, attach the Go2PolicyController, drive N control steps, and
    # assert the dogs stay in a sane standing band over the retained terrain.
    add_executable(nuka_go2_play_smoke
        ${CMAKE_SOURCE_DIR}/examples/demo/go2_play_smoke.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/app/viewer/editor_scene.cpp
    )
    add_dependencies(nuka_go2_play_smoke nuka_codegen)
    target_compile_definitions(nuka_go2_play_smoke
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_go2_play_smoke
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_go2_play_smoke
        PRIVATE
            nuka_inference
            nuka_policy_mlp
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # L1-c: the nuka_union_throughput demo exe (examples/demo/union_throughput.cpp)
    # was DELETED with the UnionCsr path -- it measured batched UnionCsr
    # H1+cup+table throughput via BuildNkUnionModel (union_nk_model.cpp, gone). The
    # GENERAL-path throughput floor is now measured by vproof_go2_throughput below.

    # V-PROOF throughput gate (one-general-solver-spec §4) -- vproof_go2_throughput:
    # measure the GENERAL-path (PairDriven + flat heightfield) go2 env-steps/sec at
    # N in {1,64,256,1024} vs the >= 10000 env-steps/s training-ready floor. Mirrors
    # nuka_union_throughput's structure (cook once per N -> nk::World -> StepPlanned
    # loop + timing + finite-q). No Vulkan (shares the gated link set for the cook).
    add_executable(vproof_go2_throughput
        ${CMAKE_SOURCE_DIR}/examples/demo/vproof_go2_throughput.cpp
    )
    add_dependencies(vproof_go2_throughput nuka_codegen)
    target_compile_definitions(vproof_go2_throughput
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(vproof_go2_throughput
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(vproof_go2_throughput
        PRIVATE
            # L1-c: dropped nuka_test_validated_sources (the deleted union bridge)
            # -- this general-path throughput exe cooks via nuka_scene_cook only.
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # L-RECON-B cook profile -- lreconb_cook_profile: split the general
    # CookToModel(PairDriven) cook of the H1 union scene across the two HEAVY
    # per-shape stages (V-HACD, sparse-SDF bake) to confirm which dominates and
    # quantify the single-hull + no-SDF fix. Shares the gated cook link set.
    add_executable(lreconb_cook_profile
        ${CMAKE_SOURCE_DIR}/examples/demo/lreconb_cook_profile.cpp
    )
    add_dependencies(lreconb_cook_profile nuka_codegen)
    target_compile_definitions(lreconb_cook_profile
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(lreconb_cook_profile
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(lreconb_cook_profile
        PRIVATE
            # L1-c: dropped nuka_test_validated_sources (the deleted union bridge)
            # -- this general-path cook profiler cooks via nuka_scene_cook only.
            nuka_app
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_contact_wrench
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # M8.5 GATE-C -- viewport_present_smoke: the RUN-VERIFIED headless present-loop
    # smoke. Creates a VK_KHR_xcb_surface under Xvfb -> swapchain -> acquire ->
    # draw -> present for N frames. Links nuka_present (the swapchain renderer + xcb
    # backend) + nuka_render (the borrowed raster renderer). SKIPs cleanly with no
    # display / no surface ext / no Vulkan device. RUN via:
    #   VK_ICD_FILENAMES=<lavapipe icd> xvfb-run -a -s '-screen 0 1280x720x24' \
    #     tests/nuka_viewport_present_smoke_test
    add_executable(nuka_viewport_present_smoke_test
        scenario/viewport_present_smoke.cpp
    )
    add_dependencies(nuka_viewport_present_smoke_test nuka_render_shaders)
    target_compile_definitions(nuka_viewport_present_smoke_test
        PRIVATE NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_viewport_present_smoke_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_viewport_present_smoke_test
        PRIVATE
            nuka_present
            nuka_render
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_viewport_present_smoke_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # M8.5 GATE-B -- viewer_frame_smoke: the RUN-VERIFIED OFFSCREEN viewer-frame
    # composite. Renders a synthetic RenderWorld + the REAL imgui_layer panels
    # (fixed UI state) into the offscreen image via the Render(world,opts,overlay)
    # seam, asserts ImDrawData.CmdListsCount>0, non_bg>0, two composited renders
    # byte-identical (D1), dumps /tmp/m8_5_viewer_frame.ppm. Links nuka_viewer
    # (imgui_layer + camera_controller) + nuka_imgui + nuka_render. SKIPs with no
    # Vulkan device. No window needed (imgui_impl_vulkan draws to an offscreen pass).
    add_executable(nuka_viewer_frame_smoke_test
        scenario/viewer_frame_smoke.cpp
    )
    add_dependencies(nuka_viewer_frame_smoke_test nuka_render_shaders)
    target_compile_definitions(nuka_viewer_frame_smoke_test
        PRIVATE NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_viewer_frame_smoke_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_viewer_frame_smoke_test
        PRIVATE
            nuka_viewer
            nuka_imgui
            nuka_present
            nuka_render
            nuka_scene
            imgui
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_viewer_frame_smoke_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # M11 INT-9 -- interop_fallback_smoke: the BUILD-LINK + graceful-degrade
    # acceptance gate for the CUDA<->Vulkan zero-copy interop publisher. Links the
    # exportable-SSBO (nuka_render), the publisher + weak fallback (nuka_app), and
    # the STRONG CUDA scatter (nuka_phi2_interop). On this lavapipe box the SSBO
    # export is unsupported -> the publisher degrades to HostDownloadPublisher; the
    # test asserts that fallback path. (Real zero-copy is owner-verified on display.)
    add_executable(nuka_interop_fallback_smoke_test
        scenario/interop_fallback_smoke.cpp
    )
    add_dependencies(nuka_interop_fallback_smoke_test nuka_render_shaders)
    target_compile_definitions(nuka_interop_fallback_smoke_test
        PRIVATE NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_interop_fallback_smoke_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_interop_fallback_smoke_test
        PRIVATE
            nuka_app
            nuka_phi2_interop
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_scene
            Vulkan::Vulkan
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_interop_fallback_smoke_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")
endif()
