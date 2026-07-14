# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(GoogleTest)

# ---------------------------------------------------------------------------
# M9 T14 -- test-suite reorganization
# ---------------------------------------------------------------------------
# 95 per-feature test exes -> 5 CATEGORY exes (oracle / scenario / perf / import
# / render) with fast/full/perf LABELS.
#
# LABEL SCHEME (single-token, ctest -L is a REGEX so substrings match):
#   full      -> EVERY test (the comprehensive run). Applied to all.
#   fast_full -> the quick unit/oracle subset. `-L fast` matches it; `-L full`
#                matches it too (contains "full").
#   perf_full -> the throughput/timing gates. `-L perf` matches it; `-L full` too.
# Therefore `-L full` == the whole suite, `-L full` + `-L perf` cover everything,
# and `-L fast` is the quick subset. (A multi-value LABELS list cannot survive
# gtest_discover_tests serialization in CMake 3.30 -- add_command flattens it --
# so the tier is encoded in ONE token instead. See M9 T14 report.)
#
# SHARED VALIDATED-NOT-WIRED SOURCES
# ----------------------------------
# L1-c: the nuka_test_validated_sources static support lib was DELETED. It existed
# to fold src/scene/cook/union_nk_model.cpp (the UnionCsr BuildNkUnionModel bridge)
# + contact_stream_driver.cpp into ONE lib for the union grasp/perf gates. The
# union path + union_nk_model.cpp are gone (cup-grasping moved to RL); the foot
# collision/solver gates that still need contact_stream_driver.cpp compile it
# directly as a per-exe extra source (they always did), so no shared lib remains.

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

        # --- Multi-env C ABI surface (p01-F T7) ---------------------------
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

    # --- v0.5 p02-B/C: differentiable rollout tape C ABI (runtime coverage) ---
    add_executable(nuka_diffsim_tape_cabi_test
        c_abi/test_diffsim_tape.cpp
    )

    target_compile_definitions(nuka_diffsim_tape_cabi_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_diffsim_tape_cabi_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(nuka_diffsim_tape_cabi_test
        PRIVATE
            nuka
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_tape_cabi_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.5 p04 N1: C-ABI sensor-noise surface (NONE no-op + apply + D1) ----
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

    # --- v0.5 p04 N2: per-episode domain randomization (sampling + apply + D1) -
    # Reads c_abi/internal.hpp (WorldRecord) for mass/gravity readback + calls
    # SampleEpisodeRandomization directly (nuka_sensor_noise), so it needs the
    # src/ include + the runtime libs (same pattern as the mass-setter FD test).
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
                nuka_articulation_gpu
                nuka_core
                nuka_phi
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_n2_domain_randomization_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")
    endif()

    # --- v0.5 p04-A0: runtime link-mass setter + C-ABI mass-gradient FD proof --
    # Operations under test are pure C-ABI (set_link_mass / step_with_tape /
    # tape_backward); the internal libs are linked only for setup + reading the
    # tape-evolved device state (the same readback pattern every diffsim test
    # uses). Includes src/ for the c_abi/internal.hpp WorldRecord bridge.
    if(CMAKE_CUDA_COMPILER)
        add_executable(nuka_diffsim_mass_setter_fd_test
            diffsim/test_mass_setter_fd.cpp
        )

        target_compile_definitions(nuka_diffsim_mass_setter_fd_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_diffsim_mass_setter_fd_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
                ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(nuka_diffsim_mass_setter_fd_test
            PRIVATE
                nuka
                nuka_diffsim
                nuka_runtime
                nuka_articulation_gpu
                nuka_core
                nuka_phi
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_diffsim_mass_setter_fd_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "full")
    endif()
endif()

# =====================================================================
# CATEGORY EXE 4/5: nuka_import_test -- cooker / scene / import (HOST).
# Consolidates the pure-host scene-unit + importer + V-HACD + sparse-SDF cooker
# gtest suites into ONE exe (all link nuka_scene/nuka_import/nuka_cooker; the
# visual-mesh cook consumer adds nuka_render). No ODR collisions (distinct suite
# names + no cross-file global-helper clashes -- verified M9 T14).
#
# NOTE: the 3 .nks/.nka roundtrip "whole-exe" gates (SceneRoundtrip,
# NksGraspSceneRoundtrip, VisualMeshCook) stay as their OWN add_test(NAME) exes
# below -- they are registered as single whole-binary ctest entries (NOT
# gtest_discover'd); folding them into this gtest exe would re-expose their 12
# inner TEST()s as 12 ctest entries (3 -> 12), CHANGING the count. Kept separate
# to preserve the exact 663 ctest count. The CUDA-region cooker/scene/c_abi
# members (scene_compose_h1, xpbd_cooker, fluid_cooker, the c_abi gtests) also
# stay separate (they live behind if(CMAKE_CUDA_COMPILER); not folded here to
# avoid a cross-region restructure -- still labeled).
# =====================================================================
# M9 T11: test_scene_pipeline.cpp (the deleted ScenePipeline orphan) dropped; the
# scene unit files only need compose/scene_ir/cooker (nuka_scene). vhacd_test_meshes.hpp
# needs the source-root include dir (added below for the V-HACD sources).

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
        nuka_articulation_gpu
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
        nuka_articulation_gpu
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
        nuka_articulation_gpu
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
        nuka_articulation_gpu
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
        nuka_articulation_gpu
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
        nuka_articulation_gpu
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_editor_script_lifecycle_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")
endif()

# --- M7 T3: the committed union grasp scene roundtrip + content gate. Loads
# examples/scenes/h1_cup_table.nks (H1 via imports + cup + table + grasp config
# + initial_state + settle) and asserts byte-identical re-save + section
# exposure. Same lightweight deps as SceneRoundtrip (no factory).
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

# --- M8.5 T5: the VISUAL-MESH COOK gate. Cooks go2 (+ h1_with_hand if present)
# import -> .nks/.nka Save -> Load -> BuildRenderWorld and asserts the robot
# renders as REAL .nka MESH triangles (MeshSource::NkaMesh), not placeholder
# boxes. Scene-side only (no Vulkan device needed); links nuka_render for
# BuildRenderWorld (the consumer) which is host-only.
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

# Fluid marching-cubes isosurface mesher (runtime::fluid::MarchFluidSurface).
# HOST-ONLY (no Vulkan device / no CUDA): runs the mesher as a pure function of a
# synthetic settled-pool / sphere particle set and asserts determinism (D1),
# watertight-ish closure, envelope tracking, and unit gradient normals. Links
# nuka_render (where the mesher lives) like the visual-mesh cook gate above.
add_executable(nuka_fluid_surface_mesher_test
    runtime/test_fluid_surface_mesher.cpp
)
target_include_directories(nuka_fluid_surface_mesher_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(nuka_fluid_surface_mesher_test
    PRIVATE
        nuka_render
        GTest::gtest_main
)
gtest_discover_tests(nuka_fluid_surface_mesher_test
    PROPERTIES LABELS "fast_full")

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

# M9 T14: nuka_scene_test, nuka_vhacd_test, nuka_sparse_sdf_test MERGED into
# nuka_import_test above (the host import/scene/cooker category exe).

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

# --- Solver tests ----------------------------------------------------------
# M9 T11-core-b2: test_row_scheduler_bridge.cpp (legacy BuildRowColorPartitions +
# DiffTestBridge/row_solver.cu replay) and test_row_scheduler_coloring_equivalence.cpp
# (the legacy O(rows^2)->O(rows*K) coloring-rewrite gate) were DELETED with the
# legacy solver trio. The coloring algorithm migrated to nk BuildColors (inside
# nk::SolveSchedule::Partition) and is exercised end-to-end by the byte-exact nk
# union gates + the nk_solve_harness foot_*/link_rigid device-solve tests.
add_executable(nuka_solver_test
    # L1-c: test_unified_solve.cpp was DELETED. It was WHOLLY a UnionCsr box-on-
    # plane oracle (MakeNkBoxModel built an nk::UnionSlot kBodyBoxPlane model with
    # contact_family=UnionCsr + union_solref/solimp -- all deleted with the
    # UnionCsr path). The general (PairDriven) box-rests-on-static-ground +
    # solref/solimp + friction-pyramid + D1 coverage lives in
    # scenario/pairdriven_solve.cpp (nuka_scenario_test). See the L1-c report:
    # this file was on the "keep" list but had no general-path content to preserve.
    # v0.8 C5b-core: ArticulationChainJ reaction arm wired into UnifiedSolve
    # (closed-form chain oracle + same-articulation coloring + link<->rigid two-
    # way + D1). Hand-built ArticulationReactionState data (no device CRBA).
    solver/test_link_rigid_twoway.cpp
    # v0.8 C5b-core: compliant-PGS regularizer R-SWEEP guard (210c11f). Drives a
    # coupled full-rank SPD A = J M^-1 J^T (dof_stride==N) compliant system through
    # UnifiedSolve and matches the converged lambda to a double-precision dense
    # (A+diag(R))lambda=b solve across R/A from ~1e-3 to ~1e+1, + a large-R
    # discriminator vs the pre-fix A*lambda=b fixed point + D1. MJX-FREE.
    solver/test_compliant_pgs_regularizer.cpp
)

target_link_libraries(nuka_solver_test
    PRIVATE
        nuka_solver
        nuka_articulation
        GTest::gtest_main
)
if(CMAKE_CUDA_COMPILER)
    # M4 re-point: the nk-driven UnifiedSolve oracle scenarios
    # (test_unified_solve.cpp NkWorld* cases).
    target_link_libraries(nuka_solver_test PRIVATE nuka_nk nuka_phi2)
endif()

gtest_discover_tests(nuka_solver_test
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
        $<$<TARGET_EXISTS:nuka_articulation_gpu>:nuka_articulation_gpu>
        $<$<TARGET_EXISTS:nuka_phi>:nuka_phi>
        # M3b: the nk::World drive entries (CookToModel -> nk core -> phi v2).
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

    # --- V3 FD adjoint validation: all four base row classes (v0.5 p01) ------
    # maximal_drive (dense), maximal_joint (dense), maximal_contact +
    # featherstone_contact (stop_grad_on_event). Each validates the GENERATED
    # primal/adjoint pair against a central-difference numerical Jacobian.
    add_executable(nuka_adjoint_fd_test
        codegen/test_adjoint_fd_maximal_drive.cpp
        codegen/test_adjoint_fd_maximal_joint.cpp
        codegen/test_adjoint_fd_maximal_contact.cpp
        codegen/test_adjoint_fd_featherstone_contact.cpp
        codegen/test_adjoint_fd_xpbd_distance.cpp
        # v0.7 p09-B: XPBD bend (id 7) + volume (id 8) adjoint FD validation.
        codegen/test_adjoint_fd_xpbd_bend.cpp
        codegen/test_adjoint_fd_xpbd_volume.cpp
        # v0.7 p09-C: XPBD shape-match (id 9) goal-projection adjoint FD validation.
        codegen/test_adjoint_fd_xpbd_shape_match.cpp
        # v0.7 p11 (K3): particle-particle contact (id 10) contact-multiplier
        # adjoint FD validation (active penetrating contacts, unilateral box).
        codegen/test_adjoint_fd_particle_particle_contact.cpp
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

    # M9 T11: nuka_h1_union_parity_test DELETED (deletion-order step 2). Its nk-vs-
    # legacy coresident union-world step-0/window parity dies with the legacy class
    # (golden/nk path is authority; the M7 vel-window benign-vs-masking verdict was
    # rendered in T10 PART A while the oracle was alive). Its UNIQUE non-legacy
    # coverage -- the union StepPlanned==Step plan-replay D1 -- was FOLDED into
    # tests/scenario/h1_grasp_lift.cpp (T10 PART B).

    # --- M8 T4: render_frame_loop_smoke — the frame-loop SPINE smoke
    # (PosePublisher + systems + Simulation, nuka_app). The G5 render-OFF
    # byte-identity assertion runs in the DEFAULT build (NOT behind the Vulkan
    # flag); the G1 publish-contract + non-bg-pixels assertions are #ifdef'd on
    # NK_BUILD_VULKAN_VALIDATION inside the source (they need a real raster
    # device). The executable links nuka_app (the spine under test) + the .nks
    # cook path (CookToModel in nuka_scene_cook, nks::Load in nuka_import) +
    # nuka_render (BuildRenderWorld / RenderWorld). Asset-gated SKIP.
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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

    # =====================================================================
    # CATEGORY EXE 2/5: nuka_scenario_test -- physics integration SCENARIOS.
    # Consolidates the nk-physics integration cluster: cook-settle determinism, nk
    # particle/soft-fluid equivalence, the multi-dog + PairDriven general-path
    # gates, and the general-path V-PROOFs. L1-c DELETED the UnionCsr-path members
    # (coupled_grasp_soft / test_coupling_framework / test_pbd_costep_unified --
    # the hand-assembled UnionSlot co-step gates -- plus the .nks union grasp gates
    # h1_grasp_lift / union_cook_golden); cup-grasping moved to RL. All sources are
    # anon-namespaced with distinct TEST suites (no ODR / name collision -- M9 T14).
    # =====================================================================
    add_executable(nuka_scenario_test
        scenario/cook_settle_determinism.cpp
        scenario/nk_particle_equivalence.cpp
        scenario/nk_soft_fluid_coresidence.cpp
        scenario/nk_mpm_xpbd_coresidence.cpp
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
        # Body/artic <-> soft/fluid particle coupling: a static box overlapping ONE
        # particle emits a kNkSideParticle row through the SAME EmitPairDrivenRows ->
        # ComputeRowMeff -> SolveRowsBlockIsland the body rows use (particle = sphere
        # on the ONE path); asserts the particle-side row solves (lambda>0 + braked
        # approach velocity) + two-run byte-identity of the coupling stream.
        scenario/body_particle_coupling.cpp
        # Body<->particle coupling against a HEIGHTFIELD: a particle dropped onto a
        # cooked terrain grid emits a kNkSideParticle face contact (the SAME deep-sink
        # SphereTriangleFace the rigid sphere-vs-heightfield handler uses) and rests at
        # the surface instead of tunnelling through -- closes the heightfield arm that
        # was a silent no-op stub (0 rows, 0 diagnostic); + two-run byte-identity.
        scenario/body_particle_heightfield.cpp
        # Body<->particle FRICTION asymmetry: one SoftFluid Model presses a SOFT and a
        # FLUID particle onto static boxes with the same (-z,+x) approach. Same pyramid,
        # same row, same kernel -- only the per-system mu differs: the soft slice (finite
        # mu) grips (friction lambda>0 + tangential bounded), the fluid slice (mu~0)
        # slides (friction lambda~0 + tangential free) while its normal row still brakes.
        scenario/body_particle_friction.cpp
        # Body<->particle contact-BUDGET disjointness: a cooked scene with a static
        # ground box, a dynamic box on it (a body<->body contact), AND >= 64 particles
        # whose reserve dwarfs the rigid candidate budget. The cook grows
        # max_contacts_per_env additively so the rigid [0, rigid_cap) and body<->particle
        # [rigid_cap, total) slot sub-ranges are disjoint -- the rigid contact SURVIVES
        # the particle writes (pre-reserve it was clobbered). + a particle over its
        # body-contact slots flags kEnvStatusPairOverflow LOUD (no silent drop).
        scenario/body_particle_budget_disjoint.cpp
        # SoftFluid finalize composition: a soft particle braked by a static-floor
        # coupling row must STAY braked through a FULL World::Step. The SoftFluid
        # finalize reproduces the projection velocity AND composes the contact
        # correction (v_now - v_pre) so the impulse the solve wrote into particle_vel
        # survives instead of being overwritten by a plain v = (pos_proj - prev)/dt.
        scenario/soft_fluid_coupling_finalize.cpp
        # Cloth <-> presser two-way coupling: a 576-particle XPBD cloth drapes over a
        # low static ridge, a rigid presser dips it (cloth Δz down + spring-back on
        # lift) and a dynamic presser is arrested by the cloth (body-side reaction
        # lambda>0 vs a no-cloth control) -- all via the body<->particle row solver.
        scenario/cloth_presser_two_way.cpp
        # 3D soft tet body <-> rigid two-way coupling on the same body<->particle row
        # a cloth particle uses (deform + rest-on-floor reaction + volume recovery).
        scenario/soft_tet_presser_two_way.cpp
        # PBF fluid <-> rigid foot two-way coupling on the same body<->particle row: a
        # foot plunges a shallow pool (splash + displacement) and is resisted by the
        # fluid (body-side reaction lambda>0 + decel vs a no-fluid control); the
        # resistance emerges from the contact rows + PBF incompressibility, no buoyancy.
        scenario/foot_splash_two_way.cpp
        # Articulated LINK <-> particle two-way coupling on the same body<->particle
        # row a foot uses on the ground: a minimal one-revolute pendulum arm presses a
        # cloth patch (coupling row whose BODY side maps to an articulation link, and
        # the reaction reaches the single revolute qdot + tip pose vs a no-patch control).
        scenario/articulated_link_particle_coupling.cpp
        # Go2 articulation + cloth (XPBD) + fluid (PBF) co-resident in ONE world, feet
        # coupling to both media's slices through the SAME body<->particle row solver.
        scenario/robot_cloth_fluid_coresident.cpp
        # Phase 1B dog cases: artic x rigid (free box on a dog's back = mixed island)
        # and artic x artic (two dogs collide via the GENERAL path; the ONE path).
        scenario/pairdriven_dog_solve.cpp
        # General contact pipeline Phase 2A: the GENERAL HEIGHTFIELD narrowphase
        # gate. A box on a FLAT grid rests; ★ a box on a real PyramidStairs STEP
        # EDGE settles finite + no 穿模 (the exact gate2 explosion fix); per-cell
        # TRIANGLE_PRISM contacts via the EXISTING cvx GJK/EPA; two-run identity.
        scenario/pairdriven_heightfield.cpp
        # V-PROOF (one-general-solver-spec §4/§L0): physical-validity harness for the
        # GENERAL contact path (PairDriven + CookHeightfieldGrid) on go2 feet-on-
        # ground / terrain. (L1-b deleted the FUSED reference half; these now assert
        # the general path alone is finite + feet-load + no-sink.)
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
        # Per-medium solver selector + the media-list (kind x method) validator (host
        # cook): a bulk-soft body cooks XPBD by default (byte-identical) and MPM when
        # selected; cloth -> MLS-MPM and MPM+XPBD/PBF mixes throw LOUDLY.
        scenario/mpm_sim_method_selector.cpp
        # MediaRecord -> cook-input equivalence (host cook): the cloth/fluid builders
        # over a translated MediaRecord match the legacy flat-descriptor builders
        # field-by-field (the byte-identity proof for the cook-orchestrator re-point).
        scenario/media_record_cook_equivalence.cpp
        # MLS-MPM elastic jelly-ball gate: a fixed-corotated ball drops onto the static
        # floor plane (grid BC), squashes <= ~25%, recovers, no collapse/spike/escape,
        # >= 5 s, finite, two-run byte-identical (the deterministic gather).
        scenario/mpm_jelly_ball.cpp
        # Two-way rigid <-> MPM coupling gate: a free rigid box rests on an MPM bed;
        # the per-substep grid->body reaction balances gravity (m*g*dt), the free-fall
        # BITE drops the box, two runs byte-identical (the deterministic per-body gather).
        scenario/rigid_on_mpm_rest.cpp
        # MLS-MPM weakly-compressible FLUID constitutive gate (model_kind == 3): a
        # fluid column settles in the domain box -> hydrostatic pressure (1-J) rises
        # with depth + flat free surface (J~1); grid mass/momentum byte-identical
        # run-to-run + grid mass == particle mass; a heavy body into the pool
        # decelerates with a non-zero reaction sink (two-way through the same grid).
        scenario/mpm_fluid_rest.cpp
        # Articulated-LINK <-> MPM coupling gate: a fixed-base prismatic-Z link rests
        # on an MPM bed; the link's grid reaction seeds qdot_flat via M^-1 J^T (support,
        # balance, BITE, two-run byte-identical, one-way bit cleared, 2-link reduction).
        scenario/artic_link_on_mpm_rest.cpp
        # MLS-MPM granular Drucker-Prager constitutive gate (model_kind == 4): a
        # released sand column piles at a dp_friction-tracking repose angle (fluid
        # control flattens, elastic control stands); a scripted capsule press leaves
        # a >= 3 s persistent crater; particle state byte-identical run-to-run.
        scenario/mpm_granular.cpp
        # Dynamic island solve == the static one-island-per-env schedule, bit-for-bit,
        # in the contact->airborne (pos_iters>0) and multi-env mixed-contact regimes.
        scenario/island_byte_identity.cpp
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_scenario_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- CouplingProvider op-list identity (the body↔particle row provider) -----
    # Pure-host structural proof: builds a cooked-Model matrix through
    # Pipeline::Build and asserts the emitted OpCall sequence + a FULL byte memcmp
    # of each moved Params POD is identical build-to-build and at the canonical
    # positions. No CUDA at runtime (Build with device==nullptr); links nuka_nk
    # (Pipeline/Model) + nuka_phi2 (the op-schema/backend types nuka_nk needs).
    add_executable(nuka_coupling_provider_test
        scenario/coupling_provider_op_list.cpp
    )
    add_dependencies(nuka_coupling_provider_test nuka_codegen)
    target_include_directories(nuka_coupling_provider_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_coupling_provider_test
        PRIVATE
            nuka_nk
            nuka_phi2
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_coupling_provider_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # M9 T14: nuka_cook_settle_determinism_test MERGED into nuka_scenario_test
    # (scenario/cook_settle_determinism.cpp; the scenario category exe above).

    # --- M7 T5b: the H1 x cup x kitchen-counter compose gate, UPGRADED to the
    # FULL .nks pipeline (Compose -> FindRestPlacement seat -> Save(.nks+.nka) ->
    # re-Save byte-identity -> Load -> CookToModel -> nk::World -> Step). GTEST_SKIPs
    # without .nuka-assets / a CUDA backend. Moved out of nuka_scene_test (CUDA-free)
    # into the CUDA region for the cook+World step. Flags the kitchen-scale cook gap
    # (single-articulation cook can't represent the whole-body H1; the union
    # h1_cup_table.nks path owns the real grasp world).
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

    # M9 T14: nuka_nk_particle_equivalence_test + nuka_nk_soft_fluid_coresidence_test
    # MERGED into nuka_scenario_test (scenario/nk_particle_equivalence.cpp +
    # scenario/nk_soft_fluid_coresidence.cpp; the scenario category exe above).

    add_executable(nuka_cuda_runtime_test
        # M9 T11-core-b2: test_cuda_solver.cpp (legacy CudaRowSolver device-kernel
        # tests) deleted with row_solver.cu; the nk device solve is covered by
        # test_unified_solve.cpp NkWorld* + the nk_solve_harness foot_*/link_rigid
        # device-solve gates.
        # v0.7 p09-A: XPBD distance analytic oscillation oracle + D1 two-run.
        runtime/test_xpbd_distance_oracle.cpp
        # v0.7 p09-B: XPBD bend (cloth) + volume (tet) sim invariants + grad-C FD.
        runtime/test_xpbd_bend_cloth.cpp
        runtime/test_xpbd_volume_tet.cpp
        # v0.7 p09-C: XPBD shape-match (cluster) sim + polar dR/dA host-FD.
        runtime/test_xpbd_shape_match.cpp
        # v0.7 p11 (K3): cross-system particle-particle contact co-step oracles
        # (separation, mass-weighted momentum, soft<->fluid, segment-reduction, D1,
        # inert byte-identity).
        runtime/test_particle_particle_contact.cpp
    )

    target_link_libraries(nuka_cuda_runtime_test
        PRIVATE
            nuka_runtime_gpu
            nuka_runtime
            nuka_articulation_gpu
            # M9 T11: the re-pointed XPBD shape-match / bend-cloth / distance /
            # volume + PBF + cross-system contact tests cook a scene -> nk::World ->
            # Step on the nk particle ops (the legacy soft/fluid steppers were
            # DELETED in Phase 2b; the cook is the scene-cook layer).
            nuka_nk
            nuka_scene_cook
            nuka_phi2
            nuka_phi
            nuka_core
            GTest::gtest_main
    )
    add_dependencies(nuka_cuda_runtime_test nuka_codegen)
    target_include_directories(nuka_cuda_runtime_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)

    gtest_discover_tests(nuka_cuda_runtime_test
        PROPERTIES LABELS "full")

    # --- .nks media + rigid-primitive roundtrip. Host/CPU only (no device step):
    # Saves/Loads a SceneIR with cloth + soft-tet media + an authored ground box and
    # asserts the records round-trip field-equal + the cloth cooks to the frozen
    # lattice FNV. Links nuka_scene_cook for BuildClothXpbdInput (CUDA-language lib,
    # hence inside the CMAKE_CUDA_COMPILER guard; the test runs no device code).
    add_executable(nuka_nks_media_roundtrip_test
        scenario/nks_media_roundtrip.cpp
    )
    target_include_directories(nuka_nks_media_roundtrip_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_nks_media_roundtrip_test
        PRIVATE
            nuka_scene_cook
            nuka_cooker
            nuka_scene
            nuka_import
            nuka_runtime_gpu
            nuka_nk
            GTest::gtest_main
    )
    add_dependencies(nuka_nks_media_roundtrip_test nuka_codegen)
    add_test(NAME NksMediaRoundtrip
             COMMAND nuka_nks_media_roundtrip_test
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(NksMediaRoundtrip PROPERTIES LABELS "fast_full")

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

    # --- v0.7 p09-D: XPBD soft-body COOKER tests (exit-crit 3) -------------
    # PURE host/CPU: cooks programmatic XpbdSoftBodySpecs and asserts the
    # returned XpbdConstraintSet (counts + the two param->engine mappings +
    # two-cook determinism). No device upload / step, so it needs no GPU -- but
    # it links nuka_runtime_gpu because the cooker + topology builders live
    # there (the lib is CUDA-language, hence inside the CMAKE_CUDA_COMPILER
    # guard). nuka:soft:* schema + mapping documented in
    # docs/architecture/xpbd-soft-schema.md.
    add_executable(nuka_xpbd_cooker_test
        import/test_xpbd_cooker.cpp
    )

    target_include_directories(nuka_xpbd_cooker_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_xpbd_cooker_test
        PRIVATE
            nuka_runtime_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_xpbd_cooker_test
        PROPERTIES LABELS "fast_full")

    # --- v0.7 p10-B: PBF fluid COOKER (programmatic FluidBoxSpec -> particles) --
    # PURE host/CPU: cooks a box volume + asserts the hand-computed lattice count,
    # the uniform mass = rest_density*spacing^3, cell-centered placement, two-cook
    # byte-identical determinism, degenerate-spec empties, and the optional
    # inscribed-sphere carve. Links nuka_runtime_gpu because the fluid cooker +
    # PbfParticleSet live there (the lib is CUDA-language, hence inside the
    # CMAKE_CUDA_COMPILER guard) -- but the test itself runs no device code.
    add_executable(nuka_fluid_cooker_test
        import/test_fluid_cooker.cpp
    )

    target_include_directories(nuka_fluid_cooker_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_fluid_cooker_test
        PRIVATE
            nuka_runtime_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_fluid_cooker_test
        PROPERTIES LABELS "fast_full")

    # --- v0.7 p10-A: PBF fluid core (density iteration + volume + D1) -------
    # Analytic/physical invariants for the forward density-projection fluid
    # (Macklin & Mueller 2013): rest-density block, over-compressed relaxation,
    # single column -> stable pool, density-drift volume conservation, and D1
    # two-run byte-exact. NOT a Flex golden (Flex oracle deferred to p15). rho0 is
    # calibrated numerically from the engine's own host Poly6 kernel (the CUDA-free
    # ComputePbfDensities in fluid_cooker_types.hpp); the forward runs on nk::World.
    # p10-B adds the polish-layer gates: XSPH viscosity (variance reduction +
    # off-gate skip + D1), surface-tension cohesion (blob contraction + off-gate +
    # D1 + stability), and the PBF V2 particle-count invariant (count conserved
    # with viscosity + cohesion ON; the central V2 sampler is rigid-only, so this
    # is the focused fluid count-conservation test -- see the file header).
    add_executable(nuka_pbf_test
        runtime/test_pbf_density_iteration.cpp
        runtime/test_pbf_volume_conservation.cpp
        runtime/test_pbf_viscosity.cpp
        runtime/test_pbf_surface_tension.cpp
        runtime/test_pbf_v2_invariant.cpp
        # CUDA-graph capture parity: the PBF grid op now captures (StepPlanned Ok)
        # and graph replay == eager Step byte-for-byte (pre-allocated sort scratch).
        runtime/test_pbf_graph_capture.cpp
    )

    target_link_libraries(nuka_pbf_test
        PRIVATE
            nuka_runtime_gpu
            # M9 T11 Phase 2b: ALL PBF tests cook a fluid -> nk::World -> Step on the
            # nk particle ops; rho0 calibration uses the CUDA-free host
            # ComputePbfDensities. The legacy soft/fluid steppers were DELETED.
            nuka_nk
            nuka_phi2
            nuka_phi
            nuka_core
            GTest::gtest_main
    )
    add_dependencies(nuka_pbf_test nuka_codegen)
    target_include_directories(nuka_pbf_test
        PRIVATE ${CMAKE_SOURCE_DIR}/src)

    gtest_discover_tests(nuka_pbf_test
        PROPERTIES LABELS "full")

    # --- v0.7 p08-A: GPU upload of cooked sparse SDFs ----------------------
    # COL-5 (M11): nuka_sdf_device_test (runtime/test_sdf_device_upload.cu) DELETED
    # with SdfDeviceWorld. The cooked-table->device round-trip + cell-count coverage
    # it provided is folded into the SDF precision oracle
    # (SdfPrecisionOracle.CookedTableDeviceRoundTripAndCellCount in
    # collision/test_gjk_epa_convex.cpp); device SDF sampling is covered by the rest
    # of that oracle (NarrowphaseSdf op over uploaded cell tables).

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

    # --- v0.7 p04: LBVH vs SAP differential gate + D1 byte-exact -----------
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

    # --- v0.7 p05: particle uniform grid + cross-system broadphase ---------
    # Correctness vs CPU brute-force + D1 two-run byte-exact for both the
    # particle neighbor grid and the particle->rigid cross-system query (which
    # traverses the LBVH retained by BuildLbvhForQuery).
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

    # M9 T11: nuka_sdf_tier_wired_test DELETED -- it tested the LEGACY SDF tier wired
    # through ResolveNarrowphase + find_sdf_contact_newton (narrowphase_dispatch.hpp +
    # sdf_contact.*), both deleted in T11 (the new core uses contacts_union.cu). SDF
    # cook determinism survives separately in tests/import/test_sparse_sdf_determinism.cpp.

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

    # --- v0.7 p13: the SENSOR renderer (real per-primitive leaf intersections
    # triangle/sphere/sparse-SDF + Lambert+GGX shade + hard shadows + full AOV
    # framebuffer). Matches a CPU brute-force ray-vs-all-prims oracle that REUSES
    # the same __host__ __device__ intersection + shade + closest-hit functions
    # the kernel uses (triangle hit + bary EXACT, sphere depth + normal EXACT,
    # SDF hit within tolerance, Lambert/GGX == analytic, hard-shadow occlusion);
    # NaN-guard for a surface-origin axis-parallel ray; D1 two-run byte-exact
    # across ALL AOVs. Built -ffp-contract=off to pair with the kernel's
    # --fmad=false so the shared fp64-internal intersection math is host==device
    # bit-for-bit. Links nuka_cooker for the in-test cooked sparse SDF (p07).
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

    # --- v0.7 G1-C: two-level (TLAS/BLAS) ray-tracer BENCHMARK harness -------
    # MEASURES flat p13 RenderScene (rebuild-all every frame over N*~300 baked
    # world triangles) vs the G1-A two-level RenderFrame (per-mesh BLAS built
    # ONCE + cheap per-frame N-instance TLAS rebuild). Sweeps N=1,16,64,256,1024
    # at 512x512, animates instances each frame, times the WHOLE host per-frame
    # call (wall-clock), reports ms/frame + full-per-frame-wall-time MRays/s +
    # speedup, and prints the explicit G1-B go/no-go verdict. Asserts ONLY the
    # relative structural invariant (two-level < flat at N>=64); absolute numbers
    # are hardware-dependent (nvidia-smi NVML-broken) and never asserted.
    # M9 T14: nuka_rt_two_level_bench MERGED into nuka_perf_test (the perf
    # category exe; rt/test_two_level_bench.cpp + its -ffp-contract=off there).

    # --- p08-C Channel 2/3: envelope-theorem contact-geometry adjoint -------
    # Cooks box+sphere SDFs, runs the SHIPPING find_sdf_contact_newton, and
    # FD-validates the envelope depth + normal gradients by RE-RUNNING the descent
    # under perturbation (cell value / pose). Channel 3: documents the contact
    # point's valley-degeneracy (no point gradient produced/validated).
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

    # --- Batched N-articulation replication (p01-F T1) ---------------------
    add_executable(nuka_batched_articulation_replication_test
        runtime/test_batched_articulation_replication.cpp
    )

    target_compile_definitions(nuka_batched_articulation_replication_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_batched_articulation_replication_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_articulation_replication_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    add_executable(nuka_contact_chain_jacobian_test
        runtime/test_contact_chain_jacobian.cpp
    )

    target_compile_definitions(nuka_contact_chain_jacobian_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_contact_chain_jacobian_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_contact_chain_jacobian_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    add_executable(nuka_articulation_contacts_test
        runtime/test_articulation_contacts.cpp
    )

    target_compile_definitions(nuka_articulation_contacts_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_articulation_contacts_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_articulation_contacts_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Joint-space inertia M (CRBA) + contact effective mass (p01-F T4) ---
    add_executable(nuka_articulation_inertia_m_test
        runtime/test_articulation_inertia_m.cpp
    )

    target_compile_definitions(nuka_articulation_inertia_m_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_articulation_inertia_m_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_articulation_inertia_m_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.8 SG G0: DOF honesty above 18 (whole-body H1 precondition) -------
    # Synthetic >18-DOF chains: M^-1 factorization covers the FULL dof block,
    # m_eff uses the full chain-J, implicit damping reaches DOF>18, and beyond
    # kMaxArticulationDof the launchers fail LOUDLY (never a silent clamp).
    add_executable(nuka_dof_above18_honesty_test
        runtime/test_dof_above18_honesty.cpp
    )

    target_compile_definitions(nuka_dof_above18_honesty_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_dof_above18_honesty_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_dof_above18_honesty_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.8 C4c: reaction providers (rigid invM / chain-J / particle invM) -
    # Validates the NEW, VALIDATED-NOT-WIRED src/constraint/reaction_provider.hpp
    # §0.5 interface: a 1-DOF pendulum chain-J vs a hand-computed J M^-1 J^T (with
    # an OFF-origin contact point -- the C4c arbitrary-point generalization),
    # rigid invM vs the analytic diagonal path, particle scalar invM, StaticNull
    # zero/no-op, and 2-run byte-identity (D1). Round-trips the proven device CRBA
    # M^-1 + chain-J builders the articulation provider consumes, so it links the
    # articulation GPU libs + nuka_constraint (the header-only providers).
    add_executable(nuka_reaction_providers_test
        constraint/test_reaction_providers.cpp
    )

    target_compile_definitions(nuka_reaction_providers_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_reaction_providers_test
        PRIVATE
            nuka_constraint
            nuka_runtime
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_reaction_providers_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- Articulated contact rows + block-PGS solve (p01-F T5) --------------
    add_executable(nuka_articulation_contact_rows_test
        runtime/test_articulation_contact_rows.cpp
    )

    target_compile_definitions(nuka_articulation_contact_rows_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_articulation_contact_rows_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_articulation_contact_rows_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.5 p02-A: differentiable single-step ABA reverse (FD ladder) -----
    add_executable(nuka_diffsim_aba_reverse_test
        diffsim/test_aba_reverse_fd.cpp
    )

    target_compile_definitions(nuka_diffsim_aba_reverse_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_aba_reverse_test
        PRIVATE
            nuka_diffsim
            nuka_runtime
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            # M3b: the nk-arena seam smoke (diffsim kernels on nk::World state).
            nuka_nk
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_aba_reverse_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.5 p02-B/C: multi-step diff-sim (tape/checkpoint/recompute/backward) -
    add_executable(nuka_diffsim_multistep_backward_test
        diffsim/test_multistep_backward.cpp
    )

    target_compile_definitions(nuka_diffsim_multistep_backward_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_multistep_backward_test
        PRIVATE
            nuka_diffsim
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_multistep_backward_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.5 p03 R1: self-written deterministic CG sparse solver (IFT backend) -
    # CG-vs-Eigen-LDLT dense cross-check + D1 bit-exact gate. Eigen is linked here
    # ONLY (nuka_eigen_testonly INTERFACE include dir), never into any production
    # target.
    add_executable(nuka_diffsim_cg_solver_test
        diffsim/test_cg_vs_dense.cpp
    )

    target_compile_definitions(nuka_diffsim_cg_solver_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_cg_solver_test
        PRIVATE
            nuka_diffsim
            nuka_core
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_cg_solver_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # --- v0.7 p01: hardened CG general-path solver tests --------------------
    # The v0.5 CG solver hardened + generalized for worse-conditioned systems:
    #   * test_cg_vs_dense          : Eigen-LDLT cross-check on kappa up to ~1e6 +
    #                                 D1 bit-exact (worse-conditioned coverage).
    #   * test_cg_convergence_rate  : CG A-norm error vs the kappa-theory bound in
    #                                 the informative regime + residual-history diag.
    #   * test_block_jacobi_tuning  : Jacobi-vs-BlockJacobi preconditioner quality
    #                                 sweep + the host diagnostics summarizer.
    # Eigen is linked test-only (the approved dense oracle), never a production path.
    foreach(cg_test
            test_cg_vs_dense
            test_cg_convergence_rate
            test_block_jacobi_tuning)
        add_executable(nuka_diffsim_${cg_test}
            diffsim/solver/${cg_test}.cpp)
        target_compile_definitions(nuka_diffsim_${cg_test}
            PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(nuka_diffsim_${cg_test}
            PRIVATE
                nuka_diffsim
                nuka_core
                nuka_phi
                nuka_eigen_testonly
                GTest::gtest_main)
        gtest_discover_tests(nuka_diffsim_${cg_test}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "fast_full")
    endforeach()

    # --- v0.7 p02: self-written MINRES (symmetric indefinite) + ILU(0) ----------
    #   * test_minres_indefinite          : MINRES solves a known indefinite system
    #                                       (||Ax-b|| small) + D1 bit-exact.
    #   * test_minres_vs_dense_indefinite : MINRES vs Eigen FullPivLU on indefinite
    #                                       blocks (< 1e-5; FullPivLU, NOT LDLT).
    #   * test_ilu0_factorization         : ILU(0) factor matches a hand reference +
    #                                       host==device + D1 bit-exact.
    #   * test_ilu0_vs_jacobi_convergence : ILU(0)-MINRES converges in <= 0.5x iters
    #                                       vs Jacobi-MINRES on an ill-conditioned
    #                                       structurally-sparse DEFINITE block.
    # Eigen is linked test-only (the approved dense oracle), never a production path.
    foreach(minres_test
            test_minres_indefinite
            test_minres_vs_dense_indefinite
            test_ilu0_factorization
            test_ilu0_vs_jacobi_convergence)
        add_executable(nuka_diffsim_${minres_test}
            diffsim/solver/${minres_test}.cpp)
        target_compile_definitions(nuka_diffsim_${minres_test}
            PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(nuka_diffsim_${minres_test}
            PRIVATE
                nuka_diffsim
                nuka_core
                nuka_phi
                nuka_eigen_testonly
                GTest::gtest_main)
        gtest_discover_tests(nuka_diffsim_${minres_test}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "fast_full")
    endforeach()

    # --- v0.7 p03-A: self-written GMRES (general NON-SYMMETRIC Krylov) -----------
    #   * test_gmres_nonsymmetric : GMRES vs Eigen PartialPivLU/FullPivLU on
    #                               genuinely non-symmetric blocks (< 1e-5); the m<n
    #                               restart-coverage case; the non-symmetric detector
    #                               byte-safety (symmetric -> 0, non-symmetric -> 1);
    #                               and the D1 two-run byte-exact gate.
    # Eigen is linked test-only (the approved dense oracle), never a production path.
    foreach(gmres_test
            test_gmres_nonsymmetric)
        add_executable(nuka_diffsim_${gmres_test}
            diffsim/solver/${gmres_test}.cpp)
        target_compile_definitions(nuka_diffsim_${gmres_test}
            PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(nuka_diffsim_${gmres_test}
            PRIVATE
                nuka_diffsim
                nuka_core
                nuka_phi
                nuka_eigen_testonly
                GTest::gtest_main)
        gtest_discover_tests(nuka_diffsim_${gmres_test}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            PROPERTIES LABELS "fast_full")
    endforeach()

    # --- v0.5 p03 R2: contact KKT (Delassus) builder ------------------------
    # Builder (A = J M^-1 J^T) vs an INDEPENDENT host Gram recompute (Eigen,
    # < 1e-5 rel), exact bit-symmetry, the diagonal == 1/effective_mass identity,
    # and the D1 two-run byte-exact gate. Eigen is linked test-only.
    add_executable(nuka_diffsim_kkt_build_test
        diffsim/test_kkt_build.cpp
    )

    target_compile_definitions(nuka_diffsim_kkt_build_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_kkt_build_test
        PRIVATE
            nuka_diffsim
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_kkt_build_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # --- v0.5 p03 R3a: IFT-at-convergence contact gradient ------------------
    # IFT analytic vs finite-difference through the REAL engine contact step
    # (< 1e-3 rel, the PRIMARY oracle -- p02 is contact-free so there is no
    # contact-tape to compare against), vs a dense Eigen::LDLT solve+distribute
    # reference (< 1e-5, second oracle), and the D1 two-run byte-exact gate.
    # Eigen is linked test-only.
    add_executable(nuka_diffsim_ift_backward_test
        diffsim/test_ift_vs_tape_backward.cpp
    )

    target_compile_definitions(nuka_diffsim_ift_backward_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_ift_backward_test
        PRIVATE
            nuka_diffsim
            nuka_articulation_gpu
            nuka_articulation
            nuka_core
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_ift_backward_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.7 p08-C Channel 1 (§4.B): SDF-contact IFT d/dM, d/dJ ------------
    # The §4.B acceptance the v0.5 ift_runner DEFERRED. The FD PERTURBS actual
    # entries of M^-1 / J and RE-SOLVES the constraint system to convergence with
    # an INDEPENDENT Eigen LDLT oracle (not the module's own assemble/solve, so a
    # transpose/index bug is caught), then compares to the analytical IFT
    # dlambda = A^-1 (dr - dA lambda). Eigen linked test-only.
    add_executable(nuka_sdf_contact_ift_test
        diffsim/test_sdf_contact_ift_fd.cpp
    )

    target_link_libraries(nuka_sdf_contact_ift_test
        PRIVATE
            nuka_diffsim
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sdf_contact_ift_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.5 p03 R3b: OSC differentiable adjoint (R8 / R9) -----------------
    # Reverse-mode gain (Kp,Kd) + task-target gradients of the REAL forward OSC
    # drive kernel: analytic adjoint vs central finite-difference of the live
    # LaunchApplyOscDriveKernels (< 1e-3 rel), the D1 two-run byte-exact gate, and
    # the R9 clamp stop-grad (saturated component -> zero param grad). Links the
    # engine (scene cook + world + forward kernel) AND nuka_diffsim (the adjoint).
    add_executable(nuka_diffsim_osc_adjoint_test
        diffsim/test_osc_adjoint.cpp
    )

    target_compile_definitions(nuka_diffsim_osc_adjoint_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_osc_adjoint_test
        PRIVATE
            nuka_diffsim
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_osc_adjoint_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.5 p04 N1: sim-to-real sensor noise (Philox Gaussian + Poisson) ---
    # Launch-level gtests: Philox KAT + D1 two-run memcmp + statistics. Link the
    # self-contained noise lib directly -- no world/scene needed (fast + robust).
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

    # M9 T11: nuka_floating_base_aba_test (test_floating_base_aba.cpp) +
    # nuka_floating_base_contact_test (test_floating_base_contact.cpp) DELETED with
    # the legacy batched articulated world. Floating-base ABA correctness (legs+base
    # byte-exact vs MJX golden) survives on nk via oracle
    # NkWorldFloatingBaseGoldensMatchCudaAbaByteExact; the floating-base foot-ground
    # contact-recoil through the 18-wide chain-J survives via foot_ground_subsume +
    # h1_grasp_lift + articulation_contact_rows. go2-stance remainder -> M10.

    # --- v0.8 C5c-1: floating-base go2 foot<->ground contact through the UNIFIED
    # collision/contact spine (contact_stream_driver -> sphere x plane narrowphase
    # -> EmitCompliantContactRows -> UnifiedSolve C5b-core ArticulationChainJ arm).
    # PROVES the foot contact force recoils into the floating base via the 18-wide
    # foot chain Jacobian (6 base + 12 leg): a structural base-column J check
    # (roll=+lever_y, pitch=-lever_x, lin_z=1), the base-z recoil through the
    # unified solve, a summed-normal-impulse ~ m_total*g*dt weight balance, and D1
    # two-run byte-identity. Compiles the host contact_stream_driver.cpp directly
    # (validated-not-wired: it is in no production lib); needs nuka_codegen for the
    # generated collidable_registry.hpp.
    add_executable(nuka_foot_ground_subsume_test
        solver/test_foot_ground_subsume.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_ground_subsume_test nuka_codegen)

    target_compile_definitions(nuka_foot_ground_subsume_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_ground_subsume_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_ground_subsume_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            # M9 T11 re-point: the nk SolveRowsBlockIsland harness (sole solve path).
            nuka_nk
            nuka_phi2
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_ground_subsume_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

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
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_link_aabb_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "fast_full")

    # --- v0.8 C5c-2: MJX-parity ACCEPTANCE for the foot-ground subsume.
    # At a FIXED penetrating contact state (go2 feet pressed delta into a +Z
    # ground plane, qpos fixed, qvel=0) one mjx.forward -> qacc is compared to
    # the qacc Nuka produces through the UNIFIED contact pipeline (reuses the
    # C5c-1 wiring). PRIMARY assertion: the BASE-6 qacc match (the foot contact
    # force recoils into the floating base MJX-exactly). The MJX golden
    # (go2_foot_contact_step.bin, generated via generate_mjx_golden.py --mode
    # floating-contact-step) is a SOURCE TRANSLATION of the SAME go2_float.usda
    # Nuka cooks, so both sides share M by construction. Single-step, NOT a
    # rollout. Same host contact_stream_driver.cpp compile as the subsume test.
    add_executable(nuka_foot_ground_mjx_parity_test
        solver/test_foot_ground_mjx_parity.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_ground_mjx_parity_test nuka_codegen)

    target_compile_definitions(nuka_foot_ground_mjx_parity_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_ground_mjx_parity_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_ground_mjx_parity_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            # M4 re-point: the nk SolveRowsBlockIsland harness.
            nuka_nk
            nuka_phi2
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_ground_mjx_parity_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.8 co-residence commit 3/3: THE GATE (the grasp crux). A go2
    # articulation FOOT and a MOVABLE RIGID BODY box co-resident in ONE unified
    # contact solve, BOTH reaction arms firing (foot=ArticulationChainJ, box=
    # RigidInvMass) and the box GPU BodyState velocity actually changing. De-orphans
    # commit 1 (link_aabb.hpp ExtractLinkShapeAabbs) + commit 2 (rigid_candidate_
    # pairs.cu BuildArticulationRigidCandidatePairs) by driving the FULL stack:
    # per-link AABB -> cross-type broadphase -> sphere x box narrowphase ->
    # compliant rows -> UnifiedSolve. PRIMARY: kernel lambda vs the DOUBLE exact
    # (A+R) solve of the co-resident system (J_foot M^-1 J_foot^T + box inv-mass).
    # STRUCTURAL: box linear velocity changes (sign-checked away from the foot), the
    # articulation qdot recoils (two-way), box angular ~0 (the linear-only-emitter
    # grasp-blocker finding, reported). VALIDATED-NOT-WIRED. Links nuka_collision_gpu
    # (rigid_candidate_pairs.cu) + compiles contact_stream_driver.cpp like the
    # subsume/parity tests; needs nuka_codegen for the generated registry.
    add_executable(nuka_foot_box_coresidence_test
        solver/test_foot_box_coresidence.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_box_coresidence_test nuka_codegen)

    target_compile_definitions(nuka_foot_box_coresidence_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_box_coresidence_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_box_coresidence_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            # M9 T11 re-point: the nk SolveRowsBlockIsland harness (sole solve path).
            nuka_nk
            nuka_phi2
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_box_coresidence_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # --- v0.8 W1b: the CO-RESIDENT FREE-BOX foot-contact STEP-0 MJX-parity
    # capstone -- the box-side REALITY ANCHOR to MuJoCo for the unified co-resident
    # contact path. A go2 FL foot penetrates a FREE rigid box; one mjx.forward gives
    # the ground-truth qacc for the articulation base 6 + legs 12 AND the box 6 free
    # DOFs. Nuka's unified pipeline (per-link AABB -> cross-type broadphase ->
    # sphere x box narrowphase -> compliant rows -> UnifiedSolve) must reproduce the
    # BOX 6-DOF qacc (PRIMARY) + the base-6 (SECONDARY, the C5c-2 anchor with the box
    # present). The box mass+inertia are pinned to match MJX exactly (GATE 1). New
    # golden go2_foot_box_contact_step.bin (generated, NOT hand-edited).
    # VALIDATED-NOT-WIRED. Same dep set as nuka_foot_box_coresidence_test.
    add_executable(nuka_foot_box_mjx_parity_test
        solver/test_foot_box_mjx_parity.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_box_mjx_parity_test nuka_codegen)

    target_compile_definitions(nuka_foot_box_mjx_parity_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_box_mjx_parity_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_box_mjx_parity_test
        PRIVATE
            # M4 re-point: the nk SolveRowsBlockIsland harness.
            nuka_nk
            nuka_phi2
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_box_mjx_parity_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTIES LABELS "full")

    # M9 T11: the coresident validated-not-wired test estate DELETED with the
    # legacy classes (the co-resident stepper / the batched union world / the grasp
    # scene factory): nuka_unified_coresident_stepper_test, nuka_batched_unified_world_test,
    # nuka_batched_unified_world_perf_test, nuka_grasp_hold_spike_test,
    # nuka_h1_grasp_spike_test. Their unique coverage is folded/acceptable-loss per
    # subagent-plans/m9-t10-coverage-fold.md (broadphase D1 / RNG D1 / Krylov etc.
    # survive standalone; the grasp force-balance/BITE/D1 lives in h1_grasp_lift).

    # --- M7 T6: the 4 H1 grasp DEV-SPIKE tests (power_grasp_lift /
    # scaled_cup_grasp / dense_grasp / grasp_feasibility_probe) were DELETED.
    # Their hard GO assertions (force-balance, BITE, disturbance, D1, the
    # conjoint honest gate) consolidated into tests/scenario/h1_grasp_lift.cpp
    # (the permanent .nks grasp gate); the rest were SKIP-with-finding spikes.

    # M9 T11: nuka_batched_h1_hand_grasp_test (batched H1-hand grasp vs co-resident
    # oracle byte-exact) DELETED with the coresident estate.

    # M9 T11: nuka_batched_h1_union_world_test (SG G1 batched-union parity) DELETED
    # with the coresident estate -- the batched union world vs co-resident oracle
    # parity dies with the legacy classes (golden/nk path is authority).

    # --- M7 T6: the factory byte-equality gate (nuka_h1_union_scene_factory_test)
    # and the transitional cook==factory gate (nuka_union_cook_matches_factory_test)
    # were DELETED with the factory they pinned. The permanent gate is
    # nuka_h1_grasp_lift_test below (the pure .nks-cooked grasp gate).

    # M9 T14: nuka_h1_grasp_lift_test (THE permanent .nks grasp gate) +
    # nuka_union_cook_golden_test (the cook-fidelity golden) MERGED into
    # nuka_scenario_test (scenario/h1_grasp_lift.cpp + scenario/union_cook_golden.cpp;
    # union_nk_model.cpp + contact_stream_driver.cpp now come from the shared
    # nuka_test_validated_sources lib, no longer a per-exe extra-source).

    # M9 T11: the H1 co-resident demo/transfer/bridge spikes DELETED with the
    # coresident estate: nuka_h1_cup_sequence_demo_test (C7b-2 stand+grasp scaffold),
    # nuka_h1_costand_transfer_test (S3 full-DOF stand transfer),
    # nuka_h1_bridge_spike_test (PPO->co-resident inference seam). RL-adjacent /
    # validated-not-wired; rebuild on the nk world at M10 per the SG freeze.

    # M6 RE-POINT: the ParticleInvMass reaction arm on the nk path (the device-
    # resident nk::SolveRowsBlockIsland, the union kUSlotParticleSphereBox class).
    # A particle presses a free rigid box -> both react two-way (momentum balance);
    # the box held on a static plane supports the particle through the
    # particle->box->plane chain in ONE unified solve; D1. (The legacy host
    # solver::UnifiedSolve + EmitCouplingRows are no longer consumed here.)
    # M9 T14: nuka_coupling_test MERGED into nuka_scenario_test
    # (coupling/test_coupling_framework.cpp; the scenario category exe above).

    # --- v0.8 C6a: the PBD<->rigid unified-contact CO-STEP bridge -------------
    # Drives the REAL front-end together (C2c broadphase -> C5 contact driver ->
    # C6b coupling framework -> C6b unified solver) for the Genesis-style "couple"
    # phase: a PBD particle system consumes unified manifolds + reacts TWO-WAY
    # M6 RE-POINT: a particle system co-steps against a rigid box through the ONE
    # unified row solve (nk::World, the kUSlotParticleSphereBox union class ->
    # AssembleRows -> SolveRowsBlockIsland ParticleInvMass arm). The legacy host
    # bridge (unified_costep.cpp, the M6 deletion target) is no longer consumed.
    # CASE A: multi-particle drop, TOTAL-energy non-increasing + settle + two-way.
    # CASE C: rest-on-box. CASE D: D1.
    # M9 T14: nuka_pbd_costep_test MERGED into nuka_scenario_test
    # (coupling/test_pbd_costep_unified.cpp; the scenario category exe above).

    # M9 T11: nuka_go2_pd_standing_test (test_go2_pd_standing.cpp) DELETED with the
    # legacy batched articulated world. Pure go2-quadruped PD-stance locomotion
    # precursor -> M10 carry-forward (crouch-hold height tracking, Sigma lambda=Mg/4
    # per foot, COM-in-foot-polygon static balance, no-tip, 4096-env determinism;
    # see subagent-plans/m9-t11-articulated-fold.md M10 carry-forward #1). Generic
    # foot-ground recoil/friction/D1/scale already on nk via foot_ground_subsume +
    # h1_grasp_lift + articulation_contact_rows + multi_env.

    # M9 T11: nuka_link_contact_wrench_test (test_link_contact_wrench.cpp) DELETED
    # with the legacy batched articulated world. The synthetic ComputeContactWrench-
    # Outputs kernel arm + the LINK_CONTACT_WRENCH field binary contract survive
    # (the kernel + multi_env SingleEnvStillSupported); the go2-stance-balances-
    # weight + graph==wrench arms -> M10 carry-forward (per-link contact-wrench
    # readback on a seated go2 nk world; ledger M10 carry-forward #2).

    # M9 T11: nuka_batched_reset_test (RL autoreset) + nuka_base_pose_view_test
    # (episode-boundary un-lagged base pose) DELETED -- batched-only / RL-adjacent,
    # their contracts carried forward to M10 (rebuild on the nk world when RL
    # returns; see subagent-plans/m9-t10-coverage-fold.md M10 carry-forward).

    # M9 T11: nuka_batched_articulated_world_test (test_batched_articulated_world.cpp),
    # nuka_batched_articulated_graph_test (test_batched_articulated_graph.cpp), and
    # nuka_batched_articulated_world_perf_test (test_batched_articulated_world_perf_
    # detail.cpp) ALL DELETED with the legacy batched articulated world. The class
    # orchestration (env-major + cross-env/two-run D1 + 4096 scale) survives on nk
    # via multi_env (DeterminismCrossEnvAndTwoRuns + HeadlineScale4096 +
    # SingleEnvStillSupported); batched==engine bit-exact + graph==step survive via
    # oracle NkWorldRandomSampleGoldensMatchCudaAbaByteExact +
    # NkWorldBatchedContactStepPlannedByteExact (+ h1_grasp_lift plan-replay); perf
    # via nuka_go2_4096env_step_time_test NkWorldStepPlannedMeetsGatePerEnv. The
    # perf-detail file was self-documented "measurement harness, not a correctness
    # gate" (zero physics). The velocity / non-PD control mode + per-mode two-run
    # byte-exact contract (was nuka_control_modes_test) is carried forward to M10
    # (rebuild on the nk world; see subagent-plans/m9-t10-coverage-fold.md).

    # M9 T14: the 4096-env Go2 perf gate (test_go2_4096env_step_time.cpp), the
    # cuda_perf throughput gates (test_lbvh_50k_bodies.cpp +
    # test_1m_particles_grid_perf.cpp) and the perf-timing infra contract
    # (test_perf_timing.cu) are MERGED into nuka_perf_test (the perf category exe,
    # defined above). Their link sets (nuka_runtime/_gpu + articulation_gpu +
    # collision_gpu + scene_cook + CUDA::cudart) are unioned there.
    # (M9 T11: tools/perf/perf_harness.cpp was deleted; the live throughput gates
    # are the nuka_perf_test gtest binaries, see tools/perf/run_perf_baseline.py.)
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

# Reusable whole-solver capacity probe: real scene cook/upload plus exact
# Model/Data arena segments and the emitted op list (not a local unit test).
add_executable(nuka_solver_arena_profile
    ${CMAKE_SOURCE_DIR}/examples/demo/solver_arena_profile.cpp
)
add_dependencies(nuka_solver_arena_profile nuka_codegen)
target_compile_definitions(nuka_solver_arena_profile
    PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(nuka_solver_arena_profile
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
target_link_libraries(nuka_solver_arena_profile
    PRIVATE
        nuka_scene_cook
        nuka_nk
        nuka_phi2
        nuka_runtime
        nuka_import
        nuka_cooker
        nuka_articulation_gpu
        nuka_collision_gpu
        nuka_core
        nuka_phi
)

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

    # M8 T3a -- offscreen raster renderer smoke (synthetic RenderWorld; two
    # renders byte-identical [G2] + non-background pixels > 0 [G3]).
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

    # M8 T7 -- render_physics_parity: THE M8 EXIT GATE. Builds the real H1 union
    # fixture (Load(.nks) -> CookToModel -> nk::World + BuildRenderWorld +
    # Simulation, the render_frame_loop_smoke path) and asserts G1 (1:1
    # physics<->render bit-exact over ALL physics-driven instances), G2 (two
    # renders byte-identical, RAW pixels), G3 (non-background pixels > 0). Reads
    # poses from nk::World::DownloadField directly (Decision D2). Asset-gated SKIP;
    # the whole gate is behind NK_BUILD_VULKAN_VALIDATION (real Vulkan device).
    # Links the nk physics libs + nuka_render + nuka_app (mirrors the
    # render_frame_loop_smoke link set + the raster renderer).
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
            nuka_nk
            nuka_phi2
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # The body<->particle coupling headline clip: a real XPBD cloth draped over two
    # ridges, pressed + released by a rigid sphere on the ONE general row solver.
    # Unlike the pose-replay video tools this one STEPS nk::World on CUDA (settle +
    # the scripted press) and renders the live deforming cloth offscreen. Links the
    # cloth topology builder (nuka_runtime_gpu) + the full physics + render stack.
    add_executable(nuka_cloth_presser_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/cloth_presser_demo.cpp
    )
    add_dependencies(nuka_cloth_presser_demo nuka_codegen)
    target_compile_definitions(nuka_cloth_presser_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_cloth_presser_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_cloth_presser_demo
        PRIVATE
            nuka_scene_cook
            nuka_render
            nuka_nk
            nuka_phi2
            nuka_runtime_gpu
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # Soft<->rigid coupling headline clip: a soft tet ball (a sphere rest-lattice
    # tessellated into distance + volume tets) dropped onto a static ground plane on
    # the ONE general body<->particle row solver. STEPS nk::World on CUDA and renders
    # the live deforming surface mesh on the CUDA RT backend (--gpu --beauty) or
    # lavapipe raster. Links the tet topology builder (nuka_runtime_gpu) + the CUDA
    # ray tracer (nuka_phi2_rt) + the full physics + render stack.
    add_executable(nuka_soft_ball_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/soft_ball_demo.cpp
    )
    add_dependencies(nuka_soft_ball_demo nuka_codegen)
    target_compile_definitions(nuka_soft_ball_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_soft_ball_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_soft_ball_demo
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # Demo 1 headline clip: a heavy rigid Stanford bunny dropped into an MLS-MPM
    # water pool (weakly-compressible FLUID constitutive model_kind==3, Tait EOS),
    # TWO-WAY coupled through the env-private grid (the SAME grid BC the floor uses).
    # STEPS nk::World on CUDA and renders the live MPM fluid marching-cubes isosurface
    # as a refractive dielectric with teal depth absorption (ior 1.33, transmit
    # bounces, Taubin smoothing) + the bunny (SDF collider, posed from body_pose) + a
    # checker pool bottom on the CUDA RT beauty backend. --dump/--from decouple the
    # ~once sim from cheap render iteration; writes a PNG sequence (-> mp4).
    add_executable(nuka_mpm_water_drop_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/mpm_water_drop_demo.cpp
    )
    add_dependencies(nuka_mpm_water_drop_demo nuka_codegen)
    target_compile_definitions(nuka_mpm_water_drop_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_mpm_water_drop_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_mpm_water_drop_demo
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # Cloth<->rigid coupling headline clip: a free unpinned square cloth sheet released
    # above a PD-standing Go2 (full per-link collision skeleton cooked from go2.nks)
    # drapes over the body on the ONE general body<->particle row solver. STEPS nk::World
    # on CUDA and renders the live deforming cloth surface mesh + the FK-posed Go2 visuals
    # on the CUDA RT backend (--gpu --beauty) or lavapipe raster.
    add_executable(nuka_go2_cloth_drape_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/go2_cloth_drape_demo.cpp
    )
    add_dependencies(nuka_go2_cloth_drape_demo nuka_codegen)
    target_compile_definitions(nuka_go2_cloth_drape_demo
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NK_BUILD_VULKAN_VALIDATION)
    target_include_directories(nuka_go2_cloth_drape_demo
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_go2_cloth_drape_demo
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
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
    )

    # XPBD constraint-solve micro-benchmark (serial vs colored solver timing).
    add_executable(nuka_xpbd_solve_bench
        ${CMAKE_SOURCE_DIR}/examples/demo/xpbd_solve_bench.cpp
    )
    add_dependencies(nuka_xpbd_solve_bench nuka_codegen)
    target_include_directories(nuka_xpbd_solve_bench
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_xpbd_solve_bench
        PRIVATE
            nuka_scene_cook
            nuka_nk
            nuka_phi2
            nuka_runtime_gpu
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
            nuka_articulation_gpu
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
