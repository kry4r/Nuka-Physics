cmake_minimum_required(VERSION 3.24)
project(nuka-physics
    VERSION 0.1.0
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Guard against in-source builds
include(cmake/PreventInSourceBuilds.cmake)

# Compiler-warning helper
include(cmake/Warnings.cmake)
include(CMakePackageConfigHelpers)
include(CheckCXXSourceCompiles)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NK_BUILD_TESTS "Build unit / smoke tests" ON)
option(NK_BUILD_VULKAN_VALIDATION "Build Vulkan-specific validation and timing tests" OFF)
option(NK_BUILD_DEMO_TIMING_VALIDATION "Build scene demo render timing tests" OFF)
option(NK_REQUIRE_CUDA "Require CUDA support for the preferred production physics backend" ON)
set(NK_PHYSICS_BACKEND "CUDA" CACHE STRING "Preferred production physics backend")
set_property(CACHE NK_PHYSICS_BACKEND PROPERTY STRINGS CUDA CPU_REFERENCE)
set(NK_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for production physics kernels")

check_cxx_source_compiles(
    "#include <expected>
int main() { std::expected<int, int> value = 1; return *value == 1 ? 0 : 1; }"
    NUKA_HAS_STD_EXPECTED
)

# ---------------------------------------------------------------------------
# CUDA-preferred physics backend and Vulkan renderer backend
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NK_CUDA_ARCHITECTURES STREQUAL "native"
       AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
        set(NK_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures for production physics kernels" FORCE)
        message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not support native Ada targets; using sm_86")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${NK_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to build" FORCE)
    message(STATUS "CUDA enabled for preferred physics backend: ${CMAKE_CUDA_COMPILER}")
else()
    if(NK_REQUIRE_CUDA OR NK_PHYSICS_BACKEND STREQUAL "CUDA")
        message(FATAL_ERROR "CUDA compiler not found, but CUDA is the required/preferred production physics backend.")
    endif()
    message(STATUS "CUDA compiler not found; building CPU reference backend only")
endif()

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(NUKA_CODEGEN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/codegen/generated")
set(NUKA_CODEGEN_GENERATED_SOURCES
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_joint_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_drive_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/featherstone_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/row_dispatch.cu"
)
set(NUKA_CODEGEN_GENERATED_HEADERS
    "${NUKA_CODEGEN_OUTPUT_DIR}/row_class_registry.hpp"
)
set(NUKA_CODEGEN_OUTPUTS
    ${NUKA_CODEGEN_GENERATED_SOURCES}
    ${NUKA_CODEGEN_GENERATED_HEADERS}
)
file(GLOB NUKA_ROW_IR_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/codegen/classes/*.yaml"
)
file(GLOB NUKA_ROW_TEMPLATES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/codegen/templates/*.j2"
)
add_custom_command(
    OUTPUT ${NUKA_CODEGEN_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NUKA_CODEGEN_OUTPUT_DIR}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/codegen/regen.py"
    DEPENDS
        ${NUKA_ROW_IR_FILES}
        ${NUKA_ROW_TEMPLATES}
        "${CMAKE_SOURCE_DIR}/tools/codegen/schema/row_v0_1.yaml"
        "${CMAKE_SOURCE_DIR}/tools/codegen/regen.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Regenerating row class CUDA kernels from IR"
    VERBATIM
)
add_custom_target(nuka_codegen ALL
    DEPENDS ${NUKA_CODEGEN_OUTPUTS}
)

add_custom_target(nuka_run_lint
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/lint/physics_smell.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running physics-smell lint"
)

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
include(FetchContent)

# tinyxml2 (needed by importers)
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        10.0.0
)
set(tinyxml2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinyxml2)

if(NK_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Prevent GoogleTest from overriding our compiler/linker options on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/nukaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nukaConfig.cmake"
    INSTALL_DESTINATION lib/cmake/nuka
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nukaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

if(NK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nukaConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nukaConfigVersion.cmake"
    DESTINATION lib/cmake/nuka
)
