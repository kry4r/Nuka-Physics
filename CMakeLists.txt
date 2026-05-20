cmake_minimum_required(VERSION 3.24)
project(nuka-physics
    VERSION 0.1.0
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Guard against in-source builds
include(cmake/PreventInSourceBuilds.cmake)

# Compiler-warning helper
include(cmake/Warnings.cmake)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NK_BUILD_TESTS "Build unit / smoke tests" ON)
option(NK_REQUIRE_CUDA "Require CUDA support for the preferred production physics backend" ON)
set(NK_PHYSICS_BACKEND "CUDA" CACHE STRING "Preferred production physics backend")
set_property(CACHE NK_PHYSICS_BACKEND PROPERTY STRINGS CUDA CPU_REFERENCE)
set(NK_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for production physics kernels")

# ---------------------------------------------------------------------------
# CUDA-preferred physics backend and Vulkan renderer backend
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "${NK_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to build" FORCE)
    message(STATUS "CUDA enabled for preferred physics backend: ${CMAKE_CUDA_COMPILER}")
else()
    if(NK_REQUIRE_CUDA OR NK_PHYSICS_BACKEND STREQUAL "CUDA")
        message(FATAL_ERROR "CUDA compiler not found, but CUDA is the required/preferred production physics backend.")
    endif()
    message(STATUS "CUDA compiler not found; building CPU reference backend only")
endif()

find_package(Vulkan REQUIRED)

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

if(NK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
