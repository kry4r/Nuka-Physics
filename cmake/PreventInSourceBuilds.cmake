# PreventInSourceBuilds.cmake
# Abort configuration when the build directory is the same as the source tree.

get_filename_component(_srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_bindir "${CMAKE_BINARY_DIR}" REALPATH)

if("${_srcdir}" STREQUAL "${_bindir}")
    message(FATAL_ERROR
        "\n"
        "In-source builds are not allowed.\n"
        "Please create a separate build directory and run CMake from there:\n"
        "\n"
        "  cmake -S . -B build\n"
        "  cmake --build build\n"
        "\n"
        "Remove CMakeCache.txt and the CMakeFiles/ directory that were just created "
        "in the source tree before retrying.\n"
    )
endif()
