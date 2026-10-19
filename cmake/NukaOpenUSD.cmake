# ---------------------------------------------------------------------------
# OpenUSD (Modified Apache 2.0): composition, file formats and plugins for the USD importer.
# Never vendored: the first configure builds the pinned release into NK_OPENUSD_ROOT.
# ---------------------------------------------------------------------------
set(NK_OPENUSD_ROOT "${CMAKE_SOURCE_DIR}/.nuka-deps/openusd-v26.08" CACHE PATH
    "OpenUSD install prefix; tools/deps/build_openusd.py fills it when it is empty")
if(NOT EXISTS "${NK_OPENUSD_ROOT}/pxrConfig.cmake")
    message(STATUS "Building OpenUSD into ${NK_OPENUSD_ROOT}; this first configure takes a while")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/deps/build_openusd.py"
                --prefix "${NK_OPENUSD_ROOT}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()
find_package(pxr CONFIG REQUIRED PATHS "${NK_OPENUSD_ROOT}" NO_DEFAULT_PATH)
# The pxr headers include TBB 2020's deprecated task.h, which prints a notice per translation unit.
set_property(TARGET arch APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS TBB_SUPPRESS_DEPRECATED_MESSAGES=1)
if(WIN32)
    # The pxr headers pull in TBB's <windows.h>, whose min/max macros break them; OpenUSD builds
    # itself with NOMINMAX but its exported targets do not pass it on.
    set_property(TARGET arch APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS NOMINMAX)
endif()

# Target <name> stages the USD runtime into each directory: the libraries, and one `usd/` plugin
# tree (core and file-format plugins merged), which Plug finds beside its own library.
function(nuka_stage_usd_runtime name)
    set(commands "")
    if(WIN32)
        file(GLOB dlls "${NK_OPENUSD_ROOT}/lib/*.dll" "${NK_OPENUSD_ROOT}/bin/tbb.dll")
        foreach(dir IN LISTS ARGN)
            list(APPEND commands
                COMMAND ${CMAKE_COMMAND} -E make_directory "${dir}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dlls} "${dir}"
                COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${NK_OPENUSD_ROOT}/lib/usd" "${dir}/usd"
                COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${NK_OPENUSD_ROOT}/plugin/usd" "${dir}/usd")
        endforeach()
    endif()
    add_custom_target(${name} ${commands} VERBATIM)
endfunction()
