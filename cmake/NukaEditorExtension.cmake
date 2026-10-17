# NukaEditorExtension.cmake
# nuka_add_editor_extension(<name> SOURCES <files...> [IMGUI]) builds a native editor extension.

if(WIN32)
    set(NUKA_EDITOR_EXTENSION_PLATFORM windows-x64)
else()
    set(NUKA_EDITOR_EXTENSION_PLATFORM linux-x64)
endif()

function(nuka_add_editor_extension name)
    cmake_parse_arguments(EXT "IMGUI" "" "SOURCES" ${ARGN})
    set(root "${CMAKE_BINARY_DIR}/src/extensions/${name}")
    set(bin "${root}/bin/${NUKA_EDITOR_EXTENSION_PLATFORM}")
    add_library(${name} SHARED ${EXT_SOURCES})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
    set_target_properties(${name} PROPERTIES
        PREFIX ""
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        RUNTIME_OUTPUT_DIRECTORY "${bin}"
        LIBRARY_OUTPUT_DIRECTORY "${bin}")
    if(EXT_IMGUI)
        # The Dear ImGui core without a context; the plugin draws in the editor's.
        target_link_libraries(${name} PRIVATE nuka_imgui_plugin)
        target_compile_definitions(${name} PRIVATE NUKA_EDITOR_PLUGIN_IMGUI)
    endif()
    # The extension.toml and i18n/ beside the caller's CMakeLists.txt go next to the library.
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/extension.toml" "${root}/extension.toml")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/i18n")
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/i18n" "${root}/i18n")
    endif()
    nuka_set_warnings(${name})
endfunction()

# A Python or data extension is its folder; the build copies it beside the native ones.
function(nuka_add_editor_extension_folder name)
    add_custom_target(${name} ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/${name}" "${CMAKE_BINARY_DIR}/src/extensions/${name}")
endfunction()
