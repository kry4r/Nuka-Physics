# Warnings.cmake
# Provides nuka_set_warnings(<target>) to apply strict warning flags.

option(NK_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

function(nuka_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
            /wd4819       # suppress code-page encoding warning
        )
        if(NK_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
        )
        if(NK_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()
