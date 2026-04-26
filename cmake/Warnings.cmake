# Warnings.cmake
# Provides nuka_set_warnings(<target>) to apply strict warning flags.

function(nuka_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
            /WX           # treat warnings as errors
            /wd4819       # suppress code-page encoding warning
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Werror       # treat warnings as errors
        )
    endif()
endfunction()
