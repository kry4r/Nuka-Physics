#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace nuka::render {

inline std::filesystem::path ViewerExecutableDirectory() {
#ifdef _WIN32
    wchar_t buffer[32768];
    const DWORD length = GetModuleFileNameW(nullptr, buffer, 32768u);
    if (length != 0u && length < 32768u) return std::filesystem::path(buffer).parent_path();
#else
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) return executable.parent_path();
#endif
    return {};
}

inline std::string ViewerResource(const char* relative, const char* build_path) {
    const auto executable = ViewerExecutableDirectory();
    std::error_code error;
    for (const auto& candidate : {executable / relative, executable / "../share/nuka" / relative,
                                  std::filesystem::path(build_path)}) {
        if (std::filesystem::is_regular_file(candidate, error)) return candidate.string();
    }
    return build_path;
}

}  // namespace nuka::render
