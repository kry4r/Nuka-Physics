#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nuka::render {

// The directory of the module holding this code, so a library finds the resources
// installed next to it whichever executable loads it.
inline std::filesystem::path ViewerModuleDirectory() {
    static const int anchor = 0;
#ifdef _WIN32
    HMODULE handle = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&anchor), &handle)) {
        wchar_t buffer[32768];
        const DWORD length = GetModuleFileNameW(handle, buffer, 32768u);
        if (length != 0u && length < 32768u) return std::filesystem::path(buffer).parent_path();
    }
#else
    Dl_info info{};
    if (dladdr(&anchor, &info) != 0 && info.dli_fname != nullptr)
        return std::filesystem::absolute(info.dli_fname).parent_path();
#endif
    return {};
}

inline std::string ViewerResource(const char* relative, const char* build_path) {
    const auto dir = ViewerModuleDirectory();
    std::error_code error;
    for (const auto& candidate : {dir / relative, dir / "../share/nuka" / relative,
                                  std::filesystem::path(build_path)}) {
        if (std::filesystem::is_regular_file(candidate, error)) return candidate.string();
    }
    return build_path;
}

}  // namespace nuka::render
