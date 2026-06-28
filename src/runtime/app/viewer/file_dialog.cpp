// ---------------------------------------------------------------------------
// Native "Open file" dialog: GetOpenFileNameW on Windows, zenity on Linux.
//
// Kept in its own TU so <windows.h> (and its min/max + ANSI/UNICODE macros)
// never leak into imgui_layer.cpp. HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/file_dialog.hpp"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

#include <vector>

#pragma comment(lib, "comdlg32.lib")

namespace nuka::runtime::app::viewer {
namespace {

std::wstring Widen(const char* s) {
    if (!s || !*s) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    return w;
}

std::string Narrow(const wchar_t* w) {
    if (!w || !*w) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

// Append one wide field plus its terminating null into the filter buffer.
void AppendField(std::vector<wchar_t>& buf, const std::wstring& field) {
    buf.insert(buf.end(), field.begin(), field.end());
    buf.push_back(L'\0');
}

}  // namespace

std::string OpenFileDialog(const char* title, const char* filter_label,
                           const char* filter_glob) {
    const std::wstring wglob = Widen(filter_glob && *filter_glob ? filter_glob : "*.*");
    std::wstring wlabel = Widen(filter_label && *filter_label ? filter_label : "Files");
    wlabel += L" (" + wglob + L")";

    // The API wants a double-null-terminated "display\0pattern\0...\0\0" filter.
    std::vector<wchar_t> filter;
    AppendField(filter, wlabel);
    AppendField(filter, wglob);
    AppendField(filter, L"All files (*.*)");
    AppendField(filter, L"*.*");
    filter.push_back(L'\0');

    const std::wstring wtitle = Widen(title);
    wchar_t path[MAX_PATH];
    path[0] = L'\0';

    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter.data();
    ofn.lpstrFile = path;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = wtitle.empty() ? nullptr : wtitle.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (GetOpenFileNameW(&ofn)) {
        return Narrow(path);
    }
    return std::string();
}

std::string SaveFileDialog(const char* title, const char* filter_label,
                           const char* filter_glob, const char* suggested) {
    const std::wstring wglob = Widen(filter_glob && *filter_glob ? filter_glob : "*.*");
    std::wstring wlabel = Widen(filter_label && *filter_label ? filter_label : "Files");
    wlabel += L" (" + wglob + L")";

    std::vector<wchar_t> filter;
    AppendField(filter, wlabel);
    AppendField(filter, wglob);
    AppendField(filter, L"All files (*.*)");
    AppendField(filter, L"*.*");
    filter.push_back(L'\0');

    const std::wstring wtitle = Widen(title);
    const std::wstring wsugg = Widen(suggested);
    wchar_t path[MAX_PATH];
    path[0] = L'\0';
    if (!wsugg.empty()) {
        wcsncpy(path, wsugg.c_str(), MAX_PATH - 1);
        path[MAX_PATH - 1] = L'\0';
    }

    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter.data();
    ofn.lpstrFile = path;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = wtitle.empty() ? nullptr : wtitle.c_str();
    ofn.lpstrDefExt = L"nks";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (GetSaveFileNameW(&ofn)) {
        return Narrow(path);
    }
    return std::string();
}

}  // namespace nuka::runtime::app::viewer

#else  // !_WIN32

#include <array>
#include <cstdio>

namespace nuka::runtime::app::viewer {

std::string OpenFileDialog(const char* title, const char* filter_label,
                           const char* filter_glob) {
    // Native GTK dialog via zenity when present; "" on cancel / no zenity. The
    // arguments are compile-time constants, so the command carries no external input.
    std::string cmd = "zenity --file-selection";
    if (title && *title) cmd += std::string(" --title=\"") + title + "\"";
    if (filter_glob && *filter_glob) {
        cmd += " --file-filter=\"";
        if (filter_label && *filter_label) cmd += std::string(filter_label) + " | ";
        cmd += filter_glob;
        cmd += "\"";
    }
    cmd += " 2>/dev/null";

    std::FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return std::string();
    std::string out;
    std::array<char, 1024> chunk{};
    while (std::fgets(chunk.data(), static_cast<int>(chunk.size()), pipe) != nullptr) {
        out += chunk.data();
    }
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

std::string SaveFileDialog(const char* title, const char* filter_label,
                           const char* filter_glob, const char* suggested) {
    std::string cmd = "zenity --file-selection --save --confirm-overwrite";
    if (title && *title) cmd += std::string(" --title=\"") + title + "\"";
    if (suggested && *suggested) cmd += std::string(" --filename=\"") + suggested + "\"";
    if (filter_glob && *filter_glob) {
        cmd += " --file-filter=\"";
        if (filter_label && *filter_label) cmd += std::string(filter_label) + " | ";
        cmd += filter_glob;
        cmd += "\"";
    }
    cmd += " 2>/dev/null";

    std::FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return std::string();
    std::string out;
    std::array<char, 1024> chunk{};
    while (std::fgets(chunk.data(), static_cast<int>(chunk.size()), pipe) != nullptr) {
        out += chunk.data();
    }
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

}  // namespace nuka::runtime::app::viewer

#endif  // _WIN32
