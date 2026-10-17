#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer -- native "Open file" dialog helper.
//
// A thin platform shim so the editor's Load panel can browse the filesystem for
// a scene through the OS-native file-open dialog (Windows: GetOpenFileNameW;
// Linux: zenity when present). The file.open and file.save_as actions ask it
// when they get no path.
//
// Isolated behind this seam so <windows.h> never leaks into imgui_layer.cpp.
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include <string>

namespace nuka::runtime::app::viewer {

// Open the OS file-open dialog (`filter_label` + `filter_glob` = one filter, e.g.
// "Nuka scenes", "*.nks"). Returns the chosen path, or "" on cancel / no dialog.
std::string OpenFileDialog(const char* title, const char* filter_label,
                           const char* filter_glob);

// The OS file-SAVE dialog (Windows GetSaveFileNameW; Linux zenity --save).
// `suggested` seeds the filename. Returns the chosen path, or "" on cancel.
std::string SaveFileDialog(const char* title, const char* filter_label,
                           const char* filter_glob, const char* suggested);

// The OS folder picker (Windows IFileOpenDialog with FOS_PICKFOLDERS; Linux zenity
// --directory). Returns the chosen folder, or "" on cancel.
std::string OpenFolderDialog(const char* title);

}  // namespace nuka::runtime::app::viewer
