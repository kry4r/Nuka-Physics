// nuka-editor: runs `python -m nuka.editor` in an embedded interpreter, the private CPython in
// the `python` folder beside the exe when there is one, else the interpreter of the build.

#include <Python.h>

#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>

#include <cstdio>
#include <io.h>
#include <string_view>
#endif

namespace {

std::filesystem::path ExeDir() {
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    path.resize(GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size())));
    return std::filesystem::path(path).parent_path();
#else
    return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
}

#ifdef _WIN32
bool HasStdHandle(DWORD which) {
    const HANDLE handle = GetStdHandle(which);
    return handle != nullptr && handle != INVALID_HANDLE_VALUE && GetFileType(handle) != FILE_TYPE_UNKNOWN;
}

// A GUI program starts without a console: it borrows the terminal it was started from, or
// opens one when `required`. Streams the parent redirected keep pointing where they did.
void BindConsole(bool required) {
    const bool had_in = HasStdHandle(STD_INPUT_HANDLE), had_out = HasStdHandle(STD_OUTPUT_HANDLE),
               had_err = HasStdHandle(STD_ERROR_HANDLE);
    if (had_out && had_err) return;
    if (!AttachConsole(ATTACH_PARENT_PROCESS) && !(required && AllocConsole())) return;
    FILE* stream = nullptr;
    if (!had_in && freopen_s(&stream, "CONIN$", "r", stdin) == 0) _dup2(_fileno(stdin), 0);
    if (!had_out && freopen_s(&stream, "CONOUT$", "w", stdout) == 0) _dup2(_fileno(stdout), 1);
    if (!had_err && freopen_s(&stream, "CONOUT$", "w", stderr) == 0) _dup2(_fileno(stderr), 2);
}
#endif

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    bool console = false;
    for (int i = 1; i < argc; ++i) {
        const std::wstring_view arg = argv[i];
        console = console || arg == L"--console" || arg == L"--headless" || arg == L"--help" || arg == L"-h";
    }
    BindConsole(console);
#else
int main(int argc, char** argv) {
#endif
    const std::filesystem::path home = ExeDir() / "python";
    std::error_code error;
    const bool bundled = std::filesystem::exists(home / NUKA_PY_ZIP, error);

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    // Every argument belongs to the editor, none to the interpreter.
    config.parse_argv = 0;
    PyStatus status = PyConfig_SetString(&config, &config.run_module, L"nuka.editor");
    if (bundled) {
        // The private CPython: its standard library zip, extension modules and site-packages,
        // untouched by PYTHONPATH and the user site.
        config.isolated = 1;
        config.module_search_paths_set = 1;
        const std::wstring home_text = home.wstring(), zip_text = (home / NUKA_PY_ZIP).wstring();
        if (!PyStatus_Exception(status)) status = PyConfig_SetString(&config, &config.home, home_text.c_str());
        if (!PyStatus_Exception(status)) status = PyWideStringList_Append(&config.module_search_paths, zip_text.c_str());
        if (!PyStatus_Exception(status)) status = PyWideStringList_Append(&config.module_search_paths, home_text.c_str());
    } else if (!PyStatus_Exception(status)) {
        status = PyConfig_SetBytesString(&config, &config.home, NUKA_PY_HOME);
    }
#ifdef _WIN32
    if (!PyStatus_Exception(status)) status = PyConfig_SetArgv(&config, argc, argv);
#else
    if (!PyStatus_Exception(status)) status = PyConfig_SetBytesArgv(&config, argc, argv);
#endif
    if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
#ifdef _WIN32
        const std::string text = std::string("Nuka Editor could not start its Python runtime:\n\n") +
                                 (status.err_msg != nullptr ? status.err_msg : "unknown error");
        MessageBoxA(nullptr, text.c_str(), "Nuka Editor", MB_OK | MB_ICONERROR);
#endif
        Py_ExitStatusException(status);
    }
    // Marks the app, which starts further editors through this exe.
    PyRun_SimpleString("import sys; sys.frozen = True");
    return Py_RunMain();
}
