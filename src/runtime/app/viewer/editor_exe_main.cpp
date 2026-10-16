// nuka-editor: runs `python -m nuka.editor` in an embedded interpreter, so the launcher
// and the Python entry point share one code path.

#include <Python.h>

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    // Every argument belongs to the editor, none to the interpreter.
    config.parse_argv = 0;
    PyStatus status = PyConfig_SetBytesString(&config, &config.home, NUKA_PY_HOME);
    if (!PyStatus_Exception(status)) status = PyConfig_SetString(&config, &config.run_module, L"nuka.editor");
#ifdef _WIN32
    if (!PyStatus_Exception(status)) status = PyConfig_SetArgv(&config, argc, argv);
#else
    if (!PyStatus_Exception(status)) status = PyConfig_SetBytesArgv(&config, argc, argv);
#endif
    if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);
    return Py_RunMain();
}
