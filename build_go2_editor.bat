@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "C:\Softwares\code\Nuka-Physics"
"C:\PROGRA~1\MICROS~1\2022\COMMUN~1\Common7\IDE\COMMON~1\MICROS~1\CMake\Ninja\ninja.exe" -C build-win-editor nuka_editor_exe
echo BUILD_EXIT=%ERRORLEVEL%
