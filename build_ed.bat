@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set PATH=%CUDA_PATH%\bin;%PATH%
cd /d C:\Softwares\code\Nuka-Physics
cmake --build build-win-editor --target nuka_editor_exe -j 16
echo BUILD_EXIT=%ERRORLEVEL%
