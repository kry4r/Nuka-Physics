@echo off
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d C:\Softwares\code\Nuka-Physics
"build-win-editor\src\nuka_editor.exe" --frames 6 --capture-frame 5 --capture-out C:\Users\Nidho\tree_empty.ppm
echo empty_rc=%ERRORLEVEL%
