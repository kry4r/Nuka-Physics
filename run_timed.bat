@echo off
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "C:\Softwares\code\Nuka-Physics"
set START=%TIME%
build-win-editor\src\nuka_editor.exe %* >nul 2>&1
set END=%TIME%
echo START=%START% END=%END%
