@echo off
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "C:\Softwares\code\Nuka-Physics"
build-win-editor\src\nuka_editor.exe --scene examples\scenes\go2_terrain30.nks --policy out\go2_terrain_hs\go2_policy.bin --play --dt 0.005 --frames %1 --capture-frame %2 --capture-out %3
echo RUN_EXIT=%ERRORLEVEL%
