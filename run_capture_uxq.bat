@echo off
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d C:\Softwares\code\Nuka-Physics
set "EXE=build-win-editor\src\nuka_editor.exe"
echo === capture go2_field ===
"%EXE%" --scene examples\scenes\go2_field.nks --frames 6 --capture-frame 5 --capture-out C:\Users\Nidho\tree.ppm
echo field_rc=%ERRORLEVEL%
echo === capture go2 ===
"%EXE%" --scene examples\scenes\go2.nks --frames 6 --capture-frame 5 --capture-out C:\Users\Nidho\tree_go2.ppm
echo go2_rc=%ERRORLEVEL%
dir C:\Users\Nidho\tree*.ppm
