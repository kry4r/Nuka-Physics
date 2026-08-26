@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\CMake\bin\cmake.exe" --build C:\Softwares\code\Nuka-Physics\build-win-editor --target nuka_go2_walk_video --config Release -j 8
exit /b %ERRORLEVEL%
