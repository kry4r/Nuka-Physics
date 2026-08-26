@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
dumpbin /DEPENDENTS C:\Softwares\code\Nuka-Physics\build-win-editor\tests\nuka_go2_walk_video.exe
