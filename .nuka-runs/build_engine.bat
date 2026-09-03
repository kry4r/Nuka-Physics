@echo off
set "PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\CMake\bin"
cd /d "C:\Softwares\code\Nuka-Physics"
cmake -S . -B build-win-editor -DCMAKE_MAKE_PROGRAM="%NINJA%" 1>.nuka-runs\cfg.log 2>&1
if errorlevel 1 ( echo CONFIGURE_FAILED & tail -30 .nuka-runs\cfg.log & exit /b 1 )
echo CONFIGURE_OK
"%NINJA%" -C build-win-editor -j 8 src/nuka.dll
