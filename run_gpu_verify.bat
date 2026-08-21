@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set PATH=%CUDA_PATH%\bin;%PATH%
cd /d C:\Softwares\code\Nuka-Physics
set SC=examples\scenes\go2_terrain30.nks
set POL=out\go2_terrain_hs\go2_policy.bin
set EXE=build-win-editor\src\nuka_editor.exe

echo ==BUILD==
cmake --build build-win-editor --target nuka_editor_exe -j 16 1>build_gpu.log 2>&1
echo BUILD_RC=%ERRORLEVEL%
if not exist %EXE% ( echo NO_EXE & exit /b 1 )
echo ==BUILD_OK==

echo ==RUN1_GPU_play_late==
%EXE% --scene %SC% --policy %POL% --play --dt 0.005 --frames 600 --capture-frame 550 --capture-out gpu_late.ppm 1>run_gpu_late.log 2>&1
echo RUN1_RC=%ERRORLEVEL%

echo ==RUN2_GPU_play_early==
%EXE% --scene %SC% --policy %POL% --play --dt 0.005 --frames 70 --capture-frame 8 --capture-out gpu_early.ppm 1>run_gpu_early.log 2>&1
echo RUN2_RC=%ERRORLEVEL%

echo ==RUN3_HOST_play==
%EXE% --scene %SC% --policy %POL% --play --host-policy --dt 0.005 --frames 150 1>run_host.log 2>&1
echo RUN3_RC=%ERRORLEVEL%

echo ==RUN4_play_no_policy==
%EXE% --scene %SC% --play --dt 0.005 --frames 200 1>run_nopol.log 2>&1
echo RUN4_RC=%ERRORLEVEL%

echo ==RUN5_static==
%EXE% --scene %SC% --policy %POL% --dt 0.005 --frames 200 1>run_static.log 2>&1
echo RUN5_RC=%ERRORLEVEL%

echo ==VERIFY_DONE==
endlocal
