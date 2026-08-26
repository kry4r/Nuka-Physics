@echo off
cd /d C:\Softwares\code\Nuka-Physics
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;C:\VulkanSDK\1.4.321.1\Bin;C:\Windows\System32;%PATH%"
build-win-editor\tests\nuka_go2_skill_video_cuda.exe --bin C:\Softwares\code\Nuka-Physics\out\go2_bf_start.bin --out-dir C:\Softwares\code\Nuka-Physics\out\go2_bf_start_frames --frames 60 --stride 1 --samples 2
echo EXIT=%ERRORLEVEL%
