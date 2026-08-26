@echo off
cd /d C:\Softwares\code\Nuka-Physics
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;C:\VulkanSDK\1.4.321.1\Bin;C:\Windows\System32;%PATH%"
build-win-editor\tests\nuka_go2_skill_video_cuda.exe --bin C:\Softwares\code\Nuka-Physics\out\go2_hs_cuda.bin --out-dir C:\Softwares\code\Nuka-Physics\out\go2_hs_cuda_frames --frames 200 --stride 2 --samples 2
echo EXIT=%ERRORLEVEL%
