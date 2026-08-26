@echo off
cd /d C:\Softwares\code\Nuka-Physics
build-win-editor\tests\nuka_go2_walk_video.exe --bin C:\Softwares\code\Nuka-Physics\out\go2_hs_visual_test.bin --out-dir C:\Softwares\code\Nuka-Physics\out\go2_vulkan_test --frames 1 --probe
echo EXIT=%ERRORLEVEL%
