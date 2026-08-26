@echo off
cd /d C:\Softwares\code\Nuka-Physics
set "PATH=C:\VulkanSDK\1.4.321.1\Bin;C:\Windows\System32;%PATH%"
build-win-editor\tests\nuka_go2_walk_video.exe --bin out\go2_hs_visual_test.bin --out-dir out\go2_hs_visual_test --width 1280 --height 720 --probe
echo EXIT=%ERRORLEVEL%
