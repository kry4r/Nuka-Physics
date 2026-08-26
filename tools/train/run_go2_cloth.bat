@echo off
cd /d C:\Softwares\code\Nuka-Physics
set "PATH=C:\VulkanSDK\1.4.321.1\Bin;C:\Windows\System32;%PATH%"
build-win-editor\tests\nuka_go2_cloth_drape_demo.exe --frames 1 --stride 1 --out-dir out\go2_cloth_visual_test --png-dir out\go2_cloth_visual_test\png
echo EXIT=%ERRORLEVEL%
