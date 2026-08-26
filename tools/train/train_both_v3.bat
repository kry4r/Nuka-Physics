@echo off
cd /d C:\Softwares\code\Nuka-Physics
set PYTHONPATH=python
C:\Users\Nidho\AppData\Local\Programs\Python\Python313\python.exe tools\train\train_skill.py --config configs\rl_games\go2_backflip_pd.yaml --max-frames 9830400 > logs\train_backflip_pd_v3.log 2>&1
echo BACKFLIP_V3_EXIT=%ERRORLEVEL%
C:\Users\Nidho\AppData\Local\Programs\Python\Python313\python.exe tools\train\train_skill.py --config configs\rl_games\go2_front_handstand_walk.yaml --max-frames 9830400 > logs\train_hs_walk_v3.log 2>&1
echo HANDSTAND_V3_EXIT=%ERRORLEVEL%
