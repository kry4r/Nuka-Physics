@echo off
cd /d C:\Softwares\code\Nuka-Physics
set PYTHONPATH=python
C:\Users\Nidho\AppData\Local\Programs\Python\Python313\python.exe tools\train\train_skill.py --config configs\rl_games\go2_backflip_pd.yaml --max-frames 307200
