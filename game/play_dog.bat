@echo off
rem The dog game (Panda3D). Double-click for the Lab; or: play_dog.bat --dog mal
rem Type words and press Enter to speak. F1 sit, F2 down, F3 come, F4 stand, F5 jump, F6 shake, F8 shove, Tab camera, Esc quit.
cd /d "%~dp0.."
set OMP_NUM_THREADS=1
"body\.venv\Scripts\python.exe" game\dog_game.py %*
if errorlevel 1 pause
