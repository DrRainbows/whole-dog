@echo off
rem Both dogs (the Lab and the Malinois) in one world. Speak to both, or to one by name: "mal, sit".
rem F1 sit, F2 down, F3 come, F4 stand, F5 jump, F6 shake, F7 next dog, F8 shove, arrows walk, Tab camera, Esc quit.
cd /d "%~dp0.."
set OMP_NUM_THREADS=1
"body\.venv\Scripts\python.exe" game\dog_game.py --dogs lab,mal --cam 3 %*
if errorlevel 1 pause
