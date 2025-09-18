@echo off
set PYTHON=C:\Users\Lenovo\miniforge3\envs\ov\python.exe
set OV_CACHE_DIR=%~dp0\.ov_cache
set SF2_PATH=%~dp0\assets\FluidR3_GM.sf2
start "" "%PYTHON%" -m uvicorn compose_full_server:app --host 127.0.0.1 --port 9010
