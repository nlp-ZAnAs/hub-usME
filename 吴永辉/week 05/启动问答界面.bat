@echo off
setlocal
cd /d "%~dp0"

rem Python interpreter: edit this line if your torch lives elsewhere.
set "PY=F:\anaconda3\envs\py312\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0chat_ui.py" %*

if errorlevel 1 (
    echo.
    echo Failed to start. Make sure PyTorch is installed, or edit the PY path in this file.
    pause
)
endlocal
