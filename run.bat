@echo off
rem Double-click to run. The first run creates a virtual environment and installs
rem the dependencies. It lives in %LOCALAPPDATA% so cloud-sync folders like
rem OneDrive don't try to sync it.
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\ar-face-filters\venv"

if exist "%VENV%\Scripts\python.exe" goto launch

echo Setting up AR Face Filters. This only happens once and needs internet access.
set "BASE="
py -3.13 --version >nul 2>&1 && set "BASE=py -3.13"
if not defined BASE py -3.12 --version >nul 2>&1 && set "BASE=py -3.12"
if not defined BASE (
    echo Python 3.12 or 3.13 is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)
%BASE% -m venv "%VENV%" || goto failed
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto failed

:launch
"%VENV%\Scripts\python.exe" main.py %*
if errorlevel 1 pause
exit /b

:failed
echo Setup failed. See the messages above.
rmdir /s /q "%VENV%" 2>nul
pause
exit /b 1
