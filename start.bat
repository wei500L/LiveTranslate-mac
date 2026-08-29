@echo off
cd /d "%~dp0"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

if not exist ".venv\Scripts\python.exe" (
    echo Environment not found; running the installer first...
    call "%~dp0install.bat"
    if errorlevel 1 exit /b 1
)

if not exist ".venv\.livetranslate-ready" (
    echo Setup is incomplete; running the installer first...
    call "%~dp0install.bat"
    if errorlevel 1 exit /b 1
)

echo Starting LiveTranslate...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
