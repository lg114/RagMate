@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo Starting RagMate Unified Launcher...
echo ==========================================

if exist "backend\.venv\Scripts\python.exe" (
    "backend\.venv\Scripts\python.exe" run.py %*
) else (
    python run.py %*
)

if errorlevel 1 (
    echo.
    echo [RagMate] Startup exited with error.
    pause
)
