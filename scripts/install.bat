@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo Installing UAV Pothole Recognition Environment...
echo ==================================================

:: Check if python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is required but not found in PATH.
    exit /b 1
)

:: Get project root directory (parent of scripts/)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo Error: Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment and install requirements
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo Error: Failed to install dependencies.
        exit /b 1
    )
) else (
    echo Error: requirements.txt not found.
    exit /b 1
)

echo ==================================================
echo Environment successfully installed!
echo To activate virtual environment: .venv\Scripts\activate.bat
echo ==================================================

pause
