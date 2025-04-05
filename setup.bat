@echo off
setlocal enabledelayedexpansion

echo Checking Python version...

REM Run Python to get version information
for /f "tokens=2" %%V in ('python -c "import sys; print(sys.version.split()[0])"') do (
    set PYTHON_VERSION=%%V
)

echo Detected Python version: %PYTHON_VERSION%

REM Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

echo Major version: %MAJOR%
echo Minor version: %MINOR%

REM Check if version is above 3.12
if %MAJOR% GTR 3 (
    echo ERROR: Python version %PYTHON_VERSION% is above 3.12
    exit /b 1
)

if %MAJOR% EQU 3 (
    if %MINOR% GTR 12 (
        echo ERROR: Python version %PYTHON_VERSION% is above 3.12
        exit /b 1
    )
)

echo SUCCESS: Python version %PYTHON_VERSION% is 3.12 or below

python -m venv env
call env/Scripts/activate.bat
pip install -r requirements.txt