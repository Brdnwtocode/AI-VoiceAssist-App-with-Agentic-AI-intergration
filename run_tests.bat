@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  python -m venv venv
  if errorlevel 1 exit /b 1
)

call venv\Scripts\activate.bat
python -m pip install -r requirements.txt -q
if errorlevel 1 exit /b 1

set "SERVICE_URL=http://127.0.0.1:8000"
set "MOCK_OPENAI=1"

echo Starting uvicorn in background...
start /B "" "%~dp0venv\Scripts\python.exe" -m uvicorn src.main:app --host 127.0.0.1 --port 8000
ping -n 5 127.0.0.1 >nul

echo Running contract tests...
"%~dp0venv\Scripts\python.exe" -m src.test_contract --mock-openai
set EC=!ERRORLEVEL!

echo Stopping server on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%a >nul 2>&1
)

exit /b %EC%
