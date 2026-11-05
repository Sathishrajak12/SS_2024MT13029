@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto :try_python
set "PYTHON_COMMAND=py -3.12"
goto :python_found

:try_python
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found.
  echo Install 64-bit Python 3.12 from https://www.python.org/downloads/
  pause
  exit /b 1
)
set "PYTHON_COMMAND=python"

:python_found

if not exist "venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  %PYTHON_COMMAND% -m venv venv
  if errorlevel 1 goto :failed
)

echo Installing project requirements. This can take several minutes...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Setup completed successfully.
echo Run run_windows.bat to start the application.
exit /b 0

:failed
echo.
echo Setup failed. Review the error above and confirm internet access is available.
pause
exit /b 1
