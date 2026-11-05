@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
  echo Project environment not found. Running first-time setup...
  call setup_windows.bat
  if errorlevel 1 exit /b 1
)
call venv\Scripts\activate.bat
venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType=none

