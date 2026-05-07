@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\packaging\windows\build_windows.ps1"
pause
