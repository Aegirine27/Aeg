@echo off
cd /d "%~dp0"
python gui_main.py
if errorlevel 1 pause
