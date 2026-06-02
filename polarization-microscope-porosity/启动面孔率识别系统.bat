@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo  偏光显微镜面孔率识别系统
echo ========================================
echo.
python gui_main.py
if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查Python是否安装
    echo.
    pause
)
