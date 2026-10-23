@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   CRIPT SUBS - Transcripcion y notas en local
echo ============================================================
echo.
echo Capturando el AUDIO DEL SISTEMA (lo que suena en el PC) con modelo large-v3.
echo Para otras opciones edita este .bat o ejecuta:
echo   .venv\Scripts\python.exe escuchar.py --help
echo.
".venv\Scripts\python.exe" escuchar.py --fuente sistema --modelo large-v3
echo.
pause
