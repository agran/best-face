@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if "%~1"=="" (
    echo Перетащите папку с фотографиями на этот файл.
    echo Либо запустите из консоли: run.bat "путь\к\папке" [--limit N] [--output папка]
    pause
    exit /b 1
)

set "SRC=%~1"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

if not exist "%SRC%\" goto notdir

python best_face.py "%SRC%" %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" goto failed
echo Готово. Открываю папку с лучшими фото...
if exist "%SRC%\best_face_output\best" start "" explorer "%SRC%\best_face_output\best"
goto done

:failed
echo Завершено с ошибкой, код %RC%.
goto done

:notdir
echo Это не папка: "%SRC%"
set "RC=1"

:done
pause
exit /b %RC%