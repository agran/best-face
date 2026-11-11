@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

rem Внимание: в этом файле нет блоков в скобках (if ... ( ... )). Путь перетащенной
rem папки может содержать скобки, например "Новая папка (26)", и ломал бы такой блок.

if "%~1"=="" goto usage

set "SRC=%~1"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
if not exist "%SRC%\" goto notdir

call :ensure_env
if errorlevel 1 goto envfailed

".venv\Scripts\python.exe" best_face.py "%SRC%" %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" goto failed
echo Готово. Открываю папку с лучшими фото...
if exist "%SRC%\best_face_output\best" start "" explorer "%SRC%\best_face_output\best"
goto done

:usage
echo Перетащите папку с фотографиями на этот файл.
echo Либо запустите из консоли: run.bat "путь\к\папке" [--limit N] [--output папка]
set "RC=1"
goto done

:notdir
echo Это не папка: "%SRC%"
set "RC=1"
goto done

:envfailed
echo.
echo Не удалось подготовить окружение. См. сообщения выше.
set "RC=1"
goto done

:failed
echo Завершено с ошибкой, код %RC%.
if "%RC%"=="2" echo Скорее всего, не найдены файлы моделей: проверьте интернет и запустите download_models.py, либо положите файлы в папку models\ ^(см. README.md^).
goto done

:done
pause
exit /b %RC%


rem ── Подготовка окружения ───────────────────────────────────────────────────
rem Создаёт .venv и ставит requirements.txt при первом запуске, а также повторно,
rem если requirements.txt изменился.
:ensure_env
if exist ".venv\Scripts\python.exe" goto check_deps

set "BASEPY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASEPY=py -3"
if defined BASEPY goto make_venv
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASEPY=python"
if defined BASEPY goto make_venv

echo Python 3.10 или новее не найден.
echo Установите его с https://www.python.org/downloads/ и отметьте "Add python.exe to PATH",
echo затем запустите этот файл снова.
exit /b 1

:make_venv
echo Первый запуск: создаю окружение .venv ...
%BASEPY% -m venv .venv
if errorlevel 1 exit /b 1

:check_deps
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if not errorlevel 1 exit /b 0

echo Устанавливаю зависимости из requirements.txt ^(несколько минут, нужен интернет^)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
copy /y requirements.txt .venv\requirements.installed >nul
exit /b 0
