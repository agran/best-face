#!/usr/bin/env bash
# Запуск Best-Face на macOS и Linux.
#   ./run.sh "/путь/к/папке/с/фото" [--limit N] [--output папка] [--no-contact-sheet]
#   ./run.sh "/папка/1" "/папка/2" ...  — несколько папок: у каждой своя подпапка best_face_output
#   ./run.sh "/путь/к/manual_choices.json"   — применить ручной выбор из контактного листа
# Без аргументов откроется диалог выбора папки (macOS/zenity/kdialog) или запрос в консоли.
# При первом запуске создаётся окружение .venv и ставятся зависимости.
set -u
# Папка, из которой запустили: относительные пути к фото считаем от неё, а не от папки скрипта
START_DIR="$PWD"
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8

pause() {
    if [ -t 0 ] && [ -z "${BEST_FACE_NO_PAUSE:-}" ]; then
        read -r -n 1 -p "Нажмите любую клавишу для выхода..." _ || true
        echo
    fi
}

# ── Окружение ────────────────────────────────────────────────────────────────
# Создаётся при первом запуске и обновляется, если изменился requirements.txt
PY="$PWD/.venv/bin/python"
if [ ! -x "$PY" ]; then
    BASE_PY=""
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 \
            && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            BASE_PY="$cand"; break
        fi
    done
    if [ -z "$BASE_PY" ]; then
        echo "Python 3.10+ не найден. Установите его (macOS: brew install python; Linux: пакет python3 + python3-venv)."
        pause; exit 1
    fi
    echo "Первый запуск: создаю окружение .venv ..."
    if ! "$BASE_PY" -m venv .venv; then
        echo "Не удалось создать окружение (на Debian/Ubuntu нужен пакет python3-venv)."
        pause; exit 1
    fi
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
    echo "Устанавливаю зависимости из requirements.txt (несколько минут, нужен интернет)..."
    if ! "$PY" -m pip install --upgrade pip || ! "$PY" -m pip install -r requirements.txt; then
        echo "Не удалось установить зависимости."
        pause; exit 1
    fi
    cp requirements.txt .venv/requirements.installed
fi

# ── Папки с фото ─────────────────────────────────────────────────────────────
# Можно передать несколько папок и/или manual_choices.json; результат каждой папки
# сохраняется в её подпапке best_face_output.
if [ $# -eq 0 ]; then
    SRC=""
    if [ "$(uname)" = "Darwin" ]; then
        SRC="$(osascript -e 'POSIX path of (choose folder with prompt "Папка с фотографиями")' 2>/dev/null || true)"
    elif command -v zenity >/dev/null 2>&1; then
        SRC="$(zenity --file-selection --directory --title="Папка с фотографиями" 2>/dev/null || true)"
    elif command -v kdialog >/dev/null 2>&1; then
        SRC="$(kdialog --getexistingdirectory "$HOME" 2>/dev/null || true)"
    fi
    if [ -z "$SRC" ]; then
        read -r -p "Путь к папке с фотографиями: " SRC
    fi
    if [ -z "$SRC" ]; then
        echo "Папка не указана."
        pause; exit 1
    fi
    set -- "$SRC"
fi

# Относительные пути считаем от папки, где запустили скрипт; ключи (--limit, ...) не трогаем.
ARGS=()
PREV=""
for a in "$@"; do
    case "$a" in
        --*) ;;
        *)
            a="${a/#\~/$HOME}"
            case "$PREV" in
                --limit) ;;
                --output|--config|--choices)
                    case "$a" in /*) ;; *) a="$START_DIR/$a" ;; esac ;;
                *)
                    case "$a" in
                        /*) ;;
                        *) [ -e "$START_DIR/$a" ] && a="$START_DIR/$a" ;;
                    esac
                    if [ "${#a}" -gt 1 ]; then a="${a%/}"; fi ;;
            esac ;;
    esac
    ARGS+=("$a")
    PREV="$a"
done

# ── Запуск ───────────────────────────────────────────────────────────────────
"$PY" best_face.py --open "${ARGS[@]}"
RC=$?

echo
if [ "$RC" -eq 0 ]; then
    echo "Готово."
else
    echo "Завершено с ошибкой, код $RC."
    if [ "$RC" -eq 2 ]; then
        echo "Скорее всего, не найдены файлы моделей: проверьте интернет и запустите download_models.py, либо положите файлы в папку models/ (см. README.md)."
    fi
fi
pause
exit "$RC"
